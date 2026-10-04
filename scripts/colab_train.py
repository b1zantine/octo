"""Three-epoch CUDA trainer with durable checkpoints and direct W&B tracking."""
import argparse
import gc
import json
import os
from pathlib import Path
import random
import signal
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
import torch
from transformers import AutoTokenizer
from octo import Encoder, Limits, OctoModel, StructuralTokens, make_optimizer, train_step, save_checkpoint, load_checkpoint
from octo.cli import read_records
from octo.dataset import digest
from octo.evaluation import evaluate


def write(path, data):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(data, indent=2) + '\n'); temp.replace(path)


def batch_order(lengths, size):
    indices=list(range(len(lengths))); random.shuffle(indices)
    batches=[]
    for offset in range(0,len(indices),size*32):
        window=sorted(indices[offset:offset+size*32],key=lengths.__getitem__)
        batches.extend(window[i:i+size] for i in range(0,len(window),size))
    random.shuffle(batches)
    return batches


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True)
    parser.add_argument('--approve-training',action='store_true')
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args()
    if not args.approve_training: parser.error('training requires explicit approval')
    os.chdir(ROOT); config=json.loads(Path(args.config).read_text())
    if config['epochs'] not in (1,2,3) or config['batch_size'] not in (1,16) or config['precision']!='float32':
        raise ValueError('unsupported training configuration')
    output=ROOT/'artifacts/runs'/config['run_name'];output.mkdir(parents=True,exist_ok=True)
    import fcntl
    lock=(output/'training.lock').open('a');fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    start=time.monotonic(); stopped=False
    def request_stop(*_):
        nonlocal stopped
        stopped=True
    signal.signal(signal.SIGTERM,request_stop);signal.signal(signal.SIGINT,request_stop)
    def expired():return stopped or time.monotonic()-start>=config['recommended_max_runtime_hours']*3600
    def status(phase,**values):
        write(output/'status.json',{'phase':phase,'pid':os.getpid(),'updated_unix':time.time(),**values})
    status('validating_data')
    torch.manual_seed(config['seed']);random.seed(config['seed']);torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32=config['tf32'];torch.backends.cudnn.allow_tf32=config['tf32']
    tokenizer=AutoTokenizer.from_pretrained('artifacts/tokenizers/qwen3',local_files_only=True)
    encoder=Encoder(tokenizer,StructuralTokens.from_tokenizer(tokenizer),Limits(),tokenizer.pad_token_id or tokenizer.eos_token_id)
    records=[];dev=[]
    for domain in ('bfsi','chess'):
        records.extend(read_records(f'artifacts/datasets/{domain}-v2/train.jsonl'))
        dev.extend(read_records(f'artifacts/datasets/{domain}-v2/dev.jsonl'))
    lengths=[]
    for row in records:
        encoded=encoder.encode(row);lengths.append(len(encoded.input_ids))
        if any(q.target is None for q in encoded.questions):raise ValueError('unlabeled training record')
    for row in dev:encoder.encode(row)
    if {r['source_group_id'] for r in records}&{r['source_group_id'] for r in dev}:raise ValueError('data leakage')
    config['dataset_sha256']={f'{d}/{s}':digest(f'artifacts/datasets/{d}-v2/{s}.jsonl') for d in ('bfsi','chess') for s in ('train','dev')}
    config.update(torch=torch.__version__,gpu=torch.cuda.get_device_name(),training_approved=True)
    latest=output/'latest_checkpoint.json'
    if latest.exists() and not args.resume:raise ValueError('existing checkpoint; use --resume')
    status('loading_model')
    if args.resume:
        prior=json.loads((output/'training_config.json').read_text())
        for key in ('dataset_sha256','base_revision','seed','precision','tf32','batch_size','batching','adapter_lr','pointer_lr','epochs'):
            if config[key]!=prior[key]:raise ValueError('resume configuration mismatch: '+key)
        saved=Path(json.loads(latest.read_text())['path'])
        model,encoder,_=load_checkpoint(saved,device='cuda')
        state=torch.load(saved/'training_state.pt',map_location='cpu',weights_only=False)
        optimizer=make_optimizer(model,config['adapter_lr'],config['pointer_lr']);optimizer.load_state_dict(state['optimizer'])
        for values in optimizer.state.values():
            for key,value in values.items():
                if torch.is_tensor(value) and key!='step':values[key]=value.to('cuda')
        epoch=state['epoch'];order=state['order'];cursor=state['cursor'];step=state['step']
        random.setstate(state['python_rng']);torch.set_rng_state(state['torch_rng']);torch.cuda.set_rng_state_all(state['cuda_rng'])
    else:
        model=OctoModel.from_pretrained(device='cuda',dtype=torch.float32,backend=config['backend'],local_files_only=True)
        model.backbone.config._commit_hash=config['base_revision'];model.backbone.config._name_or_path=config['base_model']
        optimizer=make_optimizer(model,config['adapter_lr'],config['pointer_lr'])
        epoch=cursor=step=0;order=batch_order(lengths,config['batch_size'])
    write(output/'training_config.json',config)
    import wandb
    identity_path=output/'tracking_run.json'
    previous=json.loads(identity_path.read_text()) if identity_path.exists() else {}
    run=wandb.init(entity=config['wandb_entity'],project=config['wandb_project'],name=config['run_name'],
                   id=previous.get('id'),resume='allow',config=config,dir=str(output),
                   settings=wandb.Settings(disable_git=True,save_code=False))
    run.config.update(config,allow_val_change=True)
    cloud_step=max(run.step,config.get('wandb_last_logged_step',-1)+1)
    write(identity_path,{'id':run.id,'url':run.url})
    for reference in config['dataset_artifacts']:
        run.use_artifact(reference,type='dataset')
    queued=[]
    def _publish_checkpoint(path,aliases=None,initial=False):
        saved_metadata=json.loads((path/'octo.json').read_text())['metadata']
        saved_step=saved_metadata['step'];saved_epoch=saved_metadata['epoch']
        aliases=aliases or [f'step-{saved_step}',f'epoch-{saved_epoch}']
        full=wandb.Artifact(config['run_name']+'-training-checkpoint',type='training-checkpoint',
                            metadata={'step':saved_step,'epoch':saved_epoch,'base_revision':config['base_revision']})
        full.add_dir(str(path));uploaded=run.log_artifact(full,aliases=aliases)
        model_artifact=wandb.Artifact(config['run_name']+'-model',type='model',
                                     metadata={'step':saved_step,'epoch':saved_epoch,'base_revision':config['base_revision']})
        for file in path.rglob('*'):
            if file.is_file() and file.name not in ('training_state.pt','file_checksums.json'):
                model_artifact.add_file(str(file),name=str(file.relative_to(path)))
        inference=run.log_artifact(model_artifact,aliases=aliases)
        queued.extend([uploaded,inference])
        if initial:
            uploaded.wait(timeout=180);inference.wait(timeout=180)
            write(output/'initial_cloud_receipt.json',{'training_checkpoint':uploaded.qualified_name,
                                                       'model':inference.qualified_name,'url':run.url})
    pending=[]
    def publish_checkpoint(path,aliases=None,initial=False):
        try:
            _publish_checkpoint(path,aliases,initial)
        except Exception as exc:
            if initial: raise
            pending.append((str(path),aliases))
            write(output/'pending_uploads.json',{'checkpoints':pending,'last_error':type(exc).__name__})
    def retry_uploads():
        previous=list(pending);pending.clear()
        for path,aliases in previous:publish_checkpoint(Path(path),aliases)
        write(output/'pending_uploads.json',{'checkpoints':pending})
    def checkpoint(label='',cloud=False):
        path=output/'checkpoints'/f'step-{step:08d}{label}'
        if not path.exists():
            staging=path.with_name(path.name+'.saving')
            if staging.exists():staging.rename(staging.with_name(staging.name+f'-{time.time_ns()}'))
            save_checkpoint(staging,model,encoder,metadata={**config,'step':step,'epoch':epoch+1})
            torch.save({'optimizer':optimizer.state_dict(),'epoch':epoch,'order':order,'cursor':cursor,'step':step,
                        'python_rng':random.getstate(),'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all()},staging/'training_state.pt')
            write(staging/'file_checksums.json',{str(p.relative_to(staging)):digest(p) for p in staging.rglob('*') if p.is_file()})
            staging.rename(path)
        if cloud:publish_checkpoint(path)
        write(latest,{'path':str(path),'step':step,'epoch':epoch+1})
        return path
    best_path=output/'best_checkpoint.json'
    best=json.loads(best_path.read_text()) if best_path.exists() else {'loss':float('inf')}
    history=json.loads((output/'development_metrics.json').read_text()) if (output/'development_metrics.json').exists() else {}
    checkpoint();phase='training'
    try:
        with (output/'metrics.jsonl').open('a') as log:
            while epoch<config['epochs']:
                status('training',step=step,epoch=epoch+1,total_steps=len(order)*config['epochs'])
                while cursor<len(order):
                    if expired():raise InterruptedError('training runtime limit or stop requested')
                    before=time.monotonic()
                    torch.cuda.reset_peak_memory_stats()
                    loss=train_step(model,encoder.batch([records[i] for i in order[cursor]]).to('cuda'),optimizer,config['gradient_clip_norm'])
                    cursor+=1;step+=1
                    metrics={'step':step,'epoch':epoch+1,'train/loss':loss,'train/step_seconds':time.monotonic()-before,
                             'train/records':len(order[cursor-1]),'gpu/peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30,'gpu/peak_reserved_gib':torch.cuda.max_memory_reserved()/2**30,'lr/adapter':config['adapter_lr'],'lr/pointer':config['pointer_lr']}
                    log.write(json.dumps(metrics)+'\n');log.flush()
                    if step>=cloud_step:run.log({k:v for k,v in metrics.items() if k!='step'},step=step)
                    if step%10==0:status('training',step=step,epoch=epoch+1,total_steps=len(order)*config['epochs'],loss=loss)
                    if step%config['checkpoint_every_steps']==0:
                        retry_uploads();checkpoint()
                saved=checkpoint(f'-epoch-{epoch+1}',cloud=True)
                status('development_evaluation',step=step,epoch=epoch+1)
                def batches():
                    for offset in range(0,len(dev),config['batch_size']):
                        if expired():raise InterruptedError('evaluation runtime limit or stop requested')
                        yield encoder.batch(dev[offset:offset+config['batch_size']]).to('cuda')
                scores=evaluate(model,batches());history[str(epoch+1)]={'step':step,**scores}
                write(output/'development_metrics.json',history)
                run.log({f'dev/{k}':v for k,v in scores.items()},step=step)
                if scores['loss']<best['loss']:
                    best={'path':str(saved),'loss':scores['loss'],'epoch':epoch+1,'step':step};write(best_path,best)
                elif config.get('early_stop_on_dev_loss_increase',True) and scores['loss']>best['loss']:
                    phase='early_stopped';break
                epoch+=1
                if epoch<config['epochs']:order=batch_order(lengths,config['batch_size']);cursor=0
        if phase!='early_stopped':phase='complete'
        write(output/'final_checkpoint.json',{'path':str(saved),'step':step,'best':best})
        publish_checkpoint(Path(best['path']),aliases=['best'])
        publish_checkpoint(saved,aliases=['final'])
        results=wandb.Artifact(config['run_name']+'-evaluation',type='evaluation')
        results.add_file(str(output/'development_metrics.json'));results.add_file(str(output/'training_config.json'))
        queued.append(run.log_artifact(results))
        run.summary.update({'best_dev_loss':best['loss'],'best_epoch':best['epoch'],'final_step':step,'phase':phase})
    except InterruptedError as exc:
        checkpoint('-interrupted',cloud=True);phase='paused_at_runtime_limit' if not stopped else 'interrupted'
        write(output/'interruption.json',{'reason':str(exc)})
    except BaseException as exc:
        checkpoint('-failed',cloud=True);phase='failed';write(output/'failure.json',{'type':type(exc).__name__,'error':str(exc),'traceback':traceback.format_exc(),'gpu_allocated_gib':torch.cuda.memory_allocated()/2**30});raise
    finally:
        training_phase=phase
        status('syncing_artifacts',step=step,training_phase=training_phase)
        try:
            retry_uploads()
            if pending:raise RuntimeError('checkpoint uploads remain pending')
            for artifact in queued: artifact.wait(timeout=600)
            write(output/'cloud_receipt.json',{'run_url':run.url,'artifacts':[a.qualified_name for a in queued],
                                               'best':best,'training_phase':training_phase})
            run.finish(exit_code=1 if phase=='failed' else 0)
        except Exception as exc:
            write(output/'cloud_sync_error.json',{'error':type(exc).__name__,'local_files_retained':True})
            phase='cloud_sync_pending'
        status(phase,step=step,epoch=min(epoch+1,config['epochs']),elapsed_hours=(time.monotonic()-start)/3600,best=best)
        del model,optimizer;gc.collect();torch.cuda.empty_cache()

if __name__=='__main__':main()
