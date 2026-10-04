"""Train from labeled JSONL, or run a small synthetic wiring demonstration."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import random
import time
import torch
from . import Encoder, Limits, OctoModel, StructuralTokens, make_optimizer, save_checkpoint, train_step
from .evaluation import evaluate
from .model import BASE_MODEL, BASE_REVISION


def tiny_setup(backend):
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import PreTrainedTokenizerFast, Qwen3Config, Qwen3Model
    words = ['[UNK]', '[PAD]', '<state>', '<question>', '<option>', '</option>', '<decide>',
             'Export', 'is', 'broken', 'A', 'workaround', 'available', 'Which', 'team', 'Payments',
             'Product', 'functionality', 'Rate', 'impact', 'Cosmetic', 'Broken', 'with', 'without',
             'True', 'False', 'the', 'following', 'statement', 'true', 'of', 'state', '?', 'Is']
    tokenizer_backend = Tokenizer(WordLevel({w:i for i,w in enumerate(words)}, unk_token='[UNK]'))
    tokenizer_backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer_backend, unk_token='[UNK]',
        pad_token='[PAD]', additional_special_tokens=words[2:7])
    config = Qwen3Config(vocab_size=len(words), hidden_size=64, intermediate_size=128,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2, head_dim=16,
        max_position_embeddings=512, attention_dropout=0.)
    config._attn_implementation = backend
    return OctoModel(Qwen3Model(config)), tokenizer, StructuralTokens(2,3,4,5,6)


def demo_records():
    state = 'Export is broken A workaround is available'
    questions = [
        {'id':'route','type':'choice','instruction':'Which team','options':[
            {'id':'payments','description':'Payments'},
            {'id':'product','description':'Product functionality'}], 'label_id':'product'},
        {'id':'impact','type':'score','instruction':'Rate impact','criteria':[
            'Cosmetic','Broken with workaround','Broken without workaround'],'label_id':'1'},
        {'id':'workaround','type':'noul','statement':'A workaround is available','label_id':'true'}]
    return [{'record_id':f'demo-{i}', 'source_group_id':'synthetic-demo', 'state':state,'questions':[q]}
            for i,q in enumerate(questions)]


def read_records(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train', action='append', help='Labeled training JSONL; repeat for separate corpora')
    parser.add_argument('--dev', action='append', help='Development JSONL; repeat for separate corpora')
    parser.add_argument('--tiny', action='store_true', help='Random CPU backbone for wiring checks')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--seed', type=int, default=17)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--backend', choices=['eager','sdpa'], default='sdpa')
    parser.add_argument('--checkpoint', default='artifacts/models/checkpoint')
    parser.add_argument('--wandb', choices=['online','offline','disabled'], default='disabled')
    parser.add_argument('--project', default='octo_v1')
    parser.add_argument('--run-name')
    parser.add_argument('--dataset-artifact', action='append', help='Exact W&B dataset version; repeat for multiple corpora')
    args = parser.parse_args()
    import re
    if args.dataset_artifact and any(not re.search(r':v[0-9]+$', a) for a in args.dataset_artifact):
        parser.error('--dataset-artifact requires an immutable :vN version')
    if args.dataset_artifact and args.wandb != 'online':
        parser.error('--dataset-artifact requires --wandb online')
    if args.epochs <= 0:
        parser.error('epochs must be positive')
    if not args.tiny and not args.train:
        parser.error('provide --train, or use --tiny for the synthetic demo')
    if Path(args.checkpoint).exists():
        parser.error('checkpoint destination already exists; choose a new path')
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    if args.tiny:
        torch.set_num_threads(1)
        model, tokenizer, tokens = tiny_setup(args.backend)
        model.to(args.device)
    else:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL,revision=BASE_REVISION)
        tokens = StructuralTokens.from_tokenizer(tokenizer)
        model = OctoModel.from_pretrained(device=args.device, backend=args.backend)
    encoder = Encoder(tokenizer,tokens,Limits(),tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id)
    records = [r for path in args.train for r in read_records(path)] if args.train else demo_records()
    if not records:
        parser.error('training dataset is empty')
    # Validate all supervision and budgets before creating a remote run or updating weights.
    for record in records:
        encoded = encoder.encode(record)
        if any(q.target is None for q in encoded.questions):
            parser.error('every training question must have a label')
    dev = [r for path in args.dev for r in read_records(path)] if args.dev else []
    if dev:
        if any(not isinstance(r.get('source_group_id'),str) or not r['source_group_id'] for r in records+dev):
            parser.error('train/development records require source_group_id for leakage checks')
        if {r['source_group_id'] for r in records} & {r['source_group_id'] for r in dev}:
            parser.error('train and development source groups overlap')
    for record in dev:
        encoder.encode(record)
    def batches_for(items):
        return (encoder.batch([record]).to(args.device) for record in items)
    config = {'scope':'synthetic architecture check' if args.tiny and not args.train else 'task training',
              'backbone':'tiny random Qwen3' if args.tiny else BASE_MODEL,
              'revision':None if args.tiny else BASE_REVISION,'seed':args.seed,'epochs':args.epochs,
              'training_records':len(records),'development_records':len(dev), 'precision':'float32',
              'backend':args.backend, 'limits':asdict(encoder.limits),
              'adapter_lr':1e-4, 'pointer_lr':1e-3,
              'trainable_parameters':sum(p.numel() for p in model.parameters() if p.requires_grad)}
    if args.train:
        from .dataset import digest
        config['train_file_sha256'] = {str(Path(path)):digest(path) for path in args.train}
        config['dev_file_sha256'] = {str(Path(path)):digest(path) for path in args.dev or []}
        if len(args.train)==1:
            config['train_sha256'] = digest(args.train[0])
        config['dataset_manifest_sha256'] = {
            str(Path(path).parent):digest(Path(path).parent/'manifest.json')
            for path in args.train if (Path(path).parent/'manifest.json').exists()}
    config['dataset_artifacts'] = args.dataset_artifact or []
    run = None
    if args.wandb != 'disabled':
        from .tracking import start_run
        run = start_run(project=args.project,name=args.run_name,mode=args.wandb,config=config)
    if run and args.dataset_artifact:
        try:
            artifact_files = []
            for reference in args.dataset_artifact:
                artifact = run.use_artifact(reference, type='dataset')
                artifact_files.append(artifact.metadata['files'])
            for selected in (args.train or [])+(args.dev or []):
                if not any(files.get(Path(selected).name)==digest(selected) for files in artifact_files):
                    raise ValueError('selected records do not match the referenced dataset artifacts')
        except Exception:
            run.finish(exit_code=1)
            raise
    optimizer=make_optimizer(model)
    trace=[]
    failed=True
    try:
        step=0
        for epoch in range(args.epochs):
            indices=list(range(len(records)))
            random.shuffle(indices)
            for index in indices:
                start=time.perf_counter()
                batch = encoder.batch([records[index]]).to(args.device)
                loss=train_step(model,batch,optimizer)
                gradient_norm=math_gradient_norm(model)
                step+=1
                metrics={'train/loss':loss,'train/gradient_norm':gradient_norm,
                         'train/step_seconds':time.perf_counter()-start,'epoch':epoch+1,
                         'lr/adapter':optimizer.param_groups[0]['lr'], 'lr/pointer':optimizer.param_groups[1]['lr']}
                trace.append({'step':step,**metrics})
                if run: run.log(metrics,step=step)
            metrics={f'train_eval/{k}':v for k,v in evaluate(model,batches_for(records)).items()}
            if dev: metrics.update({f'dev/{k}':v for k,v in evaluate(model,batches_for(dev)).items()})
            if run: run.log(metrics,step=step)
            print(json.dumps({'epoch':epoch+1,**metrics}),flush=True)
        save_checkpoint(args.checkpoint,model,encoder,metadata=config)
        output=Path(args.checkpoint).parent
        (output/'training_metrics.json').write_text(json.dumps(trace,indent=2)+'\n')
        (Path(args.checkpoint)/'training_metrics.json').write_text(json.dumps(trace,indent=2)+'\n')
        if run:
            run.summary.update(metrics)
            from .tracking import log_artifact, publish
            try:
                identity = log_artifact(run, args.checkpoint, name=Path(args.checkpoint).name,
                    kind='model', metadata=config)
                run.summary['model_artifact'] = identity
            except Exception:
                # The saved checkpoint remains available even if remote storage is full.
                identity = publish(args.checkpoint, kind='model', project=args.project,
                    dataset_artifact=','.join(args.dataset_artifact or []), backend='mlflow')
                run.summary['model_fallback'] = identity
            print('W&B run: '+str(run.url),flush=True)
        failed=False
    finally:
        if run: run.finish(exit_code=1 if failed else 0)


def math_gradient_norm(model):
    return float(torch.stack([p.grad.float().norm()**2 for p in model.parameters()
                              if p.grad is not None]).sum().sqrt())


if __name__ == '__main__': main()
