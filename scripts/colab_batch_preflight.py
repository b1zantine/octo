"""Compare CUDA batch throughput on disposable weights; do not start a training run."""
import gc, json, os, random, statistics, sys, time
from pathlib import Path
os.chdir('/content/octo');sys.path.insert(0,'/content/octo/src')
import torch
from transformers import AutoTokenizer
from octo import Encoder,Limits,StructuralTokens,OctoModel,make_optimizer,train_step
from octo.cli import read_records

torch.set_num_threads(8);torch.manual_seed(17);random.seed(17)
torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
t=AutoTokenizer.from_pretrained('artifacts/tokenizers/qwen3',local_files_only=True)
e=Encoder(t,StructuralTokens.from_tokenizer(t),Limits(),t.pad_token_id or t.eos_token_id)
model=OctoModel.from_pretrained(device='cuda',local_files_only=True)
opt=make_optimizer(model)
report={'scope':'disposable batch benchmarks, no retained training weights','precision':'float32','tf32':False,'results':[]}
try:
 for domain in ('bfsi','chess'):
  records=read_records(f'artifacts/datasets/{domain}-v2/train.jsonl')
  # Random examples reflect this domain; force the longest into every batch as a memory stress check.
  lengths=[len(e.encode(r).input_ids) for r in records]
  longest=records[max(range(len(lengths)),key=lengths.__getitem__)]
  random.shuffle(records)
  prior=None
  for size in (1,8,16,32,64,128):
   if prior and 6.6+(prior['peak_allocated_gib']-6.6)*size/prior['batch_size']>76:
    print(json.dumps({'domain':domain,'batch_size':size,'skipped':'estimated peak above 80% VRAM'}),flush=True);break
   samples=[longest]+records[:size-1]
   b=None
   try:
    gc.collect();torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
    b=e.batch(samples).to('cuda')
    train_step(model,b,opt);torch.cuda.synchronize()
    times=[];losses=[]
    for _ in range(3):
     start=time.perf_counter();losses.append(train_step(model,b,opt));torch.cuda.synchronize()
     times.append(time.perf_counter()-start)
    seconds=statistics.median(times)
    prior={'domain':domain,'batch_size':size,'padded_tokens':b.input_ids.shape[1],
           'step_seconds':seconds,'records_per_second':size/seconds,
           'peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30,
           'peak_reserved_gib':torch.cuda.max_memory_reserved()/2**30,'losses':losses}
    report['results'].append(prior);print(json.dumps(prior),flush=True)
   except torch.cuda.OutOfMemoryError:
    report['results'].append({'domain':domain,'batch_size':size,'oom':True})
    print(json.dumps(report['results'][-1]),flush=True);break
   finally:
    opt.zero_grad(set_to_none=True);del b;gc.collect();torch.cuda.empty_cache()
  del records
 out=Path('artifacts/runs/octo_v1_colab_1/batch_preflight.json')
 out.write_text(json.dumps(report,indent=2)+'\n')
 print('BATCH_PREFLIGHT_COMPLETE',flush=True)
finally:
 del opt,model
 gc.collect();torch.cuda.empty_cache()
