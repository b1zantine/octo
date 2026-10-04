"""Disposable CUDA feasibility probe; never saves trained weights or starts tracking."""
import gc
import json
import os
from pathlib import Path
import random
import statistics
import sys
import time
import tarfile

root = Path('/content/octo')
root.mkdir(exist_ok=True)
with tarfile.open('/content/octo-colab-preflight.tar.gz') as archive:
    archive.extractall(root, filter='data')
os.chdir(root)
sys.path.insert(0, str(root / 'src'))
import torch
from transformers import AutoTokenizer
from octo import Encoder, Limits, OctoModel, StructuralTokens, make_optimizer, train_step
from octo.cli import read_records
from octo.model import BASE_MODEL, BASE_REVISION

torch.manual_seed(17)
random.seed(17)
torch.set_num_threads(8)
# Match the fp32 baseline strictly; measure CUDA without TF32 changing its arithmetic.
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
tokenizer = AutoTokenizer.from_pretrained('artifacts/tokenizers/qwen3', local_files_only=True)
encoder = Encoder(tokenizer, StructuralTokens.from_tokenizer(tokenizer), Limits(), tokenizer.pad_token_id or tokenizer.eos_token_id)
records = []
for domain in ('bfsi', 'chess'):
    records.extend(read_records(f'artifacts/datasets/{domain}-v2/train.jsonl'))
lengths = [len(encoder.encode(r).input_ids) for r in records]
print(json.dumps({'phase': 'data_encoded', 'records': len(records), 'max_tokens': max(lengths)}), flush=True)
model = OctoModel.from_pretrained(device='cuda', dtype=torch.float32, backend='sdpa')
optimizer = make_optimizer(model)
report = {'gpu': torch.cuda.get_device_name(), 'vram_gib': torch.cuda.get_device_properties(0).total_memory / 2**30,
          'torch': torch.__version__, 'base_model': BASE_MODEL, 'revision': BASE_REVISION,
          'precision': 'float32', 'backend': 'sdpa', 'batch_size': 1, 'epochs': 1,
          'training_records': len(records), 'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
          'scope': 'disposable optimizer steps only; no checkpoint or tracking run', 'steps': []}
# Stratify by source and candidate count, then test the longest actual request.
strata = {}
for i, r in enumerate(records):
    strata.setdefault((r['domain'], len(r['questions'][0]['options'])), []).append(i)
indices = [random.choice(v) for v in strata.values()] + [max(range(len(lengths)), key=lengths.__getitem__)]
# Warm up without retaining this model for the eventual run.
for i in indices[:2]:
    train_step(model, encoder.batch([records[i]]).to('cuda'), optimizer)
torch.cuda.synchronize()
for i in indices:
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    loss = train_step(model, encoder.batch([records[i]]).to('cuda'), optimizer)
    torch.cuda.synchronize()
    item = {'domain': records[i]['domain'], 'candidates': len(records[i]['questions'][0]['options']),
            'tokens': lengths[i], 'seconds': time.perf_counter()-start, 'loss': loss,
            'peak_allocated_gib': torch.cuda.max_memory_allocated()/2**30,
            'peak_reserved_gib': torch.cuda.max_memory_reserved()/2**30}
    report['steps'].append(item)
    print(json.dumps(item), flush=True)
measured = {(v['domain'],v['candidates']):v['seconds'] for v in report['steps'][:-1]}
report['estimated_training_hours'] = sum(len(v)*measured[k] for k,v in strata.items())/3600
report['median_step_seconds'] = statistics.median(v['seconds'] for v in report['steps'])
report['max_peak_allocated_gib'] = max(v['peak_allocated_gib'] for v in report['steps'])
report['finite_nonzero_gradients'] = all(p.grad is None or torch.isfinite(p.grad).all().item() for p in model.parameters()) and any(p.grad is not None and p.grad.abs().max().item()>0 for p in model.parameters())
output = root / 'artifacts/runs/octo_v1_colab_1'
output.mkdir(parents=True, exist_ok=True)
(output / 'gpu_preflight.json').write_text(json.dumps(report,indent=2)+'\n')
print('PREFLIGHT_RESULT='+json.dumps(report), flush=True)
del optimizer, model, records
gc.collect()
torch.cuda.empty_cache()
