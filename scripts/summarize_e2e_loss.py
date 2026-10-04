"""Join recorded losses with the saved shuffle order and input lengths."""
from collections import defaultdict
import json
import math
from pathlib import Path
import statistics
import torch
import numpy as np
from transformers import AutoTokenizer
from octo import Encoder, Limits, StructuralTokens

ROOT = Path('artifacts/runs/e2e_train_1')
OUT = ROOT / 'investigation'


def stats(items):
    values = [r['loss'] for r in items]
    ordered = sorted(values)
    return {'steps': len(values), 'mean': statistics.mean(values), 'median': statistics.median(values),
            'p95': ordered[int(.95*(len(values)-1))], 'p99': ordered[int(.99*(len(values)-1))],
            'max': max(values), 'exact_zero_fraction': sum(x == 0 for x in values)/len(values),
            'loss_gt10_fraction': sum(x > 10 for x in values)/len(values),
            'nonfinite': sum(not math.isfinite(x) for x in values)}


def main():
    rows = [json.loads(line) for line in (ROOT / 'training_metrics.jsonl').read_text().splitlines() if line]
    checkpoint = Path(json.loads((ROOT / 'latest_checkpoint.json').read_text())['path'])
    state = torch.load(checkpoint / 'training_state.pt', map_location='cpu', weights_only=False)
    data = [json.loads(line) for domain in ('bfsi', 'chess')
            for line in (Path('artifacts/datasets') / f'{domain}-v2/train.jsonl').read_text().splitlines()]
    joined = [{**row, 'domain': data[state['order'][row['step']-1]]['domain'],
               'record_id': data[state['order'][row['step']-1]]['record_id'],
               'candidate_count': len(data[state['order'][row['step']-1]]['questions'][0]['options'])}
              for row in rows]
    output = {'step_count': len(rows), 'overall': stats(joined), 'windows': {}, 'domains': {},
              'largest_spikes': sorted(joined, key=lambda row: row['loss'], reverse=True)[:25]}
    for start in range(0, len(rows), 1000):
        output['windows'][f'{start+1}-{min(start+1000,len(rows))}'] = stats(joined[start:start+1000])
    for domain in sorted({r['domain'] for r in joined}):
        output['domains'][domain] = stats([r for r in joined if r['domain'] == domain])
    for start, stop in ((0, 9000), (9000, 18000)):
        if len(rows) >= stop:
            selected = joined[start:stop]
            output.setdefault('matched_windows', {})[f'{start+1}-{stop}'] = {
                'overall': stats(selected),
                'by_domain': {domain: stats([r for r in selected if r['domain'] == domain]) for domain in output['domains']}}
    tokenizer = AutoTokenizer.from_pretrained('artifacts/tokenizers/qwen3', local_files_only=True)
    encoder = Encoder(tokenizer, StructuralTokens.from_tokenizer(tokenizer), Limits(), tokenizer.eos_token_id)
    # Restrict timing correlations to normal recent steps; startup and the first
    # diagnostic GPU probe are outside this window. Later GPU contention remains
    # a limitation, so correlations are descriptive rather than causal.
    timings = []
    for row in joined[11000:min(18000,len(joined))]:
        record = data[state['order'][row['step']-1]]
        length = len(encoder.encode(record).input_ids)
        timings.append({**row, 'physical_tokens': length})
    if timings:
        output['timing'] = {'steps': len(timings), 'window': '11001-18000',
            'token_time_pearson': float(np.corrcoef([r['physical_tokens'] for r in timings], [r['step_seconds'] for r in timings])[0, 1]),
            'loss_time_pearson': float(np.corrcoef([r['loss'] for r in timings], [r['step_seconds'] for r in timings])[0, 1]),
            'by_domain': {domain: {'median_seconds': statistics.median(r['step_seconds'] for r in timings if r['domain'] == domain),
                                 'median_tokens': statistics.median(r['physical_tokens'] for r in timings if r['domain'] == domain)}
                          for domain in output['domains']}}
    OUT.mkdir(exist_ok=True)
    (OUT / 'loss_statistics.json').write_text(json.dumps(output, indent=2) + '\n')
    print(json.dumps({k:v for k,v in output.items() if k not in ('largest_spikes','windows')}, indent=2))


if __name__ == '__main__':
    main()
