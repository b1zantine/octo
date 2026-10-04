"""Read-only checkpoint probes on development examples, never locked tests."""
import argparse
from collections import defaultdict
import gc
import json
from pathlib import Path
import random
import statistics

import torch
from octo.checkpoint import load_checkpoint
from octo.cli import read_records
from octo.dataset import digest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoints', nargs='+', required=True)
    p.add_argument('--device', default='mps')
    p.add_argument('--output', default='artifacts/runs/e2e_train_1/investigation')
    p.add_argument('--per-source', type=int, default=40)
    p.add_argument('--append', action='store_true')
    args = p.parse_args()
    torch.set_num_threads(4)
    output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
    grouped = defaultdict(list)
    for domain in ('bfsi', 'chess'):
        for row in read_records(Path('artifacts/datasets') / f'{domain}-v2/dev.jsonl'):
            grouped[row['domain']].append(row)
    rng = random.Random(917)
    sample = []
    for domain, rows in sorted(grouped.items()):
        rng.shuffle(rows)
        seen = set()
        for row in rows:
            if row['source_group_id'] not in seen:
                sample.append(row); seen.add(row['source_group_id'])
            if len(seen) == args.per_source:
                break
    ids = [r['record_id'] for r in sample]
    identity_path = output / 'development_probe_ids.json'
    if args.append and identity_path.exists() and json.loads(identity_path.read_text()) != ids:
        raise ValueError('development sample identity changed')
    identity_path.write_text(json.dumps(ids, indent=2) + '\n')
    results_path = output / 'development_probe.json'
    results = json.loads(results_path.read_text()) if args.append and results_path.exists() else {}
    for path in args.checkpoints:
        print('Loading ' + path, flush=True)
        model, encoder, metadata = load_checkpoint(path, device=args.device)
        model.eval()
        rows = []
        with torch.inference_mode():
            for i, record in enumerate(sample):
                batch = encoder.batch([record]).to(args.device)
                result = model(batch)
                count = len(record['questions'][0]['options'])
                logits = result.logits[0, :count].float().cpu()
                prob = result.probabilities[0, :count].float().cpu()
                label = record['questions'][0]['label_id']
                labels = [o['id'] for o in record['questions'][0]['options']]
                target = labels.index(label)
                prediction = int(prob.argmax())
                rows.append({'record_id': record['record_id'], 'domain': record['domain'],
                    'correct': prediction == target, 'nll': float(-logits.log_softmax(-1)[target]),
                    'confidence': float(prob.max()), 'target_probability': float(prob[target]),
                    'entropy': float(-torch.special.xlogy(prob, prob).sum()),
                    'logit_range': float(logits.max() - logits.min()),
                    'candidate_hidden_rms': float(result.candidate_states.square().mean().sqrt()),
                    'decision_hidden_rms': float(result.decision_states.square().mean().sqrt())})
                if (i + 1) % 50 == 0:
                    print(f'{Path(path).name}: {i+1}/{len(sample)}', flush=True)
        def summarize(entries):
            wrong = [r for r in entries if not r['correct']]
            return {'records': len(entries), 'accuracy': statistics.mean(r['correct'] for r in entries),
                    'nll': statistics.mean(r['nll'] for r in entries),
                    'mean_confidence': statistics.mean(r['confidence'] for r in entries),
                    'mean_confidence_when_wrong': statistics.mean(r['confidence'] for r in wrong) if wrong else None,
                    'median_logit_range': statistics.median(r['logit_range'] for r in entries),
                    'max_logit_range': max(r['logit_range'] for r in entries),
                    'median_candidate_hidden_rms': statistics.median(r['candidate_hidden_rms'] for r in entries),
                    'median_decision_hidden_rms': statistics.median(r['decision_hidden_rms'] for r in entries)}
        values = {'step': metadata['step'], 'checkpoint_sha256': digest(Path(path) / 'octo.json'),
                  'overall': summarize(rows), 'by_domain': {d: summarize([r for r in rows if r['domain'] == d]) for d in sorted(grouped)},
                  'pointer_query_frobenius': float(model.query.weight.norm()), 'pointer_key_frobenius': float(model.key.weight.norm()),
                  'rows': rows}
        results[Path(path).name] = values
        (output / 'development_probe.json').write_text(json.dumps(results, indent=2) + '\n')
        print(json.dumps({k:v for k,v in values.items() if k != 'rows'}, indent=2), flush=True)
        del model, encoder, result, batch
        gc.collect()
        if args.device == 'mps':
            torch.mps.empty_cache()


if __name__ == '__main__':
    main()
