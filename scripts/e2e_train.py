"""Durable full-corpus training, retained checkpoints, and three-model benchmarks."""
import argparse
import gc
import html
import json
import os
from pathlib import Path
import platform
import random
import shutil
import signal
import subprocess
import sys
import time

import torch
from octo import Encoder, Limits, OctoModel, StructuralTokens, make_optimizer, save_checkpoint, train_step
from octo.checkpoint import load_checkpoint
from octo.cli import read_records
from octo.dataset import digest
from octo.evaluation import evaluate
from octo.model import BASE_MODEL, BASE_REVISION

ROOT = Path('artifacts/runs/e2e_train_1')


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(data, indent=2) + '\n')
    temp.replace(path)


def status(phase, **details):
    write_json(ROOT / 'status.json', {'phase': phase, 'updated_unix': time.time(), 'pid': os.getpid(), **details})
    print(json.dumps({'phase': phase, **details}), flush=True)


def manifest(directory):
    return {str(p.relative_to(directory)): digest(p) for p in sorted(directory.rglob('*'))
            if p.is_file() and p.name != 'checkpoint_manifest.json'}


def checkpoint(model, encoder, optimizer, config, step, order, cursor, *, suffix=''):
    name = f'step-{step:08d}' + suffix
    target = ROOT / 'checkpoints' / name
    staging = target.with_name(name + '.saving')
    if target.exists():
        return target
    if staging.exists():
        # Preserve any incomplete save for inspection rather than deleting it.
        staging.rename(staging.with_name(staging.name + f'.interrupted-{time.time_ns()}'))
    save_checkpoint(staging, model, encoder, metadata={**config, 'step': step})
    state = {'optimizer': optimizer.state_dict(), 'step': step, 'order': order, 'cursor': cursor,
             'python_rng': random.getstate(), 'torch_rng': torch.get_rng_state()}
    if config['device'] == 'mps':
        state['mps_rng'] = torch.mps.get_rng_state()
    torch.save(state, staging / 'training_state.pt')
    write_json(staging / 'checkpoint_manifest.json', manifest(staging))
    staging.rename(target)
    write_json(ROOT / 'latest_checkpoint.json', {'path': str(target), 'step': step, 'cursor': cursor})
    return target


def report():
    config = json.loads((ROOT / 'config.json').read_text()) if (ROOT / 'config.json').exists() else {}
    state = json.loads((ROOT / 'status.json').read_text())
    labels = {'qwen-prompt': 'Qwen3 · prompted', 'jev': 'Jev 1.13.0', 'octo': 'Octo · trained'}
    comparisons = {}
    for domain in ('bfsi', 'chess'):
        path = ROOT / 'evaluations' / f'{domain}-comparison' / 'comparison.json'
        if path.exists():
            comparisons[domain] = json.loads(path.read_text())
    all_done = len(comparisons) == 2
    n = sum(d['scores']['octo']['records'] for d in comparisons.values())
    headline = 'The first full-corpus test.'
    subtitle = 'Training and evaluation in progress. Results will appear here after each frozen comparison completes.'
    takeaways = []
    if all_done:
        mean = {a: sum(d['scores'][a]['accuracy'] * d['scores'][a]['records'] for d in comparisons.values()) / n
                for a in labels}
        delta_q = (mean['octo'] - mean['qwen-prompt']) * 100
        delta_j = (mean['octo'] - mean['jev']) * 100
        headline = (f'Octo leads by {delta_j:.1f} points over Jev.' if delta_j > 0 and delta_q > 0 else
                    f'Octo leads by {delta_q:.1f} points over prompted Qwen3.' if delta_q > 0 else
                    'A measured starting point for Octo.')
        subtitle = f'Observed accuracy across {n:,} identical held-out decisions. BFSI and chess results are shown separately below.'
        takeaways.append(f'Combined observed accuracy: Octo {mean["octo"]:.1%}, Jev {mean["jev"]:.1%}, prompted Qwen3 {mean["qwen-prompt"]:.1%}. '
                         'The combined figure weights the 800-example BFSI suite four times more than the 200-example chess suite.')
        for domain, data in comparisons.items():
            for other in ('qwen-prompt', 'jev'):
                pair = next((v for k, v in data['paired'].items() if k == 'octo minus ' + other or k == other + ' minus octo'), None)
                if pair:
                    sign = 1 if 'octo minus ' + other in data['paired'] else -1
                    lo, hi = sorted(sign * x * 100 for x in pair['ci95'])
                    delta = sign * pair['accuracy_difference'] * 100
                    takeaways.append(f'{domain.upper()}: Octo vs {labels[other]} = {delta:+.1f} percentage points '
                                     f'(paired 95% interval {lo:+.1f} to {hi:+.1f}). ' +
                                     ('The interval includes zero; the difference is inconclusive.' if lo <= 0 <= hi else
                                      'The source-family bootstrap interval excludes zero.'))
    sections = ['<header><div class="brand">OCTO <span>RESEARCH / RUN 001</span></div><div class="status">' +
                html.escape(state['phase'].replace('_', ' ').upper()) + '</div></header>',
                '<section class="hero"><div class="eyebrow">E2E_TRAIN_1 · FROZEN BENCHMARK</div><h1>' + headline +
                '</h1><p>' + subtitle + '</p><div class="chips"><span>Same inputs, three models</span>'
                '<span>800 BFSI + 200 chess</span><span>All checkpoints retained</span></div></section>',
                '<section class="grid">']
    for domain in ('bfsi', 'chess'):
        sections += [f'<article class="card"><div class="eyebrow">{"FINANCIAL INTENT SELECTION" if domain == "bfsi" else "TACTICAL MOVE SELECTION"}</div><h2>{domain.upper()}</h2>']
        if domain not in comparisons:
            sections += ['<div class="pending">Awaiting measured results<span>No benchmark scores yet</span></div></article>']
            continue
        data = comparisons[domain]
        for name in ('qwen-prompt', 'jev', 'octo'):
            s = data['scores'][name]
            lo, hi = s['accuracy_ci95']
            sections += [f'<div class="metric {name}"><div class="metric-label"><b>{labels[name]}</b><strong>{s["accuracy"]:.1%}</strong></div>'
                         f'<div class="track"><div style="width:{s["accuracy"]*100:.3f}%"></div></div>'
                         f'<div class="range">95% CI {lo:.1%}–{hi:.1%} · valid {s["valid_response_rate"]:.1%}</div></div>']
        s = data['scores']['octo']
        sections += [f'<div class="card-note">N = {s["records"]:,} · chance = {s["uniform_accuracy"]:.1%} · errors count as wrong</div></article>']
    sections += ['</section><div class="method-strip">Pinned Qwen3-1.7B-Base · LoRA + pointer head · 1 full epoch · '
                 f'{config.get("training_records", 79336):,} training examples · fp32 / SDPA</div>',
                 '<div class="share-note">Screenshot the summary above. This file is self-contained and opens without a server. '
                 'A lead in observed accuracy is not a significance claim; paired confidence intervals are below.</div>',
                 '<section class="details"><h2>What the results tell us</h2>']
    if takeaways:
        sections += ['<ul>' + ''.join('<li>' + html.escape(t) + '</li>' for t in takeaways) + '</ul>']
    else:
        sections += ['<p>Measurements are pending. The report makes no performance claims before evaluation completes.</p>']
    if all_done:
        lower = [domain for domain, data in comparisons.items()
                 if any(data['scores']['octo']['accuracy'] <= data['scores'][other]['accuracy'] for other in ('qwen-prompt', 'jev'))]
        sections += ['<h2>Retrospective and the next experiment</h2>']
        if lower:
            sections += ['<p>Octo does not exceed both comparators on ' + ', '.join(d.upper() for d in lower) +
                         '. Treat the following as hypotheses for a new run, not fixes validated by this benchmark.</p><ul>'
                         '<li>Inspect development errors by source, intent, candidate count, and chess rating; separate probability '
                         'quality from response-format failures.</li><li>Try source-balanced sampling to reduce the BFSI-heavy training '
                         'mixture. Measure chess and each BFSI source independently.</li><li>Select learning rates and additional epochs '
                         'using development data, saving each epoch and choosing the checkpoint before a new locked evaluation.</li>'
                         '<li>For chess, test richer board representations and engine-ranked hard negatives on development positions.</li>'
                         '<li>If prompted Qwen3 has many invalid completions, add a separately named constrained-decoding baseline; '
                         'retain this frozen plain-prompt comparison.</li></ul>']
        else:
            sections += ['<p>Octo has higher observed accuracy on both suites. Replicate with additional seeds and a new held-out '
                         'source before making broad generalization claims. Inspect paired intervals and per-source results below.</p>']
    sections += ['<h2>Accuracy, validity, and operational latency</h2>']
    for domain, data in comparisons.items():
        sections += [f'<h3>{domain.upper()}</h3><div class="table-wrap"><table><tr><th>Model</th><th>Accuracy</th><th>Valid</th>'
                     '<th>Median / p95 latency</th><th>Macro source accuracy</th></tr>']
        for name, s in data['scores'].items():
            sections += [f'<tr><td>{labels[name]}</td><td>{s["accuracy"]:.2%}</td><td>{s["valid_response_rate"]:.2%}</td>'
                         f'<td>{s["latency_median_seconds"]:.3f}s / {s["latency_p95_seconds"]:.3f}s</td>'
                         f'<td>{s["macro_domain_accuracy"]:.2%}</td></tr>']
        sections += ['</table></div><details><summary>Paired differences, probability metrics, and source breakdown</summary><pre>' +
                     html.escape(json.dumps({'paired': data['paired'], 'scores': data['scores']}, indent=2)) + '</pre></details>']
    sections += ['<h2>How to read this comparison</h2><p>All models receive identical states, instructions, and neutral-letter '
                 'options. Answers are separate. Qwen3 is the same pinned Base model, using greedy zero-shot plain prompting; '
                 'an instruction-tuned model could behave differently. API errors and malformed answers count as wrong. '
                 'Intervals bootstrap source families. Local GPU and hosted network latencies measure different deployment paths.</p>'
                 '<p>BFSI template grouping is approximate and public sources may overlap pretraining. Chess measures selecting '
                 'a tactical puzzle solution supplied among legal candidates, not Elo or full-game strength. Core suites are '
                 'subsets of locked tests. Dollar costs are not inferred from missing billing data.</p>',
                 '<details><summary>Training configuration, hardware, and provenance</summary><pre>' +
                 html.escape(json.dumps(config, indent=2)) + '</pre></details>']
    if (ROOT / 'heldout_metrics.json').exists():
        sections += ['<details><summary>Complete development, calibration, and locked-test evaluation</summary><pre>' +
                     html.escape((ROOT / 'heldout_metrics.json').read_text()) + '</pre></details>']
    sections += ['<details><summary>Run state</summary><pre>' + html.escape(json.dumps(state, indent=2)) + '</pre></details>',
                 '</section><footer>OCTO / e2e_train_1 · Preserved weights. Reproducible inputs. Measured outcomes.</footer>']
    css = '''*{box-sizing:border-box}body{margin:0;background:#eff2ed;color:#182c28;font:16px/1.55 system-ui,-apple-system,sans-serif}
    main{max-width:1120px;margin:auto;padding:36px 42px}header{display:flex;justify-content:space-between;align-items:center}
    .brand{font-size:28px;font-weight:850;letter-spacing:-1px}.brand span{font-size:10px;letter-spacing:2px;margin-left:14px;color:#66746e}
    .status{font-size:10px;letter-spacing:1.4px;background:#dee6dc;border-radius:20px;padding:7px 12px}
    .hero{padding:42px 0 30px}.eyebrow{font-size:10px;font-weight:700;letter-spacing:1.8px;color:#6a796e}
    h1{font-size:clamp(32px,4.5vw,54px);line-height:1.1;letter-spacing:-2px;max-width:890px;margin:14px 0 18px}
    .hero p{max-width:800px;color:#53665f;font-size:17px}.chips{display:flex;gap:10px;flex-wrap:wrap;margin-top:24px}
    .chips span{border:1px solid #ced8cf;padding:5px 11px;border-radius:6px;font-size:11px}.grid{display:grid;grid-template-columns:1fr 1fr;gap:20px}
    .card{background:#fbfcf8;border:1px solid #d5ded3;border-radius:16px;padding:26px}.card h2{font-size:26px;margin:8px 0 24px}
    .metric{margin:17px 0}.metric-label{display:flex;justify-content:space-between;align-items:center;font-size:13px}
    .metric-label strong{font-size:24px;letter-spacing:-.8px}.track{height:10px;border-radius:10px;background:#e8eee5;margin:8px 0 5px;overflow:hidden}
    .track div{height:100%;background:#9eaaa2;border-radius:10px}.jev .track div{background:#c59a50}.octo .track div{background:#237663}
    .octo .metric-label{color:#1d6a57}.range{font-size:10px;color:#748076}.card-note{font-size:10px;color:#69786e;border-top:1px solid #e0e6dc;margin-top:23px;padding-top:13px}
    .method-strip{margin-top:21px;font-size:11px;text-align:center;border-top:1px solid #cad4c9;border-bottom:1px solid #cad4c9;padding:15px}
    .share-note{font-size:11px;color:#7b877e;margin:15px 0 45px;max-width:830px}.pending{height:230px;display:flex;flex-direction:column;justify-content:center;color:#66746e;font-size:20px}
    .pending span{font-size:12px;margin-top:10px}.details{border-top:1px solid #cad4c9;padding-top:22px}.details h2{font-size:24px;margin-top:36px}
    .details p,.details li{color:#4f645b}.details li{margin:12px 0}.table-wrap{overflow:auto}table{width:100%;border-collapse:collapse;font-size:13px}
    td,th{padding:12px 8px;border-bottom:1px solid #d0dacd;text-align:left}th{color:#69776e;font-size:11px}
    details{background:#e7ede3;margin:12px 0;border-radius:8px;padding:14px}summary{cursor:pointer;font-size:13px;font-weight:600}
    pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:11px}footer{font-size:10px;letter-spacing:1px;border-top:1px solid #cad4c9;margin-top:45px;padding-top:18px;color:#6d7a70}
    @media(max-width:720px){main{padding:24px}.grid{grid-template-columns:1fr}.brand span{display:none}h1{letter-spacing:-1px}}
    @media print{body{background:white}.share-note{display:none}.hero{padding-top:26px}main{padding:15px}details{break-inside:avoid}}'''
    (ROOT / 'report.html').write_text('<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" '
        'content="width=device-width,initial-scale=1"><title>Octo · e2e_train_1</title><style>' + css + '</style><main>' +
        ''.join(sections) + '</main></html>')


def train(device, resume, *, tracking_enabled=True):
    from transformers import AutoTokenizer
    from huggingface_hub import snapshot_download
    torch.set_num_threads(8)
    torch.manual_seed(17)
    random.seed(17)
    if device == 'mps' and not torch.backends.mps.is_available():
        raise RuntimeError('MPS GPU access is required for this run')
    paths = [Path('artifacts/datasets') / f'{d}-v2' for d in ('bfsi', 'chess')]
    config_path = ROOT / 'config.json'
    train_paths = [p / 'train.jsonl' for p in paths]
    config = {'run_name': 'e2e_train_1', 'epochs': 1, 'seed': 17, 'device': device, 'precision': 'float32',
              'backend': 'sdpa', 'base_model': BASE_MODEL, 'base_revision': BASE_REVISION,
              'batch_size': 1, 'adapter_lr': 1e-4, 'pointer_lr': 1e-3, 'checkpoint_every_steps': 1000,
              'train_file_sha256': {str(p): digest(p) for p in train_paths},
              'dataset_manifest_sha256': {str(p): digest(p / 'manifest.json') for p in paths},
              'benchmark_manifest_sha256': {d: digest(Path('artifacts/benchmarks') / f'{d}-v1/manifest.json') for d in ('bfsi', 'chess')},
              'runner_sha256': digest(__file__), 'platform': platform.platform(), 'torch': torch.__version__,
              'started_unix': time.time()}
    config['dataset_artifacts'] = [json.loads((Path('artifacts/tracking/receipts') / f'{d}-v2.json').read_text()) for d in ('bfsi', 'chess')]
    if config_path.exists():
        old = json.loads(config_path.read_text())
        for key in ('train_file_sha256', 'dataset_manifest_sha256', 'benchmark_manifest_sha256', 'device'):
            if old[key] != config[key]:
                raise ValueError('resume identity mismatch: ' + key)
        if old['runner_sha256'] != config['runner_sha256']:
            if (ROOT / 'latest_checkpoint.json').exists():
                raise ValueError('runner changed after a checkpoint was saved')
            old.setdefault('prelaunch_runner_sha256', []).append(old['runner_sha256'])
            old['runner_sha256'] = config['runner_sha256']
        config = old
    else:
        write_json(config_path, config)
    status('validating_data')
    records = [r for p in train_paths for r in read_records(p)]
    train_groups = {r['source_group_id'] for r in records}
    heldout = {f'{p.name}/{split}': read_records(p / f'{split}.jsonl')
               for p in paths for split in ('dev', 'calibration', 'test_locked')}
    for name, rows in heldout.items():
        if train_groups & {r['source_group_id'] for r in rows}:
            raise ValueError('source group leakage: ' + name)
    tokenizer = AutoTokenizer.from_pretrained('artifacts/tokenizers/qwen3', local_files_only=True)
    for name, expected in json.loads(Path('configs/bfsi_v2.json').read_text())['tokenizer_checksums'].items():
        if digest(Path('artifacts/tokenizers/qwen3') / name) != expected:
            raise ValueError('tokenizer checksum mismatch')
    tokens = StructuralTokens.from_tokenizer(tokenizer)
    encoder = Encoder(tokenizer, tokens, Limits(), tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id)
    for row in records:
        if any(q.target is None for q in encoder.encode(row).questions):
            raise ValueError('unlabeled training row')
    for rows in heldout.values():
        for row in rows:
            encoder.encode(row)
    config['training_records'] = len(records)
    config['heldout_records'] = {k: len(v) for k, v in heldout.items()}
    write_json(config_path, config)
    status('retaining_base_weights')
    base = ROOT / 'base_weights'
    if not (base / 'manifest.json').exists():
        cached = Path(snapshot_download(BASE_MODEL, revision=BASE_REVISION, local_files_only=True))
        shutil.copytree(cached, base, symlinks=False, dirs_exist_ok=True)
        write_json(base / 'manifest.json', {'model': BASE_MODEL, 'revision': BASE_REVISION, 'files': manifest(base)})
    if resume and (ROOT / 'latest_checkpoint.json').exists():
        saved = Path(json.loads((ROOT / 'latest_checkpoint.json').read_text())['path'])
        for name, expected in json.loads((saved / 'checkpoint_manifest.json').read_text()).items():
            if digest(saved / name) != expected:
                raise ValueError('checkpoint checksum mismatch')
        model, encoder, _ = load_checkpoint(saved, device=device)
        state = torch.load(saved / 'training_state.pt', map_location='cpu', weights_only=False)
        optimizer = make_optimizer(model)
        optimizer.load_state_dict(state['optimizer'])
        for values in optimizer.state.values():
            for key, value in values.items():
                if torch.is_tensor(value) and key != 'step':
                    values[key] = value.to(device)
        step, order, cursor = state['step'], state['order'], state['cursor']
        random.setstate(state['python_rng']); torch.set_rng_state(state['torch_rng'])
        if device == 'mps':
            torch.mps.set_rng_state(state['mps_rng'])
    else:
        model = OctoModel.from_pretrained(device=device, local_files_only=True)
        # Offline Transformers loading can omit this field. The snapshot above
        # is the explicitly pinned revision and its raw weights are retained.
        model.backbone.config._commit_hash = BASE_REVISION
        model.backbone.config._name_or_path = BASE_MODEL
        optimizer = make_optimizer(model)
        order = list(range(len(records))); random.shuffle(order)
        step = cursor = 0
        checkpoint(model, encoder, optimizer, config, step, order, cursor)
    run = None
    try:
        if not tracking_enabled:
            raise RuntimeError('external tracking disabled for local phase')
        from octo.tracking import start_run
        run = start_run(name='e2e_train_1', config=config, directory=str(ROOT / 'tracking'))
        for receipt in config['dataset_artifacts']:
            if receipt['backend'] == 'wandb':
                run.use_artifact(receipt['artifact'], type='dataset')
        write_json(ROOT / 'tracking_run.json', {'url': run.url, 'id': run.id})
    except Exception as exc:
        write_json(ROOT / 'tracking_error.json', {'error': type(exc).__name__, 'local_files_retained': True})
        if run:
            run.finish(exit_code=1)
        run = None
    stopped = False
    def stop(*_):
        nonlocal stopped
        stopped = True
    signal.signal(signal.SIGTERM, stop); signal.signal(signal.SIGINT, stop)
    status('training', step=step, total_steps=len(records))
    report()
    try:
        with (ROOT / 'training_metrics.jsonl').open('a') as log:
            while cursor < len(order):
                start = time.perf_counter()
                loss = train_step(model, encoder.batch([records[order[cursor]]]).to(device), optimizer)
                cursor += 1; step += 1
                metrics = {'step': step, 'loss': loss, 'step_seconds': time.perf_counter() - start}
                log.write(json.dumps(metrics) + '\n'); log.flush()
                if run:
                    run.log({'train/loss': loss, 'train/step_seconds': metrics['step_seconds']}, step=step)
                if step % 25 == 0:
                    status('training', step=step, total_steps=len(records), **{k:v for k,v in metrics.items() if k != 'step'})
                if step % 1000 == 0 or stopped:
                    checkpoint(model, encoder, optimizer, config, step, order, cursor)
                    report()
                if stopped:
                    raise InterruptedError('training interrupted; checkpoint retained')
        final = checkpoint(model, encoder, optimizer, config, step, order, cursor, suffix='-final')
        write_json(ROOT / 'final_checkpoint.json', {'path': str(final)})
        if run:
            run.finish(); run = None
        status('heldout_evaluation', step=step, checkpoint=str(final))
        scores = {}
        for name, rows in heldout.items():
            scores[name] = evaluate(model, (encoder.batch([r]).to(device) for r in rows))
            write_json(ROOT / 'heldout_metrics.json', scores)
            status('heldout_evaluation', completed_split=name, checkpoint=str(final))
        write_json(ROOT / 'training_complete.json', {'checkpoint': str(final), 'step': step})
        return final
    except BaseException:
        checkpoint(model, encoder, optimizer, config, step, order, cursor, suffix='-interrupted')
        raise
    finally:
        if run:
            run.finish(exit_code=1)


def command(arguments):
    subprocess.run([sys.executable, '-m', 'octo.benchmark', *arguments], check=True)


def benchmark(final, device, *, allow_jev=True):
    for domain in ('bfsi', 'chess'):
        root = Path('artifacts/benchmarks') / f'{domain}-v1'
        outputs = []
        for adapter in (('qwen-prompt', 'jev', 'octo') if allow_jev else ('qwen-prompt', 'octo')):
            status('benchmarking', domain=domain, adapter=adapter)
            out = ROOT / 'evaluations' / f'{domain}-{adapter}'
            args = ['run', '--benchmark', str(root), '--suite', 'core', '--adapter', adapter,
                    '--device', device, '--output', str(out)]
            if adapter == 'octo':
                args += ['--checkpoint', str(final)]
            if out.exists():
                args += ['--resume']
            command(args); outputs.append(str(out))
        comparison = ROOT / 'evaluations' / f'{domain}-comparison'
        if allow_jev and not (comparison / 'manifest.json').exists():
            command(['compare', '--benchmark', str(root), '--suite', 'core', '--predictions', *outputs,
                     '--output', str(comparison)])
        report()


def publish(final):
    from octo.tracking import publish as publish_artifact
    # Publish inference weights without duplicating the optimizer in cloud storage.
    inference = ROOT / 'model_artifact'
    if not inference.exists():
        shutil.copytree(final, inference, ignore=shutil.ignore_patterns('training_state.pt', 'checkpoint_manifest.json'))
    status('publishing')
    config = json.loads((ROOT / 'config.json').read_text())
    model_receipt = (json.loads((ROOT / 'model_receipt.json').read_text()) if (ROOT / 'model_receipt.json').exists()
                     else publish_artifact(inference, kind='model', name='e2e_train_1',
                         dataset_artifact=next((r.get('artifact') for r in config['dataset_artifacts'] if r.get('artifact')), None)))
    write_json(ROOT / 'model_receipt.json', model_receipt)
    for domain in ('bfsi', 'chess'):
        if (ROOT / f'{domain}_evaluation_receipt.json').exists():
            continue
        dataset = next((r.get('artifact') for r in config['dataset_artifacts'] if domain in r.get('artifact', '')), None)
        benchmark_receipt = Path('artifacts/tracking/receipts') / f'{domain}-benchmark-v1.json'
        benchmark_id = json.loads(benchmark_receipt.read_text()).get('artifact') if benchmark_receipt.exists() else None
        receipt = publish_artifact(ROOT / 'evaluations' / f'{domain}-comparison', kind='evaluation',
            name=f'e2e_train_1-{domain}-comparison', dataset_artifact=dataset, benchmark_artifact=benchmark_id,
            model_artifact=model_receipt.get('artifact'))
        write_json(ROOT / f'{domain}_evaluation_receipt.json', receipt)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='mps'); parser.add_argument('--resume', action='store_true')
    parser.add_argument('--report-only', action='store_true')
    parser.add_argument('--local-only', action='store_true', help='No Jev calls or external artifact uploads')
    args = parser.parse_args()
    if args.report_only:
        report()
        return
    if ROOT.exists() and not args.resume:
        raise ValueError('run already exists; use --resume')
    ROOT.mkdir(parents=True, exist_ok=True)
    import fcntl
    lock = (ROOT / 'run.lock').open('a')
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    try:
        if (ROOT / 'training_complete.json').exists():
            final = Path(json.loads((ROOT / 'training_complete.json').read_text())['checkpoint'])
        else:
            final = train(args.device, args.resume, tracking_enabled=not args.local_only)
        gc.collect()
        if args.device == 'mps':
            torch.mps.empty_cache()
        benchmark(final, args.device, allow_jev=not args.local_only)
        if args.local_only:
            status('awaiting_external_approval', checkpoint=str(final), report=str(ROOT / 'report.html'))
        else:
            publish(final)
            status('complete', checkpoint=str(final), report=str(ROOT / 'report.html'))
    except BaseException as exc:
        status('interrupted' if isinstance(exc, (KeyboardInterrupt, InterruptedError)) else 'failed', error=type(exc).__name__)
        raise
    finally:
        report()


if __name__ == '__main__':
    main()
