"""Render evidence from the ongoing run's read-only loss investigation."""
import base64
import html
import json
from pathlib import Path
import statistics

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path('artifacts/runs/e2e_train_1')
OUT = ROOT / 'investigation'


def main():
    stats = json.loads((OUT / 'loss_statistics.json').read_text())
    probes = json.loads((OUT / 'development_probe.json').read_text())
    entries = [json.loads(line) for line in (ROOT / 'training_metrics.jsonl').read_text().splitlines()][:stats['step_count']]
    checkpoints = sorted(probes.values(), key=lambda v: v['step'])
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.5), constrained_layout=True)
    fig.suptitle('e2e_train_1 · loss investigation', fontsize=20, weight='bold')
    x = [r['step'] for r in entries]; y = [r['loss'] for r in entries]
    axes[0, 0].scatter(x, y, s=2, color='#adb9b2', alpha=.5, label='Individual example')
    smooth_x = []; smooth_y = []
    for start in range(0, len(entries), 200):
        block = entries[start:start+200]
        smooth_x.append(block[-1]['step']); smooth_y.append(statistics.mean(r['loss'] for r in block))
    axes[0, 0].plot(smooth_x, smooth_y, color='#217563', linewidth=2, label='200-step mean')
    axes[0, 0].set(yscale='symlog', xlabel='Training step', ylabel='Loss (symmetric log scale)', title='Spikes persist; average loss falls')
    axes[0, 0].legend(fontsize=9)
    domains = list(stats['domains'])
    axes[0, 1].barh(domains, [stats['domains'][d]['mean'] for d in domains], color=['#718c83']*4+['#c7984e'])
    axes[0, 1].set(xlabel='Mean per-example training loss', title='Chess and BANKING77 are harder')
    steps = [r['step'] for r in checkpoints]
    for label, color in [('overall', '#217563'), ('banking77', '#637aa1'), ('chess', '#c7984e')]:
        values = [r['overall'] if label == 'overall' else r['by_domain'][label] for r in checkpoints]
        axes[1, 0].plot(steps, [r['accuracy']*100 for r in values], marker='o', label=label, color=color)
    axes[1, 0].set(xlabel='Saved checkpoint step', ylabel='Development accuracy (%)', ylim=(0, 105), title='Fixed sample: 40 examples per source')
    axes[1, 0].legend()
    axes[1, 1].plot(steps, [r['overall']['nll'] for r in checkpoints], color='#217563', marker='o', label='Mean development loss')
    axes[1, 1].set(xlabel='Saved checkpoint step', ylabel='Mean development loss', title='Loss improves despite confident mistakes')
    secondary = axes[1, 1].twinx()
    secondary.plot(steps, [r['overall']['mean_confidence_when_wrong']*100 if r['overall']['mean_confidence_when_wrong'] is not None else float('nan') for r in checkpoints], color='#b86859', marker='s', linestyle='--')
    secondary.set(ylabel='All-source confidence when wrong (%)', ylim=(0, 105))
    axes[1, 1].legend(loc='center left')
    fig.savefig(OUT / 'loss_diagnostics.png', dpi=160)
    plt.close(fig)
    latest = checkpoints[-1]
    wrong_count = sum(not r['correct'] for r in latest['rows'] if r['domain'] == 'banking77')
    bank_confidence = latest['by_domain']['banking77']['mean_confidence_when_wrong']
    confidence_text = (f'Incorrect BANKING77 predictions average {bank_confidence:.1%} confidence ({wrong_count} errors out of 40 examples).'
                       if bank_confidence is not None else 'There are no BANKING77 errors in the latest 40-example sample, so confidence on its errors cannot be estimated.')
    windows = stats.get('matched_windows', {})
    recent = windows.get('9001-18000')
    trend_text = (f'Comparing equal 9,000-step windows, mean loss falls from {windows["1-9000"]["overall"]["mean"]:.3f} '
                  f'to {recent["overall"]["mean"]:.3f}; the fraction of losses above 10 falls from '
                  f'{windows["1-9000"]["overall"]["loss_gt10_fraction"]:.2%} to {recent["overall"]["loss_gt10_fraction"]:.2%}.'
                  if recent else 'The first 1,000-step average loss was 1.218; steps 8,001–9,000 averaged 0.373.')
    timing = stats.get('timing')
    timing_text = (f'<h2>Step-time spikes</h2><p>Across {timing["steps"]:,} recent examples, input length and step duration have '
                   f'Pearson correlation {timing["token_time_pearson"]:.3f}. Chess median length is '
                   f'{timing["by_domain"]["chess"]["median_tokens"]:.0f} tokens and median step duration '
                   f'{timing["by_domain"]["chess"]["median_seconds"]:.3f}s, versus approximately 60 tokens and 0.22s '
                   'for the BFSI sources. This is a descriptive association; occasional contention and startup also affect timings.</p>'
                   if timing else '')
    expanded_text = ''
    expanded_path = OUT / 'expanded-18000' / 'comparison.json'
    if expanded_path.exists():
        expanded = json.loads(expanded_path.read_text())
        expanded_rows = ''.join(f'<tr><td>{r["step"]:,}</td><td>{r["overall"]["accuracy"]:.1%}</td>'
                                f'<td>{r["overall"]["nll"]:.3f}</td><td>{r["by_domain"]["chess"]["accuracy"]:.1%}</td></tr>' for r in expanded['checkpoints'])
        expanded_text = f'''<h2>Larger confirmation: 1,000 development examples</h2><p>200 distinct source families per source,
        same requests at each checkpoint. Still development data; no locked benchmarks used.</p>
        <table><tr><th>Step</th><th>Accuracy</th><th>Average loss</th><th>Chess accuracy</th></tr>{expanded_rows}</table>
        <p>Paired accuracy difference: {expanded['paired']['accuracy_difference']*100:+.2f} percentage points
        (source-stratified family bootstrap 95% interval {expanded['paired']['ci95'][0]*100:+.2f} to {expanded['paired']['ci95'][1]*100:+.2f}).
        {expanded['paired']['wins']} previously wrong examples become correct; {expanded['paired']['losses']} previously correct examples become wrong.</p>'''
    rows = ''.join(f'<tr><td>{r["step"]:,}</td><td>{r["overall"]["accuracy"]:.1%}</td>'
                   f'<td>{r["overall"]["nll"]:.3f}</td><td>{r["overall"]["median_logit_range"]:.2f}</td></tr>' for r in checkpoints)
    image = base64.b64encode((OUT / 'loss_diagnostics.png').read_bytes()).decode()
    text = f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
    <title>Octo loss investigation</title><style>body{{background:#f2f5ef;color:#19372e;font:16px/1.6 system-ui;margin:36px auto;max-width:1120px;padding:0 24px}}
    h1{{font-size:42px;line-height:1.15}}img{{width:100%;border-radius:12px}}table{{border-collapse:collapse;width:100%}}td,th{{text-align:left;padding:10px;border-bottom:1px solid #ccd8cb}}
    pre{{white-space:pre-wrap;font-size:11px}}.note{{padding:15px;background:#e4ece0;border-radius:10px}}li{{margin:10px 0}}</style>
    <p>OCTO / E2E_TRAIN_1 / DEVELOPMENT DIAGNOSTIC</p><h1>Improving accuracy, with overconfident errors.</h1>
    <p>Investigated the first {stats['step_count']:,} steps. Training continues with its original settings. No locked-test or Jev results were used.</p>
    <img src="data:image/png;base64,{image}" alt="Training loss, source loss, and development checkpoint comparisons">
    <h2>What the evidence shows</h2><ul>
    <li>No non-finite losses among the inspected steps. The trainer checks finite logits, loss, and gradients before each update and clips the gradient norm to 1.</li>
    <li>{trend_text}</li>
    <li>{stats['overall']['exact_zero_fraction']:.1%} of steps have exactly zero fp32 loss. This reflects saturated probabilities on easy examples, not proof of perfect generalization.</li>
    <li>{stats['overall']['loss_gt10_fraction']:.1%} of steps exceed loss 10. Chess is only {stats['domains']['chess']['steps']/stats['step_count']:.1%} of inspected examples, but has mean loss {stats['domains']['chess']['mean']:.2f}.</li>
    <li>On a fixed source-balanced development sample of 200 distinct families, accuracy reaches {latest['overall']['accuracy']:.1%} at step {latest['step']:,}. This is a small diagnostic sample, not the final benchmark.</li>
    <li>The median candidate-logit range grows from {checkpoints[0]['overall']['median_logit_range']:.2f} before training to {latest['overall']['median_logit_range']:.2f}. {confidence_text}</li></ul>
    <table><tr><th>Checkpoint</th><th>Development accuracy</th><th>Development loss</th><th>Median logit range</th></tr>{rows}</table>
    {expanded_text}{timing_text}
    <h2>Why an individual loss spikes</h2><p>Each update uses one example. Shuffling changes the source, difficulty, and candidate count at each step.
    For a hard correct label, cross-entropy is −log of its predicted probability: near-certain correct answers have near-zero loss;
    a confidently wrong answer can have very large loss. Gradient clipping limits gradients, not the reported loss.
    See <a href="https://docs.pytorch.org/docs/2.14/generated/torch.nn.CrossEntropyLoss.html">PyTorch cross-entropy</a> and
    <a href="https://docs.pytorch.org/docs/2.14/generated/torch.nn.utils.clip_grad_norm_.html">gradient clipping</a>.</p>
    <h2>Source-label audit</h2><p>Some high-loss source annotations conflict with the text or omit necessary context. The pinned original BANKING77 parquet was checked directly:
    row 5268 says “my card was stolen” but is labeled lost_or_stolen_phone (training step 1950, loss 81.96).
    Row 2903 asks how to top up by card but is labeled supported_cards_and_currencies (step 5024, loss 85.23).
    These labels were inherited from the source, not introduced by the builder. This small targeted review does not estimate the overall noise rate.
    No labels or test answers have been changed.</p>
    <h2>What to change in a separate experiment</h2><ul>
    <li>Use accumulated gradients across 8–16 examples and monitor source-specific mean loss, accuracy, logit scale, and the fraction of updates clipped.</li>
    <li>Test a smaller pointer-head learning rate (the current rate is 1e-3), warmup, and mild label smoothing on development data. The learning-rate contribution has not been isolated experimentally.</li>
    <li>Audit noisy or ambiguous source rows with preserved provenance and documented exclusions; improve vague candidate descriptions.</li>
    <li>Give chess its own development trend and consider source-balanced sampling. Its 40-example sampled accuracy was 70% at step 9,000 and {latest['by_domain']['chess']['accuracy']:.1%} at step {latest['step']:,}.</li>
    <li>Calibrate confidence on the separate calibration split after training. Preserve this run as the unchanged first-run reference.</li></ul>
    <p class="note">Conclusion: spikiness comes from single-example updates, mixed task difficulty, source-label ambiguity, and saturated predictions.
    Accuracy and average loss are improving on the inspected development sample. Overconfidence is a real quality issue to track; it is not evidence here of numerical divergence.</p>
    <details><summary>Full statistics</summary><pre>{html.escape(json.dumps(stats, indent=2))}</pre></details></html>'''
    (OUT / 'report.html').write_text(text)
    print(str(OUT / 'report.html'))


if __name__ == '__main__':
    main()
