"""Render the frozen Colab checkpoint's locally executed core benchmark report."""
import html
import json
import platform
import base64
from pathlib import Path
from datetime import datetime, timezone

ROOT = Path('artifacts/runs/octo_v1_colab_1')
LABELS = {'octo': 'Octo · trained', 'qwen-prompt': 'Qwen3 Base · prompted', 'jev': 'Jev 1.13.0 · hosted'}

def table(headers, rows):
    return '<div class="table-wrap"><table><thead><tr>'+''.join('<th>'+html.escape(str(x))+'</th>' for x in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+html.escape(str(x))+'</td>' for x in row)+'</tr>' for row in rows)+'</tbody></table></div>'

def percent(v):
    return f'{v:.2%}'

def main():
    evaluations=ROOT/'evaluations'
    hardware=json.loads((evaluations/'hardware.json').read_text())
    data={'generated_utc':datetime.now(timezone.utc).isoformat(),'hardware':hardware,'domains':{}}
    data['input_validation']=json.loads((evaluations/'input_validation.json').read_text())
    sections=[]
    for domain,title in [('bfsi','BFSI · 800 locked examples'),('chess','Chess · 200 locked examples')]:
        comparison=evaluations/f'{domain}-comparison'/'comparison.json'
        if not comparison.exists():
            sections.append(f'<section><h2>{title}</h2><p>Comparison incomplete. No complete three-model result is claimed.</p></section>')
            continue
        result=json.loads(comparison.read_text());data['domains'][domain]=result
        scores=result['scores'];rows=[]
        for adapter in ['octo','qwen-prompt','jev']:
            s=scores[adapter];ci=s['accuracy_ci95']
            rows.append([LABELS[adapter],percent(s['accuracy']),f'{percent(ci[0])} – {percent(ci[1])}',percent(s['valid_response_rate']),f"{s['latency_median_seconds']:.3f}",f"{s['latency_p95_seconds']:.3f}"])
        content=table(['Model','Accuracy','95% family CI','Valid responses','Median (s)','p95 (s)'],rows)
        content+=f"<p>Uniform random choice: <strong>{percent(scores['octo']['uniform_accuracy'])}</strong>. Failures count as incorrect in the full denominator.</p>"
        subgroups=['by_domain','by_candidate_count']+(['by_playing_side','by_puzzle_rating_band'] if domain=='chess' else [])
        for field in subgroups:
            groups=scores['octo'][field]
            content+=f'<h3>{field.removeprefix("by_").replace("_"," ").capitalize()}</h3>'
            content+=table(['Group','Examples','Octo','Qwen Base','Jev'],[[group,entry['records']]+[percent(scores[a][field][group]['accuracy']) for a in ['octo','qwen-prompt','jev']] for group,entry in sorted(groups.items())])
        content+='<h3>Paired comparisons</h3>'+table(['Accuracy difference','Difference (pp)','95% CI (pp)','First wins','Second wins','Ties'],[[pair,f"{entry['accuracy_difference']*100:+.2f}",f"{entry['ci95'][0]*100:+.2f} to {entry['ci95'][1]*100:+.2f}",entry['a_wins'],entry['b_wins'],entry['ties']] for pair,entry in result['paired'].items()])
        content+='<h3>Probability quality</h3>'+table(['Model','Probability coverage','NLL','Multiclass Brier'],[[LABELS[a],percent(s['probability_coverage']),f"{s['nll_on_probability_rows']:.4f}" if s['nll_on_probability_rows'] is not None else 'Unavailable',f"{s['brier_on_probability_rows']:.4f}" if s['brier_on_probability_rows'] is not None else 'Unavailable'] for a,s in scores.items()])
        usage_rows=[]
        for a in ['octo','qwen-prompt','jev']:
            predictions=[json.loads(line) for line in (evaluations/f'{domain}-{a}'/'predictions.jsonl').read_text().splitlines()]
            errors={}
            for p in predictions:
                if p.get('error'):errors[p['error']]=errors.get(p['error'],0)+1
            usage=[p.get('usage') or {} for p in predictions]
            usage_rows.append([LABELS[a],json.dumps(errors,sort_keys=True) if errors else 'None',sum(u.get('input_tokens',0) for u in usage) if a=='jev' else 'N/A',sum(u.get('output_tokens',0) for u in usage) if a=='jev' else 'N/A'])
        content+='<h3>Failures and hosted usage</h3>'+table(['Model','Failures','Input tokens','Output tokens'],usage_rows)
        sections.append('<section><h2>'+title+'</h2>'+content+'</section>')
    completed=len(data['domains'])==2
    outcome='Complete · 1,000 paired test examples' if completed else 'Partial · comparisons still pending'
    findings=[]
    for domain,r in data['domains'].items():
        s=r['scores'];findings.append(f"{domain.upper()}: Octo {percent(s['octo']['accuracy'])}, prompted Qwen {percent(s['qwen-prompt']['accuracy'])}, Jev {percent(s['jev']['accuracy'])}.")
        paired=r['paired']['jev minus octo']
        if paired['ci95'][0] <= 0 <= paired['ci95'][1]:
            findings.append(f"{domain.upper()}: the paired Octo–Jev difference is not resolved by this suite's 95% bootstrap interval; the point estimate alone does not establish a winner.")
    if 'chess' in data['domains']:
        difficult=data['domains']['chess']['scores']['octo']['by_puzzle_rating_band'].get('2400+')
        if difficult:
            findings.append(f"Octo accuracy on 2400+ chess puzzles is {percent(difficult['accuracy'])} across {difficult['records']} examples. The high combined training-development accuracy does not imply equal chess performance.")
    checkpoint=ROOT/'checkpoints/step-00014877-epoch-3'
    metadata=json.loads((checkpoint/'octo.json').read_text())
    data['checkpoint']={'path':str(checkpoint),'base_model':metadata['base_model'],'base_revision':metadata['base_revision'],'step':14877,'epoch':3}
    (evaluations/'report_data.json').write_text(json.dumps(data,indent=2)+'\n')
    page='''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Octo v1 · final benchmark report</title><style>
    :root{color-scheme:light}body{margin:0;background:#f4f6f9;color:#182332;font:15px/1.55 system-ui,sans-serif}main{max-width:1120px;margin:48px auto;padding:0 24px}header{padding:30px 0}h1{font-size:38px;letter-spacing:-1px;margin:8px 0}h2{font-size:23px}h3{font-size:16px;margin-top:28px}.eyebrow{color:#286657;text-transform:uppercase;letter-spacing:2px;font-size:12px;font-weight:700}section{background:white;border:1px solid #dce3ea;border-radius:16px;padding:28px;margin:22px 0}.table-wrap{overflow:auto}table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}th,td{text-align:left;border-bottom:1px solid #e4e9ee;padding:11px 10px}th{font-size:12px;color:#536476;background:#f6f8fa}td:first-child{font-weight:600}a{color:#246e9e}code{overflow-wrap:anywhere;font-size:12px}li{margin:9px 0}.muted{color:#596b7b}@media print{body{background:white}main{margin:0;max-width:none}section{break-inside:avoid;border-radius:0;padding:12px}h1{font-size:28px}}</style><main>'''
    page+=f'<header><div class="eyebrow">Octo · post-training evaluation</div><h1>Final local benchmark report</h1><p>{outcome}</p><p class="muted">Run octo_v1_colab_1 · final/best checkpoint · epoch 3 · step 14,877</p></header>'
    figure=''
    if completed:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,axes=plt.subplots(1,2,figsize=(10,3.7),layout='constrained')
        for axis,(domain,r) in zip(axes,data['domains'].items()):
            names=['octo','qwen-prompt','jev'];values=[r['scores'][a]['accuracy']*100 for a in names]
            lows=[values[i]-r['scores'][a]['accuracy_ci95'][0]*100 for i,a in enumerate(names)]
            highs=[r['scores'][a]['accuracy_ci95'][1]*100-values[i] for i,a in enumerate(names)]
            axis.barh(['Octo','Qwen Base','Jev'],values,xerr=[lows,highs],color=['#28766b','#7394b6','#a596bf'],capsize=3)
            axis.axvline(r['scores']['octo']['uniform_accuracy']*100,color='#6b7280',linestyle='--',label='Uniform random')
            axis.set_xlim(0,105);axis.invert_yaxis();axis.set_xlabel('Accuracy (%)');axis.set_title(domain.upper());axis.spines[['top','right']].set_visible(False)
            for i,value in enumerate(values):axis.text(1,i,f'{value:.2f}%',va='center',color='white',weight='bold')
            axis.legend(loc='lower right',fontsize=8)
        fig.savefig(evaluations/'accuracy.png',dpi=180)
        fig.savefig(evaluations/'accuracy.svg')
        plt.close(fig)
        encoded=base64.b64encode((evaluations/'accuracy.png').read_bytes()).decode()
        figure='<img style="width:100%;height:auto" alt="BFSI and chess accuracy comparison with family-bootstrap confidence intervals" src="data:image/png;base64,'+encoded+'"><p class="muted">Bars show accuracy; error bars show 95% source-family bootstrap intervals.</p>'
    page+='<section><h2>Measured results</h2>'+figure+''.join('<p>'+html.escape(x)+'</p>' for x in findings)+'</section>'+''.join(sections)
    page+='<section><h2>Method and interpretation</h2><ul><li>Frozen core suites: 200 examples from each of four BFSI sources, plus 200 chess examples. Gold answers stay separate from model inputs. Identical option descriptions, order, and neutral letters A–H are used across models.</li><li>Octo uses the trained fp32 LoRA and pointer checkpoint. Qwen uses the exact same unfine-tuned Qwen3-1.7B-Base revision, greedy zero-shot plain completion and at most eight generated tokens. Malformed completions count as failures. This is a Base-model comparison, not an Instruct-model or prompt-search comparison.</li><li>95% intervals use 1,000 bootstrap resamples of source families. Paired differences compare predictions on identical record IDs; API and format errors count as incorrect.</li><li>Octo and Qwen run locally on Apple MPS. The older training process was temporarily paused. Timing excludes model loading but includes the first inference; no separate warm-up was excluded. Jev latency includes remote processing and network transport, so it is an operational comparison.</li><li>Probability scores cover only valid distributions. Qwen greedy answers provide no distribution. NLL uses a 1e-12 probability floor. Provider token usage is recorded; no dollar cost is inferred.</li><li>Public-source contamination in pretraining cannot be ruled out. BFSI tests choice among supplied intent candidates, not full-taxonomy classification. Chess tests supplied legal tactical candidates, not full-game strength or Elo. Core is a subset of full; the optional full suite was not run.</li><li>This is held-out core performance, distinct from the training development accuracy of 97.83%. No optimizer step or test-driven prompt tuning occurs during this evaluation.</li></ul></section>'
    page+='<section><h2>Reproducibility and files</h2><p>'+html.escape(json.dumps(hardware,sort_keys=True))+'</p><p>Base revision: <code>'+metadata['base_revision']+'</code>. Per-model run manifests preserve checkpoint, runner and input checksums. Raw predictions, per-source metrics and paired comparisons are stored alongside this report.</p><p><a href="report_data.json">Complete report data</a> · <a href="https://wandb.ai/octo-team/octo_v1/runs/i9oqzj5c">Training run</a> · <a href="https://docs.typesafe.ai/api">Jev API reference</a> · <a href="https://docs.typesafe.ai/primitives/choice">Choice contract</a></p><p class="muted">Generated '+html.escape(data['generated_utc'])+'</p></section></main></html>'
    (evaluations/'report.html').write_text(page)
    print(json.dumps({'report':str((evaluations/'report.html').resolve()),'complete':completed,'findings':findings},indent=2))

if __name__=='__main__':main()
