"""Frozen, label-free benchmarking inputs and separate evaluator answer keys."""
from collections import Counter, defaultdict, deque
import hashlib
import json
from pathlib import Path
import shutil
from .dataset import digest, write_json
from .data import normalize_record


def records(path):
    with Path(path).open() as source:
        for line in source:
            if line.strip():
                yield json.loads(line)


def neutralize(row):
    q = row['questions'][0]
    if len(row['questions']) != 1 or q['type'] != 'choice':
        raise ValueError('benchmark v1 supports one Choice question per record')
    options, labels = [], {}
    for i, option in enumerate(q['options']):
        identity = chr(ord('A') + i)
        labels[option['id']] = identity
        options.append({'id':identity,'description':option['description']})
    example_id = 'eval-' + hashlib.sha256(row['record_id'].encode()).hexdigest()[:24]
    request = {'record_id':example_id,'state':row['state'],'questions':[{
        'id':'decision','type':'choice','instruction':q['instruction'],'options':options}]}
    normalize_record(request)
    gold = {'record_id':example_id,'label_id':labels[q['label_id']],
        'source_group_id':row['source_group_id'],'domain':row['domain'],
        'candidate_count':len(options),'original_record_id':row['record_id'],
        'provenance':row['provenance']}
    return request, gold


def sample(rows, config, count):
    # Round-robin label/count strata, at most one member per source family.
    chosen = []
    for domain in config['domains']:
        bins = defaultdict(list)
        for row in rows:
            if row['domain'] != domain:
                continue
            q = row['questions'][0]
            key = (q['label_id'],len(q['options'])) if domain!='chess' else (
                row['provenance']['solver_fen'].split()[1],
                min(3,row['provenance']['rating']//800),len(q['options']))
            bins[key].append(row)
        queues = {k:deque(sorted(rs,key=lambda r:hashlib.sha256((str(config['seed'])+r['record_id']).encode()).hexdigest())) for k,rs in bins.items()}
        used, selected = set(), []
        if domain != 'chess':
            # Visit every intent before taking a second example of any intent.
            # Rotate candidate-count bands independently so short core suites
            # do not exclude alphabetically later BANKING77 labels.
            by_label = defaultdict(list)
            for key in sorted(queues):
                by_label[key[0]].append(key)
            cycles = {}
            for label, keys in by_label.items():
                offset = int(hashlib.sha256((label+str(config['seed'])).encode()).hexdigest()[:8],16)%len(keys)
                cycles[label] = deque(keys[offset:]+keys[:offset])
            while cycles and len(selected)<count:
                for label in sorted(list(cycles)):
                    cycle=cycles[label]
                    picked=None
                    while cycle:
                        key=cycle.popleft();queue=queues[key]
                        while queue and queue[0]['source_group_id'] in used:
                            queue.popleft()
                        if not queue:
                            continue
                        picked=queue.popleft();cycle.append(key);break
                    if picked is None:
                        del cycles[label];continue
                    used.add(picked['source_group_id']);selected.append(picked)
                    if len(selected)==count:break
        else:
            while queues and len(selected)<count:
                for key in sorted(list(queues)):
                    queue = queues[key]
                    while queue and queue[0]['source_group_id'] in used:
                        queue.popleft()
                    if not queue:
                        del queues[key]
                        continue
                    row=queue.popleft();used.add(row['source_group_id']);selected.append(row)
                    if len(selected)==count:
                        break
        if len(selected)<count:
            raise ValueError(f'insufficient held-out families: {domain} {len(selected)}/{count}')
        chosen.extend(selected)
    return sorted(chosen,key=lambda r:r['record_id'])


def build(config_path, output_dir):
    config=json.loads(Path(config_path).read_text())
    dataset=Path(config['dataset_dir']);output=Path(output_dir)
    if output.exists():
        raise ValueError('benchmark already exists; create a new version')
    parent=json.loads((dataset/'manifest.json').read_text())
    for split in ['train','dev','calibration','diagnostic','test_locked']:
        if digest(dataset/f'{split}.jsonl')!=parent['files'][f'{split}.jsonl']:
            raise ValueError('dataset split checksum mismatch: '+split)
    non_test_groups={r['source_group_id'] for split in ['train','dev','calibration','diagnostic'] for r in records(dataset/f'{split}.jsonl')}
    locked=list(records(dataset/'test_locked.jsonl'))
    if {r['source_group_id'] for r in locked} & non_test_groups:
        raise ValueError('benchmark source families overlap training/tuning data')
    suites={'dev':sample(list(records(dataset/'dev.jsonl')),config,config['dev_per_domain']),
            'core':sample(locked,config,config['core_per_domain']), 'full':locked}
    output.mkdir(parents=True)
    counts={}
    for suite, rows in suites.items():
        inputs=output/f'{suite}.inputs.jsonl';answers=output/f'{suite}.gold.jsonl'
        domains,candidates=Counter(),Counter()
        with inputs.open('w') as source,answers.open('w') as scorer:
            for row in rows:
                request,gold=neutralize(row)
                source.write(json.dumps(request,sort_keys=True)+'\n')
                scorer.write(json.dumps(gold,sort_keys=True)+'\n')
                domains[row['domain']]+=1;candidates[len(request['questions'][0]['options'])]+=1
        counts[suite]={'records':len(rows),'domains':dict(domains),'candidate_counts':dict(candidates),
            'uniform_accuracy':sum(v/int(k) for k,v in candidates.items())/len(rows)}
    write_json(output/'config.json',config)
    shutil.copyfile('docs/octo_benchmarks.md',output/'BENCHMARK_CARD.md')
    shutil.copytree(dataset/'sources',output/'source_notices',ignore=shutil.ignore_patterns('*.jsonl'))
    manifest={'benchmark_name':config['benchmark_name'],'dataset_name':parent['dataset_name'],
        'dataset_manifest_sha256':digest(dataset/'manifest.json'),'historical_parent_manifest_sha256':parent.get('parent_manifest_sha256'),
        'training_file_sha256':parent['files']['train.jsonl'],'historical_training_file_sha256':parent.get('parent_train_sha256'),
        'source_test_sha256':parent['files']['test_locked.jsonl'],'seed':config['seed'],'suites':counts,
        'sources':parent['sources'],'adapters':['qwen-prompt','jev','octo'],'prompt_version':1,
        'builder_sha256':digest(__file__),'files':{str(p.relative_to(output)):digest(p) for p in sorted(output.rglob('*')) if p.is_file()},
        'policy':{'primary_metric':'accuracy including errors as wrong','frozen_inputs':True,'gold_separate':True,
            'core_is_subset_of_full':True,'tune_only_on':'dev','test_runs_after_training_only':True}}
    write_json(output/'manifest.json',manifest)
    print(json.dumps({'benchmark':str(output),'suites':counts},indent=2))
    return manifest


def main():
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--domain',choices=['bfsi','chess'],required=True)
    parser.add_argument('--output')
    args=parser.parse_args()
    build(f'configs/{args.domain}_benchmark_v1.json',args.output or f'artifacts/benchmarks/{args.domain}-v1')

if __name__=='__main__':
    main()
