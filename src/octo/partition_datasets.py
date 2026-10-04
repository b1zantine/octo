"""Separate immutable BFSI/chess corpora without changing existing source families."""
from collections import Counter
import json
from pathlib import Path
import shutil
from .dataset import digest, write_json


def separate(parent='artifacts/datasets/bfsi-chess-v2'):
    parent = Path(parent)
    manifest = json.loads((parent / 'manifest.json').read_text())
    for name in ('bfsi', 'chess'):
        config_path = Path(f'configs/{name}_v2.json')
        config = json.loads(config_path.read_text())
        output = Path(f'artifacts/datasets/{name}-v2')
        if output.exists():
            raise ValueError('dataset already exists: ' + str(output))
        output.mkdir(parents=True)
        summary = {}
        selected = set()
        for split in manifest['splits']:
            counts, domains, candidates = Counter(), Counter(), Counter()
            with (parent / f'{split}.jsonl').open() as source, (output / f'{split}.jsonl').open('w') as target:
                for line in source:
                    row = json.loads(line)
                    if (row['domain']=='chess') != (name=='chess'):
                        continue
                    target.write(line)
                    selected.add(row['record_id'])
                    domains[row['domain']] += 1
                    candidates[len(row['questions'][0]['options'])] += 1
                    counts[row['questions'][0]['label_id']] += 1
            summary[split] = {'records':sum(domains.values()),'domains':dict(domains),'candidate_counts':dict(candidates)}
        groups = json.loads((parent / 'source_groups.json').read_text())
        write_json(output/'source_groups.json',[g for g in groups if set(g['members']) & selected])
        write_json(output/'config.json', config)
        sources = output/'sources';sources.mkdir()
        for domain in config['sources']:
            shutil.copytree(parent/'sources'/domain,sources/domain)
        if name=='bfsi':
            shutil.copyfile(parent/'sources/bfsi_source_rows.jsonl',sources/'bfsi_source_rows.jsonl')
            descriptions=json.loads((parent/'candidate_descriptions.json').read_text())
            write_json(output/'candidate_descriptions.json',descriptions)
        (output/'DATASET_CARD.md').write_text(f'# {name.upper()} corpus v2\n\nSeparated from the immutable BFSI/chess source artifact, preserving all records, labels, candidate orders, source groups, and partitions.\nSources and original licenses are in sources/. Only this domain family is included in this corpus and configuration.\nSee docs/octo_bfsi_chess_dataset.md for source/labeling/splitting details.\n')
        result={'dataset_name':config['dataset_name'],'sources':config['sources'],'splits':summary,
            'parent_artifact':'octo-team/octo_v1/bfsi-chess-v2:v0','parent_manifest_sha256':digest(parent/'manifest.json'),
            'parent_train_sha256':manifest['files']['train.jsonl'],'parent_dev_sha256':manifest['files']['dev.jsonl'],
            'transformation':'domain partition only; no changes to records or split assignments',
            'builder_sha256':digest(__file__),'files':{str(p.relative_to(output)):digest(p) for p in sorted(output.rglob('*')) if p.is_file()}}
        write_json(output/'manifest.json',result)
        print(name,summary,flush=True)

if __name__=='__main__':
    separate()
