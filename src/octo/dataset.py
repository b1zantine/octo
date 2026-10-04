"""Reproducible, group-separated BANKING77 Choice pilot. No remote dataset code."""
import argparse
from collections import Counter, defaultdict
from difflib import SequenceMatcher
import hashlib
import json
from pathlib import Path
import random
import re
import unicodedata

from .data import normalize_record


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical(text):
    return ' '.join(re.findall(r'\w+', unicodedata.normalize('NFKC', text).casefold()))


def grounded(text, intent):
    words = set(canonical(text).split())
    if intent == 'change_pin':
        return 'pin' in words and bool(words & {'change', 'changing', 'reset', 'new', 'update'})
    if intent == 'lost_or_stolen_card':
        return 'card' in words and bool(words & {'lost', 'stolen', 'missing'})
    if intent == 'activate_my_card':
        return 'card' in words and bool(words & {'activate', 'activating', 'activation'})
    return False


def group_rows(rows):
    """Conservative lexical near-duplicate components across both source splits."""
    parent = list(range(len(rows)))
    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    texts = [canonical(r['text']) for r in rows]
    words = [set(t.split()) for t in texts]
    for i in range(len(rows)):
        for j in range(i):
            overlap = len(words[i] & words[j]) / max(1, len(words[i] | words[j]))
            if texts[i] == texts[j] or overlap >= .80 or SequenceMatcher(None, texts[i], texts[j]).ratio() >= .92:
                parent[root(i)] = root(j)
    groups = defaultdict(list)
    for i, row in enumerate(rows):
        groups[root(i)].append(row)
    return list(groups.values())


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')


def build(config_path, source_dir, output_dir):
    import pyarrow.parquet as pq
    from transformers import AutoTokenizer
    from .encoding import Encoder, StructuralTokens
    from .model import BASE_MODEL, BASE_REVISION
    config = json.loads(Path(config_path).read_text())
    source_dir, output = Path(source_dir), Path(output_dir)
    if output.exists():
        raise ValueError('destination already exists; choose a new version directory')
    source = json.loads((source_dir / 'source.json').read_text())
    if any(source[k] != config['source'][k] for k in config['source']):
        raise ValueError('source identity does not match configuration')
    for name, expected in config['source_checksums'].items():
        if digest(source_dir / name) != expected:
            raise ValueError(f'source checksum mismatch: {name}')
    for name, expected in config['tokenizer_checksums'].items():
        if digest(Path('artifacts/tokenizers/qwen3') / name) != expected:
            raise ValueError(f'tokenizer checksum mismatch: {name}')
    tokenizer = AutoTokenizer.from_pretrained(Path('artifacts/tokenizers/qwen3'), local_files_only=True)
    encoder = Encoder(tokenizer, StructuralTokens.from_tokenizer(tokenizer))
    rows, excluded = [], []
    for split in ('train', 'test'):
        table = pq.read_table(source_dir / f'{split}.parquet')
        names = json.loads(table.schema.metadata[b'huggingface'])['info']['features']['label']['names']
        for index, row in enumerate(table.to_pylist()):
            intent = names[row['label']]
            ref = {'split': split, 'row_index': index, 'intent': intent}
            if intent not in config['intents']:
                continue
            if not grounded(row['text'], intent):
                excluded.append({**ref, 'reason': 'explicit task evidence filter'})
                continue
            rows.append({**row, **ref})
    rng = random.Random(config['seed'])
    pools = defaultdict(list)
    for group in group_rows(rows):
        identity = hashlib.sha256('\n'.join(sorted(canonical(r['text']) for r in group)).encode()).hexdigest()
        if len({r['intent'] for r in group}) != 1:
            excluded.extend({**r, 'reason': 'conflicting labels in lexical source group'} for r in group)
            continue
        upstream = 'test' if any(r['split'] == 'test' for r in group) else 'train'
        representative = sorted((r for r in group if r['split'] == upstream), key=lambda r: (len(r['text']), r['row_index']))[0]
        pools[(representative['intent'], upstream)].append((identity, representative, group))
    partitions, assignments, max_tokens = {}, [], 0
    for split, count in config['per_intent'].items():
        records = []
        for intent in sorted(config['intents']):
            pool = pools[(intent, 'test' if split == 'test_locked' else 'train')]
            rng.shuffle(pool)
            accepted = 0
            while pool and accepted < count:
                group_id, row, group = pool.pop()
                ids = sorted(config['intents'])
                if accepted % 2 == 0:
                    ids.remove(rng.choice([i for i in ids if i != intent]))
                rng.shuffle(ids)
                record = {
                    'record_id': f"banking77-{row['split']}-{row['row_index']}",
                    'source_group_id': 'banking77-' + group_id,
                    'state': row['text'],
                    'questions': [{'id': 'intent', 'type': 'choice',
                        'instruction': 'Which support issue is explicitly described by this customer?',
                        'options': [{'id': i, 'description': config['intents'][i]} for i in ids],
                        'label_id': intent}],
                    'provenance': {**source, 'source_split': row['split'], 'row_index': row['row_index'],
                        'original_label': row['label'], 'original_intent': intent,
                        'text_sha256': hashlib.sha256(row['text'].encode()).hexdigest(),
                        'transformation_version': 1, 'partition': split}}
                try:
                    normalize_record(record)
                    encoded = encoder.encode(record)
                except ValueError as e:
                    excluded.append({**row, 'reason': str(e)})
                    continue
                max_tokens = max(max_tokens, len(encoded.input_ids))
                records.append(record)
                assignments.append({'source_group_id': record['source_group_id'], 'partition': split,
                    'members': [{'split': r['split'], 'row_index': r['row_index']} for r in group]})
                accepted += 1
            if accepted != count:
                raise ValueError(f'insufficient independent groups for {split}/{intent}: {accepted}/{count}')
        rng.shuffle(records)
        partitions[split] = records
    output.mkdir(parents=True)
    import shutil
    shutil.copytree(source_dir, output / 'source')
    for split, records in partitions.items():
        (output / f'{split}.jsonl').write_text(''.join(json.dumps(r, sort_keys=True) + '\n' for r in records))
    write_json(output / 'source_groups.json', assignments)
    write_json(output / 'exclusions.json', excluded)
    write_json(output / 'config.json', config)
    (output / 'DATASET_CARD.md').write_bytes(Path('docs/octo_dataset.md').read_bytes())
    (output / 'SOURCE_CARD.md').write_bytes((source_dir / 'README.md').read_bytes())
    (output / 'LICENSE.txt').write_text('BANKING77 by PolyAI / Casanueva et al. (2020).\nCreative Commons Attribution 4.0 International.\nhttps://creativecommons.org/licenses/by/4.0/\nSource and attribution: https://huggingface.co/datasets/PolyAI/banking77\nPaper: https://arxiv.org/abs/2003.04807\nChanges: filtered intents, evidence filtering, lexical grouping, subsampling, and Octo Choice formatting. Original utterance text and intent labels preserved.\n')
    summary = {s: {'records': len(rs), 'labels': dict(Counter(r['questions'][0]['label_id'] for r in rs)),
        'candidate_counts': dict(Counter(len(r['questions'][0]['options']) for r in rs))} for s, rs in partitions.items()}
    manifest = {'dataset_name': config['dataset_name'], 'source': source, 'transformation_version': 1,
        'seed': config['seed'], 'splits': summary, 'builder_sha256': digest(__file__), 'tokenizer': {'repo': BASE_MODEL, 'revision': BASE_REVISION},
        'max_physical_tokens': max_tokens, 'source_files': {p.name: digest(p) for p in sorted(source_dir.iterdir())},
        'files': {str(p.relative_to(output)): digest(p) for p in sorted(output.rglob('*')) if p.is_file()},
        'limitations': ['Choice only; source supplies no severity or truth supervision.',
            'Lexical groups approximate scenario families; semantic paraphrase leakage remains possible.',
            'Locked test is drawn only from upstream test; other partitions use upstream train.',
            'Legacy HF Parquet mirror is deprecated; canonical provenance is PolyAI/banking77.',
            'A small three-intent teaching pilot, not a full 77-intent benchmark.']}
    write_json(output / 'manifest.json', manifest)
    print(json.dumps({'dataset': str(output), 'splits': summary, 'max_physical_tokens': max_tokens}, indent=2))
    return manifest



def fetch(config_path, source_dir):
    import requests
    from .model import BASE_MODEL, BASE_REVISION
    config = json.loads(Path(config_path).read_text())
    source = config['source']
    files = {
        'train.parquet': f"https://huggingface.co/datasets/{source['repo']}/resolve/{source['revision']}/default/train/0000.parquet",
        'test.parquet': f"https://huggingface.co/datasets/{source['repo']}/resolve/{source['revision']}/default/test/0000.parquet",
        'README.md': f"https://huggingface.co/datasets/{source['canonical_repo']}/resolve/{source['card_revision']}/README.md"}
    destination = Path(source_dir)
    destination.mkdir(parents=True, exist_ok=True)
    for name, url in files.items():
        response = requests.get(url, timeout=60)
        response.raise_for_status()
        payload = response.content
        if hashlib.sha256(payload).hexdigest() != config['source_checksums'][name]:
            raise ValueError(f'source checksum mismatch: {name}')
        (destination / name).write_bytes(payload)
    (destination / 'source.json').write_text(json.dumps(source, indent=2) + '\n')
    tokenizer_dir = Path('artifacts/tokenizers/qwen3')
    tokenizer_dir.mkdir(parents=True, exist_ok=True)
    for name, expected in config['tokenizer_checksums'].items():
        response = requests.get(f'https://huggingface.co/{BASE_MODEL}/resolve/{BASE_REVISION}/{name}', timeout=60)
        response.raise_for_status()
        if hashlib.sha256(response.content).hexdigest() != expected:
            raise ValueError(f'tokenizer checksum mismatch: {name}')
        (tokenizer_dir / name).write_bytes(response.content)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', default='configs/banking77_pilot.json')
    p.add_argument('--source', default='artifacts/sources/banking77')
    p.add_argument('--fetch', action='store_true', help='Download pinned source and tokenizer before building')
    p.add_argument('--output', default='artifacts/datasets/banking77-choice-v1')
    args = p.parse_args()
    if args.fetch:
        fetch(args.config, args.source)
    build(args.config, args.source, args.output)

if __name__ == '__main__':
    main()
