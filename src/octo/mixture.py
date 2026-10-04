"""Pinned BFSI intent datasets plus engine-supervised Lichess chess choices."""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import random
import re
import shutil

from .dataset import canonical, digest, write_json
from .chess_task import make_chess_request
from .data import normalize_record


class Groups:
    def __init__(self, n):
        self.parent = list(range(n))
    def root(self, i):
        while self.parent[i] != i:
            self.parent[i] = self.parent[self.parent[i]]
            i = self.parent[i]
        return i
    def union(self, i, j):
        a, b = self.root(i), self.root(j)
        self.parent[max(a, b)] = min(a, b)


def text_groups(rows):
    """Exact and approximate lexical families across all BFSI sources."""
    from datasketch import MinHash, MinHashLSH
    groups = Groups(len(rows))
    lsh = MinHashLSH(threshold=.85, num_perm=64)
    exact, seen_sets = {}, []
    for i, row in enumerate(rows):
        text = canonical(row['state'])
        # Values and template entities are not independent scenarios.
        text = re.sub(r'\b\d+\b', '<number>', text)
        tokens = set(text.split())
        seen_sets.append(tokens)
        if text in exact:
            groups.union(i, exact[text])
        else:
            exact[text] = i
        if len(tokens) >= 4:
            mh = MinHash(num_perm=64, seed=17)
            mh.update_batch([t.encode() for t in sorted(tokens)])
            for key in lsh.query(mh):
                j = int(key)
                if len(tokens & seen_sets[j]) / max(1, len(tokens | seen_sets[j])) >= .85:
                    groups.union(i, j)
            lsh.insert(str(i), mh)
        if i and i % 10000 == 0:
            print(f'Grouped {i} BFSI rows', flush=True)
    components = defaultdict(list)
    for i in range(len(rows)):
        components[groups.root(i)].append(i)
    for indices in components.values():
        key = hashlib.sha256('\n'.join(sorted(rows[i]['record_id'] for i in indices)).encode()).hexdigest()
        for i in indices:
            rows[i]['source_group_id'] = 'bfsi-' + key
    return components


def load_bfsi(config):
    import pyarrow.parquet as pq
    rows, descriptions = [], {}
    for domain in (d for d in ('banking77', 'bitext-banking', 'bitext-insurance', 'bitext-wealth') if d in config['sources']):
        source = config['sources'][domain]
        input_files = [(k, v) for k, v in source['files'].items() if k.endswith(('.parquet', '.csv'))]
        for filename, spec in input_files:
            path = Path(spec['path'])
            table = pq.read_table(path) if path.suffix == '.parquet' else None
            originals = table.to_pylist() if table is not None else list(csv.DictReader(path.open(newline='', encoding='utf-8-sig')))
            names = json.loads(table.schema.metadata[b'huggingface'])['info']['features']['label']['names'] if domain == 'banking77' else None
            for index, raw in enumerate(originals):
                intent = names[raw['label']] if names else raw['intent']
                state = raw['text'] if names else raw['instruction']
                stable_id = domain + ':' + intent
                descriptions[stable_id] = 'Customer request concerning ' + intent.replace('_', ' ').rstrip('?') + '.'
                rows.append({'record_id': f'{domain}-{filename}-{index}', 'domain': domain, 'state': state,
                    'label': stable_id, 'provenance': {'source': domain, 'repo': source['repo'],
                        'revision': source['revision'], 'file': filename, 'row_index': index,
                        'original_intent': intent, 'original_label': raw['label'] if names else intent,
                        'source_split': filename.split('.')[0] if names else 'train',
                        'text_sha256': hashlib.sha256(state.encode()).hexdigest(), 'transformation_version': 2}})
        print(f'Loaded {domain}: {sum(r["domain"] == domain for r in rows)} rows', flush=True)
    return rows, descriptions


def chess_rows(config, output, exclusions):
    import chess
    import pyarrow.parquet as pq
    source = config['sources']['lichess']
    spec = next(v for k, v in source['files'].items() if k.endswith('.parquet'))
    collected = []
    sampled = output / 'sources' / 'lichess' / 'selected_source_rows.jsonl'
    sampled.parent.mkdir(parents=True, exist_ok=True)
    scanned = 0
    with sampled.open('w') as snapshot:
        for batch in pq.ParquetFile(spec['path']).iter_batches(batch_size=4096):
            for raw in batch.to_pylist():
                index = scanned
                scanned += 1
                if scanned > config['chess_scan_limit']:
                    break
                if raw['Popularity'] < 80 or raw['NbPlays'] < 1000 or raw['RatingDeviation'] > 100:
                    exclusions.append({'source': 'lichess', 'row_index': index, 'reason': 'puzzle quality filter'})
                    continue
                try:
                    board = chess.Board(raw['FEN'])
                    moves = raw['Moves'].split()
                    if not board.is_valid() or len(moves) < 2:
                        raise ValueError('invalid puzzle board or missing continuation')
                    # Lichess FEN is BEFORE the opponent setup move.
                    continuation = board.copy()
                    for uci in moves:
                        move = chess.Move.from_uci(uci)
                        if move not in continuation.legal_moves:
                            raise ValueError('illegal puzzle continuation')
                        continuation.push(move)
                    board.push_uci(moves[0])
                    solution = chess.Move.from_uci(moves[1])
                    legal = sorted(board.legal_moves, key=lambda m: m.uci())
                    distractors = []
                    for move in legal:
                        if move == solution:
                            continue
                        if 'mateIn1' in raw['Themes']:
                            probe = board.copy()
                            probe.push(move)
                            if probe.is_checkmate():
                                continue  # do not label another valid mate as wrong
                        distractors.append(move.uci())
                    if not distractors:
                        raise ValueError('no unambiguous distractor available')
                    seed = int(hashlib.sha256((str(config['seed']) + raw['PuzzleId']).encode()).hexdigest()[:16], 16)
                    rng = random.Random(seed)
                    count = min(rng.choice(config['candidate_counts']), len(distractors) + 1)
                    options = [solution.uci()] + rng.sample(distractors, count - 1)
                    rng.shuffle(options)
                    side = 'white' if board.turn else 'black'
                    record = make_chess_request(board.fen(), side, options, record_id='lichess-' + raw['PuzzleId'])
                    record['questions'][0]['label_id'] = solution.uci()
                    record.update({'domain': 'chess', 'source_group_id': 'game-' + raw['GameId'].split('/')[0].split('#')[0],
                        'board_group': 'board-' + hashlib.sha256(' '.join(board.fen().split()[:4]).encode()).hexdigest(),
                        'provenance': {'source': 'lichess', 'repo': source['repo'], 'revision': source['revision'],
                            'file': Path(spec['path']).name, 'row_index': index, 'puzzle_id': raw['PuzzleId'],
                            'game_id': raw['GameId'], 'original_fen': raw['FEN'], 'setup_move': moves[0],
                            'solver_fen': board.fen(), 'solution_move': solution.uci(), 'rating': raw['Rating'],
                            'themes': raw['Themes'], 'transformation_version': 2, 'label_source': 'Lichess Stockfish-derived puzzle solution'}})
                    collected.append(record)
                    snapshot.write(json.dumps({'row_index': index, **raw}, sort_keys=True) + '\n')
                except ValueError as exc:
                    exclusions.append({'source': 'lichess', 'row_index': index, 'reason': str(exc)})
                if len(collected) == config['chess_positions']:
                    break
            if len(collected) == config['chess_positions'] or scanned > config['chess_scan_limit']:
                break
    if len(collected) < config['chess_positions']:
        raise ValueError(f'only {len(collected)} quality chess positions within scan limit')
    # Merge all positions from a game and equivalent boards across games.
    groups, seen = Groups(len(collected)), {}
    for i, row in enumerate(collected):
        for key in (row['source_group_id'], row['board_group']):
            if key in seen:
                groups.union(i, seen[key])
            seen[key] = i
    components = defaultdict(list)
    for i in range(len(collected)):
        components[groups.root(i)].append(i)
    for indices in components.values():
        identity = hashlib.sha256('\n'.join(sorted(collected[i]['record_id'] for i in indices)).encode()).hexdigest()
        for i in indices:
            collected[i]['source_group_id'] = 'chess-' + identity
            del collected[i]['board_group']
    print(f'Prepared {len(collected)} legal chess positions from {scanned} source rows', flush=True)
    return collected


def assign_partitions(rows, config):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row['source_group_id']].append(row)
    diagnostic = set()
    for domain in sorted({r['domain'] for r in rows}):
        eligible = [g for g, rs in grouped.items() if any(r['domain'] == domain for r in rs)
            and not any(r['domain'] == 'banking77' and r['provenance']['source_split'] == 'test' for r in rs)]
        for group in sorted(eligible, key=lambda g: hashlib.sha256((g + str(config['seed'])).encode()).hexdigest())[:config['diagnostic_per_domain']]:
            diagnostic.add(group)
    partitions = defaultdict(list)
    for group, records in grouped.items():
        if group in diagnostic:
            split = 'diagnostic'
            # Keep one representative per domain; all siblings are quarantined from quality splits.
            records = [next(r for r in records if r['domain'] == domain) for domain in sorted({r['domain'] for r in records})]
        elif any(r['domain'] == 'banking77' and r['provenance']['source_split'] == 'test' for r in records):
            split = 'test_locked'
        else:
            value = int(hashlib.sha256((group + str(config['seed'])).encode()).hexdigest()[:16], 16) / 2**64
            cumulative = 0.
            for split, fraction in config['split_fractions'].items():
                cumulative += fraction
                if value < cumulative:
                    break
        for r in records:
            r['provenance']['partition'] = split
            partitions[split].append(r)
    return partitions, grouped


def fetch(config):
    import requests
    from .model import BASE_MODEL, BASE_REVISION
    tokenizer_dir = Path('artifacts/tokenizers/qwen3')
    tokenizer_dir.mkdir(parents=True, exist_ok=True)
    for name, expected in config['tokenizer_checksums'].items():
        target = tokenizer_dir / name
        if target.exists() and digest(target) == expected:
            continue
        response = requests.get(f'https://huggingface.co/{BASE_MODEL}/resolve/{BASE_REVISION}/{name}', timeout=120)
        response.raise_for_status()
        if hashlib.sha256(response.content).hexdigest() != expected:
            raise ValueError('tokenizer checksum mismatch: ' + name)
        target.write_bytes(response.content)
    for source in config['sources'].values():
        for file in source['files'].values():
            path = Path(file['path'])
            if path.exists() and digest(path) == file['sha256']:
                continue
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(path.suffix + '.partial')
            with requests.get(file['url'], stream=True, timeout=120) as response:
                response.raise_for_status()
                with temporary.open('wb') as target:
                    for chunk in response.iter_content(1024 * 1024):
                        target.write(chunk)
            if digest(temporary) != file['sha256']:
                raise ValueError('source checksum mismatch: ' + str(path))
            temporary.replace(path)


def build(config_path, output_dir):
    from transformers import AutoTokenizer
    from .encoding import Encoder, StructuralTokens
    from .model import BASE_MODEL, BASE_REVISION
    config = json.loads(Path(config_path).read_text())
    output = Path(output_dir)
    if output.exists():
        raise ValueError('dataset destination already exists; choose a new version directory')
    for source in config['sources'].values():
        for spec in source['files'].values():
            if digest(spec['path']) != spec['sha256']:
                raise ValueError('source checksum mismatch: ' + spec['path'])
    for name, expected in config['tokenizer_checksums'].items():
        if digest(Path('artifacts/tokenizers/qwen3') / name) != expected:
            raise ValueError('tokenizer checksum mismatch')
    tokenizer = AutoTokenizer.from_pretrained('artifacts/tokenizers/qwen3', local_files_only=True)
    encoder = Encoder(tokenizer, StructuralTokens.from_tokenizer(tokenizer))
    output.mkdir(parents=True)
    rows, descriptions = load_bfsi(config)
    components = text_groups(rows)
    exclusions, accepted, source_members = [], [], []
    for indices in components.values():
        group = [rows[i] for i in indices]
        labels_by_domain = defaultdict(set)
        for row in group:
            labels_by_domain[row['domain']].add(row['label'])
        source_members.append({'source_group_id': group[0]['source_group_id'],
            'members': [r['record_id'] for r in group]})
        if any(len(labels) > 1 for labels in labels_by_domain.values()):
            exclusions.extend({'record_id': r['record_id'], 'reason': 'conflicting intent labels in lexical family'} for r in group)
            continue
        accepted.extend(group)
    taxonomies = {domain: sorted(k for k in descriptions if k.startswith(domain + ':')) for domain in {r['domain'] for r in accepted}}
    for row in accepted:
        rng = random.Random(int(hashlib.sha256((row['record_id'] + str(config['seed'])).encode()).hexdigest()[:16], 16))
        pool = [k for k in taxonomies[row['domain']] if k != row['label']]
        count = rng.choice(config['candidate_counts'])
        chosen = [row['label']] + rng.sample(pool, count - 1)
        rng.shuffle(chosen)
        row['questions'] = [{'id': 'intent', 'type': 'choice',
            'instruction': 'Select the support request category explicitly described by this customer.',
            'options': [{'id': k, 'description': descriptions[k]} for k in chosen], 'label_id': row.pop('label')}]
    chess = chess_rows(config, output, exclusions) if 'lichess' in config['sources'] else []
    all_rows = accepted + chess
    valid, token_lengths = [], {}
    for row in all_rows:
        try:
            normalize_record(row)
            encoded = encoder.encode(row)
            token_lengths[row['record_id']] = len(encoded.input_ids)
            valid.append(row)
        except ValueError as exc:
            exclusions.append({'record_id': row['record_id'], 'reason': str(exc)})
    partitions, all_groups = assign_partitions(valid, config)
    # For every accepted original text, keep a compact source snapshot. Full input files remain local.
    if accepted:
        source_snapshot = output / 'sources' / 'bfsi_source_rows.jsonl'
        source_snapshot.parent.mkdir(parents=True, exist_ok=True)
        with source_snapshot.open('w') as handle:
            for row in accepted:
                handle.write(json.dumps({'record_id': row['record_id'], 'text': row['state'], 'provenance': row['provenance']}, sort_keys=True) + '\n')
    summary = {}
    for split in ('diagnostic', 'train', 'dev', 'calibration', 'test_locked'):
        records = sorted(partitions[split], key=lambda r: r['record_id'])
        with (output / f'{split}.jsonl').open('w') as handle:
            for row in records:
                handle.write(json.dumps(row, sort_keys=True) + '\n')
        summary[split] = {'records': len(records), 'source_groups': len({r['source_group_id'] for r in records}),
            'domains': dict(Counter(r['domain'] for r in records)),
            'candidate_counts': dict(Counter(len(r['questions'][0]['options']) for r in records)),
            'labels': dict(Counter(r['questions'][0]['label_id'] for r in records if r['domain'] != 'chess')),
            'max_physical_tokens': max(token_lengths[r['record_id']] for r in records)}
    write_json(output / 'source_groups.json', source_members + [{'source_group_id': g, 'members': [r['record_id'] for r in rs]} for g, rs in all_groups.items() if rs[0]['domain'] == 'chess'])
    write_json(output / 'exclusions.json', exclusions)
    write_json(output / 'config.json', config)
    write_json(output / 'candidate_descriptions.json', descriptions)
    for domain, source in config['sources'].items():
        directory = output / 'sources' / domain
        directory.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source['files']['README.md']['path'], directory / 'SOURCE_CARD.md')
        (directory / 'LICENSE.txt').write_text(f"Source: {source['repo']}\nRevision: {source['revision']}\nLicense: {source['license']}\nOriginal source card accompanies this file. Original dataset licensing is retained per source; the mixture does not relicense these records.\n" + ('https://cdla.dev/sharing-1-0/\n' if source['license']=='cdla-sharing-1.0' else 'https://creativecommons.org/publicdomain/zero/1.0/\n' if source['license']=='cc0-1.0' else 'https://creativecommons.org/licenses/by/4.0/\n'))
    shutil.copyfile('docs/octo_bfsi_chess_dataset.md', output / 'DATASET_CARD.md')
    manifest = {'dataset_name': config['dataset_name'], 'transformation_version': 2, 'seed': config['seed'],
        'builder_sha256': digest(__file__), 'chess_request_builder_sha256': digest(Path(__file__).with_name('chess_task.py')),
        'sources': config['sources'], 'splits': summary, 'exclusion_counts': dict(Counter(e['reason'] for e in exclusions)),
        'tokenizer': {'repo': BASE_MODEL, 'revision': BASE_REVISION},
        'files': {str(p.relative_to(output)): digest(p) for p in sorted(output.rglob('*')) if p.is_file()},
        'limitations': ['BFSI synthetic lexical families approximate template IDs; semantic leakage remains possible.',
            'Choice candidate sets always contain the source correct answer; not full-taxonomy classification.',
            'Chess targets are tactical puzzle solutions, not labels for arbitrary candidate subsets without the solution.',
            'Chess sample comes from a bounded scan of one pinned shard; not a representative general-game corpus.',
            'Dataset preparation is not evidence of trained model strength.']}
    write_json(output / 'manifest.json', manifest)
    print(json.dumps({'dataset': str(output), 'splits': {k: {f: v for f, v in s.items() if f != 'labels'} for k, s in summary.items()}, 'exclusions': manifest['exclusion_counts']}, indent=2), flush=True)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--domain', choices=['bfsi','chess'], required=True)
    parser.add_argument('--config')
    parser.add_argument('--output')
    parser.add_argument('--fetch', action='store_true')
    args = parser.parse_args()
    config_path = args.config or f'configs/{args.domain}_v2.json'
    output = args.output or f'artifacts/datasets/{args.domain}-v2'
    if args.fetch:
        fetch(json.loads(Path(config_path).read_text()))
    build(config_path, output)

if __name__ == '__main__':
    main()
