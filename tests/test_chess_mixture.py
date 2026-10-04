from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import unittest
import chess

from octo.chess_task import make_chess_request
from octo.dataset import digest
from octo.mixture import assign_partitions


class ChessRequestTests(unittest.TestCase):
    def test_uci_and_san_keep_move_identity(self):
        fen = chess.STARTING_FEN
        a = make_chess_request(fen, 'white', ['e2e4', 'g1f3'])
        b = make_chess_request(fen, 'white', ['Nf3', 'e4'])
        self.assertEqual(a['state'], b['state'])
        self.assertEqual({o['id']:o['description'] for o in a['questions'][0]['options']},
                         {o['id']:o['description'] for o in b['questions'][0]['options']})
        self.assertNotIn('label_id', a['questions'][0])

    def test_bad_side_illegal_and_duplicate_moves_rejected(self):
        for side, moves in [('black', ['e2e4','d2d4']), ('white', ['e2e5','d2d4']),
                            ('white', ['e2e4','e4'])]:
            with self.assertRaises(ValueError):
                make_chess_request(chess.STARTING_FEN, side, moves)

    def test_castling_and_promotion_candidates(self):
        castle = 'r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1'
        request = make_chess_request(castle, 'white', ['O-O','O-O-O'])
        self.assertEqual({o['id'] for o in request['questions'][0]['options']}, {'e1g1','e1c1'})
        promotion = '7k/P7/8/8/8/8/8/7K w - - 0 1'
        request = make_chess_request(promotion, 'white', ['a7a8q','a7a8n'])
        self.assertIn('promote to queen', request['questions'][0]['options'][0]['description'])


class SplitTests(unittest.TestCase):
    def test_official_test_and_diagnostic_siblings_cannot_enter_train(self):
        rows = [{'domain':'banking77','source_group_id':'family','record_id':str(i),
                 'provenance':{'source_split':split}} for i, split in enumerate(['train','test'])]
        config = {'seed':17,'diagnostic_per_domain':1,'split_fractions':{'train':.75,'dev':.1,'calibration':.05,'test_locked':.1}}
        partitions, _ = assign_partitions(rows,config)
        self.assertEqual(len(partitions['test_locked']),2)
        self.assertFalse(partitions['train'])
        rows = [{'domain':'chess','source_group_id':'game','record_id':str(i),'provenance':{}} for i in range(3)]
        partitions, _ = assign_partitions(rows,config)
        self.assertEqual(len(partitions['diagnostic']),1)
        self.assertFalse(partitions['train'])
        self.assertFalse(partitions['test_locked'])


class ExpandedBundleTests(unittest.TestCase):
    def test_bundle_provenance_legality_and_isolation(self):
        root = Path('artifacts/datasets/bfsi-chess-v2')
        if not (root / 'manifest.json').exists():
            self.skipTest('build expanded dataset first')
        manifest = json.loads((root / 'manifest.json').read_text())
        for name, expected in manifest['files'].items():
            self.assertEqual(digest(root / name), expected, name)
        seen_groups, seen_ids, game_partition, board_partition = set(), set(), {}, {}
        source_index = {}
        with (root/'sources/bfsi_source_rows.jsonl').open() as handle:
            for line in handle:
                row = json.loads(line)
                source_index[row['record_id']] = row
        chess_count, sides = 0, Counter()
        for split, stats in manifest['splits'].items():
            groups, domains, count = set(), Counter(), 0
            with (root / f'{split}.jsonl').open() as handle:
                for line in handle:
                    row = json.loads(line)
                    count += 1
                    domains[row['domain']] += 1
                    self.assertNotIn(row['record_id'],seen_ids)
                    seen_ids.add(row['record_id'])
                    groups.add(row['source_group_id'])
                    q,p = row['questions'][0],row['provenance']
                    self.assertEqual(p['partition'],split)
                    self.assertEqual(sum(o['id']==q['label_id'] for o in q['options']),1)
                    if row['domain'] != 'chess':
                        original = source_index[row['record_id']]
                        self.assertEqual(row['state'], original['text'])
                        self.assertEqual(hashlib.sha256(row['state'].encode()).hexdigest(),p['text_sha256'])
                        self.assertEqual(q['label_id'],row['domain']+':'+p['original_intent'])
                        if row['domain']=='banking77' and p['source_split']=='test':
                            self.assertEqual(split,'test_locked')
                        continue
                    chess_count += 1
                    original = chess.Board(p['original_fen'])
                    original.push_uci(p['setup_move'])
                    self.assertEqual(original.fen(),p['solver_fen'])
                    board = chess.Board(p['solver_fen'])
                    sides['white' if board.turn else 'black'] += 1
                    self.assertEqual(q['label_id'],p['solution_move'])
                    self.assertIn(p['solver_fen'],row['state'])
                    for option in q['options']:
                        move = chess.Move.from_uci(option['id'])
                        self.assertIn(move,board.legal_moves)
                        if 'mateIn1' in p['themes'] and option['id']!=q['label_id']:
                            probe = board.copy();probe.push(move)
                            self.assertFalse(probe.is_checkmate())
                    game = p['game_id'].split('/')[0].split('#')[0]
                    board_key = ' '.join(board.fen().split()[:4])
                    for key, mapping in [(game,game_partition),(board_key,board_partition)]:
                        self.assertEqual(mapping.setdefault(key,split),split)
            self.assertFalse(groups & seen_groups)
            seen_groups |= groups
            self.assertEqual(count,stats['records'])
            self.assertEqual(dict(domains),stats['domains'])
        self.assertEqual(chess_count,10000)
        self.assertGreater(sides['white'],1000)
        self.assertGreater(sides['black'],1000)

if __name__=='__main__':
    unittest.main()
