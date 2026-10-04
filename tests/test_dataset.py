import itertools
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from octo.dataset import build, canonical, digest, group_rows
from octo.tracking import log_artifact


class DatasetTests(unittest.TestCase):
    def test_near_duplicates_cross_upstream_splits_share_component(self):
        rows = [{'text': 'I lost my card!', 'split': 'train'},
                {'text': 'i lost my card', 'split': 'test'},
                {'text': 'Change my PIN please', 'split': 'train'}]
        groups = group_rows(rows)
        self.assertEqual(sorted(map(len, groups)), [1, 2])
        self.assertEqual(canonical(rows[0]['text']), canonical(rows[1]['text']))

    def test_generated_bundle_integrity_and_partition_isolation(self):
        root = Path('artifacts/datasets/banking77-choice-v1')
        if not root.exists():
            self.skipTest('build the public dataset first')
        manifest = json.loads((root / 'manifest.json').read_text())
        for name, expected in manifest['files'].items():
            self.assertEqual(digest(root / name), expected, name)
        import pyarrow.parquet as pq
        source_rows = {s: pq.read_table(root / 'source' / f'{s}.parquet').to_pylist() for s in ('train', 'test')}
        all_groups, all_ids = set(), set()
        for split, stats in manifest['splits'].items():
            rows = [json.loads(line) for line in (root / f'{split}.jsonl').read_text().splitlines()]
            groups = {r['source_group_id'] for r in rows}
            ids = {r['record_id'] for r in rows}
            self.assertFalse(all_groups & groups)
            self.assertFalse(all_ids & ids)
            all_groups |= groups
            all_ids |= ids
            self.assertEqual(len(rows), stats['records'])
            for r in rows:
                self.assertEqual(r['provenance']['source_split'], 'test' if split == 'test_locked' else 'train')
                provenance = r['provenance']
                original = source_rows[provenance['source_split']][provenance['row_index']]
                self.assertEqual(r['state'], original['text'])
                self.assertEqual(provenance['original_label'], original['label'])
                q = r['questions'][0]
                for order in itertools.permutations(q['options']):
                    self.assertEqual(sum(o['id'] == q['label_id'] for o in order), 1)

    def test_rebuild_is_byte_identical(self):
        original = Path('artifacts/datasets/banking77-choice-v1')
        if not original.exists():
            self.skipTest('build the public dataset first')
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'rebuilt'
            build('configs/banking77_pilot.json', 'artifacts/sources/banking77', output)
            for p in original.rglob('*'):
                if p.is_file():
                    self.assertEqual(p.read_bytes(), (output / p.relative_to(original)).read_bytes(), str(p))


class ArtifactTests(unittest.TestCase):
    def test_upload_budget_rejects_before_any_upload(self):
        with tempfile.TemporaryDirectory() as directory:
            Path(directory, 'weights.bin').write_bytes(b'12345')
            run = MagicMock()
            with self.assertRaises(ValueError):
                log_artifact(run, directory, name='model', kind='model', max_bytes=4)
            run.log_artifact.assert_not_called()

    def test_credentials_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            Path(directory, '.env').write_text('secret')
            with self.assertRaises(ValueError):
                log_artifact(MagicMock(), directory, name='dataset', kind='dataset')

    def test_model_upload_waits_for_persistence(self):
        with tempfile.TemporaryDirectory() as directory, patch('wandb.Artifact') as artifact:
            Path(directory, 'pointer.pt').write_bytes(b'weights')
            run = MagicMock()
            run.log_artifact.return_value.qualified_name = 'team/octo/model:v0'
            result = log_artifact(run, directory, name='model', kind='model', metadata={'dataset_artifact':'team/octo/data:v0'})
            self.assertEqual(result, 'team/octo/model:v0')
            self.assertEqual(artifact.call_args.kwargs['type'], 'model')
            self.assertEqual(artifact.call_args.kwargs['metadata']['dataset_artifact'], 'team/octo/data:v0')
            run.log_artifact.return_value.wait.assert_called_once()

if __name__ == '__main__':
    unittest.main()
