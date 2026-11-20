"""Reproduce the Kaggle failure: Drive inaccessible before model loading."""
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import kaggle_run
from dataset_source import prepare_snapshot, digest


class DriveUnavailable(RuntimeError):
    pass


class DatasetSourceTests(unittest.TestCase):
    def test_both_snapshots_preserve_locked_bytes_and_replace_partial_cache(self):
        import json
        root = Path(__file__).resolve().parents[1]
        manifest = json.loads((root/'resources/datasets/manifest.json').read_text(encoding='utf-8'))
        with tempfile.TemporaryDirectory() as tmp:
            for dataset in ['fewrel', 'wikizsl']:
                output = Path(tmp)/f'{dataset}.json'
                output.write_text('partial download', encoding='utf-8')
                metadata = prepare_snapshot(root, dataset, output)
                self.assertEqual(digest(output), manifest[dataset]['raw_sha256'])
                self.assertEqual(output.stat().st_size, metadata['raw_bytes'])
                self.assertFalse(metadata['author_identity_verified'])
                rows = json.loads(output.read_text(encoding='utf-8'))
                self.assertEqual(len(rows), metadata['samples'])
                self.assertEqual(len({row['relation'] for row in rows}), metadata['relations'])

    def test_corrupt_archive_is_rejected(self):
        import json
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)/'resources/datasets'
            base.mkdir(parents=True)
            (base/'bad.gz').write_bytes(b'corrupt')
            (base/'manifest.json').write_text(json.dumps({'fewrel':dict(filename='bad.gz',archive_sha256='wrong')}), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'Corrupt dataset archive'):
                prepare_snapshot(Path(tmp),'fewrel',Path(tmp)/'raw.json')

    def test_default_fewrel_reaches_next_stage_without_google_drive(self):
        fake_torch = types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: True))
        def blocked_drive(**kwargs):
            raise DriveUnavailable('Cannot retrieve the public link of the file')
        fake_gdown = types.SimpleNamespace(download=blocked_drive)
        fake_hub = types.SimpleNamespace(snapshot_download=lambda *a, **k: None)
        with tempfile.TemporaryDirectory() as tmp:
            previous = Path.cwd()
            os.chdir(tmp)
            try:
                with patch.dict(sys.modules, {'torch':fake_torch, 'gdown':fake_gdown, 'huggingface_hub':fake_hub}), \
                     patch.object(sys, 'argv', ['kaggle_run', '--dataset', 'fewrel', '--m', '10',
                                               '--run-id', '1', '--deadline', '9999999999', '--smoke']), \
                     patch('urllib.request.urlopen', side_effect=RuntimeError('reached-property-stage')):
                    with self.assertRaisesRegex(RuntimeError, 'reached-property-stage'):
                        kaggle_run.main()
            finally:
                os.chdir(previous)


if __name__ == '__main__':
    unittest.main()
