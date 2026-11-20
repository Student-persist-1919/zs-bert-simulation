import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from collect_results import import_zip, rebuild, validate


def evidence(seed=300, smoke=False, status='complete'):
    scientific = dict(dataset='fewrel', m=5, epochs=10, smoke=smoke)
    sig = hashlib.sha256(json.dumps(scientific, sort_keys=True).encode()).hexdigest()
    cfg = dict(scientific, seed=seed, config_hash=sig)
    history = [dict(epoch=i, precision=.5, recall=.6, f1=.55) for i in range(1,11)]
    result = dict(complete=True, m=5, seed=seed, config_hash=sig,
                  train_relations=['seen'], test_relations=list('abcde'),
                  history=history, best=history[0], final=history[-1])
    objects = {'experiment_config.json':cfg, 'status.json':dict(status=status,seed=seed,config_hash=sig),
               'provenance.json':{}, 'relation_descriptions.json':{},
               f'run_seed_{seed}.json':result, f'history_seed_{seed}.json':history}
    files = {k:json.dumps(v).encode() for k,v in objects.items()}
    files.update({'attributes.npz':b'fixture', f'train_seed_{seed}.log':b'fixture log'})
    files['manifest.json'] = json.dumps({k:hashlib.sha256(v).hexdigest() for k,v in files.items()}).encode()
    return files


class EvidenceTests(unittest.TestCase):
    def test_checksum_and_smoke_rejected(self):
        files=evidence(); files['attributes.npz']=b'tampered'
        with self.assertRaises(ValueError):validate(files)
        with self.assertRaises(ValueError):validate(evidence(smoke=True))

    def test_import_aggregate_and_duplicate_attempt(self):
        with tempfile.TemporaryDirectory() as tmp:
            base=Path(tmp)/'results'
            for seed in range(300,305):
                archive=Path(tmp)/f'{seed}.zip'
                with zipfile.ZipFile(archive,'w') as z:
                    for k,v in evidence(seed).items():z.writestr(f'run/{k}',v)
                import_zip(archive,'original','https://www.kaggle.com/code/user/run?scriptVersionId=1',base)
            rebuild(base)
            self.assertIn('55.00 ± 0.00',(base/'RESULTS.md').read_text(encoding='utf-8'))
            with self.assertRaises(FileExistsError):
                import_zip(archive,'original','https://www.kaggle.com/code/user/run',base)
            import_zip(archive,'retry','https://www.kaggle.com/code/user/run',base)
            rebuild(base)
            self.assertIn('Chưa tổng hợp',(base/'RESULTS.md').read_text(encoding='utf-8'))

    def test_paused_not_aggregated(self):
        cfg,status=validate(evidence(status='paused'))
        self.assertEqual(status['status'],'paused')

    def test_zip_traversal_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'bad.zip'
            with zipfile.ZipFile(path,'w') as z:z.writestr('../evil.json','{}')
            with self.assertRaises(ValueError):
                import_zip(path,'original','https://www.kaggle.com/code/user/run',Path(tmp)/'results')


if __name__=='__main__':unittest.main()
