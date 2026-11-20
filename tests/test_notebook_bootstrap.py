import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
FULL = 'b4b84208cf43b4274f2d2aaaf14beead0daf4707'
URL = 'https://github.com/Student-persist-1919/zs-bert-simulation.git'


class BootstrapTests(unittest.TestCase):
    def test_short_commit_is_resolved_locally_before_checkout(self):
        for m in [5,10,15]:
            nb=json.loads((ROOT/f'notebooks/kaggle/zs-bert-m-{m}.ipynb').read_text(encoding='utf-8'))
            source=next(''.join(c['source']) for c in nb['cells'] if 'Unexpected repository origin' in ''.join(c['source']))
            commands=[]
            def run(cmd, **kwargs):
                commands.append(cmd)
                if 'fetch' in cmd and cmd[-1]=='b4b8420':
                    raise subprocess.CalledProcessError(128,cmd,stderr='could not find remote ref b4b8420')
                return subprocess.CompletedProcess(cmd,0)
            def output(cmd, **kwargs):
                if 'get-url' in cmd:return URL+'\n'
                if '--verify' in cmd and 'refs/remotes/origin/b4b8420' in cmd[-1]:
                    raise subprocess.CalledProcessError(128,cmd)
                return FULL+'\n'
            with tempfile.TemporaryDirectory() as tmp:
                replacement=source.replace('Path("/kaggle/working")',f'Path({tmp!r})')
                with patch('subprocess.run',side_effect=run), patch('subprocess.check_output',side_effect=output):
                    exec(compile(replacement,'notebook_setup','exec'),{'CODE_REV':'b4b8420'})
                checkout=next(c for c in commands if 'checkout' in c)
                self.assertEqual(checkout[-1],FULL)


if __name__=='__main__':unittest.main()
