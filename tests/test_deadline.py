"""Exercise the actual trainer's deadline helper without importing GPU libraries."""
import ast
from pathlib import Path
import unittest


class DeadlineTests(unittest.TestCase):
    def test_pause_only_when_due_and_preserve_resume_position(self):
        source = Path('scripts/train_run.py').read_text(encoding='utf-8')
        tree = ast.parse(source)
        helper = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == 'pause_if_due')
        calls, records = [], []
        class Clock:
            now = 9
            @classmethod
            def time(cls):return cls.now
        namespace = dict(time=Clock, cfg={'deadline_unix':10, 'config_hash':'test'}, seed=301,
                         out=Path('unused'), save_checkpoint=lambda *a:calls.append(a),
                         write_json=lambda p,v:records.append(v))
        exec(compile(ast.fix_missing_locations(ast.Module(body=[helper],type_ignores=[])), 'trainer_helper', 'exec'),namespace)
        totals=[3.5, 7]
        self.assertFalse(namespace['pause_if_due'](2,7,totals))
        self.assertEqual(calls,[])
        Clock.now=10
        self.assertTrue(namespace['pause_if_due'](2,7,totals))
        self.assertEqual(calls,[(2,7,totals)])
        self.assertEqual(records[0]['next_step'],7)
        self.assertEqual(records[0]['status'],'paused')


if __name__=='__main__':unittest.main()
