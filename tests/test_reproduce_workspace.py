import tempfile
import unittest
import os
import subprocess
import sys
from pathlib import Path
from ml.reproduce_workspace import create_workspace, validate_name

ROOT = Path(__file__).resolve().parents[1]

class ReproductionWorkspaceTests(unittest.TestCase):
    def test_name_does_not_allow_path_traversal(self):
        for name in ('..', '../escape', 'a/b', 'a\\\\b', '', '.', 'CON', 'con.txt'):
            with self.subTest(name=name), self.assertRaises(ValueError):
                validate_name(name)
    def test_separate_workspace_excludes_user_data_and_refuses_overwrite(self):
        work = ROOT/'work'; work.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=work) as parent:
            target = Path(parent)/'fresh'
            create_workspace(ROOT,target)
            self.assertTrue((target/'ml/training.py').is_file())
            self.assertTrue((target/'data/it_support_prepare.py').is_file())
            self.assertTrue((target/'frontend/src/main.jsx').is_file())
            self.assertFalse((target/'data/bitext.csv').exists())
            self.assertFalse((target/'data/it_support_raw').exists())
            self.assertFalse((target/'work').exists())
            self.assertFalse((target/'.venv').exists())
            self.assertEqual(list((target/'artifacts').iterdir()),[])
            marker=target/'do-not-touch';marker.write_text('preserve',encoding='utf-8')
            with self.assertRaises(FileExistsError):create_workspace(ROOT,target)
            self.assertEqual(marker.read_text(encoding='utf-8'),'preserve')

    def test_experiment_commands_import_inside_isolated_copy(self):
        work = ROOT/'work'; work.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=work) as parent:
            target = Path(parent)/'fresh'
            create_workspace(ROOT, target)
            env = os.environ.copy()
            env['PYTHONPATH'] = str(target)
            env['PYTHONNOUSERSITE'] = '1'
            for module in ('ml.it_experiment', 'ml.it_v2'):
                with self.subTest(module=module):
                    result = subprocess.run([sys.executable, '-m', module, '--help'], cwd=target,
                                            env=env, capture_output=True, text=True)
                    self.assertEqual(result.returncode, 0, result.stderr)
                    self.assertIn('select', result.stdout)

if __name__=='__main__':unittest.main()
