import unittest,tempfile
from pathlib import Path
from unittest.mock import patch
from ml.it_long import require_memory, require_new, is_better

class LongProtocolTests(unittest.TestCase):
 def test_free_memory_gate(self):
  with patch('ml.it_long.available_memory_bytes',return_value=699*1024*1024):
   with self.assertRaises(MemoryError):require_memory()
  with patch('ml.it_long.available_memory_bytes',return_value=701*1024*1024):require_memory()
 def test_evidence_files_not_overwritten(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'x';require_new(p);p.write_text('keep')
   with self.assertRaises(FileExistsError):require_new(p)
   self.assertEqual(p.read_text(),'keep')
 def test_best_checkpoint_only_on_improvement(self):
  self.assertTrue(is_better(.717,.71618));self.assertFalse(is_better(.71618,.71618))
if __name__=='__main__':unittest.main()
