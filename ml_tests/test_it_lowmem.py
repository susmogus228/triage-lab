import unittest
from pathlib import Path
from unittest.mock import patch
from ml.it_lowmem import require_memory, require_new, planned_batch_count
class LowMemoryTests(unittest.TestCase):
 def test_memory_threshold_prevents_start(self):
  with patch('ml.it_lowmem.available_memory_bytes',return_value=699*1024*1024):
   with self.assertRaises(MemoryError):require_memory(700*1024*1024)
 def test_no_silent_overwrite(self):
  with self.assertRaises(FileExistsError):require_new(Path(__file__))
 def test_partial_accumulation_included(self):
  self.assertEqual(planned_batch_count(9,4),3)
  self.assertEqual(planned_batch_count(8,4),2)
if __name__=='__main__':unittest.main()
