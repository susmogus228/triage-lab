import unittest
import tempfile
import json
from unittest.mock import patch
from pathlib import Path
from ml.it_inference import predict, _load

class InferenceTests(unittest.TestCase):
    def test_untrained_model_gives_clear_failure(self):
        with patch('ml.it_inference.SELECTION',Path('definitely-not-present.json')):
            with self.assertRaises(RuntimeError): predict('Account login fails')
    def test_stale_selection_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            sel=Path(d)/'sel.json';new=Path(d)/'absent.json';report=Path(d)/'report.json'
            sel.write_text(json.dumps({'selected':'tfidf','labels':['A']}),encoding='utf-8')
            report.write_text(json.dumps({'selected':'tfidf','selection_sha256':'bad'}),encoding='utf-8')
            _load.cache_clear()
            with patch('ml.it_inference.SELECTION',sel), patch('ml.it_inference.SELECTION_V2',new), patch('ml.it_inference.REPORT',report):
                with self.assertRaises(RuntimeError):predict('Test request')
            _load.cache_clear()
    def test_empty_is_rejected(self):
        with self.assertRaises(ValueError): predict('   ')
if __name__ == '__main__': unittest.main()
