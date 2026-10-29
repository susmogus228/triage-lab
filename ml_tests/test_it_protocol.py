import tempfile, unittest
from unittest.mock import patch
from pathlib import Path
from ml.it_experiment import verify_groups, select_model, verify_frozen, sha256, classification, refine_phase

class ProtocolTests(unittest.TestCase):
    def test_refine_refuses_existing_report(self):
        with tempfile.TemporaryDirectory() as d:
            report=Path(d)/'report.json';report.write_text('existing')
            with patch('ml.it_experiment.REPORT',report), patch('ml.it_experiment.REFINED',Path(d)/'fresh.pt'), patch('ml.it_experiment.SELECTION_V2',Path(d)/'fresh.json'):
                with self.assertRaises(FileExistsError):refine_phase()
    def test_train_dev_group_overlap_rejected(self):
        with self.assertRaises(ValueError):
            verify_groups([{'group_id':'a'}],[{'group_id':'a'}])
    def test_select_only_development_macro_f1(self):
        self.assertEqual(select_model({'tfidf':{'macro_f1':0.4},'multilingual':{'macro_f1':0.6}}),'multilingual')
        self.assertEqual(select_model({'tfidf':{'macro_f1':0.6},'multilingual':{'macro_f1':0.6}}),'tfidf')
    def test_exact_groups_have_equal_weight(self):
        rows=[{'text':'alpha','label':'A','group_id':'a'}, {'text':'  ALPHA ','label':'A','group_id':'a'}, {'text':'beta','label':'B','group_id':'b'}]
        result=classification(rows,['A','A','A'],['A','B'])
        self.assertAlmostEqual(result['accuracy'],2/3)
        self.assertAlmostEqual(result['exact_accuracy'],0.5)
        self.assertEqual(result['exact_unique_groups'],2)
    def test_frozen_input_change_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'a';p.write_text('initial')
            h=sha256(p);verify_frozen({'a':h},{'a':p})
            p.write_text('changed')
            with self.assertRaises(ValueError): verify_frozen({'a':h},{'a':p})

if __name__ == '__main__': unittest.main()
