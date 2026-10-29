import json
import unittest
import tempfile
from pathlib import Path
from ml.evolution import combine_examples, metrics, choose_candidate, assert_disjoint, normalize_expected, require_new_outputs

class EvolutionTests(unittest.TestCase):
    def test_dev_labels_normalized_as_review(self):
        self.assertEqual(normalize_expected('MULTI'),'REVIEW')
        self.assertEqual(normalize_expected('OUT_OF_SCOPE'),'REVIEW')
        self.assertEqual(normalize_expected('ACCOUNT'),'ACCOUNT')
    def test_existing_evidence_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'report.json'
            require_new_outputs([path])
            path.write_text('historical',encoding='utf-8')
            with self.assertRaises(FileExistsError): require_new_outputs([path])
            self.assertEqual(path.read_text(encoding='utf-8'),'historical')
    def test_no_cross_group_source(self):
        assert_disjoint({'fit': [{'origin_ids':['a','b']}], 'feedback':[{'origin_ids':['c']}]})
        with self.assertRaises(ValueError): assert_disjoint({'fit':[{'origin_ids':['a']}], 'feedback':[{'origin_ids':['a','c']}]})
    def test_same_connector_can_produce_single_and_multi(self):
        x=combine_examples('hello','world',' and ')
        self.assertEqual(x,'hello and world')
    def test_failed_candidate_never_promoted(self):
        base={'unsafe_count':5,'coverage':0.5}
        candidates=[{'name':'worse','unsafe_count':6,'coverage':0.8},{'name':'overreject','unsafe_count':2,'coverage':0.1}]
        self.assertEqual(choose_candidate(base,candidates)['name'],'baseline')
    def test_metric_review_and_error(self):
        cases=[{'expected':'ACCOUNT','predicted':'ACCOUNT','needs_review':False},{'expected':'REVIEW','predicted':'ACCOUNT','needs_review':False},{'expected':'ACCOUNT','predicted':'ORDER','needs_review':True}]
        result=metrics(cases)
        self.assertEqual(result['unsafe_count'],1)
        self.assertEqual(result['review_recall'],0)
        self.assertEqual(result['known_accuracy'],0.5)

if __name__=='__main__': unittest.main()
