import unittest, tempfile
from pathlib import Path
import numpy as np
from ml.it_v2 import choose_ood_threshold, decision_metrics, ensure_fresh, group_folds, prediction_summary, verify_baseline

class ProtocolTests(unittest.TestCase):
    def test_ood_threshold_controls_known_rejection(self):
        # Larger scores mean more likely unknown; reject if >= threshold.
        result=choose_ood_threshold([0.1,0.2,0.3,0.9],[0.7,0.8,0.9], max_reject=.25)
        self.assertLessEqual(result['known_reject_rate'],.25)
        self.assertGreater(result['unknown_recall'],0)
    def test_counts_separate_known_and_unknown(self):
        m=decision_metrics([False,False,True],[False,True,True])
        self.assertEqual(m['known_n'],2);self.assertEqual(m['unknown_n'],1)
        self.assertEqual(m['known_rejected'],1);self.assertEqual(m['unknown_detected'],1)
    def test_existing_evidence_never_overwritten(self):
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp)/'x';ensure_fresh(p);p.write_text('keep')
            with self.assertRaises(FileExistsError):ensure_fresh(p)
            self.assertEqual(p.read_text(),'keep')
    def test_unknown_and_ambiguous_counts_remain_distinct(self):
        rows=[{'expected':'Software'},{'expected':'UNKNOWN'},{'expected':'AMBIGUOUS'}]
        result=prediction_summary(rows,['Software','Software','Software'],[False,True,False],['Software'])
        self.assertEqual(result['known_accuracy'],1.0)
        self.assertEqual(result['unknown_recall'],1.0)
        self.assertEqual(result['ambiguous_recall'],0.0)
        self.assertEqual(result['unsafe_count'],1)
        self.assertNotIn('text',str(result))
    def test_no_unknown_does_not_fake_zero(self):
        result=prediction_summary([{'expected':'Software'}],['Software'],[True],['Software'])
        self.assertIsNone(result['unknown_recall'])
        self.assertIsNone(result['ambiguous_recall'])

    def test_baseline_must_be_frozen_before_final(self):
        import hashlib
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'baseline.joblib';path.write_bytes(b'trusted')
            with self.assertRaises(ValueError):verify_baseline({'hashes':{}},path)
            with self.assertRaises(ValueError):verify_baseline({'hashes':{'baseline':'wrong'}},path)
            verify_baseline({'hashes':{'baseline':hashlib.sha256(b'trusted').hexdigest()}},path)

    def test_groups_never_cross_folds(self):
        labels=np.array(['a','a','a','b','b','b']);groups=np.array(['1','1','2','3','3','4'])
        folds=group_folds(labels,groups,n_splits=2)
        for a,b in folds:self.assertFalse(set(groups[a])&set(groups[b]))
if __name__=='__main__':unittest.main()

class NewRunStatusTests(unittest.TestCase):
    def test_neural_status_is_current_run_not_historical(self):
        from ml.it_v2 import neural_status_for_selection
        self.assertEqual(neural_status_for_selection()['status'], 'not_attempted_in_this_run')
