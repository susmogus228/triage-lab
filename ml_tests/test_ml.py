import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch
from ml.training import clean_rows, split_indices, has_known_content


class SplitTests(unittest.TestCase):
    def test_conflicting_labels_and_duplicates_are_handled(self):
        rows = [('Password reset', 'ACCOUNT'), (' password  reset ', 'ACCOUNT'),
                ('Return order', 'ORDER'), ('return order', 'REFUND')]
        cleaned, stats = clean_rows(rows)
        self.assertEqual(cleaned, [('Password reset', 'ACCOUNT')])
        self.assertEqual(stats['duplicate_rows'], 1)
        self.assertEqual(stats['conflicting_groups'], 1)

    def test_training_refuses_to_overwrite_historical_output(self):
        from ml.training import train
        with tempfile.TemporaryDirectory() as tmp:
            report = Path(tmp) / 'history.json'
            report.write_text('original', encoding='utf-8')
            with patch('ml.training.REPORT', report), patch('ml.training.ARTIFACT', Path(tmp) / 'model.joblib'):
                with self.assertRaises(FileExistsError): train()
            self.assertEqual(report.read_text(encoding='utf-8'), 'original')

    def test_disjoint_group_splits(self):
        labels = [str(i % 2) for i in range(60)]
        groups = [str(i // 2) for i in range(60)]
        train, val, test = split_indices(labels, groups)
        self.assertTrue(set(groups[i] for i in train).isdisjoint(groups[i] for i in val))
        self.assertTrue(set(groups[i] for i in train).isdisjoint(groups[i] for i in test))
        self.assertTrue(set(groups[i] for i in val).isdisjoint(groups[i] for i in test))

    def test_only_common_words_do_not_count_as_known_content(self):
        vocabulary = {'the': 0, 'on': 1, 'password': 2}
        self.assertFalse(has_known_content('Unicorns dance on the moon', vocabulary))
        self.assertTrue(has_known_content('My password is broken', vocabulary))

if __name__ == '__main__':
    unittest.main()
