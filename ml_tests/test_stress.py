import json
import unittest
from unittest.mock import patch
import numpy as np
from ml.stress_eval import metrics, model_identity, active_cases

class Model:
    classes_=np.array(['ACCOUNT','ORDER'])
    named_steps={'tfidf':type('Vectorizer',(),{'vocabulary_':{'account':0,'order':1}})()}
    def predict_proba(self,texts):return np.array([[.9,.1],[.2,.8],[.8,.2]])

class EvaluationTests(unittest.TestCase):
    def test_dev_selection_never_implies_activation(self):
        models=[{'key':'baseline','cases':['baseline-example'],**model_identity('baseline','candidate')},
                {'key':'candidate','cases':['candidate-example'],**model_identity('candidate','candidate')}]
        self.assertEqual(models[0]['role'],'active')
        self.assertTrue(models[1]['selected_on_dev'])
        self.assertEqual(models[1]['role'],'comparison')
        self.assertEqual(active_cases(models),['baseline-example'])
    def test_known_error_and_unknown_auto_are_both_unsafe(self):
        rows=[{'id':'1','text':'account help','expected':'ACCOUNT','kind':'in_scope'},
              {'id':'2','text':'account cannot login','expected':'ACCOUNT','kind':'in_scope'},
              {'id':'3','text':'order pizza','expected':'REVIEW','kind':'out_of_scope','scenario':'unknown'}]
        result=metrics(rows,{'pipeline':Model()})
        self.assertEqual(result['accuracy'],.5)
        self.assertEqual(result['unsafe_auto_count'],2)
        self.assertEqual(result['should_review_recall'],0)
        self.assertEqual(result['counts']['accepted_in_scope'],2)
        self.assertEqual(result['review_scenarios']['unknown'],{'total':1,'reviewed':0})

    def test_unknown_vocabulary_forces_review(self):
        rows=[{'id':'1','text':'nonexistent words','expected':'ACCOUNT','kind':'in_scope'},
              {'id':'2','text':'account','expected':'ACCOUNT','kind':'in_scope'},
              {'id':'3','text':'pizza','expected':'REVIEW','kind':'out_of_scope','scenario':'unknown'}]
        result=metrics(rows,{'pipeline':Model()})
        self.assertTrue(result['cases'][0]['needs_review'])
        self.assertTrue(result['cases'][2]['needs_review'])
        self.assertEqual(result['should_review_recall'],1)
if __name__=='__main__':unittest.main()
