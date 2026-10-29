import os
import tempfile
import unittest
from unittest.mock import patch
from fastapi.testclient import TestClient

class ApiTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        os.environ['TRIAGELAB_DB'] = os.path.join(self.tmp.name, 'test.sqlite')
        from backend.app import app, init_db
        init_db()
        self.client = TestClient(app)
        self.pred = patch('backend.app.predict', return_value={'label':'account','confidence':0.91,'alternatives':[], 'needs_review':False,'model_version':'test'})
        self.pred.start()
    def tearDown(self):
        self.pred.stop(); self.client.close(); self.tmp.cleanup()
    def test_create_read_correct_and_timeline(self):
        res = self.client.post('/api/tickets', json={'subject':'Cannot login','body':'Access is denied', 'source':'storefront'})
        self.assertEqual(res.status_code, 201)
        ticket = res.json()['ticket']; self.assertEqual(ticket['predicted_topic'], 'account')
        self.assertEqual(self.client.get('/api/tickets').json()['tickets'][0]['id'],ticket['id'])
        result = self.client.patch('/api/tickets/%s' % ticket['id'], json={'corrected_topic':'billing'})
        self.assertEqual(result.status_code,200)
        detail = self.client.get('/api/tickets/%s' % ticket['id']).json()
        self.assertEqual(detail['ticket']['corrected_topic'],'billing')
        self.assertGreaterEqual(len(detail['events']),3)
    def test_validation_and_not_found(self):
        self.assertEqual(self.client.post('/api/tickets',json={'subject':'','body':'','source':'other'}).status_code,422)
        self.assertEqual(self.client.get('/api/tickets/123').status_code,404)
    def test_duplicate_mail_import(self):
        from backend.app import import_message
        item={'ID':'same-id','Subject':'Cannot sign in','From':{'Address':'test@example.com'},'Text':'Please help me sign in'}
        self.assertTrue(import_message(item)); self.assertFalse(import_message(item))
        self.assertEqual(len(self.client.get('/api/tickets').json()['tickets']),1)
    def test_evaluation_available_and_missing(self):
        from pathlib import Path
        import json
        path=Path(self.tmp.name)/'report.json'
        with patch('backend.app.EVALUATION_REPORT',path):
            self.assertFalse(self.client.get('/api/evaluation').json()['available'])
            payload={'available':True,'dataset':{'count':1},'models':[], 'cases':[{'id':'case-1'}]}
            path.write_text(json.dumps(payload),encoding='utf-8')
            self.assertEqual(self.client.get('/api/evaluation').json(),payload)

    def test_operator_can_record_out_of_scope_review(self):
        ticket=self.client.post('/api/tickets',json={'subject':'Unrelated','body':'Something else','source':'storefront'}).json()['ticket']
        response=self.client.patch('/api/tickets/%s' % ticket['id'],json={'corrected_topic':'REVIEW'})
        self.assertEqual(response.status_code,200)
        self.assertEqual(response.json()['ticket']['corrected_topic'],'REVIEW')
        self.assertEqual(response.json()['ticket']['status'],'reviewed')
        self.assertEqual(response.json()['ticket']['predicted_topic'],'account')

    def test_evolution_endpoint_missing_and_manifest(self):
        import json
        from pathlib import Path
        report=Path(self.tmp.name)/'evolution.json'; manifest=Path(self.tmp.name)/'manifest.json'
        with patch('backend.app.EVOLUTION_REPORT',report), patch('backend.app.EVOLUTION_MANIFEST',manifest):
            self.assertFalse(self.client.get('/api/evolution').json()['available'])
            report.write_text(json.dumps({'available':True, 'active_model':'baseline'}),encoding='utf-8')
            manifest.write_text(json.dumps({'counts':{'fit':10}}),encoding='utf-8')
            payload=self.client.get('/api/evolution').json()
            self.assertEqual(payload['active_model'],'baseline')
            self.assertEqual(payload['data_manifest']['counts']['fit'],10)

    def test_it_experiment_available_and_absent(self):
        import json
        from pathlib import Path
        report=Path(self.tmp.name)/'it_report.json'
        with patch('backend.app.IT_EXPERIMENT_REPORT',report):
            self.assertFalse(self.client.get('/api/it-experiment').json()['available'])
            payload={'available':True,'dataset':{'name':'IT tickets'},'models':[]}
            report.write_text(json.dumps(payload),encoding='utf-8')
            self.assertEqual(self.client.get('/api/it-experiment').json(),payload)

    def test_it_prediction_is_local_and_always_needs_review(self):
        with patch('backend.app.predict_it',return_value={'label':'Software','confidence':0.91,'needs_review':True,'model_version':'it-test'}):
            response=self.client.post('/api/it-predict',json={'text':'My software fails to start'})
        self.assertEqual(response.status_code,200)
        self.assertEqual(response.json()['label'],'Software')
        self.assertTrue(response.json()['needs_review'])
        self.assertEqual(self.client.post('/api/it-predict',json={'text':'   '}).status_code,422)

    def test_it_hash_failure_returns_unavailable(self):
        with patch('ml.it_inference.predict',side_effect=ValueError('Frozen input changed')):
            r=self.client.post('/api/it-predict',json={'text':'Cannot log in'})
        self.assertEqual(r.status_code,503)

    def test_it_v2_report_missing_and_present(self):
        import json
        from pathlib import Path
        report=Path(self.tmp.name)/'report.json'
        with patch('backend.app.IT_V2_REPORT',report):
            self.assertFalse(self.client.get('/api/it-v2').json()['available'])
            value={'available':True,'active':False,'models':[]}
            report.write_text(json.dumps(value),encoding='utf-8')
            self.assertEqual(self.client.get('/api/it-v2').json(),value)

    def test_it_v2_adds_optional_historical_diagnostic(self):
        import json
        from pathlib import Path
        report=Path(self.tmp.name)/'report.json'
        historical=report.parent/'it_v2_historical_diagnostic.json'
        report.write_text(json.dumps({'available':True,'active':False}),encoding='utf-8')
        historical.write_text(json.dumps({'eol':{'exact_unique':2}}),encoding='utf-8')
        with patch('backend.app.IT_V2_REPORT',report):
            body=self.client.get('/api/it-v2').json()
        self.assertEqual(body['historical_diagnostic']['eol']['exact_unique'],2)

    def test_test_email_rejects_header_injection(self):
        for payload in (
            {'subject':'hello\nBCC: attacker@example.com','body':'help','email':'test@example.com'},
            {'subject':'hello','body':'help','email':'test@example.com\r\nBCC: attacker@example.com'},
            {'subject':'hello','body':'help','email':'not-an-address'},
            {'subject':'hello','body':'help','email':'a,b@example.com'},
            {'subject':'hello','body':'help','email':'a@ex:ample.com'},
        ):
            with patch('backend.app.smtplib.SMTP') as smtp:
                result=self.client.post('/api/test-email',json=payload)
            self.assertEqual(result.status_code,422)
            smtp.assert_not_called()

    def test_missing_model_report_state(self):
        from pathlib import Path
        with patch('ml.inference.ARTIFACT',Path(self.tmp.name)/'missing.joblib'):
            r=self.client.get('/api/model')
        self.assertEqual(r.status_code,200)
        self.assertFalse(r.json()['available'])
    def test_low_score_requires_review(self):
        with patch('backend.app.predict',return_value={'label':'ORDER','confidence':0.39,'needs_review':False,'model_version':'test'}):
            r=self.client.post('/api/tickets',json={'subject':'Status','body':'Where is my order','source':'storefront'})
        self.assertEqual(r.status_code,201)
        self.assertTrue(r.json()['ticket']['needs_review'])
        self.assertEqual(r.json()['ticket']['status'],'needs_review')
    def test_html_only_email_is_not_lost(self):
        from backend.app import import_message
        item={'ID':'html123','Subject':'Help','From':{'Address':'test@example.com'},'HTML':'<p>Login broken</p><script>bad()</script>'}
        self.assertTrue(import_message(item))
        ticket=self.client.get('/api/tickets').json()['tickets'][0]
        self.assertIn('Login broken',ticket['body'])
        self.assertNotIn('bad()',ticket['body'])
    def test_mail_sent_even_when_import_temporarily_fails(self):
        from fastapi import HTTPException
        with patch('backend.app.smtplib.SMTP') as smtp, patch('backend.app.sync_mail',side_effect=HTTPException(503,'offline')):
            r=self.client.post('/api/test-email',json={'subject':'Help','body':'Please help','email':'test@example.com'})
        self.assertEqual(r.status_code,200); self.assertTrue(r.json()['sent']); self.assertIsNone(r.json()['imported'])
    def test_unavailable_model_is_not_fabricated(self):
        from backend.app import ModelUnavailable
        with patch('backend.app.predict', side_effect=ModelUnavailable('model missing')):
            r=self.client.post('/api/tickets',json={'subject':'Hello','body':'Help please','source':'help-center'})
        self.assertEqual(r.status_code,201)
        self.assertIsNone(r.json()['ticket']['predicted_topic'])
        self.assertTrue(r.json()['ticket']['needs_review'])

if __name__ == '__main__': unittest.main()
