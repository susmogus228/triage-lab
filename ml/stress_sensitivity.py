"""Post-hoc sensitivity of frozen H059 annotation; no model or holdout changes."""
import json
from datetime import datetime, timezone
from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score
from ml.stress_eval import ROOT, sha

ORIGINAL=ROOT/'artifacts/stress_report.json'
OUTPUT=ROOT/'artifacts/stress_sensitivity.json'

def recalculate(cases):
    known=[r for r in cases if r['kind']=='in_scope']
    review=[r for r in cases if r['kind']!='in_scope']
    accepted=[r for r in known if not r['needs_review']]
    unsafe=[r for r in cases if not r['needs_review'] and (r['kind']!='in_scope' or r['expected']!=r['predicted'])]
    return {'accuracy':float(accuracy_score([r['expected'] for r in known],[r['predicted'] for r in known])),
            'macro_f1':float(f1_score([r['expected'] for r in known],[r['predicted'] for r in known],average='macro',zero_division=0)),
            'coverage':sum(not r['needs_review'] for r in cases)/len(cases),
            'accepted_in_scope_accuracy':sum(r['predicted']==r['expected'] for r in accepted)/len(accepted) if accepted else None,
            'should_review_recall':sum(r['needs_review'] for r in review)/len(review),
            'unsafe_auto_count':len(unsafe),
            'counts':{'total':len(cases),'in_scope':len(known),'should_review':len(review),'accepted_in_scope':len(accepted)}}

def main():
    source=json.loads(ORIGINAL.read_text(encoding='utf-8'))
    result={'schema_version':'1.0','created_utc':datetime.now(timezone.utc).isoformat(),
            'case_id':'H059','original_expected':'REVIEW','sensitivity_expected':'ORDER',
            'rationale':'Bitext has ORDER/track_order and DELIVERY/delivery_period intents; an uncertain parcel-tracking status request may reasonably be ORDER, but overlaps with DELIVERY. This alternative is post-hoc and unverified by a second human annotator.',
            'selection_changed':False,'holdout_changed':False,'source_holdout_sha256':source['dataset']['sha256'],
            'source_selection_sha256':sha(ROOT/'artifacts/stress_selection.json'),'models':[]}
    for m in source['models']:
        cases=[dict(r) for r in m['cases']]
        case=next(r for r in cases if r['id']=='H059');case['expected']='ORDER';case['kind']='in_scope'
        before={key:m[key] for key in ['accuracy','macro_f1','coverage','accepted_in_scope_accuracy','should_review_recall','unsafe_auto_count','counts']}
        result['models'].append({'name':m['name'],'key':m['key'],'role':m['role'],'original':before,'sensitivity':recalculate(cases), 'case':{'predicted':case['predicted'],'confidence':case['confidence'],'needs_review':case['needs_review']}})
    OUTPUT.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    for m in result['models']:
        print(m['key'],json.dumps(m['sensitivity']))
if __name__=='__main__':main()
