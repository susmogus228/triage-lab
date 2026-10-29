"""Post-hoc diagnostic alternative annotations; no model choice is changed."""
import json
from pathlib import Path
import joblib
from ml.it_v2 import DATA, ART, MODEL, DETECTOR, SELECTION, REPORT, read_jsonl, prediction_summary, sha, ensure_fresh

OUTPUT=ART/'it_v2_sensitivity.json'; BLIND=DATA/'it_v2_blind.jsonl'
EXPECTED='01ce1d6aa441b25abc282c61ac51d95d0fc9d3be85403249de14741d7c2c4a92'
ALTER={'ITV2-003','ITV2-006','ITV2-025','ITV2-039','ITV2-046','ITV2-052','ITV2-054'} | {f'ITV2-{i:03}' for i in range(71,81)} | {'ITV2-085'}

def main():
 ensure_fresh(OUTPUT)
 if not REPORT.exists():raise RuntimeError('Final report not available')
 report=json.loads(REPORT.read_text(encoding='utf8'));selection=json.loads(SELECTION.read_text(encoding='utf8'))
 if sha(SELECTION)!=report['selection']['sha256'] or sha(BLIND)!=EXPECTED:raise ValueError('Frozen artifact changed')
 if sha(MODEL)!=selection['hashes']['topic'] or sha(DETECTOR)!=selection['hashes']['detector']:raise ValueError('Model changed')
 rows=read_jsonl(BLIND)
 if not ALTER.issubset({r['id'] for r in rows}):raise ValueError('Missing sensitivity ID')
 texts=[r['text'] for r in rows];model=joblib.load(MODEL);det=joblib.load(DETECTOR);base=joblib.load(ART/'it_tfidf.joblib');p=model.predict(texts).tolist();bp=base.predict(texts).tolist()
 policy=selection['ood_policy'];score=det.predict_proba(texts)[:,1] if policy['name']=='binary' else 1-model.predict_proba(texts).max(axis=1);flags=(score>=policy['threshold']).tolist()
 alternate=[dict(r,expected='AMBIGUOUS') if r['id'] in ALTER else r for r in rows]
 result={'description':'Post-hoc label sensitivity based on blind reviewer caveats. Alternate labels represent one possible taxonomy interpretation, not established truth. No thresholds, model, or original labels were changed.', 'changed_case_count':len(ALTER),'baseline':prediction_summary(alternate,bp,[True]*len(rows),selection['classes']),'candidate':prediction_summary(alternate,p,flags,selection['classes']),'original_report_sha256':sha(REPORT),'selection_sha256':sha(SELECTION),'blind_sha256':EXPECTED}
 OUTPUT.write_text(json.dumps(result,indent=2)+'\n',encoding='utf8');print(json.dumps({'changed':len(ALTER),'baseline':result['baseline'],'candidate':result['candidate']}),flush=True)
if __name__=='__main__':main()
