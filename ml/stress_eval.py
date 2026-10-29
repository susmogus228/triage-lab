"""Frozen development selection and separately evaluated stress benchmark.

Data are synthetic examples. They do not establish real-world performance.
The held-out file must never influence candidate fitting, selection, or threshold.
"""
import argparse, csv, hashlib, json
from pathlib import Path
from datetime import datetime, timezone
from collections import Counter
import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score
from ml.training import clean_rows, split_indices, normalize, has_known_content, DATA
ROOT=Path(__file__).resolve().parents[1]
ART=ROOT/'artifacts'; DEV=ROOT/'data/stress_dev.json'; HOLD=ROOT/'data/holdout_v1.jsonl'
BASE=ART/'ticket_classifier.joblib'; CANDIDATE=ART/'stress_char_candidate.joblib'; SELECTION=ART/'stress_selection.json'; REPORT=ART/'stress_report.json'
THRESHOLD=0.6

def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def metrics(rows,artifact):
 model=artifact['pipeline']; classes=model.classes_; p=model.predict_proba([r['text'] for r in rows]); ix=p.argmax(axis=1)
 vocab=artifact.get('content_vocabulary',model.named_steps['tfidf'].vocabulary_)
 cases=[]
 for row,index,probs in zip(rows,ix,p):
  score=float(probs[index]); known=has_known_content(row['text'],vocab)
  review=not known or score < THRESHOLD
  cases.append({**row,'predicted':str(classes[index]),'confidence':score,'needs_review':bool(review),'known_content':bool(known)})
 in_scope=[c for c in cases if c['kind']=='in_scope']; reject=[c for c in cases if c['kind']!='in_scope']
 accepted=[c for c in in_scope if not c['needs_review']]
 unsafe=[c for c in cases if not c['needs_review'] and (c['kind']!='in_scope' or c['predicted']!=c['expected'])]
 return {'accuracy':float(accuracy_score([c['expected'] for c in in_scope],[c['predicted'] for c in in_scope])) if in_scope else None,
 'macro_f1':float(f1_score([c['expected'] for c in in_scope],[c['predicted'] for c in in_scope],average='macro',zero_division=0)) if in_scope else None,
 'coverage':sum(not c['needs_review'] for c in cases)/len(cases),
 'accepted_in_scope_accuracy':sum(c['predicted']==c['expected'] for c in accepted)/len(accepted) if accepted else None,
 'should_review_recall':sum(c['needs_review'] for c in reject)/len(reject) if reject else None,
 'unsafe_auto_count':len(unsafe),'unsafe_auto_rate':len(unsafe)/len(cases),
 'counts':{'total':len(cases),'in_scope':len(in_scope),'should_review':len(reject),'accepted_in_scope':len(accepted)},
 'review_scenarios':{scenario:{'total':len(group),'reviewed':sum(c['needs_review'] for c in group)} for scenario in sorted({c.get('scenario','unknown') for c in reject}) for group in [[c for c in reject if c.get('scenario','unknown')==scenario]]},
 'per_category':{label:{'total':len(group),'correct':sum(c['predicted']==c['expected'] for c in group),'reviewed':sum(c['needs_review'] for c in group)} for label in sorted({c['expected'] for c in in_scope}) for group in [[c for c in in_scope if c['expected']==label]]},'cases':cases}

def select():
 rows=json.loads(DEV.read_text(encoding='utf-8')); baseline=joblib.load(BASE)
 with DATA.open(encoding='utf-8-sig',newline='') as f: raw=[(r['instruction'],r['category']) for r in csv.DictReader(f)]
 cleaned,_=clean_rows(raw); texts=np.array([r[0] for r in cleaned]); labels=np.array([r[1] for r in cleaned]); train_i,_,_=split_indices(labels,[normalize(t) for t in texts])
 if set(normalize(r['text']) for r in rows) & set(normalize(t) for t in texts): raise ValueError('Exact overlap between development and source data')
 candidate=Pipeline([('tfidf',TfidfVectorizer(analyzer='char_wb',ngram_range=(3,5),min_df=2,sublinear_tf=True,max_features=60000)),('classifier',LogisticRegression(max_iter=300,random_state=42))])
 candidate.fit(texts[train_i],labels[train_i])
 c={'pipeline':candidate,'threshold':THRESHOLD,'model_version':'bitext-char-logreg-candidate-v2','content_vocabulary':baseline['pipeline'].named_steps['tfidf'].vocabulary_}
 joblib.dump(c,CANDIDATE)
 b=metrics(rows,baseline); m=metrics(rows,c)
 # Predeclared conservative gate: measurable dev accuracy gain without more unsafe automation.
 promote=(m['accuracy']>=b['accuracy']+0.03 and m['unsafe_auto_count']<=b['unsafe_auto_count'])
 selection={'created_utc':datetime.now(timezone.utc).isoformat(),'protocol':'Candidate and gate fixed before reading independent holdout. Threshold 0.6 fixed for both, matching app policy.', 'dev_sha256':sha(DEV),'baseline_sha256':sha(BASE),'candidate_sha256':sha(CANDIDATE),'source_sha256':sha(DATA),'selected':'candidate' if promote else 'baseline','reason':'At least 3 percentage points higher in-scope accuracy and no extra unsafe automatic decisions on dev' if promote else 'Candidate did not meet predeclared accuracy and unsafe-decision gates on dev', 'development':{'baseline':{k:v for k,v in b.items() if k!='cases'},'candidate':{k:v for k,v in m.items() if k!='cases'}}}
 SELECTION.write_text(json.dumps(selection,indent=2)+'\n',encoding='utf-8'); print(json.dumps(selection,indent=2))

def model_identity(key, selected_on_dev):
 return {'role':'active' if key == 'baseline' else 'comparison', 'selected_on_dev':key == selected_on_dev}

def active_cases(models):
 return next(m['cases'] for m in models if m['key']=='baseline')

def final():
 sel=json.loads(SELECTION.read_text(encoding='utf-8'))
 for key,path in [('dev_sha256',DEV),('baseline_sha256',BASE),('candidate_sha256',CANDIDATE),('source_sha256',DATA)]:
  if sel[key]!=sha(path): raise ValueError(f'Frozen input changed: {path}')
 if not HOLD.exists(): raise FileNotFoundError(f'Independent holdout missing: {HOLD}')
 rows=[json.loads(line) for line in HOLD.read_text(encoding='utf-8-sig').splitlines() if line.strip()]
 # Do not tune after this point. Resolve labels with holdout owner if schema differs.
 if any(not all(k in row for k in ('id','text','expected','scenario')) for row in rows): raise ValueError('Holdout requires id,text,expected,scenario')
 for row in rows:
  row['kind'] = 'in_scope' if row['expected'] != 'REVIEW' else 'multi_intent' if row['scenario'] == 'multi-intent' else 'out_of_scope'
 if any(row['kind'] not in ('in_scope','out_of_scope','multi_intent') for row in rows):raise ValueError('Unexpected kind')
 source_norm=set()
 with DATA.open(encoding='utf-8-sig',newline='') as f:
  source_norm={normalize(r['instruction']) for r in csv.DictReader(f)}
 norm=[normalize(r['text']) for r in rows]; dev_norm={normalize(r['text']) for r in json.loads(DEV.read_text(encoding='utf-8'))}
 if len(set(norm))!=len(norm) or set(norm)&dev_norm or set(norm)&source_norm: raise ValueError('Holdout overlap detected')
 models=[]
 for key,path in [('baseline',BASE),('candidate',CANDIDATE)]:
  art=joblib.load(path); evaluation=metrics(rows,art)
  models.append({'name':art['model_version'],'key':key,**model_identity(key,sel['selected']),**evaluation})
 report={'available':True,'dataset':{'description':'Independently authored synthetic support-ticket stress holdout','provenance':'assistant-authored synthetic; distinct author from dev and model experiments','count':len(rows),'sha256':sha(HOLD),'kind_counts':dict(Counter(r['kind'] for r in rows))},'selection':sel,'active_model':'baseline','selected_on_dev':sel['selected'],'models':models,'cases':active_cases(models),'limitations':['Synthetic authored examples, not customer requests; small convenience sample.', 'Category meanings inherit the Bitext dataset and can be ambiguous.', 'Scores are uncalibrated; unknown and multi-intent messages can be assigned a high score.', 'Thresholds and model choice were fixed on dev before this holdout; exact-match checks do not rule out semantic similarity.', 'Synthetic cases do not contain demographic attributes; fairness across demographic groups is not measured.', 'Dev model selection does not automatically activate the candidate; active inference remains baseline.', 'This test evaluates one text field. The live API classifies subject plus body, which is not separately evaluated here.', 'H059 tracking status is ambiguous: Bitext includes ORDER/track_order and DELIVERY/delivery_period. Original REVIEW label is retained. A post-hoc alternative ORDER analysis is in artifacts/stress_sensitivity.json; neither model selection nor holdout changed.']}
 REPORT.write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n',encoding='utf-8');print(json.dumps({m['key']:{k:m[k] for k in ('accuracy','macro_f1','coverage','unsafe_auto_count','should_review_recall')} for m in models},indent=2))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('phase',choices=['select','final']);a=p.parse_args();select() if a.phase=='select' else final()
