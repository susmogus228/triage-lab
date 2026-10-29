"""Secondary diagnostic on previously inspected source test; not a fresh test."""
import json,hashlib
from pathlib import Path
import joblib
from ml.it_v2 import ROOT,ART,DATA,MODEL,DETECTOR,SELECTION,REPORT,sha,ensure_fresh,read_jsonl
OUTPUT=ART/'it_v2_historical_diagnostic.json';TEST=DATA/'it_support_prepared'/'test.jsonl';MANIFEST=ART/'it_data_manifest.json'
def main():
 ensure_fresh(OUTPUT)
 select=json.loads(SELECTION.read_text(encoding='utf8'));rep=json.loads(REPORT.read_text(encoding='utf8'));manifest=json.loads(MANIFEST.read_text(encoding='utf8'))
 if sha(SELECTION)!=rep['selection']['sha256'] or sha(TEST)!=manifest['output_sha256']['test']:raise ValueError('Frozen input changed')
 if sha(MODEL)!=select['hashes']['topic'] or sha(DETECTOR)!=select['hashes']['detector']:raise ValueError('Model artifact changed')
 rows=read_jsonl(TEST);model=joblib.load(MODEL);det=joblib.load(DETECTOR);texts=[r['text'] for r in rows]
 pred=model.predict(texts).tolist();policy=select['ood_policy'];scores=det.predict_proba(texts)[:,1] if policy['name']=='binary' else 1-model.predict_proba(texts).max(axis=1);flags=(scores>=policy['threshold']).tolist()
 def agg(indices):
  exact={}
  for i in indices:
   key=hashlib.sha256(' '.join(texts[i].casefold().split()).encode()).hexdigest();exact.setdefault(key,[]).append(i)
  return {'rows':len(indices),'flagged_rows':sum(flags[i] for i in indices),'exact_unique':len(exact),'flagged_exact_unique':sum(any(flags[i] for i in group) for group in exact.values())}
 labels=set(select['classes']);known=[i for i,r in enumerate(rows) if r['label'] in labels];eol=[i for i,r in enumerate(rows) if r['label']=='EOL'];other=[i for i,r in enumerate(rows) if r['label'] not in labels and r['label']!='EOL']
 if other:raise ValueError('Unexpected unseen label')
 report={'description':'Secondary diagnostic using original IT source test, previously inspected and transductively used for deduplication. No threshold/model tuning or independence claim. EOL has only two unique texts, so row counts are not independent unknown cases.', 'known':agg(known),'eol':agg(eol),'known_topic_correct_rows':sum(pred[i]==rows[i]['label'] for i in known),'selection_sha256':sha(SELECTION),'historical_test_sha256':sha(TEST),'limitations':['EOL unseen after removing overlap with test; two unique texts are not representative unknown requests.','Previously inspected, not an untouched evaluation.','Test texts were used without labels during grouping.','No organization/author independence or OOD production validity established.']}
 OUTPUT.write_text(json.dumps(report,indent=2)+'\n',encoding='utf8');print(json.dumps(report),flush=True)
if __name__=='__main__':main()
