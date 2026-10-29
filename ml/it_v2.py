"""Isolated IT topic and proxy unknown experiment; historical test is not read."""
import argparse, json, hashlib, time
from pathlib import Path
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.linear_model import LogisticRegression
import joblib
from ml.it_experiment import read_jsonl, classification, build_network, partial_state, REVISION, PRETRAINED
from ml.evolution import UNRELATED
ROOT=Path(__file__).resolve().parents[1]; ART=ROOT/'artifacts'; DATA=ROOT/'data'
TRAIN=DATA/'it_support_prepared'/'train.jsonl'; DEV=DATA/'it_support_prepared'/'dev.jsonl'; CLINC=DATA/'evolution_clinc_full.json'
BASELINE=ART/'it_tfidf.joblib'; MODEL=ART/'it_v2_topic.joblib'; DETECTOR=ART/'it_v2_detector.joblib'; NEURAL=ART/'it_v2_neural.pt'; SELECTION=ART/'it_v2_selection.json'; REPORT=ART/'it_v2_report.json'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def verify_baseline(selection,path=BASELINE):
 expected=selection.get('hashes',{}).get('baseline')
 if not expected:raise ValueError('Baseline hash not frozen during selection')
 if not Path(path).is_file() or sha(path)!=expected:raise ValueError('Frozen baseline changed')

def ensure_fresh(*paths):
 old=[str(p) for p in paths if Path(p).exists()]
 if old:raise FileExistsError('Refusing to overwrite historical evidence: '+', '.join(old))
def group_folds(labels,groups,n_splits=3):
 return list(StratifiedGroupKFold(n_splits=n_splits,shuffle=True,random_state=812).split(np.zeros(len(labels)),labels,groups))
def decision_metrics(expected_unknown, flagged):
 y=np.asarray(expected_unknown,dtype=bool);f=np.asarray(flagged,dtype=bool)
 if len(y)!=len(f):raise ValueError('Length mismatch')
 known=~y;unk=y
 return {'known_n':int(known.sum()),'unknown_n':int(unk.sum()),'known_rejected':int((known&f).sum()),'unknown_detected':int((unk&f).sum()),'known_reject_rate':float((known&f).sum()/known.sum()) if known.sum() else None,'unknown_recall':float((unk&f).sum()/unk.sum()) if unk.sum() else None,'coverage':float((~f).mean()) if len(f) else None}
def prediction_summary(rows,predicted,flags,labels):
 if not (len(rows)==len(predicted)==len(flags)):raise ValueError('Length mismatch')
 known=[i for i,r in enumerate(rows) if r['expected'] in labels]
 unknown=[i for i,r in enumerate(rows) if r['expected']=='UNKNOWN']
 ambiguous=[i for i,r in enumerate(rows) if r['expected']=='AMBIGUOUS']
 if len(known)+len(unknown)+len(ambiguous)!=len(rows):raise ValueError('Unexpected expected category')
 yt=[rows[i]['expected'] for i in known];yp=[predicted[i] for i in known]
 return {'total':len(rows),'known_n':len(known),'unknown_n':len(unknown),'ambiguous_n':len(ambiguous),
   'known_accuracy':float(accuracy_score(yt,yp)) if known else None,
   'known_macro_f1':float(f1_score(yt,yp,labels=labels,average='macro',zero_division=0)) if known else None,
   'known_rejected':sum(bool(flags[i]) for i in known),
   'known_reject_rate':sum(bool(flags[i]) for i in known)/len(known) if known else None,
   'unknown_detected':sum(bool(flags[i]) for i in unknown),
   'unknown_recall':sum(bool(flags[i]) for i in unknown)/len(unknown) if unknown else None,
   'ambiguous_detected':sum(bool(flags[i]) for i in ambiguous),
   'ambiguous_recall':sum(bool(flags[i]) for i in ambiguous)/len(ambiguous) if ambiguous else None,
   'unsafe_count':sum(not bool(flags[i]) and predicted[i]!=rows[i]['expected'] for i in known)+sum(not bool(flags[i]) for i in unknown+ambiguous),
   'coverage':sum(not bool(x) for x in flags)/len(flags) if flags else None,
   'known_confusion_matrix':confusion_matrix(yt,yp,labels=labels).tolist() if known else [],'labels':labels}
def choose_ood_threshold(known_scores,unknown_scores,max_reject=.40):
 k=np.asarray(known_scores,float);u=np.asarray(unknown_scores,float)
 if not len(k) or not len(u):raise ValueError('Need known and unknown validation scores')
 valid=[]
 for t in sorted(set(np.concatenate([k,u,[0.,1.+1e-8]]))):
  m=decision_metrics([False]*len(k)+[True]*len(u),list(k>=t)+list(u>=t))
  if m['known_reject_rate']<=max_reject:valid.append({'threshold':float(t),**m})
 return max(valid,key=lambda x:(x['unknown_recall'],-x['known_reject_rate'],x['threshold']))
def linear_pipeline(kind):
 word=TfidfVectorizer(ngram_range=(1,2),min_df=2,sublinear_tf=True,max_features=70000)
 char=TfidfVectorizer(analyzer='char',ngram_range=(3,5),min_df=2,sublinear_tf=True,max_features=70000)
 C=.3 if kind=='regularized' else 2.0 if kind=='stronger' else 4.0 if kind=='strongest' else 1.;weight=None if kind=='unweighted' else 'balanced'
 return Pipeline([('features',FeatureUnion([('word',word),('char',char)])),('classifier',LogisticRegression(C=C,max_iter=500,class_weight=weight,random_state=42))])
def neural_scores(model,tok,texts,cap=192,batch=8):
 import torch
 encoded=tok(texts,padding='max_length',truncation=True,max_length=cap,return_tensors='pt');encoded={k:v for k,v in encoded.items() if k in ('input_ids','attention_mask','token_type_ids')};result=[];model.eval()
 with torch.no_grad():
  for i in range(0,len(texts),batch):result.extend(torch.softmax(model(**{k:v[i:i+batch] for k,v in encoded.items()}),dim=1).cpu().numpy())
 return np.asarray(result)
def fit_neural192(train,dev,labels):
 import torch
 torch.manual_seed(812);torch.set_num_threads(3);tokenizer,model=build_network(labels);index={name:i for i,name in enumerate(labels)}
 tokens=tokenizer([r['text'] for r in train],padding='max_length',truncation=True,max_length=192,return_tensors='pt');tokens={k:v for k,v in tokens.items() if k in ('input_ids','attention_mask','token_type_ids')};truth=[index[r['label']] for r in train]
 opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=4e-5,weight_decay=.01)
 counts=np.bincount(truth,minlength=len(labels));weights=torch.tensor(len(truth)/(len(labels)*np.maximum(counts,1)),dtype=torch.float32)
 loss_fn=torch.nn.CrossEntropyLoss(weight=weights);rng=np.random.default_rng(812);best=-1;history=[]
 for epoch in range(2):
  model.train();idx=rng.permutation(len(truth));t=time.time();batch=4
  for start in range(0,len(idx),batch):
   ix=idx[start:start+batch].tolist();inputs={k:v[ix] for k,v in tokens.items()};target=torch.tensor([truth[i] for i in ix]);opt.zero_grad(set_to_none=True);loss=loss_fn(model(**inputs),target);loss.backward();opt.step()
   if start%160==0:print(json.dumps({'event':'train','epoch':epoch+1,'done':start,'total':len(idx),'elapsed':round(time.time()-t,1)}),flush=True)
  probs=neural_scores(model,tokenizer,[r['text'] for r in dev]);pred=[labels[int(i)] for i in probs.argmax(axis=1)];m=classification(dev,pred,labels)
  history.append({'epoch':epoch+1,'accuracy':m['accuracy'],'macro_f1':m['macro_f1'],'seconds':round(time.time()-t,1)});print(json.dumps({'event':'neural_epoch',**history[-1]}),flush=True)
  if m['macro_f1']>best:
   best=m['macro_f1'];torch.save({'labels':labels,'state':partial_state(model),'epoch':epoch+1,'model':PRETRAINED,'revision':REVISION,'max_length':192},NEURAL)
 return history

def neural_status_for_selection():
 return {'status':'not_attempted_in_this_run','reason':'This selection runs only linear candidates; neural experiments are documented separately.'}

def select():
 ensure_fresh(MODEL,DETECTOR,SELECTION,REPORT)
 train=read_jsonl(TRAIN);dev=read_jsonl(DEV);labels=sorted(set(r['label'] for r in train))
 if set(r['group_id'] for r in train)&set(r['group_id'] for r in dev):raise ValueError('Train/dev group overlap')
 texts=[r['text'] for r in train];targets=np.array([r['label'] for r in train]);groups=np.array([r['group_id'] for r in train]);folds=group_folds(targets,groups);records={};pipes={}
 for kind in ('balanced','regularized','unweighted','stronger','strongest'):
  cv=[]
  for i,j in folds:
   pipe=linear_pipeline(kind);pipe.fit([texts[k] for k in i],targets[i]);pred=pipe.predict([texts[k] for k in j]);cv.append(float(f1_score(targets[j],pred,labels=labels,average='macro',zero_division=0)))
  pipe=linear_pipeline(kind);pipe.fit(texts,targets);pred=pipe.predict([r['text'] for r in dev]);m=classification(dev,pred.tolist(),labels)
  records[kind]={'cv_macro_f1':cv,'dev_accuracy':m['accuracy'],'dev_macro_f1':m['macro_f1']};pipes[kind]=pipe;print(json.dumps({'event':'linear_dev','kind':kind,**records[kind]}),flush=True)
 # MiniLM 192 was attempted, but stopped before training because system
 # available RAM dropped below 0.5 GiB. Do not report a nonexistent score.
 best=max(records,key=lambda n:(records[n]['dev_macro_f1'],n=='balanced'))
 linear_best=best;joblib.dump(pipes[linear_best],MODEL)
 known_conf=pipes[best].predict_proba([r['text'] for r in dev]).max(axis=1)
 clinc=json.loads(CLINC.read_text(encoding='utf-8'));norm=lambda x:' '.join(x.casefold().split())
 unknown_train=[x for x,intent in clinc['train'] if intent in UNRELATED];unknown_dev=[x for x,intent in clinc['val'] if intent in UNRELATED];known_norm={norm(r['text']) for r in train+dev}
 unknown_train=[x for x in unknown_train if norm(x) not in known_norm];unknown_dev=[x for x in unknown_dev if norm(x) not in known_norm]
 if {norm(x) for x in unknown_train}&{norm(x) for x in unknown_dev}:raise ValueError('CLINC train/dev exact overlap')
 detector=Pipeline([('tfidf',TfidfVectorizer(ngram_range=(1,2),min_df=2,sublinear_tf=True,max_features=60000)),('clf',LogisticRegression(max_iter=500,class_weight='balanced',random_state=812))]);detector.fit(texts+unknown_train,[0]*len(texts)+[1]*len(unknown_train));joblib.dump(detector,DETECTOR)
 unknown_conf=pipes[best].predict_proba(unknown_dev).max(axis=1)
 conf=choose_ood_threshold(1-known_conf,1-unknown_conf);binary=choose_ood_threshold(detector.predict_proba([r['text'] for r in dev])[:,1],detector.predict_proba(unknown_dev)[:,1]);policy=max([{'name':'confidence',**conf},{'name':'binary',**binary}],key=lambda x:(x['unknown_recall'],-x['known_reject_rate']))
 selection={'protocol':'Group CV within train, select model and threshold on dev; CLINC train and validation with 21 unrelated intents as proxy OOD; historical IT test not read.','chosen_model':best,'linear_candidate':linear_best,'classes':labels,'candidate_dev':records,'ood_policy':policy,'ood_policy_candidates':{'confidence':conf,'binary':binary},'ood_dev':{'known_n':len(dev),'unknown_n':len(unknown_dev),'clinc_train_n':len(unknown_train)},'neural_192':neural_status_for_selection(),'hashes':{'train':sha(TRAIN),'dev':sha(DEV),'clinc':sha(CLINC),'topic':sha(MODEL),'detector':sha(DETECTOR),'baseline':sha(BASELINE)},'limitations':['CLINC unrelated intents are limited proxy OOD, not verified real IT out-of-scope.','Maximum 40% known rejection on dev is an experimental gate, not production guarantee.','Old IT test previously viewed and not used for selection.','Demographic fairness not evaluated.']}
 SELECTION.write_text(json.dumps(selection,indent=2,ensure_ascii=False)+'\n',encoding='utf-8');print(json.dumps({'event':'selection','chosen_model':best,'policy':policy}),flush=True)
def final():
 ensure_fresh(REPORT)
 blind=DATA/'it_v2_blind.jsonl';expected_blind='01ce1d6aa441b25abc282c61ac51d95d0fc9d3be85403249de14741d7c2c4a92'
 selection=json.loads(SELECTION.read_text(encoding='utf8'))
 verify_baseline(selection)
 for key,path in [('train',TRAIN),('dev',DEV),('clinc',CLINC),('topic',MODEL),('detector',DETECTOR)]:
  if selection['hashes'][key]!=sha(path):raise ValueError('Frozen input changed: '+key)
 if sha(blind)!=expected_blind:raise ValueError('Frozen blind sample checksum changed')
 rows=read_jsonl(blind);labels=selection['classes'];texts=[r['text'] for r in rows]
 norm=lambda x:' '.join(x.casefold().split())
 if len(set(r['id'] for r in rows))!=len(rows) or len(set(map(norm,texts)))!=len(rows):raise ValueError('Duplicate blind entries')
 train=read_jsonl(TRAIN);dev=read_jsonl(DEV);clinc=json.loads(CLINC.read_text(encoding='utf8'))
 source={norm(r['text']) for r in train+dev}|{norm(x) for split in ('train','val') for x,intent in clinc[split] if intent in UNRELATED}
 if any(norm(t) in source for t in texts):raise ValueError('Blind exact overlap with training/development')
 topic=joblib.load(MODEL);detector=joblib.load(DETECTOR);baseline=joblib.load(BASELINE)
 # The active separate IT mode deliberately flags all requests for review.
 baseline_pred=baseline.predict(texts).tolist(); baseline_flags=[True]*len(texts)
 pred=topic.predict(texts).tolist();policy=selection['ood_policy']
 scores=detector.predict_proba(texts)[:,1] if policy['name']=='binary' else 1-topic.predict_proba(texts).max(axis=1)
 flags=(scores>=policy['threshold']).tolist()
 b=prediction_summary(rows,baseline_pred,baseline_flags,labels);c=prediction_summary(rows,pred,flags,labels)
 external=DATA/'it_v2_external'/'sample_unlabelled_private.jsonl'; manifest=DATA/'it_v2_external'/'sample_manifest.json';unlabelled=None
 if external.exists() and manifest.exists():
  meta=json.loads(manifest.read_text(encoding='utf8'))
  if sha(external)!=meta['sample_sha256']:raise ValueError('External sample checksum changed')
  ext=[r for r in read_jsonl(external) if r['split']=='test'];ext_text=[r['text'] for r in ext]
  ext_scores=detector.predict_proba(ext_text)[:,1] if policy['name']=='binary' else 1-topic.predict_proba(ext_text).max(axis=1)
  ext_flags=ext_scores>=policy['threshold']
  unlabelled={'count':len(ext),'review_flagged':int(ext_flags.sum()),'flag_rate':float(ext_flags.mean()) if len(ext) else None,'baseline_review_flagged':len(ext),'source':meta['source_url'],'source_revision':meta['revision'],'warning':'Unlabelled external sample: flag frequency is not OOD recall, precision, or accuracy. Source labels are not mapped to this task; masking is not independently verified.'}
 b.update({'name':'Current IT TF-IDF; all review','kind':'baseline'});c.update({'name':'Candidate IT TF-IDF + proxy OOD detector','kind':'candidate'})
 report={'available':True,'dataset':{'name':'Blind independently authored AI-synthetic IT/OOD diagnostic','provenance':'AI-authored synthetic; single annotator, taxonomy ambiguities remain; not real customer tickets','known_dev_n':selection['ood_dev']['known_n'],'unknown_dev_n':selection['ood_dev']['unknown_n'],'external_known_n':c['known_n'],'external_unknown_n':c['unknown_n'],'external_ambiguous_n':c['ambiguous_n'],'blind_sha256':expected_blind},'baseline':b,'candidate':c,'selection':{'chosen_model':selection['chosen_model'],'ood_policy':selection['ood_policy']['name'],'threshold':selection['ood_policy']['threshold'],'sha256':sha(SELECTION),'development':selection['ood_dev'],'neural_192':selection['neural_192']},'development':{'models':selection['candidate_dev'],'ood_policy':selection['ood_policy'],'candidates':selection['ood_policy_candidates']},'unlabelled_external':unlabelled,'active':False,'limitations':selection['limitations']+['Blind sample created by one AI author, not an independent real-user benchmark.','External Mendeley sample is unlabelled for the target taxonomy; flag count alone is not evidence of quality.','Training source is one organisation, duplicates/near-duplicates and future shifts remain.','Previous IT test was already viewed and was not reused for tuning.']}
 REPORT.write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n',encoding='utf8')
 print(json.dumps({'event':'final','baseline':b,'candidate':c,'unlabelled_external':unlabelled},ensure_ascii=False),flush=True)

if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('phase',choices=['select','final']);args=parser.parse_args();globals()[args.phase]()
