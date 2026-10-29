"""Frozen model comparison on de-identified IT support tickets.

All input text remains local. Development guides model selection; final test
is only read after the selection and all artifacts are frozen.
"""
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'data'/'it_support_prepared'
ART=ROOT/'artifacts'

def sha256(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda:handle.read(1024*1024),b''):digest.update(chunk)
    return digest.hexdigest()
def verify_groups(a,b):
    left={x['group_id'] for x in a}; right={x['group_id'] for x in b}
    if left & right: raise ValueError('Source groups overlap')
def select_model(metrics):
    # deterministic tie-break prefers simpler, lighter baseline
    return max(metrics, key=lambda name:(metrics[name]['macro_f1'],name=='tfidf'))
def verify_frozen(expected,paths):
    for key,path in paths.items():
        if sha256(path)!=expected[key]: raise ValueError('Frozen input changed: '+key)

import argparse, csv, time, os
import joblib
import numpy as np
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

REVISION='e8f8c211226b894fcb81acc59f3b34ba3efd5f42'
PRETRAINED='sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
TRAIN=DATA/'train.jsonl'; DEV=DATA/'dev.jsonl'; TEST=DATA/'test.jsonl'
BASE=ART/'it_tfidf.joblib'; DRAFT=ART/'it_baseline_draft.json'; NEURAL=ART/'it_neural.pt'; REFINED=ART/'it_neural_refined.pt'; SELECTION=ART/'it_selection.json'; SELECTION_V2=ART/'it_selection_v2.json'; SEAL=ART/'it_final_seal.json'; DATA_MANIFEST=ART/'it_data_manifest.json'; REPORT=ART/'it_experiment_report.json'
HFHOME=ROOT/'work'/'it_huggingface_cache'
PRETRAINED_DIR=ROOT/'work'/'it_pretrained'
PRETRAINED_SHA256='eaa086f0ffee582aeb45b36e34cdd1fe2d6de2bef61f8a559a1bbc9bd955917b'
os.environ.setdefault('HF_HOME',str(HFHOME))
os.environ.setdefault('HF_HUB_DISABLE_TELEMETRY','1')

def read_jsonl(p):
    return [json.loads(line) for line in p.read_text(encoding='utf-8-sig').splitlines() if line.strip()]
def require_new(*paths):
    exists=[str(p) for p in paths if p.exists()]
    if exists: raise FileExistsError('Evidence already exists; refusing to overwrite: '+', '.join(exists))
def classification(rows,predictions,labels):
    truth=[r['label'] for r in rows]
    if len(truth)!=len(predictions):raise ValueError('Prediction length mismatch')
    report=classification_report(truth,predictions,labels=labels,output_dict=True,zero_division=0)
    groups={}
    exact={}
    for row,pred in zip(rows,predictions):
        groups.setdefault(row['group_id'],[]).append((row['label'],pred))
        key=hashlib.sha256(' '.join(row['text'].casefold().split()).encode('utf-8')).hexdigest()
        exact.setdefault(key,[]).append((row['label'],pred))
    group_truth=[];group_pred=[]; ambiguous_groups=0
    from collections import Counter
    for group in groups.values():
        true_set={v[0] for v in group}
        if len(true_set)!=1: ambiguous_groups+=1;continue
        group_truth.append(group[0][0]); group_pred.append(Counter(v[1] for v in group).most_common(1)[0][0])
    exact_truth=[];exact_pred=[];exact_conflicts=0
    for group in exact.values():
        if len({v[0] for v in group})!=1: exact_conflicts+=1;continue
        exact_truth.append(group[0][0]);exact_pred.append(Counter(v[1] for v in group).most_common(1)[0][0])
    return {'exact_unique_groups':len(exact_truth),'exact_accuracy':float(accuracy_score(exact_truth,exact_pred)) if exact_truth else None,'exact_macro_f1':float(f1_score(exact_truth,exact_pred,labels=labels,average='macro',zero_division=0)) if exact_truth else None,'exact_conflicting_groups':exact_conflicts, 'accuracy':float(accuracy_score(truth,predictions)), 'macro_f1':float(f1_score(truth,predictions,labels=labels,average='macro',zero_division=0)),
            'labels':labels,'confusion_matrix':confusion_matrix(truth,predictions,labels=labels).tolist(),
            'per_class':{k:report[k] for k in labels},'rows':len(rows),'unique_groups':len(group_truth),
            'group_accuracy':float(accuracy_score(group_truth,group_pred)) if group_truth else None,
            'group_macro_f1':float(f1_score(group_truth,group_pred,labels=labels,average='macro',zero_division=0)) if group_truth else None,
            'excluded_conflicting_groups':ambiguous_groups}

def prepare_tfidf(train,dev,labels):
    # Both vectorizers are fitted on training data only.
    pipe=Pipeline([('features',FeatureUnion([('word',TfidfVectorizer(ngram_range=(1,2),min_df=2,sublinear_tf=True,max_features=70000)),('char',TfidfVectorizer(analyzer='char',ngram_range=(3,5),min_df=2,sublinear_tf=True,max_features=70000))])),('classifier',LogisticRegression(max_iter=500,class_weight='balanced',random_state=42))])
    pipe.fit([r['text'] for r in train],[r['label'] for r in train])
    devpred=pipe.predict([r['text'] for r in dev]).tolist()
    joblib.dump(pipe,BASE)
    return classification(dev,devpred,labels)

def imports():
    import torch
    from transformers import AutoTokenizer, AutoModel
    return torch,AutoTokenizer,AutoModel

def verify_pretrained_assets():
    expected={'config.json':('git','c06d5b49495f044e6380e68a60538be17a6bd5d1'),'tokenizer_config.json':('git','3c1b565ae10a15a1d0c31096f834af2fd9359e91'),'special_tokens_map.json':('git','2ea7ad0e45a9d1d1591782ba7e29a703d0758831'),'tokenizer.json':('sha256','2c3387be76557bd40970cec13153b3bbf80407865484b209e655e5e4729076b8')}
    for name,(kind,digest) in expected.items():
        data=(PRETRAINED_DIR/name).read_bytes()
        actual=hashlib.sha1(f'blob {len(data)}\0'.encode()+data).hexdigest() if kind=='git' else hashlib.sha256(data).hexdigest()
        if actual != digest: raise ValueError('Pretrained asset checksum mismatch: '+name)

def build_network(labels,local_files_only=False):
    torch,AutoTokenizer,AutoModel=imports()
    if sha256(PRETRAINED_DIR/'model.safetensors') != PRETRAINED_SHA256: raise ValueError('Pretrained weights checksum mismatch')
    verify_pretrained_assets()
    tok=AutoTokenizer.from_pretrained(PRETRAINED_DIR,local_files_only=True)
    base=AutoModel.from_pretrained(PRETRAINED_DIR,local_files_only=True)
    for p in base.parameters(): p.requires_grad=False
    for p in base.encoder.layer[-1].parameters(): p.requires_grad=True
    class TicketNet(torch.nn.Module):
        def __init__(self,encoder,n):
            super().__init__();self.encoder=encoder;self.classifier=torch.nn.Linear(encoder.config.hidden_size,n)
        def forward(self,input_ids,attention_mask,token_type_ids=None):
            h=self.encoder(input_ids=input_ids,attention_mask=attention_mask,token_type_ids=token_type_ids).last_hidden_state
            mask=attention_mask.unsqueeze(-1).to(h.dtype)
            pooled=(h*mask).sum(dim=1)/mask.sum(dim=1).clamp(min=1)
            return self.classifier(pooled)
    return tok,TicketNet(base,len(labels))

def tensorize(rows,tokenizer,labels):
    index={name:i for i,name in enumerate(labels)}
    # Reduces memory and compute, with token truncation documented.
    tokens=tokenizer([r['text'] for r in rows],padding='max_length',truncation=True,max_length=96,return_tensors='pt')
    tokens={k:v for k,v in tokens.items() if k in ('input_ids','attention_mask','token_type_ids')}
    return tokens,[index[r['label']] for r in rows]

def neural_predict(model,tokens,labels,batch=16):
    torch,_,_=imports();model.eval();out=[]
    with torch.no_grad():
        for start in range(0,len(tokens['input_ids']),batch):
            args={k:v[start:start+batch] for k,v in tokens.items()}
            out.extend(model(**args).argmax(dim=1).tolist())
    return [labels[i] for i in out]

def partial_state(model):
    # Save only trained last encoder layer and classification head; trusted pinned pretrained weights loaded separately.
    return {k:v.detach().cpu() for k,v in model.state_dict().items() if k.startswith('encoder.encoder.layer.11.') or k.startswith('classifier.')}

def fit_neural(train,dev,labels, resume=False, output=None, epochs=2):
    torch,_,_=imports();torch.manual_seed(42);np.random.seed(42);torch.set_num_threads(3)
    output=output or NEURAL
    tokenizer,model=load_neural(labels) if resume else build_network(labels);training,truth=tensorize(train,tokenizer,labels);validation,_=tensorize(dev,tokenizer,labels)
    trainable=[p for p in model.parameters() if p.requires_grad]
    optimizer=torch.optim.AdamW(trainable,lr=4e-5,weight_decay=0.01)
    counts=np.bincount(truth,minlength=len(labels));weights=torch.tensor(len(truth)/(len(labels)*np.maximum(counts,1)),dtype=torch.float32)
    criterion=torch.nn.CrossEntropyLoss(weight=weights)
    best=-1;best_metrics=None;history=[];rng=np.random.default_rng(42);started=time.time()
    for epoch in range(epochs):
        model.train();indices=rng.permutation(len(truth)); epoch_start=time.time();batch=8;running=0.
        for start in range(0,len(indices),batch):
            ix=indices[start:start+batch].tolist();args={k:v[ix] for k,v in training.items()};target=torch.tensor([truth[i] for i in ix])
            optimizer.zero_grad(set_to_none=True);loss=criterion(model(**args),target);loss.backward();optimizer.step();running+=float(loss.detach())*len(ix)
            if start == 0 or start % (batch*10) == 0:
                elapsed=time.time()-epoch_start;done=min(start+batch,len(indices));
                print(json.dumps({'event':'training_progress','epoch':epoch+1,'done':done,'total':len(indices),'elapsed_seconds':round(elapsed,2),'eta_epoch_seconds':round(elapsed*(len(indices)-done)/done,2)}),flush=True)
        pred=neural_predict(model,validation,labels); m=classification(dev,pred,labels)
        hist={'epoch':epoch+1,'loss':running/len(truth),'dev_accuracy':m['accuracy'],'dev_macro_f1':m['macro_f1'],'elapsed_seconds':round(time.time()-epoch_start,2)};history.append(hist);print(json.dumps(hist),flush=True)
        if m['macro_f1']>best:
            best=m['macro_f1'];best_metrics=m
            torch.save({'labels':labels,'state':partial_state(model),'epoch':epoch+1,'model':PRETRAINED,'revision':REVISION,'max_length':96},output)
    return best_metrics,history

def load_neural(labels,path=None):
    torch,_,_=imports();torch.set_num_threads(3)
    checkpoint=torch.load(path or NEURAL,map_location='cpu',weights_only=True)
    if checkpoint['labels']!=labels or checkpoint['revision']!=REVISION:raise ValueError('Neural model contract mismatch')
    tokenizer,model=build_network(labels,local_files_only=True)
    missing,unexpected=model.load_state_dict(checkpoint['state'],strict=False)
    if unexpected or not checkpoint['state']:raise ValueError('Invalid partial model checkpoint')
    return tokenizer,model

def baseline_phase():
    require_new(BASE,DRAFT)
    train,dev=read_jsonl(TRAIN),read_jsonl(DEV);verify_groups(train,dev)
    labels=sorted({r['label'] for r in train})
    result=prepare_tfidf(train,dev,labels)
    draft={'train_sha256':sha256(TRAIN),'dev_sha256':sha256(DEV),'model_sha256':sha256(BASE),'labels':labels,'dev':result}
    DRAFT.write_text(json.dumps(draft,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'event':'baseline_dev','accuracy':result['accuracy'],'macro_f1':result['macro_f1'],'rows':len(dev)}),flush=True)

def select_phase():
    require_new(REPORT,NEURAL,SELECTION)
    train,dev=read_jsonl(TRAIN),read_jsonl(DEV)
    verify_groups(train,dev)
    labels=sorted({r['label'] for r in train})
    if not labels or any(r['label'] not in labels for r in dev):raise ValueError('Unexpected development label')
    print(json.dumps({'event':'data','train_rows':len(train),'dev_rows':len(dev),'labels':labels}),flush=True)
    if not BASE.exists() and not DRAFT.exists(): baseline_phase()
    if not BASE.exists() or not DRAFT.exists(): raise ValueError('Baseline draft incomplete')
    draft=json.loads(DRAFT.read_text(encoding='utf-8'))
    if draft['train_sha256']!=sha256(TRAIN) or draft['dev_sha256']!=sha256(DEV) or draft['model_sha256']!=sha256(BASE) or draft['labels']!=labels: raise ValueError('Baseline draft does not match inputs')
    baseline=draft['dev']
    neural,history=fit_neural(train,dev,labels)
    selected=select_model({'tfidf':baseline,'multilingual':neural})
    selection={'created_utc':__import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat(),
      'protocol':'Train and dev only; test never opened for model selection; select dev macro F1 with tie to tfidf. 6 trained labels, missing label treated as unseen.',
      'selected':selected,'labels':labels,'models':{'tfidf':{'name':'TF-IDF word + character logistic regression','kind':'tfidf','dev':baseline},'multilingual':{'name':'Multilingual MiniLM, trainable last encoder layer + classifier','kind':'partial_finetune','dev':neural,'history':history}},
      'pretrained':{'model':PRETRAINED,'revision':REVISION,'license':'Apache-2.0','max_length':96,'training':'Only last transformer layer and classifier trainable; not full encoder finetune'},
      'sha256':{'train':sha256(TRAIN),'dev':sha256(DEV),'tfidf':sha256(BASE),'neural':sha256(NEURAL),'pretrained_weights':sha256(PRETRAINED_DIR/'model.safetensors')}}
    SELECTION.write_text(json.dumps(selection,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');print(json.dumps({'event':'selection','selected':selected,'baseline_dev':baseline['macro_f1'],'neural_dev':neural['macro_f1']}),flush=True)

def refine_phase():
    require_new(REPORT,REFINED,SELECTION_V2)
    first=json.loads(SELECTION.read_text(encoding='utf-8'))
    verify_frozen(first['sha256'],{'train':TRAIN,'dev':DEV,'tfidf':BASE,'neural':NEURAL,'pretrained_weights':PRETRAINED_DIR/'model.safetensors'})
    train,dev=read_jsonl(TRAIN),read_jsonl(DEV);verify_groups(train,dev);labels=first['labels']
    refined,history=fit_neural(train,dev,labels,resume=True,output=REFINED,epochs=4)
    old=first['models']['multilingual']['dev']
    if refined['macro_f1']>old['macro_f1']:
        neural_path=REFINED; neural_dev=refined; choice='refined'
    else:
        neural_path=NEURAL;neural_dev=old;choice='initial'
    selected=select_model({'tfidf':first['models']['tfidf']['dev'],'multilingual':neural_dev})
    first['selected']=selected
    first['protocol']+=' Dev-only additional 4 epochs from initial best checkpoint; optimizer restarted. Final test not read; original selection preserved in separate it_selection.json.'
    first['models']['multilingual']['dev']=neural_dev
    first['models']['multilingual']['refinement_history']=history
    first['models']['multilingual']['checkpoint']=choice
    first['sha256']['neural']=sha256(neural_path)
    first['neural_checkpoint']=neural_path.name
    first['initial_selection_sha256']=sha256(SELECTION)
    SELECTION_V2.write_text(json.dumps(first,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'event':'refined_selection','selected':selected,'neural_checkpoint':choice,'neural_dev':neural_dev['macro_f1'],'baseline_dev':first['models']['tfidf']['dev']['macro_f1']}),flush=True)

def seal_phase():
    # Seal all artifacts after the dev-only selection, before opening test records.
    require_new(REPORT,SEAL)
    selection_path=SELECTION_V2 if SELECTION_V2.exists() else SELECTION
    selection=json.loads(selection_path.read_text(encoding='utf-8'))
    neural_path=ART/selection.get('neural_checkpoint',NEURAL.name)
    verify_frozen(selection['sha256'],{'train':TRAIN,'dev':DEV,'tfidf':BASE,'neural':neural_path,'pretrained_weights':PRETRAINED_DIR/'model.safetensors'})
    manifest=json.loads(DATA_MANIFEST.read_text(encoding='utf-8'))
    expected=manifest['output_sha256']['test'];actual=sha256(TEST)
    if actual!=expected:raise ValueError('Test file differs from original data manifest')
    seal={'selection_file':selection_path.name,'selection_sha256':sha256(selection_path),'test_sha256':actual,'data_manifest_sha256':sha256(DATA_MANIFEST),'protocol':'Hashes fixed after dev selection and before test text/labels read for evaluation. Test text was used without labels for near-duplicate removal when preparing data.'}
    SEAL.write_text(json.dumps(seal,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'event':'sealed','selection':selection_path.name,'test_sha256':actual}),flush=True)

def final_phase():
    require_new(REPORT)
    seal=json.loads(SEAL.read_text(encoding='utf-8'))
    selection_path=SELECTION_V2 if SELECTION_V2.exists() else SELECTION
    if seal['selection_file']!=selection_path.name or seal['selection_sha256']!=sha256(selection_path) or seal['test_sha256']!=sha256(TEST) or seal['data_manifest_sha256']!=sha256(DATA_MANIFEST):raise ValueError('Final seal mismatch')
    selection=json.loads(selection_path.read_text(encoding='utf-8'))
    neural_path=ART/selection.get('neural_checkpoint',NEURAL.name)
    verify_frozen(selection['sha256'],{'train':TRAIN,'dev':DEV,'tfidf':BASE,'neural':neural_path,'pretrained_weights':PRETRAINED_DIR/'model.safetensors'})
    train,dev,test=read_jsonl(TRAIN),read_jsonl(DEV),read_jsonl(TEST)
    verify_groups(train,dev);verify_groups(train,test);verify_groups(dev,test)
    labels=selection['labels'];all_labels=sorted(set(labels)|{r['label'] for r in test});known=[r for r in test if r['label'] in labels]; unseen=[r for r in test if r['label'] not in labels]
    if not known:raise ValueError('No known test examples')
    baseline=joblib.load(BASE);bp=baseline.predict([r['text'] for r in test]).tolist()
    tok,net=load_neural(labels,neural_path); truncation_counts={name:sum(len(x)>96 for x in tok([r['text'] for r in rows],truncation=False)['input_ids']) for name,rows in [('train',train),('dev',dev),('test',test)]};tokens,_=tensorize([{**r,'label':r['label'] if r['label'] in labels else labels[0]} for r in test],tok,labels);npred=neural_predict(net,tokens,labels)
    models=[]
    for key,pred in [('tfidf',bp),('multilingual',npred)]:
        known_indices=[i for i,r in enumerate(test) if r['label'] in labels]
        all_m=classification(test,pred,all_labels);known_m=classification(known,[pred[i] for i in known_indices],labels)
        models.append({**selection['models'][key],'key':key,'test':all_m,'known_test':known_m,'selected':selection['selected']==key,'active':False})
    report={'available':True,'selection_sha256':seal['selection_sha256'],'dataset':{'name':'Classification of IT Support Tickets (Zenodo 7648117)','source':'https://zenodo.org/records/7648117','language':'multilingual; no language-specific performance verified','splits':{'train':len(train),'dev':len(dev),'test':len(test),'test_known':len(known),'test_known_exact_unique_groups':len({hashlib.sha256(' '.join(r['text'].casefold().split()).encode('utf-8')).hexdigest() for r in known}),'test_unseen':len(unseen),'test_unseen_unique':len({hashlib.sha256(' '.join(r['text'].casefold().split()).encode('utf-8')).hexdigest() for r in unseen})},'test_sha256':sha256(TEST),'truncated_over_96_tokens':truncation_counts,'limitations':['Original de-identified tickets; exact and near duplicates affect representativeness. Unlabelled test text was used transductively to identify and remove overlapping near-duplicates before model training.','One EOL class absent in training; no model can predict it as a known label.','No demographic labels; fairness cannot be evaluated.','The published labels represent the provider taxonomy, not universal support routing.']},'models':models,'selected':selection['selected'],'pretrained':selection['pretrained'],'limitations':['This is a retrospective dataset from one IT support organisation; generalisation to another domain unknown.','Closed-set predictions do not reliably identify unseen categories; rejection not validated.','Test includes an unseen category and is reported both in full and on known labels.','Input text and error examples are intentionally excluded to limit privacy exposure.']}
    REPORT.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');print(json.dumps({'event':'test','selected':report['selected'],'scores':{x['key']:{'all_accuracy':x['test']['accuracy'],'known_accuracy':x['known_test']['accuracy'],'known_macro_f1':x['known_test']['macro_f1'],'near_group_macro_f1':x['known_test']['group_macro_f1']} for x in models}},indent=2),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=['baseline','select','refine','seal','final']);args=parser.parse_args()
    {'baseline':baseline_phase,'select':select_phase,'refine':refine_phase,'seal':seal_phase,'final':final_phase}[args.phase]()
