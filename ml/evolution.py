"""Local generations of a review detector, with frozen selection before holdout.

Training composites are synthetic distinct-Bitext-category examples, not true
semantic multi-intent labels; same-category requests can contain several intents.
Never run select or retrain in response to an inspected final holdout.
"""
import argparse, csv, hashlib, json, random
from collections import Counter, defaultdict
from datetime import datetime,timezone
from pathlib import Path
import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from ml.training import clean_rows,split_indices,normalize,has_known_content

ROOT=Path(__file__).resolve().parents[1]; ART=ROOT/'artifacts'; DATA=ROOT/'data'
BITEXT=DATA/'bitext.csv'; CLINC=DATA/'evolution_clinc_full.json'
FIT=DATA/'evolution_fit.jsonl'; FEEDBACK=DATA/'evolution_feedback.jsonl'
DEV=DATA/'evolution_dev.jsonl'; HOLD=DATA/'evolution_holdout.jsonl'
BASE=ART/'ticket_classifier.joblib'; G1=ART/'evolution_g1.joblib'; G2=ART/'evolution_g2.joblib'
SELECTION=ART/'evolution_selection.json'; REPORT=ART/'evolution_report.json'
# Labels below are reviewed against the project's category taxonomy, but still
# allow ambiguous utterances and are not human-verified in this specific use.
UNRELATED={'recipe','weather','calories','cook_time','ingredient_substitution','nutrition_info','roll_dice','flip_coin','tell_joke','meaning_of_life','tire_pressure','oil_change_how','gas_type','travel_suggestion','traffic','vaccines','next_song','what_song','play_music','timezone','next_holiday'}
CONNECTORS=[' and ','. Also, ','; another thing: ','\n','; ','. Next: ','. One more request: ',' plus ',' / ','. Additionally, ']
SEED=97831

def require_new_outputs(paths):
    existing=[str(path) for path in paths if path.exists()]
    if existing:
        raise FileExistsError('Historical evidence already exists; use a fresh project copy for a new run: '+', '.join(existing))

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def norm_hash(t): return hashlib.sha256(normalize(t).encode('utf-8')).hexdigest()
def normalize_expected(label): return 'REVIEW' if label in ('OUT_OF_SCOPE','MULTI') else label
def combine_examples(a,b,connector): return a.strip()+connector+b.strip()
def read_jsonl(path): return [json.loads(x) for x in path.read_text(encoding='utf-8-sig').splitlines() if x.strip()]
def write_jsonl(path,rows): path.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows),encoding='utf-8')
def assert_disjoint(partitions):
    used={}
    for name,rows in partitions.items():
        ids={x for r in rows for x in r['origin_ids']}
        for prior,prev in used.items():
            if ids & prev: raise ValueError(f'Overlapping origins {prior}/{name}')
        used[name]=ids

def metrics(cases):
    known=[r for r in cases if r['expected']!='REVIEW']; review=[r for r in cases if r['expected']=='REVIEW']
    notflagged=[r for r in cases if not r['needs_review']]
    unsafe=[r for r in notflagged if r['expected']=='REVIEW' or r['predicted']!=r['expected']]
    return {'count':len(cases),'known_n':len(known),'known_correct':sum(r['predicted']==r['expected'] for r in known),
        'known_accuracy':sum(r['predicted']==r['expected'] for r in known)/len(known) if known else None,
        'review_n':len(review),'review_found':sum(r['needs_review'] for r in review),
        'review_recall':sum(r['needs_review'] for r in review)/len(review) if review else None,
        'unflagged_n':len(notflagged),'coverage':len(notflagged)/len(cases),
        'unsafe_count':len(unsafe),'known_flagged_n':sum(r['needs_review'] for r in known),
        'scenario_counts':{s:{'total':len(sub),'flagged':sum(x['needs_review'] for x in sub)} for s in sorted({r.get('scenario','unknown') for r in review}) for sub in [[r for r in review if r.get('scenario','unknown')==s]]}}

def choose_candidate(base,candidates):
    # Do not silently claim success by routing nearly everyone to manual review.
    eligible=[c for c in candidates if c['unsafe_count'] <= base['unsafe_count']-2 and c['coverage'] >= max(.25,base['coverage']-.10)]
    if not eligible:return {'name':'baseline',**base}
    return min(eligible,key=lambda c:(c['unsafe_count'],-c['coverage']))

def make_set(sources,n_pairs,rng,name):
    by=defaultdict(list)
    for item in sources: by[item['category']].append(item)
    cats=sorted(by); rows=[]
    # Single source and two examples within same *broad* category are controls.
    for _ in range(n_pairs):
        a=rng.choice(sources); rows.append({'text':a['text'],'review':0,'kind':'single','origin_ids':[a['id']]})
        category=rng.choice(cats); a,b=rng.sample(by[category],2)
        rows.append({'text':combine_examples(a['text'],b['text'],rng.choice(CONNECTORS)),'review':0,'kind':'same_category_composite','origin_ids':[a['id'],b['id']]})
        c1,c2=rng.sample(cats,2); a,b=rng.choice(by[c1]),rng.choice(by[c2])
        rows.append({'text':combine_examples(a['text'],b['text'],rng.choice(CONNECTORS)),'review':1,'kind':'distinct_category_composite','origin_ids':[a['id'],b['id']]})
        a,b=rng.choice(by[c2]),rng.choice(by[c1]); rows.append({'text':combine_examples(a['text'],b['text'],rng.choice(CONNECTORS)),'review':1,'kind':'distinct_category_composite','origin_ids':[a['id'],b['id']]})
    return rows

def prepare():
    require_new_outputs([FIT, FEEDBACK, DEV, ART/'evolution_data_manifest.json'])
    rng=random.Random(SEED)
    with BITEXT.open(encoding='utf-8-sig',newline='') as f: raw=[(r['instruction'],r['category']) for r in csv.DictReader(f)]
    clean,_=clean_rows(raw); labels=[r[1] for r in clean]; train,_,_=split_indices(labels,[normalize(r[0]) for r in clean])
    groups=defaultdict(list)
    for i in train:
        text,cat=clean[int(i)];groups[cat].append({'text':text,'category':cat,'id':'bitext:'+norm_hash(text)})
    fit_src=[];feedback_src=[]
    for cat in sorted(groups):
        values=groups[cat];rng.shuffle(values); n=int(.8*len(values));fit_src+=values[:n];feedback_src+=values[n:]
    fit=make_set(fit_src,9000,rng,'fit'); feedback=make_set(feedback_src,2000,rng,'feedback')
    clinc=json.loads(CLINC.read_text(encoding='utf-8'))
    examples=[{'text':t,'review':1,'kind':'clinc_unrelated','origin_ids':['clinc:'+norm_hash(t)],'clinc_intent':intent} for t,intent in clinc['train'] if intent in UNRELATED]
    # Source-level grouping: identical utterances never split across training phases.
    unique={r['origin_ids'][0]:r for r in examples}; examples=list(unique.values());rng.shuffle(examples)
    n=int(.8*len(examples));fit+=examples[:n]; feedback+=examples[n:]
    assert_disjoint({'fit':fit,'feedback':feedback})
    # The earlier holdout is consumed as dev; preserve frozen historic data files.
    dev=read_jsonl(DATA/'holdout_v1.jsonl')
    old=json.loads((DATA/'stress_dev.json').read_text(encoding='utf-8'))
    dev += [{'id':'old-dev-'+str(i),'text':r['text'],'expected':normalize_expected(r['expected']),'scenario':r.get('scenario',r.get('kind','unknown'))} for i,r in enumerate(old)]
    source=set(normalize(x[0]) for x in clean)
    if any(normalize(r['text']) in source for r in dev):raise ValueError('Exact overlap dev/Bitext')
    if len({normalize(r['text']) for r in dev})!=len(dev):raise ValueError('Duplicate within dev')
    write_jsonl(FIT,fit);write_jsonl(FEEDBACK,feedback);write_jsonl(DEV,dev)
    manifest={'kind':'source-ID-partition before synthetic generation','seed':SEED,'bitext_training_sources':len(train),'fit_sources':len(fit_src),'feedback_sources':len(feedback_src),'counts':{'fit':len(fit),'feedback':len(feedback),'dev':len(dev)},'fit_kinds':dict(Counter(r['kind'] for r in fit)),'feedback_kinds':dict(Counter(r['kind'] for r in feedback)),'bitext_sha256':sha(BITEXT),'clinc_sha256':sha(CLINC),'fit_sha256':sha(FIT),'feedback_sha256':sha(FEEDBACK),'dev_sha256':sha(DEV),'clinc_revision':'828f809','clinc_train_intent_allowlist':sorted(UNRELATED),'limitations':['Distinct broad-category composites do not prove recognition of multiple intents within one category.','Synthetic template overlap and annotation ambiguity remain; CLINC crowd prompts are not real customer tickets.','Only exact normalised duplicates/source IDs are controlled; semantic similarity can remain.']}
    (ART/'evolution_data_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8'); print(json.dumps(manifest,indent=2))

def base_predictions(rows,base):
    model=base['pipeline']; texts=[r['text'] for r in rows]; probs=model.predict_proba(texts); labels=model.classes_; idx=probs.argmax(axis=1); scores=probs.max(axis=1)
    return [{'predicted':str(labels[int(i)]),'confidence':float(sc),'base_review':bool(not has_known_content(r['text'],model.named_steps['tfidf'].vocabulary_) or sc < .6)} for r,i,sc in zip(rows,idx,scores)]

def eval_cases(rows,base,detector=None,threshold=None):
    predictions=base_predictions(rows,base)
    scores=detector.predict_proba([r['text'] for r in rows])[:,1] if detector else [0.0]*len(rows)
    cases=[]
    for row,p,s in zip(rows,predictions,scores):
        cases.append({**row,'predicted':p['predicted'],'confidence':p['confidence'],'review_detector_score':float(s) if detector else None,'needs_review':bool(p['base_review'] or (detector is not None and s>=threshold))})
    return cases

def select():
    require_new_outputs([G1, G2, SELECTION])
    # Never inspect or open HOLD in this phase.
    meta=json.loads((ART/'evolution_data_manifest.json').read_text(encoding='utf-8'))
    for key,path in [('bitext_sha256',BITEXT),('clinc_sha256',CLINC),('fit_sha256',FIT),('feedback_sha256',FEEDBACK),('dev_sha256',DEV)]:
        if meta[key]!=sha(path):raise ValueError(f'Frozen input changed: {path}')
    fit=read_jsonl(FIT);feedback=read_jsonl(FEEDBACK);dev=read_jsonl(DEV);assert_disjoint({'fit':fit,'feedback':feedback})
    base=joblib.load(BASE)
    train_text=[x['text'] for x in fit]; y=np.array([x['review'] for x in fit]); t2=[x['text'] for x in feedback]; y2=np.array([x['review'] for x in feedback])
    vector=TfidfVectorizer(ngram_range=(1,2),min_df=2,sublinear_tf=True,max_features=90000,dtype=np.float32)
    x1=vector.fit_transform(train_text);x2=vector.transform(t2)
    m1=LogisticRegression(max_iter=250,random_state=SEED,class_weight='balanced');m1.fit(x1,y)
    missed=(m1.predict(x2)!=y2); weights=np.concatenate([np.ones(len(y)),np.where(missed,3.,1.)])
    from scipy.sparse import vstack
    m2=LogisticRegression(max_iter=250,random_state=SEED,class_weight='balanced');m2.fit(vstack([x1,x2],format='csr'),np.concatenate([y,y2]),sample_weight=weights)
    from sklearn.pipeline import Pipeline
    p1=Pipeline([('tfidf',vector),('classifier',m1)]);p2=Pipeline([('tfidf',vector),('classifier',m2)])
    joblib.dump(p1,G1);joblib.dump(p2,G2)
    b=metrics(eval_cases(dev,base)); candidates=[]
    for name,model in [('g1',p1),('g2',p2)]:
        for threshold in [.25,.35,.45,.55,.65,.75,.85,.95]:
            measure=metrics(eval_cases(dev,base,model,threshold));candidates.append({'name':name,'threshold':threshold,**measure})
    selected=choose_candidate(b,candidates)
    selection={'protocol':'G1 source-ID fit; G2 retrained from scratch using fit + disjoint feedback and weight 3 for feedback prediction errors; dev only selects threshold/generation; holdout not read', 'selected':selected['name'],'threshold':selected.get('threshold'), 'dev_baseline':b,'dev_candidates':candidates,'feedback_error_count':int(missed.sum()),'feedback_total':len(feedback),'sample_weights':{'fit':1.0,'feedback_correct':1.0,'feedback_incorrect':3.0},'gate':'At least 2 fewer unflagged errors than baseline; coverage >= max(25%, baseline minus 10 percentage points).', 'sha256':{'bitext':sha(BITEXT),'clinc':sha(CLINC),'fit':sha(FIT),'feedback':sha(FEEDBACK),'dev':sha(DEV),'baseline':sha(BASE),'g1':sha(G1),'g2':sha(G2)},'created_utc':datetime.now(timezone.utc).isoformat()}
    SELECTION.write_text(json.dumps(selection,indent=2)+'\n',encoding='utf-8');print(json.dumps({'selected':selected,'dev_baseline':b,'feedback_error_count':int(missed.sum()),'feedback_total':len(feedback)},indent=2))

def final():
    require_new_outputs([REPORT])
    selection=json.loads(SELECTION.read_text(encoding='utf-8'))
    for k,p in [('bitext',BITEXT),('clinc',CLINC),('fit',FIT),('feedback',FEEDBACK),('dev',DEV),('baseline',BASE),('g1',G1),('g2',G2)]:
        if selection['sha256'][k]!=sha(p):raise ValueError(f'Frozen experiment input changed: {p}')
    rows=read_jsonl(HOLD)
    norm=[normalize(r['text']) for r in rows];dev={normalize(r['text']) for r in read_jsonl(DEV)}
    if len(set(norm))!=len(norm) or set(norm)&dev: raise ValueError('duplicate or overlap in holdout/dev')
    # Exact overlap with ALL training texts including generated composites is rejected.
    known_texts={normalize(r['text']) for r in read_jsonl(FIT)}|{normalize(r['text']) for r in read_jsonl(FEEDBACK)}
    if set(norm)&known_texts:raise ValueError('holdout/training exact overlap')
    # Baseline was trained on Bitext, so check the whole source too, not merely
    # the subset sampled into the detector's generated training examples.
    with BITEXT.open(encoding='utf-8-sig',newline='') as f:
        original_bitext={normalize(row['instruction']) for row in csv.DictReader(f)}
    original_clinc={normalize(row[0]) for row in json.loads(CLINC.read_text(encoding='utf-8'))['train']}
    if set(norm)&(original_bitext|original_clinc):raise ValueError('holdout/source exact overlap')
    base=joblib.load(BASE);baseline=eval_cases(rows,base)
    name=selection['selected']; threshold=selection['threshold']; selected=baseline if name=='baseline' else eval_cases(rows,base,joblib.load(G1 if name=='g1' else G2),threshold)
    report={'available':True,'dataset':{'description':'Independent AI-authored synthetic holdout, not customer submissions','count':len(rows),'sha256':sha(HOLD)},'selection':selection,'baseline':{'name':'baseline','metrics':metrics(baseline),'cases':baseline},'selected':{'name':name,'threshold':threshold,'metrics':metrics(selected),'cases':selected},'active_model':'baseline (application not modified by experiment)','limitations':['Synthetic training and test examples; annotation not independently verified by human; real-world performance unknown.', 'Original Bitext categories do not detect multiple intents within the same broad category.', 'Splits group original exact normalized examples; semantically similar templates can overlap.','The detector is combined with current score rule; unflagged items are not automatically sent anywhere.','CLINC intent allowlist is only a proxy for unrelatedness and can contain ambiguous rows.','This holdout uses message text; live API uses subject and body.']}
    REPORT.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');print(json.dumps({'baseline':report['baseline']['metrics'],'selected':{'name':name,**report['selected']['metrics']}},indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['prepare','select','final']);args=p.parse_args();globals()[args.phase]()
