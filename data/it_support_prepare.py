"""Prepare the Zenodo v2 real-ticket corpus with preserved source test.

Original x/y files remain unmodified. Test data is used without labels only for
pre-declared near-duplicate grouping; downstream model transforms must fit train only.
"""
import csv, json, hashlib, re
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import StratifiedGroupKFold

ROOT=Path(__file__).resolve().parents[1]
RAW=ROOT/'data/it_support_raw'; OUT=ROOT/'data/it_support_prepared'; ART=ROOT/'artifacts/it_data_manifest.json'
EXPECTED={'X_train.csv':'0a79f0c2f95fcdb407f24b2c0ad7f7d1','X_test.csv':'02328c4e09a78a41acb22ce17e6e0312','y_train.csv':'90876921486a26a67bc1976da87305d9','y_test.csv':'2f054c0e3d8cbbbd76adf30d13a7710a'}
SEED=42; THRESHOLD=0.95

def norm(text):return ' '.join(text.casefold().split())
def digest(text):return hashlib.sha256(norm(text).encode()).hexdigest()
def read(split):
    def load(name):
        with (RAW/name).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
    x=load(f'X_{split}.csv'); y=load(f'y_{split}.csv')
    if len(x)!=len(y) or [r['id'] for r in x]!=[r['id'] for r in y]:raise ValueError('Mismatched source IDs')
    rows=[]
    for a,b in zip(x,y):
        if not a['text'].strip() or not b['category_truth'].strip():raise ValueError('Blank text or label')
        rows.append({'id':f"{split}:{a['id']}",'text':a['text'],'label':b['category_truth'],'source_split':split,'exact_group':digest(a['text'])})
    return rows

def union_groups(texts):
    """Threshold connected-components of normalized text; input includes unlabeled source test."""
    mat=TfidfVectorizer(analyzer='char',ngram_range=(3,5),min_df=1,sublinear_tf=True,dtype=np.float32).fit_transform(texts)
    similarities=cosine_similarity(mat,dense_output=False).tocoo();n=len(texts);parent=list(range(n))
    def find(i):
        while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
        return i
    def union(a,b):
        a,b=find(a),find(b)
        if a!=b:parent[max(a,b)]=min(a,b)
    for i,j,v in zip(similarities.row,similarities.col,similarities.data):
        if i<j and v>=THRESHOLD:union(i,j)
    return [find(i) for i in range(n)]

def save(path,text):
    if path.exists():
        if path.read_text(encoding='utf-8')!=text:raise FileExistsError(f'Refusing to overwrite nonidentical historical data: {path}')
        return
    path.write_text(text,encoding='utf-8')

def main():
    for name,expected in EXPECTED.items():
        if hashlib.md5((RAW/name).read_bytes()).hexdigest()!=expected:raise ValueError('Unexpected source digest: '+name)
    train,test=read('train'),read('test'); rows=train+test
    byexact=defaultdict(list)
    for r in rows:byexact[r['exact_group']].append(r)
    unique=sorted(byexact)
    normalized=[norm(byexact[key][0]['text']) for key in unique]
    components=union_groups(normalized); compmap={key:f'near-{components[i]:04d}' for i,key in enumerate(unique)}
    for r in rows:r['group_id']=compmap[r['exact_group']]
    test_comp={r['group_id'] for r in test}
    train_labels=defaultdict(set)
    for r in train:train_labels[r['exact_group']].add(r['label'])
    conflicts={group for group,values in train_labels.items() if len(values)>1}
    usable=[r for r in train if r['group_id'] not in test_comp and r['exact_group'] not in conflicts]
    # Deduplicate train exact normalized texts with same label (keep source order).
    unique_train={}
    for r in usable:unique_train.setdefault(r['exact_group'],r)
    usable=list(unique_train.values()); groups=[r['group_id'] for r in usable]; labels=[r['label'] for r in usable]
    splitter=StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=SEED)
    fit_idx,dev_idx=next(splitter.split(np.zeros(len(usable)),labels,groups))
    output={'train':[usable[int(i)] for i in fit_idx], 'dev':[usable[int(i)] for i in dev_idx], 'test':test}
    for name,part in output.items():
        for r in part:r.pop('exact_group')
        assert len({r['id'] for r in part})==len(part)
    assert not ({r['group_id'] for r in output['train']}&{r['group_id'] for r in output['dev']})
    assert not (({r['group_id'] for r in output['train']}|{r['group_id'] for r in output['dev']})&test_comp)
    OUT.mkdir(parents=True,exist_ok=True)
    for name,part in output.items():save(OUT/f'{name}.jsonl',''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in part))
    manifest={'source':'https://zenodo.org/records/7648117','doi':'10.5281/zenodo.7648117','source_license':'CC-BY-4.0','source_md5':EXPECTED,'source_counts':{'train':len(train),'test':len(test)},'split_method':'Original source test retained. Exact and near-duplicate components defined before splits using unlabeled normalized text and char TF-IDF cosine >=0.95; exclude train components touching test; remove conflicting exact-label training groups; deduplicate exact training groups; StratifiedGroupKFold 5, shuffle seed 42, fold 1 train/dev.','near_similarity_threshold':THRESHOLD,'seed':SEED,'near_components_total':len(set(components)),'test_near_components':len(test_comp),'source_train_rows_in_test_components':sum(r['group_id'] in test_comp for r in train),'source_train_conflicting_exact_groups':len(conflicts),'source_train_unique_after_purge':len(usable),'counts':{name:len(part) for name,part in output.items()},'group_counts':{name:len({r['group_id'] for r in part}) for name,part in output.items()},'class_counts':{name:dict(sorted(Counter(r['label'] for r in part).items())) for name,part in output.items()},'output_sha256':{name:hashlib.sha256((OUT/f'{name}.jsonl').read_bytes()).hexdigest() for name in output},'privacy':'Author reports de-identification; automated pattern checks are not proof of zero PII. Keep raw text local.','limitations':['Pre-declared unlabeled near-duplicate grouping fitted on all text including test; it is used only to construct conservative splits, never as model input representation.', 'Cosine threshold 0.95 detects only very similar surface forms, not all semantic or templated overlap.', 'Source dates, organizations, speakers, and language labels are not included; cannot establish chronological, client, or linguistic generalization.', 'Test includes exact duplicate source rows; report both row-weighted and unique-exact-group metrics.', 'No independent human validation of labels or anonymization was performed.']}
    save(ART,json.dumps(manifest,indent=2,ensure_ascii=False)+'\n');print(json.dumps(manifest,indent=2,ensure_ascii=False))
if __name__=='__main__':main()
