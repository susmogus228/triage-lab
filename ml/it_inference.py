"""Local inference for separately validated IT taxonomy; always manual review."""
import json
from functools import lru_cache
import joblib
from ml.it_experiment import SELECTION, SELECTION_V2, REPORT, BASE, NEURAL, ART, PRETRAINED_DIR, verify_frozen, sha256, load_neural

@lru_cache(maxsize=1)
def _load():
    if not SELECTION.is_file() or not REPORT.is_file(): raise RuntimeError('IT experiment has not been evaluated.')
    selection_path=SELECTION_V2 if SELECTION_V2.exists() else SELECTION
    sel=json.loads(selection_path.read_text(encoding='utf-8')); result=json.loads(REPORT.read_text(encoding='utf-8'))
    if result.get('selection_sha256')!=sha256(selection_path) or result['selected']!=sel['selected']:raise RuntimeError('Evaluation and selected model disagree.')
    key=sel['selected']; labels=sel['labels']
    if key=='tfidf':
        verify_frozen(sel['sha256'],{'tfidf':BASE});return key,labels,joblib.load(BASE)
    if key=='multilingual':
        checkpoint=ART/sel.get('neural_checkpoint',NEURAL.name)
        verify_frozen(sel['sha256'],{'neural':checkpoint,'pretrained_weights':PRETRAINED_DIR/'model.safetensors'})
        return key,labels,load_neural(labels,checkpoint)
    raise RuntimeError('Unknown selected IT model')

def predict(text):
    if not isinstance(text,str) or not text.strip():raise ValueError('Ticket text is required.')
    key,labels,model=_load()
    if key=='tfidf':
        probabilities=model.predict_proba([text])[0];labels=list(model.classes_)
    else:
        import torch
        tokenizer,network=model;network.eval()
        tokens=tokenizer([text],padding='max_length',truncation=True,max_length=96,return_tensors='pt')
        tokens={k:v for k,v in tokens.items() if k in ('input_ids','attention_mask','token_type_ids')}
        with torch.no_grad():probabilities=torch.softmax(network(**tokens),dim=1)[0].cpu().numpy()
    i=int(probabilities.argmax())
    return {'label':str(labels[i]),'confidence':float(probabilities[i]),'needs_review':True,'model_version':'it-support-'+key,'notice':'Known categories only; unseen categories and OOD are not reliably detected.'}
