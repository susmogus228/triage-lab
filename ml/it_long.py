"""Bounded long-context IT model experiment; never reads historical test."""
import ctypes, hashlib, json, os, time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
ART=ROOT/'artifacts'
REPORT=ART/'it_long_report.json'
MIN_RUNNING=700*1024*1024
MIN_START=1600*1024*1024
MAX_EPOCHS=6
BASELINE_DEV_MACRO_F1=0.7161779001767451

class _MemoryStatus(ctypes.Structure):
 _fields_=[('dwLength',ctypes.c_ulong),('dwMemoryLoad',ctypes.c_ulong),('ullTotalPhys',ctypes.c_ulonglong),('ullAvailPhys',ctypes.c_ulonglong),('ullTotalPageFile',ctypes.c_ulonglong),('ullAvailPageFile',ctypes.c_ulonglong),('ullTotalVirtual',ctypes.c_ulonglong),('ullAvailVirtual',ctypes.c_ulonglong),('sullAvailExtendedVirtual',ctypes.c_ulonglong)]

def available_memory_bytes():
 mem=_MemoryStatus();mem.dwLength=ctypes.sizeof(mem)
 if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(mem)):raise RuntimeError('Cannot check physical memory')
 return mem.ullAvailPhys

def require_memory(min_bytes=MIN_RUNNING):
 available=available_memory_bytes()
 if available<min_bytes:raise MemoryError(f'Available physical memory {available//(1024*1024)} MiB, required {min_bytes//(1024*1024)} MiB')
 return available

def require_new(*paths):
 existing=[str(p) for p in paths if p.exists()]
 if existing:raise FileExistsError('Refusing to overwrite: '+', '.join(existing))

def is_better(score,baseline):return float(score)>float(baseline)
def sha(path):
 digest=hashlib.sha256()
 with open(path,'rb') as f:
  for block in iter(lambda:f.read(1024*1024),b''):digest.update(block)
 return digest.hexdigest()
def write_json(report):
 with REPORT.open('x',encoding='utf8') as f:json.dump(report,f,indent=2);f.write('\n')

def main():
 outputs=[REPORT]+[ART/f'it_long_epoch_{i}.pt' for i in range(1,MAX_EPOCHS+1)]
 require_new(*outputs)
 report={'status':'not_started','protocol':'192-token MiniLM; last encoder layer + classification head; 6 epochs max; choose checkpoint by previously used dev macro F1; historical and external test never read', 'available_at_start_mib':available_memory_bytes()//(1024*1024),'minimum_start_mib':MIN_START//(1024*1024),'minimum_running_mib':MIN_RUNNING//(1024*1024),'baseline_dev_macro_f1':BASELINE_DEV_MACRO_F1,'epochs':[],'selected_checkpoint':None,'limitations':['Prior IT dev was used repeatedly for model selection and is not independent.','No comparison on a new independent final test.','Unknown request detection is a separate task and is not evaluated in this experiment.','One organisation dataset with 6 known categories; language and demographic fairness not validated.']}
 try:
  require_memory(MIN_START)
  import numpy as np
  import torch
  from ml.it_experiment import build_network,read_jsonl,classification,partial_state,PRETRAINED,REVISION,TRAIN,DEV
  torch.manual_seed(42);torch.set_num_threads(2);np.random.seed(42)
  require_memory()
  train,dev=read_jsonl(TRAIN),read_jsonl(DEV)
  if {r['group_id'] for r in train}&{r['group_id'] for r in dev}:raise ValueError('Train/dev group overlap')
  labels=sorted({r['label'] for r in train});idx={label:i for i,label in enumerate(labels)}
  # Weight hash and tokenizer assets are checked by build_network.
  tokenizer,model=build_network(labels,local_files_only=True)
  require_memory()
  def tokenise(rows):
   return {k:v for k,v in tokenizer([r['text'] for r in rows],padding='max_length',truncation=True,max_length=192,return_tensors='pt').items() if k in ('input_ids','attention_mask','token_type_ids')}
  t=tokenise(train);v=tokenise(dev);truth=[idx[r['label']] for r in train]
  optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=4e-5,weight_decay=.01)
  counts=np.bincount(truth,minlength=len(labels));weights=torch.tensor(len(truth)/(len(labels)*np.maximum(counts,1)),dtype=torch.float32)
  criterion=torch.nn.CrossEntropyLoss(weight=weights);rng=np.random.default_rng(42);best=-1.0
  report['status']='training'
  for epoch in range(MAX_EPOCHS):
   model.train();order=rng.permutation(len(truth));started=time.time();batch=4
   for start in range(0,len(order),batch):
    require_memory()
    ids=order[start:start+batch].tolist();inputs={k:value[ids] for k,value in t.items()};target=torch.tensor([truth[i] for i in ids])
    batch_start=time.time();optimizer.zero_grad(set_to_none=True);loss=criterion(model(**inputs),target);loss.backward();optimizer.step()
    if time.time()-batch_start>30:raise RuntimeError('Batch stalled for more than 30 seconds; stopped to avoid prolonged swapping')
    if start%160==0:print(json.dumps({'event':'progress','epoch':epoch+1,'examples':start,'total':len(order),'available_mib':available_memory_bytes()//(1024*1024),'elapsed':round(time.time()-started,1)}),flush=True)
   pred=[];model.eval()
   with torch.no_grad():
    for start in range(0,len(dev),4):
     require_memory();inputs={k:value[start:start+4] for k,value in v.items()};pred.extend(model(**inputs).argmax(dim=1).tolist())
   result=classification(dev,[labels[i] for i in pred],labels)
   row={'epoch':epoch+1,'dev_accuracy':result['accuracy'],'dev_macro_f1':result['macro_f1'],'seconds':round(time.time()-started,1)};report['epochs'].append(row)
   checkpoint=ART/f'it_long_epoch_{epoch+1}.pt'
   torch.save({'labels':labels,'state':partial_state(model),'epoch':epoch+1,'model':PRETRAINED,'revision':REVISION,'max_length':192},checkpoint)
   row['checkpoint_sha256']=sha(checkpoint)
   if is_better(result['macro_f1'],best):best=result['macro_f1'];report['selected_checkpoint']=checkpoint.name
   print(json.dumps({'event':'epoch',**row}),flush=True)
  report['status']='completed'
 except MemoryError as exc:
  report['status']='stopped_memory' if report['epochs'] else 'unavailable_memory';report['reason']=str(exc)
 except RuntimeError as exc:
  report['status']='stopped_runtime';report['reason']=str(exc)[:300]
 report['available_at_end_mib']=available_memory_bytes()//(1024*1024)
 if report['epochs']:
  best=max(report['epochs'],key=lambda x:x['dev_macro_f1']);report['best_dev_macro_f1']=best['dev_macro_f1'];report['best_epoch']=best['epoch'];report['outperformed_baseline_on_dev']=is_better(best['dev_macro_f1'],BASELINE_DEV_MACRO_F1)
 write_json(report)
 print(json.dumps(report),flush=True)
if __name__=='__main__':main()
