"""Low-memory local IT classifier experiment. No historical test data are read."""
import ctypes, hashlib, json, math, os, threading, time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
ART=ROOT/'artifacts'
REPORT=ART/'it_lowmem_report.json'
WATCHDOG=ART/'it_lowmem_watchdog.json'
MIN_START=1600*1024*1024
MIN_RUNNING=700*1024*1024
BASELINE_DEV_MACRO_F1=0.7161779001767451
MAX_EPOCHS=6

class MemoryStatus(ctypes.Structure):
 _fields_=[('size',ctypes.c_ulong),('load',ctypes.c_ulong),('total',ctypes.c_ulonglong),('avail',ctypes.c_ulonglong),('totalpage',ctypes.c_ulonglong),('availpage',ctypes.c_ulonglong),('totalvirtual',ctypes.c_ulonglong),('availvirtual',ctypes.c_ulonglong),('extended',ctypes.c_ulonglong)]
def available_memory_bytes():
 mem=MemoryStatus();mem.size=ctypes.sizeof(mem)
 if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(mem)):raise RuntimeError('Cannot check available physical memory')
 return mem.avail

def require_memory(minimum=MIN_RUNNING):
 avail=available_memory_bytes()
 if avail<minimum:raise MemoryError(f'Available physical memory {avail//1048576} MiB, required {minimum//1048576} MiB')
 return avail

def require_new(*paths):
 old=[str(p) for p in paths if Path(p).exists()]
 if old:raise FileExistsError('Historical evidence exists: '+', '.join(old))
def planned_batch_count(total,accumulation):return math.ceil(total/accumulation)
def sha(path):
 h=hashlib.sha256()
 with open(path,'rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def write_new(path,data):
 with open(path,'x',encoding='utf-8') as f:json.dump(data,f,indent=2);f.write('\n')

def main():
 outputs=[REPORT,WATCHDOG]+[ART/f'it_lowmem_epoch_{i}.pt' for i in range(1,MAX_EPOCHS+1)]
 require_new(*outputs)
 initial=available_memory_bytes()
 report={'status':'not_started','protocol':'192-token MiniLM, batch 1, accumulation 4, dynamic padding, max 6 epochs, best dev checkpoint; no historical test','available_at_start_mib':initial//1048576,'minimum_start_mib':MIN_START//1048576,'minimum_running_mib':MIN_RUNNING//1048576,'baseline_dev_macro_f1':BASELINE_DEV_MACRO_F1,'epochs':[], 'selected_checkpoint':None, 'limitations':['Minimum physical memory guard does not reserve memory; other programs may change memory availability.', 'The previously used development split is not independent.', 'No new independent test or real-world performance claim.', 'Unknown-category detection is not addressed.']}
 if initial<MIN_START:
  report['status']='unavailable_memory';report['reason']=f'Available physical memory {initial//1048576} MiB, required {MIN_START//1048576} MiB';report['available_at_end_mib']=available_memory_bytes()//1048576
  write_new(REPORT,report);print(json.dumps(report),flush=True);return
 stop=threading.Event()
 def watchdog():
  while not stop.wait(.25):
   try:
    avail=available_memory_bytes()
    if avail<MIN_RUNNING:
     try:write_new(WATCHDOG,{'status':'watchdog_stopped','available_mib':avail//1048576,'limit_mib':MIN_RUNNING//1048576,'note':'Process stopped; inspect completed checkpoints and report.'})
     finally:os._exit(75)
   except Exception:
    try:write_new(WATCHDOG,{'status':'watchdog_failed','note':'Memory could not be monitored; stopping.'})
    finally:os._exit(76)
 thread=threading.Thread(target=watchdog,daemon=True);thread.start()
 try:
  import numpy as np
  import torch
  from ml.it_experiment import build_network,read_jsonl,classification,partial_state,PRETRAINED,REVISION,TRAIN,DEV
  torch.manual_seed(42);torch.set_num_threads(2);np.random.seed(42)
  require_memory();train,dev=read_jsonl(TRAIN),read_jsonl(DEV)
  if {r['group_id'] for r in train}&{r['group_id'] for r in dev}:raise ValueError('Train/dev group overlap')
  labels=sorted({r['label'] for r in train});label_ids={label:i for i,label in enumerate(labels)}
  tokenizer,model=build_network(labels,local_files_only=True);require_memory()
  optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=4e-5,weight_decay=.01)
  truth=[label_ids[r['label']] for r in train];counts=np.bincount(truth,minlength=len(labels));weights=torch.tensor(len(truth)/(len(labels)*np.maximum(counts,1)),dtype=torch.float32)
  criterion=torch.nn.CrossEntropyLoss(weight=weights,reduction='sum');rng=np.random.default_rng(42);best=-1.0
  def tok(text):
   return {k:v for k,v in tokenizer(text,padding=False,truncation=True,max_length=192,return_tensors='pt').items() if k in ('input_ids','attention_mask','token_type_ids')}
  for epoch in range(MAX_EPOCHS):
   model.train();order=rng.permutation(len(train));started=time.time();batch_start=time.time();optimizer.zero_grad(set_to_none=True);weight_sum=0.
   for step,i in enumerate(order):
    require_memory();target=torch.tensor([truth[int(i)]]);loss=criterion(model(**tok(train[int(i)]['text'])),target);loss.backward();weight_sum+=float(weights[target].item())
    if (step+1)%4==0 or step+1==len(order):
     for p in model.parameters():
      if p.grad is not None:p.grad.div_(weight_sum)
     optimizer.step();optimizer.zero_grad(set_to_none=True);weight_sum=0.
    if time.time()-batch_start>30:raise RuntimeError('One training step took more than 30 seconds')
    batch_start=time.time()
    if step%100==0:print(json.dumps({'event':'progress','epoch':epoch+1,'processed':step,'total':len(order),'available_mib':available_memory_bytes()//1048576}),flush=True)
   pred=[];model.eval()
   with torch.no_grad():
    for row in dev:
     require_memory();pred.append(labels[int(model(**tok(row['text'])).argmax(dim=1).item())])
   result=classification(dev,pred,labels);checkpoint=ART/f'it_lowmem_epoch_{epoch+1}.pt'
   require_new(checkpoint);torch.save({'labels':labels,'state':partial_state(model),'epoch':epoch+1,'model':PRETRAINED,'revision':REVISION,'max_length':192},checkpoint)
   row={'epoch':epoch+1,'dev_accuracy':result['accuracy'],'dev_macro_f1':result['macro_f1'],'seconds':round(time.time()-started,1),'checkpoint_sha256':sha(checkpoint)};report['epochs'].append(row)
   if row['dev_macro_f1']>best:best=row['dev_macro_f1'];report['selected_checkpoint']=checkpoint.name
   print(json.dumps({'event':'epoch',**row}),flush=True)
  report['status']='completed'
 except MemoryError as exc:report['status']='stopped_memory' if report['epochs'] else 'unavailable_memory';report['reason']=str(exc)
 except Exception as exc:report['status']='stopped_error';report['reason']=f'{type(exc).__name__}: {str(exc)[:240]}'
 finally:stop.set();thread.join(timeout=1)
 report['available_at_end_mib']=available_memory_bytes()//1048576
 if report['epochs']:
  row=max(report['epochs'],key=lambda x:x['dev_macro_f1']);report['best_epoch']=row['epoch'];report['best_dev_macro_f1']=row['dev_macro_f1'];report['outperformed_baseline_on_dev']=row['dev_macro_f1']>BASELINE_DEV_MACRO_F1
 write_new(REPORT,report);print(json.dumps(report),flush=True)
if __name__=='__main__':main()
