"""Fetch pinned multilingual MiniLM weights and tokeniser; verify upstream hashes."""
import hashlib,urllib.request,concurrent.futures,time
from pathlib import Path
base='https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2/resolve/e8f8c211226b894fcb81acc59f3b34ba3efd5f42/'
root=Path('work/it_pretrained');root.mkdir(parents=True,exist_ok=True)
small={'config.json':('git','c06d5b49495f044e6380e68a60538be17a6bd5d1'), 'tokenizer_config.json':('git','3c1b565ae10a15a1d0c31096f834af2fd9359e91'),'special_tokens_map.json':('git','2ea7ad0e45a9d1d1591782ba7e29a703d0758831'), 'tokenizer.json':('sha256','2c3387be76557bd40970cec13153b3bbf80407865484b209e655e5e4729076b8')}
for name,(kind,expected) in small.items():
 data=urllib.request.urlopen(base+name,timeout=90).read()
 actual=hashlib.sha1(f'blob {len(data)}\0'.encode()+data).hexdigest() if kind=='git' else hashlib.sha256(data).hexdigest()
 if actual!=expected:raise ValueError(f'{name} digest mismatch')
 (root/name).write_bytes(data);print(name,'verified',flush=True)
size=470641600;chunk=4*1024*1024;url=base+'model.safetensors';path=root/'model.safetensors'
def get(i):
 start=i*chunk;end=min(size-1,start+chunk-1)
 for attempt in range(5):
  try:
   req=urllib.request.Request(url,headers={'Range':f'bytes={start}-{end}'})
   with urllib.request.urlopen(req,timeout=80) as r:
    data=r.read(); expected_range=f'bytes {start}-{end}/{size}'
    if r.status!=206 or r.headers.get('Content-Range')!=expected_range or len(data)!=(end-start+1):raise ValueError('range mismatch')
    return i,data
  except Exception:
   if attempt==4:raise
   time.sleep(attempt+1)
with path.open('wb') as f:
 f.truncate(size)
 with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
  futures=[pool.submit(get,i) for i in range((size+chunk-1)//chunk)]
  for completed,future in enumerate(concurrent.futures.as_completed(futures),1):
   i,data=future.result();f.seek(i*chunk);f.write(data)
   if completed%10==0:print(f'model {completed}/{len(futures)} chunks',flush=True)
h=hashlib.sha256()
with path.open('rb') as f:
 for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
if h.hexdigest()!='eaa086f0ffee582aeb45b36e34cdd1fe2d6de2bef61f8a559a1bbc9bd955917b':raise ValueError('model checksum mismatch')
print('Pinned model SHA-256 verified',flush=True)
