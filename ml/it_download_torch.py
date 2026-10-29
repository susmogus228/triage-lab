"""Download the official CPU PyTorch wheel in ranges and verify its SHA-256.
URL and digest obtained from https://download.pytorch.org/whl/cpu/torch/.
"""
import hashlib,os,time,urllib.request,concurrent.futures
from pathlib import Path
url='https://download-r2.pytorch.org/whl/cpu/torch-2.6.0%2Bcpu-cp311-cp311-win_amd64.whl'
expected='24c9d3d13b9ea769dd7bd5c11cfa1fc463fd7391397156565484565ca685d908'
size=206540444;chunk=2*1024*1024
path=Path('work/it_wheels/torch-2.6.0+cpu-cp311-cp311-win_amd64.whl');path.parent.mkdir(parents=True,exist_ok=True)
def get(i):
 start=i*chunk;end=min(size-1,start+chunk-1)
 for attempt in range(4):
  try:
   req=urllib.request.Request(url,headers={'Range':f'bytes={start}-{end}'})
   with urllib.request.urlopen(req,timeout=40) as r:
    data=r.read();expected_range=f'bytes {start}-{end}/{size}'
    if r.status!=206 or r.headers.get('Content-Range')!=expected_range or len(data)!=(end-start+1):raise ValueError('range mismatch')
    return i,data
  except Exception:
   if attempt==3:raise
   time.sleep(attempt+1)
with path.open('wb') as f:
 f.truncate(size)
 with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
  futures=[pool.submit(get,i) for i in range((size+chunk-1)//chunk)]
  for completed,future in enumerate(concurrent.futures.as_completed(futures),1):
   i,data=future.result();f.seek(i*chunk);f.write(data)
   if completed%10==0:print(f'{completed}/{len(futures)} chunks',flush=True)
h=hashlib.sha256()
with path.open('rb') as f:
 for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
if h.hexdigest()!=expected:raise ValueError('PyTorch wheel checksum mismatch')
print('Official wheel SHA-256 verified',flush=True)
