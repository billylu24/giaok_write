"""Download HWDB1.1 archives locally; preserve origin and transport provenance."""
import argparse, concurrent.futures, hashlib, json, subprocess, time, zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def checksum(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
 return h.hexdigest()

def download(record):
 dest=ROOT/'data/casia/raw'/record['name'];dest.parent.mkdir(parents=True,exist_ok=True)
 started=time.time();r=dict(record)
 if not dest.exists():
  part=dest.with_suffix(dest.suffix+'.part')
  cmd=['curl','-fL','--retry','5','--retry-all-errors','--continue-at','-','--connect-timeout','30','--max-time','3600',record['url'],'-o',str(part)]
  with (dest.parent/(record['name']+'.download.log')).open('w') as log:
   subprocess.run(cmd,stdout=log,stderr=log,check=True)
  if part.stat().st_size!=record['size']:raise ValueError('Unexpected size: '+record['name'])
  actual=checksum(part)
  if actual!=record['sha256']:raise ValueError('Checksum mismatch: '+record['name'])
  part.replace(dest)
 else:
  actual=checksum(dest)
  if dest.stat().st_size!=record['size'] or actual!=record['sha256']:raise ValueError('Existing archive is corrupt')
 with zipfile.ZipFile(dest) as z:
  names=z.namelist();gnt=[n for n in names if n.lower().endswith('.gnt')]
  nested_alz=[n for n in names if n.lower().endswith('.alz')]
  r.update(verified_sha256=actual,gnt_files=len(gnt),nested_alz_members=nested_alz,uncompressed_bytes=sum(x.file_size for x in z.infolist()),archive_bytes=dest.stat().st_size,download_seconds=time.time()-started)
 print(json.dumps(r,ensure_ascii=False),flush=True);return r

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--config',default='configs/casia_download.json');a=ap.parse_args()
 manifest=json.loads((ROOT/a.config).read_text())
 with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:records=list(pool.map(download,manifest['resources']))
 manifest.update(status='downloaded_and_checksum_verified',resources=records)
 (ROOT/'data/casia/download_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
 out=ROOT/'experiments/casia_001';out.mkdir(parents=True,exist_ok=True)
 (out/'download_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
