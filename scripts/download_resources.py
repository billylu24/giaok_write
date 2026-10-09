"""Download pinned official sources, validate PDFs, and record checksums."""
import argparse
import concurrent.futures
import hashlib
import json
import subprocess
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HEADERS = {'User-Agent': 'giaok-write-research/0.1'}
SOURCE_COMMIT = '7b28ce9c3b357f4fb23296622f458cf169803539'
WEIGHT_REVISION = '7bd89f3932f30e0b0062b4ba852a4270a20fd0e4'

def get(url):
    with urllib.request.urlopen(urllib.request.Request(url, headers=HEADERS), timeout=60) as r:
        return r.read()

def download(url, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        return hashlib.sha256(destination.read_bytes()).hexdigest()
    part = destination.with_suffix(destination.suffix + '.part')
    result = subprocess.run(['curl','-sSfL','--retry','4','--retry-all-errors','--continue-at','-','--connect-timeout','30','--max-time','1800',url,'-o',str(part)],capture_output=True,text=True)
    if result.returncode:
        raise RuntimeError(result.stderr.strip())
    part.replace(destination)
    return hashlib.sha256(destination.read_bytes()).hexdigest()

def paper(item):
    dest = ROOT/'papers'/f"{item['id']}.pdf"
    record = dict(item)
    try:
        sha = download(item['pdf_url'],dest)
        if not dest.read_bytes().startswith(b'%PDF-'):
            raise ValueError('Response is not a PDF')
        info = subprocess.run(['pdfinfo',str(dest)],capture_output=True,text=True,check=True).stdout
        subprocess.run(['pdftotext','-layout',str(dest),str(dest.with_suffix('.txt'))],check=True)
        record.update(status='downloaded',sha256=sha,file=str(dest.relative_to(ROOT)),pdfinfo=info)
        print(f"PDF OK: {item['id']}",flush=True)
    except Exception as e:
        record.update(status='failed',error=str(e))
        print(f"PDF FAILED: {item['id']}: {e}",flush=True)
    return record

def sources():
    tree=json.loads(get(f'https://api.github.com/repos/yeungchenwa/FontDiffuser/git/trees/{SOURCE_COMMIT}?recursive=1'))
    sha=tree['sha']
    # Fetch only code, documentation and small examples, avoiding large website images.
    selected=[x['path'] for x in tree['tree'] if x['type']=='blob' and (x['path'].endswith(('.py','.sh','.md','.txt')) or x['path'].startswith(('data_examples/','figures/ref_imgs/','figures/source_imgs/')))]
    def one(path):
        url=f'https://raw.githubusercontent.com/yeungchenwa/FontDiffuser/{sha}/'+urllib.parse.quote(path)
        data=get(url)
        dst=ROOT/'vendor'/'FontDiffuser'/path; dst.parent.mkdir(parents=True,exist_ok=True);dst.write_bytes(data)
        return dict(path=path,sha256=hashlib.sha256(data).hexdigest())
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
        files=list(pool.map(one,selected))
    record=dict(repository='https://github.com/yeungchenwa/FontDiffuser',commit=sha,download_method='GitHub raw files, pinned commit',files=files)
    (ROOT/'vendor'/'FontDiffuser'/'SOURCE_PROVENANCE.json').write_text(json.dumps(record,indent=2)+'\n')
    print(f'SOURCE OK: {len(files)} files, commit {sha}',flush=True)

def weights():
    repo='yeungchenwa/FontDiffuser-Gradio'
    rev=WEIGHT_REVISION
    entries=json.loads(get('https://huggingface.co/api/spaces/'+repo+'/tree/'+rev+'/ckpt'))
    def one(item):
        path=item['path']; dest=ROOT/'checkpoints'/'fontdiffuser'/Path(path).name
        sha=download(f'https://huggingface.co/spaces/{repo}/resolve/{rev}/{path}?download=true',dest)
        expected=item.get('lfs',{}).get('oid')
        if expected and sha != expected:
            raise ValueError('Weight checksum mismatch: '+path)
        print('WEIGHT OK: '+path,flush=True)
        return dict(path=path,size=dest.stat().st_size,sha256=sha)
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        files=list(pool.map(one,[x for x in entries if x['type']=='file']))
    (ROOT/'checkpoints'/'fontdiffuser'/'manifest.json').write_text(json.dumps(dict(repository=repo,revision=rev,files=files),indent=2)+'\n')

def papers():
    items=json.loads((ROOT/'papers'/'manifest.json').read_text())
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as pool:
        records=list(pool.map(paper,items))
    (ROOT/'papers'/'download_report.json').write_text(json.dumps(records,ensure_ascii=False,indent=2)+'\n')
    failures=[x['id'] for x in records if x['status']!='downloaded']
    print('PAPER FAILURES:',failures,flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('resource',choices=['sources','weights','papers']);args=ap.parse_args()
    globals()[args.resource]()
