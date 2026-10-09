"""Convert all HWDB1.1 records to compact, writer-isolated NumPy arrays."""
import argparse,concurrent.futures,hashlib,json,re,sys,time,zipfile
from collections import Counter
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from handwriting.gnt import read_records,is_hanzi,normalize_image

def process_writer(job):
    archive,member,origin,resolution,extent,out=job
    stem=Path(member).stem
    match=re.match(r'(\d+)',stem)
    writer_id=match.group(1) if match else stem
    target=Path(out)/'writers'/writer_id;target.mkdir(parents=True,exist_ok=True)
    images=[];codes=[];other=Counter();record_count=0;dimensions=[]
    import contextlib
    with contextlib.ExitStack() as stack:
        if Path(archive).is_dir():
            f=stack.enter_context((Path(archive)/member).open('rb'))
        else:
            z=stack.enter_context(zipfile.ZipFile(archive));f=stack.enter_context(z.open(member))
        for char,image,offset in read_records(f):
            record_count+=1
            if not is_hanzi(char):other[char]+=1;continue
            images.append(normalize_image(image,resolution,extent));codes.append(ord(char));dimensions.append(image.shape)
    if not images:raise ValueError('No Chinese samples in '+member)
    duplicate_labels={chr(n):count for n,count in Counter(codes).items() if count>1}
    np.save(target/'images.npy',np.stack(images),allow_pickle=False)
    np.save(target/'codepoints.npy',np.asarray(codes,dtype=np.int32),allow_pickle=False)
    report={'writer_id':writer_id,'original_partition':origin,'archive':Path(archive).name,'member':member,'records':record_count,'hanzi_samples':len(images),'duplicate_labels':duplicate_labels,'non_hanzi_samples':sum(other.values()),'non_hanzi_labels':dict(other),'min_original_height':min(x[0] for x in dimensions),'max_original_height':max(x[0] for x in dimensions),'min_original_width':min(x[1] for x in dimensions),'max_original_width':max(x[1] for x in dimensions),'images_sha256':hashlib.sha256((target/'images.npy').read_bytes()).hexdigest()}
    (target/'metadata.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    return report,codes

def template(char,font,resolution):
    im=Image.new('L',(resolution,resolution),255);d=ImageDraw.Draw(im);b=d.textbbox((0,0),char,font=font)
    d.text(((resolution-b[2]+b[0])/2-b[0],(resolution-b[3]+b[1])/2-b[1]),char,font=font,fill=0)
    return np.asarray(im)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',default='data/casia/processed');p.add_argument('--workers',type=int,default=4);p.add_argument('--resolution',type=int,default=96);p.add_argument('--glyph-extent',type=int,default=80);p.add_argument('--seed',type=int,default=123);p.add_argument('--validation-writers',type=int,default=24);a=p.parse_args();started=time.time()
    out=ROOT/a.output;out.mkdir(parents=True,exist_ok=True);jobs=[];ids=[]
    for partition,name in [('official_train','HWDB1.1trn_gnt.zip'),('official_test','HWDB1.1tst_gnt.zip')]:
        archive=ROOT/'data/casia/raw'/name
        if partition=='official_train':
            archive=ROOT/'data/casia/raw/train_gnt'
            members=sorted(str(n.relative_to(archive)) for n in archive.rglob('*.gnt'))
            if not members:raise ValueError('Extract nested training ALZ archive first')
        else:
            with zipfile.ZipFile(archive) as z:members=sorted(n for n in z.namelist() if n.lower().endswith('.gnt'))
        for member in members:
            ids.append(re.match(r'(\d+)',Path(member).stem).group(1));jobs.append((str(archive),member,partition,a.resolution,a.glyph_extent,str(out)))
    if len(ids)!=len(set(ids)):raise ValueError('Writer IDs overlap between archives or multiple members share an ID')
    reports=[];all_codes=set();class_counts=Counter()
    with concurrent.futures.ProcessPoolExecutor(max_workers=a.workers) as pool:
        for i,(report,codes) in enumerate(pool.map(process_writer,jobs),1):
            reports.append(report);all_codes.update(codes);class_counts.update(codes)
            if i%10==0:print(f'Converted {i}/{len(jobs)} writers',flush=True)
    codepoints=sorted(all_codes);characters=[chr(n) for n in codepoints]
    np.save(out/'content_codepoints.npy',np.asarray(codepoints,dtype=np.int32),allow_pickle=False)
    font_path='/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc';font=ImageFont.truetype(font_path,int(a.resolution*.8),index=2)
    content=np.stack([template(c,font,a.resolution) for c in characters]);np.save(out/'content_images.npy',content,allow_pickle=False)
    train_ids=sorted(r['writer_id'] for r in reports if r['original_partition']=='official_train')
    test_ids=sorted(r['writer_id'] for r in reports if r['original_partition']=='official_test')
    rng=np.random.default_rng(a.seed);shuffled=list(rng.permutation(train_ids));val_ids=sorted(str(x) for x in shuffled[:a.validation_writers]);train_ids=sorted(set(train_ids)-set(val_ids))
    assert not (set(train_ids)&set(val_ids) or set(train_ids)&set(test_ids) or set(val_ids)&set(test_ids))
    for r in reports:r['split']='train' if r['writer_id'] in train_ids else 'validation' if r['writer_id'] in val_ids else 'test'
    summary={'dataset':'CASIA-HWDB1.1','resolution':a.resolution,'glyph_extent':a.glyph_extent,'font':font_path,'font_index':2,'seed':a.seed,'writers':reports,'splits':{'train':train_ids,'validation':val_ids,'test':test_ids},'sample_counts':{s:sum(r['hanzi_samples'] for r in reports if r['split']==s) for s in ['train','validation','test']},'total_gnt_records':sum(r['records'] for r in reports),'hanzi_samples':sum(r['hanzi_samples'] for r in reports),'non_hanzi_samples':sum(r['non_hanzi_samples'] for r in reports),'character_count':len(characters),'characters':characters,'per_class_samples':{chr(n):class_counts[n] for n in codepoints},'writer_disjoint':True,'processing_seconds':time.time()-started,'storage':'Per-writer uint8 images.npy and codepoints.npy; targets are memory-mapped, no raw image redistribution.'}
    (out/'dataset_manifest.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    report_dir=ROOT/'experiments/casia_001';report_dir.mkdir(exist_ok=True,parents=True)
    (report_dir/'dataset_manifest.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    # Input sanity gallery stays private/local; no real handwriting enters public GitHub.
    private=report_dir/'private';private.mkdir(exist_ok=True)
    selected=characters[:24];board=Image.new('RGB',(8*128,3*160),'white');d=ImageDraw.Draw(board);label_font=ImageFont.truetype('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',20,index=2)
    r=reports[0];imgs=np.load(out/'writers'/r['writer_id']/'images.npy',mmap_mode='r');codes=np.load(out/'writers'/r['writer_id']/'codepoints.npy');lookup={int(n):i for i,n in enumerate(codes)}
    for i,c in enumerate(selected):
        if ord(c) not in lookup:continue
        x=(i%8)*128;y=(i//8)*160;d.text((x+8,y+4),c,fill='black',font=label_font);board.paste(Image.fromarray(imgs[lookup[ord(c)]]).convert('RGB').resize((120,120)),(x+4,y+30))
    board.save(private/'input_sanity.png')
    print(json.dumps({k:summary[k] for k in ['sample_counts','character_count','hanzi_samples','total_gnt_records','non_hanzi_samples','writer_disjoint','processing_seconds']},ensure_ascii=False),flush=True)
