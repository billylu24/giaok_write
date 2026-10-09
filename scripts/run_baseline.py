"""Run the pinned official FontDiffuser on reproducible Chinese character cases."""
import argparse
import csv
import hashlib
import json
import os
import platform
import random
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'vendor'/'FontDiffuser'))
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms
from configs.fontdiffuser import get_parser
from src import (build_unet, build_style_encoder, build_content_encoder,
                 build_ddpm_scheduler, FontDiffuserModelDPM, FontDiffuserDPMPipeline)


def render_character(char, font, size):
    image=Image.new('RGB',(size,size),'white')
    d=ImageDraw.Draw(image)
    box=d.textbbox((0,0),char,font=font)
    d.text(((size-(box[2]-box[0]))/2-box[0],(size-(box[3]-box[1]))/2-box[1]),char,fill='black',font=font)
    return image


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--config',default='configs/baseline_001.json');ap.add_argument('--checkpoint-dir',default='checkpoints/fontdiffuser')
    options=ap.parse_args();cfg=json.loads((ROOT/options.config).read_text())
    out=ROOT/cfg['output_dir'];out.mkdir(parents=True,exist_ok=True)
    for sub in ['inputs/content','inputs/references','outputs']:(out/sub).mkdir(parents=True,exist_ok=True)
    (out/'config.json').write_text(json.dumps(cfg,ensure_ascii=False,indent=2)+'\n')
    args=get_parser().parse_args([])
    args.resolution=cfg['resolution'];args.style_image_size=(args.resolution,)*2;args.content_image_size=(args.resolution,)*2
    args.guidance_scale=cfg['guidance_scale'];device=torch.device(cfg['device'])
    if device.type=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA required by this experiment config')
    torch.set_num_threads(8);torch.backends.cudnn.benchmark=False
    random.seed(cfg['seed']);np.random.seed(cfg['seed']);torch.manual_seed(cfg['seed'])
    torch.cuda.set_device(device)
    torch.cuda.init()
    torch.empty(1,device=device)
    torch.cuda.reset_peak_memory_stats(device)
    environment={'utc_time':datetime.now(timezone.utc).isoformat(),'python':sys.version,'platform':platform.platform(),'torch':torch.__version__,'torch_cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(device),'command':sys.argv,'notes':[cfg.get('run_note','Research font-generation weights; not a CASIA-trained personal handwriting evaluation.')]}
    (out/'environment.json').write_text(json.dumps(environment,ensure_ascii=False,indent=2)+'\n')
    (out/'pip_freeze.txt').write_text(subprocess.run(['uv','pip','freeze','--python',sys.executable],capture_output=True,text=True).stdout)
    (out/'nvidia_smi.txt').write_text(subprocess.run(['nvidia-smi'],capture_output=True,text=True).stdout)
    print('Loading official models...',flush=True);start=time.perf_counter()
    components={};weight_checksums={}
    for name,builder in [('unet',build_unet),('style_encoder',build_style_encoder),('content_encoder',build_content_encoder)]:
        path=ROOT/options.checkpoint_dir/f'{name}.pth'
        weight_checksums[name]=hashlib.sha256(path.read_bytes()).hexdigest()
        model=builder(args);model.load_state_dict(torch.load(path,map_location='cpu',weights_only=True),strict=True);components[name]=model
    model=FontDiffuserModelDPM(**components).to(device).eval()
    pipe=FontDiffuserDPMPipeline(model,build_ddpm_scheduler(args),guidance_scale=cfg['guidance_scale'])
    load_seconds=time.perf_counter()-start
    tensorize=transforms.Compose([transforms.Resize((args.resolution,)*2,interpolation=transforms.InterpolationMode.BILINEAR),transforms.ToTensor(),transforms.Normalize([.5],[.5])])
    font=ImageFont.truetype(cfg['content_font'],int(args.resolution*.8),index=cfg['font_index'])
    content=[]
    for char in cfg['characters']:
        im=(Image.open(ROOT/cfg['content_images'][char]).convert('RGB').resize((args.resolution,)*2) if 'content_images' in cfg else render_character(char,font,args.resolution));im.save(out/f'inputs/content/U{ord(char):04X}.png');content.append(im)
    refs=[]
    for i,path in enumerate(cfg['references']):
        im=Image.open(ROOT/path).convert('RGB').resize((args.resolution,)*2);im.save(out/f'inputs/references/style_{i}.png');refs.append(im)
    def generate(ci,ri,seed):
        torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
        torch.cuda.synchronize(device);started=time.perf_counter()
        with torch.inference_mode():
            ims=pipe.generate(content_images=tensorize(ci).unsqueeze(0).to(device),style_images=tensorize(ri).unsqueeze(0).to(device),batch_size=1,order=2,num_inference_step=cfg['steps'],content_encoder_downsample_size=args.content_encoder_downsample_size,dm_size=(args.resolution,)*2,algorithm_type='dpmsolver++',skip_type='time_uniform',method='multistep')
        torch.cuda.synchronize(device);return ims[0],time.perf_counter()-started
    # Preserve the repository's exact input examples for an initial sanity case.
    official_content=Image.open(ROOT/'vendor/FontDiffuser/data_examples/sampling/example_content.jpg').convert('RGB')
    official_style=Image.open(ROOT/'vendor/FontDiffuser/data_examples/sampling/example_style.jpg').convert('RGB')
    official_content.save(out/'inputs/official_content.png');official_style.save(out/'inputs/official_style.png')
    official,official_seconds=generate(official_content,official_style,cfg['seed']);official.save(out/'outputs/official_example.png')
    records=[];images={};original=None
    with (out/'samples.jsonl').open('w') as log:
        for ci,char in enumerate(cfg['characters']):
            for si,ref in enumerate(refs):
                seed=cfg['seed']+ci
                image,seconds=generate(content[ci],ref,seed)
                relative=f'outputs/U{ord(char):04X}_style{si}_seed{seed}.png';image.save(out/relative);images[(ci,si)]=image
                if ci==0 and si==0:original=np.array(image)
                record={'character':char,'reference':cfg['references'][si],'style_index':si,'seed':seed,'seconds':seconds,'file':relative,'sha256':hashlib.sha256((out/relative).read_bytes()).hexdigest()}
                records.append(record);log.write(json.dumps(record,ensure_ascii=False)+'\n');log.flush();print(json.dumps(record,ensure_ascii=False),flush=True)
    repeat,repeat_seconds=generate(content[0],refs[0],cfg['seed']);repeat.save(out/'outputs/repeat_first.png')
    exact=bool(np.array_equal(original,np.array(repeat)))
    cell=144;label=40;margin=20;rows=len(content);cols=len(refs)+1
    board=Image.new('RGB',(margin*2+cols*cell,margin*2+label+cell+rows*(cell+label)), '#e9eef3');d=ImageDraw.Draw(board)
    label_font=ImageFont.truetype('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',20,index=2)
    d.text((margin,margin),'目标字形',font=label_font,fill='black')
    for si,im in enumerate(refs):
        x=margin+(si+1)*cell;d.text((x,margin),f'参考 {si+1}',font=label_font,fill='black');board.paste(im.resize((cell,cell)),(x,margin+label))
    for ci,char in enumerate(cfg['characters']):
        y=margin+label+cell+ci*(cell+label);d.text((margin,y),char,font=label_font,fill='black');board.paste(content[ci].resize((cell,cell)),(margin,y+label))
        for si in range(len(refs)):board.paste(images[(ci,si)].resize((cell,cell)),(margin+(si+1)*cell,y+label))
    board.save(out/'contact_sheet.png')
    with (out/'manual_review.csv').open('w',newline='') as f:
        fields=['character','style_index','file','content_correct','stroke_issue','style_similarity','notes'];w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for r in records:w.writerow({k:r[k] for k in fields if k in r})
    metrics={'status':'completed','kind':cfg.get('kind','official_pretrained_inference'),'characters':len(content),'references':len(refs),'generated_comparison_images':len(records),'official_example_images':1,'repeat_images':1,'total_generated_images':len(records)+2,'mean_sampling_seconds':float(np.mean([r['seconds'] for r in records])),'min_sampling_seconds':min(r['seconds'] for r in records),'max_sampling_seconds':max(r['seconds'] for r in records),'official_example_seconds':official_seconds,'repeat_seconds':repeat_seconds,'repeat_bitwise_equal':exact,'load_seconds':load_seconds,'peak_allocated_gpu_bytes':torch.cuda.max_memory_allocated(device),'peak_reserved_gpu_bytes':torch.cuda.max_memory_reserved(device),'weight_checksums':weight_checksums,'content_accuracy':None,'style_score':None,'trained_on_CASIA_here':False,'limitations':cfg.get('limitations',['Font examples are not user handwriting.','No statistically valid OCR/FID/HWD evaluation on this tiny set.','This run does not train a general foundation model.'])}
    (out/'metrics.json').write_text(json.dumps(metrics,ensure_ascii=False,indent=2)+'\n');print('COMPLETE',json.dumps(metrics,ensure_ascii=False),flush=True)

if __name__=='__main__':main()
