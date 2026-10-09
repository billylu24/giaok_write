"""Paired held-out character generation, with independent OCR calibration."""
import argparse,hashlib,json,os,sys,time,warnings
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'vendor/FontDiffuser'))
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import numpy as np
import torch
from PIL import Image,ImageDraw,ImageFont
from configs.fontdiffuser import get_parser
from src import FontDiffuserModelDPM,FontDiffuserDPMPipeline,build_unet,build_style_encoder,build_content_encoder,build_ddpm_scheduler
from handwriting.casia import CASIADataset
from rapidocr_onnxruntime import RapidOCR
warnings.filterwarnings('ignore',category=FutureWarning,module=r'(diffusers|src).*')

def aggregate(rows):
 correct=[r for r in rows if r['real_ocr_correct']]
 return {'samples':len(rows),'generated_ocr_exact':float(np.mean([r['generated_ocr_correct'] for r in rows])),'real_handwriting_ocr_exact':float(np.mean([r['real_ocr_correct'] for r in rows])),'generated_ocr_on_real_ocr_correct_subset':float(np.mean([r['generated_ocr_correct'] for r in correct])) if correct else None,'real_ocr_correct_subset_size':len(correct),'mean_sampling_seconds':float(np.mean([r['sampling_seconds'] for r in rows])),'mean_pixel_mae':float(np.mean([r['pixel_mae'] for r in rows])),'blank_outputs':sum(r['ink_fraction']<.003 for r in rows),'metric_note':'OCR is a generic recognition proxy; its real-handwriting accuracy limits interpretation. Pixel MAE is not content accuracy or an independent style score.'}

def main():
 p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--label',required=True);p.add_argument('--steps',type=int,default=20);p.add_argument('--guidance',type=float,default=7.5);a=p.parse_args()
 out=ROOT/'experiments/casia_001'/f'eval_{a.label}';out.mkdir(parents=True,exist_ok=True);(out/'generated').mkdir(exist_ok=True)
 private=ROOT/'experiments/casia_001/private'/f'eval_{a.label}';private.mkdir(parents=True,exist_ok=True)
 casefile=ROOT/'experiments/casia_001/evaluation_cases.json';cases=json.loads(casefile.read_text())['cases'];seed=json.loads(casefile.read_text())['seed']
 if (out/'samples.jsonl').exists():raise FileExistsError('Evaluation already contains samples; use a new label to avoid mixing results.')
 datasets={s:CASIADataset(ROOT/'data/casia/processed',s,seed) for s in ['validation','test']}
 torch.set_num_threads(8);torch.cuda.set_device(0);device=torch.device('cuda:0');torch.empty(1,device=device);torch.cuda.reset_peak_memory_stats(device)
 torch.backends.cudnn.benchmark=False;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 args=get_parser().parse_args([]);args.style_image_size=(96,96);args.content_image_size=(96,96);components={};hashes={}
 for name,builder in [('unet',build_unet),('style_encoder',build_style_encoder),('content_encoder',build_content_encoder)]:
  path=ROOT/a.checkpoint/f'{name}.pth';hashes[name]=hashlib.sha256(path.read_bytes()).hexdigest();m=builder(args);m.load_state_dict(torch.load(path,map_location='cpu',weights_only=True),strict=True);components[name]=m
 model=FontDiffuserModelDPM(**components).to(device).eval();pipe=FontDiffuserDPMPipeline(model,build_ddpm_scheduler(args),guidance_scale=a.guidance)
 ocr=RapidOCR(intra_op_num_threads=4,inter_op_num_threads=1)
 font=ImageFont.truetype('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',18,index=2)
 rows=[];first_output=None
 for i,case in enumerate(cases):
  ds=datasets[case['split']];sample=ds[case['dataset_index']]
  if 'reference_index' in case:
   images,codes=ds.writer_arrays(case['writer']);ri=case['reference_index'];sample['reference']=ds.tensor(images[ri]);sample['reference_codepoint']=int(codes[ri])
  assert case['writer']==sample['writer'] and case['target']==chr(sample['target_codepoint']) and case['reference']==chr(sample['reference_codepoint'])
  assert case['target']!=case['reference']
  torch.manual_seed(case['seed']);torch.cuda.manual_seed_all(case['seed']);torch.cuda.synchronize();started=time.perf_counter()
  with torch.inference_mode():
   image=pipe.generate(sample['content'].unsqueeze(0).to(device),sample['reference'].unsqueeze(0).to(device),1,2,a.steps,args.content_encoder_downsample_size,dm_size=(96,96),skip_type='time_uniform')[0]
  torch.cuda.synchronize();seconds=time.perf_counter()-started
  path=out/'generated'/f"{case['case_id']}.png";image.save(path)
  target=((sample['target'][0].numpy()+1)*127.5).round().astype(np.uint8);ref=((sample['reference'][0].numpy()+1)*127.5).round().astype(np.uint8);content=((sample['content'][0].numpy()+1)*127.5).round().astype(np.uint8)
  generated=np.asarray(image)
  recs,_=ocr.text_rec([np.repeat(target[:,:,None],3,axis=2),generated[:,:,::-1].copy()]);real_text,real_score=recs[0];generated_text,generated_score=recs[1]
  row={**case,'file':str(path.relative_to(out)),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'sampling_seconds':seconds,'real_ocr_text':real_text,'real_ocr_confidence':float(real_score),'real_ocr_correct':real_text==case['target'],'generated_ocr_text':generated_text,'generated_ocr_confidence':float(generated_score),'generated_ocr_correct':generated_text==case['target'],'pixel_mae':float(np.abs(generated.astype(float)-target[:,:,None].astype(float)).mean()/255),'ink_fraction':float((generated.mean(axis=2)<200).mean())}
  rows.append(row)
  with (out/'samples.jsonl').open('a') as f:f.write(json.dumps(row,ensure_ascii=False)+'\n')
  # Side-by-side real input/target gallery is local only.
  local=Image.new('RGB',(4*144,180),'#edf2f7');d=ImageDraw.Draw(local)
  for col,(title,im) in enumerate([('标准模板',Image.fromarray(content).convert('RGB')),('真实参考',Image.fromarray(ref).convert('RGB')),('真实目标',Image.fromarray(target).convert('RGB')),('模型生成',image)]):
   d.text((col*144+4,4),title,font=font,fill='black');local.paste(im.resize((144,144)),(col*144,32))
  local.save(private/f"{case['case_id']}.png")
  if i%10==0:print(json.dumps({'completed':i+1,'total':len(cases),'target':case['target'],'prediction':generated_text,'gt_prediction':real_text},ensure_ascii=False),flush=True)
 summary={'status':'completed','label':a.label,'checkpoint':a.checkpoint,'checkpoint_sha256':hashes,'torch':torch.__version__,'inference_steps':a.steps,'guidance_scale':a.guidance,'validation':aggregate([r for r in rows if r['split']=='validation']),'test_random':aggregate([r for r in rows if r['split']=='test' and r['kind']=='random_characters']),'test_gallery':aggregate([r for r in rows if r['kind']=='gallery_characters']),'peak_allocated_gpu_bytes':torch.cuda.max_memory_allocated(),'OCR':'rapidocr-onnxruntime 1.4.4, bundled Chinese PP-OCRv4 recognizer, recognition-only CPU','data_publishing':'Real reference and real target images are private/local; generated images and numerical records are public.'}
 (out/'metrics.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n');print('COMPLETE '+json.dumps(summary,ensure_ascii=False),flush=True)
 gallery=[r for r in rows if r['kind']=='gallery_characters'];board=Image.new('RGB',(4*240,6*280),'#edf2f7');d=ImageDraw.Draw(board)
 for i,r in enumerate(gallery):
  x=(i%4)*240;y=(i//4)*280;d.text((x+4,y+4),f"{r['target']} / writer {r['writer']}",font=font,fill='black');board.paste(Image.open(out/r['file']).resize((224,224)),(x+8,y+32))
 board.save(out/'generated_gallery.png')

if __name__=='__main__':main()
