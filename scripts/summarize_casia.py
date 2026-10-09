"""Build paired CASIA figures and report honest before/after measurements."""
import argparse,csv,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image,ImageDraw,ImageFont
ROOT=Path(__file__).resolve().parents[1]


def summarize():
 out=ROOT/'experiments/casia_001'
 train=[json.loads(s) for s in (out/'train_steps.jsonl').read_text().splitlines()]
 val=[json.loads(s) for s in (out/'validation.jsonl').read_text().splitlines()]
 with (out/'train_steps.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(train[0]),lineterminator="\n");w.writeheader();w.writerows(train)
 fig,axes=plt.subplots(1,2,figsize=(12,4.5),layout='constrained')
 steps=np.array([r['step'] for r in train]);window=min(100,len(train))
 for key,label,color in [('total','Total training loss','#2563eb'),('diffusion','Diffusion MSE','#059669')]:
  values=np.array([r[key] for r in train]);axes[0].plot(steps[window-1:],np.convolve(values,np.ones(window)/window,mode='valid'),label=label,color=color)
 axes[0].set(xlabel='Optimizer step',ylabel='Loss',title=f'{window}-step training mean');axes[0].legend();axes[0].grid(alpha=.2)
 axes[1].plot([r['step'] for r in val],[r['total'] for r in val],marker='o',label='Validation total')
 axes[1].plot([r['step'] for r in val],[r['diffusion'] for r in val],marker='o',label='Validation diffusion')
 axes[1].set(xlabel='Optimizer step',ylabel='Loss',title='Fixed held-out-writer denoising validation');axes[1].legend();axes[1].grid(alpha=.2)
 fig.savefig(out/'loss_curves.png',dpi=180);plt.close(fig)
 before=json.loads((out/'eval_before/metrics.json').read_text());after=json.loads((out/'eval_after/metrics.json').read_text())
 b={r['case_id']:r for r in map(json.loads,(out/'eval_before/samples.jsonl').read_text().splitlines())}
 a={r['case_id']:r for r in map(json.loads,(out/'eval_after/samples.jsonl').read_text().splitlines())}
 assert b.keys()==a.keys()
 # Publish all fixed random cases, avoiding selection of only attractive examples.
 font=ImageFont.truetype('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',16,index=2)
 for folder,records in [('eval_before',b),('eval_after',a)]:
  for split in ['validation','test']:
   selected=[r for r in records.values() if r['split']==split and r['kind']=='random_characters']
   canvas=Image.new('RGB',(8*128,math_ceil(len(selected)/8)*160),'#edf2f7');draw=ImageDraw.Draw(canvas)
   for i,r in enumerate(selected):
    x=(i%8)*128;y=(i//8)*160
    draw.text((x+4,y+4),f"{r['target']} → {r['generated_ocr_text']}",font=font,fill='black')
    with Image.open(out/folder/r['file']) as generated:canvas.paste(generated.resize((120,120)),(x+4,y+32))
   canvas.save(out/folder/f'random_{split}_gallery.png')
 paired=[]
 for case_id in b:
  x,y=b[case_id],a[case_id]
  for k in ['writer','target','reference','seed']:assert x[k]==y[k]
  paired.append({'case_id':case_id,'split':x['split'],'kind':x['kind'],'target':x['target'],'writer':x['writer'],'before_correct':x['generated_ocr_correct'],'after_correct':y['generated_ocr_correct'],'real_ocr_correct':x['real_ocr_correct'],'before_pixel_mae':x['pixel_mae'],'after_pixel_mae':y['pixel_mae'],'before_text':x['generated_ocr_text'],'after_text':y['generated_ocr_text']})
 (out/'paired_results.json').write_text(json.dumps(paired,ensure_ascii=False,indent=2)+'\n')
 with (out/'paired_results.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(paired[0]),lineterminator="\n");w.writeheader();w.writerows(paired)
 groups={}
 for group in ['validation','test_random','test_gallery']:
  if group=='validation':subset=[r for r in paired if r['split']=='validation']
  elif group=='test_random':subset=[r for r in paired if r['split']=='test' and r['kind']=='random_characters']
  else:subset=[r for r in paired if r['kind']=='gallery_characters']
  improved=sum(not r['before_correct'] and r['after_correct'] for r in subset);regressed=sum(r['before_correct'] and not r['after_correct'] for r in subset)
  groups[group]={'before':before[group],'after':after[group],'paired_correctness_improvements':improved,'paired_correctness_regressions':regressed}
 gallery=[r for r in paired if r['kind']=='gallery_characters'];font=ImageFont.truetype('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',18,index=2)
 rows=len(gallery);board=Image.new('RGB',(4*240,math_ceil(rows/2)*280),'#edf2f7');d=ImageDraw.Draw(board)
 for i,r in enumerate(gallery):
  base_x=(i%2)*480;base_y=(i//2)*280
  for j,(label,folder) in enumerate([('训练前','eval_before'),('训练后','eval_after')]):
   x=base_x+j*240;d.text((x+8,base_y+6),f"{r['target']} / {r['writer']} / {label}",font=font,fill='black')
   records=b if folder=='eval_before' else a
   image=Image.open(out/folder/records[r['case_id']]['file']);board.paste(image.resize((224,224)),(x+8,base_y+32))
 board.save(out/'before_after_gallery.png')
 # Compact generated-only comparison: one common reference per writer, seven targets.
 writers=list(dict.fromkeys(r['writer'] for r in gallery));characters=list(dict.fromkeys(r['target'] for r in gallery))
 compact=Image.new('RGB',(180+144*len(characters),44+176*2*len(writers)),'#edf2f7');cd=ImageDraw.Draw(compact)
 for j,char in enumerate(characters):cd.text((180+j*144+60,10),char,font=font,fill='black')
 for wi,writer in enumerate(writers):
  for phase,(title,folder,records) in enumerate([('训练前','eval_before',b),('训练后','eval_after',a)]):
   y=44+(wi*2+phase)*176;cd.text((8,y+40),f'{writer} / {title}',font=font,fill='black')
   for j,char in enumerate(characters):
    r=next(r for r in gallery if r['writer']==writer and r['target']==char)
    with Image.open(out/folder/records[r['case_id']]['file']) as generated:
     compact.paste(generated.resize((136,136)),(184+j*144,y+8))
 compact.save(out/'before_after_overview.png')
 summary=json.loads((out/'training_summary.json').read_text())
 review=[{'case_id':r['case_id'],'target':r['target'],'before_file':f"eval_before/{b[r['case_id']]['file']}",'after_file':f"eval_after/{a[r['case_id']]['file']}",'human_content_correct':'','stroke_issue':'','style_observation':''} for r in paired]
 if not (out/'manual_review.csv').exists():
  with (out/'manual_review.csv').open('w',newline='') as f:
   w=csv.DictWriter(f,fieldnames=list(review[0]),lineterminator="\n");w.writeheader();w.writerows(review)
 combined={'training':summary,'comparison':groups,'evaluation_cases':len(paired),'decision_policy':'Checkpoint selected on fixed validation denoising loss; test does not select checkpoint.','limitations':['Short public-data training, not full convergence.','Generic OCR has errors on real handwriting.','Pixel similarity is not independent style verification.','No writer-style classifier or formal FID/HWD benchmark.']}
 if (out/'reference_sensitivity/metrics.json').exists():
  combined['reference_sensitivity']=json.loads((out/'reference_sensitivity/metrics.json').read_text())
 (out/'results.json').write_text(json.dumps(combined,ensure_ascii=False,indent=2)+'\n')
 print(json.dumps(combined,ensure_ascii=False))

def math_ceil(x):return int(np.ceil(x))

if __name__=='__main__':summarize()
