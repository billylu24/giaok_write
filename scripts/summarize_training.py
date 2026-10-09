"""Export true per-step TensorBoard losses and a training-only summary."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

ROOT=Path(__file__).resolve().parents[1]

def main():
    p=argparse.ArgumentParser();p.add_argument('--experiment',default='experiments/training_demo_001');args=p.parse_args()
    out=ROOT/args.experiment
    events=sorted((out/'logs').rglob('events.out.tfevents.*'))
    if len(events)!=1:raise RuntimeError('Expected exactly one event file for this isolated run')
    ea=EventAccumulator(str(events[0]));ea.Reload()
    rows=[{'step':e.step,'loss':e.value,'wall_time':e.wall_time} for e in ea.Scalars('train_loss')]
    (out/'all_steps_loss.json').write_text(json.dumps(rows,indent=2)+'\n')
    with (out/'loss_curve.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['step','loss','wall_time']);w.writeheader();w.writerows(rows)
    loss=np.array([r['loss'] for r in rows]);steps=np.array([r['step'] for r in rows]);window=min(50,len(loss))
    fig,ax=plt.subplots(figsize=(9,4.5),layout='constrained')
    ax.plot(steps,loss,color='#94a3b8',alpha=.45,lw=.6,label='Per-step training loss')
    ax.plot(steps[window-1:],np.convolve(loss,np.ones(window)/window,mode='valid'),color='#2563eb',lw=2,label=f'{window}-step mean')
    ax.set(xlabel='Optimization step',ylabel='Training loss',title='FontDiffuser: 12-example training demo (not a held-out evaluation)')
    ax.grid(alpha=.2);ax.legend();fig.savefig(out/'loss_curve.png',dpi=180);plt.close(fig)
    ckpt=out/'global_step_1000'
    hashes={name:hashlib.sha256((ckpt/f'{name}.pth').read_bytes()).hexdigest() for name in ['unet','style_encoder','content_encoder']}
    metrics={'status':'completed','kind':'random_initialization_training_demo','initialization':'random; official pretrained weights NOT loaded','steps':len(rows),'styles':3,'characters':['氮','潮','舶','镀'],'target_images':12,'batch_size':1,'learning_rate':0.0001,'mixed_precision':'no','seed':123,'loss_first50_mean':float(loss[:50].mean()),'loss_last50_mean':float(loss[-50:].mean()),'last_step_loss':float(loss[-1]),'logged_training_wall_seconds':rows[-1]['wall_time']-rows[0]['wall_time'],'all_losses_finite':bool(np.isfinite(loss).all()),'checkpoint':'global_step_1000','checkpoint_sha256':hashes,'held_out_evaluation':False,'CASIA_training':False}
    (out/'metrics.json').write_text(json.dumps(metrics,ensure_ascii=False,indent=2)+'\n');print(json.dumps(metrics,ensure_ascii=False))

if __name__=='__main__':main()
