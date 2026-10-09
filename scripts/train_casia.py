"""Continue the shared FontDiffuser model on public CASIA writers, with held-out validation."""
import argparse,hashlib,json,math,os,platform,random,sys,time,warnings
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'vendor/FontDiffuser'))
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader,Subset
from configs.fontdiffuser import get_parser
from src import FontDiffuserModel,ContentPerceptualLoss,build_unet,build_style_encoder,build_content_encoder,build_ddpm_scheduler
from handwriting.casia import CASIADataset
warnings.filterwarnings('ignore',category=FutureWarning,module=r'(diffusers|src).*')

def sha256(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()

def save_weights(model,path):
 path.mkdir(parents=True,exist_ok=True);hashes={}
 for name in ['unet','style_encoder','content_encoder']:
  module=getattr(model,name);dest=path/f'{name}.pth';part=dest.with_suffix('.pth.part')
  torch.save(module.state_dict(),part);part.replace(dest);hashes[name]=sha256(dest)
 (path/'manifest.json').write_text(json.dumps(hashes,indent=2)+'\n');return hashes

def seed_worker(worker_id):
 seed=torch.initial_seed()%2**32;np.random.seed(seed);random.seed(seed)

def main():
 p=argparse.ArgumentParser();p.add_argument('--config',default='configs/casia_001.json');p.add_argument('--resume');a=p.parse_args()
 cfg=json.loads((ROOT/a.config).read_text());out=ROOT/cfg['output_dir'];out.mkdir(parents=True,exist_ok=True)
 if not a.resume and ((out/'train_steps.jsonl').exists() or (out/'validation.jsonl').exists()):
  raise FileExistsError('Training logs already exist; use a new output_dir or --resume to avoid mixing runs.')
 (out/'train_config.json').write_text(json.dumps(cfg,ensure_ascii=False,indent=2)+'\n')
 torch.set_num_threads(8);torch.cuda.set_device(0);device=torch.device('cuda:0')
 torch.backends.cudnn.benchmark=False;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 random.seed(cfg['seed']);np.random.seed(cfg['seed']);torch.manual_seed(cfg['seed']);torch.cuda.manual_seed_all(cfg['seed'])
 torch.empty(1,device=device)
 torch.cuda.reset_peak_memory_stats(device)
 args=get_parser().parse_args([]);args.resolution=cfg['resolution'];args.style_image_size=(args.resolution,)*2;args.content_image_size=(args.resolution,)*2
 components={};source={}
 for name,builder in [('unet',build_unet),('style_encoder',build_style_encoder),('content_encoder',build_content_encoder)]:
  path=ROOT/cfg['initial_checkpoint']/f'{name}.pth';source[name]=sha256(path)
  module=builder(args);module.load_state_dict(torch.load(path,map_location='cpu',weights_only=True),strict=True);components[name]=module
 model=FontDiffuserModel(**components).to(device)
 scheduler=build_ddpm_scheduler(args);alphas=scheduler.alphas_cumprod.to(device)
 perceptual=ContentPerceptualLoss().to(device).eval();perceptual.requires_grad_(False)
 train=CASIADataset(ROOT/cfg['dataset_dir'],'train',cfg['seed']);val=CASIADataset(ROOT/cfg['dataset_dir'],'validation',cfg['seed'])
 generator=torch.Generator().manual_seed(cfg['seed'])
 loader=DataLoader(train,batch_size=cfg['batch_size'],shuffle=True,num_workers=cfg['num_workers'],pin_memory=True,drop_last=True,worker_init_fn=seed_worker,generator=generator,persistent_workers=cfg['num_workers']>0)
 validation_indices=np.random.default_rng(cfg['seed']).choice(len(val),cfg['validation_batches']*cfg['batch_size'],replace=False).tolist()
 (out/'validation_indices.json').write_text(json.dumps(validation_indices)+'\n')
 val_loader=DataLoader(Subset(val,validation_indices),batch_size=cfg['batch_size'],num_workers=0)
 optimizer=torch.optim.AdamW(model.parameters(),lr=cfg['learning_rate'],betas=(.9,.999),weight_decay=cfg['weight_decay'],eps=1e-8)
 def lr_multiplier(n):
  if n<cfg['warmup_steps']:return max((n+1)/cfg['warmup_steps'],.01)
  progress=(n-cfg['warmup_steps'])/max(1,cfg['max_steps']-cfg['warmup_steps'])
  return .1+.9*.5*(1+math.cos(math.pi*min(1,progress)))
 lr_schedule=torch.optim.lr_scheduler.LambdaLR(optimizer,lr_multiplier)
 begin_step=0
 if a.resume:
  checkpoint=torch.load(ROOT/a.resume,map_location='cpu',weights_only=False)
  model.load_state_dict(checkpoint['model'],strict=True);optimizer.load_state_dict(checkpoint['optimizer']);lr_schedule.load_state_dict(checkpoint['lr_scheduler']);begin_step=checkpoint['step']
  random.setstate(checkpoint['python_rng']);np.random.set_state(checkpoint['numpy_rng']);torch.set_rng_state(checkpoint['torch_rng']);torch.cuda.set_rng_state_all(checkpoint['cuda_rng'])
  generator.set_state(checkpoint['loader_generator_rng'])
  # Loader resumes with a new shuffle, not an identical mid-epoch cursor; report it.
 mean=torch.tensor([.485,.456,.406],device=device)[None,:,None,None];std=torch.tensor([.229,.224,.225],device=device)[None,:,None,None]
 environment={'utc':datetime.now(timezone.utc).isoformat(),'python':sys.version,'platform':platform.platform(),'torch':torch.__version__,'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(0),'initial_checkpoint_sha256':source,'samples':len(train),'train_writers':len(train.writer_ids),'validation_samples':len(validation_indices),'validation_writers':len(val.writer_ids),'optimizer_parameters':sum(p.numel() for p in model.parameters()),'resume':a.resume,'source_commit':'7b28ce9c3b357f4fb23296622f458cf169803539','resume_note':'When resumed, random generator states are restored but training starts a new DataLoader shuffle.'}
 (out/'train_environment.json').write_text(json.dumps(environment,ensure_ascii=False,indent=2)+'\n')
 def components_loss(batch,noise,timesteps,drop=False):
  target=batch['target'].to(device,non_blocking=True);ref=batch['reference'].to(device,non_blocking=True);content=batch['content'].to(device,non_blocking=True)
  if drop:
   mask=torch.rand(len(target),device=device)<cfg['drop_probability'];ref=ref.clone();content=content.clone();ref[mask]=1;content[mask]=1
  noisy=scheduler.add_noise(target,noise,timesteps)
  prediction,offset=model(x_t=noisy,timesteps=timesteps,style_images=ref,content_images=content,content_encoder_downsample_size=args.content_encoder_downsample_size)
  diffusion=F.mse_loss(prediction.float(),noise.float())
  alpha=alphas[timesteps].view(-1,1,1,1)
  x0=((noisy-(1-alpha).sqrt()*prediction)/alpha.sqrt()).clamp(-1,1)
  generated=(x0+1)/2;real=(target+1)/2
  perception=perceptual.calculate_loss((generated-mean)/std,(real-mean)/std,device)
  offset_loss=offset/2;loss=diffusion+cfg['perceptual_coefficient']*perception+cfg['offset_coefficient']*offset_loss
  return loss,{'total':float(loss.detach()),'diffusion':float(diffusion.detach()),'perceptual':float(perception.detach()),'offset':float(offset_loss.detach())}
 def validate(step):
  model.eval();entries=[]
  with torch.random.fork_rng(devices=[0]):
   torch.manual_seed(cfg['seed']+9000);torch.cuda.manual_seed_all(cfg['seed']+9000)
   with torch.no_grad():
    for batch in val_loader:
     n=len(batch['target']);noise=torch.randn((n,3,args.resolution,args.resolution),device=device);t=torch.randint(0,1000,(n,),device=device)
     loss,parts=components_loss(batch,noise,t);entries.append(parts)
  record={'step':step,**{k:float(np.mean([r[k] for r in entries])) for k in entries[0]}}
  with (out/'validation.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
  print('VALIDATION '+json.dumps(record),flush=True);model.train();return record
 baseline=validate(begin_step);best=baseline['total'];best_step=begin_step
 started=time.perf_counter();iterator=iter(loader);epoch=0;interval=[];all_steps=begin_step
 log=(out/'train_steps.jsonl').open('a',buffering=1)
 try:
  for step in range(begin_step+1,cfg['max_steps']+1):
   try:batch=next(iterator)
   except StopIteration:epoch+=1;iterator=iter(loader);batch=next(iterator)
   model.train();optimizer.zero_grad(set_to_none=True)
   n=len(batch['target']);noise=torch.randn((n,3,args.resolution,args.resolution),device=device);t=torch.randint(0,1000,(n,),device=device)
   loss,parts=components_loss(batch,noise,t,drop=True)
   if not torch.isfinite(loss):raise FloatingPointError(f'Non-finite loss at {step}')
   loss.backward();grad=torch.nn.utils.clip_grad_norm_(model.parameters(),cfg['gradient_clip'])
   if not torch.isfinite(grad):raise FloatingPointError(f'Non-finite gradient at {step}')
   optimizer.step();lr_schedule.step();all_steps=step
   record={'step':step,**parts,'gradient_norm':float(grad),'learning_rate':optimizer.param_groups[0]['lr'],'elapsed_seconds':time.perf_counter()-started,'epoch_completed':epoch}
   log.write(json.dumps(record)+'\n');interval.append(record)
   if step%cfg['log_interval']==0:
    view={'step':step,'total':float(np.mean([r['total'] for r in interval])),'diffusion':float(np.mean([r['diffusion'] for r in interval])),'seconds':record['elapsed_seconds'],'lr':record['learning_rate']}
    print('TRAIN '+json.dumps(view),flush=True);interval=[]
   if step%cfg['validation_interval']==0 or step==cfg['max_steps']:
    val_record=validate(step)
    if val_record['total']<best:best=val_record['total'];best_step=step;save_weights(model,out/'checkpoints/best')
   if step%cfg['checkpoint_interval']==0 or step==cfg['max_steps']:
    hashes=save_weights(model,out/f'checkpoints/step_{step}')
    checkpoint={'model':model.state_dict(),'optimizer':optimizer.state_dict(),'lr_scheduler':lr_schedule.state_dict(),'step':step,'python_rng':random.getstate(),'numpy_rng':np.random.get_state(),'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all(),'loader_generator_rng':generator.get_state()}
    part=out/'checkpoints/resume_latest.pth.part';torch.save(checkpoint,part);part.replace(out/'checkpoints/resume_latest.pth')
    print('CHECKPOINT '+str(step),flush=True)
 finally:log.close()
 torch.cuda.synchronize();summary={'status':'completed','steps':all_steps,'max_steps':cfg['max_steps'],'training_seconds':time.perf_counter()-started,'best_validation_step':best_step,'baseline_validation':baseline,'best_validation_loss':best,'final_validation':val_record,'final_checkpoint':f'checkpoints/step_{all_steps}','checkpoint_sha256':hashes,'optimizer_examples':(all_steps-begin_step)*cfg['batch_size'],'train_dataset_samples':len(train),'epochs_completed':epoch,'peak_allocated_gpu_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_gpu_bytes':torch.cuda.max_memory_reserved(),'initialization':'Public pretrained FontDiffuser; shared CASIA training, not user adaptation','full_convergence_claimed':False}
 (out/'training_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n');print('COMPLETE '+json.dumps(summary),flush=True)

if __name__=='__main__':main()
