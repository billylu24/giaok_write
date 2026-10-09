"""Freeze paired evaluation cases before any CASIA optimizer updates."""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from handwriting.casia import CASIADataset

if __name__=='__main__':
 root=ROOT/'data/casia/processed';seed=123;cases=[]
 for split,n in [('validation',32),('test',64)]:
  ds=CASIADataset(root,split,seed);rng=np.random.default_rng(seed+(1 if split=='validation' else 2))
  for i,index in enumerate(rng.choice(len(ds),n,replace=False)):
   sample=ds[int(index)];cases.append({'case_id':f'{split}_random_{i:03d}','split':split,'kind':'random_characters','dataset_index':int(index),'writer':sample['writer'],'target':chr(sample['target_codepoint']),'reference':chr(sample['reference_codepoint']),'seed':20000+i,'reference_rule':'same writer, different character; deterministic from dataset seed'})
 ds=CASIADataset(root,'test',seed)
 for wi,writer in enumerate(ds.writer_ids[:3]):
  images,codes=ds.writer_arrays(writer);lookup={int(n):i for i,n in enumerate(codes)}
  for ci,char in enumerate('湖海明好我清永鑫'):
   if ord(char) not in lookup:continue
   index=ds.cumulative[wi]+lookup[ord(char)];sample=ds[index]
   reference_index=lookup.get(ord('人'),lookup.get(ord('的')))
   if reference_index is None:raise ValueError('Gallery reference character missing')
   cases.append({'case_id':f'test_gallery_{wi}_{ord(char):04X}','split':'test','kind':'gallery_characters','dataset_index':index,'reference_index':reference_index,'writer':writer,'target':char,'reference':chr(int(codes[reference_index])),'seed':30000+ci,'reference_rule':'same writer, different character; deterministic from dataset seed'})
 out=ROOT/'experiments/casia_001';out.mkdir(exist_ok=True,parents=True)
 (out/'evaluation_cases.json').write_text(json.dumps({'seed':seed,'policy':'Cases fixed before CASIA training; validation picks checkpoint. Test does not tune training.','cases':cases},ensure_ascii=False,indent=2)+'\n')
 print('Frozen evaluation cases:',len(cases))
