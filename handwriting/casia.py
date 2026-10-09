"""Memory-mapped CASIA dataset with same-writer, different-character references."""
import json
from bisect import bisect_right
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset

class CASIADataset(Dataset):
    def __init__(self,root,split,seed=123):
        self.root=Path(root);self.manifest=json.loads((self.root/'dataset_manifest.json').read_text())
        self.writer_ids=self.manifest['splits'][split];self.split=split
        self.writer_counts={r['writer_id']:r['hanzi_samples'] for r in self.manifest['writers']}
        self.cumulative=np.cumsum([0]+[self.writer_counts[w] for w in self.writer_ids]).tolist()
        self.content=np.load(self.root/'content_images.npy',mmap_mode='r')
        codes=np.load(self.root/'content_codepoints.npy')
        self.content_lookup={int(n):i for i,n in enumerate(codes)}
        self.cache={};self.seed=seed
    def __len__(self):return self.cumulative[-1]
    def writer_arrays(self,writer):
        if writer not in self.cache:
            self.cache[writer]=(np.load(self.root/'writers'/writer/'images.npy',mmap_mode='r'),np.load(self.root/'writers'/writer/'codepoints.npy',mmap_mode='r'))
        return self.cache[writer]
    @staticmethod
    def tensor(image):
        # Independent writable copy avoids mutating source arrays or read-only buffers.
        return torch.from_numpy(np.array(image,copy=True)).unsqueeze(0).repeat(3,1,1).float().div(127.5).sub(1)
    def __getitem__(self,index):
        wi=bisect_right(self.cumulative,index)-1;writer=self.writer_ids[wi];target=index-self.cumulative[wi]
        images,codes=self.writer_arrays(writer);n=len(codes)
        if n<2:raise ValueError('At least two samples required for reference sampling')
        rng=np.random if self.split=='train' else np.random.default_rng(self.seed+index)
        ref=int(rng.randint(n-1)) if self.split=='train' else int(rng.integers(n-1))
        if ref>=target:ref+=1
        while codes[ref]==codes[target]:
            ref=int(rng.randint(n)) if self.split=='train' else int(rng.integers(n))
        return {'target':self.tensor(images[target]),'reference':self.tensor(images[ref]),'content':self.tensor(self.content[self.content_lookup[int(codes[target])]]),'writer':writer,'target_codepoint':int(codes[target]),'reference_codepoint':int(codes[ref])}
