import json,tempfile,unittest
from pathlib import Path
import numpy as np
from handwriting.casia import CASIADataset

class PairingTests(unittest.TestCase):
    def test_duplicate_labels_cannot_leak_reference_content(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);(root/'writers/001').mkdir(parents=True)
            codes=np.array([ord('湖'),ord('湖'),ord('海')],dtype=np.int32)
            np.save(root/'writers/001/codepoints.npy',codes)
            np.save(root/'writers/001/images.npy',np.full((3,96,96),255,dtype=np.uint8))
            np.save(root/'content_codepoints.npy',np.array([ord('湖'),ord('海')],dtype=np.int32))
            np.save(root/'content_images.npy',np.full((2,96,96),255,dtype=np.uint8))
            manifest={'splits':{'train':['001'],'validation':['001'],'test':['001']},'writers':[{'writer_id':'001','hanzi_samples':3}]}
            (root/'dataset_manifest.json').write_text(json.dumps(manifest))
            for split in ['train','validation','test']:
                ds=CASIADataset(root,split,123)
                for index in range(3):
                    for repeat in range(10):
                        sample=ds[index]
                        self.assertNotEqual(sample['target_codepoint'],sample['reference_codepoint'])
                        self.assertEqual(tuple(sample['target'].shape),(3,96,96))
                if split!='train':
                    self.assertEqual(ds[0]['reference_codepoint'],ds[0]['reference_codepoint'])

if __name__=='__main__':unittest.main()
