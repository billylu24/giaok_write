import io,struct,unittest
import numpy as np
from handwriting.gnt import read_records,normalize_image

class GNTTests(unittest.TestCase):
    def test_gb2312_byte_order_and_pixels(self):
        data=bytes([0,255,127,80,255,0]);record=struct.pack('<I2sHH',16,'湖'.encode('gb2312'),3,2)+data
        char,image,offset=next(read_records(io.BytesIO(record)))
        self.assertEqual(char,'湖');self.assertEqual(offset,0);self.assertEqual(image.tolist(),[[0,255,127],[80,255,0]])
    def test_truncation_is_not_silently_dropped(self):
        with self.assertRaises(ValueError):list(read_records(io.BytesIO(b'\x01\x00')))
        record=struct.pack('<I2sHH',16,'海'.encode('gb2312'),3,2)+b'\x00'
        with self.assertRaises(ValueError):list(read_records(io.BytesIO(record)))
    def test_invalid_size_is_rejected(self):
        record=struct.pack('<I2sHH',99,'明'.encode('gb2312'),1,1)+b'\x00'
        with self.assertRaises(ValueError):list(read_records(io.BytesIO(record)))
    def test_aspect_ratio_and_background(self):
        normalized=normalize_image(np.zeros((10,20),dtype=np.uint8),96,80)
        self.assertEqual(normalized.shape,(96,96));self.assertTrue(np.all(normalized[:28]==255));self.assertTrue(np.all(normalized[28:68,8:88]==0))

if __name__=='__main__':unittest.main()
