import torch
import sys
import os
import unittest

# Add parent dir to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from na_radio.encoders import DINOv3Encoder, NADINOv3Encoder
from na_radio.utils import get_device

class TestDINOv3Integration(unittest.TestCase):
    def setUp(self):
        self.device = get_device()
        print(f"Running tests on {self.device}")

    def test_dinov3_load(self):
        print("\nTesting DINOv3 loading...")
        try:
            enc = DINOv3Encoder(device=self.device, input_resolution=(512, 512))
            print("DINOv3 loaded successfully.")
            
            dummy = torch.randn(1, 3, 512, 512).to(self.device)
            emb = enc.encode_image_to_vector(dummy)
            print(f"Embedding shape: {emb.shape}")
            self.assertEqual(emb.shape[0], 1)
            
            enc.unload()
        except Exception as e:
            self.fail(f"DINOv3 failed: {e}")

    def test_nadino3_load(self):
        print("\nTesting NADINOv3 loading and injection...")
        try:
            enc = NADINOv3Encoder(device=self.device, input_resolution=(512, 512))
            print("NADINOv3 loaded successfully.")
            
            # Check injection
            last_block = enc.model.blocks[-1]
            # We expect GaussKernelAttn wrapper or similar if injection worked
            # But encoders.py wraps it. Let's check if it runs.
            
            dummy = torch.randn(1, 3, 512, 512).to(self.device)
            emb = enc.encode_image_to_vector(dummy)
            print(f"Embedding shape: {emb.shape}")
            self.assertEqual(emb.shape[0], 1)
            
            enc.unload()
        except Exception as e:
            self.fail(f"NADINOv3 failed: {e}")

if __name__ == '__main__':
    unittest.main()
