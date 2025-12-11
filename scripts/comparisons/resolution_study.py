import argparse
import time
import torch
import numpy as np
import sys
import os

# Add parent dir to path to import encoders
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from encoders import NARadioEncoder, DINOv2Encoder, NADINOv2Encoder, DINOv3Encoder, NADINOv3Encoder
from utils import get_device

def evaluate_resolution(encoder_name, resolutions, device=None):
    if device is None:
        device = get_device()
    
    print(f"Evaluating {encoder_name} on {device}...")
    
    results = []
    
    for res in resolutions:
        H, W = res
        print(f"Testing resolution: {H}x{W}")
        
        try:
            if encoder_name == 'radio':
                enc = NARadioEncoder(device=device, input_resolution=(H, W), return_radio_features=False)
            elif encoder_name == 'radio_no_naclip':
                enc = NARadioEncoder(device=device, input_resolution=(H, W), return_radio_features=False, use_naclip=False)
            elif encoder_name == 'dinov2':
                enc = DINOv2Encoder(device=device, input_resolution=(H, W))
            elif encoder_name == 'nadino':
                enc = NADINOv2Encoder(device=device, input_resolution=(H, W))
            elif encoder_name == 'dinov3':
                enc = DINOv3Encoder(device=device, input_resolution=(H, W))
            elif encoder_name == 'nadino3':
                enc = NADINOv3Encoder(device=device, input_resolution=(H, W))
            else:
                print(f"Unknown encoder: {encoder_name}")
                return

            # Create dummy input
            dummy_input = torch.randn(1, 3, H, W).to(device)
            
            # Warmup
            for _ in range(5):
                _ = enc.encode_image_to_vector(dummy_input)
                
            # Measure time
            start_time = time.time()
            n_iters = 20
            for _ in range(n_iters):
                _ = enc.encode_image_to_vector(dummy_input)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            end_time = time.time()
            avg_time = (end_time - start_time) / n_iters
            
            # Measure patch size / feature map size
            feat_map = enc.encode_image_to_feat_map(dummy_input)
            # feat_map shape: (B, C, H_feat, W_feat)
            _, _, H_feat, W_feat = feat_map.shape
            
            patch_h = H / H_feat
            patch_w = W / W_feat
            
            print(f"  Avg Inference Time: {avg_time*1000:.2f} ms")
            print(f"  Feature Map Size: {H_feat}x{W_feat}")
            print(f"  Effective Patch Size: {patch_h:.2f}x{patch_w:.2f}")
            
            results.append({
                'resolution': res,
                'time_ms': avg_time * 1000,
                'feat_map_size': (H_feat, W_feat),
                'patch_size': (patch_h, patch_w)
            })
            
            enc.unload()
            del enc
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
        except Exception as e:
            print(f"  Failed: {e}")
            results.append({
                'resolution': res,
                'error': str(e)
            })
            
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--encoder", type=str, default="radio", help="Encoder to evaluate")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default="docs", help="Directory to save results")
    args = parser.parse_args()
    
    # Test resolutions
    resolutions = [
        (224, 224),
        (384, 384),
        (512, 512),
        (518, 518), # DINOv2 preferred
        (640, 640),
        (1024, 1024)
    ]
    
    results = evaluate_resolution(args.encoder, resolutions, args.device)
    
    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, f"resolution_study_{args.encoder}.md")
    with open(output_path, "w") as f:
        f.write(f"# Resolution Study for {args.encoder}\n\n")
        f.write("| Resolution | Time (ms) | Feature Map | Patch Size |\n")
        f.write("|---|---|---|---|\n")
        for r in results:
            if 'error' in r:
                f.write(f"| {r['resolution']} | Error | {r['error']} | - |\n")
            else:
                f.write(f"| {r['resolution']} | {r['time_ms']:.2f} | {r['feat_map_size']} | {r['patch_size']} |\n")
    
    print(f"Results saved to {output_path}")
