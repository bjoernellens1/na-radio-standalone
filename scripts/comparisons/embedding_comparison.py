import torch
from torchvision import transforms
from PIL import Image
import json
import argparse
import sys
import os
import time

# Add parent dir to path to import encoders
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from encoders import NARadioEncoder, DINOv2Encoder, NADINOv2Encoder, DINOv3Encoder, NADINOv3Encoder
from utils import get_device

class ModelComparator:
    def __init__(self, models=['naclip', 'radio'], device=None):
        self.device = device if device else get_device()
        self.models = self._load_models(models)
        self.metrics = {}
    
    def _load_models(self, model_names):
        loaded_models = {}
        for name in model_names:
            print(f"Loading {name}...")
            try:
                if name == 'radio':
                    # Default RADIO
                    loaded_models[name] = NARadioEncoder(device=self.device, use_naclip=True)
                elif name == 'radio_no_naclip':
                    loaded_models[name] = NARadioEncoder(device=self.device, use_naclip=False)
                elif name == 'dinov2':
                    loaded_models[name] = DINOv2Encoder(device=self.device)
                elif name == 'nadino':
                    loaded_models[name] = NADINOv2Encoder(device=self.device)
                elif name == 'dinov3':
                    loaded_models[name] = DINOv3Encoder(device=self.device)
                elif name == 'nadino3':
                    loaded_models[name] = NADINOv3Encoder(device=self.device)
                elif name == 'naclip':
                     # Alias for radio with naclip (or specific naclip implementation if different)
                     loaded_models[name] = NARadioEncoder(device=self.device, use_naclip=True)
                else:
                    print(f"Unknown model: {name}")
            except Exception as e:
                print(f"Failed to load {name}: {e}")
        return loaded_models

    def compare_images(self, image_paths, resolutions=[256, 512, 1024]):
        """Compare embedding consistency across resolutions"""
        results = {}
        
        # Pre-load images
        images = []
        for p in image_paths:
            try:
                img = Image.open(p).convert('RGB')
                images.append(img)
            except Exception as e:
                print(f"Failed to load image {p}: {e}")

        if not images:
            print("No valid images found.")
            return {}

        for res in resolutions:
            print(f"Evaluating at resolution {res}x{res}...")
            res_results = {}
            
            for name, model in self.models.items():
                model_results = {
                    'inference_time': [],
                    'embedding_dim': 0
                }
                
                # Update resolution if supported
                if hasattr(model, 'update_input_resolution'):
                    model.update_input_resolution((res, res))
                elif hasattr(model, 'input_resolution'):
                     # Re-init might be needed for some models if they bake resolution in
                     # For now assume update_input_resolution or dynamic handling
                     pass

                # Run inference
                start_time = time.time()
                embeddings = []
                for img in images:
                    # Handle resolution constraints
                    target_res = res
                    if 'dinov2' in name or 'nadino' in name:
                        # Ensure multiple of 14
                        target_res = int(round(res / 14) * 14)
                    
                    # Resize image
                    img_res = img.resize((target_res, target_res))
                    
                    # Update model resolution if changed
                    if hasattr(model, 'update_input_resolution'):
                        model.update_input_resolution((target_res, target_res))

                    # Preprocess (simple to tensor for now, encoders handle their own norm usually or expect tensor)
                    # Encoders in this repo expect tensor [1, 3, H, W] usually
                    # But encode_image_to_vector handles preprocessing often?
                    # Let's check encoders.py. NARadioEncoder takes tensor.
                    # We need a unified preprocess.
                    
                    # Using a basic transform for now, should align with utils.preprocess_frame
                    transform = transforms.Compose([
                        transforms.ToTensor(),
                    ])
                    tensor = transform(img_res).unsqueeze(0).to(self.device)
                    
                    try:
                        emb = model.encode_image_to_vector(tensor)
                        embeddings.append(emb.detach().cpu().numpy().tolist())
                    except Exception as e:
                        print(f"Inference failed for {name} at {target_res}: {e}")

                end_time = time.time()
                avg_time = (end_time - start_time) / len(images) if images else 0
                
                model_results['inference_time'] = avg_time
                if embeddings:
                    model_results['embedding_dim'] = len(embeddings[0][0])
                
                res_results[name] = model_results
            
            results[res] = res_results
            
        self.metrics = results
        return results
    
    def save_results(self, filepath):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(self.metrics, f, indent=2)
        print(f"Results saved to {filepath}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs='+', default=['radio', 'nadino'], help="Models to compare")
    parser.add_argument("--resolutions", nargs='+', type=int, default=[256, 512, 1024], help="Resolutions to test")
    parser.add_argument("--images", nargs='+', required=True, help="Path to test images")
    parser.add_argument("--output", default="results/comparison.json", help="Output JSON file")
    parser.add_argument("--device", default=None, help="Device (cuda/cpu)")
    
    args = parser.parse_args()
    
    comp = ModelComparator(models=args.models, device=args.device)
    results = comp.compare_images(args.images, resolutions=args.resolutions)
    comp.save_results(args.output)
