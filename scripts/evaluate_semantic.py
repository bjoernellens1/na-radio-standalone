import argparse
import torch
import numpy as np
import sys
import os
import cv2
from typing import List

# Add parent dir to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from encoders import NARadioEncoder, DINOv2Encoder, NADINOv2Encoder, DINOv3Encoder, NADINOv3Encoder
from utils import get_device

def evaluate_semantic(encoder_name, image_dir, labels, device=None):
    if device is None:
        device = get_device()
        
    print(f"Evaluating {encoder_name} on {image_dir}...")
    
    # Load encoder
    if encoder_name == 'radio':
        enc = NARadioEncoder(device=device, return_radio_features=False)
    elif encoder_name == 'radio_no_naclip':
        enc = NARadioEncoder(device=device, return_radio_features=False, use_naclip=False)
    elif encoder_name == 'dinov2':
        enc = DINOv2Encoder(device=device)
    elif encoder_name == 'nadino':
        enc = NADINOv2Encoder(device=device)
    elif encoder_name == 'dinov3':
        enc = DINOv3Encoder(device=device)
    elif encoder_name == 'nadino3':
        enc = NADINOv3Encoder(device=device)
    else:
        print(f"Unknown encoder: {encoder_name}")
        return

    # Get images
    image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if not image_files:
        print("No images found.")
        return

    # Encode labels
    label_list = [l.strip() for l in labels.split(",")]
    print(f"Labels: {label_list}")
    text_features = enc.encode_labels(label_list)
    
    results = []
    
    for img_file in image_files:
        img_path = os.path.join(image_dir, img_file)
        frame = cv2.imread(img_path)
        if frame is None:
            continue
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Preprocess
        # Basic preprocessing: resize to encoder input size?
        # Encoders usually handle tensor input.
        # We need to convert to tensor (1, 3, H, W) float 0..1
        H, W = enc.input_resolution
        frame_resized = cv2.resize(frame_rgb, (W, H))
        tensor = torch.from_numpy(frame_resized).permute(2, 0, 1).float() / 255.0
        tensor = tensor.unsqueeze(0)
        
        # Encode image
        img_features = enc.encode_image_to_vector(tensor)
        
        # Similarity
        sim = (img_features @ text_features.T).squeeze()
        best_idx = torch.argmax(sim).item()
        pred_label = label_list[best_idx]
        score = sim[best_idx].item()
        
        print(f"Image: {img_file}, Pred: {pred_label} ({score:.2f})")
        
        results.append({
            'image': img_file,
            'prediction': pred_label,
            'score': score
        })
        
    # Save results
    os.makedirs("evaluation_results", exist_ok=True)
    with open(f"evaluation_results/semantic_{encoder_name}.md", "w") as f:
        f.write(f"# Semantic Evaluation for {encoder_name}\n\n")
        f.write(f"Labels: {label_list}\n\n")
        f.write("| Image | Prediction | Score |\n")
        f.write("|---|---|---|\n")
        for r in results:
            f.write(f"| {r['image']} | {r['prediction']} | {r['score']:.4f} |\n")
            
    print(f"Results saved to evaluation_results/semantic_{encoder_name}.md")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--encoder", type=str, default="radio")
    parser.add_argument("--image-dir", type=str, required=True)
    parser.add_argument("--labels", type=str, default="cat,dog,car,person")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()
    
    evaluate_semantic(args.encoder, args.image_dir, args.labels, args.device)
