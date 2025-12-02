
import torch
import open_clip
import sys

def inspect_siglip():
    print("Loading SigLIP...")
    try:
        model, _, _ = open_clip.create_model_and_transforms('ViT-B-16-SigLIP', pretrained='webli')
        model.eval()
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    print("Model visual type:", type(model.visual))
    print("Visual attributes:", dir(model.visual))
    
    if hasattr(model.visual, 'trunk'):
        print("Trunk type:", type(model.visual.trunk))
        print("Trunk attributes:", dir(model.visual.trunk))

    dummy_img = torch.randn(1, 3, 224, 224)
    
    print("\nTrying forward_features on visual...")
    try:
        feat = model.visual.forward_features(dummy_img)
        print("Success! Shape:", feat.shape)
    except Exception as e:
        print("Failed:", e)

    print("\nTrying trunk.forward_features...")
    try:
        if hasattr(model.visual, 'trunk'):
            feat = model.visual.trunk.forward_features(dummy_img)
            print("Success! Shape:", feat.shape)
    except Exception as e:
        print("Failed:", e)

if __name__ == "__main__":
    inspect_siglip()
