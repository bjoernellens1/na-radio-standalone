
import torch
import open_clip
import sys

def inspect_openclip():
    print("Inspecting OpenCLIP model...")
    try:
        model_name = "ViT-B-32"
        pretrained = "laion400m_e32"
        print(f"Loading {model_name} ({pretrained})...")
        model, _, _ = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
        
        visual = model.visual
        print(f"Visual module type: {type(visual)}")
        print(f"Has forward_features: {hasattr(visual, 'forward_features')}")
        
        if hasattr(visual, 'trunk'):
            print("Has 'trunk' attribute.")
            print(f"Trunk type: {type(visual.trunk)}")
            print(f"Trunk has forward_features: {hasattr(visual.trunk, 'forward_features')}")
            
        # Create dummy input
        dummy_input = torch.randn(1, 3, 224, 224)
        
        if hasattr(visual, 'forward_features'):
            print("Trying visual.forward_features()...")
            try:
                out = visual.forward_features(dummy_input)
                print(f"Output type: {type(out)}")
                if isinstance(out, torch.Tensor):
                    print(f"Output shape: {out.shape}")
                elif isinstance(out, tuple):
                    print(f"Output tuple len: {len(out)}")
                    print(f"Output[0] shape: {out[0].shape}")
            except Exception as e:
                print(f"Error calling forward_features: {e}")
        else:
            print("visual.forward_features not found.")
            
            # Try standard forward to see if we can hook or if it returns features
            # Some implementations might just use forward()
            print("Trying visual(dummy_input)...")
            try:
                out = visual(dummy_input)
                print(f"Output shape: {out.shape}")
            except Exception as e:
                print(f"Error calling visual(): {e}")

    except Exception as e:
        print(f"Failed to load or inspect model: {e}")

if __name__ == "__main__":
    inspect_openclip()
