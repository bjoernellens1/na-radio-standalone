import torch
import sys
import os

try:
    import intel_extension_for_pytorch as ipex
    print(f"IPEX version: {ipex.__version__}")
except ImportError:
    print("IPEX not installed")
    sys.exit(0)

print(f"Torch version: {torch.__version__}")

def optimize_model(model):
    try:
        print("Optimizing model...")
        model = ipex.optimize(model)
        print("Optimization successful")
    except Exception as e:
        print(f"Optimization failed: {e}")
        import traceback
        traceback.print_exc()

# Try to load RADIO model
try:
    print("Loading RADIO model...")
    model = torch.hub.load("NVlabs/RADIO", "radio_model",
                            version="radio_v2.5-b",
                            progress=True,
                            skip_validation=True,
                            adaptor_names=["siglip"])
    model.eval()
    
    print("Model loaded. Structure:")
    # print(model) 
    
    optimize_model(model)

except Exception as e:
    print(f"Failed to load model: {e}")
