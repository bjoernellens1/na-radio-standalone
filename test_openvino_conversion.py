import torch
import openvino as ov
import time
import numpy as np
import os
from encoders import NARadioEncoder

def test_conversion():
    print("Loading RADIO model...")
    # Initialize encoder (force CPU for base comparison)
    encoder = NARadioEncoder(device="cpu")
    model = encoder.model
    model.eval()

    # Create dummy input
    dummy_input = torch.randn(1, 3, 512, 512)

    print("Converting to OpenVINO...")
    try:
        # Convert model
        ov_model = ov.convert_model(model, example_input=dummy_input)
        
        # Save model
        ov.save_model(ov_model, "radio_model.xml")
        print("Model saved to radio_model.xml")
        
        # Compile model
        core = ov.Core()
        compiled_model = core.compile_model(ov_model, "CPU")
        
        print("Benchmarking...")
        # PyTorch
        t0 = time.time()
        with torch.no_grad():
            for _ in range(10):
                _ = model(dummy_input)
        pt_time = (time.time() - t0) / 10
        print(f"PyTorch (CPU) avg time: {pt_time*1000:.2f} ms")
        
        # OpenVINO
        infer_request = compiled_model.create_infer_request()
        input_tensor = ov.Tensor(dummy_input.numpy())
        
        t0 = time.time()
        for _ in range(10):
            infer_request.infer([input_tensor])
        ov_time = (time.time() - t0) / 10
        print(f"OpenVINO (CPU) avg time: {ov_time*1000:.2f} ms")
        
        print(f"Speedup: {pt_time/ov_time:.2f}x")
        
    except Exception as e:
        print(f"Conversion failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_conversion()
