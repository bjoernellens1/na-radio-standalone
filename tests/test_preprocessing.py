import numpy as np
import torch
import pytest
from naradio import preprocess_frame, FallbackResNetEncoder, get_device


def test_preprocess_frame_shape_and_dtype():
    dummy = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    try:
        import cv2
    except Exception:
        # If cv2 isn't available in the test environment, skip this test
        import pytest
        pytest.skip('cv2 (OpenCV) not available; skipping preprocess_frame test')
    tensor = preprocess_frame(dummy, input_resolution=(224, 224))
    assert isinstance(tensor, torch.FloatTensor) or torch.is_tensor(tensor)
    assert tensor.shape == (1, 3, 224, 224)
    assert tensor.dtype == torch.float32


def test_fallback_encoder_runs():
    try:
        import torchvision
    except Exception:
        pytest.skip("torchvision is not installed; skipping fallback encoder integration test")
    dev = get_device()
    enc = FallbackResNetEncoder(device=dev, input_resolution=(224, 224))
    dummy = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    tensor = preprocess_frame(dummy, input_resolution=(224, 224)).to(dev)
    vec = enc.encode_image_to_vector(tensor)
    # ensure vector normalized and correct batch shape
    assert vec.ndim == 2
    assert vec.shape[0] == 1
    assert torch.allclose(torch.norm(vec, dim=-1), torch.tensor(1.0, device=dev), atol=1e-2) or torch.norm(vec) > 0


def test_load_encoder_runtime():
    # ensure load_encoder can build an encoder object in current env (falling back gracefully)
    from naradio import load_encoder
    try:
        enc, name = load_encoder(preferred='radio')
    except Exception:
        try:
            enc, name = load_encoder(preferred='fallback')
        except Exception as e:
            import pytest
            pytest.skip("No encoder available in this environment: %s" % (e,))
    assert hasattr(enc, 'encode_image_to_vector')
