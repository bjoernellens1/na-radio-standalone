
import torch
from encoders import NARadioEncoder
from utils import preprocess_frame, cosine_similarity_matrix
import numpy as np

def test_radio():
    print("Initializing NARadioEncoder...")
    try:
        encoder = NARadioEncoder(device="cpu") # Use CPU to avoid CUDA issues for now, or "cuda" if avail
    except Exception as e:
        print(f"Failed to init encoder: {e}")
        return

    print("Encoding labels...")
    labels = ["cat", "dog"]
    label_vecs = encoder.encode_labels(labels)
    print(f"Label vecs shape: {label_vecs.shape}")

    print("Encoding image...")
    # Create dummy image (H, W, 3)
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    input_t = preprocess_frame(dummy_frame, input_resolution=(512, 512))
    
    img_vec = encoder.encode_image_to_vector(input_t)
    print(f"Image vec shape: {img_vec.shape}")

    print("Computing similarity...")
    try:
        sims = cosine_similarity_matrix(img_vec, label_vecs)
        print(f"Similarity shape: {sims.shape}")
        print("Success!")
    except Exception as e:
        print(f"Matmul error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_radio()
