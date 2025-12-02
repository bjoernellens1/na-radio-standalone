import torch
import cv2
import numpy as np
import math
from typing import Optional

try:
    import intel_extension_for_pytorch as ipex
    IPEX_AVAILABLE = True
except ImportError:
    IPEX_AVAILABLE = False

def get_device() -> str:
    """
    Get the best available device for inference.
    Prioritizes Intel XPU if available, then CUDA, then CPU.
    """
    if hasattr(torch, 'xpu') and torch.xpu.is_available():
        return "xpu"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"

def optimize_model(model, dtype=None):
    """
    Optimize model using IPEX if available.
    """
    if IPEX_AVAILABLE:
        try:
            # IPEX optimization
            # For now, we just run ipex.optimize.
            # In a real scenario, we might want to cast to bfloat16 if supported.
            if dtype is None:
                dtype = torch.float32
                # Check for AMX/AVX512 bf16 support?
                # For simplicity, let's stick to float32 or whatever the model is,
                # unless user asks for bf16.

            model = ipex.optimize(model, dtype=dtype)
            print("Model optimized with IPEX")
        except Exception as e:
            print(f"IPEX optimization failed: {e}")
    return model

def is_cuda_compatible(min_major=7):
    """Return True if the current GPU(s) has a major compute capability >= min_major.

    The function gracefully handles no-CUDA available case.
    """
    # Legacy check, keep for compatibility or remove?
    # We'll keep it returning False if no CUDA
    if not torch.cuda.is_available():
        return False
    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    return props.major >= min_major

def preprocess_frame(frame: np.ndarray, input_resolution=(512, 512)) -> torch.FloatTensor:
  # Convert BGR to RGB and resize
  rgb = frame[:, :, ::-1]
  # resize and convert to float
  h, w = input_resolution
  resized = cv2.resize(rgb, (w, h), interpolation=cv2.INTER_LINEAR)
  arr = resized.astype(np.float32) / 255.0
  # HWC -> CHW
  chw = np.transpose(arr, (2, 0, 1))
  tensor = torch.from_numpy(chw).unsqueeze(0)
  return tensor

def cosine_similarity_matrix(image_vec: torch.FloatTensor, label_vecs: torch.FloatTensor):
  """Compute cosine similarity between image vectors (B x D) and label vectors (L x D).

  Returns (B x L) similarity scores.
  """
  # ensure shapes
  image_vec = image_vec / (image_vec.norm(dim=-1, keepdim=True) + 1e-8)
  label_vecs = label_vecs / (label_vecs.norm(dim=-1, keepdim=True) + 1e-8)
  return image_vec @ label_vecs.T

def pca_2d_projection(vectors: np.ndarray) -> np.ndarray:
  """Project Nxd vectors to Nx2 using SVD for a stable 2D projection."""
  # center
  X = vectors - vectors.mean(axis=0, keepdims=True)
  # compute SVD
  u, s, vh = np.linalg.svd(X, full_matrices=False)
  proj = X @ vh.T[:, :2]
  return proj
