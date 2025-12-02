import torch
import numpy as np
import cv2
from typing import Optional

def get_device(prefer_cuda: bool = True) -> str:
  if prefer_cuda and torch.cuda.is_available():
    return "cuda"
  return "cpu"

def get_gpu_info():
  import torch
  if not torch.cuda.is_available():
    return None
  dev = torch.cuda.current_device()
  props = torch.cuda.get_device_properties(dev)
  return {
    'name': props.name,
    'total_memory': props.total_memory,
    'capability': f"{props.major}.{props.minor}",
  }

def is_cuda_compatible(min_major=7):
  """Return True if the current GPU(s) has a major compute capability >= min_major.

  The function gracefully handles no-CUDA available case.
  """
  if not torch.cuda.is_available():
    return False
  props = torch.cuda.get_device_properties(torch.cuda.current_device())
  return props.major >= min_major

def preprocess_frame(frame: np.ndarray, input_resolution=(512, 512)) -> torch.FloatTensor:
  """Converts a BGR numpy frame from OpenCV to a tensor ready for the encoder.

  Returns a torch.FloatTensor shaped (1, 3, H, W).
  """
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
