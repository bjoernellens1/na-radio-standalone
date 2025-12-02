"""Standalone demo script for using a RADIO/CLIP style image encoder.

This module contains the NARadioEncoder class (from the RayFronts project)
but also provides a command-line demo to capture a webcam stream, compute
embeddings and display the live stream with top label predictions overlaid.

It is intended to be a standalone, runnable demo using a physical webcam and
GPU-backed PyTorch where available. If NVlabs/RADIO is not available, it falls
back to a smaller CLIP-like encoder so the demo can be tried without extra
dependencies.

Features:
- Webcam capture using OpenCV
- GPU detection and optional CUDA usage
- Fallback encoder if RADIO or CLIP is not installed or fails to load
- Live overlays of top-K labels and a small embedding projection window
Dependencies: torch, torchvision (or RADIO/open_clip when available), opencv-python, numpy, matplotlib, timm.
"""

import argparse
import time
from collections import deque
from typing import Optional, List

import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt

from utils import get_device, preprocess_frame, cosine_similarity_matrix, pca_2d_projection, is_cuda_compatible
from encoders import NARadioEncoder, FallbackResNetEncoder, CLIPFallbackEncoder, SigLIPEncoder, DINOv2Encoder, YoloWorldEncoder, LangSpatialGlobalImageEncoder

def load_encoder(preferred='radio', device: Optional[str] = None, input_resolution=(512,512), force_gpu: bool = False, min_cc: float = 7.0):
  """Try to load RADIO via the NARadioEncoder; if not possible, fallback to ResNet.
  Returns an encoder instance and a short string describing it.
  """
  # Determine device
  if device is None:
      device = get_device()
  
  print(f"Loading encoder '{preferred}' on {device}...")

  if preferred == 'radio':
      try:
          # Check GPU compatibility for RADIO
          if device.startswith('cuda') and not force_gpu:
              if not is_cuda_compatible(min_major=int(min_cc)):
                   print(f"GPU compute capability too low for RADIO (requires >={min_cc}). Falling back to CPU/ResNet or forcing CPU.")
      
          enc = NARadioEncoder(device=device, input_resolution=input_resolution, return_radio_features=False)
          return enc, "RADIO-v2.5"
      except Exception as e:
          print(f"Failed to load RADIO: {e}")
          print("Falling back to CLIP...")
          preferred = 'clip'

  if preferred == 'siglip':
      try:
          enc = SigLIPEncoder(device=device, input_resolution=input_resolution)
          return enc, "SigLIP-ViT-SO400M"
      except Exception as e:
          print(f"Failed to load SigLIP: {e}")
          preferred = 'clip'

  if preferred == 'dinov2':
      try:
          # DINOv2 requires resolution multiple of 14 (patch size). 518 = 14 * 37.
          enc = DINOv2Encoder(device=device, input_resolution=(518, 518))
          return enc, "DINOv2-ViT-S-14"

      except Exception as e:
          print(f"Failed to load DINOv2: {e}")
          preferred = 'resnet'

  if preferred == 'yolo':
      try:
          enc = YoloWorldEncoder(device=device)
          return enc, "Yolo-World-v8s"
      except Exception as e:
          print(f"Failed to load Yolo-World: {e}")
          preferred = 'resnet'


  if preferred == 'clip':
      try:
          enc = CLIPFallbackEncoder(device=device, input_resolution=input_resolution)
          return enc, "CLIP-ViT-B-32"
      except Exception as e:
          print(f"Failed to load CLIP: {e}")
          print("Falling back to ResNet...")
          preferred = 'resnet'

  if preferred == 'resnet':
      enc = FallbackResNetEncoder(device=device, input_resolution=input_resolution)
      return enc, "ResNet50-Fallback"
  
  raise ValueError(f"Unknown encoder preference: {preferred}")



if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--camera", type=int, default=0, help="Camera index")
  parser.add_argument("--labels", type=str, default="person,cat,dog,chair,plant", help="Comma separated labels")
  parser.add_argument("--no-plot", action="store_true", help="Disable embedding plot")
  parser.add_argument("--video-file", type=str, default=None, help="Path to video file")
  parser.add_argument("--encoder", type=str, default="radio", help="Preferred encoder: radio, clip, resnet")
  parser.add_argument("--device", type=str, default=None, help="Device (cuda or cpu)")
  
  args = parser.parse_args()
  
  labels = [l.strip() for l in args.labels.split(",")]
  
  encoder, name = load_encoder(preferred=args.encoder, device=args.device)
  print(f"Using encoder: {name}")
  
  run_webcam_demo(encoder, labels, camera_index=args.camera, enable_plot=not args.no_plot, video_file=args.video_file)
