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

try:
  from typing_extensions import override, List, Tuple, Optional
except Exception:
  from typing import List, Tuple, Optional
  def override(f):
    return f

import os
import argparse
import json
import time
from collections import deque

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from timm.layers import use_fused_attn
import math

try:
  # optional import from rayfronts; the class here uses the base type if present
  from rayfronts.image_encoders.base import LangSpatialGlobalImageEncoder
except Exception:
  # Minimal base class stub so the encoder is importable without rayfronts
  class LangSpatialGlobalImageEncoder:
    def __init__(self, device: Optional[str] = None):
      self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    def encode_image_to_vector(self, rgb_image: torch.FloatTensor):
      raise NotImplementedError
    def encode_image_to_feat_map(self, rgb_image: torch.FloatTensor):
      raise NotImplementedError


class GaussKernelAttn(nn.Module):
  """Encompases the NACLIP attention mechanism."""

  def __init__(
    self,
    orig_attn,
    input_resolution: tuple,
    gauss_std: float,
    device,
    chosen_cls_id: int,
    dim: int,
    qk_norm: bool = False,
    num_prefix_tokens: int = 8,
  ) -> None:
    super().__init__()
    num_heads = orig_attn.num_heads
    assert dim % num_heads == 0, "dim should be divisible by num_heads"
    self.num_heads = num_heads
    self.head_dim = dim // num_heads
    self.scale = self.head_dim ** -0.5
    self.fused_attn = use_fused_attn()
    self.input_resolution = input_resolution

    h, w = input_resolution
    n_patches = (w // 16, h //16)
    window_size = [side * 2 - 1 for side in n_patches]
    window = GaussKernelAttn.gaussian_window(*window_size, std=gauss_std,
                                             device=device)
    self.attn_addition = GaussKernelAttn.get_attention_addition(
      *n_patches, window, num_prefix_tokens
    ).unsqueeze(0)

    self.chosen_cls_id = chosen_cls_id
    self.gauss_std = gauss_std

    self.qkv = orig_attn.qkv
    self.q_norm = orig_attn.q_norm if qk_norm else nn.Identity()
    self.k_norm = orig_attn.k_norm if qk_norm else nn.Identity()
    self.attn_drop = orig_attn.attn_drop
    self.proj = orig_attn.proj
    self.proj_drop = orig_attn.proj_drop
    self.device = device
    self.num_prefix_tokens = num_prefix_tokens

  def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
    B, N, C = x.shape
    x_out = self.custom_attn(x.permute(1, 0, 2))
    x_out = x_out.permute(1, 0, 2)
    return x_out

  @staticmethod
  def gaussian_window(dim1, dim2, std=5., device="cuda"):
    constant = 1 / (std * math.sqrt(2))
    start = -(dim1 - 1) / 2.0
    k1 = torch.linspace(start=start * constant,
                        end=(start + (dim1 - 1)) * constant,
                        steps=dim1,
                        dtype=torch.float, device=device)
    start = -(dim2 - 1) / 2.0
    k2 = torch.linspace(start=start * constant,
                        end=(start + (dim2 - 1)) * constant,
                        steps=dim2,
                        dtype=torch.float, device=device)
    dist_square_to_mu = (torch.stack(torch.meshgrid(
      k1, k2, indexing="ij")) ** 2).sum(0)

    return torch.exp(-dist_square_to_mu)

  @staticmethod
  def get_attention_addition(dim1, dim2, window, num_prefix_tokens=8):
    d = window.device
    m = torch.einsum("ij,kl->ijkl",
                     torch.eye(dim1, device=d),
                     torch.eye(dim2, device=d))
    m = m.permute((0, 3, 1, 2)).contiguous()
    out = F.conv2d(m.view(-1, dim1, dim2).unsqueeze(1),
                   window.unsqueeze(0).unsqueeze(1),
                   padding='same').squeeze(1)

    out = out.view(dim1 * dim2, dim1 * dim2)
    if num_prefix_tokens > 0:
      v_adjusted = torch.vstack(
        [torch.zeros((num_prefix_tokens, dim1 * dim2), device=d), out])
      out = torch.hstack([torch.zeros(
        (dim1 * dim2 + num_prefix_tokens, num_prefix_tokens), device=d),
        v_adjusted])

    return out

  def custom_attn(self, x):
    num_heads = self.num_heads
    num_tokens, bsz, embed_dim = x.size()
    head_dim = embed_dim // num_heads
    scale = head_dim ** -0.5

    q, k, v = self.qkv(x).chunk(3, dim=-1)
    q, k = self.q_norm(q), self.k_norm(k)

    q = q.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
    k = k.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
    v = v.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)

    # kk.T vs kq.T has the most impact
    attn_weights = torch.bmm(k, k.transpose(1, 2)) * scale

    # Gaussian attention seems to have minimal impact
    attn_weights += self.attn_addition
    attn_weights = F.softmax(attn_weights, dim=-1)

    attn_output = torch.bmm(attn_weights, v)
    attn_output = attn_output.transpose(0, 1).contiguous().view(
      -1, bsz, embed_dim)
    attn_output = self.proj(attn_output)
    attn_output = self.proj_drop(attn_output)

    return attn_output

  def update_input_resolution(self, input_resolution):
    h, w = input_resolution
    n_patches = (w // 16, h //16)
    window_size = [side * 2 - 1 for side in n_patches]
    window = GaussKernelAttn.gaussian_window(*window_size, std=self.gauss_std,
                                             device=self.device)
    self.attn_addition = GaussKernelAttn.get_attention_addition(
      *n_patches, window, self.num_prefix_tokens
    ).unsqueeze(0)


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
  import cv2 as _cv2
  resized = _cv2.resize(rgb, (w, h), interpolation=_cv2.INTER_LINEAR)
  arr = resized.astype(np.float32) / 255.0
  # HWC -> CHW
  chw = np.transpose(arr, (2, 0, 1))
  tensor = torch.from_numpy(chw).unsqueeze(0)
  return tensor


class FallbackResNetEncoder(LangSpatialGlobalImageEncoder):
  """A minimal, dependency-light fallback encoder using torchvision ResNet.

  It produces a global embedding vector for an RGB image frame.
  Label encoding is simulated with a simple tokenization/averaging process.
  """
  def __init__(self, device: str = None, input_resolution=(224, 224)):
    super().__init__(device)
    self.device = device or get_device()
    self.input_resolution = input_resolution
    try:
      import torchvision.models as models
    except Exception:
      models = None
    if models is None:
      raise ImportError("torchvision is required for FallbackResNetEncoder")
    self.model = models.resnet50(pretrained=True)
    # remove final classifier
    self.model = torch.nn.Sequential(*(list(self.model.children())[:-1]))
    self.model.eval()
    self.model = self.model.to(self.device)

  @override
  def encode_image_to_vector(self, rgb_image: torch.FloatTensor) -> torch.FloatTensor:
    # rgb_image expected to be (1,3,H,W) 0..1 float
    with torch.no_grad():
      img = rgb_image.to(self.device)
      out = self.model(img)
      out = out.reshape(out.shape[0], -1)
      out = out / (out.norm(dim=-1, keepdim=True) + 1e-8)
    return out

  @override
  def encode_labels(self, labels: List[str]) -> torch.FloatTensor:
    # naive label encoder: map to a reproducible vector using hashing
    # derive a vector size from the model output dim by running a dummy forward
    with torch.no_grad():
      dummy = torch.zeros((1, 3, self.input_resolution[0], self.input_resolution[1]), dtype=torch.float).to(self.device)
      size = self.model(dummy).reshape(1, -1).shape[-1]
    out = []
    for lab in labels:
      np.random.seed(abs(hash(lab)) % (2 ** 32))
      v = np.random.randn(size)
      v = v / (np.linalg.norm(v) + 1e-8)
      out.append(torch.from_numpy(v).unsqueeze(0).float())
    return torch.cat(out, dim=0).to(self.device)


class CLIPFallbackEncoder(LangSpatialGlobalImageEncoder):
  """If open_clip or CLIP library is available, use it as a lightweight CLIP fallback."""
  def __init__(self, device: str = None, model_name="ViT-B-32", input_resolution=(224, 224)):
    super().__init__(device)
    self.device = device or get_device()
    self.input_resolution = input_resolution
    self.clip_model = None
    self.clip_preprocess = None
    self._clip_expected_resolution = None
    self._clip_mean = None
    self._clip_std = None
    # Try to import open_clip
    try:
      import open_clip
      self.clip_model, _, self.clip_preprocess = open_clip.create_model_and_transforms(model_name, pretrained='laion400m_e32')
      self.tokenizer = open_clip.get_tokenizer(model_name)
      self.clip_model.eval()
      self.clip_model = self.clip_model.to(self.device)
      self._init_clip_resolution()
      self._init_clip_normalization()
    except Exception:
      self.clip_model = None
      print("open_clip not available; CLIP-like fallback will not be used. To install, try: pip install open_clip_torch")

  def _init_clip_resolution(self):
    if self.clip_model is None:
      return
    visual = getattr(self.clip_model, 'visual', None)
    size = None
    if visual is not None:
      size = getattr(visual, 'image_size', None)
      if size is None:
        size = getattr(visual, 'input_resolution', None)
    if isinstance(size, (tuple, list)):
      if len(size) == 2:
        self._clip_expected_resolution = (int(size[0]), int(size[1]))
      elif len(size) == 1:
        self._clip_expected_resolution = (int(size[0]), int(size[0]))
    elif isinstance(size, int):
      self._clip_expected_resolution = (int(size), int(size))
    if self._clip_expected_resolution is not None:
      self.input_resolution = self._clip_expected_resolution

  def _init_clip_normalization(self):
    if self.clip_preprocess is None:
      return
    transforms = getattr(self.clip_preprocess, 'transforms', None)
    if transforms is None:
      return
    for t in transforms:
      # torchvision Normalize exposes mean/std attributes
      mean = getattr(t, 'mean', None)
      std = getattr(t, 'std', None)
      if mean is not None and std is not None:
        mean = torch.tensor(mean, dtype=torch.float32).view(1, -1, 1, 1)
        std = torch.tensor(std, dtype=torch.float32).view(1, -1, 1, 1)
        self._clip_mean = mean.to(self.device)
        self._clip_std = std.to(self.device)
        break

  def _prepare_image(self, rgb_image: torch.FloatTensor) -> torch.FloatTensor:
    image = rgb_image
    if image.dim() == 3:
      image = image.unsqueeze(0)
    image = image.to(self.device, dtype=torch.float32)
    if self._clip_expected_resolution is not None:
      target_h, target_w = self._clip_expected_resolution
      if image.shape[-2:] != (target_h, target_w):
        image = F.interpolate(
          image, size=(target_h, target_w), mode='bicubic', align_corners=False)
    if self._clip_mean is not None and self._clip_std is not None:
      # If values are still in [0,1], apply the CLIP normalization; otherwise assume caller already normalized.
      img_min = torch.amin(image, dim=(-2, -1), keepdim=True)
      img_max = torch.amax(image, dim=(-2, -1), keepdim=True)
      if torch.all(img_min >= -0.1) and torch.all(img_max <= 1.1):
        image = (image - self._clip_mean) / self._clip_std
    return image

  @override
  def encode_image_to_vector(self, rgb_image: torch.FloatTensor):
    # Uses CLIP image encoder
    if self.clip_model is None:
      raise RuntimeError("No CLIP model available")
    # ensure shape (B,3,H,W)
    with torch.no_grad():
      image = self._prepare_image(rgb_image)
      img_features = self.clip_model.encode_image(image)
      img_features = img_features / (img_features.norm(dim=-1, keepdim=True) + 1e-8)
    return img_features

  @override
  def encode_labels(self, labels: List[str]):
    if self.clip_model is None:
      raise RuntimeError("No CLIP model available")
    text = self.tokenizer(labels).to(self.device)
    with torch.no_grad():
      text_features = self.clip_model.encode_text(text)
      text_features = text_features / (text_features.norm(dim=-1, keepdim=True) + 1e-8)
    return text_features



class NARadioEncoder(LangSpatialGlobalImageEncoder):
  """The RayFronts Encoder based on NACLIP + RADIO models.

  The model modifies the attention of the last layer of RADIO following the
  example of NACLIP improving spatial structure. And uses the Summary CLS 
  projection to project the patch-wise tokens to SIGLIP or CLIP language aligned
  feature spaces. The model computes na-radio spatial or global features by
  default and exposes functions to project those features to Siglip, or CLIP
  feature spaces.
  """

  DEFAULT_PROMPT_TEMPLATES = [
    "a photo of a {}.",
    "a blurry photo of a {}.",
    "a close-up photo of a {}.",
    "a cropped photo of the {}.",
    "a rendering of a {}.",
    "a photo of the {}.",
    "an image of a {}.",
    "a clean photo of a {}.",
    "a creative photo of a {}.",
    "a picture of a {}.",
    "{}.",
  ]

  def __init__(self, device: str =None,
               model_version: str = "radio_v2.5-b",
               lang_model: str ="siglip",
               input_resolution: Tuple[int,int] = [512,512],
               gauss_std: float = 7.0,
               return_radio_features: bool = True,
               compile: bool = True,
               amp: bool = True):
    """

    Args:
      device: "cpu" or "cuda", set to None to use CUDA if available.
      model_version: Choose from "radio_v2.5-x" where x can be b,l, or g.
        More models can be found on https://github.com/NVlabs/RADIO/
      lang_model: choose from ["siglip", "clip"]
      input_resolution: Tuple of ints (height, width) of the input images.
        Needed to initialize the guassian attention window.
      gauss_std: Standard deviation of the gaussian kernel.
      return_radio_features: Whether to return radio features which are not
        language aligned or whether to project them to the language aligned
        space directly. If True, then the user can always later use the
        functions `align_global_features_with_language` or 
        `align_spatial_features_with_language` to project the radio features
        to be language aligned.
      compile: Whether to compile the model or not. Compiling may be faster but may increase memory usage.
      amp: Whether to use automatic mixed percision or not.
    """

    requested_device = device or get_device()
    if requested_device.startswith('cuda') and not torch.cuda.is_available():
      print("CUDA requested but torch.cuda.is_available() is False; falling back to CPU.")
      requested_device = 'cpu'

    super().__init__(requested_device)

    self.model_version = model_version
    self.return_radio_features = return_radio_features
    self.prompt_templates = list(self.DEFAULT_PROMPT_TEMPLATES)
    self._using_cuda = self.device.startswith('cuda') and torch.cuda.is_available()
    if self._using_cuda:
      props = torch.cuda.get_device_properties(torch.cuda.current_device())
      if props.major < 7:
        print(f"Warning: GPU compute capability {props.major}.{props.minor} is too old for Triton compiler (requires >= 7.0). Disabling compilation.")
        compile = False
    self.compile = bool(compile and self._using_cuda)
    
    # Safeguard: Suppress Triton/Inductor errors on older GPUs
    if self._using_cuda and not self.compile:
      try:
        import torch._dynamo as dynamo
        dynamo.config.suppress_errors = True
        print("Explicitly suppressed TorchDynamo errors for compatibility.")
      except ImportError:
        pass
    self.amp = bool(amp and self._using_cuda)
    self._autocast_device_type = "cuda" if self._using_cuda else "cpu"
    self._autocast_dtype = torch.float16 if self._using_cuda else torch.bfloat16
    self._autocast_enabled = self.amp and self._using_cuda
    self.model = torch.hub.load("NVlabs/RADIO", "radio_model",
                                version=model_version, progress=True,
                                skip_validation=True,
                                adaptor_names=[lang_model])
    self.model.eval()
    self.model = self.model.to(self.device)
    self.model.make_preprocessor_external()
    # Steal adaptors from RADIO so it does not auto compute adaptor output.
    # We want to control when that happens.
    self.lang_adaptor = self.model.adaptors[lang_model]
    self.model.adaptors = None
    last_block = self.model.model.blocks[-1]
    last_block.attn = GaussKernelAttn(
      last_block.attn,
      input_resolution,
      gauss_std,
      dim=self.model.model.embed_dim,
      chosen_cls_id=self.lang_adaptor.head_idx,
      device=self.device,
      num_prefix_tokens=self.model.num_summary_tokens)

    self.times = list()
    if self.compile:
      try:
        self.model.compile(fullgraph=True, options={"triton.cudagraphs":True})
        self.lang_adaptor.compile(fullgraph=True, options={"triton.cudagraphs":True})
      except Exception as e:
        print("RADIO compile() failed; falling back to eager execution:", repr(e))
        self.compile = False

  def _autocast(self):
    return torch.autocast(self._autocast_device_type,
                          dtype=self._autocast_dtype,
                          enabled=self._autocast_enabled)

  @property
  def input_resolution(self):
    return self.model.model.blocks[-1].attn.input_resolution

  @input_resolution.setter
  def input_resolution(self, value):
    if hasattr(value, "__len__") and len(value) == 2:
      if self.is_compatible_size(*value):
        self.model.model.blocks[-1].attn.update_input_resolution(value)
        if self.compile:
          self.model.compile(fullgraph=True, options={"triton.cudagraphs":True})
      else:
        raise ValueError(f"Incompatible input resolution {value}")
    else:
      raise ValueError("Input resolution must be a tuple of two ints")

  @override
  def encode_labels(self, labels: List[str]) -> torch.FloatTensor:
    prompts_per_label = self.insert_labels_into_templates(labels)
    all_text_features = list()
    for i in range(len(labels)):
      text_features = self.encode_prompts(prompts_per_label[i])
      text_features = text_features.mean(dim=0, keepdim=True)
      all_text_features.append(text_features)

    all_text_features = torch.cat(all_text_features, dim=0)
    return all_text_features

  def insert_labels_into_templates(self, labels: List[str]) -> List[List[str]]:
    """Expand each label into a list of prompts following CLIP-style templates."""
    templates = self.prompt_templates or self.DEFAULT_PROMPT_TEMPLATES
    expanded = []
    for label in labels:
      prompts = []
      for tmpl in templates:
        try:
          prompts.append(tmpl.format(label))
        except Exception:
          prompts.append(tmpl.replace("{}", label))
      if not prompts:
        prompts = [label]
      expanded.append(prompts)
    return expanded

  @override
  def encode_prompts(self, prompts: List[str]) -> torch.FloatTensor:
    with self._autocast():
      text = self.lang_adaptor.tokenizer(prompts).to(self.device)
      text_features = self.lang_adaptor.encode_text(text)
      text_features /= text_features.norm(dim=-1, keepdim=True)
    return text_features

  @override
  def encode_image_to_vector(
    self, rgb_image: torch.FloatTensor) -> torch.FloatTensor:

    with self._autocast():
      out = self.model(rgb_image)
      C = out.summary.shape[-1] // 3
      i = self.lang_adaptor.head_idx
      out = out.summary[:, C*i: C*(i+1)]

      if not self.return_radio_features:
        out = self.lang_adaptor.head_mlp(out)

    return out

  @override
  def encode_image_to_feat_map(
    self, rgb_image: torch.FloatTensor) -> torch.FloatTensor:
    B, C, H, W = rgb_image.shape
    H_, W_ = H // self.model.patch_size, W // self.model.patch_size
    with self._autocast():
      out = self.model(rgb_image).features
      if not self.return_radio_features:
        out = self.lang_adaptor.head_mlp(out)
    return out.permute(0, 2, 1).reshape(B, -1, H_, W_)

  @override
  def encode_image_to_feat_map_and_vector(self, rgb_image: torch.FloatTensor) \
      -> Tuple[torch.FloatTensor, torch.FloatTensor]:
    B, C, H, W = rgb_image.shape
    H_, W_ = H // self.model.patch_size, W // self.model.patch_size
    with self._autocast():
      out = self.model(rgb_image)

      C = out.summary.shape[-1] // 3
      i = self.lang_adaptor.head_idx
      global_vector = out.summary[:, C*i: C*(i+1)]

      feat_map = out.features

      if not self.return_radio_features:
        global_vector = self.lang_adaptor.head_mlp(global_vector)
        feat_map = self.lang_adaptor.head_mlp(feat_map)

    return feat_map, global_vector

  @override
  def align_global_features_with_language(self, features: torch.FloatTensor):
    if self.lang_adaptor is None:
      raise ValueError("Cannot align to language without a lang model")
    if not self.return_radio_features:
      return features
    B,C = features.shape
    with self._autocast():
      return self.lang_adaptor.head_mlp(features)

  @override
  def align_spatial_features_with_language(self, features: torch.FloatTensor):
    if self.lang_adaptor is None:
      raise ValueError("Cannot align to language without a lang model")
    if not self.return_radio_features:
      return features
    B,C,H,W = features.shape
    features = features.permute(0, 2, 3, 1).reshape(B, -1, C)
    with self._autocast():
      out = self.lang_adaptor.head_mlp(features)
    return out.permute(0, 2, 1).reshape(B, -1, H, W)

  @override
  def is_compatible_size(self, h: int, w: int):
    hh, ww = self.get_nearest_size(h, w)
    return hh == h and ww == w

  @override
  def get_nearest_size(self, h, w):
    return self.model.get_nearest_supported_resolution(h, w)


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


def run_webcam_demo(encoder, labels, input_resolution=(512, 512), camera_index=0,
                    topk=3, enable_plot=True, video_file: Optional[str] = None):
  """Run a webcam loop that computes embeddings and overlays labels.

  encoder must provide encode_image_to_vector and encode_labels.
  """
  try:
    import cv2
    _cv2_import_ok = True
    cv2_import_exception = None
  except Exception as ex:
    cv2 = None
    _cv2_import_ok = False
    cv2_import_exception = ex
  import matplotlib.pyplot as plt
  device = get_device()
  # init camera
  if not _cv2_import_ok and video_file is None:
    print("OpenCV import failed. Here's the error:\n", cv2_import_exception)
    print("Possible fixes:")
    print(" - On Debian/Ubuntu: sudo apt-get install libglib2.0-0")
    print(" - On Fedora/RHEL: sudo dnf install glib2")
    print(" - On NixOS: add 'glib' and a proper OpenCV package to your dev environment, or install a prebuilt opencv package.")
    print(" - Alternatively, install opencv-python-headless in your environment if you don't need GUI windows: pip install opencv-python-headless")
    return
  cap = cv2.VideoCapture(camera_index) if video_file is None else None
  if not cap.isOpened():
    raise RuntimeError(f"Cannot open camera index {camera_index}")

  # prepare label vectors
  label_vecs = encoder.encode_labels(labels).to(device)

  emb_history = deque(maxlen=128)
  # plotting setup
  if enable_plot:
    plt.ion()
    fig, ax = plt.subplots(figsize=(5, 5))

  try:
    while True:
      if video_file is not None:
        # read frame using imageio as a fallback; create reader lazily
        import imageio
        if not hasattr(run_webcam_demo, '_video_reader') or run_webcam_demo._video_reader is None:
          run_webcam_demo._video_reader = imageio.get_reader(video_file)
          run_webcam_demo._video_iter = iter(run_webcam_demo._video_reader)
        try:
          frame = next(run_webcam_demo._video_iter)
          ret = True
        except StopIteration:
          break
      else:
        ret, frame = cap.read()
      if not ret:
        break
      input_t = preprocess_frame(frame, input_resolution=input_resolution)
      # if input resolution different than model expected, resize in encoder
      input_t = input_t.to(device)

      with torch.no_grad():
        img_vec = encoder.encode_image_to_vector(input_t)
        # ensure 2D batch shape
        if img_vec.dim() == 1:
          img_vec = img_vec.unsqueeze(0)

      # compute similarities
      sims = cosine_similarity_matrix(img_vec, label_vecs)
      sims = sims.cpu().numpy()[0]
      # pick topk
      idxs = sims.argsort()[-topk:][::-1]
      # overlay topk on frame
      overlay_frame = frame.copy()
      y = 30
      for i in idxs:
        label_text = f"{labels[i]}: {sims[i]:.3f}"
        cv2.putText(overlay_frame, label_text, (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 255, 0), 2)
        y += 30

      # show frame
      cv2.imshow("naradio webcam demo", overlay_frame)
      # update emb history
      v = img_vec.cpu().numpy()[0]
      emb_history.append(v)

      if enable_plot and len(emb_history) >= 4:
        arr = np.stack(list(emb_history))
        pts = pca_2d_projection(arr)
        ax.clear()
        ax.scatter(pts[:, 0], pts[:, 1], c=np.linspace(0, 1, pts.shape[0]), cmap="viridis")
        ax.set_title("Embedding PCA (last {} frames)".format(len(emb_history)))
        plt.draw()
        plt.pause(0.001)

      key = cv2.waitKey(1) & 0xFF
      if key == ord('q') or key == 27:  # q or ESC
        break
  finally:
    cap.release()
    cv2.destroyAllWindows()
    if enable_plot:
      plt.ioff()
      plt.close(fig)
    if hasattr(run_webcam_demo, '_video_reader') and run_webcam_demo._video_reader is not None:
      run_webcam_demo._video_reader.close()


def load_encoder(preferred='radio', device: Optional[str] = None, input_resolution=(512,512), force_gpu: bool = False, min_cc: float = 7.0):
  """Try to load RADIO via the NARadioEncoder; if not possible, fallback to ResNet.
  Returns an encoder instance and a short string describing it.
  """
  device = device or get_device()
  if device.startswith('cuda') and not torch.cuda.is_available():
    print("CUDA selected but torch.cuda.is_available() returned False; falling back to CPU.")
    device = 'cpu'
  if device.startswith('cuda') and torch.cuda.is_available():
    info = get_gpu_info()
    if info is not None:
      print(f"Detected GPU: {info['name']} (compute capability {info['capability']})")
  # If user requested CUDA but the current GPU compute capability is too low
  # for modern PyTorch builds, warn and fallback to CPU to avoid hard crashes.
  # If the user wants to use GPU but the compute capability is too low
  # for this build of PyTorch, warn and fallback to CPU. The demo accepts
  # a `min_cc` parameter which can be lowered when the user explicitly
  # wants to try to run on older compute capability GPUs.
  if device == 'cuda' and not force_gpu and not is_cuda_compatible(min_major=int(min_cc)):
    info = get_gpu_info()
    if info is not None:
      cc = info.get('capability', 'unknown')
      print(f"Warning: GPU compute capability {cc} looks low for modern PyTorch builds. Falling back to CPU to avoid runtime incompatibilities.")
      print("If you want to use the GPU, install a PyTorch build compatible with your GPU or compile PyTorch from source.")
    device = 'cpu'
  if preferred == 'radio':
    try:
      enc = NARadioEncoder(device=device,
                           input_resolution=input_resolution,
                           return_radio_features=False)
      return enc, "radio"
    except Exception as e:
      msg = str(e)
      print("RADIO not available or failed to load; falling back to ResNet encoder:", e)
      if 'einops' in msg or 'einops' in getattr(e, 'name', ''):
        print("It looks like RADIO failed due to missing 'einops'. You can install it with: pip install einops.")
  # try CLIP fallback
  try:
    enc_clip = CLIPFallbackEncoder(device=device, input_resolution=input_resolution)
    return enc_clip, "open-clip-fallback"
  except Exception:
    pass
  # fallback resnet
  enc = FallbackResNetEncoder(device=device, input_resolution=input_resolution)
  return enc, "resnet-fallback"


def main(argv=None):
  parser = argparse.ArgumentParser(description="na-radio webcam demo")
  parser.add_argument('--mode', choices=['webcam'], default='webcam')
  parser.add_argument('--camera', type=int, default=0)
  parser.add_argument('--video-file', type=str, default=None, help='Optional video file to use instead of webcam')
  parser.add_argument('--labels', type=str, default='person,car,dog,cat,tree')
  parser.add_argument('--input-resolution', type=str, default='512,512')
  parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
  parser.add_argument('--force-cpu', action='store_true', help='Force CPU even if CUDA is available')
  parser.add_argument('--force-gpu', action='store_true', help='Force GPU even if compute capability checks fail (risk runtime error)')
  parser.add_argument('--min-cc', type=float, default=7.0, help='Minimum required GPU compute capability major version (default: 7.0). Lower to support older GPUs like P100 (6.0)')
  parser.add_argument('--model', type=str, default='radio')
  parser.add_argument('--gpu-info', action='store_true', help='Print GPU details and exit')
  args = parser.parse_args(argv)

  labels = [x.strip() for x in args.labels.split(',') if x.strip()]
  w, h = [int(x) for x in args.input_resolution.split(',')]
  input_resolution = (h, w)
  # allow forcing CPU if requested
  if args.force_cpu:
    args.device = 'cpu'
  enc, name = load_encoder(preferred=args.model, device=args.device, input_resolution=input_resolution, force_gpu=args.force_gpu, min_cc=args.min_cc)
  chosen_device = getattr(enc, 'device', args.device)
  print(f"Using encoder: {name} on device {chosen_device} with input resolution {input_resolution}")
  if args.mode == 'webcam':
    if args.gpu_info:
      info = get_gpu_info()
      if info is None:
        print('No GPU available (torch.cuda.is_available() is False)')
      else:
        print(json.dumps(info, indent=2))
      return
  run_webcam_demo(enc, labels, input_resolution=input_resolution, camera_index=args.camera, enable_plot=True if args.video_file is None else False)


if __name__ == '__main__':
  main()
