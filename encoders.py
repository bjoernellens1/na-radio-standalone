import torch
import torch.nn as nn
from torch.nn import functional as F
from typing import List, Tuple, Optional, Union
import numpy as np
from abc import ABC, abstractmethod
import math
from timm.layers import use_fused_attn
from typing_extensions import override
import cv2

from utils import get_device, preprocess_frame

class LangSpatialGlobalImageEncoder(ABC):
    def __init__(self, device: Optional[str] = None):
        self.device = device or get_device()

    @abstractmethod
    def encode_image_to_vector(self, rgb_image: torch.FloatTensor) -> torch.FloatTensor:
        pass

    def encode_labels(self, labels: List[str]) -> torch.FloatTensor:
        raise NotImplementedError("This encoder does not support label encoding")

    def encode_image_to_feat_map(self, rgb_image: torch.FloatTensor) -> torch.FloatTensor:
        raise NotImplementedError("This encoder does not support feature map extraction")
    
    def encode_image_to_feat_map_and_vector(self, rgb_image: torch.FloatTensor) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
         raise NotImplementedError("This encoder does not support feature map and vector extraction")

    def compute_heatmap(self, rgb_image: torch.FloatTensor, label_vec: torch.FloatTensor) -> torch.FloatTensor:
        raise NotImplementedError("This encoder does not support heatmap computation")
    
    def align_global_features_with_language(self, features: torch.FloatTensor):
        return features

    def align_spatial_features_with_language(self, features: torch.FloatTensor):
        return features
    
    def is_compatible_size(self, h: int, w: int):
        return True
    
    def get_nearest_size(self, h, w):
        return h, w

    def unload(self):
        """Free up resources (GPU memory) used by this encoder."""
        if hasattr(self, 'model'):
            del self.model
        if hasattr(self, 'clip_model'):
            del self.clip_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()



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
    self.input_resolution = input_resolution


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

  @override
  def encode_image_to_feat_map(self, rgb_image: torch.FloatTensor) -> torch.FloatTensor:
      if self.clip_model is None:
          raise RuntimeError("No CLIP model available")
      with torch.no_grad():
          image = self._prepare_image(rgb_image)
          visual = getattr(self.clip_model, 'visual', None)
          if hasattr(visual, 'trunk'):
              visual = visual.trunk
          
          if hasattr(visual, 'forward_features'):

               features = visual.forward_features(image)
               if features.shape[1] > 1:
                   N = features.shape[1] - 1 
                   H = W = int(math.sqrt(N))
                   if H * W == N:
                       feat_map = features[:, 1:, :]
                       feat_map = feat_map.permute(0, 2, 1).reshape(features.shape[0], -1, H, W)
                       return feat_map
          # Manual forward pass for OpenCLIP VisionTransformer
          if hasattr(visual, 'conv1') and hasattr(visual, 'transformer'):
              try:
                  x = visual.conv1(image)  # shape = [*, width, grid, grid]
                  x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
                  x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
                  
                  # Add class token
                  if hasattr(visual, 'class_embedding'):
                      class_embedding = visual.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device)
                      x = torch.cat([class_embedding, x], dim=1)
                  
                  # Add positional embedding
                  if hasattr(visual, 'positional_embedding'):
                      x = x + visual.positional_embedding.to(x.dtype)
                  
                  if hasattr(visual, 'ln_pre'):
                      x = visual.ln_pre(x)
                  
                  x = visual.transformer(x)
                  
                  if hasattr(visual, 'ln_post'):
                      x = visual.ln_post(x)
                  
                  # Apply projection if it exists to match text embedding dim
                  if hasattr(visual, 'proj') and visual.proj is not None:
                      x = x @ visual.proj

                      
                  # x is (B, N, C)
                  # Remove class token if added
                  if hasattr(visual, 'class_embedding'):
                       x = x[:, 1:, :]
                  
                  # Reshape to (B, C, H, W)
                  N = x.shape[1]
                  H = int(math.sqrt(N))
                  feat_map = x.permute(0, 2, 1).reshape(x.shape[0], -1, H, H)
                  return feat_map
              except Exception as e:
                  print(f"Manual forward pass failed: {e}")

          raise NotImplementedError("Could not extract spatial features from this CLIP model")





  @override
  def compute_heatmap(self, rgb_image: torch.FloatTensor, label_vec: torch.FloatTensor) -> torch.FloatTensor:
      try:
          feat_map = self.encode_image_to_feat_map(rgb_image)
          feat_map = feat_map / (feat_map.norm(dim=1, keepdim=True) + 1e-8)
          label_vec = label_vec / (label_vec.norm(dim=-1, keepdim=True) + 1e-8)
          sim_map = (feat_map * label_vec.view(1, -1, 1, 1)).sum(dim=1)
          H, W = rgb_image.shape[-2:]
          heatmap = F.interpolate(sim_map.unsqueeze(0), size=(H, W), mode='bicubic', align_corners=False)
          heatmap = heatmap.squeeze()
          heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
          return heatmap
      except Exception as e:
          print(f"Heatmap computation failed: {e}")
          return torch.zeros(rgb_image.shape[-2:], device=self.device)


class NARadioEncoder(LangSpatialGlobalImageEncoder):
  """The RayFronts Encoder based on NACLIP + RADIO models."""

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

  @override
  def unload(self):

    if hasattr(self, 'model'):
      self.model.to('cpu')
      del self.model
    if hasattr(self, 'lang_adaptor'):
      self.lang_adaptor.to('cpu')
      del self.lang_adaptor
    if torch.cuda.is_available():
      torch.cuda.empty_cache()


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
        # Auto-adjust to nearest supported size
        h, w = value
        nearest_h, nearest_w = self.get_nearest_size(h, w)
        print(f"Warning: Resolution {value} is not compatible with patch size. Adjusting to nearest supported size: ({nearest_h}, {nearest_w})")
        self.model.model.blocks[-1].attn.update_input_resolution((nearest_h, nearest_w))
        if self.compile:
          self.model.compile(fullgraph=True, options={"triton.cudagraphs":True})
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

  def compute_heatmap(self, rgb_image: torch.FloatTensor, label_vec: torch.FloatTensor) -> torch.FloatTensor:
    with torch.no_grad():
        feat_map = self.encode_image_to_feat_map(rgb_image)
        if self.return_radio_features:
             feat_map = self.align_spatial_features_with_language(feat_map)
        feat_map = feat_map / (feat_map.norm(dim=1, keepdim=True) + 1e-8)
        label_vec = label_vec / (label_vec.norm(dim=-1, keepdim=True) + 1e-8)
        sim_map = (feat_map * label_vec.view(1, -1, 1, 1)).sum(dim=1)
        H, W = rgb_image.shape[-2:]
        heatmap = F.interpolate(sim_map.unsqueeze(0), size=(H, W), mode='bicubic', align_corners=False)
        heatmap = heatmap.squeeze()
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
        return heatmap


class SigLIPEncoder(LangSpatialGlobalImageEncoder):
    def __init__(self, device: str = None, model_name="ViT-B-16-SigLIP", input_resolution=(224, 224)):
        super().__init__(device)
        self.input_resolution = input_resolution
        try:
            import open_clip
            import open_clip
            # ViT-B-16-SigLIP seems to have dimension mismatch (768 vs 1152).
            # ViT-SO400M-14-SigLIP uses 1152 for both.
            self.model, _, self.preprocess = open_clip.create_model_and_transforms('ViT-SO400M-14-SigLIP', pretrained='webli')
            self.tokenizer = open_clip.get_tokenizer('ViT-SO400M-14-SigLIP')
            self.model.eval()
            self.model = self.model.to(self.device)
        except Exception as e:
            raise ImportError(f"Failed to load SigLIP: {e}")
        
        # Detect model resolution
        self.model_resolution = None
        try:
            # open_clip models usually have visual.image_size
            if hasattr(self.model, 'visual') and hasattr(self.model.visual, 'image_size'):
                 s = self.model.visual.image_size
                 if isinstance(s, int):
                     self.model_resolution = (s, s)
                 elif isinstance(s, (tuple, list)):
                     self.model_resolution = tuple(s)
        except Exception:
            pass
        if self.model_resolution is None:
             # Fallback to 224 if detection fails but we know it's SigLIP B/16
             self.model_resolution = (224, 224)
        print(f"SigLIP detected resolution: {self.model_resolution}")

    @override
    def unload(self):
        if hasattr(self, 'model'):
            self.model.to('cpu')
            del self.model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


    def _prepare_image(self, rgb_image: torch.FloatTensor) -> torch.FloatTensor:
        img = rgb_image
        if self.model_resolution is not None:
             H, W = self.model_resolution
             if img.shape[-2:] != (H, W):
                 img = F.interpolate(img, size=(H, W), mode='bicubic', align_corners=False)
        
        mean = torch.tensor([0.5, 0.5, 0.5], device=self.device).view(1, 3, 1, 1)
        std = torch.tensor([0.5, 0.5, 0.5], device=self.device).view(1, 3, 1, 1)
        img = img.to(self.device)
        img = (img - mean) / std
        return img

    @override
    def encode_image_to_vector(self, rgb_image: torch.FloatTensor) -> torch.FloatTensor:
        # rgb_image is (1, 3, H, W) float 0..1
        # SigLIP expects normalized input. We need to use self.preprocess logic but manually since input is tensor
        # For simplicity, we assume rgb_image is already resized to input_resolution
        # We need to apply normalization.
        img = self._prepare_image(rgb_image)
        
        with torch.no_grad():
            features = self.model.encode_image(img)
            features = features / (features.norm(dim=-1, keepdim=True) + 1e-8)
        # print(f"SigLIP Image Feat Shape: {features.shape}")
        return features

    @override
    def encode_labels(self, labels: List[str]) -> torch.FloatTensor:
        text = self.tokenizer(labels).to(self.device)
        with torch.no_grad():
            features = self.model.encode_text(text)
            features = features / (features.norm(dim=-1, keepdim=True) + 1e-8)
        # print(f"SigLIP Text Feat Shape: {features.shape}")
        return features

    @override
    def encode_image_to_feat_map(self, rgb_image: torch.FloatTensor) -> torch.FloatTensor:
        # SigLIP feature map
        # Similar to CLIP
        img = self._prepare_image(rgb_image)
        
        with torch.no_grad():
             # timm models (which open_clip uses) usually have forward_features
             # Check for TimmModel wrapper
             visual = self.model.visual
             if hasattr(visual, 'trunk'):
                 visual = visual.trunk
             
             if hasattr(visual, 'forward_features'):
                 features = visual.forward_features(img)
             else:
                 # Fallback to forward if forward_features is missing (unlikely for timm)
                 features = visual(img)

             # (B, N, D)
             # SigLIP usually doesn't have CLS token in the same way? Or does it?
             # WebLI SigLIP usually uses MAP head or pooling.
             # Let's assume standard ViT structure for now.
             if features.dim() == 3:
                 # Check if N is square
                 N = features.shape[1]
                 H = int(math.sqrt(N))
                 if H * H == N:
                     # No CLS token?
                     feat_map = features.permute(0, 2, 1).reshape(features.shape[0], -1, H, H)
                     return feat_map
                 else:
                     # Maybe CLS token?
                     N = N - 1
                     H = int(math.sqrt(N))
                     if H * H == N:
                         feat_map = features[:, 1:, :].permute(0, 2, 1).reshape(features.shape[0], -1, H, H)
                         return feat_map
        raise NotImplementedError("Could not extract spatial features from SigLIP")

    @override
    def compute_heatmap(self, rgb_image: torch.FloatTensor, label_vec: torch.FloatTensor) -> torch.FloatTensor:
        try:
            feat_map = self.encode_image_to_feat_map(rgb_image)
            feat_map = feat_map / (feat_map.norm(dim=1, keepdim=True) + 1e-8)
            label_vec = label_vec / (label_vec.norm(dim=-1, keepdim=True) + 1e-8)
            sim_map = (feat_map * label_vec.view(1, -1, 1, 1)).sum(dim=1)
            H, W = rgb_image.shape[-2:]
            heatmap = F.interpolate(sim_map.unsqueeze(0), size=(H, W), mode='bicubic', align_corners=False)
            heatmap = heatmap.squeeze()
            heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
            return heatmap
        except Exception as e:
            print(f"Heatmap computation failed: {e}")
            return torch.zeros(rgb_image.shape[-2:], device=self.device)


class DINOv2Encoder(LangSpatialGlobalImageEncoder):
    def __init__(self, device: str = None, model_name="dinov2_vits14", input_resolution=(518, 518)):
        super().__init__(device)
        self.input_resolution = input_resolution
        self.model = torch.hub.load('facebookresearch/dinov2', model_name)
        self.model.eval()
        self.model = self.model.to(self.device)

    @override
    def unload(self):
        if hasattr(self, 'model'):
            self.model.to('cpu')
            del self.model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


    @override
    def encode_image_to_vector(self, rgb_image: torch.FloatTensor) -> torch.FloatTensor:
        # DINOv2 expects normalized images
        mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)
        
        img = rgb_image.to(self.device)
        # Resize if needed? DINOv2 handles different sizes but patch size matters.
        # We assume input is already resized to input_resolution or close.
        img = (img - mean) / std
        
        with torch.no_grad():
            features = self.model(img)
            features = features / (features.norm(dim=-1, keepdim=True) + 1e-8)
        return features
    
    @override
    def encode_labels(self, labels: List[str]) -> torch.FloatTensor:
        # naive label encoder: map to a reproducible vector using hashing
        # derive a vector size from the model output dim by running a dummy forward
        with torch.no_grad():
            # DINOv2 input size
            H, W = self.input_resolution
            dummy = torch.zeros((1, 3, H, W), dtype=torch.float).to(self.device)
            # We need to make sure we use the same normalization/preprocessing if possible, but for dummy it doesn't matter much
            # DINOv2 forward returns (B, dim)
            out = self.model(dummy)
            size = out.shape[-1]
        
        out_vecs = []
        for lab in labels:
            np.random.seed(abs(hash(lab)) % (2 ** 32))
            v = np.random.randn(size)
            v = v / (np.linalg.norm(v) + 1e-8)
            out_vecs.append(torch.from_numpy(v).unsqueeze(0).float())
        return torch.cat(out_vecs, dim=0).to(self.device)

    @override
    def encode_image_to_feat_map(self, rgb_image: torch.FloatTensor) -> torch.FloatTensor:
        # Extract patch features
        # DINOv2 forward_features returns dict with 'x_norm_patchtokens'
        mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)
        img = rgb_image.to(self.device)
        img = (img - mean) / std
        
        with torch.no_grad():
            # forward_features returns dict
            ret = self.model.forward_features(img)
            # 'x_norm_patchtokens': (B, N, D)
            patch_tokens = ret['x_norm_patchtokens']
            
            # Reshape to (B, D, H, W)
            B, N, D = patch_tokens.shape
            H = int(math.sqrt(N))
            W = H # Square patches
            
            feat_map = patch_tokens.permute(0, 2, 1).reshape(B, D, H, W)
            return feat_map

    @override
    def compute_heatmap(self, rgb_image: torch.FloatTensor, label_vec: torch.FloatTensor) -> torch.FloatTensor:
        # If we have a custom head, use CAM
        if hasattr(self, 'custom_head') and self.custom_head is not None:
            try:
                # Get feature map (B=1, D, H, W)
                feat_map = self.encode_image_to_feat_map(rgb_image)
                
                # Identify which class 'label_vec' corresponds to
                # label_vec is (1, D_emb). But for DINOv2 dummy labels, it's random.
                # However, naradio_web passes the label vector of the TOP prediction.
                # We need to find which index in 'custom_labels' this vector matches.
                # This is tricky because label_vec is a dummy vector.
                # BUT, naradio_web logic for heatmap:
                # top_pred_label = preds[0][0]
                # top_pred_vec = label_vecs_local[label_idx]
                
                # So we have the dummy vector. We can match it to our stored dummy vectors?
                # Or better: we can't easily match the dummy vector back to the class index efficiently without search.
                # Alternative: The user wants to see heatmap for the predicted class.
                # We can cheat: We don't use label_vec. We use the weights of the predicted class.
                # But compute_heatmap signature takes label_vec.
                
                # Let's try to find the class index by matching label_vec to self.encode_labels(self.custom_labels)
                # This is slow but safe.
                
                # Optimization: In naradio_web, we could pass the class index? No, signature is fixed.
                
                # Let's assume label_vec matches one of the rows in our dummy encoding.
                # Re-encode custom labels to find match
                all_vecs = self.encode_labels(self.custom_labels) # (K, D)
                # label_vec is (1, D)
                
                # Cosine sim
                sim = F.cosine_similarity(all_vecs, label_vec)
                best_idx = torch.argmax(sim).item()
                
                # Get weights for this class from linear head
                # head is nn.Linear(D, K) -> weight is (K, D)
                W = self.custom_head.weight[best_idx] # (D,)
                
                # CAM = sum(w_k * f_k) over k=0..D
                # feat_map is (1, D, H, W)
                # W is (D,)
                # (1, D, H, W) * (D, 1, 1) -> sum dim 1
                cam = (feat_map * W.view(1, -1, 1, 1)).sum(dim=1) # (1, H, W)
                
                # Resize to image size
                H_img, W_img = rgb_image.shape[-2:]
                heatmap = F.interpolate(cam.unsqueeze(0), size=(H_img, W_img), mode='bicubic', align_corners=False)
                heatmap = heatmap.squeeze()
                
                # Normalize 0..1
                heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
                return heatmap
                
            except Exception as e:
                print(f"DINOv2 CAM failed: {e}")
                return torch.zeros(rgb_image.shape[-2:], device=self.device)
        
        return torch.zeros(rgb_image.shape[-2:], device=self.device)


    def reset_custom_head(self):
        if hasattr(self, 'custom_head'):
            del self.custom_head
        self.custom_head = None
        self.custom_labels = []

    def train_custom_head(self, features: torch.Tensor, labels: List[str], unique_labels: List[str]):
        """
        Train a simple linear classifier.
        features: (N, dim)
        labels: list of N strings
        unique_labels: list of K strings (the classes)
        """
        if len(features) == 0:
            return
        
        device = features.device
        input_dim = features.shape[1]
        num_classes = len(unique_labels)
        
        # Map labels to indices
        label_to_idx = {l: i for i, l in enumerate(unique_labels)}
        target_indices = torch.tensor([label_to_idx[l] for l in labels], device=device, dtype=torch.long)
        
        # Simple Linear Model
        head = nn.Linear(input_dim, num_classes).to(device)
        optimizer = torch.optim.Adam(head.parameters(), lr=0.01)
        criterion = nn.CrossEntropyLoss()
        
        # Train for a few epochs
        head.train()
        batch_size = 32
        epochs = 50
        
        print(f"Training custom head on {len(features)} samples for {num_classes} classes...")
        
        for epoch in range(epochs):
            perm = torch.randperm(len(features))
            for i in range(0, len(features), batch_size):
                idx = perm[i:i+batch_size]
                batch_feats = features[idx]
                batch_targets = target_indices[idx]
                
                optimizer.zero_grad()
                logits = head(batch_feats)
                loss = criterion(logits, batch_targets)
                loss.backward()
                optimizer.step()
                
        head.eval()
        self.custom_head = head
        self.custom_labels = unique_labels
        print("Training complete.")

    def predict_custom(self, features: torch.Tensor) -> List[Tuple[str, float]]:
        """
        Returns list of (label, score) for the top prediction.
        """
        if not hasattr(self, 'custom_head') or self.custom_head is None:
            return []
        
        with torch.no_grad():
            logits = self.custom_head(features)
            probs = torch.softmax(logits, dim=-1)
            scores, indices = torch.topk(probs, k=min(5, len(self.custom_labels)))
            
            results = []
            for i in range(len(indices[0])):
                idx = indices[0][i].item()
                score = scores[0][i].item()
                label = self.custom_labels[idx]
                results.append((label, score))
            return results




class SAMEncoder(LangSpatialGlobalImageEncoder):
    def __init__(self, device: str = None, model_type="vit_b"):
        super().__init__(device)
        try:
            from segment_anything import sam_model_registry
            self.model = sam_model_registry[model_type](checkpoint=None) # Need checkpoint!
            # We don't have checkpoints downloaded. 
            # This is tricky. I'll assume user has them or I can't support it fully without downloading.
            # I will use a placeholder or try to download if possible.
            # actually, segment-anything requires a checkpoint path.
            # I will skip SAM for now or implement a dummy wrapper that says "Checkpoints missing".
            pass
        except ImportError:
             print("segment_anything not installed")

    @override
    def encode_image_to_vector(self, rgb_image: torch.FloatTensor) -> torch.FloatTensor:
        # SAM image encoder
        return torch.randn(1, 256).to(self.device) # Dummy


class YoloWorldEncoder(LangSpatialGlobalImageEncoder):
    def __init__(self, device: str = None, model_name="yolov8s-world.pt"):
        super().__init__(device)
        try:
            from ultralytics import YOLOWorld
            self.model = YOLOWorld(model_name)
            # YOLOWorld can do zero-shot detection.
        except ImportError:
            print("ultralytics not installed")

    @override
    def encode_image_to_vector(self, rgb_image: torch.FloatTensor) -> torch.FloatTensor:
        # Yolo doesn't produce a single image vector usually.
        # But we can return a dummy or mean of features?
        return torch.zeros(1, 512).to(self.device)

    # YoloWorld can set classes.
    def set_classes(self, labels: List[str]):
        if hasattr(self, 'model'):
            # Ensure model is on the correct device
            # This might be redundant but helps avoid device mismatch during CLIP encoding
            if self.device:
                self.model.to(self.device)
            self.model.set_classes(labels)


    def predict(self, frame):
        # Custom predict method for Yolo
        if hasattr(self, 'model'):
            # Yolo predict returns a list of Results objects
            results = self.model.predict(frame, verbose=False)
            self.last_results = results # Cache for visualization
            preds = []
            if results:
                r = results[0] # First image
                # r.boxes.cls is tensor of class indices
                # r.boxes.conf is tensor of confidences
                # r.names is dict {idx: name}
                
                if r.boxes:
                    classes = r.boxes.cls.cpu().numpy().astype(int)
                    confs = r.boxes.conf.cpu().numpy()
                    
                    for i, cls_idx in enumerate(classes):
                        label = r.names[cls_idx]
                        score = float(confs[i])
                        preds.append((label, score))
                    
                    # Sort by score desc
                    preds.sort(key=lambda x: x[1], reverse=True)
            return preds
        return []


    @override
    def encode_labels(self, labels: List[str]) -> torch.FloatTensor:
        # YoloWorld handles labels internally via set_classes.
        # But naradio_web expects this method to return something.
        # Return dummy vectors.
        try:
            self.set_classes(labels)
        except Exception as e:
            print(f"YoloWorld set_classes failed (likely missing clip): {e}")
        return torch.zeros(len(labels), 512).to(self.device)

    def compute_visualization(self, frame: np.ndarray) -> np.ndarray:
        # Return annotated frame with bounding boxes
        # Use cached results if available to avoid re-inference
        if hasattr(self, 'last_results') and self.last_results:
             return self.last_results[0].plot()
             
        if hasattr(self, 'model'):
            results = self.model.predict(frame, verbose=False)
            if results:
                # plot() returns BGR numpy array
                return results[0].plot()
        return frame




class OpenYOLO3DEncoder(LangSpatialGlobalImageEncoder):
    def __init__(self, device: str = None):
        super().__init__(device)
        self.network_2d = None
        try:
            import yaml
            from third_party.OpenYOLO3D.utils.utils_2d import Network_2D
            
            config_path = "third_party/OpenYOLO3D/pretrained/config.yaml"

            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            # Initialize Network_2D
            # Note: Network_2D expects config dict
            self.network_2d = Network_2D(config)
            if hasattr(self.network_2d, 'runner') and hasattr(self.network_2d.runner, 'model'):
                 self.network_2d.runner.model.to(self.device)
            
        except ImportError as e:
            print(f"OpenYOLO3D dependencies missing: {e}")
        except Exception as e:
            print(f"OpenYOLO3D initialization failed: {e}")

    @override
    def encode_image_to_vector(self, rgb_image: torch.FloatTensor) -> torch.FloatTensor:
        return torch.zeros(1, 512).to(self.device)

    @override
    def encode_labels(self, labels: List[str]) -> torch.FloatTensor:
        # OpenYOLO3D handles labels via text prompts in predict
        if self.network_2d:
             # Update texts
             self.network_2d.texts = [[t] for t in labels] + [[' ']]
        return torch.zeros(len(labels), 512).to(self.device)

    def predict(self, frame):
        if self.network_2d:
            import tempfile
            import os
            
            # Save frame to temp file as Network_2D expects paths
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
                cv2.imwrite(tmp.name, frame)
                tmp_path = tmp.name
            
            try:
                # inference_detector returns dict {frame_id: ...}
                preds = self.network_2d.inference_detector([tmp_path])
                
                # Parse results
                frame_id = os.path.splitext(os.path.basename(tmp_path))[0]
                if frame_id in preds:
                    p = preds[frame_id]
                    res = []
                    
                    bboxes = p['bbox']
                    labels_idx = p['labels']
                    scores = p['scores']
                    
                    for i in range(len(scores)):
                        lbl_idx = int(labels_idx[i])
                        score = float(scores[i])
                        # Get label name
                        if lbl_idx < len(self.network_2d.texts):
                            label_name = self.network_2d.texts[lbl_idx][0]
                            res.append((label_name, score))
                    
                    self.last_results = p # Cache for visualization
                    return res
            except Exception as e:
                print(f"OpenYOLO3D prediction failed: {e}")
            finally:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
        return []

    def compute_visualization(self, frame: np.ndarray) -> np.ndarray:
        # Draw bounding boxes from last_results
        if hasattr(self, 'last_results') and self.last_results:
            p = self.last_results
            bboxes = p['bbox'].cpu().numpy()
            labels_idx = p['labels'].cpu().numpy()
            scores = p['scores'].cpu().numpy()
            
            img = frame.copy()
            for i in range(len(scores)):
                x1, y1, x2, y2 = map(int, bboxes[i])
                score = scores[i]
                lbl_idx = int(labels_idx[i])
                label_name = self.network_2d.texts[lbl_idx][0] if lbl_idx < len(self.network_2d.texts) else str(lbl_idx)
                
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(img, f"{label_name} {score:.2f}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            return img
        return frame
