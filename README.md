# na-radio-standalone

Standalone RADIO/CLIP webcam demo with a lightweight Flask UI. The app streams
frames from a webcam (or video file), encodes them with the RADIO model when
available, falls back to CLIP/open_clip, and finally a ResNet-50 encoder if
nothing else can run.

## Highlights
- OpenCV capture with live label predictions and optional PCA overlay.
- Automatic encoder selection with CUDA/CPU fallbacks.
- Flask web front-end (`docker compose up` exposes http://localhost:5000).
- Persistent Torch/Hugging Face caches when running through Docker Compose.

## Quick start (pip venv)

1. Create a virtual environment and install PyTorch that matches your GPU (or
   CPU-only wheel). The helper script can do this for you:

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   # Examples:
   ./setup_pip_venv.sh --cuda 11.8                # installs CUDA 11.8 wheel
   ./setup_pip_venv.sh --cuda 11.6 --cpu          # force CPU-only build
   ./setup_pip_venv.sh --cuda 11.6 --torch-version 1.13.1+cu116
   ./setup_pip_venv.sh --cuda 11.6 --torch-wheel-url https://download.pytorch.org/whl/cu116/torch-1.13.1%2Bcu116-cp310-cp310-manylinux_2_17_x86_64.whl
   ```

2. Install optional extras for better zero-shot performance:

   ```bash
   pip install einops open_clip_torch transformers
   ```

3. Run the CLI demo (defaults to the RADIO encoder on CUDA when possible):

   ```bash
   python naradio.py --mode webcam --labels "person,car,dog,cat,tree"
   ```

### Notes for NixOS or pip-on-Nix
- Use `nix-shell` (see `shell.nix`) to ensure required system libraries such as
  `libstdc++` and `libgthread` are available before creating your pip venv.
- OpenCV requirements differ per distro; for headless runs use
  `opencv-python-headless`, otherwise install system GUI libs.

## Docker / Docker Compose

The project supports both Intel (CPU/XPU) and NVIDIA (CUDA) architectures via a multi-stage Dockerfile.

### Prebuilt Images

Prebuilt images are available on GHCR:
- **Intel/CPU**: `ghcr.io/bjoernellens1/na-radio-standalone:intel-optimized`
- **NVIDIA/CUDA**: `ghcr.io/bjoernellens1/na-radio-standalone:cuda`

### Building the Image

You can build the image for your specific target architecture using the `TARGET_ARCH` build argument.

**Intel (CPU / XPU)**:
```bash
docker build --build-arg TARGET_ARCH=intel -t na-radio-intel .
```

**NVIDIA (CUDA)**:
```bash
docker build --build-arg TARGET_ARCH=nvidia -t na-radio-nvidia .
```

### Running with Docker Compose

We recommend using `docker compose` for local development. You can set the `TARGET_ARCH` in your `.env` file or pass it as an environment variable.

1. **Intel Mode (Default)**:
   ```bash
   TARGET_ARCH=intel docker compose up
   ```
   *Note: For Intel GPU support, ensure `/dev/dri` is mounted (default in `docker-compose.override.yml`).*

2. **NVIDIA Mode**:
   ```bash
   TARGET_ARCH=nvidia docker compose -f docker-compose.yml -f docker-compose.override.yml -f docker-compose.gpu.yml up
   ```
   *Note: Ensure you have the NVIDIA Container Toolkit installed.*

### Customizing Base Images
You can override the base images if you need specific versions (e.g., for legacy hardware like Tesla P100):

```bash
# Example: Build for NVIDIA P100 (needs CUDA 11.x)
docker build \
  --build-arg TARGET_ARCH=nvidia \
  --build-arg NVIDIA_BASE_IMAGE=pytorch/pytorch:1.13.1-cuda11.6-cudnn8-runtime \
  -t na-radio-p100 .
```

## Troubleshooting

- **Intel IPEX Errors**: Ensure you are using the correct `TARGET_ARCH=intel` build.
- **CUDA Errors**: Ensure you are using `TARGET_ARCH=nvidia` and have passed the GPU to the container (`--gpus all` or via compose).
- **Webcam**: If no webcam is found, map the device correctly (e.g., `/dev/video0`) or use a video file via `VIDEO_FILE` env var.

## Contributing

Issues and pull requests are welcome. Useful areas:
- Better fallbacks or new language adaptors.
- Web UI polish (status indicators, controls).
- Tests for encoder edge cases and preprocessing.
