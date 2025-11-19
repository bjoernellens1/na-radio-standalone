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

```bash
# Build the image
docker build -t naradio-demo:latest --build-arg PYTORCH_CUDA=cpu .

# Simple run with GPU and webcam
docker run --gpus all --rm \
  -p 5000:5000 \
  --device /dev/video0:/dev/video0 \
  -v /dev/dri:/dev/dri \
  naradio-demo:latest
```

Compose is preferred for local development (adds caches and device mappings):

```bash
# CPU / generic run
docker compose up --build

# With webcam + GPU overrides
docker compose -f docker-compose.yml \
  -f docker-compose.override.yml \
  -f docker-compose.gpu.yml up --build
```

### Customizing the PyTorch base image
The Docker image now inherits from `pytorch/pytorch`. Override
`PYTORCH_BASE_IMAGE` (via `.env` or `--build-arg`) to match your GPU/driver.
Example for Tesla P100 (needs CUDA 11.6 with `sm_60` support):

```bash
docker compose -f docker-compose.yml -f docker-compose.override.yml -f docker-compose.gpu.yml \
  build --build-arg PYTORCH_BASE_IMAGE=pytorch/pytorch:1.13.1-cuda11.6-cudnn8-runtime
```

You can also set the same value in `.env`:

```
PYTORCH_BASE_IMAGE=pytorch/pytorch:1.13.1-cuda11.6-cudnn8-runtime
```

After the container is running, you can confirm CUDA visibility with:

```bash
docker exec -it na-radio-standalone-naradio-1 python - <<'PY'
import torch
print("cuda?", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device", torch.cuda.get_device_name(0), "capability", torch.cuda.get_device_capability(0))
PY
```

The compose files mount two volumes:

| Volume        | Container path             | Purpose                                 |
|---------------|----------------------------|-----------------------------------------|
| `torch-cache` | `/root/.cache/torch`       | TorchHub repo + RADIO checkpoints       |
| `hf-cache`    | `/root/.cache/huggingface` | Transformer tokenizers / secondary data |

Keep these volumes around to avoid multi-GB downloads on every start.

### Forcing GPU usage (legacy cards)
The container reads a few optional environment variables before launching:

- `ENCODER_DEVICE` (`cuda` / `cpu`) to explicitly pick the encoder device.
- `FORCE_GPU` (`1`/`true`) to skip compute-capability safety checks.
- `MIN_CC` (e.g., `6.0`) to set the minimum CUDA compute capability used by the safety check.
- `GPU_DEVICE` (default `/dev/nvidia0`) to map a specific GPU device into the container if you use manual `--device` rules instead of `gpus: all`.

Example for Tesla P100 (compute capability 6.0):

```bash
export ENCODER_DEVICE=cuda
export FORCE_GPU=1
export MIN_CC=6.0
docker compose -f docker-compose.yml -f docker-compose.override.yml -f docker-compose.gpu.yml up --build
```

## Older GPUs (Tesla P100, etc.)

Recent PyTorch wheels drop support for `sm_60`. If you see warnings about
unsupported compute capability, pick an older `pytorch/pytorch` base (e.g.,
`pytorch/pytorch:1.13.1-cuda11.6-cudnn8-runtime`) or run everything in CPU mode:

```bash
python naradio.py --mode webcam --device cpu --force-cpu
```

You can also pass `--min-cc 6.0` to `naradio.py` to relax the automatic CUDA
compatibility check if you know your wheel/image is compatible.

## Troubleshooting

- **Missing `libgthread-2.0.so.0` / other OpenCV deps:** install
  `libglib2.0-0` (Debian/Ubuntu) or add `glib` via Nix.
- **No webcam available:** Use `--video-file path.mp4` or inject the file via
  the `VIDEO_FILE` environment variable for the web app.
- **RADIO downloads every run:** ensure the cache volumes exist (Compose) or
  mount `~/.cache/torch` inside the container manually.
- **Label encoding failures:** install `transformers` (already listed in
  `requirements.txt`) so the RADIO adaptor tokenizer can load.

## Contributing

Issues and pull requests are welcome. Useful areas:
- Better fallbacks or new language adaptors.
- Web UI polish (status indicators, controls).
- Tests for encoder edge cases and preprocessing.
