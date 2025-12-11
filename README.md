# na-radio-standalone

Standalone RADIO/CLIP webcam demo with a lightweight Flask UI. The app streams
frames from a webcam (or video file), encodes them with the RADIO model when
available, falls back to CLIP/open_clip, and finally a ResNet-50 encoder if
nothing else can run.

## Highlights
- **Modular Architecture**: Core logic separated into `na_radio` package.
- **CLI & WebUI**: Run headless via CLI or with a Flask-based WebUI.
- **Multi-Model Support**: RADIO, CLIP, SigLIP, DINOv2, DINOv3, Yolo-World.
- **Automatic Fallbacks**: CUDA -> CPU, RADIO -> CLIP -> ResNet.

## Installation

1. Create a virtual environment:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

2. Install dependencies (helper script available):
   ```bash
   ./setup_pip_venv.sh --cuda 11.8
   ```

3. Install optional extras:
   ```bash
   pip install einops open_clip_torch transformers
   ```

## Usage

### CLI (Command Line Interface)

The project now includes a robust CLI for running inference without the WebUI.

```bash
# Run with default settings (webcam 0, RADIO model)
python -m na_radio.cli

# Process a video file
python -m na_radio.cli --input my_video.mp4 --model dinov2

# Process an image folder
python -m na_radio.cli --input /path/to/images --duration 10

# Specify device and labels
python -m na_radio.cli --device cuda --labels "cat,dog,person"
```

**Arguments:**
- `--input`: Input source. Integer for webcam index (e.g., `0`), string for file path (video or folder).
- `--model`: Model to use (`radio`, `clip`, `dinov2`, `yolo`, etc.).
- `--labels`: Comma-separated list of labels for zero-shot classification.
- `--device`: Force device (`cuda`, `cpu`).
- `--duration`: Run duration in seconds (0 for infinite).

### WebUI

The WebUI provides a visual interface for the application.

```bash
python web/naradio_web.py
```
Access at `http://localhost:5000`.

### Scripts

The `scripts/` directory contains utilities for evaluation and comparisons.

- **Resolution Study**: Evaluate model performance across different resolutions.
  ```bash
  python scripts/comparisons/resolution_study.py --encoder radio --output-dir results/
  ```

- **Embedding Comparison**: Compare embeddings from different models.
  ```bash
  python scripts/comparisons/embedding_comparison.py --models radio clip --resolutions 512
  ```

- **Semantic Evaluation**: Run semantic segmentation evaluation (requires dataset).
  ```bash
  python scripts/evaluate_semantic.py --dataset /path/to/ade20k
  ```

## API Documentation

The core logic is available in the `na_radio` package.

### `na_radio.manager.Manager`

The central class that orchestrates capture and inference.

```python
from na_radio.manager import Manager

# Initialize
manager = Manager(device_index=0, encoder_name='radio')

# Start background threads
manager.start()

# Access status
status = manager.get_status()
print(status['predictions'])

# Change model dynamically
manager.load_model('dinov2')

# Stop
manager.stop()
```

### `na_radio.encoders`

Contains encoder implementations.

- `load_encoder(preferred, device, ...)`: Factory function to load encoders.
- `NARadioEncoder`: Wrapper for NVlabs RADIO.
- `CLIPFallbackEncoder`: Wrapper for OpenCLIP.
- `DINOv2Encoder`, `SigLIPEncoder`, etc.

### `na_radio.utils`

Utility functions.

- `get_device()`: Returns best available device (xpu > cuda > cpu).
- `preprocess_frame(frame, resolution)`: Prepares image for encoder.

## Docker

See `docker-compose.yml` for running with Docker.

```bash
TARGET_ARCH=nvidia docker compose up
```
