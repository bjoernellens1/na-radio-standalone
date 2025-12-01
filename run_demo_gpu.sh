#!/usr/bin/env bash
# Script to run na-radio with GPU support using docker run directly
# Bypasses podman-compose issues with GPU flags

# Ensure the image is built
docker compose build naradio

# Run the container
# --gpus all: Enable all GPUs
# --net host: Use host networking (simplest for local demos) or map ports
# -v: Map cache directories
# --env-file: Load environment variables
echo "Starting na-radio with GPU support..."
docker run --rm -it \
  --gpus all \
  -p 5001:5000 \
  --env-file .env \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -v torch-cache:/root/.cache/torch \
  -v hf-cache:/root/.cache/huggingface \
  -v /dev/video0:/dev/video0 \
  ghcr.io/na-radio-standalone/naradio:latest
