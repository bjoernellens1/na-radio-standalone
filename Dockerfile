ARG PYTORCH_BASE_IMAGE=pytorch/pytorch:2.2.2-cuda11.8-cudnn8-runtime
FROM ${PYTORCH_BASE_IMAGE}

# Install additional system packages required by the app
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY requirements.txt /app/requirements.txt
RUN python -m pip install --upgrade pip && \
    python -m pip install -r /app/requirements.txt --extra-index-url https://download.pytorch.org/whl/cu118

# Emit build info for easier debugging
RUN echo "Using base image: ${PYTORCH_BASE_IMAGE}" && \
    python - <<'PY'
import torch, sys
print("Torch version:", torch.__version__)
print("Torch CUDA build:", torch.version.cuda)
print("CUDA available at build time:", torch.cuda.is_available())
print("Python:", sys.version.replace("\n", " "))
PY

# Copy application code
COPY . /app

EXPOSE 5000

ENV PYTHONUNBUFFERED=1

ENTRYPOINT ["python", "-m", "web.naradio_web"]
CMD ["--host", "0.0.0.0", "--port", "5000"]
