ARG PYTORCH_BASE_IMAGE=pytorch/pytorch:2.2.2-cuda11.8-cudnn8-runtime
FROM ${PYTORCH_BASE_IMAGE}

# Install additional system packages required by the app
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 git \
    && rm -rf /var/lib/apt/lists/*


WORKDIR /app

# Install Python dependencies
COPY requirements.txt /app/requirements.txt
RUN python -m pip install --upgrade pip && \
    python -m pip install -r /app/requirements.txt --extra-index-url https://download.pytorch.org/whl/cu118 && \
    python -m pip install openvino openvino-dev

# Install ROCm and ZLUDA for AMD support
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget gnupg2 curl ca-certificates && \
    mkdir --parents --mode=0755 /etc/apt/keyrings && \
    wget https://repo.radeon.com/rocm/rocm.gpg.key -O - | gpg --dearmor | tee /etc/apt/keyrings/rocm.gpg > /dev/null && \
    echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/rocm.gpg] https://repo.radeon.com/rocm/apt/6.1.3 jammy main" | tee /etc/apt/sources.list.d/rocm.list && \
    echo -e 'Package: *\nPin: release o=repo.radeon.com\nPin-Priority: 600' | tee /etc/apt/preferences.d/rocm-pin-600 && \
    apt-get update && \
    apt-get install -y --no-install-recommends rocm-hip-runtime-dev && \
    rm -rf /var/lib/apt/lists/*

# Download ZLUDA (v3)
RUN mkdir -p /opt/zluda && \
    wget https://github.com/vosen/ZLUDA/releases/download/v3/zluda-linux-amd64.tar.gz -O /tmp/zluda.tar.gz || \
    wget https://github.com/vosen/ZLUDA/releases/download/v3/zluda-v3-linux-amd64.tar.gz -O /tmp/zluda.tar.gz && \
    tar -xvf /tmp/zluda.tar.gz -C /opt/zluda --strip-components=1 && \
    rm /tmp/zluda.tar.gz


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
RUN chmod +x /app/entrypoint.sh

EXPOSE 5000

ENV PYTHONUNBUFFERED=1

ENTRYPOINT ["/app/entrypoint.sh"]
CMD ["python", "-m", "web.naradio_web", "--host", "0.0.0.0", "--port", "5000"]

