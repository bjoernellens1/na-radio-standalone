ARG TARGET_ARCH=intel
# Define base images
ARG INTEL_BASE_IMAGE=python:3.10-slim
ARG NVIDIA_BASE_IMAGE=pytorch/pytorch:2.2.2-cuda11.8-cudnn8-runtime

# Intermediate stage to select base image
FROM ${INTEL_BASE_IMAGE} as base-intel
ENV ARCH_TYPE=intel

FROM ${NVIDIA_BASE_IMAGE} as base-nvidia
ENV ARCH_TYPE=nvidia

# Final stage
FROM base-${TARGET_ARCH} as final

# Install system packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements
COPY requirements-base.txt /app/requirements-base.txt
COPY requirements-intel.txt /app/requirements-intel.txt

# Install dependencies based on ARCH_TYPE
RUN python -m pip install --upgrade pip && \
    if [ "$ARCH_TYPE" = "intel" ]; then \
        echo "Installing Intel dependencies..." && \
        python -m pip install -r /app/requirements-intel.txt && \
        python -m pip install -r /app/requirements-base.txt; \
    else \
        echo "Installing Base dependencies for NVIDIA build..." && \
        python -m pip install -r /app/requirements-base.txt; \
    fi

# Emit build info
RUN echo "Build Architecture: ${ARCH_TYPE}" && \
    python - <<'PY'
import torch, sys
try:
    import intel_extension_for_pytorch as ipex
    print("IPEX version:", ipex.__version__)
except ImportError:
    print("IPEX not found")
print("Torch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("Python:", sys.version.replace("\n", " "))
PY

# Copy application code
COPY . /app

EXPOSE 5000

ENV PYTHONUNBUFFERED=1

ENTRYPOINT ["python", "-m", "web.naradio_web"]
CMD ["--host", "0.0.0.0", "--port", "5000"]
