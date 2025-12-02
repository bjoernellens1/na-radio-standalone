ARG TARGET_ARCH=intel
ARG INTEL_BASE_IMAGE=python:3.10-slim-bullseye
ARG NVIDIA_BASE_IMAGE=pytorch/pytorch:2.2.2-cuda11.8-cudnn8-runtime

# --- Intel Builder Stage ---
FROM ${INTEL_BASE_IMAGE} as build-intel
ENV ARCH_TYPE=intel
WORKDIR /app

# Install system packages for Intel
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements-base.txt /app/requirements-base.txt
COPY requirements-intel.txt /app/requirements-intel.txt
RUN python -m pip install --upgrade pip && \
    python -m pip install -r /app/requirements-intel.txt && \
    python -m pip install -r /app/requirements-base.txt

# Fix for "cannot enable executable stack" error with IPEX
RUN apt-get update && apt-get install -y execstack && \
    find /usr/local/lib/python3.10/site-packages/intel_extension_for_pytorch -name "*.so" -exec execstack -c {} \; && \
    rm -rf /var/lib/apt/lists/*

# --- NVIDIA Builder Stage ---
FROM ${NVIDIA_BASE_IMAGE} as build-nvidia
ENV ARCH_TYPE=nvidia
WORKDIR /app

# Install system packages for NVIDIA (Ubuntu based usually)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies (Torch already present)
COPY requirements-base.txt /app/requirements-base.txt
RUN python -m pip install --upgrade pip && \
    python -m pip install -r /app/requirements-base.txt

# --- Final Stage ---
FROM build-${TARGET_ARCH} as final
WORKDIR /app

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
