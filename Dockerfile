ARG PYTHON_BASE_IMAGE=python:3.10-slim
FROM ${PYTHON_BASE_IMAGE}

# Install system packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY requirements.txt /app/requirements.txt
RUN python -m pip install --upgrade pip && \
    python -m pip install -r /app/requirements.txt

# Emit build info
RUN echo "Using base image: ${PYTHON_BASE_IMAGE}" && \
    python - <<'PY'
import torch, sys
try:
    import intel_extension_for_pytorch as ipex
    print("IPEX version:", ipex.__version__)
except ImportError:
    print("IPEX not found")
print("Torch version:", torch.__version__)
print("Python:", sys.version.replace("\n", " "))
PY

# Copy application code
COPY . /app

EXPOSE 5000

ENV PYTHONUNBUFFERED=1

ENTRYPOINT ["python", "-m", "web.naradio_web"]
CMD ["--host", "0.0.0.0", "--port", "5000"]
