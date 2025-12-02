#!/usr/bin/env bash
set -euo pipefail

# Helper script to create a python venv and install pip requirements.
# Usage:
#   ./setup_pip_venv.sh [--cuda 11.8] [--cpu] [--venv .venv]
# If --cpu is given, install CPU-only PyTorch. If --cuda is given, try to install matching torch wheel.

VENV_DIR=${VENV_DIR:-.venv}
CUDA_TOOLKIT=${CUDA_TOOLKIT:-11.8}
FORCE_CPU=0
TORCH_WHEEL_URL=""
TORCH_VERSION=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --cuda)
      CUDA_TOOLKIT="$2"; shift 2;;
    --cpu)
      FORCE_CPU=1; shift;;
    --venv)
      VENV_DIR="$2"; shift 2;;
    --torch-wheel-url)
      TORCH_WHEEL_URL="$2"; shift 2;;
    --torch-version)
      TORCH_VERSION="$2"; shift 2;;
    *)
      echo "Usage: $0 [--cuda 11.8] [--cpu] [--venv .venv]"; exit 1;;
  esac
done

python3 -m venv ${VENV_DIR}
source ${VENV_DIR}/bin/activate
python -m pip install --upgrade pip

if [[ "$FORCE_CPU" -eq 1 ]]; then
  echo "Installing CPU-only PyTorch (pip)"
  pip install --upgrade pip
  pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cpu
else
  echo "Attempting to install GPU PyTorch for CUDA ${CUDA_TOOLKIT}"
  # Install matching wheel for the specified CUDA. This isn't perfect for all GPU compute capabilities,
  # but provides a starting point for pip-only installs. Adjust versions if necessary.
  pip install --upgrade pip
  EXTRA_INDEX_URL="https://download.pytorch.org/whl/cu${CUDA_TOOLKIT//./}"
    if [[ -n "${TORCH_WHEEL_URL}" ]]; then
      echo "Installing PyTorch wheel from specified URL: ${TORCH_WHEEL_URL}"
      pip install "${TORCH_WHEEL_URL}"
    elif [[ -n "${TORCH_VERSION}" ]]; then
      echo "Installing PyTorch version ${TORCH_VERSION} for CUDA ${CUDA_TOOLKIT}"
  pip install "torch==${TORCH_VERSION}" "torchvision==${TORCH_VERSION}" --extra-index-url "${EXTRA_INDEX_URL}"
    else
      case "${CUDA_TOOLKIT}" in
      11.8|11.7|11.6|11.3|11.1)
  pip install torch torchvision --extra-index-url "${EXTRA_INDEX_URL}"
        ;;
      12.8|12.6)
  pip install torch torchvision --extra-index-url "${EXTRA_INDEX_URL}"
        ;;
      *)
        echo "CUDA ${CUDA_TOOLKIT} not recognized; trying default wheels (may be cuda-compiled or cpu-only)"
        pip install torch torchvision -f https://download.pytorch.org/whl/torch_stable.html
        ;;
      esac
    fi
fi

echo "Installing other pip requirements"
pip install -r requirements.txt

echo "Virtualenv ready. Activate with: source ${VENV_DIR}/bin/activate"

# Quick post-install sanity check for common NixOS issues (missing libstdc++). If the import fails,
# print helpful instructions for Nix users or Debian/Ubuntu users.
set +e
source ${VENV_DIR}/bin/activate
python - <<'PY'
import sys
try:
    import numpy
except Exception as e:
    print('Warning: numpy import failed after pip install. See error below:')
    print(e)
    # detect common missing lib
    msg = str(e)
    if 'libstdc++.so.6' in msg or 'libgthread-2.0.so.0' in msg:
        print('\nIt looks like a system shared library is missing. On NixOS, ensure you')
        print('entered `nix-shell` which sets up system libraries for wheels, or run')
        print('this script inside the `nix-shell` to ensure the libraries are available. You can run:')
        print('  nix-shell --run "./setup_pip_venv.sh --cuda <CUDA_VERSION>"')
        print('\nOr if you are not on NixOS, install the missing system packages:')
        print('  Debian/Ubuntu: sudo apt-get install libstdc++6 libglib2.0-0')
    sys.exit(0)
print('numpy import OK')
PY
set -e
