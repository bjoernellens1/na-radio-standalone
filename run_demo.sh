#!/usr/bin/env bash
# Small convenience script to run the demo using the local venv if present
set -euo pipefail

# Prefer to activate local venv if present; otherwise run the emulator with system Python
if [[ -d .venv ]]; then
  echo "Activating venv at .venv"
  . .venv/bin/activate
else
  echo "No local .venv found, running system python. Consider running './setup_pip_venv.sh' to create a venv first."
fi

python naradio.py --mode webcam
