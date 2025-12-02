import torch
import sys
import os

# Mock IPEX if not installed (for local verification if running on non-Intel machine)
# But we want to verify the logic.

print("Verifying imports...")
try:
    from utils import get_device, optimize_model
    print("utils imported successfully")
except ImportError as e:
    print(f"Failed to import utils: {e}")
    sys.exit(1)

try:
    from encoders import NARadioEncoder
    print("encoders imported successfully")
except ImportError as e:
    print(f"Failed to import encoders: {e}")
    sys.exit(1)

print(f"Detected device: {get_device()}")

# Check if OpenYOLO3D is really gone
try:
    from encoders import OpenYOLO3DEncoder
    print("ERROR: OpenYOLO3DEncoder still exists!")
    sys.exit(1)
except ImportError:
    print("OpenYOLO3DEncoder correctly removed.")

print("Verification passed.")
