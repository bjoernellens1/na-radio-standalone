# na-radio-standalone (demo)

This repository contains a standalone demo script that uses a RADIO/CLIP-like
image encoder to extract embeddings from a webcam and display label
predictions in realtime.

Key features:
- Webcam capture with OpenCV
- GPU detection and use (CUDA-enabled PyTorch) with fallback to CPU
- Tries to use RADIO from NVLabs (via `torch.hub`) and falls back to a
  ResNet-based encoder if RADIO isn't available
- A live PCA plot of recent embeddings

Getting started
--------------

1. Create a Python virtual environment (pip) and install packages. Adjust `cudatoolkit` and `pytorch` according to your CUDA version. For NVIDIA Tesla P100 (CUDA 11.7/11.8 supported), use a pip wheel that matches your CUDA or install CPU-only PyTorch.

  Example (pip venv, default path):

```bash
python3 -m venv .venv
source .venv/bin/activate
# Install a CUDA-enabled wheel if you have a compatible GPU; otherwise use --cpu in the helper below
./setup_pip_venv.sh --cuda 11.8
```
- The demo tries to load the RADIO model where possible. If unavailable or
  too heavy to load, it falls back to a ResNet-50 encoder for embeddings.
- For best performance with RADIO or large models, use an appropriate
  CUDA-enabled PyTorch installation and sufficient GPU memory.
- The fallback uses naive label embeddings if a proper text encoder is not
  present; this is meant to provide a functional demo but isn't a true
  zero-shot classification like CLIP.
- If you want real CLIP-like zero-shot behavior, install `open_clip` or a
  CLIP implementation, and the demo will use that when present.
  Install packages if needed:

  ```bash
  pip install einops open_clip_torch
  # or with pip: pip install einops
  ```

2. Run the webcam demo (default uses the `radio` model and device `cuda` if
   available):

```bash
python naradio.py --mode webcam --labels "person,car,dog,cat,tree"
```

Nix users can enter the nix shell and run the demo. Example (pip venv preferred):

```bash
nix-shell # starts the shell defined in shell.nix
# Option A: Use pip + venv (no conda) within the nix-shell (recommended for reproducible system libs via Nix):
./setup_pip_venv.sh --cuda 11.8  # or --cpu to install CPU-only PyTorch
source .venv/bin/activate
python naradio.py --mode webcam --labels "person,car,dog,cat,tree"

Note: This repository now uses a pip-based venv workflow instead of conda.
If you are on NixOS, create the venv from inside `nix-shell` so the system libraries (like libstdc++.so.6) are available. Example:

```bash
nix-shell
./setup_pip_venv.sh --cuda 11.8
source .venv/bin/activate
```

If you see an ImportError referencing `libstdc++.so.6` or `libgthread-2.0.so.0`, it often means the pip wheel depends on system libraries not present in your environment. Try `nix-shell` or install the appropriate system libraries as described in the Troubleshooting section.
```

3. Controls:
- Press `q` or Escape to exit

Notes and caveats
-----------------
- The demo tries to load the RADIO model where possible. If unavailable or
  too heavy to load, it falls back to a ResNet-50 encoder for embeddings.
- For best performance with RADIO or large models, use an appropriate
  CUDA-enabled PyTorch installation and sufficient GPU memory.
- The fallback uses naive label embeddings if a proper text encoder is not
  present; this is meant to provide a functional demo but isn't a true
  zero-shot classification like CLIP.

Contributing
------------
Add issues and PRs for improvements: better fallback text encoding,
web UI, integration with specific language models, and dataset examples.

Notes for NVIDIA Tesla P100
-------------------------
Tesla P100 GPUs are compatible with CUDA 11.x. To make the best use of this
demo and any heavy models (RADIO / CLIP), ensure to install a PyTorch build
linked to a CUDA toolkit compatible with your system. Example installation for
CUDA 11.8 is shown above; adjust it if your driver is older or different.

The RADIO models can be large; they may require more GPU memory. If you run
into OOM errors, try lowering `--input-resolution` or using a fallback encoder.

Using Tesla P100 (and older GPUs)
---------------------------------
If you see runtime warnings like "Tesla P100... is not compatible with the current PyTorch installation" this means the PyTorch build you installed was compiled for newer GPU architectures and your P100 (sm_60) isn't included. You have several options:

- Use CPU: Run `python naradio.py --mode webcam --device cpu` to avoid GPU errors.
  You can also use: `--force-cpu` to ensure the script doesn't attempt to use CUDA even if available:

  ```bash
  python naradio.py --mode webcam --device cpu --force-cpu --labels "person,car"
  ```
- Install a PyTorch binary that supports your GPU compute capability. For older GPUs (like P100), this may require installing an older PyTorch wheel or a wheel compiled for the CUDA toolkit that includes sm_60. Check the PyTorch archive on https://download.pytorch.org/whl/torch_stable.html and the "Get started" page for legacy builds.
  If you prefer to install a wheel for older architectures, choose a wheel that matches your CUDA and GPU compatibility. For example, if you have CUDA 11.6, pick a 'cu116' wheel. Use the official wheel links here:

  ```bash
  # This is an example; pick the exact version for your driver and cuda version
  pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 -f https://download.pytorch.org/whl/torch_stable.html
  ```
- Compile PyTorch from source with support for your GPU (this requires a suitable compiler and CUDA toolkit).
- If it's feasible, upgrade to a newer GPU that is supported by the latest PyTorch wheels (sm_70/75/80+).

Troubleshooting OpenCV / NixOS issues
------------------------------------
If you run into an error similar to "ImportError: libgthread-2.0.so.0: cannot open shared object file", this indicates a missing glib dependency for the OpenCV Python package in your environment. On NixOS, the recommended approaches are:

- Use a Nix-enabled Python env or package that bundles OpenCV with the correct glib, or add `glib` to your shell environment, for example in a `shell.nix`/`flake.nix`.
- Use `opencv-python-headless` in the Python environment if you do not need the GUI windows and prefer to avoid system GUI libraries; note that webcam capture may still require system libs for video input.
- On Debian/Ubuntu: `sudo apt-get install libglib2.0-0` will provide the missing library.
- For system packaging on Debian/Ubuntu, consider installing OpenCV via apt (`sudo apt-get install libopencv-dev python3-opencv`) or using `opencv-python` from pip. On NixOS, rely on the `shell.nix` for system OpenCV.

NixOS example
--------------
If you are on NixOS, add `glib` and `opencv` to your `shell.nix` or development shell to ensure the correct system libraries are present. A minimal `shell.nix` example:

```nix
{ pkgs ? import <nixpkgs> {} }:
pkgs.mkShell {
  buildInputs = [ pkgs.python310 pkgs.opencv pkgs.glib pkgs.ffmpeg ];
  shellHook = ''
    echo "Nix shell for naradio demo: OpenCV + glib available"
  '';
}
```

You can then enter the shell with `nix-shell` and run the demo (or use `nix develop`/`flake` style if preferred).

If you still see issues, please check whether your OpenCV installation supports Video4Linux (v4l) for the camera capture on Linux.


