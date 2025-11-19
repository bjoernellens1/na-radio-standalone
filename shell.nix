{ pkgs ? import <nixpkgs> {} }:
pkgs.mkShell {
  buildInputs = [ pkgs.python310 pkgs.glib pkgs.opencv pkgs.ffmpeg pkgs.v4l-utils pkgs.pkg-config pkgs.libGL pkgs.gcc pkgs.pkg-config];
  nativeBuildInputs = [ pkgs.git pkgs.wget pkgs.cmake pkgs.gnumake ];
  shellHook = ''
    echo "Nix shell for naradio demo: OpenCV, glib available. Python interpreter: ${pkgs.python310}" 
  echo "This shell will create a Python venv at .venv and attempt to install pip requirements if missing."
  echo "Recommended workflow inside the shell: use a pip venv to create a local .venv with a suitable PyTorch wheel and install the requirements (example below):"
  echo "Example (recommended for Tesla P100):"
  echo "  ./setup_pip_venv.sh --cuda 11.8"
  echo "  source .venv/bin/activate"

    # Auto-create a Python venv and pip install requirements to replicate pip environment within nix-shell
    if [ ! -d .venv ]; then
      echo "Creating a Python venv at .venv and installing pip requirements..."
      python -m venv .venv
      . .venv/bin/activate
      python -m pip install --upgrade pip
  python -m pip install -r requirements.txt || echo "pip install failed; try using ./setup_pip_venv.sh or adjust your build inputs for system libs"
    else
      . .venv/bin/activate
    fi
  '';
}