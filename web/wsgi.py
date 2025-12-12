import sys
import pathlib

# Add project root to path
proj_root = pathlib.Path(__file__).resolve().parent.parent
if str(proj_root) not in sys.path:
    sys.path.insert(0, str(proj_root))

from web.naradio_web import app, get_manager

# Initialize Manager
get_manager()

if __name__ == "__main__":
    app.run()
