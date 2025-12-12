import sys
import pathlib

# Add project root to path
proj_root = pathlib.Path(__file__).resolve().parent.parent
if str(proj_root) not in sys.path:
    sys.path.insert(0, str(proj_root))

from web.naradio_web import app, get_manager

# Initialize Manager
get_manager()

# Apply ProxyFix (already in naradio_web.py, but good to double check)
from werkzeug.middleware.proxy_fix import ProxyFix
app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1, x_prefix=1)

if __name__ == "__main__":
    app.run()
