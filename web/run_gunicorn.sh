#!/bin/bash
# Install gunicorn if not present
./.venv/bin/python -m pip install gunicorn

# Run gunicorn
# -w 1: 1 worker (since we have stateful manager/camera access, multiple workers might be problematic)
# --threads 4: Multi-threaded for concurrent requests
# -b 0.0.0.0:5000: Bind address
# --access-logfile -: Log to stdout
# --timeout 120: Increase timeout for long requests
./.venv/bin/python -m gunicorn -w 1 --threads 4 -b 0.0.0.0:5000 --timeout 120 web.wsgi:app
