#!/bin/bash
set -e

if [ "$USE_ZLUDA" = "1" ] || [ "$USE_ZLUDA" = "true" ]; then
    echo "Enabling ZLUDA support..."
    export LD_LIBRARY_PATH=/opt/zluda/lib:$LD_LIBRARY_PATH
    echo "LD_LIBRARY_PATH set to: $LD_LIBRARY_PATH"
fi

exec "$@"
