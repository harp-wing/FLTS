#!/bin/sh

# Exit immediately if a command exits with a non-zero status.
set -e

echo "Entrypoint: Starting training as appuser..."docker run --rm -it -v barcelona_cell_data:/data alpine ls -l /data

chown -R appuser:appuser /app/data || echo "Warning: Could not chown /app/data, permissions may cause errors."

exec gosu appuser "$@"
