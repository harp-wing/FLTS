#!/bin/sh

# Exit immediately if a command exits with a non-zero status.
set -e

# This script is now running as root.
# We can now correctly change ownership of the mounted volume.
echo "Entrypoint: Taking ownership of /app/data..."
chown -R appuser:appuser /app/data

# Drop privileges and execute the main command (CMD) as the 'appuser'.
# The "$@" is a shell variable that represents all the arguments passed to the script.
# In our case, it will be ["uvicorn", "main:app", "--host=0.0.0.0", "--port=8000"]
echo "Entrypoint: Starting application as appuser..."
exec gosu appuser "$@"