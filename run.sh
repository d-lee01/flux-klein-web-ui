#!/bin/bash
set -e

# Navigate to script directory
cd "$(dirname "$0")"

echo "Checking dependencies directly (pip install -e flux2)..."
# We assume flux2 is already installed in editable mode or user has run installation commands
# But for safety we can try to install requirement if missing, but let's just warn.

if ! python3 -c "import flux2" &> /dev/null; then
    echo "Warning: flux2 module not found. Installing..."
    pip install -r requirements.txt
    pip install -e flux2
fi

echo "Starting Flux 2.1 Klein Server..."
echo "NOTE: On the first run, this will download ~10-15GB of model weights."
echo "Please wait..."

python3 app_server.py
