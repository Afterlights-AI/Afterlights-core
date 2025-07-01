#!/bin/bash

# Navigate to the script's directory (optional)
cd "$(dirname "$0")/../src"
echo "Running from directory: $(pwd)"
# Run uvicorn with the specified app
uvicorn api:app --port 8755