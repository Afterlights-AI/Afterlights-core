#!/bin/bash
cd "$(dirname "$0")/../src"
echo "Running from directory: $(pwd)"

python ../evaluation/evaluate.py
