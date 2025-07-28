#!/usr/bin/env python3
"""
Command-line interface wrappers for Afterlights-core scripts.
These functions provide console script entry points for the shell scripts.
"""

import os
import sys
import subprocess
from pathlib import Path


def get_project_root():
    """Get the project root directory."""
    current_file = Path(__file__).resolve()
    # Go up from src/cli.py to project root
    return current_file.parent.parent


def train_cli():
    """Console script entry point for training."""
    project_root = get_project_root()
    train_script = project_root / "src" / "train.py"
    
    if not train_script.exists():
        print(f"Error: {train_script} not found", file=sys.stderr)
        sys.exit(1)
    
    try:
        subprocess.run([sys.executable, str(train_script)], cwd=project_root, check=True)
    except subprocess.CalledProcessError as e:
        sys.exit(e.returncode)


def retrieve_cli():
    """Console script entry point for retrieval."""
    project_root = get_project_root()
    retrieve_script = project_root / "src" / "retrieve.py"
    
    # Default parameters from retrieve.sh
    file_path = os.environ.get("AFTERLIGHTS_FILE_PATH", "dataset/path")
    model_output_path = os.environ.get("AFTERLIGHTS_MODEL_PATH", "trained_model/path")
    top_k = os.environ.get("AFTERLIGHTS_TOP_K", "3")
    query = os.environ.get("AFTERLIGHTS_QUERY", "query")
    mode = os.environ.get("AFTERLIGHTS_MODE", "contextual")
    
    if not retrieve_script.exists():
        print(f"Error: {retrieve_script} not found", file=sys.stderr)
        sys.exit(1)
    
    cmd = [
        sys.executable, str(retrieve_script),
        "--file_path", file_path,
        "--model_output_path", model_output_path,
        "--query", query,
        "--top_k", top_k,
        "--mode", mode,
        "--qdrant"
    ]
    
    try:
        subprocess.run(cmd, cwd=project_root, check=True)
    except subprocess.CalledProcessError as e:
        sys.exit(e.returncode)


def evaluate_cli():
    """Console script entry point for evaluation."""
    project_root = get_project_root()
    eval_script = project_root / "evaluation" / "evaluate.py"
    
    if not eval_script.exists():
        print(f"Error: {eval_script} not found", file=sys.stderr)
        sys.exit(1)
    
    try:
        subprocess.run([sys.executable, str(eval_script)], cwd=project_root, check=True)
    except subprocess.CalledProcessError as e:
        sys.exit(e.returncode)


def serve_cli():
    """Console script entry point for serving the API."""
    project_root = get_project_root()
    src_dir = project_root / "src"
    
    if not (src_dir / "api.py").exists():
        print(f"Error: {src_dir / 'api.py'} not found", file=sys.stderr)
        sys.exit(1)
    
    port = os.environ.get("AFTERLIGHTS_PORT", "8755")
    
    try:
        subprocess.run([
            "uvicorn", "api:app", "--port", port
        ], cwd=src_dir, check=True)
    except subprocess.CalledProcessError as e:
        sys.exit(e.returncode)
    except FileNotFoundError:
        print("Error: uvicorn not found. Please install with: pip install uvicorn", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    # Allow running individual functions for testing
    if len(sys.argv) > 1:
        func_name = sys.argv[1]
        if func_name in ["train", "retrieve", "evaluate", "serve"]:
            globals()[f"{func_name}_cli"]()
        else:
            print(f"Unknown command: {func_name}")
            print("Available commands: train, retrieve, evaluate, serve")
            sys.exit(1)
    else:
        print("Available commands: train, retrieve, evaluate, serve")
        print("Use: python cli.py <command>")