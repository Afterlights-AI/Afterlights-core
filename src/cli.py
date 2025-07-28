#!/usr/bin/env python3
"""
Command-line interface wrappers for Afterlights-core scripts.
These functions provide console script entry points for the shell scripts.
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path


def get_project_root():
    """Get the project root directory."""
    current_file = Path(__file__).resolve()
    # Go up from src/cli.py to project root
    return current_file.parent.parent


def train_cli():
    """Console script entry point for training."""
    parser = argparse.ArgumentParser(
        description="Train the Afterlights model using contrastive learning",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  afterlights-train
  
This command runs the training script which will:
- Load data from evaluation/eval_dataset/locomo/locomo_conv-26.csv
- Train using sentence-transformers/all-MiniLM-L6-v2 as base model
- Save trained model to trained_model/ directory
        """
    )
    parser.add_argument(
        "-v", "--verbose", 
        action="store_true", 
        help="Enable verbose output"
    )
    
    args = parser.parse_args()
    
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
    parser = argparse.ArgumentParser(
        description="Retrieve relevant documents using trained embeddings",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  afterlights-retrieve --query "What is courage?" --file_path dataset/the_courage_to_be_disliked.csv
  afterlights-retrieve --query "Iron Man" --model_output_path trained_model/iron_man_model --top_k 5
  
Modes:
  - naive_csv: Simple CSV-based retrieval
  - contextual: Advanced contextual retrieval (default)
        """
    )
    parser.add_argument(
        "--file_path",
        default=os.environ.get("AFTERLIGHTS_FILE_PATH", "dataset/path"),
        help="Path to the dataset file (default: dataset/path)"
    )
    parser.add_argument(
        "--model_output_path", 
        default=os.environ.get("AFTERLIGHTS_MODEL_PATH", "trained_model/path"),
        help="Path to the trained model (default: trained_model/path)"
    )
    parser.add_argument(
        "--query",
        default=os.environ.get("AFTERLIGHTS_QUERY", "query"),
        help="Query string to search for (default: 'query')"
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=int(os.environ.get("AFTERLIGHTS_TOP_K", "3")),
        help="Number of top results to return (default: 3)"
    )
    parser.add_argument(
        "--mode",
        choices=["naive_csv", "contextual"],
        default=os.environ.get("AFTERLIGHTS_MODE", "contextual"),
        help="Retrieval mode (default: contextual)"
    )
    parser.add_argument(
        "--qdrant",
        action="store_true",
        default=True,
        help="Use Qdrant vector database (default: enabled)"
    )
    
    args = parser.parse_args()
    
    project_root = get_project_root()
    retrieve_script = project_root / "src" / "retrieve.py"
    
    if not retrieve_script.exists():
        print(f"Error: {retrieve_script} not found", file=sys.stderr)
        sys.exit(1)
    
    cmd = [
        sys.executable, str(retrieve_script),
        "--file_path", args.file_path,
        "--model_output_path", args.model_output_path,
        "--query", args.query,
        "--top_k", str(args.top_k),
        "--mode", args.mode,
    ]
    
    if args.qdrant:
        cmd.append("--qdrant")
    
    try:
        subprocess.run(cmd, cwd=project_root, check=True)
    except subprocess.CalledProcessError as e:
        sys.exit(e.returncode)


def evaluate_cli():
    """Console script entry point for evaluation."""
    parser = argparse.ArgumentParser(
        description="Evaluate the trained Afterlights model performance",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  afterlights-evaluate
  afterlights-evaluate --verbose
  
This command runs model evaluation which will:
- Test retrieval accuracy on evaluation datasets
- Generate performance metrics and reports
- Output results to console and log files
        """
    )
    parser.add_argument(
        "-v", "--verbose", 
        action="store_true", 
        help="Enable verbose evaluation output"
    )
    
    args = parser.parse_args()
    
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
    parser = argparse.ArgumentParser(
        description="Start the Afterlights retrieval API server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  afterlights-serve
  afterlights-serve --port 9000
  afterlights-serve --host 0.0.0.0 --port 8080
  
This command starts a FastAPI server that provides:
- Document retrieval endpoints
- Model management APIs  
- Health check endpoints
        """
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host to bind the server to (default: 127.0.0.1)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("AFTERLIGHTS_PORT", "8755")),
        help="Port to bind the server to (default: 8755)"
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload for development"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of worker processes (default: 1)"
    )
    
    args = parser.parse_args()
    
    project_root = get_project_root()
    src_dir = project_root / "src"
    
    if not (src_dir / "api.py").exists():
        print(f"Error: {src_dir / 'api.py'} not found", file=sys.stderr)
        sys.exit(1)
    
    cmd = [
        "uvicorn", "api:app", 
        "--host", args.host,
        "--port", str(args.port),
        "--workers", str(args.workers)
    ]
    
    if args.reload:
        cmd.append("--reload")
    
    try:
        subprocess.run(cmd, cwd=src_dir, check=True)
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