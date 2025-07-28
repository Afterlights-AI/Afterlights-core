#!/usr/bin/env python3

from setuptools import setup, find_packages
import os

# Read requirements from requirements.txt
def read_requirements():
    req_path = os.path.join(os.path.dirname(__file__), 'requirements.txt')
    if os.path.exists(req_path):
        with open(req_path, 'r') as f:
            return [line.strip() for line in f if line.strip() and not line.startswith('#')]
    return []

# Read long description from README
def read_readme():
    readme_path = os.path.join(os.path.dirname(__file__), 'README.md')
    if os.path.exists(readme_path):
        with open(readme_path, 'r', encoding='utf-8') as f:
            return f.read()
    return ''

setup(
    name="Afterlights-core",
    version="0.1.0",
    description="Add your description here",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.11",
    install_requires=[
        "accelerate>=1.6.0",
        "ace-tools>=0.0",
        "datasets>=3.5.1",
        "einops>=0.8.1",
        "fastapi>=0.115.14",
        "nltk>=3.9.1",
        "openai>=1.79.0",
        "pytest>=8.4.1",
        "qdrant-client>=1.14.2",
        "scikit-learn>=1.6.1",
        "sentence-transformers>=4.1.0",
        "thefuck>=3.32",
        "uvicorn>=0.35.0",
    ],
    entry_points={
        'console_scripts': [
            'afterlights-train=cli:train_cli',
            'afterlights-retrieve=cli:retrieve_cli',
            'afterlights-evaluate=cli:evaluate_cli',
            'afterlights-serve=cli:serve_cli',
        ],
    },
    package_data={
        'afterlights_core': ['../scripts/*.sh'],
    },
    include_package_data=True,
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
)