#!/usr/bin/env bash
# Build script for Render

echo "Installing dependencies..."
pip install -r requirements.txt

echo "Preloading DeepFace models..."
python preload_models.py

echo "Build completed!" 