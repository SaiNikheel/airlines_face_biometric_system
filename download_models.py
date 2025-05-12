#!/usr/bin/env python
"""
Pre-downloads ONLY the GhostFaceNet model to prevent runtime timeout errors.
Run this during the build phase on Render.
"""
import os
import sys
import time
print("Starting targeted model download process...")

try:
    from deepface import DeepFace
    from deepface.commons import functions
    
    # Create output directory for logs
    os.makedirs("model_downloads", exist_ok=True)
    
    # Only download GhostFaceNet model - no fallbacks
    print("Downloading only GhostFaceNet model...")
    model_name = "GhostFaceNet"
    
    try:
        print(f"Downloading {model_name} model...")
        model = DeepFace.build_model(model_name)
        print(f"✓ {model_name} downloaded successfully")
        primary_model = model_name
    except Exception as e:
        print(f"× Error downloading {model_name}: {str(e)}")
        print("CRITICAL: Failed to download GhostFaceNet model")
        sys.exit(1)  # Exit with error
    
    # Download only opencv detector backend - it's faster and more reliable
    print("Downloading minimal face detector...")
    detector = 'opencv'  # Single detector only
    try:
        print(f"Testing {detector} detector...")
        DeepFace.extract_faces(img_path="https://github.com/serengil/deepface/raw/master/tests/dataset/img1.jpg", 
                              detector_backend=detector)
        print(f"✓ {detector} detector downloaded and working")
    except Exception as e:
        print(f"× Error with {detector} detector: {str(e)}")
    
    print("Targeted model download completed!")
    
    # List all downloaded model files
    base_dir = os.path.join(os.path.expanduser('~'), '.deepface')
    if os.path.exists(base_dir):
        print("\nDownloaded model files:")
        for root, dirs, files in os.walk(base_dir):
            for file in files:
                full_path = os.path.join(root, file)
                rel_path = os.path.relpath(full_path, base_dir)
                size_mb = os.path.getsize(full_path) / (1024 * 1024)
                print(f"- {rel_path} ({size_mb:.2f} MB)")
    
    # Create a marker file to indicate successful download
    with open("model_downloads/success.txt", "w") as f:
        f.write(f"GhostFaceNet model downloaded successfully at {time.ctime()}")
    
    sys.exit(0)  # Success
    
except Exception as e:
    print(f"ERROR during model download: {str(e)}")
    with open("model_downloads/error.txt", "w") as f:
        f.write(f"Error during model download at {time.ctime()}: {str(e)}")
    sys.exit(1)  # Failure 
