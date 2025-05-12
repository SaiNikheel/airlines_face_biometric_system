#!/usr/bin/env python
"""
Pre-downloads only the GhostFaceNet model to prevent runtime timeout errors.
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
    
    # Only attempt to download GhostFaceNet and fallback model
    print("Downloading targeted face recognition model...")
    models = [
        "GhostFaceNet",  # Primary model
    ]
    
    # Try to download only the specified models
    primary_model = None
    for model_name in models:
        try:
            print(f"Downloading {model_name} model...")
            model = DeepFace.build_model(model_name)
            print(f"✓ {model_name} downloaded successfully")
            
            if primary_model is None:
                primary_model = model_name
                print(f"Successfully set {model_name} as primary model")
        except Exception as e:
            print(f"× Error downloading {model_name}: {str(e)}")
    
    if primary_model is None:
        print("WARNING: Failed to download any of the specified models!")
    else:
        print(f"Primary model set to: {primary_model}")
    
    # Download only necessary detector backends
    print("Downloading minimal set of face detectors...")
    detectors = ['opencv', 'retinaface']  # Only the ones we actually use
    for detector in detectors:
        try:
            print(f"Testing detector: {detector}")
            DeepFace.extract_faces(img_path="https://github.com/serengil/deepface/raw/master/tests/dataset/img1.jpg", 
                                  detector_backend=detector)
            print(f"✓ {detector} detector downloaded and working")
        except Exception as e:
            print(f"× Error with {detector}: {str(e)}")
    
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
        f.write(f"Models downloaded successfully at {time.ctime()}, primary model: {primary_model}")
    
    sys.exit(0)  # Success
    
except Exception as e:
    print(f"ERROR during model download: {str(e)}")
    with open("model_downloads/error.txt", "w") as f:
        f.write(f"Error during model download at {time.ctime()}: {str(e)}")
    sys.exit(1)  # Failure 
