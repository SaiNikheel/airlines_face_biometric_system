#!/usr/bin/env python
"""
Pre-downloads all DeepFace models to prevent runtime timeout errors.
Run this during the build phase on Render.
"""
import os
import sys
import time
print("Starting model download process...")

try:
    from deepface import DeepFace
    from deepface.commons import functions
    
    # Create output directory for logs
    os.makedirs("model_downloads", exist_ok=True)
    
    # Force download of main models
    print("Downloading face detection models...")
    models = [
        "VGG-Face", 
        "Facenet", 
        "Facenet512", 
        "OpenFace", 
        "DeepFace", 
        "DeepID", 
        "ArcFace", 
        "SFace"
    ]
    
    # Download all face recognition models
    for model_name in models:
        print(f"Downloading {model_name} model...")
        model = DeepFace.build_model(model_name)
        print(f"✓ {model_name} downloaded successfully")
    
    # Download detector models (RetinaFace, MTCNN, etc.)
    print("Downloading face detector models...")
    detectors = ['opencv', 'ssd', 'dlib', 'mtcnn', 'retinaface', 'mediapipe', 'yolov8']
    for detector in detectors:
        try:
            print(f"Testing detector: {detector}")
            DeepFace.extract_faces(img_path="https://github.com/serengil/deepface/raw/master/tests/dataset/img1.jpg", 
                                  detector_backend=detector)
            print(f"✓ {detector} detector downloaded and working")
        except Exception as e:
            print(f"× Error with {detector}: {str(e)}")
    
    # Download auxiliary models
    print("Downloading auxiliary models (age, gender, emotion, race)...")
    DeepFace.analyze(img_path="https://github.com/serengil/deepface/raw/master/tests/dataset/img1.jpg", 
                    actions=['age', 'gender', 'emotion', 'race'])
    
    print("All models downloaded successfully!")
    
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
        f.write(f"Models downloaded successfully at {time.ctime()}")
    
    sys.exit(0)  # Success
    
except Exception as e:
    print(f"ERROR during model download: {str(e)}")
    with open("model_downloads/error.txt", "w") as f:
        f.write(f"Error during model download at {time.ctime()}: {str(e)}")
    sys.exit(1)  # Failure 