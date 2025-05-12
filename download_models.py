#!/usr/bin/env python
"""
Pre-downloads ONLY the GhostFaceNet model to prevent runtime timeout errors.
Run this during the build phase on Render.
"""
import os
import sys
import time
import numpy as np
print("Starting targeted model download process...")

try:
    from deepface import DeepFace
    from deepface.commons import functions
    from PIL import Image
    
    # Create output directory for logs
    os.makedirs("model_downloads", exist_ok=True)
    
    # Create a tiny test image instead of downloading a large one
    print("Creating tiny test image...")
    test_img = np.zeros((100, 100, 3), dtype=np.uint8)
    # Add a simple face-like shape for testing
    test_img[30:70, 40:60] = [220, 190, 170]  # Face color
    test_img[40:45, 45:50] = [50, 50, 50]  # Left eye
    test_img[40:45, 55:60] = [50, 50, 50]  # Right eye
    test_img[55:60, 50:55] = [150, 90, 90]  # Mouth
    test_img_path = "tiny_test_face.jpg"
    Image.fromarray(test_img).save(test_img_path)
    print(f"Created test image at {test_img_path}")
    
    # Only set up paths and check if model exists, don't load it yet
    print("Checking GhostFaceNet model paths...")
    model_name = "GhostFaceNet"
    
    # Just verify the paths exist
    home = os.path.expanduser("~")
    weights_path = os.path.join(home, '.deepface', 'weights')
    model_path = os.path.join(weights_path, 'ghostfacenet_v1.h5')
    
    # Create directory if it doesn't exist
    os.makedirs(weights_path, exist_ok=True)
    
    # Only check if file exists or download it, but don't load the model
    if not os.path.isfile(model_path):
        print(f"GhostFaceNet model not found at {model_path}, will download...")
        # This only downloads but doesn't load the model
        from deepface.commons import functions
        functions.initialize_folder()
        url = "https://github.com/HamadYA/GhostFaceNets/releases/download/v1.2/GhostFaceNet_W1.3_S1_ArcFace.h5"
        functions.download_file(url, model_path)
        print(f"✓ {model_name} downloaded successfully to {model_path}")
    else:
        print(f"✓ {model_name} already exists at {model_path}")
    
    # Only check if detector works with minimal memory usage
    detector_backend = 'opencv'
    try:
        print(f"Testing basic {detector_backend} detector functionality...")
        # Just import and test basic functionality without loading model
        import cv2
        from deepface.detectors import OpenCvWrapper
        # Don't actually run detection, just check if OpenCV is available
        print(f"✓ {detector_backend} detector is available")
    except Exception as e:
        print(f"× Error with {detector_backend} detector: {str(e)}")
    
    print("GhostFaceNet model setup completed successfully!")
    
    # List model files without loading them
    base_dir = os.path.join(os.path.expanduser('~'), '.deepface')
    if os.path.exists(base_dir):
        print("\nVerified model files:")
        for root, dirs, files in os.walk(base_dir):
            for file in files:
                full_path = os.path.join(root, file)
                rel_path = os.path.relpath(full_path, base_dir)
                if os.path.exists(full_path):  # Make sure file exists
                    size_mb = os.path.getsize(full_path) / (1024 * 1024)
                    print(f"- {rel_path} ({size_mb:.2f} MB)")
    
    # Clean up test image
    if os.path.exists(test_img_path):
        os.remove(test_img_path)
    
    # Create a marker file to indicate successful download
    with open("model_downloads/success.txt", "w") as f:
        f.write(f"GhostFaceNet model downloaded successfully at {time.ctime()}")
    
    sys.exit(0)  # Success
    
except Exception as e:
    print(f"ERROR during model download: {str(e)}")
    with open("model_downloads/error.txt", "w") as f:
        f.write(f"Error during model download at {time.ctime()}: {str(e)}")
    sys.exit(1)  # Failure 
