"""
Preload DeepFace models during the build phase to prevent runtime downloads.
Run this script during the build process before starting the application.
"""
print("Preloading DeepFace models...")

# Import CPU configuration before any TensorFlow imports
import cpu_config

import os
import time
from deepface import DeepFace
from deepface.commons import functions

# Force CPU usage
print("Configuring DeepFace to use CPU only...")
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Force download of models during build
print("Downloading face detection model...")
start_time = time.time()
functions.initialize_detector("retinaface")
print(f"Detection model download took {time.time() - start_time:.2f} seconds")

print("Downloading face recognition model...")
start_time = time.time()
try:
    # Turn off normalization to improve speed on CPU
    DeepFace.represent(
        img_path="https://github.com/serengil/deepface/raw/master/tests/dataset/img1.jpg", 
        model_name="Facenet", 
        detector_backend="retinaface",
        enforce_detection=False  # Don't enforce face detection as a fallback
    )
    print(f"Recognition model download took {time.time() - start_time:.2f} seconds")
    print("Model preloading complete!")
except Exception as e:
    print(f"Error during model preloading: {e}")
    # Continue anyway as we've already downloaded the models 