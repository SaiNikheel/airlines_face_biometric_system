"""
Preload DeepFace models during the build phase to prevent runtime downloads.
Run this script during the build process before starting the application.
"""
print("Preloading DeepFace models...")

from deepface import DeepFace
from deepface.commons import functions

# Force download of models during build
print("Downloading face detection model...")
functions.initialize_detector("retinaface")

print("Downloading face recognition model...")
DeepFace.represent(img_path="https://github.com/serengil/deepface/raw/master/tests/dataset/img1.jpg", 
                  model_name="Facenet", 
                  detector_backend="retinaface")

print("Model preloading complete!") 