"""
Preload DeepFace models during the build phase to prevent runtime downloads.
Run this script during the build process before starting the application.
"""
print("Preloading DeepFace models...")

# Force TensorFlow to use CPU
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_FORCE_CPU_ALLOW_GROWTH'] = 'true'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow warnings

from deepface import DeepFace
from deepface.commons import functions
import tensorflow as tf

# Verify CPU is being used
print("TensorFlow devices:", tf.config.list_physical_devices())
print("Using CPU only mode:", not tf.config.list_physical_devices('GPU'))

# Force download of models during build
print("Downloading face detection model...")
functions.initialize_detector("retinaface")

print("Downloading face recognition model...")
DeepFace.represent(img_path="https://github.com/serengil/deepface/raw/master/tests/dataset/img1.jpg", 
                  model_name="Facenet", 
                  detector_backend="retinaface")

print("Model preloading complete!") 