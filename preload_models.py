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

# Import DeepFace for model preloading
from deepface import DeepFace
import tensorflow as tf

# Verify CPU is being used
print("TensorFlow devices:", tf.config.list_physical_devices())
print("Using CPU only mode:", not tf.config.list_physical_devices('GPU'))

# Force download of face detection models by running a test
print("Downloading face detection models...")
try:
    # This will automatically download the necessary detector models
    sample_img = "https://github.com/serengil/deepface/raw/master/tests/dataset/img1.jpg"
    
    # Try both detectors we use in the app
    print("Testing OpenCV detector...")
    DeepFace.extract_faces(img_path=sample_img, detector_backend="opencv")
    
    print("Testing MTCNN detector...")
    DeepFace.extract_faces(img_path=sample_img, detector_backend="mtcnn")
    
    print("Face detection models downloaded successfully")
except Exception as e:
    print(f"Error loading detection models: {str(e)}")

# Force download of face recognition models with a test embed
print("Downloading face recognition models...")
try:
    # Generate an embedding with Facenet model
    DeepFace.represent(img_path=sample_img, model_name="Facenet")
    print("Face recognition model (Facenet) downloaded successfully")
    
    # Also download Facenet512 if we're using it
    DeepFace.represent(img_path=sample_img, model_name="Facenet512")
    print("Face recognition model (Facenet512) downloaded successfully")
except Exception as e:
    print(f"Error loading recognition models: {str(e)}")

print("Model preloading complete!") 