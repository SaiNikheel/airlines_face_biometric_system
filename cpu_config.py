"""
Configure TensorFlow to use CPU only and disable GPU.
Import this before any TensorFlow imports.
"""
import os
import tensorflow as tf

# Force CPU usage
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow logging

# Configure TensorFlow to use CPU
try:
    # Disable GPU memory allocation
    physical_devices = tf.config.list_physical_devices('GPU')
    if len(physical_devices) > 0:
        for device in physical_devices:
            tf.config.set_visible_devices([], 'GPU')
            print(f"Disabled GPU device: {device}")
    print("TensorFlow configured to use CPU only")
except Exception as e:
    print(f"Error disabling GPU: {e}") 