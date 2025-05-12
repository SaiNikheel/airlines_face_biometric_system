"""
CPU configuration for TensorFlow to prevent GPU errors.
This file should be imported at the beginning of app.py.
"""
import os
import logging

logging.info("Configuring TensorFlow for CPU-only mode")

# Force TensorFlow to use CPU
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_FORCE_CPU_ALLOW_GROWTH'] = 'true'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow warnings

# For safety, disable eager execution which can sometimes cause issues
try:
    import tensorflow as tf
    if hasattr(tf, 'compat') and hasattr(tf.compat, 'v1'):
        # TensorFlow 2.x
        tf.compat.v1.disable_eager_execution()
    logging.info("TensorFlow CPU configuration applied successfully")
except ImportError:
    logging.warning("TensorFlow import failed - configuration will be applied when TF is imported")
except Exception as e:
    logging.warning(f"Error during TensorFlow configuration: {str(e)}") 