# Gunicorn configuration for face recognition app

# Timeout settings - increased for model loading
timeout = 240  # 4 minutes to allow for face processing
graceful_timeout = 120

# Worker settings optimized for memory usage
workers = 1  # Single worker to reduce memory usage
worker_class = 'sync'
worker_connections = 500
threads = 2
max_requests = 200  # Restart worker occasionally to release memory

# Pre-load app optimizations
preload_app = False  # Don't preload to reduce startup memory usage

# Server settings
bind = "0.0.0.0:$PORT"

# Logging
accesslog = '-'
errorlog = '-'
loglevel = 'info'

# Memory optimization
worker_tmp_dir = '/dev/shm'  # Use shared memory for temp
keepalive = 3                # Reduce keep-alive connections

# Function to check resource usage
def on_starting(server):
    import os
    import psutil
    
    # Log initial memory usage
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    print(f"Initial memory usage: {memory_info.rss / (1024 * 1024):.2f} MB")
    
    # Force garbage collection
    import gc
    gc.collect()
    
    # Set lower memory limits for TensorFlow if available
    try:
        import tensorflow as tf
        gpus = tf.config.experimental.list_physical_devices('GPU')
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("Set TensorFlow memory growth to True")
    except:
        pass 