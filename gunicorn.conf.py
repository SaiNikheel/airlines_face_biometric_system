# Gunicorn configuration file
import multiprocessing

# Increase timeout to handle face recognition processing (even more time)
timeout = 180

# Optimize for CPU-bound workloads (face recognition)
# Use fewer workers with more threads for better memory efficiency
cpu_count = multiprocessing.cpu_count()
workers = max(2, cpu_count)  # Use at least 2 workers
threads = 4  # More threads per worker

# Use gthread worker for better handling of CPU-intensive loads
worker_class = 'gthread'

# Prevent worker heartbeat from timing out
worker_timeout = 180

# Logging
loglevel = 'info'
accesslog = '-'  # Log to stdout
errorlog = '-'   # Log to stderr

# Maximum number of requests a worker will process before restarting
# This helps prevent memory leaks
max_requests = 1000
max_requests_jitter = 100  # Add randomness to max_requests

# Prevent workers from taking too much memory
limit_request_line = 4096
limit_request_fields = 100 