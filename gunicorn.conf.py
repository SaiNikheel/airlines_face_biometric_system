# Gunicorn configuration file
import multiprocessing

# Increase timeout to handle face recognition processing
timeout = 300  # Increased to 5 minutes from previous 120 seconds
workers = multiprocessing.cpu_count() * 2 + 1
threads = 2
worker_class = 'gthread'
loglevel = 'info'
keepalive = 120  # Keep connections alive for 2 minutes
worker_tmp_dir = '/tmp'  # Use tmp directory for worker heartbeats
graceful_timeout = 120   # Grace period for workers to finish

# Increase timeout to handle face recognition processing
timeout = 120
workers = multiprocessing.cpu_count() * 2 + 1
threads = 2
worker_class = 'gthread'
loglevel = 'info' 