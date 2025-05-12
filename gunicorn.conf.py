# Gunicorn configuration for face recognition app

# Timeout settings
timeout = 120  # 2 minutes to allow for face processing
graceful_timeout = 120

# Worker settings
workers = 1  # For face recognition, multiple workers can consume too much memory
worker_class = 'sync'
worker_connections = 1000
threads = 2

# Server settings
bind = "0.0.0.0:$PORT"

# Restart workers occasionally to help with memory issues
max_requests = 1000
max_requests_jitter = 100

# Logging
accesslog = '-'
errorlog = '-'
loglevel = 'info'

# Optimize for longer requests
keepalive = 5 