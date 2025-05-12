# Gunicorn configuration file
import multiprocessing

# Increase timeout to handle face recognition processing
timeout = 120
workers = multiprocessing.cpu_count() * 2 + 1
threads = 2
worker_class = 'gthread'
loglevel = 'info' 