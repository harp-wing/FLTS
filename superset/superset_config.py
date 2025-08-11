# superset/superset_config.py

import os

# Database configuration
SQLALCHEMY_DATABASE_URI = os.environ.get('SQLALCHEMY_DATABASE_URI', 
    'postgresql+psycopg2://superset:superset@superset-postgres:5432/superset')

# Redis configuration
REDIS_HOST = os.environ.get('REDIS_HOST', 'redis')
REDIS_PORT = os.environ.get('REDIS_PORT', 6379)

# Cache configuration
CACHE_CONFIG = {
    'CACHE_TYPE': 'redis',
    'CACHE_DEFAULT_TIMEOUT': 300,
    'CACHE_KEY_PREFIX': 'superset_',
    'CACHE_REDIS_HOST': REDIS_HOST,
    'CACHE_REDIS_PORT': REDIS_PORT,
    'CACHE_REDIS_DB': 1,
    'CACHE_REDIS_URL': f'redis://{REDIS_HOST}:{REDIS_PORT}/1'
}

DATA_CACHE_CONFIG = CACHE_CONFIG

# Security configuration
SECRET_KEY = os.environ.get('SUPERSET_SECRET_KEY', 'your-secret-key-here-change-this')

# Enable CORS for API access
ENABLE_CORS = True
CORS_OPTIONS = {
    'supports_credentials': True,
    'allow_headers': [
        'X-CSRFToken', 'Content-Type', 'Origin', 'X-Requested-With', 'Accept',
        'Authorization', 'X-CSRF-Token'
    ],
    'resources': {
        '/api/*': {'origins': '*'},
        '/superset/*': {'origins': '*'}
    }
}

# Feature flags
FEATURE_FLAGS = {
    'DASHBOARD_NATIVE_FILTERS': True,
    'DASHBOARD_CROSS_FILTERS': True,
    'DASHBOARD_FILTERS_EXPERIMENTAL': True,
    'ENABLE_TEMPLATE_PROCESSING': True,
    'ALERT_REPORTS': True,
}

# SQL Lab configuration
SUPERSET_WEBSERVER_PORT = 8088
SUPERSET_WEBSERVER_TIMEOUT = 300

# Default row limit for SQL Lab
DEFAULT_SQLLAB_LIMIT = 5000
SQL_MAX_ROW = 100000

# Enable SQL Lab
SQLLAB_ASYNC_TIME_LIMIT_SEC = 300
SQLLAB_TIMEOUT = 300

# Logging configuration
ENABLE_TIME_ROTATE = True
TIME_ROTATE_LOG_LEVEL = 'DEBUG'

# Define the data directory for Superset
DATA_DIR = '/app/superset_home'
FILENAME = os.path.join(DATA_DIR, 'superset.log')

# Custom CSS
CUSTOM_CSS = """
.navbar-brand {
    color: #007bff !important;
}
"""

# Email configuration (optional)
# SMTP_HOST = 'your-smtp-server'
# SMTP_STARTTLS = True
# SMTP_SSL = False
# SMTP_USER = 'your-email@example.com'
# SMTP_PORT = 587
# SMTP_PASSWORD = 'your-password'
# SMTP_MAIL_FROM = 'your-email@example.com'