
import logging
import sys
# Configure root logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)  # log to stdout (so Docker can capture it)
    ]
)
logger = logging.getLogger("file-ingest-microservice")  # custom logger name