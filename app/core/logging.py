import logging
from pythonjsonlogger import json

from app.core.config import settings 
# logger -> handler -> formatter->O/P
def setup_logging() -> None:
    logger = logging.getLogger()
    logger.setLevel(settings.log_level)

    #Remove default handlers to avoid duplicate logs
    for h in list(logger.handlers):
        logger.removeHandler(h)
    
    handler = logging.StreamHandler()
    formatter = json.JsonFormatter(
        "%(asctime)s %(levelname)s %(name)s %(message)s %(request_id)s %(path)s %(method)s %(status_code)s %(latency_ms)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
