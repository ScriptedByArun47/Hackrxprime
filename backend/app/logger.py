# app/logger.py

import logging
import os

# Setup logs directory relative to project root
LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs")
os.makedirs(LOG_DIR, exist_ok=True)

# Allow log level override from environment
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

def setup_logger(name: str = "hackrx") -> logging.Logger:
    """
    Initializes and returns a logger with both file and console handlers.
    Ensures handlers are not duplicated on repeated imports.
    """
    logger = logging.getLogger(name)
    if logger.hasHandlers():
        return logger  # Prevent duplicate handlers

    logger.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))

    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

    # Log to file
    file_handler = logging.FileHandler(os.path.join(LOG_DIR, "hackrx.log"))
    file_handler.setFormatter(formatter)

    # Log to stdout (console)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    return logger

# Global logger instance
logger = setup_logger()
