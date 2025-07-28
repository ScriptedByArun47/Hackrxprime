# app/logger.py

import logging
import os

# Ensure the logs directory exists
os.makedirs("logs", exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("logs/hackrx.log"),
        logging.StreamHandler()
    ]
)

# Logger instance for use across the app
logger = logging.getLogger("hackrx")
