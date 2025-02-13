import os
import logging
from datetime import datetime

# Generate log file name
LOG_FILE = f'{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.log'

# Create logs directory if it doesn't exist
log_path = os.path.join(os.getcwd(), 'logs', LOG_FILE)
os.makedirs(os.path.dirname(log_path), exist_ok=True)

# Define the full log file path
LOG_FILE_PATH = log_path

# Configure logging
logging.basicConfig(
    filename=LOG_FILE_PATH,
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
