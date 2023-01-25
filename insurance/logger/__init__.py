import logging
from datetime import datetime
import os

# defining the directory to store log file
# defining log file location
LOG_DIR = "insurance_log"

# giving current timestamp
CURRENT_TIME_STAMP = f"{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}"

# name of log file according to timestamp
LOG_FILE_NAME = f"log_{CURRENT_TIME_STAMP}.log"

# checking log directory available or not
# if no then it will create one.
os.makedirs(LOG_DIR, exist_ok=True)

# defining log file path
LOG_FILE_PATH = os.path.join(LOG_DIR, LOG_FILE_NAME)

# main log file
logging.basicConfig(
    filename=LOG_FILE_PATH,
    filemode="w",
    format="[%(asctime)s] %(name)s - %(levelname)s - %(message)s",
    level=logging.DEBUG,
)
