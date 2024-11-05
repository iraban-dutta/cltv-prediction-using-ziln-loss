import os
import logging
from datetime import datetime


LOG_FOLDER_PATH = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
logs_path = os.path.join(os.getcwd(), 'logs', LOG_FOLDER_PATH)
os.makedirs(logs_path, exist_ok=True)

LOG_FILE_PATH = os.path.join(logs_path, f'{LOG_FOLDER_PATH}.log')

logging.basicConfig(
    filename=LOG_FILE_PATH,
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO, 
)

if __name__=='__main__':

    # Testing Logging
    logging.info('Testing logging functionality')