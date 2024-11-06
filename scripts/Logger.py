import os
import sys
import logging

def setup_logger():
    '''
    This function is used to setup logger for logging error and Info

    **Returns**:
    -----------
        a `logger` instance
    '''

    log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'logs')
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    log_file_info = os.path.join(log_dir, 'Info.log')
    log_file_error = os.path.join(log_dir, 'Error.log')

    logger = logging.getLogger(__name__)
    
    # Check if logger has handlers already (prevents adding multiple handlers)
    if logger.hasHandlers():
        return logger
    
    logger.setLevel(logging.DEBUG)  # Set the base level

    # Create handlers for file and console
    info_handler = logging.FileHandler(log_file_info)
    error_handler = logging.FileHandler(log_file_error)
    console_handler = logging.StreamHandler()

    # Set logging levels for each handler
    info_handler.setLevel(logging.INFO)
    error_handler.setLevel(logging.ERROR)
    console_handler.setLevel(logging.DEBUG)

    # Define formatter and set it for each handler
    formatter = logging.Formatter('%(asctime)s - %(levelname)s :: %(message)s',
                                  datefmt='%Y-%m-%d %H:%M')
    info_handler.setFormatter(formatter)
    error_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Add handlers to logger
    logger.addHandler(info_handler)
    logger.addHandler(error_handler)
    logger.addHandler(console_handler)
    
    # Disable propagation to prevent duplicate logs
    logger.propagate = False

    mlflow_logger = logging.getLogger("mlflow")
    mlflow_logger.setLevel(logging.ERROR)

    return logger

# Create a global instance of the logger
LOGGER = setup_logger()