import logging
import os
from datetime import datetime

def setup_logger(name, log_file, level=logging.INFO):
    """Function to setup as many loggers as you want"""
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    handler = logging.FileHandler(log_file)
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)

    # Also log to console
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger

# Setup standard loggers
execution_logger = setup_logger('execution', 'logs/execution.log')
trade_logger = setup_logger('trade', 'logs/trades.log')
error_logger = setup_logger('error', 'logs/errors.log', level=logging.ERROR)
