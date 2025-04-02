import logging
import os
import sys
from logging.handlers import RotatingFileHandler

def setup_logger(name, log_level=logging.DEBUG):
    """
    Set up a logger with the specified name and level
    
    Args:
        name: Name for the logger
        log_level: Logging level (default: DEBUG)
        
    Returns:
        Logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(log_level)
    
    # Create logs directory if it doesn't exist
    logs_dir = "logs"
    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir)
    
    # Set up file handler with rotation
    file_handler = RotatingFileHandler(
        os.path.join(logs_dir, f"{name}.log"),
        maxBytes=5*1024*1024,  # 5 MB
        backupCount=5
    )
    
    # Set up console handler
    console_handler = logging.StreamHandler(sys.stdout)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Set formatter for handlers
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

def get_logger(name):
    """
    Get a logger with the specified name.
    Creates a new one if it doesn't exist.
    
    Args:
        name: Name for the logger
        
    Returns:
        Logger instance
    """
    logger = logging.getLogger(name)
    
    if not logger.handlers:
        return setup_logger(name)
    
    return logger
