import logging
import os
from datetime import datetime

def setup_logger(name):
    # Create logs directory if it doesn't exist
    if not os.path.exists('logs'):
        os.makedirs('logs')
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # Create formatters
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # File handler (logs everything to file)
    file_handler = logging.FileHandler(
        f'logs/{datetime.now().strftime("%Y-%m-%d")}.log'
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    
    # Console handler (logs INFO and above to console)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    
    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    def log_to_file(content: str, filename: str) -> None:
        """Write content to a specific file in the logs directory."""
        if not os.path.exists('logs'):
            os.makedirs('logs')
        filepath = os.path.join('logs', filename)
        with open(filepath, 'a', encoding='utf-8') as f:
            f.write(content + '\n')
    
    # Attach the method to the logger object
    logger.log_to_file = log_to_file
    
    return logger 