import sys

from loguru import logger


def setup_logging(log_file="logs/app.log"):
    # Remove any existing handlers
    logger.remove()
    
    # Add a handler for stdout with INFO level
    logger.add(sys.stdout, level="INFO", format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>")
    
    # Add a handler for file with DEBUG level
    logger.add(log_file, rotation="10 MB", level="DEBUG", format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}")

    return logger

# Create a global logger instance
logger = setup_logging()