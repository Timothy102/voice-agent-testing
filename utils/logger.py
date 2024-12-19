from loguru import logger
import sys
from datetime import datetime

class LoggerFactory:
    def __init__(self):
        self.logger = self.setup_logger()

    @staticmethod
    def setup_logger():
        # Remove any existing handlers
        logger.remove()
        
        # Add custom formatted handler
        logger.add(
            sys.stdout,
            format="<green>{time:HH:mm:ss}</green> | {message}",
            colorize=True,
            level="INFO"
        )
        
        return logger

    def info(self, message: str):
        self.logger.info(message)

    def debug(self, message: str):
        self.logger.debug(message)

    def success(self, message: str):
        self.logger.success(message)

    def error(self, message: str):
        self.logger.error(message)

    def warning(self, message: str):
        self.logger.warning(message)

    @staticmethod
    def get_logger():
        return LoggerFactory.setup_logger()
