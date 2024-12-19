import sys

from loguru import logger

from modules.logger.interface import LoggerInterface


class LoggerFactory(LoggerInterface):
    def __init__(self):
        self.logger = self.setup_logger()

    @staticmethod
    def setup_logger():
        # Remove any existing handlers
        logger.remove()

        # Add custom formatted handler
        logger.add(
            sys.stdout,
            format="<blue>{time:HH:mm:ss}</blue> | <green>{message}</green>",
            colorize=True,
            level="INFO",
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
