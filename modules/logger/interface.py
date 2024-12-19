import abc


class LoggerInterface(abc.ABC):
    @abc.abstractmethod
    def info(self, message: str):
        pass

    @abc.abstractmethod
    def debug(self, message: str):
        pass

    @abc.abstractmethod
    def success(self, message: str):
        pass

    @abc.abstractmethod
    def error(self, message: str):
        pass

    @abc.abstractmethod
    def warning(self, message: str):
        pass

    @staticmethod
    @abc.abstractmethod
    def get_logger():
        pass
