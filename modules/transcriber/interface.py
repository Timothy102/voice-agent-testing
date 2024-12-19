import abc
from pathlib import Path


class TranscriberInterface(abc.ABC):
    def __init__(self, api_token: str):
        self.api_token = api_token

    @abc.abstractmethod
    async def transcribe(self, filepath: Path) -> str:
        pass
