from abc import ABC, abstractmethod
from typing import Optional


class LLMClientInterface(ABC):
    """
    Interface for a Language Model Client.

    This interface defines the methods that any language model client should implement.
    """

    @abstractmethod
    def __init__(self, api_token: str, model: Optional[str] = None):
        """
        Initialize the LLM client with the given API token and optional model.

        Args:
            api_token (str): The API token for authenticating with the language model service.
            model (Optional[str]): The model to use for generating responses. Defaults to None.
        """
        self.api_token = api_token
        self.model = model

    @abstractmethod
    async def generate(
        self, user_prompt: str, system_prompt: Optional[str] = None
    ) -> str:
        """
        Generate a response based on the given prompt.

        Args:
            user_prompt (str): The prompt provided by the user.

        Returns:
            str: The generated response from the language model.
        """
        pass
