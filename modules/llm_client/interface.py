from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


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

    @abstractmethod
    async def generate_scenarios(
        self, node: Dict[str, Any], max_scenarios: int = 3
    ) -> List[Dict[str, str]]:
        """Generate alternative conversation paths for a decision point.

        This function uses the LLM to generate diverse test scenarios for exploring different
        conversation branches at key decision points in the dialogue. It analyzes the current
        conversation node and generates alternative customer responses to test the system's
        handling of various scenarios.

        Args:
            node (Dict[str, Any]): The current conversation node containing the dialogue
                context and available response edges
            max_scenarios (int, optional): Maximum number of test scenarios to generate.
                Defaults to 3.

        Returns:
            List[Dict[str, str]]: A list of scenario dictionaries, where each dictionary
                contains:
                - 'response': A brief description of the customer's response
                - 'prompt': The full conversation prompt to test that scenario

        Raises:
            ValueError: If the LLM response cannot be parsed as valid JSON
            ClientError: If there is an error communicating with the LLM service
        """
        pass

    @abstractmethod
    async def get_nodes_from_transcript(
        self,
        transcript: str,
        depth_patterns: Dict[int, List[str]],
        current_depth: int = 0,
        max_retries: int = 3,
    ) -> tuple[List[Dict[str, Any]], Dict[int, List[str]]]:
        """Convert a conversation transcript into a graph representation with concise nodes.

        Args:
            transcript (str): The conversation transcript to convert into nodes
            depth_patterns (Dict[int, List[str]]): Dictionary mapping depths to lists of known node patterns
            current_depth (int, optional): Current depth in the conversation tree. Defaults to 0.
            max_retries (int, optional): Maximum number of retries for failed conversions. Defaults to 3.

        Returns:
            tuple[List[Dict[str, Any]], Dict[int, List[str]]]: Tuple containing:
                - List of node dictionaries, each containing:
                    - node_id (int): Unique identifier for the node
                    - content (str): Concise 1-4 word description of the node
                    - depth (int): Depth of node in conversation tree
                    - edges (List[Dict]): List of edges to other nodes
                - Updated depth_patterns dictionary with any new patterns found

        Raises:
            ValueError: If transcript cannot be converted to valid nodes
            ClientError: If there is an error communicating with the LLM service
        """
        pass
