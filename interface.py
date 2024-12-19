from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from modules.graph_visualizer.interface import GraphVisualizerInterface
from modules.llm_client.interface import LLMClientInterface
from modules.logger.factory import LoggerFactory
from modules.phone.interface import PhoneInterface
from modules.transcriber.interface import TranscriberInterface
from utils.settings import Settings


class IGraphConstructor(ABC):
    def __init__(
        self,
        llm_client: LLMClientInterface,
        logger: LoggerFactory,
        phone_client: PhoneInterface,
        transcriber: TranscriberInterface,
        visualizer: GraphVisualizerInterface,
        settings: Optional[Settings] = None,
    ) -> None:
        """
        Initialize the IGraphConstructor with the necessary components.

        Args:
            llm_client (LLMClientInterface): The LLM client for generating responses.
            graph_viz (GraphVisualizerInterface): The graph visualizer for rendering graphs.
            transcriber (TranscriberInterface): The transcriber for processing audio inputs.
            logger (LoggerFactory): The logger for logging activities.
        """
        self.llm_client = llm_client
        self.logger = logger
        self.phone_client = phone_client
        self.settings = settings
        self.transcriber = transcriber
        self.visualizer = visualizer

    @abstractmethod
    async def generate_test_scenarios(
        self, node: Dict[str, Any], max_scenarios: int = 3
    ) -> List[Dict[str, str]]:
        """
        Generate test scenarios based on a given node.

        Args:
            node (Dict[str, Any]): The node to base the scenarios on.
            max_scenarios (int): The maximum number of scenarios to generate. Defaults to 3.

        Returns:
            List[Dict[str, str]]: A list of generated test scenarios.
        """
        pass

    @abstractmethod
    async def get_nodes_from_transcript(
        self, transcript: str, current_depth: int = 0
    ) -> List[Dict[str, Any]]:
        """
        Extract nodes from a given transcript.

        Args:
            transcript (str): The transcript to process.
            current_depth (int): The current depth in the graph. Defaults to 0.

        Returns:
            List[Dict[str, Any]]: A list of nodes extracted from the transcript.
        """
        pass

    @abstractmethod
    async def _process_scenario(
        self,
        scenario: Dict[str, str],
        node: Dict[str, Any],
        visited: Set[str],
        depth: int,
    ) -> None:
        """
        Process a given scenario within the context of a node.

        Args:
            scenario (Dict[str, str]): The scenario to process.
            node (Dict[str, Any]): The current node in the graph.
            visited (Set[str]): A set of visited nodes.
            depth (int): The current depth in the graph.
        """
        pass

    @abstractmethod
    async def dfs(
        self, node: Dict[str, Any], visited: Optional[Set[str]] = None, depth: int = 0
    ) -> None:
        """
        Perform a depth-first search starting from a given node.

        Args:
            node (Dict[str, Any]): The starting node for the DFS.
            visited (Optional[Set[str]]): A set of visited nodes. Defaults to None.
            depth (int): The current depth in the graph. Defaults to 0.
        """
        pass

    @abstractmethod
    async def run(self, phone_number: str, initial_prompt: str) -> List[Dict[str, Any]]:
        """
        Run the graph construction process with a given phone number and initial prompt.

        Args:
            phone_number (str): The phone number to use.
            initial_prompt (str): The initial prompt for the process.
        """
        pass
