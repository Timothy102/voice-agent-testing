"""Main module for constructing and exploring conversation graphs."""

import asyncio
import json
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from interface import GraphConstructorInterface
from modules.graph_visualizer.dash_viz import DynamicGraphVisualizer
from modules.graph_visualizer.interface import GraphVisualizerInterface
from modules.llm_client.claude import ClaudeClient
from modules.llm_client.interface import LLMClientInterface
from modules.logger.factory import LoggerFactory
from modules.logger.interface import LoggerInterface
from modules.phone.hamming import HammingClient
from modules.phone.interface import PhoneInterface
from modules.transcriber.assembly_ai import AssemblyAITranscriber
from modules.transcriber.interface import TranscriberInterface
from utils.settings import Settings


@dataclass
class ConversationNode:
    """Represents a node in the conversation graph.

    Attributes:
        content: The text content of the node
        next_nodes: List of nodes that follow this one
        node_id: Unique identifier for the node
    """

    content: str
    next_nodes: List["ConversationNode"]
    node_id: str


class GraphConstructor(GraphConstructorInterface):
    """Constructs and explores conversation graphs through automated testing.

    This class handles building conversation graphs by making calls, transcribing responses,
    and exploring different conversation paths through automated testing.
    """

    def __init__(
        self,
        llm_client: LLMClientInterface,
        logger: LoggerInterface,
        phone_client: PhoneInterface,
        transcriber: TranscriberInterface,
        visualizer: GraphVisualizerInterface,
        settings: Optional[Settings] = None,
    ) -> None:
        """Initialize the GraphConstructor with required components.

        Args:
            llm_client (LLMClientInterface): The LLM client for generating responses
            phone_client (PhoneInterface): The phone client for making calls
            visualizer (GraphVisualizerInterface): The visualizer for rendering graphs
            transcriber (TranscriberInterface): The transcriber for processing audio
            logger (LoggerInterface): The logger for tracking execution
            settings (Optional[Settings]): Optional settings configuration
        """
        super().__init__(
            llm_client=llm_client,
            logger=logger,
            phone_client=phone_client,
            transcriber=transcriber,
            visualizer=visualizer,
            settings=settings,
        )
        self.nodes: List[Dict[str, Any]] = []
        self.depth_patterns: Dict[int, List[str]] = {}
        self.logger.info("INFO: GraphConstructor initialized")

    async def run_scenario(
        self, scenario: Dict[str, str], node: Dict[str, Any], visited: set, depth: int
    ) -> None:
        """Process a single test scenario by making an API call and updating the conversation graph.

        Takes a test scenario, makes a call to generate a response, transcribes it, and updates
        the conversation graph with any new nodes or edges discovered.

        Args:
            scenario (Dict[str, str]): The test scenario containing prompt and expected response
            node (Dict[str, Any]): Current node in the conversation graph
            visited (set): Set of already visited node IDs to avoid cycles
            depth (int): Current depth in the conversation tree

        Returns:
            None

        Raises:
            Exception: If API call, transcription or graph update fails
        """
        self.logger.info(f"Testing scenario: {scenario['response']}")
        self.prompt = scenario["prompt"]

        # Go generate this scenario #
        call_id = await self.phone_client.make_call(
            phone_number=self.phone_number, prompt=self.prompt
        )
        audio_filepath = await self.phone_client.poll_for_response(call_id)
        new_transcript = await self.transcriber.transcribe(audio_filepath)
        self.logger.info(f"Transcript: {new_transcript}")

        # Generate new nodes from transcript #
        new_nodes, depth_patterns = await self.llm_client.get_nodes_from_transcript(
            transcript=new_transcript,
            depth_patterns=self.depth_patterns,
            current_depth=depth + 1,
        )
        self.logger.info(f"New path discovered: {json.dumps(new_nodes, indent=2)}")
        self.depth_patterns = depth_patterns

        if new_nodes:
            self.visualizer.update_graph(self.nodes)
            # Check if first node in new path matches any existing node
            existing_node = next(
                (n for n in self.nodes if n["content"] == new_nodes[0]["content"]),
                None,
            )

            if existing_node:
                self.logger.info(
                    f"Found matching existing node: {existing_node['content']}"
                )
                new_edge = {
                    "target_node_id": existing_node["node_id"],
                    "label": scenario["response"],
                }

                if new_edge not in node["edges"]:
                    node["edges"].append(new_edge)
                    self.visualizer.update_graph(self.nodes)
                    await self.dfs(existing_node, visited, depth + 1)
            else:
                new_edge = {
                    "target_node_id": new_nodes[0]["node_id"],
                    "label": scenario["response"],
                }

                if new_edge not in node["edges"]:
                    node["edges"].append(new_edge)
                    self.nodes.extend(new_nodes)
                    self.visualizer.update_graph(self.nodes)
                    await self.dfs(new_nodes[0], visited, depth + 1)

    async def dfs(
        self, node: Dict[str, Any], visited: set = None, depth: int = 0
    ) -> None:
        """Perform depth-first search traversal with dynamic path exploration.

        This method traverses the conversation graph using DFS while dynamically exploring
        new conversation paths at each node. It generates and tests alternative scenarios
        at decision points to discover new branches.

        Args:
            node (Dict[str, Any]): The current node being explored in the graph
            visited (set, optional): Set of already visited node IDs. Defaults to None.
            depth (int, optional): Current depth in the graph traversal. Defaults to 0.

        Returns:
            None

        Side Effects:
            - Updates self.nodes with newly discovered conversation paths
            - Updates self.depth_patterns with patterns found at each depth
            - Updates the graph visualization via self.visualizer
        """
        if visited is None:
            visited = set()

        if node["node_id"] in visited:
            return

        self.logger.info(
            f"Exploring node {node['node_id']}: {node['content']} at depth {depth}"
        )

        # Add this node's content to patterns
        if depth not in self.depth_patterns:
            self.depth_patterns[depth] = []
        if node["content"] not in self.depth_patterns[depth]:
            self.depth_patterns[depth].append(node["content"])
            self.logger.info(f"New pattern at depth {depth}: {node['content']}")

        visited.add(node["node_id"])

        # Generate alternative scenarios for decision points #
        self.logger.info(f"Found decision point: {node['content']}")

        # We do not want to run scenarios on greetings, and starting conversations #
        if depth != 0:
            scenarios = await self.llm_client.generate_scenarios(node)
            self.logger.info(f"Generated {len(scenarios)} scenarios")

            if not scenarios:
                self.logger.warning(
                    f"No scenarios generated for node: {node['content']}"
                )

            scenario_tasks = []
            for scenario in scenarios:
                task = self.run_scenario(scenario, node, visited, depth)
                scenario_tasks.append(task)

            # Wait for all scenario tasks to complete #
            if scenario_tasks:
                await asyncio.gather(*scenario_tasks)

        # Continue DFS on existing edges #
        edge_tasks = []
        for edge in node["edges"]:
            target_node = next(
                (n for n in self.nodes if n["node_id"] == edge["target_node_id"]), None
            )
            if target_node:
                task = self.dfs(target_node, visited, depth + 1)
                edge_tasks.append(task)

        # Wait for all edge exploration tasks to complete
        if edge_tasks:
            await asyncio.gather(*edge_tasks)

    async def run(self, phone_number: str, initial_prompt: str) -> List[Dict[str, Any]]:
        """Run the automated testing process for exploring conversation paths.

        This method initiates the conversation graph exploration by:
        1. Making an initial call using the provided phone number and prompt
        2. Transcribing the response
        3. Constructing the initial graph structure
        4. Performing depth-first search to explore conversation paths
        5. Visualizing the graph throughout the process

        Args:
            phone_number (str): The phone number to call for testing
            initial_prompt (str): The initial conversation prompt to start with

        Returns:
            None

        Raises:
            ValueError: If no nodes are found in the initial transcript
        """

        self.phone_number = phone_number
        self.initial_prompt = initial_prompt
        self.logger.info("START: Starting conversation graph exploration")

        # Start the Discovery Process #
        call_id = await self.phone_client.make_call(
            phone_number=phone_number, prompt=initial_prompt
        )
        audio_filepath = await self.phone_client.poll_for_response(call_id)
        transcript = await self.transcriber.transcribe(audio_filepath)

        # Get the initial graph #
        self.nodes, depth_patterns = await self.llm_client.get_nodes_from_transcript(
            transcript=transcript, depth_patterns=self.depth_patterns
        )
        self.logger.info(f"Initial graph structure: {json.dumps(self.nodes, indent=2)}")
        self.depth_patterns = depth_patterns
        self.visualizer.update_graph(self.nodes)

        if not self.nodes:
            raise ValueError("No nodes found in the initial transcript.")

        # Start DFS from root
        root_node = next(node for node in self.nodes if node["content"] == "Start")
        await self.dfs(root_node)

        self.logger.info(f"Final graph structure: {json.dumps(self.nodes, indent=2)}")
        return self.nodes


async def main():
    # Initialize modules #
    settings = Settings()
    logger = LoggerFactory()
    hamming_client = HammingClient(
        api_token=settings.hammingai_api_token, api_endpoint=settings.api_endpoint
    )
    llm_client = ClaudeClient(api_token=settings.claude_api_token)
    transcriber = AssemblyAITranscriber(api_token=settings.assemblyai_api_token)
    visualizer = DynamicGraphVisualizer()

    # Initialize graph constructor required modules. #
    gc = GraphConstructor(
        llm_client=llm_client,
        phone_client=hamming_client,
        visualizer=visualizer,
        transcriber=transcriber,
        logger=logger,
        settings=settings,
    )

    # Init Arguments #
    air_conditioning_phone_number = "+14153580761"
    initial_air_conditioning_prompt = """Hi, my AC unit isn't cooling properly - it's blowing warm air and making 
    unusual noises. The unit is about 5 years old. I'd like to schedule a service appointment to have someone 
    take a look at it. What are your availability and rates for AC repairs?"""

    return await gc.run(
        phone_number=air_conditioning_phone_number,
        initial_prompt=initial_air_conditioning_prompt,
    )


if __name__ == "__main__":
    result = asyncio.run(main())
    print(json.dumps(result, indent=2))
