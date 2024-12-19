"""Main module for constructing and exploring conversation graphs."""

import asyncio
import json
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from interface import IGraphConstructor
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


class GraphConstructor(IGraphConstructor):
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

    async def generate_test_scenarios(
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
        self.logger.info(f"INFO: Generating test scenarios for node: {node['content']}")

        system_prompt = """You are an expert at generating test scenarios for conversation branches.
        For each decision point, generate alternative paths that test different customer responses."""

        user_prompt = f"""At this point in the conversation:
        Current node: {node['content']}
        Current response: {[edge['label'] for edge in node['edges']]}

        Generate {max_scenarios} alternative scenarios that test different customer responses.
        For example, if asking about membership, consider:
        - Being a gold member
        - Being a silver member
        - Not being a member

        Return ONLY a JSON array like this:
        [
            {{
                "response": "I am a gold member",
                "prompt": "I'm a gold member interested in buying a used BMW..."
            }},
            {{
                "response": "I am a silver member",
                "prompt": "I have a silver membership and I'm looking for used BMWs..."
            }}
        ]
        """

        try:
            response_data = await self.llm_client.generate(
                user_prompt=user_prompt, system_prompt=system_prompt
            )

            # Log raw response
            self.logger.debug(f"DEBUG: Claude response: {response_data}")

            # Extract JSON from response
            response_text = response_data

            # Find JSON content
            if "```json" in response_text:
                json_text = response_text.split("```json")[1].split("```")[0].strip()
            else:
                # Try to find array brackets
                start_idx = response_text.find("[")
                end_idx = response_text.rfind("]")
                if start_idx == -1 or end_idx == -1:
                    raise ValueError("No valid JSON array found in response")
                json_text = response_text[start_idx : end_idx + 1]

            # Clean up any trailing commas
            json_text = json_text.strip().replace(",]", "]")

            # Parse scenarios
            scenarios = json.loads(json_text)

            # Validate scenario format
            for scenario in scenarios:
                if not isinstance(scenario, dict):
                    raise ValueError("Scenario is not a dictionary")
                if "response" not in scenario or "prompt" not in scenario:
                    raise ValueError("Scenario missing required fields")

            self.logger.info(
                f"INFO: Parsed scenarios: {json.dumps(scenarios, indent=2)}"
            )
            return scenarios

        except Exception as e:
            self.logger.error(f"ERROR: Error generating scenarios: {str(e)}")
            if "response_data" in locals():
                self.logger.error(f"ERROR: Failed to parse response: {response_data}")
            else:
                self.logger.error("ERROR: No response received from Claude")
            return []

    async def get_nodes_from_transcript(
        self, transcript: str, current_depth: int = 0, max_retries: int = 3
    ) -> List[Dict[str, Any]]:
        """Convert a conversation transcript into a graph representation with concise nodes.

        Args:
            transcript (str): The conversation transcript to convert into nodes
            current_depth (int, optional): Current depth in the conversation tree. Defaults to 0.
            max_retries (int, optional): Maximum number of retries for failed conversions. Defaults to 3.

        Returns:
            List[Dict[str, Any]]: List of node dictionaries, each containing:
                - node_id (int): Unique identifier for the node
                - content (str): Concise 1-4 word description of the node
                - depth (int): Depth of node in conversation tree
                - edges (List[Dict]): List of edges to other nodes

        Raises:
            ValueError: If transcript cannot be converted to valid nodes
        """
        for attempt in range(max_retries):
            try:
                depth_context = ""
                if self.depth_patterns:
                    depth_context = "Known decision points by depth:\n"
                    for depth, patterns in self.depth_patterns.items():
                        depth_context += f"Depth {depth}: {', '.join(patterns)}\n"

                system_prompt = """You are an expert at converting customer service conversations into clear, concise decision flows.
                Think like a minimalist flowchart designer - use the shortest possible phrases that capture the decision point.
                Every node should be 1-4 words maximum.
                
                You must return ONLY a valid JSON array, nothing else."""

                user_prompt = f"""Convert this conversation into a minimal decision flow with extremely concise nodes.

                Key rules for node content:
                1. Maximum 4 words per node
                2. Use shortest possible phrases:
                - "Member?" instead of "Are you an existing customer with us?"
                - "Budget?" instead of "Please share your budget and preferences"
                - "Schedule Visit?" instead of "Would you like to schedule an appointment?"
                - "Transfer to Agent" instead of "Transfer customer to an agent who can help"
                3. For questions, use format: "Topic?"
                4. For actions, use imperative form: "Schedule Call", "Transfer"
                5. Never just use information like: "Provide Information", but always something more, like "Providing Model Information"  
                5. If you see a question/action that matches these known patterns at depth {current_depth}, use the EXACT same content:
                {depth_context}

                The first node should always have content be "Start". ALWAYS.
                Return ONLY a valid JSON array. No text before or after.

                Format each node as:
                {{
                    "node_id": number,
                    "content": "very short phrase",
                    "depth": number,
                    "edges": [
                        {{
                            "target_node_id": number,
                            "label": "brief label"
                        }}
                    ]
                }}

                Transcript:
                {transcript}"""

                # Get response from LLM
                response = await self.llm_client.generate(
                    user_prompt=user_prompt, system_prompt=system_prompt
                )

                self.logger.debug(
                    f"Raw LLM response (attempt {attempt + 1}): {response}"
                )

                # Parse response string to extract JSON
                if isinstance(response, str):
                    # Clean the string and find the JSON array
                    response = response.strip()
                    start_idx = response.find("[")
                    end_idx = response.rfind("]")

                    if start_idx == -1 or end_idx == -1:
                        raise ValueError("No valid JSON array found in response")

                    json_str = response[start_idx : end_idx + 1]
                    # Clean up common JSON issues
                    json_str = re.sub(
                        r",(\s*[}\]])", r"\1", json_str
                    )  # Remove trailing commas
                    json_str = re.sub(
                        r'\\([^"])', r"\1", json_str
                    )  # Fix escaped characters

                    self.logger.debug(
                        f"Attempting to parse JSON (attempt {attempt + 1}): {json_str}"
                    )
                    nodes = json.loads(json_str)
                else:
                    nodes = response

                # Validate and process nodes
                if not isinstance(nodes, list):
                    raise ValueError(f"Expected list of nodes, got {type(nodes)}")

                # Update depth patterns
                for node in nodes:
                    depth = node.get("depth", 0)
                    if depth not in self.depth_patterns:
                        self.depth_patterns[depth] = []
                    if node["content"] not in self.depth_patterns[depth]:
                        self.depth_patterns[depth].append(node["content"])

                return nodes

            except (json.JSONDecodeError, ValueError) as e:
                self.logger.warning(
                    f"Attempt {attempt + 1}/{max_retries} failed: {str(e)}"
                )
                if attempt == max_retries - 1:  # Last attempt
                    self.logger.error("All attempts to parse nodes failed")
                    self.logger.error(f"Final response that failed: {response}")
                    raise
                await asyncio.sleep(1)  # Wait before retrying
            except Exception as e:
                self.logger.error(f"Unexpected error: {str(e)}")
                self.logger.error(f"Full response: {response}")
                raise

    async def _process_scenario(
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

        # Update prompt for this scenario
        self.prompt = scenario["prompt"]

        # Make new call
        call_id = await self.phone_client.make_call(
            phone_number=self.phone_number, prompt=self.prompt
        )
        audio_filepath = await self.poll_for_response(call_id)

        new_transcript = await self.transcriber.transcribe(audio_filepath)
        self.logger.info(f"Transcript: {new_transcript}")

        # Get new nodes with depth context
        new_nodes = await self.get_nodes_from_transcript(new_transcript, depth + 1)
        self.logger.info(f"New path discovered: {json.dumps(new_nodes, indent=2)}")

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

        # Generate alternative scenarios for decision points
        self.logger.info(f"Found decision point: {node['content']}")

        # We do not want to run scenarios on greetings, and starting conversations #
        if depth != 0:
            scenarios = await self.generate_test_scenarios(node)
            self.logger.info(f"Generated {len(scenarios)} scenarios")

            if not scenarios:
                self.logger.warning(
                    f"No scenarios generated for node: {node['content']}"
                )

            scenario_tasks = []
            for scenario in scenarios:
                task = self._process_scenario(scenario, node, visited, depth)
                scenario_tasks.append(task)

            # Wait for all scenario tasks to complete
            if scenario_tasks:
                await asyncio.gather(*scenario_tasks)

        # Continue DFS on existing edges
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

        # Make initial call
        call_id = await self.phone_client.make_call(
            phone_number=phone_number, prompt=initial_prompt
        )
        audio_filepath = await self.phone_client.poll_for_response(call_id)
        transcript = await self.transcriber.transcribe(audio_filepath)

        # Get initial graph
        self.nodes = await self.get_nodes_from_transcript(transcript)
        self.logger.info(f"Initial graph structure: {json.dumps(self.nodes, indent=2)}")

        self.visualizer.update_graph(self.nodes)

        if not self.nodes:
            raise ValueError("No nodes found in initial transcript")

        # Start DFS from root
        root_node = next(node for node in self.nodes if node["content"] == "Start")
        await self.dfs(root_node)

        self.logger.info(f"Final graph structure: {json.dumps(self.nodes, indent=2)}")
        return self.nodes


async def main():
    # Initialize components
    settings = Settings()
    logger = LoggerFactory()  # Move logger initialization to top
    hamming_client = HammingClient(
        api_token=settings.hammingai_api_token, api_endpoint=settings.api_endpoint
    )
    llm_client = ClaudeClient(api_token=settings.claude_api_token)
    transcriber = AssemblyAITranscriber(api_token=settings.assemblyai_api_token)
    visualizer = DynamicGraphVisualizer()

    # Initialize graph constructor with components
    gc = GraphConstructor(
        llm_client=llm_client,
        phone_client=hamming_client,
        visualizer=visualizer,
        transcriber=transcriber,
        logger=logger,
        settings=settings,
    )

    # Init Arguments
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
