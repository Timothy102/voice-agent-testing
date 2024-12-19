"""Main module for constructing and exploring conversation graphs."""

import asyncio
import json
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import aiofiles
import aiohttp
import requests

from interface import IGraphConstructor
from modules.graph_visualizer.dash_viz import DynamicGraphVisualizer
from modules.graph_visualizer.interface import GraphVisualizerInterface
from modules.llm_client.claude import ClaudeClient
from modules.llm_client.interface import LLMClientInterface
from modules.logger.factory import LoggerFactory
from modules.logger.interface import LoggerInterface
from modules.transcriber.assembly_ai import AssemblyAITranscriber
from modules.transcriber.interface import TranscriberInterface
from utils.settings import Settings


@dataclass
class ConversationNode:
    """Represents a node in the conversation graph.

    Attributes:
        content: The text content of the node
        speaker: Who is speaking ('agent' or 'customer')
        next_nodes: List of nodes that follow this one
        node_id: Unique identifier for the node
    """

    content: str
    speaker: str  # 'agent' or 'customer'
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
        visualizer: GraphVisualizerInterface,
        transcriber: TranscriberInterface,
        logger: LoggerInterface,
        settings: Optional[Settings] = None,
    ) -> None:
        """Initialize the GraphConstructor with required components.

        Args:
            llm_client: Client for generating LLM responses
            graph_viz: Visualizer for rendering conversation graphs
            transcriber: Service for transcribing audio to text
            logger: Logger for tracking execution
        """
        super().__init__(llm_client, visualizer, transcriber, logger, settings)
        self.nodes: List[Dict[str, Any]] = []
        self.depth_patterns: Dict[int, List[str]] = {}
        self.logger.info("INFO: GraphConstructor initialized")

    async def generate_test_scenarios(
        self, node: Dict[str, Any], max_scenarios: int = 3
    ) -> List[Dict[str, str]]:
        """Generate alternative conversation paths for a decision point.

        Args:
            node: The current conversation node
            max_scenarios: Maximum number of scenarios to generate

        Returns:
            List of scenario dictionaries with 'response' and 'prompt' keys
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
            print("response_data", response_data)
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

    async def call(self) -> int:
        """Makes a call to the Hamming API Endpoint.

        Returns:
            The call ID from the Hamming API response
        """
        headers = {
            "Authorization": f"Bearer {self.settings.hammingai_api_token}",
            "Content-Type": "application/json",
        }

        payload = {
            "phone_number": self.phone_number,
            "prompt": self.initial_prompt,
            "webhook_url": "https://your-webhook-url.com/callback",
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(
                self.settings.api_endpoint, json=payload, headers=headers
            ) as response:
                self.logger.info(
                    f"INFO: Calling phone number {self.phone_number} via API call to {self.settings.api_endpoint}"
                )
                response_data = await response.json()
                return response_data["id"]

    async def poll_for_response(
        self, call_id: int, time_in_between_polls: int = 200, max_time: int = 480
    ) -> str:
        """Poll the API for a response recording.

        Args:
            call_id: ID of the call to poll for from Hamming AI
            time_in_between_polls: Time to wait between polls in seconds
            max_time: Maximum time to poll in seconds before raising a TimeoutError

        Returns:
            Path to the downloaded audio file

        Raises:
            TimeoutError: If max_time is exceeded
            ValueError: If server returns an error
        """
        headers = {
            "Authorization": f"Bearer {self.settings.hammingai_api_token}",
            "Content-Type": "application/json",
        }

        recording_url = f"https://app.hamming.ai/api/media/exercise?id={call_id}"

        self.logger.info(f"INFO: Starting to poll for response with call ID: {call_id}")

        start_time = asyncio.get_event_loop().time()
        while True:
            recording_response = requests.get(recording_url, headers=headers)

            if recording_response.status_code == 200:
                audio_content = recording_response.content

                filepath = f"transcripts/{call_id}.mp3"
                async with aiofiles.open(filepath, "wb") as f:
                    await f.write(audio_content)

                self.logger.info(f"INFO: Successfully downloaded audio to {filepath}")
                return filepath

            elif recording_response.status_code >= 500:
                raise ValueError(
                    f"Server error {recording_response.status_code}, retrying..."
                )

            if asyncio.get_event_loop().time() - start_time > max_time:
                raise TimeoutError(
                    f"Recording not available after {max_time} minutes of polling"
                )

            await asyncio.sleep(time_in_between_polls)

    async def get_nodes_from_transcript(
        self, transcript: str, current_depth: int = 0
    ) -> List[Dict[str, Any]]:
        """Convert a transcript into graph nodes with concise content."""
        depth_context = ""
        if self.depth_patterns:
            depth_context = "Known decision points by depth:\n"
            for depth, patterns in self.depth_patterns.items():
                depth_context += f"Depth {depth}: {', '.join(patterns)}\n"

        system_prompt = """You are an expert at converting customer service conversations into clear, concise decision flows.
        Think like a minimalist flowchart designer - use the shortest possible phrases that capture the decision point.
        Every node should be 1-4 words maximum."""

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
        5. If you see a question/action that matches these known patterns at depth {current_depth}, use the EXACT same content:
        {depth_context}

        Return your response as a SINGLE, COMPLETE JSON array.
        The first node should always have content be "Start". ALWAYS.
        The array must start with '[' and end with ']'.
        No text before or after the JSON array.

        Format each node as:
        {{
            "node_id": number,
            "content": "very short phrase",
            "speaker": "agent" or "system",
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

        try:
            # Get response from LLM
            response = await self.llm_client.generate(
                user_prompt=user_prompt, system_prompt=system_prompt
            )

            self.logger.debug(f"Raw LLM response: {response}")

            # Parse response string to extract JSON
            if isinstance(response, str):
                # Clean the string and find the JSON array
                response = response.strip()
                start_idx = response.find("[")
                end_idx = response.rfind("]")

                if start_idx == -1 or end_idx == -1:
                    raise ValueError("No valid JSON array found in response")

                json_str = response[start_idx : end_idx + 1]
                nodes = json.loads(json_str)
            else:
                # If response is already parsed JSON (dict or list)
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

        except json.JSONDecodeError as e:
            self.logger.error(f"JSON parsing error: {str(e)}")
            self.logger.debug(f"Failed to parse response: {response}")
            raise
        except Exception as e:
            self.logger.error(f"Error processing transcript: {str(e)}")
            raise

    def _extract_nodes(self, response_text: str) -> List[Dict[str, Any]]:
        """Extract and validate nodes from Claude's response.

        Args:
            response_text: Raw response text from Claude

        Returns:
            List of validated node dictionaries

        Raises:
            ValueError: If node validation fails
        """
        try:
            # Forcing JSON structure extraction #
            if "```json" in response_text:
                json_content = response_text.split("```json")[1].split("```")[0].strip()
            else:
                start_idx = response_text.find("[")
                end_idx = response_text.rfind("]")

                if start_idx == -1 or end_idx == -1:
                    self.logger.error("ERROR: No complete JSON array found in response")
                    self.logger.debug(f"DEBUG: Response text: {response_text}")
                    raise ValueError("No valid JSON array found in response")

                json_content = response_text[start_idx : end_idx + 1]

            # Clean up the JSON content
            json_content = json_content.strip()

            # Remove any trailing commas before closing brackets
            json_content = re.sub(r",(\s*})", r"\1", json_content)
            json_content = re.sub(r",(\s*])", r"\1", json_content)

            # Parse the complete array
            try:
                nodes = json.loads(json_content)
            except json.JSONDecodeError as e:
                self.logger.error(f"ERROR: JSON decode error: {str(e)}")
                self.logger.debug(f"DEBUG: Attempted to parse: {json_content}")
                raise

            # Validate nodes
            has_start = False
            for node in nodes:
                if node.get("content") == "Start":
                    has_start = True
                required_fields = ["node_id", "content", "speaker", "edges"]
                missing_fields = [
                    field for field in required_fields if field not in node
                ]
                if missing_fields:
                    self.logger.warning(
                        f"WARNING: Node {node.get('node_id', 'unknown')} missing fields: {missing_fields}"
                    )

            if not has_start:
                self.logger.warning("WARNING: No Start node found in the tree")

            return nodes

        except Exception as e:
            self.logger.error(f"ERROR: Error extracting nodes: {str(e)}")
            raise

    async def _process_scenario(
        self, scenario: Dict[str, str], node: Dict[str, Any], visited: set, depth: int
    ) -> None:
        """Process a single test scenario.

        Args:
            scenario: The scenario to test
            node: Current conversation node
            visited: Set of visited node IDs
            depth: Current depth in conversation tree
        """
        self.logger.info(f"Testing scenario: {scenario['response']}")

        # Update prompt for this scenario
        self.prompt = scenario["prompt"]

        # Make new call
        call_id = await self.call()
        audio_filepath = await self.poll_for_response(call_id)
        print("audio_filepath", audio_filepath)
        new_transcript = await self.transcriber.transcribe(audio_filepath)
        self.logger.info(f"Transcript: {new_transcript}")

        # Get new nodes with depth context
        new_nodes = await self.get_nodes_from_transcript(new_transcript, depth + 1)
        self.logger.info(f"New path discovered: {json.dumps(new_nodes, indent=2)}")

        if new_nodes:
            self.visualizer.update_graph(self.nodes)
            # Check if first node in new path matches any existing node
            existing_node = next(
                (
                    n
                    for n in self.nodes
                    if n["speaker"] == new_nodes[0]["speaker"]
                    and n["content"] == new_nodes[0]["content"]
                ),
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
        """Perform DFS with dynamic path exploration"""
        if visited is None:
            visited = set()

        if node["node_id"] in visited:
            return

        self.logger.info(
            f"Exploring node {node['node_id']}: {node['content']} at depth {depth}"
        )

        # Add this node's content to patterns if it's an agent node
        if node["speaker"] == "agent":
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

    async def run(self, phone_number: str, initial_prompt: str) -> None:
        """Run the automated testing process"""

        self.phone_number = phone_number
        self.initial_prompt = initial_prompt
        self.visualizer = DynamicGraphVisualizer()

        self.logger.info("START: Starting conversation graph exploration")

        # Make initial call
        call_id = await self.call()
        audio_filepath = await self.poll_for_response(call_id)
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


async def main():
    # Initialize components
    settings = Settings()
    llm_client = ClaudeClient(api_token=settings.claude_api_token)
    logger = LoggerFactory()
    transcriber = AssemblyAITranscriber(api_token=settings.assemblyai_api_token)
    visualizer = DynamicGraphVisualizer()

    # Initialize graph constructor with components
    gc = GraphConstructor(
        llm_client=llm_client,
        logger=logger,
        transcriber=transcriber,
        visualizer=visualizer,
        settings=settings,
    )

    auto_dealership_phone_number = "+1 (650) 879-8564"
    initial_prompt = """I'm interested in buying a used BMW. I'd like to know what models you have available 
    and what the price ranges are. I'm particularly interested in models from the last 5 years."""

    await gc.run(
        phone_number=auto_dealership_phone_number, initial_prompt=initial_prompt
    )


if __name__ == "__main__":
    asyncio.run(main())