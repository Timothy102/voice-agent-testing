import asyncio
import json
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import aiofiles
import aiohttp
import anthropic
import assemblyai as aai
import requests
from pydantic import BaseModel

from utils.dash_viz import DynamicGraphVisualizer
from utils.logger import LoggerFactory
from utils.settings import Settings
from utils.visualizer import DecisionTreeVisualizer


@dataclass
class ConversationNode:
    content: str
    speaker: str  # 'agent' or 'customer'
    next_nodes: List["ConversationNode"]
    node_id: str


class GraphConstructor(BaseModel):
    phone_number: str
    prompt: str
    nodes: List[Dict[str, Any]] = []
    visited: set = set()
    settings: Settings = None
    logger: LoggerFactory = None
    transcriber: Optional[aai.Transcriber] = None
    llm_client: Optional[anthropic.Anthropic] = None
    visualizer: Optional[DynamicGraphVisualizer] = None

    depth_patterns: Dict[int, List[str]] = {}  # depth -> list of known questions

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, **data):
        super().__init__(**data)
        self.settings = Settings()
        self.logger = LoggerFactory()
        aai.settings.api_key = self.settings.assemblyai_api_token
        self.transcriber = aai.Transcriber()
        self.llm_client = anthropic.Anthropic(
            api_key=self.settings.claude_api_token,
        )

    async def generate_test_scenarios(
        self, node: Dict[str, Any], max_scenarios: int = 3
    ) -> List[Dict[str, str]]:
        """Generate alternative paths for a decision point"""
        self.logger.info(f"Generating test scenarios for node: {node['content']}")

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
            # Use aiohttp directly instead of anthropic client
            headers = {
                "x-api-key": self.settings.claude_api_token,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            }

            payload = {
                "model": "claude-3-opus-20240229",
                "max_tokens": 1500,
                "temperature": 0,
                "system": system_prompt,
                "messages": [{"role": "user", "content": user_prompt}],
            }

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    "https://api.anthropic.com/v1/messages",
                    headers=headers,
                    json=payload,
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        raise Exception(f"Claude API error: {error_text}")

                    response_data = await response.json()

            # Log raw response
            self.logger.info(
                f"Raw Claude response: {response_data['content'][0]['text']}"
            )

            # Extract JSON from response
            response_text = response_data["content"][0]["text"]

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

            self.logger.info(f"Parsed scenarios: {json.dumps(scenarios, indent=2)}")
            return scenarios

        except Exception as e:
            self.logger.error(f"Error generating scenarios: {str(e)}")
            if "response_data" in locals():
                self.logger.error(
                    f"Failed to parse response: {response_data['content'][0]['text']}"
                )
            else:
                self.logger.error("No response received from Claude")
            return []

    async def call(self) -> int:
        """Makes a call to the Hamming API Endpoint."""
        headers = {
            "Authorization": f"Bearer {self.settings.hammingai_api_token}",
            "Content-Type": "application/json",
        }

        payload = {
            "phone_number": self.phone_number,
            "prompt": self.prompt,
            "webhook_url": "https://your-webhook-url.com/callback",
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(
                self.settings.api_endpoint, json=payload, headers=headers
            ) as response:
                self.logger.info(
                    f"Calling phone number {self.phone_number} via API call to {self.settings.api_endpoint}"
                )
                response_data = await response.json()
                return response_data["id"]

    async def poll_for_response(
        self, call_id: int, time_in_between_polls: int = 2, max_time: int = 480
    ) -> str:
        headers = {
            "Authorization": f"Bearer {self.settings.hammingai_api_token}",
            "Content-Type": "application/json",
        }

        # URL to get response from #
        recording_url = f"https://app.hamming.ai/api/media/exercise?id={call_id}"

        # Start polling #
        start_time = asyncio.get_event_loop().time()
        while True:
            recording_response = requests.get(recording_url, headers=headers)

            if recording_response.status_code == 200:
                audio_content = recording_response.content

                # Save the audio content to a file (optional but recommended for debugging)
                filepath = f"transcripts/{call_id}.mp3"
                async with aiofiles.open(filepath, "wb") as f:
                    await f.write(audio_content)

                return filepath

            # Breaking the code for server errors. #
            elif recording_response.status_code >= 500:
                raise ValueError(
                    f"Server error {recording_response.status_code}, retrying..."
                )

            # We cannot poll forever. Max time should define how long we ought to wait before breaking the program #
            if asyncio.get_event_loop().time() - start_time > max_time:
                raise TimeoutError(
                    f"Recording not available after {max_time} minutes of polling"
                )

            # Wait before polling again
            await asyncio.sleep(time_in_between_polls)

    async def transcribe(self, filepath: str) -> str:
        self.logger.info(f"Transcribing audio file: {filepath}")
        transcript = self.transcriber.transcribe(filepath).text
        self.logger.info(f"Transcription result: {transcript}")
        return transcript

    async def get_nodes_from_transcript(
        self, transcript: str, current_depth: int = 0
    ) -> List[Dict[str, Any]]:
        depth_context = ""
        if self.depth_patterns:
            depth_context = "Known decision points by depth:\n"
            for depth, patterns in self.depth_patterns.items():
                depth_context += f"Depth {depth}: {patterns}\n"

        system_prompt = """You are an expert at converting customer service conversations into clear decision flows.
        Think like a flowchart designer - focus on key decision points and their explicit outcomes.
        Structure nodes as clear questions with specific possible answers."""

        user_prompt = f"""Convert this conversation into a decision flow similar to a customer service flowchart.

        Key rules:
        1. Start with "Start" node
        2. If you see a question similar to any known one at the same depth, USE THE EXACT KNOWN QUESTION. Use these depth questions to compare the input to:
        {depth_context}
        3. Format decision points as clear questions with "?" (e.g., "Are you a member?")
        4. Keep node content very brief and human-like (e.g., "Collect Customer Info" not "Agent proceeds to gather customer information")
        5. Edge labels should be specific answers/choices (e.g., "Gold Member", "Not a Member")
        6. Agent actions should be simple and clear (e.g., "Schedule Appointment" not "Agent initiates appointment scheduling process")
        7. Think in terms of a visual flowchart - each node should represent one clear decision or action

        Do not include formalities, like greeting customers, thanking them, etc. Stricly decision-making.
        Return your response as a SINGLE, COMPLETE JSON array. 
        The array must start with '[' and end with ']' and contain all nodes.
        Do not include any text before or after the JSON array.

        Example decision point:
        Node: "Are you a member?"
        Possible edges: 
        - "Gold Member" -> Transfer to Premium
        - "Silver Member" -> Describe Issue
        - "Not a Member" -> Collect New Member Info

        DO NOT return separate JSON objects. Put everything in one array.

        Format each node as:
        {{
            "node_id": number,
            "content": "clear question or action",
            "speaker": "agent" or "system", 
            "edges": [
                {{
                    "target_node_id": number,
                    "label": "specific customer choice or outcome"
                }}
            ]
        }}

        Transcript:
        {transcript}"""

        headers = {
            "x-api-key": self.settings.claude_api_token,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }

        payload = {
            "model": "claude-3-opus-20240229",
            "max_tokens": 1500,
            "temperature": 0.0,
            "system": system_prompt,
            "messages": [{"role": "user", "content": user_prompt}],
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    "https://api.anthropic.com/v1/messages",
                    headers=headers,
                    json=payload,
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        raise Exception(f"Claude API error: {error_text}")

                    response_data = await response.json()

            self.logger.info(f"Completion: {response_data}")

            response_text = response_data["content"][0]["text"]
            return self._extract_nodes(response_text)

        except Exception as e:
            self.logger.error(f"Error processing transcript: {str(e)}")
            raise

    def _extract_nodes(self, response_text: str) -> List[Dict[str, Any]]:
        """Extract and validate nodes from Claude's response"""
        try:
            # First try to find a complete JSON array
            if "```json" in response_text:
                json_content = response_text.split("```json")[1].split("```")[0].strip()
            else:
                # Look for the complete array structure
                start_idx = response_text.find("[")
                end_idx = response_text.rfind("]")

                if start_idx == -1 or end_idx == -1:
                    self.logger.error("No complete JSON array found in response")
                    self.logger.debug(f"Response text: {response_text}")
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
                self.logger.error(f"JSON decode error: {str(e)}")
                self.logger.debug(f"Attempted to parse: {json_content}")
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
                        f"Node {node.get('node_id', 'unknown')} missing fields: {missing_fields}"
                    )

            if not has_start:
                self.logger.warning("No Start node found in the tree")

            return nodes

        except Exception as e:
            self.logger.error(f"Error extracting nodes: {str(e)}")
            raise

    async def _process_scenario(
        self, scenario: Dict[str, str], node: Dict[str, Any], visited: set, depth: int
    ) -> None:
        """Process a single scenario"""
        self.logger.info(f"Testing scenario: {scenario['response']}")

        # Update prompt for this scenario
        self.prompt = scenario["prompt"]

        # Make new call
        call_id = await self.call()
        audio_filepath = await self.poll_for_response(call_id)
        new_transcript = await self.transcribe(audio_filepath)

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
                    self.visualizer.update_graph(
                        self.nodes
                    )  # Update after edge addition
                    await self.dfs(existing_node, visited, depth + 1)
            else:
                new_edge = {
                    "target_node_id": new_nodes[0]["node_id"],
                    "label": scenario["response"],
                }

                if new_edge not in node["edges"]:
                    node["edges"].append(new_edge)
                    self.nodes.extend(new_nodes)
                    self.visualizer.update_graph(
                        self.nodes
                    )  # Update after edge addition
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
        scenarios = await self.generate_test_scenarios(node)
        self.logger.info(f"Generated {len(scenarios)} scenarios")

        if not scenarios:
            self.logger.warning(f"No scenarios generated for node: {node['content']}")

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

    async def run(self) -> None:
        """Run the automated testing process"""

        self.visualizer = DynamicGraphVisualizer()

        self.logger.info("START: Starting conversation graph exploration")

        # Make initial call
        call_id = await self.call()
        audio_filepath = await self.poll_for_response(call_id)
        transcript = await self.transcribe(audio_filepath)

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

        # Visualize final graph
        visualizer = DecisionTreeVisualizer()
        visualizer.create_graph(self.nodes)
        visualizer.save_graph("conversation_tree")


async def main():
    auto_dealership_phone_number = "+1 (650) 879-8564"
    initial_prompt = """I'm interested in buying a used BMW. I'd like to know what models you have available 
    and what the price ranges are. I'm particularly interested in models from the last 5 years."""

    gc = GraphConstructor(
        phone_number=auto_dealership_phone_number, prompt=initial_prompt
    )
    await gc.run()


if __name__ == "__main__":
    asyncio.run(main())
