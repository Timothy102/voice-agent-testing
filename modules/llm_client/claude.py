import asyncio
import json
import re
from typing import Any, Dict, List, Optional

import aiohttp

from modules.llm_client.interface import LLMClientInterface
from modules.logger.factory import LoggerFactory


class ClaudeClient(LLMClientInterface):
    def __init__(self, api_token: str, model: str = "claude-3-5-sonnet-20240620"):
        """
        Initialize the Claude client with the given API token and model.

        Args:
            api_token (str): The API token for authenticating with the Anthropic service.
            model (str): The model to use for generating responses. Defaults to "claude-3-sonnet-20240229".
        """
        self.api_token = api_token
        self.model = model
        self.logger = LoggerFactory()

    async def generate(
        self, user_prompt: str, system_prompt: Optional[str] = None
    ) -> str:
        """
        Generate a response from the Claude model based on the given user and optional system prompts.

        Args:
            user_prompt (str): The prompt provided by the user.
            system_prompt (Optional[str]): An optional system prompt to guide the model's response.

        Returns:
            str: The generated response from the Claude model.

        Raises:
            Exception: If there is a network error, unexpected response format, or other error when calling the Anthropic API.
        """
        headers = {
            "x-api-key": self.api_token,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }

        messages = [{"role": "user", "content": user_prompt}]

        payload = {
            "model": self.model,
            "max_tokens": 1500,
            "temperature": 0,
            "messages": messages,
        }

        if system_prompt:
            payload["system"] = system_prompt

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    "https://api.anthropic.com/v1/messages",
                    json=payload,
                    headers=headers,
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        raise Exception(f"Anthropic API error: {error_text}")

                    response_data = await response.json()
                    return response_data["content"][0]["text"]

        except aiohttp.ClientError as e:
            raise Exception(f"Network error when calling Anthropic API: {str(e)}")
        except KeyError as e:
            raise Exception(f"Unexpected response format from Anthropic API: {str(e)}")
        except Exception as e:
            raise Exception(f"Error calling Anthropic API: {str(e)}")

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
            response_data = await self.generate(
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
        self,
        transcript: str,
        depth_patterns: Dict[int, List[str]],
        current_depth: int = 0,
        max_retries: int = 3,
    ) -> tuple[List[Dict[str, Any]], Dict[int, List[str]]]:
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
                if depth_patterns:
                    depth_context = "Known decision points by depth:\n"
                    for depth, patterns in depth_patterns.items():
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
                response = await self.generate(
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
                    if depth not in depth_patterns:
                        depth_patterns[depth] = []
                    if node["content"] not in depth_patterns[depth]:
                        depth_patterns[depth].append(node["content"])

                return nodes, depth_patterns

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
