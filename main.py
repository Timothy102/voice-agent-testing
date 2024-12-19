import json
from pydantic import BaseModel
import asyncio
import requests
import aiofiles
import aiohttp
import assemblyai as aai
import anthropic

from dataclasses import dataclass
from typing import List, Dict, Optional

from utils.settings import Settings
from utils.logger import LoggerFactory
from utils.visualizer import DecisionTreeVisualizer

@dataclass
class ConversationNode:
    content: str
    speaker: str  # 'agent' or 'customer'
    next_nodes: List['ConversationNode']
    node_id: str

class GraphConstructor(BaseModel):
    phone_number: str
    prompt: str
    nodes: List[ConversationNode] = []
    visited: set = set()
    settings: Settings = None
    logger: LoggerFactory = None
    transcriber: Optional[aai.Transcriber] = None
    llm_client: Optional[anthropic.Anthropic] = None

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
    
    async def call(self) -> int:
        """
        Makes a call to the Hamming API Endpoint.
        """
        # Make API call to start phone call
        headers = {
            "Authorization": f"Bearer {self.settings.hammingai_api_token}",
            "Content-Type": "application/json"
        }

        payload = {
            "phone_number": self.phone_number,
            "prompt": self.prompt,
            "webhook_url": "https://your-webhook-url.com/callback"  # Replace with actual webhook URL
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(self.settings.api_endpoint, json=payload, headers=headers) as response:
                self.logger.info(f"Calling phone number {self.phone_number} via API call to {self.settings.api_endpoint}")
                response_data = await response.json()
                return response_data["id"]
    
    async def poll_for_response(self, call_id: int, time_in_between_polls: int = 2, max_time: int = 480) -> str:
        headers = {
            "Authorization": f"Bearer {self.settings.hammingai_api_token}",
            "Content-Type": "application/json"
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
                filepath = f'transcripts/{call_id}.mp3'
                async with aiofiles.open(filepath, 'wb') as f:
                    await f.write(audio_content)

                self.logger.info(f"Successfully downloaded audio file for call {call_id} to {filepath}")
                return filepath
            
            # Breaking the code for server errors. #
            elif recording_response.status_code >= 500:
                raise ValueError(f"Server error {recording_response.status_code}, retrying...")
            
            # We cannot poll forever. Max time should define how long we ought to wait before breaking the program #
            if asyncio.get_event_loop().time() - start_time > max_time:  # 480 seconds = 8 minutes
                raise TimeoutError(f"Recording not available after {max_time} minutes of polling")
            
            # Wait before polling again
            await asyncio.sleep(time_in_between_polls)

    async def transcribe(self, filepath: str) -> str:
        return self.transcriber.transcribe(filepath).text
    
    async def get_nodes_from_transcript(self, transcript: str) -> List[Dict[str, Any]]:
        """Extract conversation flow from transcript with customer choices as edges"""
        
        system_prompt = """You are an expert at mapping customer service conversations into process flows.
        Focus only on what actually happened in the conversation.
        Customer responses should be treated as edge labels between decision points."""

        # Using triple quotes and raw string to avoid formatting issues
        user_prompt = f"""Map this conversation into a process flow.
        Rules:
        1. Start with a "Start" node
        2. Create nodes only for key system/agent decision points (e.g., "Membership Check", "Service Type")
        3. Customer responses should be edge labels, not nodes
        4. Only include paths that were actually taken in the conversation
        5. Each node should represent a distinct decision point or state
        6. End nodes should represent final outcomes

        Format as a JSON array where each node has:
        - node_id: unique identifier
        - content: the decision point or state 
        - speaker: "agent" or "system"
        - edges: list of objects containing:
            - target_node_id: ID of the next node
            - label: customer's response/choice that led to that path

        Example format:
        [
            {{
                "node_id": 1,
                "content": "Start",
                "speaker": "system",
                "edges": [
                    {{
                        "target_node_id": 2,
                        "label": ""
                    }}
                ]
            }},
            {{
                "node_id": 2,
                "content": "Membership Check",
                "speaker": "agent",
                "edges": [
                    {{
                        "target_node_id": 3,
                        "label": "not a member"
                    }}
                ]
            }}
        ]

        Transcript:
        {transcript}"""

        headers = {
            "x-api-key": self.settings.claude_api_token,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json"
        }

        payload = {
            "model": "claude-3-opus-20240229",
            "max_tokens": 1500,
            "temperature": 0.0,
            "system": system_prompt,
            "messages": [
                {"role": "user", "content": user_prompt}
            ]
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    "https://api.anthropic.com/v1/messages",
                    headers=headers,
                    json=payload
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
            if "```json" in response_text:
                json_content = response_text.split("```json")[1].split("```")[0].strip()
            else:
                json_start = response_text.find('[')
                json_end = response_text.rfind(']')
                
                if json_start == -1 or json_end == -1:
                    raise ValueError("No valid JSON array found in response")
                    
                json_content = response_text[json_start:json_end + 1]

            json_content = json_content.strip().replace(',]', ']')
            nodes = json.loads(json_content)
            
            # Validate node structure and ensure Start node exists
            has_start = False
            for node in nodes:
                if node.get('content') == 'Start':
                    has_start = True
                required_fields = ['node_id', 'content', 'speaker', 'edges']
                missing_fields = [field for field in required_fields if field not in node]
                if missing_fields:
                    self.logger.warning(f"Node {node.get('node_id', 'unknown')} missing fields: {missing_fields}")
            
            if not has_start:
                self.logger.warning("No Start node found in the tree")
            
            return nodes
            
        except Exception as e:
            self.logger.error(f"Error extracting nodes: {str(e)}")
            raise
            
    async def run(self) -> None:
        # Initialize nodes dictionary if not already done
        if not hasattr(self, 'nodes'):
            self.nodes = {}
        if not hasattr(self, 'visited'):
            self.visited = set()
            
        # Make initial call
        call_id = await self.call()
        
        # Poll for response and get audio file
        audio_filepath = await self.poll_for_response(call_id)
        
        # Get transcript
        transcript = await self.transcribe(audio_filepath)
        
        # Get root node and build graph
        root_node = await self.get_nodes_from_transcript(transcript)
        
        # Start DFS from root node
        self.logger.info(f"Starting DFS traversal from root node: {root_node.node_id}")
        await self.dfs(root_node)

    async def dfs(self, node: ConversationNode, depth: int = 0) -> None:
        """
        Perform DFS on conversation nodes. For each unvisited node:
        1. Mark as visited
        2. Process node (make call if needed)
        3. Recursively visit children
        """
        if node.node_id in self.visited:
            return
            
        # Mark node as visited
        self.visited.add(node.node_id)
        self.logger.info(f"Currently visiting node: {node.node_id}")

        
        # Process this node - make a call if needed
        if depth > 0:  # Skip initial node since we already made that call
            call_id = await self.call()
            audio_filepath = await self.poll_for_response(call_id)
            transcript = await self.transcribe(audio_filepath)
            
            # Update graph with new nodes from this transcript
            new_node = await self.get_nodes_from_transcript(transcript)
            node.next_nodes.extend(new_node.next_nodes)
        
        # Visit all unvisited children
        for next_node in node.next_nodes:
            if next_node.node_id not in self.visited:
                await self.dfs(next_node, depth + 1)

    def print_graph(self, node: ConversationNode, visited=None, level=0):
        """Helper function to visualize the graph"""
        if visited is None:
            visited = set()
            
        if node.node_id in visited:
            print("  " * level + f"[Cycle to {node.node_id}]")
            return
            
        visited.add(node.node_id)
        print("  " * level + f"[{node.speaker}] {node.content}")
        
        for next_node in node.next_nodes:
            self.print_graph(next_node, visited, level + 1)

async def main():
    auto_dealership_phone_number = "+1 (650) 879-8564"
    initial_prompt = """I'm interested in buying a used BMW. I'd like to know what models you have available 
    and what the price ranges are. I'm particularly interested in models from the last 5 years."""
    
    gc = GraphConstructor(phone_number=auto_dealership_phone_number, prompt=initial_prompt)
    await gc.run()
    
    # Print the resulting graph
    print("\nConversation Graph:")
    gc.print_graph(list(gc.nodes.values())[0])

    visualizer = DecisionTreeVisualizer()
    visualizer.create_graph(gc.nodes)
    visualizer.save_graph('conversation_tree')


if __name__ == "__main__":
    asyncio.run(main())