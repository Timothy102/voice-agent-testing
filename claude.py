import json
import aiohttp
import assemblyai as aai
import anthropic
import graphviz
from typing import List, Dict, Any
from utils.settings import Settings
from utils.logger import LoggerFactory

class TranscriptionService:
    def __init__(self):
        self.settings = Settings()
        self.logger = LoggerFactory()
        aai.settings.api_key = self.settings.assemblyai_api_token
        self.transcriber = aai.Transcriber()

    async def transcribe(self, filepath: str) -> str:
        """Transcribe audio file using AssemblyAI"""
        return self.transcriber.transcribe(filepath).text

class ClaudeService:
    def __init__(self):
        self.settings = Settings()
        self.logger = LoggerFactory()
        self.llm_client = anthropic.Anthropic(
            api_key=self.settings.claude_api_token,
        )

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

async def main():
    # Example usage
    transcription_service = TranscriptionService()
    claude_service = ClaudeService()

    # Test transcription
    audio_file = "transcripts/cm4u5chwj00luirz8bepiwfzc.mp3"  # Replace with actual test file
    transcript = await transcription_service.transcribe(audio_file)
    print(f"Transcript: {transcript}")

    # Test Claude analysis
    nodes = await claude_service.get_nodes_from_transcript(transcript)
    claude_service.logger.info(f"Conversation nodes: {json.dumps(nodes, indent=2)}")

    from utils.visualizer import DecisionTreeVisualizer
    visualizer = DecisionTreeVisualizer()
    graph = visualizer.create_graph(nodes)
    visualizer.save_graph('conversation_tree')

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
