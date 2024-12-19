# Conversation Graph Explorer

A tool for automatically exploring and mapping out conversational AI agent behaviors through synthetic phone calls.

## Overview

This tool takes a phone number and initial prompt as input, then systematically explores all possible conversation paths with the agent through automated testing. It builds a comprehensive graph visualization showing the different branches and decision points in the conversation flow.

## How It Works

### Core Components

1. **GraphConstructor** - Main class orchestrating the exploration process
   - Manages the overall conversation flow and graph construction
   - Implements depth-first search to systematically explore paths
   - Tracks conversation patterns at each depth level

2. **LLM Client** - Handles interaction with language models
   - Generates alternative conversation scenarios
   - Helps analyze and structure conversation flows

3. **Transcriber** - Converts audio to text
   - Uses AssemblyAI for accurate speech-to-text conversion
   - Processes recorded phone call audio

4. **Graph Visualizer** - Creates interactive graph displays
   - Built with Dash for dynamic visualization
   - Updates in real-time as new paths are discovered

### Process Flow

1. **Initial Call**
   - Makes first call using provided phone number and prompt
   - Records and transcribes the conversation

2. **Graph Construction**
   - Converts transcript into structured node format
   - Each node contains:
     - Content (1-4 word description)
     - Speaker (agent/system)
     - Depth level
     - Connected edges

3. **Path Exploration**
   - Uses DFS to systematically explore conversation branches
   - Generates alternative scenarios at each decision point
   - Tracks patterns at each conversation depth
   - Avoids redundant paths by matching existing nodes

4. **Visualization**
   - Dynamically updates graph as new paths are discovered
   - Shows conversation flow and decision points
   - Highlights different speakers and depth levels

## Key Features

- **Automated Discovery**: Systematically explores all conversation paths
- **Pattern Recognition**: Identifies common patterns at each depth
- **Concise Representation**: Converts verbose conversations into minimal decision flows
- **Real-time Visualization**: Dynamic graph updates during exploration
- **Depth-First Approach**: Ensures thorough path exploration
- **Deduplication**: Prevents redundant conversation branches

## Usage

1. **Installation**
   ```bash
   # Clone the repository
   git clone https://github.com/Timothy102/hamming-ai-takehome
   cd conversation-graph

   # Install dependencies using Poetry
   poetry install
   ```

2. **API Keys Setup**
   Create a `.env` file in the root directory with your API keys:
   ```
   ASSEMBLY_AI_API_KEY=your_assembly_ai_key
   ANTHROPIC_API_KEY=your_claude_key 
   HAMMING_API_KEY=your_hamming_key
   ```

3. **Running the Tool**
   ```python
   # main.py
   import asyncio
   from modules.llm_client.claude import ClaudeClient
   from modules.graph_viz.dash_viz import DynamicGraphVisualizer
   from modules.transcriber.assembly_ai import AssemblyAITranscriber
   from modules.logger.logger import LoggerFactory
   from main2 import GraphConstructor

   async def main():
       # Initialize components
       llm_client = ClaudeClient()
       graph_viz = DynamicGraphVisualizer()
       transcriber = AssemblyAITranscriber()
       logger = LoggerFactory()

       # Initialize graph constructor
       gc = GraphConstructor(
           llm_client=llm_client,
           graph_viz=graph_viz,
           transcriber=transcriber,
           logger=logger
       )

       # Configure call parameters
       phone_number = "+1 (555) 123-4567"  # Target phone number
       initial_prompt = "I'm interested in your services"  # Opening message

       # Run the conversation exploration
       await gc.run(phone_number=phone_number, initial_prompt=initial_prompt)

   if __name__ == "__main__":
       asyncio.run(main())
   ```

4. **Viewing Results**
   - The graph visualization will automatically open in your default browser
   - Audio recordings and transcripts are stored in `transcripts`