Ollama MCP Integration
Guide for integrating Ollama models with PraisonAI agents using MCP

​
Add Ollama Tool to AI Agent
In
AI Agent
Airbnb MCP
Out
​
Quick Start

1 Create a file

Create a new file ollama_airbnb.py with the following code:


from praisonaiagents import Agent, MCP

search_agent = Agent(
    instructions="""You help book apartments on Airbnb.""",
    llm="ollama/llama3.2",
    tools=MCP("npx -y @openbnb/mcp-server-airbnb --ignore-robots-txt")
)

search_agent.start("MUST USE airbnb_search Tool to Search. Search for Apartments in Paris for 2 nights. 04/28 - 04/30 for 2 adults. All Your Preference")
2
Install Dependencies

Make sure you have Node.js installed, as the MCP server requires it:


pip install "praisonaiagents[llm]"
4
Run the Agent

Execute your script:


python ollama_airbnb.py
Requirements

Python 3.10 or higher
Node.js installed on your system
Ollama installed and running locally