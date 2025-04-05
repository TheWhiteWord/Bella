Getting Started
Quick Start
Create AI Agents and make them work for you in just a few lines of code.

​
Basic
Code
No Code
JavaScript
TypeScript
1
Install Package

Install the PraisonAI Agents package:


pip install praisonaiagents
2
Set API Key


export OPENAI_API_KEY=your_openai_key
Generate your OpenAI API key from OpenAI. Use other LLM providers like Ollama, Anthropic, Groq, Google, etc. Please refer to the Models for more information.

3
Create Agents

Create app.py:


Single Agent
```python
from praisonaiagents import Agent, PraisonAIAgents

# Create a simple agent
summarise_agent = Agent(instructions="Summarise Photosynthesis")

# Run the agent
agents = PraisonAIAgents(agents=[summarise_agent])
agents.start()
```


Multiple Agents

```python

from praisonaiagents import Agent, PraisonAIAgents

# Create agents with specific roles
diet_agent = Agent(
    instructions="Give me 5 healthy food recipes",
)

blog_agent = Agent(
    instructions="Write a blog post about the food recipes",
)

# Run multiple agents
agents = PraisonAIAgents(agents=[diet_agent, blog_agent])
agents.start()
```
4
Run Agents

Execute your script:


python app.py
