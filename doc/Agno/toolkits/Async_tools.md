Tools
Async Tools
Agno Agents can execute multiple tools concurrently, allowing you to process function calls that the model makes efficiently. This is especially valuable when the functions involve time-consuming operations. It improves responsiveness and reduces overall execution time.

Here is an example:

async_tools.py

Copy
import asyncio
import time

from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.utils.log import logger

async def atask1(delay: int):
    """Simulate a task that takes a random amount of time to complete
    Args:
        delay (int): The amount of time to delay the task
    """
    logger.info("Task 1 has started")
    for _ in range(delay):
        await asyncio.sleep(1)
        logger.info("Task 1 has slept for 1s")
    logger.info("Task 1 has completed")
    return f"Task 1 completed in {delay:.2f}s"


async def atask2(delay: int):
    """Simulate a task that takes a random amount of time to complete
    Args:
        delay (int): The amount of time to delay the task
    """
    logger.info("Task 2 has started")
    for _ in range(delay):
        await asyncio.sleep(1)
        logger.info("Task 2 has slept for 1s")
    logger.info("Task 2 has completed")
    return f"Task 2 completed in {delay:.2f}s"


async def atask3(delay: int):
    """Simulate a task that takes a random amount of time to complete
    Args:
        delay (int): The amount of time to delay the task
    """
    logger.info("Task 3 has started")
    for _ in range(delay):
        await asyncio.sleep(1)
        logger.info("Task 3 has slept for 1s")
    logger.info("Task 3 has completed")
    return f"Task 3 completed in {delay:.2f}s"


async_agent = Agent(
    model=OpenAIChat(id="gpt-4o-mini"),
    tools=[atask2, atask1, atask3],
    show_tool_calls=True,
    markdown=True,
)

asyncio.run(
    async_agent.aprint_response("Please run all tasks with a delay of 3s", stream=True)
)
Run the Agent:


Copy
pip install -U agno openai

export OPENAI_API_KEY=***

python async_tools.py
How to use:

Provide your Agent with a list of tools, preferably asynchronous for optimal performance. However, synchronous functions can also be used since they will execute concurrently on separate threads.
Run the Agent using either the arun or aprint_response method, enabling concurrent execution of tool calls.
Concurrent execution of tools requires a model that supports parallel function calling. For example, OpenAI models have a parallel_tool_calls parameter (enabled by default) that allows multiple tool calls to be requested and executed simultaneously.