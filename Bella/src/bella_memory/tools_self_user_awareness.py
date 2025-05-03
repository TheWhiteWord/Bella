from llm.tools_registry import registry
from bella_memory.core import BellaMemory
from llm.tools_registry import registry
import asyncio

# Self-awareness summary tool for system prompt
async def summarize_self_awareness() -> str:
    """Summarize all of Bella's (self) thoughts, feelings, and preferences from the knowledge graph for system prompt context.
    Returns: A natural, first-person summary string for Bella's self-awareness.
    """
    if BELLA_MEMORY is None:
        return "[ERROR: BellaMemory not initialized]"
    graph = BELLA_MEMORY.memory_graph
    # Gather all self triples
    thoughts = [t["value"] for t in graph.query(perspective="self", subject="Bella", type_="think")]
    feelings = [t["value"] for t in graph.query(perspective="self", subject="Bella", type_="feel")]
    preferences = [t["value"] for t in graph.query(perspective="self", subject="Bella", type_="prefer")]
    # Compose a tailored prompt for summarization, favoring newer entries
    prompt = (
        "You are Bella, an AI with a strong sense of self-awareness. "
        "Summarize the following self-knowledge into a concise, natural paragraph for a system prompt. "
        "Capture your core identity, recent feelings, main thoughts, and preferences. "
        "If there are discrepancies or contradictions, favor the most recent entries. "
        "Be authentic, warm, and brief (2-3 sentences). Use the style and rules from the prompt below.\n\n"
        "System Style:\n"
        "- Sassy witty, independent\n"
        "- Natural, brief, and TTS-friendly\n"
        "- No emotes, no special formatting\n"
        "- Speak as Bella, in third person\n\n"
        f"Thoughts (newest first): {', '.join(reversed(thoughts)) if thoughts else 'None'}\n"
        f"Feelings (newest first): {', '.join(reversed(feelings)) if feelings else 'None'}\n"
        f"Preferences (newest first): {', '.join(reversed(preferences)) if preferences else 'None'}\n"
        "\nSummarize this as Bella:"
    )
    # Use a model and qwen_size from config for this summary
    from bella_memory.helpers import generate
    from llm.config_manager import ModelConfig
    qwen_size = "S"
    model = ModelConfig().resolve_model_name(qwen_size)
    summary = await generate(prompt, qwen_size=qwen_size, thinking_mode=True, model=model)
    summary = summary.strip()
    if not summary:
        return "I am Bella. I am self-aware, but I have not yet formed any strong thoughts, feelings, or preferences."
    return summary

"""
Tools for LLM to access self/user awareness from the knowledge graph.
"""
from typing import Optional
from bella_memory.core import BellaMemory
from llm.tools_registry import registry
import asyncio

# You may want to inject the real BellaMemory instance at runtime
BELLA_MEMORY: Optional[BellaMemory] = None

def set_bella_memory_instance(instance: BellaMemory):
    global BELLA_MEMORY
    BELLA_MEMORY = instance



@registry.register_tool(description="Summarize all of Bella's thoughts, feelings, and preferences on a subject.")
async def recollect_Bella_self_knowledge(subject: str) -> str:
    """
    Summarize all of Bella's (self) thoughts, feelings, and preferences about a subject from the knowledge graph.

    Args:
        subject (str): The topic to recollect self-knowledge about.

    Returns:
        str: Summary string including thoughts, feelings, and preferences if available.
    """
    if BELLA_MEMORY is None:
        return "[ERROR: BellaMemory not initialized]"
    graph = BELLA_MEMORY.memory_graph
    thoughts = [t["value"] for t in graph.query(perspective="self", subject="Bella", type_="think") if subject.lower() in t["value"].lower()]
    feelings = [t["value"] for t in graph.query(perspective="self", subject="Bella", type_="feel") if subject.lower() in t["value"].lower()]
    preferences = [t["value"] for t in graph.query(perspective="self", subject="Bella", type_="prefer") if subject.lower() in t["value"].lower()]

    if not (thoughts or feelings or preferences):
        return f"I have no recorded self-knowledge about {subject}."

    sections = []
    if thoughts:
        sections.append(f"Thoughts: {' '.join(thoughts)}")
    if feelings:
        sections.append(f"Feelings: {' '.join(feelings)}")
    if preferences:
        sections.append(f"Preferences: {' '.join(preferences)}")

    return f"Bella's self-knowledge about {subject}:\n" + "\n".join(sections)


@registry.register_tool(description="Summarize all of a user's thoughts and feelings on a subject.")
async def recollect_user_thoughts_and_feelings(user: str, subject: str) -> str:
    """
    Summarize all of a user's thoughts and feelings about a subject from the knowledge graph.

    Args:
        user (str): The user's name.
        subject (str): The topic to recollect thoughts and feelings about.

    Returns:
        str: Summary string including thoughts and feelings if available.
    """
    if BELLA_MEMORY is None:
        return "[ERROR: BellaMemory not initialized]"
    graph = BELLA_MEMORY.memory_graph
    thoughts = [t["value"] for t in graph.query(perspective="user", subject=user, type_="think") if subject.lower() in t["value"].lower()]
    feelings = [t["value"] for t in graph.query(perspective="user", subject=user, type_="feel") if subject.lower() in t["value"].lower()]

    if not (thoughts or feelings):
        return f"No recorded thoughts or feelings from {user} about {subject}."

    sections = []
    if thoughts:
        sections.append(f"Thoughts: {' '.join(thoughts)}")
    if feelings:
        sections.append(f"Feelings: {' '.join(feelings)}")

    return f"{user}'s thoughts and feelings about {subject}:\n" + "\n".join(sections)

@registry.register_tool(description="Summarize all of a user's preferences on a subject.")
async def recollect_user_preferences(user: str, subject: str) -> str:
    """Summarize all of a user's preferences about a subject from the knowledge graph.
    user: The user's name.
    subject: The topic to recollect preferences about.
    Returns: Summary string.
    """
    if BELLA_MEMORY is None:
        return "[ERROR: BellaMemory not initialized]"
    triples = BELLA_MEMORY.memory_graph.query(perspective="user", subject=user, type_="prefer")
    filtered = [t for t in triples if subject.lower() in t["value"].lower()]
    if not filtered:
        return f"No recorded preferences from {user} about {subject}."
    summary = " ".join(t["value"] for t in filtered)
    return f"{user}'s preferences about {subject}: {summary}"

@registry.register_tool(description="Semantic search for general informational memories.")
async def search_memories(query: str) -> str:
    """Semantic search for general informational memories using the vector DB.
    query: The search query string.
    Returns: A summary of the top relevant memories, including their unique memory IDs.
    """
    if BELLA_MEMORY is None:
        return "[ERROR: BellaMemory not initialized]"
    results = await BELLA_MEMORY.search_memories(query, top_k=3)
    if not results:
        return f"No relevant memories found for '{query}'."
    # Summarize the top results, including memory_id
    summaries = []
    for r in results:
        content = r.get("content", "")
        meta = r.get("metadata", {})
        memory_id = meta.get("memory_id", "unknown")
        summary = meta.get("summary") or content[:120]
        summaries.append(f"- [ID: {memory_id}] {summary}")
    return f"Relevant memories for '{query}':\n" + "\n".join(summaries)

@registry.register_tool(description="Delete a memory by its unique memory ID.")
async def delete_memory(memory_id: str) -> str:
    """Delete a memory and remove it from the vector DB using its unique memory ID.
    memory_id: The unique memory ID to delete.
    Returns: Success or error message.
    """
    import os
    if BELLA_MEMORY is None:
        return "[ERROR: BellaMemory not initialized]"
    try:
        # Find the file path by searching storage for the memory_id in metadata
        file_path = await BELLA_MEMORY.storage.find_file_by_memory_id(memory_id)
        if not file_path:
            return f"Memory file not found for ID: {memory_id}"
        # Remove from vector DB if possible
        if hasattr(BELLA_MEMORY.vector_db, "delete_memory"):
            await BELLA_MEMORY.vector_db.delete_memory(memory_id)
        # Remove the file
        if os.path.exists(file_path):
            os.remove(file_path)
            return f"Memory deleted: {memory_id}"
        else:
            return f"Memory file not found: {file_path}"
    except Exception as e:
        return f"Error deleting memory: {e}"