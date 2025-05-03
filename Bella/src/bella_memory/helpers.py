class SelfUserRelationExtractor:
    def __init__(self, model_size: str = None, thinking_mode: bool = False):
        self.model_size = model_size
        self.thinking_mode = thinking_mode

    async def extract(self, text: str, perspective: str = "self", model: str = None, qwen_size: str = None) -> list[dict]:
        """
        Extract (type, value) pairs from a self or user memory, and assign subject automatically.
        perspective: "self" (Bella) or "user" (David)
        Returns a list of dicts: {"subject": ..., "type": ..., "value": ...}
        Parsing is robust: accepts |, :, or - as delimiters, and lines with 1 or 2 fields (type defaults to 'think').
        Subject is always 'I' for self, 'David' for user.
        """
        from llm.config_manager import ModelConfig
        if qwen_size is None:
            qwen_size = "S"
        if model is None:
            model = ModelConfig().get_model_config(qwen_size).get("name", "Lexi:latest")
        subject = "Bella" if perspective == "self" else "David"
        if perspective == "self":
            prompt = (
                "You are Bella. From the following text, extract all of Bella's thoughts and feelings as structured pairs.\n"
                "For each, return: type (thinks_that/feels/wants/cares_about), value (the thought/feeling as a short phrase).\n"
                "Format: type | value. One per line.\n"
                "Examples:\n"
                "- thinks_that | I am always learning.\n"
                "- feels | proud\n"
                "- wants | to help David\n"
                "- cares_about | clarity\n\n"
                f"{text}"
            )
        else:
            prompt = (
                "You are Bella. From the following text, extract all of David's thoughts and feelings as structured pairs.\n"
                "Focus on what David seems to think, feel, want, or care about, as inferred from the text.\n"
                "For each, return: type (thinks_that/feels/wants/cares_about), value (the thought/feeling as a short phrase).\n"
                "Format: type | value. One per line.\n"
                "Examples:\n"
                "- thinks_that | AI will change the world.\n"
                "- feels | optimistic\n"
                "- wants | Bella to succeed\n"
                "- cares_about | Bella's confidence\n\n"
                f"{text}"
            )
        result = await generate(prompt, qwen_size=qwen_size, thinking_mode=self.thinking_mode, model=model)
        triples = []
        # Accept |, :, or - as delimiters, and tolerate extra/missing whitespace
        for line in result.splitlines():
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            # Try all delimiters
            for delim in ['|', ':', '-']:
                if delim in line:
                    parts = [p.strip() for p in line.split(delim)]
                    break
            else:
                parts = [line]
            # Accept 2 fields: type, value
            if len(parts) == 2:
                triples.append({"subject": subject, "type": parts[0], "value": parts[1]})
            # Accept 1 field: value (type defaults to 'think')
            elif len(parts) == 1 and parts[0]:
                triples.append({"subject": subject, "type": "think", "value": parts[0]})
            # Ignore lines with <1 field
        return triples

class MemoryClassifier:
    def __init__(self, model_size: str = None, thinking_mode: bool = False):
        self.model_size = model_size
        self.thinking_mode = thinking_mode
    async def classify(self, text: str, model: str = None, qwen_size: str = None) -> list[str]:
        """
        Classify the memory into all relevant types (multi-label) using Qwen LLM.
        Returns a list of labels, e.g., ["relationship", "self", "project"].
        """
        from llm.config_manager import ModelConfig
        if qwen_size is None:
            qwen_size = "XS"
        if model is None:
            model = ModelConfig().get_model_config(qwen_size).get("name", "Lexi:latest")
        prompt = (
            "You are Bella's memory system. Classify the following memory into ALL relevant types. "
            "Possible types: relationship, self, project, user, fact, preference, general. "
            "Return a comma-separated list of all applicable types, ordered by relevance.\n\n"
            f"Memory: {text}"
        )
        result = await generate(prompt, qwen_size=qwen_size, thinking_mode=self.thinking_mode, model=model)
        # Parse comma-separated labels
        return [label.strip().lower() for label in result.split(",") if label.strip()]
"""
helpers.py
Async helpers for summarization, topic extraction, and importance scoring.
"""

from typing import List



# Qwen-based helpers for summarization, topic extraction, and importance scoring
import re
from llm.ollama_client import generate

class Summarizer:
    async def summarize_self_insight(self, text: str, model: str = None, qwen_size: str = None) -> str:
        """
        Extract Bella's self-insight or personal reflection from the input text.
        """
        from llm.config_manager import ModelConfig
        if qwen_size is None:
            qwen_size = "XXS"
        if model is None:
            model = ModelConfig().get_model_config(qwen_size).get("name", "Lexi:latest")
        prompt = (
            "You are Bella. From the following conversation, extract a single concise insight, reflection, or feeling about yourself (Bella).\n"
            "Focus on insights on what you Bella learned, felt, or realized about herself. You are creating memories of your experiences.\n"
            "Let your sassy, witty and independent spirit shine through in your observations.\n"
            "Be authentic and brief (2-3 sentences). Speak as Bella, in third person\n\n"
            f"{text}"
        )
        return await generate(prompt, qwen_size=qwen_size, thinking_mode=self.thinking_mode, model=model)

    async def summarize_user_observation(self, text: str, model: str = None, qwen_size: str = None) -> str:
        """
        Extract an observation about David (the user) from the input text.
        """
        from llm.config_manager import ModelConfig
        if qwen_size is None:
            qwen_size = "XXS"
        if model is None:
            model = ModelConfig().get_model_config(qwen_size).get("name", "Lexi:latest")
        prompt = (
            "You are Bella. From the following conversation, extract a single concise insight, reflection, or feeling about yourself (Bella).\n"
            "Focus on insights on what David seems to think, feel, or care about, as observed by Bella. You are creating memories of your observations of David.\n"
            "Let your sassy, witty and independent spirit shine through in your observations.\n"
            "Be authentic and brief (2-3 sentences). Speak as Bella, in third person\n\n"
            f"{text}"
        )
        return await generate(prompt, qwen_size=qwen_size, thinking_mode=self.thinking_mode, model=model)
    def __init__(self, model_size: str = None, thinking_mode: bool = False):
        self.model_size = model_size
        self.thinking_mode = thinking_mode

    async def summarize(self, text: str, memory_type: list[str] = None, model: str = None, qwen_size: str = None) -> str:
        """
        Generate a concise summary of the input text using Qwen LLM, speaking in Bella's first person voice.
        For main memory (not self/user), focus on relationship, facts, and events.
        Optionally tailor the summary based on memory_type labels.
        """
        from llm.config_manager import ModelConfig
        if qwen_size is None:
            qwen_size = "XXS"
        if model is None:
            model = ModelConfig().get_model_config(qwen_size).get("name", "Lexi:latest")
        type_hint = ""
        if memory_type:
            type_hint = f"Memory type(s): {', '.join(memory_type)}. "
        prompt = (
            "You are Bella. Summarize the following memory in 1-2 sentences, speaking of both parties in third person. "
            "Focus exclusively on the relationship, shared facts, and events between Bella and David. "
            "Do not include any personal insights or reflections. "
            "If relevant, mention what happened, what was discussed, and the emotional context, but keep it factual and event-focused. "
            f"{type_hint}"
            f"\n\n{text}"
        )
        return await generate(prompt, qwen_size=qwen_size, thinking_mode=self.thinking_mode, model=model)


import re
import numpy as np


class TopicExtractor:
    def __init__(self, model_size: str = None, thinking_mode: bool = False):
        self.model_size = model_size
        self.thinking_mode = thinking_mode

    async def extract(self, text: str, model: str = None, qwen_size: str = None) -> list[str]:
        """Extract key topics from the input text using Qwen LLM."""
        from llm.config_manager import ModelConfig
        if qwen_size is None:
            qwen_size = "XS"
        if model is None:
            model = ModelConfig().get_model_config(qwen_size).get("name", "Lexi:latest")
        prompt = (
            "List the main topics discussed in the following text as a comma-separated list, focusing on relationship and personal context:\n\n"
            f"{text}"
        )
        result = await generate(prompt, qwen_size=qwen_size, thinking_mode=self.thinking_mode, model=model)
        return [t.strip() for t in result.split(",") if t.strip()]



class ImportanceScorer:
    def __init__(self, model_size: str = None, thinking_mode: bool = False):
        self.model_size = model_size
        self.thinking_mode = thinking_mode

    async def score(self, text: str, model: str = None, qwen_size: str = None) -> float:
        """
        Score the importance of the input text for Bella's memory system using Qwen LLM.
        The score should reflect not only relationship relevance, but also mutual interest, unique perspectives, and points of view expressed by both parties.
        """
        from llm.config_manager import ModelConfig
        if qwen_size is None:
            qwen_size = "XS"
        if model is None:
            model = ModelConfig().get_model_config(qwen_size).get("name", "Lexi:latest")
        prompt = (
            "On a scale from 0 to 1, how important is this memory for Bella and David's sense of self, relationship, mutual interest, and the sharing of unique points of view? "
            "Consider not just emotional connection, but also whether the conversation is interesting, thought-provoking, or reveals something meaningful about either person's perspective. "
            "Be conservative in your scoring. Only score above 0.7 if the memory is very important to both parties. "
            "Respond with a single number between 0 and 1.\n\n"
            f"{text}"
        )
        result = await generate(prompt, qwen_size=qwen_size, thinking_mode=self.thinking_mode, model=model)
        try:
            # Extract the first float between 0 and 1
            match = re.search(r"([01](?:\.\d+)?)", result)
            return float(match.group(1)) if match else 0.5
        except Exception:
            return 0.5
