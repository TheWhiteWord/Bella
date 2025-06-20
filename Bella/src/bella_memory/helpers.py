"""helpers.py
Async helpers for summarization, topic extraction, and importance scoring.
"""
import re
import numpy as np
from typing import List
import re
from llm.ollama_client import generate


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
        from bella_memory.config import ConfigLoader
        if qwen_size is None:
            qwen_size = "XXS"
        if model is None:
            model = ModelConfig().get_model_config(qwen_size).get("name", "Lexi:latest")
        config = ConfigLoader().load(ConfigLoader.HELPERS_PROMPTS_PATH)
        system_prompt = config.get("memory_classifier_system", "")
        prompt = config["memory_classifier"].format(text=text)
        result = await generate(
            prompt,
            model=model,
            system_prompt=system_prompt,
            qwen_size=qwen_size,
            thinking_mode=self.thinking_mode
        )
        # Parse comma-separated labels
        return [label.strip().lower() for label in result.split(",") if label.strip()]


class Summarizer:
    """
    Async helper for summarization tasks using Qwen LLM.
    Allows per-instance control of thinking_mode for prompt generation.
    """

    def __init__(self, model_size: str = None, thinking_mode: bool = False):
        """
        Initialize the Summarizer.

        Args:
            model_size (str, optional): Model size identifier for LLM selection.
            thinking_mode (bool, optional): Whether to use thinking mode for LLM prompts.
        """
        self.model_size = model_size
        self.thinking_mode = thinking_mode

    async def summarize_self_insight(self, text: str, model: str = None, qwen_size: str = None) -> str:
        """
        Extract Bella's self-insight or personal reflection from the input text.

        Args:
            text (str): Input conversation or memory text.
            model (str, optional): LLM model name.
            qwen_size (str, optional): Qwen model size.

        Returns:
            str: Concise self-insight or reflection as Bella.
        """
        from llm.config_manager import ModelConfig
        from bella_memory.config import ConfigLoader
        if qwen_size is None:
            qwen_size = "XXS"
        if model is None:
            model = ModelConfig().get_model_config(qwen_size).get("name", "Lexi:latest")
        config = ConfigLoader().load(ConfigLoader.SUMMARIES_PROMPTS_PATH)
        system_prompt = config.get("summarize_self_insight_system", "")
        prompt = config["summarize_self_insight"].format(text=text)
        return await generate(
            prompt,
            model=model,
            system_prompt=system_prompt,
            qwen_size=qwen_size,
            thinking_mode=self.thinking_mode
        )

    async def summarize_user_observation(self, text: str, model: str = None, qwen_size: str = None) -> str:
        """
        Extract an observation about David (the user) from the input text.

        Args:
            text (str): Input conversation or memory text.
            model (str, optional): LLM model name.
            qwen_size (str, optional): Qwen model size.

        Returns:
            str: Concise observation about David as seen by Bella.
        """
        from llm.config_manager import ModelConfig
        from bella_memory.config import ConfigLoader
        if qwen_size is None:
            qwen_size = "XXS"
        if model is None:
            model = ModelConfig().get_model_config(qwen_size).get("name", "Lexi:latest")
        config = ConfigLoader().load(ConfigLoader.SUMMARIES_PROMPTS_PATH)
        system_prompt = config.get("summarize_user_observation_system", "")
        prompt = config["summarize_user_observation"].format(text=text)
        return await generate(
            prompt,
            model=model,
            system_prompt=system_prompt,
            qwen_size=qwen_size,
            thinking_mode=self.thinking_mode
        )

    async def summarize(self, text: str, memory_type: list[str] = None, model: str = None, qwen_size: str = None) -> str:
        """
        Generate a concise, factual summary of the input text using Qwen LLM, focusing on relationship, facts, and events between Bella and David.
        Optionally tailor the summary based on memory_type labels.

        Args:
            text (str): Input memory text.
            memory_type (list[str], optional): Memory type labels for context.
            model (str, optional): LLM model name.
            qwen_size (str, optional): Qwen model size.

        Returns:
            str: Concise summary of the memory.
        """
        from llm.config_manager import ModelConfig
        from bella_memory.config import ConfigLoader
        if qwen_size is None:
            qwen_size = "XXS"
        if model is None:
            model = ModelConfig().get_model_config(qwen_size).get("name", "Lexi:latest")
        config = ConfigLoader().load(ConfigLoader.SUMMARIES_PROMPTS_PATH)
        system_prompt = config.get("summarize_system", "")
        type_hint = f"{', '.join(memory_type)}. " if memory_type else ""
        prompt = config["summarize"].format(type_hint=type_hint, text=text)
        return await generate(
            prompt,
            model=model,
            system_prompt=system_prompt,
            qwen_size=qwen_size,
            thinking_mode=self.thinking_mode
        )

class TopicExtractor:
    def __init__(self, model_size: str = None, thinking_mode: bool = False):
        self.model_size = model_size
        self.thinking_mode = thinking_mode

    async def extract(self, text: str, model: str = None, qwen_size: str = None) -> list[str]:
        """Extract key topics from the input text using Qwen LLM."""
        from llm.config_manager import ModelConfig
        from bella_memory.config import ConfigLoader
        if qwen_size is None:
            qwen_size = "XXS"
        if model is None:
            model = ModelConfig().get_model_config(qwen_size).get("name", "Lexi:latest")
        config = ConfigLoader().load(ConfigLoader.HELPERS_PROMPTS_PATH)
        system_prompt = config.get("topic_extractor_system", "")
        prompt = config["topic_extractor"].format(text=text)
        result = await generate(
            prompt,
            model=model,
            system_prompt=system_prompt,
            qwen_size=qwen_size,
            thinking_mode=self.thinking_mode
        )
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
        from bella_memory.config import ConfigLoader
        if qwen_size is None:
            qwen_size = "XXS"
        if model is None:
            model = ModelConfig().get_model_config(qwen_size).get("name", "Lexi:latest")
        config = ConfigLoader().load(ConfigLoader.HELPERS_PROMPTS_PATH)
        system_prompt = config.get("importance_scorer_system", "")
        prompt = config["importance_scorer"].format(text=text)
        result = await generate(
            prompt,
            model=model,
            system_prompt=system_prompt,
            qwen_size=qwen_size,
            thinking_mode=self.thinking_mode
        )
        try:
            # Extract the first float between 0 and 1
            match = re.search(r"([01](?:\.\d+)?)", result)
            return float(match.group(1)) if match else 0.5
        except Exception:
            return 0.5

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
        from bella_memory.config import ConfigLoader
        if qwen_size is None:
            qwen_size = "XXS"
        if model is None:
            model = ModelConfig().get_model_config(qwen_size).get("name", "Lexi:latest")
        config = ConfigLoader().load(ConfigLoader.HELPERS_PROMPTS_PATH)
        subject = "Bella" if perspective == "self" else "David"
        if perspective == "self":
            system_prompt = config.get("self_relation_extractor_system", "")
            prompt = config["self_relation_extractor"].format(text=text)
        else:
            system_prompt = config.get("user_relation_extractor_system", "")
            prompt = config["user_relation_extractor"].format(text=text)
        result = await generate(
            prompt,
            model=model,
            system_prompt=system_prompt,
            qwen_size=qwen_size,
            thinking_mode=self.thinking_mode
        )
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