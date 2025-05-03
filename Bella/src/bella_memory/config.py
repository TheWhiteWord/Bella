"""
config.py
Centralized config loader for Bella's memory system (YAML-based).
"""

from typing import Any

import os
import yaml

class ConfigLoader:
    """
    Centralized config loader for Bella's memory system (YAML-based).
    Stores paths to prompt YAML files as class variables.
    """
    # Define paths to prompt YAML files (update as needed)
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    PROMPTS_DIR = os.path.join(BASE_DIR, "prompts")
    SUMMARIES_PROMPTS_PATH = os.path.join(PROMPTS_DIR, "summaries.yaml")
    HELPERS_PROMPTS_PATH = os.path.join(PROMPTS_DIR, "helpers.yaml")

    def load(self, config_path: str) -> Any:
        """Load configuration from a YAML file."""
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
