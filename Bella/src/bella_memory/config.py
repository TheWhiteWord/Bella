"""
config.py
Centralized config loader for Bella's memory system (YAML-based).
"""

from typing import Any

class ConfigLoader:
    def load(self, config_path: str) -> Any:
        """Load configuration from a YAML file."""
        ...
