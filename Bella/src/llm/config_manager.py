"""Module for managing LLM model configurations.

This module handles loading and managing model configurations from YAML files,
providing a clean interface for model settings and parameters.
"""

import os
import yaml
from typing import Dict, Any

class ModelConfig:
    def __init__(self, config_path: str = None):
        """Initialize model configuration manager.
        
        Args:
            config_path (str, optional): Path to models.yaml config file.
                If None, uses default path in config directory.
        """
        if config_path is None:
            config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config", "models.yaml")
        
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
    def get_model_config(self, model_nickname: str) -> Dict[str, Any]:
        """Get model configuration by nickname.
        
        Args:
            model_nickname (str): Nickname of the model from config
            
        Returns:
            Dict[str, Any]: Model configuration including name and parameters
        """
        if model_nickname not in self.config['models']:
            model_nickname = self.config['default_model']
        return self.config['models'][model_nickname]
        
    def list_models(self) -> Dict[str, str]:
        """Return dict of model nicknames and descriptions.
        
        Returns:
            Dict[str, str]: Mapping of model nicknames to descriptions
        """
        return {
            nickname: info['description'] 
            for nickname, info in self.config['models'].items()
        }
        
    def get_default_model(self) -> str:
        """Get the default model nickname.
        
        Returns:
            str: Default model nickname from config
        """
        return self.config['default_model']