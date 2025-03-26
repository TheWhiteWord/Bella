import os
import yaml
import ollama
import asyncio
from typing import Optional, Dict, Any
import time
from pathlib import Path
from dotenv import load_dotenv
import re

load_dotenv()

class ModelConfig:
    def __init__(self, config_path: str = None):
        if config_path is None:
            config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config", "models.yaml")
        
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
    def get_model_config(self, model_nickname: str) -> Dict[str, Any]:
        """Get model configuration by nickname"""
        if model_nickname not in self.config['models']:
            model_nickname = self.config['default_model']
        return self.config['models'][model_nickname]
        
    def list_models(self) -> Dict[str, str]:
        """Return dict of model nicknames and descriptions"""
        return {
            nickname: info['description'] 
            for nickname, info in self.config['models'].items()
        }
        
    def get_default_model(self) -> str:
        """Get the default model nickname"""
        return self.config['default_model']


def clean_response(text: str) -> str:
    """Remove emojis and emoticons from text.
    
    Args:
        text (str): Input text to clean
        
    Returns:
        str: Cleaned text without emojis/emoticons
    """
    # Remove unicode emojis
    emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        "]+", flags=re.UNICODE)
    
    # Remove ASCII-style emoticons
    ascii_pattern = re.compile(r'[:;=]-?[)(/\\|pPoO]')
    
    # Clean the text
    text = emoji_pattern.sub('', text)
    text = ascii_pattern.sub('', text)
    
    # Remove multiple spaces and trim
    text = ' '.join(text.split())
    
    return text.strip()


async def generate(
    prompt: str,
    model: str = None,
    system_prompt: str = "Keep your response short and concise.",
    verbose: bool = False,
    config_path: str = None,
    timeout: float = 5.0  # Add timeout parameter
) -> Optional[str]:
    """Generate a response using local Ollama model asynchronously.
    
    Args:
        prompt (str): The input text to generate a response for
        model (str): Model nickname from config (e.g., "hermes8b", "dolphin8b")
        system_prompt (str): System prompt for setting model behavior
        verbose (bool): Whether to print debug info
        config_path (str): Path to models.yaml config file
        timeout (float): Maximum time to wait for response in seconds
    
    Returns:
        str: Generated response, or None if there was an error
    """
    try:
        # Load model config
        model_config = ModelConfig(config_path)
        if not model:
            model = model_config.get_default_model()
            
        model_info = model_config.get_model_config(model)
        
        if verbose:
            print(f"\nGenerating response with Ollama ({model})")
            print(f"Model: {model_info['name']}")
            print(f"Parameters: {model_info['parameters']}")
            start = time.time()
        
        # Generate response using ollama library with timeout
        response_future = asyncio.create_task(
            asyncio.to_thread(
                ollama.chat,
                model=model_info['name'],
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                options=model_info['parameters']
            )
        )
        
        # Wait for response with timeout
        try:
            response = await asyncio.wait_for(response_future, timeout=timeout)
        except asyncio.TimeoutError:
            if verbose:
                print(f"LLM response timed out after {timeout}s")
            return None
        
        # Clean and process the response
        response_text = clean_response(response['message']['content'].strip())
        
        if verbose:
            print(f"Response time: {time.time() - start:.2f}s")
            print(f"Response: {response_text}")
            
        return response_text
        
    except Exception as e:
        print(f"Error generating response: {str(e)}")
        if "connection" in str(e).lower():
            print("Make sure Ollama is running (ollama serve)")
        return None

async def generate_llm_response(user_input: str, history_context: str, model: str = "Gemma3", timeout: float = 5.0) -> str:
    """Generate a response using local Ollama model.
    
    Args:
        user_input (str): The user's input text
        history_context (str): Previous conversation history
        model (str): Model nickname from config (e.g., "Gemma3", "hermes8b")
        timeout (float): Maximum time to wait for response
        
    Returns:
        str: Generated response from the model
    """
    system_prompt = """You are a helpful voice assistant. Be concise and natural in your responses.
    Keep responses under 40 words. Focus on being helpful while maintaining a conversational tone.
    Use complete sentences but be brief. No emotes or special formatting."""
    
    response = await generate(
        prompt=f"Given this conversation history:\n{history_context}\n\nRespond to: {user_input}",
        model=model,
        system_prompt=system_prompt,
        verbose=True,  # Enable verbose mode to debug
        timeout=timeout
    )
    
    if not response:
        return "I apologize, but I'm having trouble generating a response right now. Could you try again?"
    
    return response

async def list_available_models_from_ollama() -> list:
    """Get list of models directly from Ollama service"""
    try:
        response = ollama.list()
        if hasattr(response, 'models') and isinstance(response.models, list):
            return [model.model for model in response.models]
        return []
    except Exception as e:
        print(f"Error listing Ollama models: {str(e)}")
        return []

def list_available_models() -> Dict[str, str]:
    """Get list of available models and their descriptions from config"""
    try:
        config = ModelConfig()
        return config.list_models()
    except Exception as e:
        print(f"Error loading model config: {str(e)}")
        return {}


# Example usage
if __name__ == "__main__":
    async def main():
        print("\nAvailable models from config:")
        for nickname, description in list_available_models().items():
            print(f"- {nickname}: {description}")
        
        print("\nAvailable models from Ollama:")
        ollama_models = await list_available_models_from_ollama()
        for model in ollama_models:
            print(f"- {model}")
        
        prompt = "hey how are you?"
        start = time.time()
        
        # Test default model
        response = await generate(
            prompt,
            system_prompt="Be helpful and friendly",
            verbose=True
        )
        print(f"\nDefault model response time: {time.time() - start:.2f}s")
        print(f"Response: {response}\n")

    asyncio.run(main())