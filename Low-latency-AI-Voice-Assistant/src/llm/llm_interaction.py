from .llm_response import generate, list_available_models, list_available_models_from_ollama
import asyncio

async def generate_llm_response(user_input: str, history_context: str, model: str = "Gemma3", timeout: float = 15.0) -> str:
    """Generate a response using local Ollama model.
    
    Args:
        user_input (str): The user's input text
        history_context (str): Previous conversation history
        model (str): Model nickname from config (e.g., "Gemma3", "hermes8b")
        timeout (float): Maximum time to wait for response in seconds
        
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
        return "I apologize, but I'm having trouble generating a response. Please make sure Ollama is running."
    
    return response

async def get_available_models():
    """Get list of available local models and their descriptions, including runtime check"""
    config_models = list_available_models()
    ollama_models = await list_available_models_from_ollama()
    
    # Add runtime availability status
    models_status = {}
    for nickname, desc in config_models.items():
        # Check if either the nickname or the actual model name exists in ollama_models
        model_exists = any(m.lower().startswith(nickname.lower()) for m in ollama_models)
        models_status[nickname] = {
            'description': desc,
            'available': model_exists
        }
    
    return models_status