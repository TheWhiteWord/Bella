"""
embeddings.py
Async embedding model interface using transformers.
"""

from typing import List



import os
import aiohttp
import asyncio
from typing import List
from openai import AsyncOpenAI

class EmbeddingModel:
    def __init__(self, ollama_url: str = None, model_name: str = None):
        # We now point to the Llama.cpp server
        self.base_url = "http://localhost:8080/v1"
        if model_name is None:
            from llm.config_manager import ModelConfig
            # Resolve 'EMBEDDING' alias from models.yaml
            self.model_name = ModelConfig().resolve_model_name("EMBEDDING")
        else:
            self.model_name = model_name
        self.client = AsyncOpenAI(base_url=self.base_url, api_key="none")

    async def generate_embedding(self, text: str) -> List[float]:
        """
        Generate an embedding vector for the input text using Llama.cpp server.
        """
        try:
            response = await self.client.embeddings.create(
                input=text,
                model=self.model_name
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"Error generating embedding: {e}")
            return []

