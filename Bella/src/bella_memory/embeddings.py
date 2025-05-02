"""
embeddings.py
Async embedding model interface using transformers.
"""

from typing import List



import aiohttp
import os
import asyncio
from typing import List

class EmbeddingModel:
    def __init__(self, ollama_url: str = None, model_name: str = "nomic-embed-text:latest"):
        self.ollama_url = ollama_url or os.environ.get("OLLAMA_HOST", "http://localhost:11434")
        self.model_name = model_name

    async def generate_embedding(self, text: str) -> List[float]:
        """
        Generate an embedding vector for the input text using Ollama's nomic-embed-text model.
        """
        url = f"{self.ollama_url}/api/embeddings"
        payload = {"model": self.model_name, "prompt": text}
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload, timeout=30) as resp:
                resp.raise_for_status()
                data = await resp.json()
                return data["embedding"]
