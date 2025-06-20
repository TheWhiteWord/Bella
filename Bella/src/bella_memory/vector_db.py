"""
vector_db.py
Async ChromaDB interface for semantic memory indexing and search.
"""

from typing import List, Dict, Any


import chromadb
from chromadb.utils import embedding_functions
import os
import uuid
import asyncio

class VectorDB:
    def __init__(self, db_path: str = None, collection_name: str = "bella_memories"):
        self.db_path = db_path or os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../Bella/memories/chroma_db/"))
        os.makedirs(self.db_path, exist_ok=True)
        self.collection_name = collection_name
        self.client = chromadb.PersistentClient(path=self.db_path)
        self.collection = self.client.get_or_create_collection(collection_name)

    async def add_memory(self, embedding: list[float], metadata: dict) -> str:
        """Add a memory embedding and metadata to the vector database (async wrapper)."""
        return await asyncio.to_thread(self._add, embedding, metadata)

    def _add(self, embedding, metadata):
        uid = metadata.get("file_path") or str(uuid.uuid4())
        self.collection.add(
            embeddings=[embedding],
            metadatas=[metadata],
            ids=[uid],
        )
        return uid

    async def query(self, embedding: list[float], top_k: int = 5) -> List[Dict[str, Any]]:
        """Query the vector database for top-k similar memories (async wrapper)."""
        return await asyncio.to_thread(self._query, embedding, top_k)

    def _query(self, embedding, top_k):
        results = self.collection.query(query_embeddings=[embedding], n_results=top_k)
        matches = []
        for i in range(len(results["ids"][0])):
            match = {k: v[0][i] for k, v in results.items() if isinstance(v, list) and v and isinstance(v[0], list)}
            # Add metadata fields
            if "metadatas" in results and results["metadatas"][0]:
                match.update(results["metadatas"][0][i])
            # Add score if present
            if "distances" in results and results["distances"][0]:
                match["score"] = 1.0 - results["distances"][0][i]  # Convert distance to similarity
            matches.append(match)
        return matches
