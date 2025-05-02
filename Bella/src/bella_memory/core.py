class SimpleMemoryGraph:
    """A minimal in-memory graph for self/user awareness triples."""
    def __init__(self):
        self.triples = []  # List of dicts: {subject, type, value, perspective, memory_id}

    def add_triples(self, triples, perspective, memory_id):
        for t in triples:
            t = dict(t)
            t["perspective"] = perspective
            t["memory_id"] = memory_id
            self.triples.append(t)

    def query(self, perspective=None, subject=None, type_=None):
        results = self.triples
        if perspective:
            results = [t for t in results if t["perspective"] == perspective]
        if subject:
            results = [t for t in results if t["subject"].lower() == subject.lower()]
        if type_:
            results = [t for t in results if t["type"].lower() == type_.lower()]
        return results
"""
core.py
Unified async memory manager for Bella's relationship-centric memory system.
"""

from typing import List, Dict, Any

from .helpers import Summarizer, TopicExtractor, ImportanceScorer, MemoryClassifier
from .storage import MemoryStorage
from .embeddings import EmbeddingModel
from .vector_db import VectorDB
from typing import List, Dict, Any
import asyncio
import datetime

class BellaMemory:
    """Unified async memory manager for Bella."""

    def __init__(self,
                 summarizer: Summarizer,
                 topic_extractor: TopicExtractor,
                 importance_scorer: ImportanceScorer,
                 storage: MemoryStorage,
                 embedding_model: EmbeddingModel,
                 vector_db: VectorDB,
                 memory_classifier: MemoryClassifier,
                 relation_extractor=None,
                 memory_graph=None):
        self.summarizer = summarizer
        self.topic_extractor = topic_extractor
        self.importance_scorer = importance_scorer
        self.memory_classifier = memory_classifier
        self.storage = storage
        self.embedding_model = embedding_model
        self.vector_db = vector_db
        from .helpers import SelfUserRelationExtractor
        self.relation_extractor = relation_extractor or SelfUserRelationExtractor()
        self.memory_graph = memory_graph or SimpleMemoryGraph()

    async def store_memory(self, content: str, user_context: dict) -> list[str]:
        """
        Store a new memory, classify its type, summarize, tag, and index it.
        For 'self' and 'user' labels, create special memories with focused summaries.
        Args:
            content (str): The text content of the memory.
            user_context (dict): Metadata about the interaction (e.g., participants, topics, emotional tone, project, etc.).
        Returns:
            list[str]: List of unique IDs or file paths of the stored memories.
        """
        # Run helpers in parallel (including multi-label classifier)
        topics_task = self.topic_extractor.extract(content)
        importance_task = self.importance_scorer.score(content)
        memory_type_task = self.memory_classifier.classify(content)
        topics, importance, memory_type = await asyncio.gather(
            topics_task, importance_task, memory_type_task
        )

        if importance < 0.5:
            return []  # Not important enough to store

        # Multi-label: allow user_context to override or add to classifier output
        user_labels = user_context.get("memory_type")
        if user_labels:
            if isinstance(user_labels, str):
                user_labels = [user_labels]
            memory_type = list({label.lower() for label in (memory_type + user_labels)})

        # Separate out self/user and main labels
        main_labels = [l for l in memory_type if l not in ("self", "user")]
        self_present = "self" in memory_type
        user_present = "user" in memory_type

        memory_ids = []

        # Store main memory (all non-self/user labels)
        if main_labels:
            import uuid
            memory_id = str(uuid.uuid4())
            summary = await self.summarizer.summarize(content, memory_type=main_labels)
            timestamp = datetime.datetime.utcnow().isoformat()
            metadata = {
                "memory_id": memory_id,
                "timestamp": timestamp,
                "participants": user_context.get("participants", []),
                "topics": topics,
                "emotional_tone": user_context.get("emotional_tone", None),
                "summary": summary,
                "source": "autonomous",
                "memory_type": main_labels,
                "importance": importance,
            }
            if user_context.get("project"):
                metadata["project"] = user_context["project"]
            metadata.update(user_context)
            file_path = await self.storage.save_memory(content, metadata)
            embedding = await self.embedding_model.generate_embedding(content)
            # Sanitize metadata for ChromaDB (no lists)
            chroma_metadata = {
                k: (
                    ", ".join(map(str, v)) if isinstance(v, list)
                    else (v if v is not None else "")
                )
                for k, v in {**metadata, "file_path": file_path}.items()
            }
            await self.vector_db.add_memory(embedding, chroma_metadata)
            memory_ids.append(memory_id)

        # Store self memory (if present)
        if self_present:
            import uuid
            memory_id = str(uuid.uuid4())
            self_summary = await self.summarizer.summarize_self_insight(content)
            timestamp = datetime.datetime.utcnow().isoformat()
            metadata = {
                "memory_id": memory_id,
                "timestamp": timestamp,
                "participants": user_context.get("participants", []),
                "topics": topics,
                "emotional_tone": user_context.get("emotional_tone", None),
                "summary": self_summary,
                "source": "autonomous",
                "memory_type": ["self"],
                "importance": importance,
            }
            if user_context.get("project"):
                metadata["project"] = user_context["project"]
            metadata.update(user_context)
            file_path = await self.storage.save_memory(content, metadata)
            embedding = await self.embedding_model.generate_embedding(content)
            chroma_metadata = {
                k: (
                    ", ".join(map(str, v)) if isinstance(v, list)
                    else (v if v is not None else "")
                )
                for k, v in {**metadata, "file_path": file_path}.items()
            }
            await self.vector_db.add_memory(embedding, chroma_metadata)
            # --- Extract and store self triples ---
            triples = await self.relation_extractor.extract(content, perspective="self")
            self.memory_graph.add_triples(triples, perspective="self", memory_id=memory_id)
            memory_ids.append(memory_id)

        # Store user memory (if present)
        if user_present:
            import uuid
            memory_id = str(uuid.uuid4())
            user_summary = await self.summarizer.summarize_user_observation(content)
            timestamp = datetime.datetime.utcnow().isoformat()
            metadata = {
                "memory_id": memory_id,
                "timestamp": timestamp,
                "participants": user_context.get("participants", []),
                "topics": topics,
                "emotional_tone": user_context.get("emotional_tone", None),
                "summary": user_summary,
                "source": "autonomous",
                "memory_type": ["user"],
                "importance": importance,
            }
            if user_context.get("project"):
                metadata["project"] = user_context["project"]
            metadata.update(user_context)
            file_path = await self.storage.save_memory(content, metadata)
            embedding = await self.embedding_model.generate_embedding(content)
            chroma_metadata = {
                k: (
                    ", ".join(map(str, v)) if isinstance(v, list)
                    else (v if v is not None else "")
                )
                for k, v in {**metadata, "file_path": file_path}.items()
            }
            await self.vector_db.add_memory(embedding, chroma_metadata)
            # --- Extract and store user triples ---
            triples = await self.relation_extractor.extract(content, perspective="user")
            self.memory_graph.add_triples(triples, perspective="user", memory_id=memory_id)
            memory_ids.append(memory_id)

        return memory_ids

    async def search_memories(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Semantic search for relevant relationship/self memories.

        Args:
            query (str): The search query.
            top_k (int): Number of top results to return.

        Returns:
            List[Dict[str, Any]]: List of memory metadata and content.
        """
        query_embedding = await self.embedding_model.generate_embedding(query)
        results = await self.vector_db.query(query_embedding, top_k=top_k)
        # Optionally load full content for each result
        for result in results:
            file_path = result.get("file_path")
            if file_path:
                memory_data = await self.storage.load_memory(file_path)
                result["content"] = memory_data.get("content")
                result["metadata"] = memory_data.get("metadata")
        return results

    async def summarize_memory(self, memory_id: str) -> str:
        """Summarize a relationship/self memory by ID, using label-aware summarization.

        Args:
            memory_id (str): The unique ID or file path of the memory.

        Returns:
            str: The summary of the memory.
        """
        memory_data = await self.storage.load_memory(memory_id)
        content = memory_data.get("content", "")
        metadata = memory_data.get("metadata", {})
        memory_type = metadata.get("memory_type", [])
        if isinstance(memory_type, str):
            memory_type = [memory_type]
        summary = await self.summarizer.summarize(content, memory_type=memory_type)
        return summary
