import pytest
import asyncio
from bella_memory.core import BellaMemory
from bella_memory.helpers import Summarizer, TopicExtractor, ImportanceScorer, MemoryClassifier
from bella_memory.storage import MemoryStorage
from bella_memory.embeddings import EmbeddingModel
from bella_memory.vector_db import VectorDB

@pytest.mark.asyncio
async def test_bella_memory_full_workflow(tmp_path):
    """
    Integration test: store a memory, check markdown file, and search via vector DB.
    """
    # Use a temp chroma db path for isolation
    chroma_db_path = tmp_path / "chroma_db"
    chroma_db_path.mkdir()
    # Instantiate all components
    summarizer = Summarizer(model_size="XS", thinking_mode=True)
    topic_extractor = TopicExtractor(model_size="XS", thinking_mode=True)
    importance_scorer = ImportanceScorer(model_size="XS", thinking_mode=True)
    memory_classifier = MemoryClassifier(model_size="XS", thinking_mode=True)
    storage = MemoryStorage()
    embedding_model = EmbeddingModel()
    vector_db = VectorDB(db_path=str(chroma_db_path))
    bella_memory = BellaMemory(
        summarizer=summarizer,
        topic_extractor=topic_extractor,
        importance_scorer=importance_scorer,
        storage=storage,
        embedding_model=embedding_model,
        vector_db=vector_db,
        memory_classifier=memory_classifier,
    )
    # Test input
    content = """Bella: I felt proud after finishing the marathon. It made me realize how determined I can be when I set my mind to something."""
    user_context = {"participants": ["Bella", "David"], "emotional_tone": "proud, determined"}
    # Store memory
    file_paths = await bella_memory.store_memory(content, user_context)
    assert file_paths, "No memory file was created!"
    print(f"Memory file(s) created: {file_paths}")
    # Check markdown file exists and has headings
    import os
    for path in file_paths:
        assert os.path.exists(path), f"File {path} does not exist!"
        with open(path, "r", encoding="utf-8") as f:
            md = f.read()
        print(f"\n---\n{md}\n---\n")
        assert "# timestamp" in md and "## summary" in md, "Markdown headings missing!"
    # Test search
    results = await bella_memory.search_memories("marathon proud")
    assert results, "No search results returned!"
    print(f"Search results: {results}")
