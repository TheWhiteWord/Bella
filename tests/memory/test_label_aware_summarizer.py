
import asyncio
import pytest
from bella_memory.helpers import Summarizer, MemoryClassifier, TopicExtractor, ImportanceScorer
from bella_memory.core import BellaMemory

LABEL_TEST_CASES = [
    # Philosophy/AI (should trigger self, user, and main)
    (
        "David: Bella, do you think consciousness can be reduced to neural computations, or is there something fundamentally irreducible about subjective experience?\n"
        "Bella: That's a classic question in philosophy of mind. Some argue for physicalism, that everything is just brain states, but others point to qualia as evidence for something more.\n"
        "David: But if we can't measure qualia, how can we ever know if they're real or just a byproduct of language?\n"
        "Bella: Maybe it's a limitation of our current science. Or maybe subjective experience is just a different kind of fact, not accessible to third-person observation.\n"
        "David: I wonder if AI will ever have real subjective experience, or just simulate it.\n"
        "Bella: That's the hard problem, isn't it?\n"
    ),
    # Grocery shopping (should trigger self, main, maybe user)
    (
        "Bella: I went grocery shopping today.\n"
        "David: Did you remember to get eggs?\n"
        "Bella: Yes, and I got some fresh bread too.\n"
        "David: Nice. Did you see the new bakery section?\n"
        "Bella: I did! It smelled amazing. I almost bought a cake.\n"
        "David: Maybe next time.\n"
    ),
    # Emotional support (should trigger self, user, and main)
    (
        "Bella: I felt really anxious about my presentation today.\n"
        "David: You did great, Bella. I could tell you were nervous, but you handled it well.\n"
        "Bella: Thanks, David. Your encouragement helped me calm down.\n"
        "David: I'm always here for you.\n"
        "Bella: That means a lot.\n"
    ),
    # Playful teasing (should trigger user, main)
    (
        "David: You always take forever to pick a movie, Bella!\n"
        "Bella: That's because I want to find the perfect one.\n"
        "David: Or maybe you just like making me wait.\n"
        "Bella: Maybe a little. But you secretly enjoy the suspense.\n"
        "David: Only if there's popcorn.\n"
    ),
    # Project collaboration (should trigger main, maybe self/user)
    (
        "Bella: I finished the first draft of our project report.\n"
        "David: Awesome! Want me to review it?\n"
        "Bella: Yes, please. Your feedback always helps.\n"
        "David: I'll read it tonight.\n"
        "Bella: Thank you!\n"
    ),
]

# Integration test for the new BellaMemory.store_memory logic

# Enhanced DummyStorage to capture all saved memories for inspection
class DummyStorage:
    def __init__(self):
        self.saved = []  # List of (content, metadata)
    async def save_memory(self, content, metadata):
        self.saved.append((content, dict(metadata)))
        # Return a fake file path based on memory_type
        return f"/fake/path/{'-'.join(metadata['memory_type'])}.md"
    async def load_memory(self, file_path):
        # Find the memory by file_path
        for _, meta in self.saved:
            if f"/fake/path/{'-'.join(meta['memory_type'])}.md" == file_path:
                return {"content": "dummy", "metadata": meta}
        return {"content": "dummy", "metadata": {}}

class DummyEmbedding:
    async def generate_embedding(self, content):
        return [0.0] * 384

class DummyVectorDB:
    async def add_memory(self, embedding, metadata):
        return None
    async def query(self, embedding, top_k=5):
        return []

@pytest.mark.asyncio
@pytest.mark.parametrize("text", LABEL_TEST_CASES)
async def test_bella_memory_store_logic(text):
    summarizer = Summarizer(model_size="XS", thinking_mode=True)
    classifier = MemoryClassifier(model_size="XS", thinking_mode=True)
    topic_extractor = TopicExtractor(model_size="XS", thinking_mode=True)
    importance_scorer = ImportanceScorer(model_size="XS", thinking_mode=True)
    storage = DummyStorage()
    bella_memory = BellaMemory(
        summarizer=summarizer,
        topic_extractor=topic_extractor,
        importance_scorer=importance_scorer,
        storage=storage,
        embedding_model=DummyEmbedding(),
        vector_db=DummyVectorDB(),
        memory_classifier=classifier,
    )
    user_context = {"participants": ["Bella", "David"]}
    file_paths = await bella_memory.store_memory(text, user_context)
    print(f"\n[BELLA MEMORY STORE LOGIC] Input: {text}\nStored file paths: {file_paths}")
    for i, (content, meta) in enumerate(storage.saved):
        print(f"  Memory {i+1} type: {meta['memory_type']}")
        print(f"    Summary: {meta['summary']}")
        print(f"    Topics: {meta.get('topics')}")
        print(f"    Importance: {meta.get('importance')}")
        print(f"    All metadata: {meta}")
    # Should store at least one memory if important
    assert isinstance(file_paths, list)
    if file_paths:
        assert all(isinstance(fp, str) for fp in file_paths)
import pytest
from bella_memory.helpers import Summarizer, MemoryClassifier

LABEL_TEST_CASES = [
    (
        "David: Bella, do you think consciousness can be reduced to neural computations, or is there something fundamentally irreducible about subjective experience?\n"
        "Bella: That's a classic question in philosophy of mind. Some argue for physicalism, that everything is just brain states, but others point to qualia as evidence for something more.\n"
        "David: But if we can't measure qualia, how can we ever know if they're real or just a byproduct of language?\n"
        "Bella: Maybe it's a limitation of our current science. Or maybe subjective experience is just a different kind of fact, not accessible to third-person observation.\n"
        "David: I wonder if AI will ever have real subjective experience, or just simulate it.\n"
        "Bella: That's the hard problem, isn't it?\n"
    ),
    (
        "Bella: I went grocery shopping today.\n"
        "David: Did you remember to get eggs?\n"
        "Bella: Yes, and I got some fresh bread too.\n"
        "David: Nice. Did you see the new bakery section?\n"
        "Bella: I did! It smelled amazing. I almost bought a cake.\n"
        "David: Maybe next time.\n"
    ),
]

@pytest.mark.asyncio
@pytest.mark.parametrize("text", LABEL_TEST_CASES)
async def test_label_aware_summarizer(text):
    classifier = MemoryClassifier(model_size="XS", thinking_mode=True)
    summarizer = Summarizer(model_size="XS", thinking_mode=True)
    labels = await classifier.classify(text)
    summary = await summarizer.summarize(text, memory_type=labels)
    print(f"\n[LABEL-AWARE SUMMARIZER]\nInput: {text}\nLabels: {labels}\nSummary: {summary}\n")
    assert isinstance(labels, list)
    assert all(isinstance(l, str) for l in labels)
    assert len(labels) > 0
    assert isinstance(summary, str)
    assert len(summary) > 0