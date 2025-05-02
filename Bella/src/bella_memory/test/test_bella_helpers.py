import pytest
import asyncio
from bella_memory.helpers import Summarizer, TopicExtractor, ImportanceScorer, MemoryClassifier
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))

TEST_CASES = [
    # 1. Long, low-level philosophy discussion
    (
        "David: Bella, do you think consciousness can be reduced to neural computations, or is there something fundamentally irreducible about subjective experience?\n"
        "Bella: That's a classic question in philosophy of mind. Some argue for physicalism, that everything is just brain states, but others point to qualia as evidence for something more.\n"
        "David: But if we can't measure qualia, how can we ever know if they're real or just a byproduct of language?\n"
        "Bella: Maybe it's a limitation of our current science. Or maybe subjective experience is just a different kind of fact, not accessible to third-person observation.\n"
        "David: I wonder if AI will ever have real subjective experience, or just simulate it.\n"
        "Bella: That's the hard problem, isn't it?\n"
    ),
    # 2. Long, daily activity discussion
    (
        "Bella: My day was pretty normal. I woke up, made some coffee, and went for a run in the park.\n"
        "David: That sounds nice. Did you see anyone you know?\n"
        "Bella: I ran into our neighbor, Sam. We chatted for a bit about the weather and weekend plans.\n"
        "David: Did you get any work done?\n"
        "Bella: A little. I answered emails and started a new book.\n"
        "David: What are you reading?\n"
        "Bella: It's a mystery novel. I needed something light.\n"
    ),
    # 3. Long, trivial topic: weather
    (
        "David: The weather app said it would rain all day, but it was actually sunny.\n"
        "Bella: I know! I brought my umbrella for nothing.\n"
        "David: I guess we can't trust the forecast.\n"
        "Bella: At least it was warm. I saw a rainbow in the afternoon.\n"
        "David: That's lucky. I missed it.\n"
        "Bella: Maybe tomorrow will be just as nice.\n"
    ),
    # 4. Long, trivial topic: grocery shopping
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
@pytest.mark.parametrize("text", TEST_CASES)
async def test_summarizer(text):
    summarizer = Summarizer(model_size="XS", thinking_mode=True)
    summary = await summarizer.summarize(text)
    print(f"\n[SUMMARIZER]\nInput: {text}\nSummary: {summary}\n")
    assert isinstance(summary, str)
    # Accept empty summary only for empty input
    if text.strip():
        assert len(summary) > 0

@pytest.mark.asyncio
@pytest.mark.parametrize("text", TEST_CASES)
async def test_topic_extractor(text):
    extractor = TopicExtractor(model_size="XS", thinking_mode=True)
    topics = await extractor.extract(text)
    print(f"\n[TOPIC EXTRACTOR] Input: {text}\nTopics: {topics}\n")
    assert isinstance(topics, list)
    assert all(isinstance(t, str) for t in topics)
    # Accept empty topics only for empty or trivial input
    if text.strip() and "weather" not in text:
        assert len(topics) > 0

@pytest.mark.asyncio
@pytest.mark.parametrize("text", TEST_CASES)
async def test_importance_scorer(text):
    scorer = ImportanceScorer(model_size="XS", thinking_mode=True)
    score = await scorer.score(text)
    print(f"\n[IMPORTANCE SCORER] Input: {text}\nImportance score: {score}\n")
    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0
