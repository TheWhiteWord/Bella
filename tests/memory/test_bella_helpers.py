import pytest
import asyncio
from bella_memory.helpers import Summarizer, TopicExtractor, ImportanceScorer, MemoryClassifier

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

import os
RESULTS_FILE = os.path.join(os.path.dirname(__file__), "bella_helpers_llm_outputs.txt")

def log_llm_result(section, text, result):
    with open(RESULTS_FILE, "a", encoding="utf-8") as f:
        f.write(f"\n[{section}]\nInput: {text}\nOutput: {result}\n")

@pytest.mark.asyncio
@pytest.mark.parametrize("text", TEST_CASES)
async def test_summarizer_actual(text):
    summarizer = Summarizer(model_size="XXS", thinking_mode=False)
    summary = await summarizer.summarize(text)
    log_llm_result("SUMMARIZER", text, summary)
    print(f"\n[SUMMARIZER]\nInput: {text}\nSummary: {summary}\n")
    assert isinstance(summary, str)
    if text.strip():
        assert len(summary) > 0

@pytest.mark.asyncio
@pytest.mark.parametrize("text", TEST_CASES)
async def test_topic_extractor_actual(text):
    extractor = TopicExtractor(model_size="XXS", thinking_mode=False)
    topics = await extractor.extract(text)
    log_llm_result("TOPIC EXTRACTOR", text, topics)
    print(f"\n[TOPIC EXTRACTOR] Input: {text}\nTopics: {topics}\n")
    assert isinstance(topics, list)
    assert all(isinstance(t, str) for t in topics)
    if text.strip() and "weather" not in text:
        assert len(topics) > 0

@pytest.mark.asyncio
@pytest.mark.parametrize("text", TEST_CASES)
async def test_importance_scorer_actual(text):
    scorer = ImportanceScorer(model_size="XXS", thinking_mode=False)
    score = await scorer.score(text)
    log_llm_result("IMPORTANCE SCORER", text, score)
    print(f"\n[IMPORTANCE SCORER] Input: {text}\nImportance score: {score}\n")
    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0

@pytest.mark.asyncio
@pytest.mark.parametrize("text", TEST_CASES)
async def test_summarize_self_insight(text):
    summarizer = Summarizer(model_size="XXS", thinking_mode=False)
    summary = await summarizer.summarize_self_insight(text)
    log_llm_result("SELF INSIGHT", text, summary)
    print(f"\n[SELF INSIGHT]\nInput: {text}\nSelf Insight: {summary}\n")
    assert isinstance(summary, str)
    if text.strip():
        assert len(summary) > 0

@pytest.mark.asyncio
@pytest.mark.parametrize("text", TEST_CASES)
async def test_summarize_user_observation(text):
    summarizer = Summarizer(model_size="XXS", thinking_mode=False)
    summary = await summarizer.summarize_user_observation(text)
    log_llm_result("USER OBSERVATION", text, summary)
    print(f"\n[USER OBSERVATION]\nInput: {text}\nUser Observation: {summary}\n")
    assert isinstance(summary, str)
    if text.strip():
        assert len(summary) > 0

