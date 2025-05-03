import pytest
from bella_memory.helpers import SelfUserRelationExtractor
from bella_memory.core import SimpleMemoryGraph

TEST_CASES = [
    # Bella's self-reflection (subject always 'Bella')
    (
        "Bella: I felt proud after finishing the marathon. It made me realize how determined I can be when I set my mind to something.",
        "self",
        [
            {"subject": "Bella", "type": "feels", "value": "proud"},
            {"subject": "Bella", "type": "thinks_that", "value": "how determined I can be when I set my mind to something"},
        ]
    ),
    # Bella's observation about David (subject always 'David')
    (
        "David: I think AI will change the world. Bella: David seems very optimistic about technology.",
        "user",
        [
            {"subject": "David", "type": "thinks_that", "value": "AI will change the world."},
        ]
    ),
    # Mixed conversation (subject always 'Bella' for self)
    (
        "Bella: I feel nervous about the presentation. David: You'll do great! Bella: Thanks, your support helps me feel more confident.",
        "self",
        [
            {"subject": "Bella", "type": "thinks_that", "value": "I feel nervous about the presentation"},
            {"subject": "Bella", "type": "feels", "value": "more confident"},
        ]
    ),
    # Mixed conversation (subject always 'David' for user)
    (
        "Bella: I feel nervous about the presentation. David: You'll do great! Bella: Thanks, your support helps me feel more confident.",
        "user",
        [
            {"subject": "David", "type": "thinks_that", "value": "you'll do great!"},
        ]
    ),
]

@pytest.mark.asyncio
@pytest.mark.parametrize("text,perspective,expected", TEST_CASES)
async def test_self_user_relation_extractor(text, perspective, expected, monkeypatch):
    # Wrap the generate function to print prompt and result
    from bella_memory import helpers
    orig_generate = helpers.generate

    async def debug_generate(prompt, *args, **kwargs):
        result = await orig_generate(prompt, *args, **kwargs)
        print(f"\n[LLM RAW OUTPUT]\nPrompt:\n{prompt}\nResult:\n{result}\n")
        return result

    monkeypatch.setattr(helpers, "generate", debug_generate)

    extractor = SelfUserRelationExtractor(model_size="XXS", thinking_mode=False)
    triples = await extractor.extract(text, perspective=perspective)
    print(f"\n[EXTRACTOR] Perspective: {perspective}\nInput: {text}\nTriples: {triples}\n")
    assert isinstance(triples, list)
    for exp in expected:
        found = any(
            exp["subject"].lower() == t["subject"].lower()
            and exp["type"].lower() in t["type"].lower()
            and any(word in t["value"].lower() for word in exp["value"].lower().split())
            for t in triples
        )
        assert found, f"Expected subject/type/value not found: {exp}"

def test_memory_graph_add_and_query():
    graph = SimpleMemoryGraph()
    triples = [
        {"subject": "AI", "type": "think", "value": "will change the world"},
        {"subject": "technology", "type": "feel", "value": "optimistic"},
    ]
    graph.add_triples(triples, perspective="user", memory_id="/fake/path/user.md")
    results = graph.query(perspective="user", subject="AI", type_="think")
    assert len(results) == 1
    assert results[0]["subject"] == "AI"
    assert results[0]["type"] == "think"
