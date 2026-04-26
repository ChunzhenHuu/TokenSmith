import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.summary_pipeline import (
    is_multi_turn_summary_query,
    maybe_summary_guidance,
    extract_topic_queries,
    enhance_topic_query,
    build_summary_generation_query,
    reliability_warning,
)

def test_summary_query_detection():
    summary_queries = [
        "summarize what we discussed so far",
        "give me a summary of our conversation",
        "recap our chat",
        "what did we cover",
        "key points from our discussion",
    ]

    normal_queries = [
        "What is ARIES?",
        "Explain two phase locking",
        "How does recovery work?",
        "What is a B+ tree?",
    ]

    for q in summary_queries:
        assert is_multi_turn_summary_query(q)

    for q in normal_queries:
        assert not is_multi_turn_summary_query(q)


def test_maybe_summary_guidance():
    vague_query = "summarize"
    guidance = maybe_summary_guidance(vague_query)

    assert guidance != ""
    assert "summarize what we discussed" in guidance


def test_extract_topic_queries_from_history():
    chat_history = [
        {"role": "user", "content": "What is ARIES?"},
        {"role": "assistant", "content": "ARIES is a recovery algorithm."},
        {"role": "user", "content": "What is two phase locking?"},
        {"role": "assistant", "content": "2PL is a locking protocol."},
        {"role": "user", "content": "summarize what we discussed so far"},
    ]

    topics = extract_topic_queries(chat_history)

    assert "What is ARIES?" in topics
    assert "What is two phase locking?" in topics
    assert "summarize what we discussed so far" not in topics
    assert len(topics) == 2

def test_extract_topic_queries_removes_duplicates():
    chat_history = [
        {"role": "user", "content": "What is ARIES?"},
        {"role": "assistant", "content": "Answer"},
        {"role": "user", "content": "What is ARIES?"},
        {"role": "assistant", "content": "Answer again"},
        {"role": "user", "content": "What is 2PL?"},
    ]

    topics = extract_topic_queries(chat_history)

    assert topics.count("What is ARIES?") == 1
    assert "What is 2PL?" in topics


def test_extract_topic_queries_full_history_not_last_four():
    chat_history = [
        {"role": "user", "content": "Topic 1"},
        {"role": "assistant", "content": "Answer 1"},
        {"role": "user", "content": "Topic 2"},
        {"role": "assistant", "content": "Answer 2"},
        {"role": "user", "content": "Topic 3"},
        {"role": "assistant", "content": "Answer 3"},
        {"role": "user", "content": "Topic 4"},
        {"role": "assistant", "content": "Answer 4"},
        {"role": "user", "content": "Topic 5"},
        {"role": "assistant", "content": "Answer 5"},
    ]

    topics = extract_topic_queries(chat_history)

    assert len(topics) == 5
    assert "Topic 1" in topics
    assert "Topic 5" in topics


def test_query_enhancement():
    assert enhance_topic_query("What is ARIES?") == "definition of ARIES in database systems"
    assert enhance_topic_query("How does recovery work?") == "explain how does recovery work in database systems"
    assert enhance_topic_query("2PL") == "2PL in database systems"


def test_summary_generation_query_contains_topics_and_structure():
    topics = ["What is ARIES?", "What is 2PL?"]
    prompt = build_summary_generation_query(
        "summarize what we discussed so far",
        topics,
    )

    assert "What is ARIES?" in prompt
    assert "What is 2PL?" in prompt
    assert "Key Topics" in prompt
    assert "Explanation for Each Topic" in prompt
    assert "Conclusion" in prompt


def test_reliability_warning():
    assert "No clear prior user topics" in reliability_warning(num_chunks=5, num_topics=0)
    assert "small number of supporting chunks" in reliability_warning(num_chunks=1, num_topics=2)
    assert "Fewer chunks" in reliability_warning(num_chunks=2, num_topics=4)
    assert reliability_warning(num_chunks=5, num_topics=2) == ""