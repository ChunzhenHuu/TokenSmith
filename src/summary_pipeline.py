import re
from typing import List, Tuple, Dict, Optional


SUMMARY_PATTERNS = [
    r"\bsummarize (?:what|everything|all|the things)? ?(?:we )?(?:discussed|covered|talked about)(?: so far)?\b",
    r"\bsummarize (?:our|the) (?:chat|conversation|discussion)\b",
    r"\bgive me (?:a )?(?:summary|recap) of (?:our|the) (?:chat|conversation|discussion)\b",
    r"\bwhat did we (?:discuss|cover|talk about)\b",
    r"\bwhat have we (?:discussed|covered) so far\b",
    r"\brecap (?:our|the)? ?(?:chat|conversation|discussion)?\b",
    r"\bkey points from (?:our|the) (?:chat|conversation|discussion)\b",
]

SUMMARY_SIGNALS = {
    "summary", "summarize", "summarise", "recap", "overview",
    "key points", "highlights", "main points",
}
CONVERSATION_SIGNALS = {
    "chat", "conversation", "discussion", "discussed", "covered",
    "so far", "earlier", "before", "previous", "we talked",
}

# detect whether the user query is multi-turn summary query or not
def is_multi_turn_summary_query(query: str) -> bool:
    q = query.lower().strip()
    if not q:
        return False

    for pattern in SUMMARY_PATTERNS:
        if re.search(pattern, q):
            return True

    has_summary_signal = any(signal in q for signal in SUMMARY_SIGNALS)
    has_conversation_signal = any(signal in q for signal in CONVERSATION_SIGNALS)
    return has_summary_signal and has_conversation_signal


# If the user seems to want a conversation recap but uses a vague phrase, provide a small hint about supported phrasing.
def maybe_summary_guidance(query: str) -> str:
    q = query.lower().strip()
    if not q:
        return ""

    vague_summary_requests = [
        "give me a summary",
        "summary please",
        "summarize",
        "recap please",
        "can you recap",
    ]
    if any(phrase in q for phrase in vague_summary_requests) and not is_multi_turn_summary_query(q):
        return (
            "If you want a summary of this chat, try asking something like: "
            "'summarize what we discussed so far' or 'give me a recap of our conversation'."
        )
    return ""


# Extract topic queries from ALL prior user turns in the current chat,not just a fixed recent window. Exclude summary-like turns themselves.
def extract_topic_queries(chat_history: List[Dict], max_topics: Optional[int] = None) -> List[str]:
    user_queries = [turn["content"] for turn in chat_history if turn.get("role") == "user"]

    topics: List[str] = []
    seen = set()

    for q in user_queries:
        cleaned = _normalize_query_text(q)
        if not cleaned:
            continue
        if is_multi_turn_summary_query(cleaned):
            continue

        lowered = cleaned.lower()
        if lowered in seen:
            continue

        topics.append(cleaned)
        seen.add(lowered)

    if max_topics is not None and max_topics > 0:
        return topics[-max_topics:]
    return topics

def _normalize_query_text(query: str) -> str:
    q = re.sub(r"\s+", " ", query).strip()
    q = re.sub(r"^[\-–—\s]+", "", q)
    return q


# Make short or vague topic queries a little more specific before retrieval.
def enhance_topic_query(query: str) -> str:
    q = _normalize_query_text(query)
    q_lower = q.lower()

    conversational_prefixes = [
        "can you explain ",
        "could you explain ",
        "please explain ",
        "tell me about ",
        "what about ",
        "i want to know about ",
    ]
    for prefix in conversational_prefixes:
        if q_lower.startswith(prefix):
            q = q[len(prefix):].strip()
            q_lower = q.lower()
            break

    if q_lower.startswith("what is "):
        core = q[8:].strip(" ?.")
        return f"definition of {core} in database systems"
    if q_lower.startswith("what are "):
        core = q[9:].strip(" ?.")
        return f"explanation of {core} in database systems"
    if q_lower.startswith("why "):
        core = q[4:].strip(" ?.")
        return f"explain why {core} in database systems"
    if q_lower.startswith("how "):
        core = q[4:].strip(" ?.")
        return f"explain how {core} in database systems"

    token_count = len(q.split())
    if token_count <= 3:
        return f"{q} in database systems"

    return q



# Run retrieval for each topic and merge scores.
def retrieve_multi_topic_chunks(
    topic_queries: List[str],
    retrievers,
    ranker,
    chunks: List[str],
    top_k: int,
    num_candidates: int,
):
    merged_scores: Dict[int, float] = {}
    enhanced_topic_queries = [enhance_topic_query(tq) for tq in topic_queries]
    per_topic_debug = []

    for original_tq, enhanced_tq in zip(topic_queries, enhanced_topic_queries):
        raw_scores = {}
        for retriever in retrievers:
            raw_scores[retriever.name] = retriever.get_scores(
                enhanced_tq, num_candidates, chunks
            )

        ordered_ids, ordered_scores = ranker.rank(raw_scores)

        for idx, score in zip(ordered_ids, ordered_scores):
            merged_scores[idx] = merged_scores.get(idx, 0.0) + float(score)

        per_topic_debug.append({
            "original_topic_query": original_tq,
            "enhanced_topic_query": enhanced_tq,
            "num_ranked_candidates": len(ordered_ids),
            "top_candidate_ids": ordered_ids[:5],
        })

    sorted_items = sorted(merged_scores.items(), key=lambda x: x[1], reverse=True)
    topk_idxs = [idx for idx, _ in sorted_items[:top_k]]
    scores = [score for _, score in sorted_items[:top_k]]

    debug_info = {
        "topic_queries": topic_queries,
        "enhanced_topic_queries": enhanced_topic_queries,
        "num_topics": len(topic_queries),
        "per_topic_debug": per_topic_debug,
        "num_merged_candidates": len(merged_scores),
    }

    return topk_idxs, scores, debug_info

# generate structured result.
def build_summary_generation_query(original_query: str, topic_queries: List[str]) -> str:
    if topic_queries:
        topic_block = "\n".join(f"- {topic}" for topic in topic_queries)
    else:
        topic_block = "- No explicit topics detected"
    return (
        f"{original_query}\n\n"
        "Generate a structured summary of the previous discussion using only the provided textbook excerpts.\n"
        "Try to cover all main topics from the discussion.\n\n"
        "Topics that should be covered:\n"
        f"{topic_block}\n\n"
        "Organize the answer into exactly these sections:\n"
        "1. Key Topics\n"
        "2. Explanation for Each Topic\n"
        "3. Conclusion\n\n"
        "If some topics are not well supported by the retrieved excerpts, state that clearly."
    )

# Generate reliability warning based on number of retrieved chunks.
def reliability_warning(num_chunks: int, num_topics: int, threshold: int = 3) -> str:
    if num_topics <= 0:
        return "Warning: No clear prior user topics were detected, so the summary may be incomplete."
    if num_chunks < threshold:
        return "Warning: Only a small number of supporting chunks were retrieved, so the summary may be incomplete or inaccurate."
    if num_chunks < num_topics:
        return "Warning: Fewer chunks were retrieved than the number of detected topics, so some key points may be under-covered."
    return ""