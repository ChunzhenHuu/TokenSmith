from typing import List, Tuple, Dict

# detect whether the user query is multi-turn summary query or not
def is_multi_turn_summary_query(query: str) -> bool:
    q = query.lower()
    keywords = [
        "summarize what we discussed",
        "summary of conversation",
        "what we discussed",
        "what did we talk",
        "key points from our discussion",
        "recap",
        "previous discussion",
        "previous conversation",
        "previous chat",
        "previous chat",
        "summarize our chat",
        "summarize our conversation",
        "summarize the conversation",
        "give me a summary",
    ]
    return any(k in q for k in keywords)

# Extract recent user queries as topic queries.
def extract_topic_queries(chat_history: List[Dict], max_topics: int = 4) -> List[str]:

    user_queries = [turn["content"] for turn in chat_history if turn["role"] == "user"]

    # take last 4 queries in current implementation
    recent = user_queries[-max_topics:]

    topics = []
    for q in recent:
        cleaned = q.strip()
        if cleaned and not is_multi_turn_summary_query(cleaned):
            topics.append(cleaned)

    return topics

def _normalize_query_text(query: str) -> str:
    q = re.sub(r"\s+", " ", query).strip()
    q = re.sub(r"^[\-–—\s]+", "", q)
    return q


# Lightweight query enhancement after topic extraction.
def enhance_topic_query(query: str) -> str:
    """
    Reformulate a topic query into a slightly more retrieval-friendly form
    while preserving its original meaning.

    This is intentionally lightweight and deterministic so it is easy to
    analyze in the final report.
    """
    q = _normalize_query_text(query)
    q_lower = q.lower()

    # Remove very conversational wrappers.
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

    # Preserve intent but add retrieval-friendly scaffolding.
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

# generate structured result, to be completed
def build_summary_generation_query(original_query: str) -> str:

    topic_block = "\n".join(f"- {topic}" for topic in topic_queries) if topic_queries else "- No explicit topics detected"
    return (
        f"{original_query}\n\n"
        "Generate a structured summary of the previous discussion using ONLY the provided textbook excerpts.\n"
        "Focus on covering all major topics from the discussion.\n\n"
        "Topics that should be covered:\n"
        f"{topic_block}\n\n"
        "Organize the answer into exactly these sections:\n"
        "1. Key Topics\n"
        "2. Explanation for Each Topic\n"
        "3. Conclusion\n\n"
        "If some topics are not well supported by the retrieved excerpts, state that clearly."
    )

# Generate reliability warning based on number of retrieved chunks. To be completed
def reliability_warning(num_chunks: int, num_topics: int, threshold: int = 3) -> str:
    if num_topics <= 0:
        return "Warning: No clear prior user topics were detected, so the summary may be incomplete."
    if num_chunks < threshold:
        return "Warning: Only a small number of supporting chunks were retrieved, so the summary may be incomplete or inaccurate."
    if num_chunks < num_topics:
        return "Warning: Fewer chunks were retrieved than the number of detected topics, so some discussion points may be under-covered."
    return ""