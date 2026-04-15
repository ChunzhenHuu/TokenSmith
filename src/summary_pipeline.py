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
        if cleaned:
            topics.append(cleaned)

    return topics

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

    for tq in topic_queries:
        raw_scores = {}
        for retriever in retrievers:
            raw_scores[retriever.name] = retriever.get_scores(
                tq, num_candidates, chunks
            )

        ordered_ids, ordered_scores = ranker.rank(raw_scores)

        # accumulate scores
        for idx, score in zip(ordered_ids, ordered_scores):
            merged_scores[idx] = merged_scores.get(idx, 0) + score

    # sort merged results
    sorted_items = sorted(merged_scores.items(), key=lambda x: x[1], reverse=True)

    topk_idxs = [idx for idx, _ in sorted_items[:top_k]]
    scores = [score for _, score in sorted_items[:top_k]]

    debug_info = {
        "topic_queries": topic_queries,
        "num_topics": len(topic_queries),
    }

    return topk_idxs, scores, debug_info

# generate structured result, to be completed
def build_summary_generation_query(original_query: str) -> str:

    return (
        "Organize the answer into:\n"
        "- Key topics\n"
        "- Explanation for each topic\n"
        "- Conclusion\n"
    )

# Generate reliability warning based on number of retrieved chunks. To be completed
def reliability_warning(num_chunks: int, threshold: int = 3) -> str:
    if num_chunks < threshold:
        return "Warning: Summary may be incomplete or inaccurate."
    return ""