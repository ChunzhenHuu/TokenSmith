import json
from pathlib import Path
import matplotlib.pyplot as plt

CASES = [
    {
        "name": "ARIES + 2PL",
        "expected_topics": ["ARIES", "two phase locking"],
    },
    {
        "name": "ARIES + importance",
        "expected_topics": ["ARIES", "important"],
    },
    {
        "name": "Long Conversation",
        "expected_topics": ["ARIES", "two phase locking", "strict two phase locking", "serializability"],
    },
]

def load_log(path):
    path = Path(path)
    if not path.exists():
        print(f"Missing log file: {path}")
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def topic_coverage(answer, expected_topics):
    text = answer.lower()
    covered = []
    for topic in expected_topics:
        if topic.lower() in text:
            covered.append(topic)
    return covered


def count_retrieved_chunks(log):
    if not log:
        return 0
    return len(log.get("retrieved_chunks", log.get("chunks", [])))

def count_pipeline_topics(log):
    if not log:
        return 0
    summary_info = log.get("summary_pipeline", {})
    return summary_info.get("num_topics", 0)


def has_warning(log):
    if not log:
        return False
    warning = log.get("summary_warning", "")
    return bool(warning)


def evaluate():
    rows = []

    for i, case in enumerate(CASES, start=1):
        baseline_log = load_log(f"evaluation_logs/baseline/case{i}.json")
        pipeline_log = load_log(f"evaluation_logs/pipeline/case{i}.json")

        baseline_answer = baseline_log.get("full_response", "") if baseline_log else ""
        pipeline_answer = pipeline_log.get("full_response", "") if pipeline_log else ""
        baseline_covered = topic_coverage(baseline_answer, case["expected_topics"])
        pipeline_covered = topic_coverage(pipeline_answer, case["expected_topics"])

        rows.append({
            "case": case["name"],
            "num_expected": len(case["expected_topics"]),
            "baseline_score": len(baseline_covered) / len(case["expected_topics"]),
            "pipeline_score": len(pipeline_covered) / len(case["expected_topics"]),
            "baseline_covered": baseline_covered,
            "pipeline_covered": pipeline_covered,
            "baseline_chunks": count_retrieved_chunks(baseline_log),
            "pipeline_chunks": count_retrieved_chunks(pipeline_log),
            "pipeline_topics": count_pipeline_topics(pipeline_log),
            "pipeline_warning": has_warning(pipeline_log),
        })

    return rows


def print_results(rows):
    print("\nEvaluation from logs\n")

    for row in rows:
        print(row["case"])
        print(f"  baseline coverage: {len(row['baseline_covered'])}/{row['num_expected']} "
              f"{row['baseline_covered']}")
        print(f"  pipeline coverage: {len(row['pipeline_covered'])}/{row['num_expected']} "
              f"{row['pipeline_covered']}")
        print(f"  baseline retrieved chunks: {row['baseline_chunks']}")
        print(f"  pipeline retrieved chunks: {row['pipeline_chunks']}")
        print(f"  pipeline extracted topics: {row['pipeline_topics']}")
        print(f"  pipeline warning: {row['pipeline_warning']}")
        print()

    avg_base = sum(r["baseline_score"] for r in rows) / len(rows)
    avg_pipe = sum(r["pipeline_score"] for r in rows) / len(rows)

    print("Average topic coverage")
    print(f"  baseline: {avg_base:.2f}")
    print(f"  pipeline: {avg_pipe:.2f}")


def make_plots(rows):
    out_dir = Path("evaluation_results")
    out_dir.mkdir(exist_ok=True)

    names = [r["case"] for r in rows]
    x = range(len(names))
    width = 0.35

    plt.figure(figsize=(8, 4))
    plt.bar([i - width / 2 for i in x], [r["baseline_score"] for r in rows],
            width, label="Baseline")
    plt.bar([i + width / 2 for i in x], [r["pipeline_score"] for r in rows],
            width, label="Summary pipeline")
    plt.xticks(list(x), names, rotation=20, ha="right")
    plt.ylabel("Topic coverage")
    plt.ylim(0, 1.1)
    plt.title("Topic Coverage from Generated Answers")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "topic_coverage_from_logs.png")
    plt.close()

    plt.figure(figsize=(8, 4))
    plt.bar([i - width / 2 for i in x], [r["baseline_chunks"] for r in rows],
            width, label="Baseline")
    plt.bar([i + width / 2 for i in x], [r["pipeline_chunks"] for r in rows],
            width, label="Summary pipeline")
    plt.xticks(list(x), names, rotation=20, ha="right")
    plt.ylabel("Retrieved chunks")
    plt.title("Number of Retrieved Chunks")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "retrieved_chunks_from_logs.png")
    plt.close()

    print("\nSaved plots:")
    print("  evaluation_results/topic_coverage_from_logs.png")
    print("  evaluation_results/retrieved_chunks_from_logs.png")


if __name__ == "__main__":
    rows = evaluate()
    print_results(rows)
    make_plots(rows)