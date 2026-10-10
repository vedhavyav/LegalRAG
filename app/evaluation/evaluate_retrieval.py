import json
from pathlib import Path
from statistics import mean

from app.retrieval.search import search as bm25_search
from app.retrieval.semantic_search import semantic_search
from app.retrieval.hybrid_search import hybrid_search


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_PATH = PROJECT_ROOT / "data" / "evaluation" / "test_queries.json"


def load_test_queries():
    """Load queries and expected relevant judgment IDs."""
    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Evaluation dataset not found: {DATASET_PATH}"
        )

    with DATASET_PATH.open("r", encoding="utf-8") as file:
        queries = json.load(file)

    if not isinstance(queries, list) or not queries:
        raise ValueError("The evaluation dataset must be a non-empty list.")

    for index, item in enumerate(queries, start=1):
        if not item.get("query", "").strip():
            raise ValueError(f"Query {index} has an empty query.")

        if not item.get("expected_source_ids"):
            raise ValueError(
                f"Query {index} has no expected_source_ids."
            )

    return queries


def calculate_metrics(results, expected_source_ids, k=5):
    """
    Evaluate retrieval at the judgment/source level.

    Recall@k: relevant expected judgments retrieved / all expected judgments.
    Precision@k: relevant retrieved judgments / k.
    MRR: reciprocal rank of the first relevant result.
    """
    expected = set(expected_source_ids)
    top_results = results[:k]

    retrieved_ids = [
        result["source_id"] for result in top_results
    ]

    relevant_count = len(set(retrieved_ids) & expected)

    recall = relevant_count / len(expected)
    precision = relevant_count / k

    reciprocal_rank = 0.0

    for rank, source_id in enumerate(retrieved_ids, start=1):
        if source_id in expected:
            reciprocal_rank = 1.0 / rank
            break

    return {
        "recall_at_k": recall,
        "precision_at_k": precision,
        "reciprocal_rank": reciprocal_rank,
        "relevant_retrieved": relevant_count,
        "expected_relevant": len(expected),
    }


def evaluate_retrieval(k=5):
    queries = load_test_queries()

    retrieval_methods = {
        "BM25": lambda query: bm25_search(query, top_k=k),
        "Semantic": lambda query: semantic_search(query, top_k=k),
        "Hybrid": lambda query: hybrid_search(query, top_k=k),
    }

    all_metrics = {name: [] for name in retrieval_methods}

    for item in queries:
        query = item["query"]
        expected_ids = item["expected_source_ids"]

        print("\n" + "=" * 80)
        print(f"Query: {query}")
        print(f"Expected relevant judgments: {expected_ids}")

        for method_name, retrieve in retrieval_methods.items():
            results = unique_judgments(retrieve(query, ), k=k)
            metrics = calculate_metrics(
                results,
                expected_ids,
                k=k,
            )

            all_metrics[method_name].append(metrics)

            print(f"\n{method_name}")
            print(f"  Recall@{k}:     {metrics['recall_at_k']:.3f}")
            print(f"  Precision@{k}:  {metrics['precision_at_k']:.3f}")
            print(f"  MRR:            {metrics['reciprocal_rank']:.3f}")
            print(
                f"  Relevant found: "
                f"{metrics['relevant_retrieved']}/"
                f"{metrics['expected_relevant']}"
            )

            print("  Retrieved judgments:")
            for rank, result in enumerate(results[:k], start=1):
                print(
                    f"    {rank}. {result['source_id']} | "
                    f"{result.get('title', 'Untitled')} | "
                    f"chunk {result['chunk_index']}"
                )

    print("\n" + "=" * 80)
    print("AGGREGATE RESULTS (macro averages)")

    for method_name, metrics_list in all_metrics.items():
        print(f"\n{method_name}")
        print(
            f"  Mean Recall@{k}: "
            f"{mean(m['recall_at_k'] for m in metrics_list):.3f}"
        )
        print(
            f"  Mean Precision@{k}: "
            f"{mean(m['precision_at_k'] for m in metrics_list):.3f}"
        )
        print(
            f"  Mean MRR: "
            f"{mean(m['reciprocal_rank'] for m in metrics_list):.3f}"
        )

def unique_judgments(results, k=5):
    """Keep the highest-ranked chunk from each judgment."""
    unique = []
    seen_source_ids = set()

    for result in results:
        source_id = result["source_id"]

        if source_id in seen_source_ids:
            continue

        seen_source_ids.add(source_id)
        unique.append(result)

        if len(unique) >= k:
            break

    return unique


if __name__ == "__main__":
    evaluate_retrieval(k=5)
