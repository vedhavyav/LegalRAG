import json
from pathlib import Path
from statistics import mean
from time import perf_counter

from app.retrieval.search import search as bm25_search
from app.retrieval.semantic_search import semantic_search
from app.retrieval.hybrid_search import hybrid_search
from app.retrieval.query_expansion import expand_query


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


def calculate_metrics(results, expected_source_ids, k=5):
    """Calculate Recall@k, Precision@k, and MRR at judgment level."""
    expected = set(expected_source_ids)
    top_results = results[:k]
    retrieved_ids = [result["source_id"] for result in top_results]
    # relevant_count needs to be computed from the intersection
    relevant_retrieved = sum(1 for source_id in retrieved_ids if source_id in expected)
    
    recall = relevant_retrieved / len(expected) if expected else 0.0
    precision = relevant_retrieved / k if k > 0 else 0.0

    reciprocal_rank = 0.0
    for rank, source_id in enumerate(retrieved_ids, start=1):
        if source_id in expected:
            reciprocal_rank = 1.0 / rank
            break

    return {
        "recall_at_k": recall,
        "precision_at_k": precision,
        "reciprocal_rank": reciprocal_rank,
        "relevant_retrieved": relevant_retrieved,
        "expected_relevant": len(expected),
    }

def timed_retrieval(retrieve, query, k):
    """Measure retrieval time, including query expansion when applicable."""
    start = perf_counter()
    results = retrieve(query, k)
    elapsed_ms = (perf_counter() - start) * 1000
    return results, elapsed_ms


def evaluate_retrieval(k=5):
    queries = load_test_queries()

    retrieval_methods = {
        "BM25": lambda query, top_k: bm25_search(
            query, top_k=top_k
        ),
        "KeyBERT + BM25": lambda query, top_k: bm25_search(
            expand_query(query), top_k=top_k
        ),
        "Semantic": lambda query, top_k: semantic_search(
            query, top_k=top_k
        ),
        "Hybrid": lambda query, top_k: hybrid_search(
            query, top_k=top_k
        ),
        "KeyBERT + Hybrid": lambda query, top_k: hybrid_search(
            expand_query(query), top_k=top_k
        ),
    }

    all_metrics = {name: [] for name in retrieval_methods}
    all_latencies = {name: [] for name in retrieval_methods}

    for item in queries:
        query = item["query"]
        expected_ids = item["expected_source_ids"]

        print("\n" + "=" * 80)
        print(f"Original query: {query}")
        print(f"Expected judgments: {expected_ids}")

        expanded_query = expand_query(query)
        print(f"Expanded query: {expanded_query}")

        for method_name, retrieve in retrieval_methods.items():
            results, elapsed_ms = timed_retrieval(
                retrieve, query, k
            )
            results = unique_judgments(results, k=k)

            metrics = calculate_metrics(
                results,
                expected_ids,
                k=k,
            )

            all_metrics[method_name].append(metrics)
            all_latencies[method_name].append(elapsed_ms)

            print(f"\n{method_name}")
            print(f"  Latency:        {elapsed_ms:.1f} ms")
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
                    f"chunk {result.get('chunk_index', 'Unknown')}"
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
        print(
            f"  Mean retrieval latency: "
            f"{mean(all_latencies[method_name]):.1f} ms"
        )


if __name__ == "__main__":
    evaluate_retrieval(k=5)
