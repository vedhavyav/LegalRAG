
import csv
import json
from pathlib import Path

from app.retrieval.search import search as bm25_search
from app.retrieval.semantic_search import semantic_search


# --------------------------------------------------
# Configuration
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
QUERY_PATH = BASE_DIR / "dataset" / "test_queries.json"
OUTPUT_DIR = BASE_DIR / "results"
OUTPUT_PATH = OUTPUT_DIR / "candidate_pool.csv"

CANDIDATE_K = 30
RRF_K = 60


# --------------------------------------------------
# Utility functions
# --------------------------------------------------

def chunk_key(result):
    """Unique identity for a chunk."""
    return (
        str(result["source_id"]),
        int(result["chunk_index"])
    )


def reciprocal_rank_fusion(
    bm25_results,
    semantic_results,
    rrf_k=60
):
    """
    Combine BM25 and semantic rankings using RRF.
    """

    fused = {}

    result_lists = [
        ("bm25", bm25_results),
        ("semantic", semantic_results)
    ]

    for method, results in result_lists:
        for rank, result in enumerate(results, start=1):
            key = chunk_key(result)

            if key not in fused:
                fused[key] = {
                    "result": result,
                    "rrf_score": 0.0
                }

            fused[key]["rrf_score"] += (
                1.0 / (rrf_k + rank)
            )

    ranked = sorted(
        fused.values(),
        key=lambda item: item["rrf_score"],
        reverse=True
    )

    return ranked


def load_queries():
    with QUERY_PATH.open("r", encoding="utf-8") as file:
        data = json.load(file)

    # Supports either a direct list or an object containing queries.
    if isinstance(data, list):
        return data

    if isinstance(data, dict):
        return data.get("queries", [])

    raise ValueError("Unexpected test_queries.json format")


# --------------------------------------------------
# Candidate pooling
# --------------------------------------------------

def generate_candidate_pool():

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    queries = load_queries()

    if not queries:
        raise ValueError("No test queries found.")

    all_candidates = []

    print(f"Total test queries: {len(queries)}")
    print(f"Candidates per method: {CANDIDATE_K}")

    for query_number, query_item in enumerate(queries, start=1):

        query_id = query_item["query_id"]
        query_text = query_item["query"]
        category = query_item.get("category", "")

        print(
            f"\n[{query_number}/{len(queries)}] "
            f"Processing {query_id}: {query_text}"
        )

        # Retrieve candidates from both individual methods.
        bm25_results = bm25_search(
            query_text,
            top_k=CANDIDATE_K
        )

        semantic_results = semantic_search(
            query_text,
            top_k=CANDIDATE_K
        )

        # Build the hybrid ranking from the two result lists.
        hybrid_results = reciprocal_rank_fusion(
            bm25_results,
            semantic_results,
            rrf_k=RRF_K
        )[:CANDIDATE_K]

        # Store method ranks and scores for each unique chunk.
        candidates = {}

        for method, results in [
            ("bm25", bm25_results),
            ("semantic", semantic_results)
        ]:

            for rank, result in enumerate(results, start=1):

                key = chunk_key(result)

                if key not in candidates:
                    candidates[key] = {
                        "query_id": query_id,
                        "category": category,
                        "query": query_text,
                        "source_id": result["source_id"],
                        "chunk_index": result["chunk_index"],
                        "title": result.get("title", ""),
                        "court": result.get("court", ""),
                        "judgment_date": result.get("judgment_date", ""),
                        "source_url": result.get("source_url", ""),
                        "chunk_text": result.get("chunk_text", ""),
                        "bm25_rank": "",
                        "bm25_score": "",
                        "semantic_rank": "",
                        "semantic_score": "",
                        "hybrid_rank": "",
                        "hybrid_score": "",
                        "relevance_grade": ""
                    }

                candidates[key][f"{method}_rank"] = rank
                candidates[key][f"{method}_score"] = result.get(
                    "score", ""
                )

        for rank, item in enumerate(hybrid_results, start=1):

            result = item["result"]
            key = chunk_key(result)

            if key in candidates:
                candidates[key]["hybrid_rank"] = rank
                candidates[key]["hybrid_score"] = item["rrf_score"]

        all_candidates.extend(candidates.values())

        print(
            f"BM25: {len(bm25_results)} | "
            f"Semantic: {len(semantic_results)} | "
            f"Hybrid: {len(hybrid_results)} | "
            f"Unique pooled: {len(candidates)}"
        )

    # Export the complete candidate pool.
    if not all_candidates:
        raise ValueError("Candidate pool is empty.")

    fieldnames = list(all_candidates[0].keys())

    with OUTPUT_PATH.open(
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames
        )

        writer.writeheader()
        writer.writerows(all_candidates)

    print("\n-----------------------------------")
    print("CANDIDATE POOL GENERATION COMPLETE")
    print("-----------------------------------")
    print(f"Queries processed: {len(queries)}")
    print(f"Total unique query-chunk pairs: {len(all_candidates)}")
    print(f"Output: {OUTPUT_PATH}")


if __name__ == "__main__":
    generate_candidate_pool()
