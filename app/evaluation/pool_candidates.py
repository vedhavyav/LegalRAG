import csv
import json
from pathlib import Path

from app.retrieval.search import search as bm25_search
from app.retrieval.semantic_search import semantic_search
from app.retrieval.query_expansion import (
    expand_query_from_passages,
    FEEDBACK_K,
)


# --------------------------------------------------
# Configuration
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
QUERY_PATH = BASE_DIR / "dataset" / "test_queries.json"
OUTPUT_DIR = BASE_DIR / "results"
OUTPUT_PATH = OUTPUT_DIR / "candidate_pool.csv"
EXPANSION_LOG_PATH = OUTPUT_DIR / "query_expansion_log.csv"

CANDIDATE_K = 30
RRF_K = 60
EXPANSION_TOP_N = 5


# --------------------------------------------------
# Utilities
# --------------------------------------------------

def chunk_key(result):
    """Unique identity for a chunk."""
    return (
        str(result["source_id"]),
        int(result["chunk_index"]),
    )


def reciprocal_rank_fusion(
    bm25_results,
    semantic_results,
    rrf_k=60,
):
    """Fuse BM25 and semantic rankings using Reciprocal Rank Fusion."""
    if rrf_k <= 0:
        raise ValueError("rrf_k must be positive")

    fused = {}

    for results in (bm25_results, semantic_results):
        for rank, result in enumerate(results, start=1):
            key = chunk_key(result)

            if key not in fused:
                fused[key] = {
                    "result": result,
                    "rrf_score": 0.0,
                }

            fused[key]["rrf_score"] += 1.0 / (rrf_k + rank)

    return sorted(
        fused.values(),
        key=lambda item: item["rrf_score"],
        reverse=True,
    )


def load_queries():
    with QUERY_PATH.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if isinstance(data, list):
        queries = data
    elif isinstance(data, dict):
        queries = data.get("queries", [])
    else:
        raise ValueError("Unexpected test_queries.json format")

    if not queries:
        raise ValueError("No test queries found.")

    return queries


def result_with_rrf_score(item):
    """Convert an RRF item to a normal result dictionary with its score."""
    result = item["result"].copy()
    result["score"] = item["rrf_score"]
    return result


def rank_results(results):
    """Return a chunk-key -> (rank, score) mapping."""
    ranked = {}
    for rank, result in enumerate(results, start=1):
        ranked[chunk_key(result)] = (rank, result.get("score", ""))
    return ranked


# --------------------------------------------------
# Candidate pooling
# --------------------------------------------------

def generate_candidate_pool():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    queries = load_queries()
    all_candidates = []
    expansion_logs = []

    method_names = (
        "bm25",
        "semantic",
        "hybrid",
        "keybert_bm25",
        "keybert_hybrid",
    )

    print(f"Total test queries: {len(queries)}")
    print(f"Candidates per method: {CANDIDATE_K}")
    print(f"KeyBERT feedback passages: {FEEDBACK_K}")
    print(f"KeyBERT expansion phrases: {EXPANSION_TOP_N}")

    for query_number, query_item in enumerate(queries, start=1):
        query_id = query_item["query_id"]
        query_text = query_item["query"]
        category = query_item.get("category", "")

        print(
            f"\n[{query_number}/{len(queries)}] "
            f"Processing {query_id}: {query_text}"
        )

        # Baseline retrieval.
        bm25_results = bm25_search(query_text, top_k=CANDIDATE_K)
        semantic_results = semantic_search(query_text, top_k=CANDIDATE_K)

        hybrid_items = reciprocal_rank_fusion(
            bm25_results,
            semantic_results,
            rrf_k=RRF_K,
        )[:CANDIDATE_K]
        hybrid_results = [
            result_with_rrf_score(item) for item in hybrid_items
        ]

        # KeyBERT + BM25: use initial BM25 passages as feedback.
        kb_bm25_feedback = bm25_results[:FEEDBACK_K]
        kb_bm25_query, kb_bm25_terms = expand_query_from_passages(
            query_text,
            kb_bm25_feedback,
            top_n=EXPANSION_TOP_N,
        )
        keybert_bm25_results = bm25_search(
            kb_bm25_query,
            top_k=CANDIDATE_K,
        )

        # KeyBERT + Hybrid: use initial hybrid passages as feedback.
        kb_hybrid_feedback = hybrid_results[:FEEDBACK_K]
        kb_hybrid_query, kb_hybrid_terms = expand_query_from_passages(
            query_text,
            kb_hybrid_feedback,
            top_n=EXPANSION_TOP_N,
        )

        kb_hybrid_bm25 = bm25_search(
            kb_hybrid_query,
            top_k=CANDIDATE_K,
        )
        kb_hybrid_semantic = semantic_search(
            kb_hybrid_query,
            top_k=CANDIDATE_K,
        )
        kb_hybrid_items = reciprocal_rank_fusion(
            kb_hybrid_bm25,
            kb_hybrid_semantic,
            rrf_k=RRF_K,
        )[:CANDIDATE_K]
        keybert_hybrid_results = [
            result_with_rrf_score(item) for item in kb_hybrid_items
        ]

        method_results = {
            "bm25": bm25_results,
            "semantic": semantic_results,
            "hybrid": hybrid_results,
            "keybert_bm25": keybert_bm25_results,
            "keybert_hybrid": keybert_hybrid_results,
        }

        expansion_logs.append({
            "query_id": query_id,
            "original_query": query_text,
            "keybert_bm25_terms": " | ".join(
                f"{phrase} ({score:.4f})"
                for phrase, score in kb_bm25_terms
            ),
            "keybert_bm25_expanded_query": kb_bm25_query,
            "keybert_hybrid_terms": " | ".join(
                f"{phrase} ({score:.4f})"
                for phrase, score in kb_hybrid_terms
            ),
            "keybert_hybrid_expanded_query": kb_hybrid_query,
        })

        # Union of all five methods creates one shared candidate pool.
        candidates = {}

        for method, results in method_results.items():
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
                        "keybert_bm25_rank": "",
                        "keybert_bm25_score": "",
                        "keybert_hybrid_rank": "",
                        "keybert_hybrid_score": "",
                        "keybert_bm25_expanded_query": kb_bm25_query,
                        "keybert_bm25_terms": " | ".join(
                            phrase for phrase, _ in kb_bm25_terms
                        ),
                        "keybert_hybrid_expanded_query": kb_hybrid_query,
                        "keybert_hybrid_terms": " | ".join(
                            phrase for phrase, _ in kb_hybrid_terms
                        ),
                        "relevance_grade": "",
                    }

                candidates[key][f"{method}_rank"] = rank
                candidates[key][f"{method}_score"] = result.get("score", "")

        all_candidates.extend(candidates.values())

        print(
            f"BM25: {len(bm25_results)} | "
            f"Semantic: {len(semantic_results)} | "
            f"Hybrid: {len(hybrid_results)} | "
            f"KeyBERT+BM25: {len(keybert_bm25_results)} | "
            f"KeyBERT+Hybrid: {len(keybert_hybrid_results)} | "
            f"Unique pooled: {len(candidates)}"
        )

    if not all_candidates:
        raise ValueError("Candidate pool is empty.")

    # Candidate pool.
    with OUTPUT_PATH.open(
        "w", newline="", encoding="utf-8-sig"
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(all_candidates[0].keys()),
        )
        writer.writeheader()
        writer.writerows(all_candidates)

    # One row per query documents the actual expansion terms.
    with EXPANSION_LOG_PATH.open(
        "w", newline="", encoding="utf-8-sig"
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(expansion_logs[0].keys()),
        )
        writer.writeheader()
        writer.writerows(expansion_logs)

    print("\n-----------------------------------")
    print("FIVE-METHOD CANDIDATE POOL COMPLETE")
    print("-----------------------------------")
    print(f"Queries processed: {len(queries)}")
    print(f"Total unique query-chunk pairs: {len(all_candidates)}")
    print(f"Candidate pool: {OUTPUT_PATH}")
    print(f"Expansion log: {EXPANSION_LOG_PATH}")


if __name__ == "__main__":
    generate_candidate_pool()
