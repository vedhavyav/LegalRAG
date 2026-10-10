from app.retrieval.search import search as bm25_search

from app.retrieval.semantic_search import (
    semantic_search as vector_search,
)


def reciprocal_rank_fusion(
    bm25_results: list[dict],
    semantic_results: list[dict],
    rrf_k: int = 60,
) -> list[dict]:
    """
    Combine BM25 and semantic rankings using RRF.

    A chunk is identified by (source_id, chunk_index).
    """

    if rrf_k <= 0:
        raise ValueError("rrf_k must be positive")

    fused = {}

    ranking_lists = [
        ("bm25_rank", bm25_results),
        ("semantic_rank", semantic_results),
    ]

    for rank_field, results in ranking_lists:
        for rank, result in enumerate(results, start=1):
            key = (
                result["source_id"],
                result["chunk_index"],
            )

            if key not in fused:
                fused[key] = result.copy()
                fused[key]["bm25_rank"] = None
                fused[key]["semantic_rank"] = None
                fused[key]["rrf_score"] = 0.0

            fused[key][rank_field] = rank
            fused[key]["rrf_score"] += 1.0 / (rrf_k + rank)

    print("\n--- BM25 candidates ---")
    for rank, result in enumerate(bm25_results[:10], start=1):
        print(rank, result["source_id"], result["chunk_index"])

    print("\n--- Semantic candidates ---")
    for rank, result in enumerate(semantic_results[:10], start=1):
            print(rank, result["source_id"], result["chunk_index"])

    print("\n--- Fused candidates ---")

    top_fused = sorted(
        fused.values(),
        key=lambda item: item["rrf_score"],
        reverse=True,
    )[:10]

    for rank, result in enumerate(top_fused, start=1):
        print(
            rank,
            result["source_id"],
            result["chunk_index"],
            round(result["rrf_score"], 6),
            result["bm25_rank"],
            result["semantic_rank"],
        )

    return sorted(
        fused.values(),
        key=lambda item: item["rrf_score"],
        reverse=True,
    )


def hybrid_search(
    query: str,
    top_k: int = 5,
    candidate_k: int = 30,
    rrf_k: int = 60,
) -> list[dict]:
    """Retrieve candidates from both systems and fuse their rankings."""

    if not query or not query.strip():
        raise ValueError("Query cannot be empty")

    if top_k <= 0:
        raise ValueError("top_k must be positive")

    if candidate_k <= 0:
        raise ValueError("candidate_k must be positive")

    bm25_results = bm25_search(
        query,
        top_k=candidate_k,
    )

    semantic_results = vector_search(
        query,
        top_k=candidate_k,
    )

    fused_results = reciprocal_rank_fusion(
        bm25_results,
        semantic_results,
        rrf_k=rrf_k,
    )

    return diversify_by_judgment(
        fused_results,
        top_k=top_k,
        max_chunks_per_source=1,
    )



def diversify_by_judgment(results, top_k=5, max_chunks_per_source=1):
    """
    Return ranked results while limiting repeated chunks
    from the same judgment.
    """
    if top_k <= 0:
        raise ValueError("top_k must be positive")

    if max_chunks_per_source <= 0:
        raise ValueError("max_chunks_per_source must be positive")

    diversified = []
    source_counts = {}

    for result in results:
        source_id = result["source_id"]
        count = source_counts.get(source_id, 0)

        if count >= max_chunks_per_source:
            continue

        diversified.append(result)
        source_counts[source_id] = count + 1

        if len(diversified) >= top_k:
            break

    return diversified


if __name__ == "__main__":
    query = input("Enter your legal query: ")

    try:
        results = hybrid_search(
            query,
            top_k=5,
            candidate_k=30,
        )

        if not results:
            print("No matching chunks found.")
        else:
            for rank, result in enumerate(results, start=1):
                print(f"\n{'=' * 70}")
                print(f"Hybrid rank: {rank}")
                print(f"RRF score: {result['rrf_score']:.6f}")
                print(f"BM25 rank: {result['bm25_rank']}")
                print(
                    f"Semantic rank: "
                    f"{result['semantic_rank']}"
                )
                print(f"Judgment: {result['title']}")
                print(f"Source ID: {result['source_id']}")
                print(f"Chunk index: {result['chunk_index']}")
                print(f"Court: {result['court']}")
                print(f"Date: {result['judgment_date']}")
                print(f"Source: {result['source_url']}")
                print("\nPassage:")
                print(result["chunk_text"][:1200])


    
    except (ValueError, FileNotFoundError) as exc:
        print(f"Search error: {exc}")