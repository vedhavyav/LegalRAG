from app.retrieval.hybrid_search import reciprocal_rank_fusion


def test_shared_chunk_gets_two_contributions():
    bm25_results = [
        {"source_id": "A", "chunk_index": 0, "title": "A"},
        {"source_id": "B", "chunk_index": 0, "title": "B"},
    ]

    semantic_results = [
        {"source_id": "B", "chunk_index": 0, "title": "B"},
        {"source_id": "A", "chunk_index": 0, "title": "A"},
    ]

    results = reciprocal_rank_fusion(
        bm25_results,
        semantic_results,
    )

    # Both chunks rank first in one system and second in the other.
    assert len(results) == 2
    assert results[0]["rrf_score"] == results[1]["rrf_score"]


def test_chunk_ranked_first_by_both_wins():
    bm25_results = [
        {"source_id": "A", "chunk_index": 0},
        {"source_id": "B", "chunk_index": 0},
    ]

    semantic_results = [
        {"source_id": "A", "chunk_index": 0},
        {"source_id": "C", "chunk_index": 0},
    ]

    results = reciprocal_rank_fusion(
        bm25_results,
        semantic_results,
    )

    assert results[0]["source_id"] == "A"
    assert results[0]["bm25_rank"] == 1
    assert results[0]["semantic_rank"] == 1


def test_duplicate_chunk_is_merged():
    results = reciprocal_rank_fusion(
        [
            {"source_id": "A", "chunk_index": 2},
            {"source_id": "A", "chunk_index": 2},
        ],
        [],
    )

    assert len(results) == 1


def test_invalid_rrf_k():
    try:
        reciprocal_rank_fusion([], [], rrf_k=0)
    except ValueError:
        pass
    else:
        raise AssertionError("Expected invalid rrf_k to fail")


if __name__ == "__main__":
    test_shared_chunk_gets_two_contributions()
    test_chunk_ranked_first_by_both_wins()
    test_duplicate_chunk_is_merged()
    test_invalid_rrf_k()

    print("All hybrid retrieval tests passed.")