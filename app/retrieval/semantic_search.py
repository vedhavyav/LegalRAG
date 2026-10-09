import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from app.retrieval.embeddings import (
    MODEL_NAME,
    EMBEDDINGS_PATH,
    METADATA_PATH,
)


def semantic_search(
    query: str,
    top_k: int = 5,
) -> list[dict]:
    """Return the most semantically similar legal chunks."""

    if not query or not query.strip():
        raise ValueError("Query cannot be empty")

    if top_k <= 0:
        raise ValueError("top_k must be positive")

    if not EMBEDDINGS_PATH.exists() or not METADATA_PATH.exists():
        raise FileNotFoundError(
            "Embedding index not found. Run "
            "'python -m app.retrieval.embeddings' first."
        )

    embeddings = np.load(EMBEDDINGS_PATH)

    with METADATA_PATH.open("r", encoding="utf-8") as file:
        metadata = json.load(file)

    if metadata["model_name"] != MODEL_NAME:
        raise ValueError(
            "The index was created with a different model."
        )

    chunks = metadata["chunks"]

    if len(chunks) != len(embeddings):
        raise ValueError(
            "Embedding and metadata counts do not match."
        )

    model = SentenceTransformer(MODEL_NAME)

    query_embedding = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True,
    )[0]

    # Vectors are normalized, so their dot product equals
    # cosine similarity (up to floating-point precision).
    scores = embeddings @ query_embedding

    # Higher cosine similarity ranks first.
    ranked_indices = np.argsort(scores)[::-1][:top_k]

    results = []

    for index in ranked_indices:
        result = chunks[int(index)].copy()
        result["score"] = float(scores[index])
        result["retrieval_method"] = "cosine_similarity"
        results.append(result)

    return results


if __name__ == "__main__":
    query = input("Enter your legal query: ")

    try:
        results = semantic_search(query, top_k=5)

        if not results:
            print("No chunks are available.")
        else:
            for rank, result in enumerate(results, start=1):
                print(f"\n{'=' * 65}")
                print(f"Rank: {rank}")
                print(f"Cosine similarity: {result['score']:.4f}")
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