from app.retrieval.index import load_chunks, build_index
from app.retrieval.tokenizer import tokenize


def search(query: str, top_k: int = 5) -> list[dict]:
    """Retrieve the top-ranked legal chunks for a query."""

    if not query or not query.strip():
        raise ValueError("Query cannot be empty")

    if top_k <= 0:
        raise ValueError("top_k must be positive")

    if not tokenize(query):
        return []

    chunks = load_chunks()

    if not chunks:
        return []

    bm25 = build_index(chunks)
    ranked = bm25.rank(query)

    results = []
    seen_sources = set()

    for index, score in ranked:
        result = chunks[index]

        if result["source_id"] in seen_sources:
            continue

        seen_sources.add(result["source_id"])

        item = result.copy()
        item["score"] = score
        results.append(item)

        if len(results) == top_k:
            break

    return results


if __name__ == "__main__":
    query = input("Enter your legal query: ")

    try:
        results = search(query, top_k=5)

        if not results:
            print("No matching chunks found.")
        else:
            for rank, result in enumerate(results, start=1):
                print(f"\n{'=' * 65}")
                print(f"Rank: {rank}")
                print(f"BM25 score: {result['score']:.4f}")
                print(f"Judgment: {result['title']}")
                print(f"Source ID: {result['source_id']}")
                print(f"Chunk index: {result['chunk_index']}")
                print(f"Court: {result['court']}")
                print(f"Date: {result['judgment_date']}")
                print(f"Source: {result['source_url']}")
                print("\nPassage:")
                print(result["chunk_text"][:1200])

    except ValueError as exc:
        print(f"Input error: {exc}")