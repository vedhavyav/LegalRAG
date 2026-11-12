import json
from pathlib import Path

from app.retrieval.query_expansion import extract_query_keywords
from app.retrieval.search import search as bm25_search


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TEST_QUERIES_PATH = PROJECT_ROOT / "data" / "evaluation" / "test_queries.json"


def load_test_queries():
    with open(TEST_QUERIES_PATH, "r", encoding="utf-8") as file:
        return json.load(file)


def evaluate_keybert_and_bm25():
    test_queries = load_test_queries()

    for index, test_case in enumerate(test_queries, start=1):
        query = test_case["query"]

        print("\n" + "=" * 80)
        print(f"QUERY {index}: {query}")
        print("=" * 80)

        # Experiment A: KeyBERT keyword extraction
        print("\n[KEYBERT — EXTRACTED KEYWORDS]")

        keywords = extract_query_keywords(query, top_n=5)

        if keywords:
            for rank, (phrase, score) in enumerate(keywords, start=1):
                print(f"{rank}. {phrase} (score: {score:.4f})")
        else:
            print("No keywords extracted.")

        # Experiment B: Independent BM25 retrieval
        print("\n[BM25 — RETRIEVED CHUNKS]")

        results = bm25_search(query, top_k=5)

        if results:
            for rank, result in enumerate(results, start=1):
                print(f"\nRank: {rank}")
                print(f"Title: {result.get('title')}")
                print(f"Source ID: {result.get('source_id')}")
                print(f"Chunk Index: {result.get('chunk_index')}")
                print(f"BM25 Score: {result.get('score')}")
                print(f"Text: {result.get('chunk_text', '')[:300]}...")
        else:
            print("No BM25 results found.")


if __name__ == "__main__":
    evaluate_keybert_and_bm25()
