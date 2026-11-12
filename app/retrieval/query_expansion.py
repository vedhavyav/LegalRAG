from sentence_transformers import SentenceTransformer
from keybert import KeyBERT
import re

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

embedding_model = SentenceTransformer(MODEL_NAME)
keyword_model = KeyBERT(model=embedding_model)


def extract_query_keywords(
        query: str, 
        top_n: int = 5,     
) -> list[tuple[str, float]]: 
    """Extract candidate keywords and phrases from a legal query"""
    if not query or not query.strip():
        raise ValueError("Query cannot be Empty")
    if top_n <= 0:
        raise ValueError("top_n must be positive")
    keywords =  keyword_model.extract_keywords(
        query, 
        keyphrase_ngram_range=(1,3),
        stop_words="english",
        top_n=top_n
    )
    return [(phrase, float(score)) for phrase, score in keywords]


def expand_query(query: str, top_n: int = 5) -> str:
    """Expand a query with phrases that are not already present."""

    if not query or not query.strip():
        raise ValueError("Query cannot be empty")

    original = query.strip()
    original_normalized = set(
        re.findall(r"\b\w+\b", original.casefold())
    )

    keywords = extract_query_keywords(original, top_n=top_n)
    added_phrases = []
    seen_phrases = set()

    for phrase, _score in keywords:
        normalized_phrase = " ".join(
            re.findall(r"\b\w+\b", phrase.casefold())
        )

        if not normalized_phrase or normalized_phrase in seen_phrases:
            continue

        phrase_terms = set(normalized_phrase.split())

        # Skip phrases containing only words already in the query.
        if phrase_terms.issubset(original_normalized):
            continue

        seen_phrases.add(normalized_phrase)
        added_phrases.append(phrase)

    if not added_phrases:
        return original

    return f"{original} {' '.join(added_phrases)}"
