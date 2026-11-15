import re
from functools import lru_cache

from keybert import KeyBERT
from sentence_transformers import SentenceTransformer

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Keep pseudo-relevance feedback bounded: the first few retrieved passages
# provide candidate legal terminology without sending the entire corpus to KeyBERT.
FEEDBACK_K = 10
MAX_FEEDBACK_CHARS = 12000


@lru_cache(maxsize=1)
def get_keyword_model():
    """Load the shared embedding model and KeyBERT model once per process."""
    embedding_model = SentenceTransformer(MODEL_NAME)
    return KeyBERT(model=embedding_model)


def _normalize_phrase(value: str) -> str:
    return " ".join(re.findall(r"\b\w+\b", value.casefold()))

def extract_keywords_from_passages(
    passages: list[dict],
    top_n: int = 5,
) -> list[tuple[str, float]]:
    """
    Extract candidate keyphrases from initially retrieved legal passages.

    Each passage is expected to contain a `chunk_text` field.
    """
    if top_n <= 0:
        raise ValueError("top_n must be positive")

    texts = []
    seen_texts = set()

    for passage in passages:
        text = (passage.get("chunk_text") or "").strip()
        if not text or text in seen_texts:
            continue
        seen_texts.add(text)
        texts.append(text)

    if not texts:
        return []

    feedback_document = "\n\n".join(texts)[:MAX_FEEDBACK_CHARS]
    if not feedback_document.strip():
        return []

    keyword_model = get_keyword_model()
    keywords = keyword_model.extract_keywords(
        feedback_document,
        keyphrase_ngram_range=(1, 3),
        stop_words="english",
        top_n=top_n,
    )

    return [(phrase, float(score)) for phrase, score in keywords]

def expand_query_from_passages(
    query: str,
    passages: list[dict],
    top_n: int = 5,
) -> tuple[str, list[tuple[str, float]]]:
    """
    Expand a query with keyphrases extracted from initial retrieval results.

    Returns the expanded query and the selected (phrase, score) pairs.
    """
    if not query or not query.strip():
        raise ValueError("Query cannot be empty")

    original = query.strip()
    original_terms = set(_normalize_phrase(original).split())

    candidates = extract_keywords_from_passages(
        passages,
        top_n=top_n,
    )

    selected = []
    seen_phrases = set()

    for phrase, score in candidates:
        normalized = _normalize_phrase(phrase)

        if not normalized or normalized in seen_phrases:
            continue

        phrase_terms = set(normalized.split())

        # Do not append a phrase made entirely of words already in the query.
        if phrase_terms.issubset(original_terms):
            continue

        seen_phrases.add(normalized)
        selected.append((phrase, score))

    if not selected:
        return original, []

    expanded = f"{original} {' '.join(phrase for phrase, _ in selected)}"
    return expanded, selected