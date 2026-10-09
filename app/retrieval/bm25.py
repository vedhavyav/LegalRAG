import math
from collections import Counter

from app.retrieval.tokenizer import tokenize


class BM25:
    """
    A simple BM25 implementation for a collection of text chunks.

    Parameters:
        corpus: List of tokenized chunks.
        k1: Controls term-frequency saturation.
        b: Controls document-length normalization.
    """

    def __init__(
        self,
        corpus: list[list[str]],
        k1: float = 1.5,
        b: float = 0.75,
    ):
        if k1 <= 0:
            raise ValueError("k1 must be positive")

        if not 0 <= b <= 1:
            raise ValueError("b must be between 0 and 1")

        if not corpus:
            raise ValueError("Corpus cannot be empty")

        self.corpus = corpus
        self.k1 = k1
        self.b = b

        # Total number of chunks.
        self.N = len(corpus)

        # Count tokens within each chunk.
        self.term_frequencies = [
            Counter(document)
            for document in corpus
        ]

        # Number of tokens in each chunk.
        self.document_lengths = [
            len(document)
            for document in corpus
        ]

        # Average number of tokens per chunk.
        self.avgdl = (
            sum(self.document_lengths) / self.N
        )

        # Count how many chunks contain each term.
        self.document_frequencies = Counter()

        for document in self.term_frequencies:
            self.document_frequencies.update(document.keys())

        # Positive IDF variant of BM25.
        self.idf = {}

        for term, df in self.document_frequencies.items():
            self.idf[term] = math.log(
                1 + (self.N - df + 0.5) / (df + 0.5)
            )

    def score_document(
        self,
        query_tokens: list[str],
        document_index: int,
    ) -> float:
        """Calculate a BM25 score for one chunk."""

        if not 0 <= document_index < self.N:
            raise IndexError("Invalid document index")

        if not self.avgdl:
            return 0.0

        frequencies = self.term_frequencies[document_index]
        document_length = self.document_lengths[document_index]

        score = 0.0

        # Avoid counting a repeated query term multiple times.
        for term in dict.fromkeys(query_tokens):
            tf = frequencies.get(term, 0)

            # Terms absent from this chunk contribute nothing.
            if tf == 0:
                continue

            idf = self.idf.get(term, 0.0)

            length_normalization = (
                1 - self.b
                + self.b * document_length / self.avgdl
            )

            denominator = (
                tf + self.k1 * length_normalization
            )

            term_score = (
                idf
                * tf
                * (self.k1 + 1)
                / denominator
            )

            score += term_score

        return score

    def get_scores(self, query: str) -> list[float]:
        """Score every chunk against a raw text query."""

        query_tokens = tokenize(query)

        return [
            self.score_document(query_tokens, index)
            for index in range(self.N)
        ]

    def rank(self, query: str) -> list[tuple[int, float]]:
        """
        Return (document_index, score) pairs,
        sorted from highest to lowest score.
        """

        scores = self.get_scores(query)

        ranked = [
            (index, score)
            for index, score in enumerate(scores)
            if score > 0
        ]

        return sorted(
            ranked,
            key=lambda item: item[1],
            reverse=True,
        )