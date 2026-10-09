from app.retrieval.bm25 import BM25
from app.retrieval.tokenizer import tokenize


def test_tokenization():
    assert tokenize("Section 438: ANTICIPATORY Bail") == [
        "section",
        "438",
        "anticipatory",
        "bail",
    ]


def test_relevant_document_ranks_first():
    corpus = [
        tokenize("anticipatory bail application under section 438"),
        tokenize("contract dispute and breach of agreement"),
        tokenize("criminal bail and custody proceedings"),
    ]

    bm25 = BM25(corpus)
    ranked = bm25.rank("anticipatory bail")

    assert ranked
    assert ranked[0][0] == 0


def test_unmatched_query_returns_no_results():
    corpus = [
        tokenize("anticipatory bail"),
        tokenize("contract dispute"),
    ]

    bm25 = BM25(corpus)

    assert bm25.rank("quantum mechanics") == []


def test_repeated_query_term_does_not_double_count():
    corpus = [
        tokenize("anticipatory bail"),
        tokenize("contract dispute"),
    ]

    bm25 = BM25(corpus)

    assert bm25.get_scores("bail") == bm25.get_scores(
        "bail bail"
    )


def test_invalid_parameters():
    corpus = [tokenize("legal judgment")]

    try:
        BM25(corpus, k1=-1)
    except ValueError:
        pass
    else:
        raise AssertionError("Expected invalid k1 to fail")

    try:
        BM25(corpus, b=1.5)
    except ValueError:
        pass
    else:
        raise AssertionError("Expected invalid b to fail")


if __name__ == "__main__":
    test_tokenization()
    test_relevant_document_ranks_first()
    test_unmatched_query_returns_no_results()
    test_repeated_query_term_does_not_double_count()
    test_invalid_parameters()

    print("All BM25 tests passed.")