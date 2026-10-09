from app.db.connection import get_connection
from app.retrieval.bm25 import BM25
from app.retrieval.tokenizer import tokenize


def load_chunks() -> list[dict]:
    """Load stored chunks together with their judgment metadata."""

    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT
                c.source_id,
                c.chunk_index,
                c.chunk_text,
                d.title,
                d.court,
                d.judgment_date,
                d.source_url
            FROM document_chunks AS c
            JOIN legal_documents AS d
                ON d.source_id = c.source_id
            WHERE c.chunk_text IS NOT NULL
              AND TRIM(c.chunk_text) <> ''
            ORDER BY c.source_id, c.chunk_index
            """
        ).fetchall()

    return [
        {
            "source_id": row[0],
            "chunk_index": row[1],
            "chunk_text": row[2],
            "title": row[3],
            "court": row[4],
            "judgment_date": row[5],
            "source_url": row[6],
        }
        for row in rows
    ]


def build_index(chunks: list[dict]) -> BM25:
    """Build an in-memory BM25 index from stored chunks."""

    tokenized_corpus = [
        tokenize(chunk["chunk_text"])
        for chunk in chunks
    ]

    return BM25(tokenized_corpus)