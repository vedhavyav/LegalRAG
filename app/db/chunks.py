import json

from app.db.connection import get_connection
from app.chunking.chunker import chunk_text


def create_chunk_table():
    """Create the chunk table if it does not exist."""

    from pathlib import Path

    schema_path = Path(__file__).with_name("chunk_schema.sql")
    sql = schema_path.read_text(encoding="utf-8")

    with get_connection() as conn:
        conn.execute(sql)


def chunk_all_documents(
    chunk_size: int = 1500,
    overlap: int = 200,
):
    """Chunk every stored judgment and replace its existing chunks."""

    create_chunk_table()

    with get_connection() as conn:
        documents = conn.execute(
            """
            SELECT source_id, full_text
            FROM legal_documents
            WHERE full_text IS NOT NULL
              AND full_text <> ''
            ORDER BY source_id
            """
        ).fetchall()

        total_chunks = 0

        for source_id, full_text in documents:
            chunks = chunk_text(
                full_text,
                chunk_size=chunk_size,
                overlap=overlap,
            )

            # Re-running the pipeline replaces old chunks for this
            # document instead of accumulating duplicates.
            conn.execute(
                """
                DELETE FROM document_chunks
                WHERE source_id = %s
                """,
                (source_id,),
            )

            for chunk in chunks:
                metadata = {
                    "chunking_method": "paragraph_aware_character",
                    "chunk_size": chunk_size,
                    "overlap": overlap,
                }

                conn.execute(
                    """
                    INSERT INTO document_chunks (
                        source_id,
                        chunk_index,
                        chunk_text,
                        char_start,
                        char_end,
                        metadata
                    )
                    VALUES (
                        %s, %s, %s, %s, %s, %s::jsonb
                    )
                    """,
                    (
                        source_id,
                        chunk["chunk_index"],
                        chunk["chunk_text"],
                        chunk["char_start"],
                        chunk["char_end"],
                        json.dumps(metadata),
                    ),
                )

            total_chunks += len(chunks)

            print(
                f"Processed {source_id}: "
                f"{len(chunks)} chunks"
            )

    print(f"\nDocuments processed: {len(documents)}")
    print(f"Total chunks stored: {total_chunks}")


if __name__ == "__main__":
    chunk_all_documents()