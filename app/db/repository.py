import json
from app.db.connection import get_connection

def upsert_document(document: dict) -> None:
    """Insert a judgment or update it by source ID."""

    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO legal_documents (
                    source_id,
                    title,
                    court,
                    judgment_date,
                    source_url,
                    full_text,
                    num_cites,
                    num_cited_by,
                    categories,
                    related_queries,
                    metadata
                )
                VALUES (
                    %s, %s, %s, %s, %s, %s,
                    %s, %s, %s, %s, %s
                )
                ON CONFLICT (source_id)
                DO UPDATE SET
                    title = EXCLUDED.title,
                    court = EXCLUDED.court,
                    judgment_date = EXCLUDED.judgment_date,
                    source_url = EXCLUDED.source_url,
                    full_text = EXCLUDED.full_text,
                    num_cites = EXCLUDED.num_cites,
                    num_cited_by = EXCLUDED.num_cited_by,
                    categories = EXCLUDED.categories,
                    related_queries = EXCLUDED.related_queries,
                    metadata = EXCLUDED.metadata,
                    updated_at = NOW()
                """,
                (
                    document["document_id"],
                    document["title"],
                    document.get("court") or None,
                    document.get("date") or None,
                    document.get("source_url") or None,
                    document["text"],
                    document.get("num_cites", 0),
                    document.get("num_cited_by", 0),
                    json.dumps(document.get("categories", [])),
                    json.dumps(document.get("related_queries", [])),
                    json.dumps(document.get("metadata", {})),
                ),
            )
