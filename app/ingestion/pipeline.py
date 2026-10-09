import json
import logging
import re

from datetime import datetime, timezone
from pathlib import Path

from app.ingestion.parser import normalize_document


logger = logging.getLogger(__name__)

# Project root: LegalRAG/
PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MANIFEST_DIR = PROJECT_ROOT / "data" / "manifests"


def save_json(path: Path, data: dict) -> None:
    """Save a dictionary as UTF-8 JSON."""

    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        json.dump(
            data,
            file,
            ensure_ascii=False,
            indent=2,
        )


def load_json(path: Path) -> dict:
    """Load a previously saved JSON dictionary."""

    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def ingest_query(
    client,
    query: str,
    max_documents: int = 10,
    page: int = 0,
) -> dict:
    """Fetch, normalize, and save a small legal corpus."""

    if max_documents < 1:
        raise ValueError("max_documents must be at least 1")

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)

    # Step 1: Search Indian Kanoon.
    search_response = client.search(query, page=page)

    if not isinstance(search_response, dict):
        raise ValueError("Search API did not return a JSON object")

    results = search_response.get("docs", [])

    if not results:
        raise ValueError(
            f"No search results returned for query: {query!r}"
        )

    # Save the original search response for debugging.
    safe_query = re.sub(r"[^a-zA-Z0-9_-]+", "_", query).strip("_")
    search_path = (
        RAW_DIR / "searches" /
        f"{safe_query or 'query'}_page_{page}.json"
    )
    save_json(search_path, search_response)

    # Step 2: Collect unique document IDs.
    unique_results = []
    seen_ids = set()

    for result in results:
        document_id = result.get("tid")

        if document_id is None:
            continue

        document_id = str(document_id)

        if document_id in seen_ids:
            continue

        seen_ids.add(document_id)
        unique_results.append(result)

        if len(unique_results) >= max_documents:
            break

    if not unique_results:
        raise ValueError("Search results contained no valid document IDs")

    # Step 3: Process documents individually.
    manifest = {
        "query": query,
        "page": page,
        "requested_documents": max_documents,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "results": [],
    }

    for result in unique_results:
        document_id = str(result["tid"])
        processed_path = PROCESSED_DIR / f"{document_id}.json"
        raw_path = RAW_DIR / f"{document_id}.json"

        # Skip only if an existing file is valid and non-empty.
        if processed_path.exists():
            try:
                existing = load_json(processed_path)

                if (
                    existing.get("document_id") == document_id
                    and existing.get("text", "").strip()
                ):
                    manifest["results"].append({
                        "document_id": document_id,
                        "status": "skipped_existing",
                        "title": existing.get("title", ""),
                    })
                    continue
            except (OSError, json.JSONDecodeError):
                logger.warning(
                    "Existing processed file is invalid: %s",
                    processed_path,
                )

        try:
            # Step 4: Retrieve the complete judgment.
            document = client.get_document(document_id)

            if not isinstance(document, dict):
                raise ValueError(
                    "Document API did not return a JSON object"
                )

            # Step 5: Preserve the original API response.
            save_json(raw_path, document)

            # Step 6: Normalize the document.
            normalized = normalize_document(
                document,
                search_result=result,
            )

            # Step 7: Save the processed document.
            save_json(processed_path, normalized)

            manifest["results"].append({
                "document_id": document_id,
                "status": "success",
                "title": normalized["title"],
                "text_length": len(normalized["text"]),
            })

            logger.info(
                "Processed %s: %s",
                document_id,
                normalized["title"],
            )

        except Exception as exc:
            # One failed judgment must not stop the batch.
            logger.exception(
                "Failed to process document %s",
                document_id,
            )

            manifest["results"].append({
                "document_id": document_id,
                "status": "failed",
                "error": str(exc),
            })

    # Step 8: Save an ingestion report.
    manifest["finished_at"] = datetime.now(
        timezone.utc
    ).isoformat()

    manifest["summary"] = {
        "success": sum(
            r["status"] == "success"
            for r in manifest["results"]
        ),
        "skipped": sum(
            r["status"] == "skipped_existing"
            for r in manifest["results"]
        ),
        "failed": sum(
            r["status"] == "failed"
            for r in manifest["results"]
        ),
    }

    timestamp = datetime.now(timezone.utc).strftime(
        "%Y%m%dT%H%M%SZ"
    )

    manifest_path = (
        MANIFEST_DIR / f"ingestion_{timestamp}.json"
    )
    save_json(manifest_path, manifest)

    logger.info("Manifest saved to %s", manifest_path)

    return manifest
