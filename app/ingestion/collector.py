
"""
LegalRAG corpus collector.

Collects judgments until each category/court bucket reaches its target.
Uses the existing IndianKanoonClient and normalize_document implementation.
"""

import logging
import time

from datetime import datetime, timezone
from pathlib import Path

from app.ingestion.indiankanoon import IndianKanoonClient
from app.ingestion.parser import normalize_document
from app.ingestion.pipeline import save_json

from app.ingestion.collection_config import (
    CATEGORIES,
    COURT_TYPES,
    EXCLUDED_DOCSOURCES,
    MAX_PAGES_PER_QUERY,
    MAX_SEARCH_RESULTS_PER_PAGE,
    REQUEST_DELAY_SECONDS,
    TARGET_TOTAL,
)

from app.ingestion.corpus_inventory import (
    build_inventory,
    classify_court,
    is_valid_judgment,
    print_inventory,
)

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MANIFEST_DIR = PROJECT_ROOT / "data" / "manifests"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def validate_target(target_total: int) -> int:
    """
    Validate the requested corpus size.

    There are 5 categories x 2 court types = 10 collection buckets.
    The target must divide equally across all 10 buckets.
    """

    if target_total < 10:
        raise ValueError("Target must be at least 10.")

    if target_total % 10 != 0:
        raise ValueError(
            "Target must be divisible by 10 because the collector "
            "distributes judgments equally across 5 categories "
            "and 2 court types."
        )

    return target_total // 10


def save_manifest(manifest: dict) -> Path:
    """Save the current collection manifest."""

    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = MANIFEST_DIR / f"corpus_collection_{timestamp}.json"

    save_json(path, manifest)

    return path


def get_bucket_count(
    inventory: dict,
    category: str,
    court_type: str,
) -> int:
    """Return the current count for one category/court combination."""

    return inventory["bucket_counts"][(category, court_type)]


def fetch_and_save_document(
    client: IndianKanoonClient,
    search_result: dict,
    document_id: str,
    category: str,
    query: str,
    court_type: str,
) -> dict:
    """
    Fetch, normalize and save one judgment.

    The search query determines the provisional primary category.
    The full document's court source is checked again before saving.
    """

    document = client.get_document(document_id)

    if not isinstance(document, dict):
        raise ValueError("API returned an invalid document response.")

    actual_id = str(document.get("tid", document_id))

    if actual_id != document_id:
        raise ValueError(
            f"Document ID mismatch: requested {document_id}, "
            f"received {actual_id}"
        )

    actual_court = document.get("docsource", "")
    actual_court_type = classify_court(actual_court)

    if actual_court_type != court_type:
        raise ValueError(
            f"Court mismatch for {document_id}: "
            f"expected {court_type}, received {actual_court!r}"
        )

    if actual_court in EXCLUDED_DOCSOURCES:
        raise ValueError(
            f"Excluded document source: {actual_court}"
        )

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    raw_path = RAW_DIR / f"{document_id}.json"
    processed_path = PROCESSED_DIR / f"{document_id}.json"

    # Preserve the original API response.
    save_json(raw_path, document)

    normalized = normalize_document(
        document,
        search_result=search_result,
    )

    if not isinstance(normalized, dict):
        raise ValueError("Document normalization returned invalid data.")

    normalized["document_id"] = document_id
    normalized["collection_metadata"] = {
        "primary_category": category,
        "category_label": CATEGORIES[category]["label"],
        "collection_query": query,
        "court_type": court_type,
        "classification_method": "search_query",
        "collected_at": utc_now(),
    }

    if not is_valid_judgment(normalized):
        raise ValueError(
            "Normalized document is not a valid judgment "
            "or has empty text."
        )

    save_json(processed_path, normalized)

    return {
        "document_id": document_id,
        "title": normalized.get("title", ""),
        "court": normalized.get("court", ""),
        "category": category,
        "court_type": court_type,
        "text_length": len(normalized.get("text", "")),
        "status": "success",
    }


def collect_bucket(
    client: IndianKanoonClient,
    category: str,
    court_type: str,
    target_per_bucket: int,
    manifest: dict,
) -> None:
    """
    Fill one category/court bucket.

    Search results are paginated for each configured query.
    Existing document IDs are never downloaded twice.
    """

    inventory = build_inventory()
    current_count = get_bucket_count(inventory, category, court_type)

    if current_count >= target_per_bucket:
        logger.info(
            "Bucket already complete: %s / %s",
            category,
            court_type,
        )
        return

    queries = CATEGORIES[category]["queries"]
    seen_search_ids = set()

    logger.info(
        "Collecting %s | %s | current=%s | target=%s",
        category,
        court_type,
        current_count,
        target_per_bucket,
    )

    for query in queries:
        if current_count >= target_per_bucket:
            break

        logger.info("Searching: %s", query)

        for page in range(MAX_PAGES_PER_QUERY):
            if current_count >= target_per_bucket:
                break

            try:
                response = client.search(query, page=page)

                if not isinstance(response, dict):
                    raise ValueError("Invalid search response.")

                results = response.get("docs", [])

                if not isinstance(results, list) or not results:
                    logger.info(
                        "No more search results for query=%r page=%s",
                        query,
                        page,
                    )
                    break

                logger.info(
                    "Query=%r page=%s results=%s",
                    query,
                    page,
                    len(results),
                )

            except Exception as exc:
                logger.exception(
                    "Search failed for query=%r page=%s: %s",
                    query,
                    page,
                    exc,
                )

                manifest["errors"].append({
                    "stage": "search",
                    "query": query,
                    "page": page,
                    "error": str(exc),
                    "timestamp": utc_now(),
                })

                break

            for search_result in results[:MAX_SEARCH_RESULTS_PER_PAGE]:
                if current_count >= target_per_bucket:
                    break

                document_id = search_result.get("tid")

                if document_id is None:
                    continue

                document_id = str(document_id)

                if document_id in seen_search_ids:
                    continue

                seen_search_ids.add(document_id)

                # Search-result court source must match the requested bucket.
                result_court = search_result.get("docsource", "")
                result_court_type = classify_court(result_court)

                if result_court_type != court_type:
                    continue

                # Never download an already processed document again.
                processed_path = PROCESSED_DIR / f"{document_id}.json"

                if processed_path.exists():
                    logger.info(
                        "Skipping existing document: %s",
                        document_id,
                    )
                    continue

                try:
                    result = fetch_and_save_document(
                        client=client,
                        search_result=search_result,
                        document_id=document_id,
                        category=category,
                        query=query,
                        court_type=court_type,
                    )

                    manifest["results"].append(result)

                    # Rebuild inventory so counts reflect saved documents.
                    inventory = build_inventory()
                    current_count = get_bucket_count(
                        inventory,
                        category,
                        court_type,
                    )

                    logger.info(
                        "Collected %s | %s | %s/%s",
                        document_id,
                        category,
                        current_count,
                        target_per_bucket,
                    )

                except Exception as exc:
                    logger.exception(
                        "Failed to collect document %s: %s",
                        document_id,
                        exc,
                    )

                    manifest["results"].append({
                        "document_id": document_id,
                        "query": query,
                        "category": category,
                        "court_type": court_type,
                        "status": "failed",
                        "error": str(exc),
                    })

                time.sleep(REQUEST_DELAY_SECONDS)

        logger.info(
            "Finished query=%r for bucket=%s/%s",
            query,
            category,
            court_type,
        )

    inventory = build_inventory()
    current_count = get_bucket_count(inventory, category, court_type)

    if current_count < target_per_bucket:
        manifest["incomplete_buckets"].append({
            "category": category,
            "court_type": court_type,
            "current": current_count,
            "target": target_per_bucket,
            "remaining": target_per_bucket - current_count,
        })

        logger.warning(
            "Bucket incomplete: %s | %s | %s/%s",
            category,
            court_type,
            current_count,
            target_per_bucket,
        )


def collect_corpus(target_total: int = TARGET_TOTAL) -> dict:
    """Run the complete corpus collection process."""

    target_per_bucket = validate_target(target_total)

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)

    client = IndianKanoonClient()

    initial_inventory = build_inventory()

    manifest = {
        "target_total": target_total,
        "target_per_bucket": target_per_bucket,
        "started_at": utc_now(),
        "initial_inventory": {
            "total_valid_judgments":
                initial_inventory["total_valid_judgments"],
            "classified_judgments":
                initial_inventory["classified_judgments"],
            "unclassified_judgments":
                initial_inventory["unclassified_judgments"],
        },
        "results": [],
        "errors": [],
        "incomplete_buckets": [],
    }

    print_inventory(initial_inventory)

    logger.info("Starting corpus collection.")
    logger.info("Target total: %s", target_total)
    logger.info("Target per category/court bucket: %s", target_per_bucket)

    for category in CATEGORIES:
        for court_type in COURT_TYPES:
            collect_bucket(
                client=client,
                category=category,
                court_type=court_type,
                target_per_bucket=target_per_bucket,
                manifest=manifest,
            )

    final_inventory = build_inventory()

    manifest["finished_at"] = utc_now()
    manifest["final_inventory"] = {
        "total_valid_judgments":
            final_inventory["total_valid_judgments"],
        "classified_judgments":
            final_inventory["classified_judgments"],
        "unclassified_judgments":
            final_inventory["unclassified_judgments"],
        "court_counts": dict(final_inventory["court_counts"]),
        "category_counts": dict(final_inventory["category_counts"]),
        "bucket_counts": {
            f"{category}|{court_type}": count
            for (category, court_type), count
            in final_inventory["bucket_counts"].items()
        },
    }

    manifest["summary"] = {
        "successful_downloads": sum(
            item.get("status") == "success"
            for item in manifest["results"]
        ),
        "failed_downloads": sum(
            item.get("status") == "failed"
            for item in manifest["results"]
        ),
        "incomplete_buckets": len(manifest["incomplete_buckets"]),
        "target_reached": (
            final_inventory["classified_judgments"] >= target_total
            and not manifest["incomplete_buckets"]
        ),
    }

    manifest_path = save_manifest(manifest)

    print_inventory(final_inventory)

    print("\n========== COLLECTION SUMMARY ==========")
    print(manifest["summary"])
    print(f"Manifest saved: {manifest_path}")
    print("========================================\n")

    return manifest
