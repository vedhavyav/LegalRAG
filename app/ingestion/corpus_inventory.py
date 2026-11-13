
"""
Discover and summarize the existing LegalRAG corpus.

Processed JSON files are the source of truth for collection inventory.
Only documents with valid collection metadata count toward category targets.
"""

import json
import logging
from collections import Counter
from pathlib import Path

from app.ingestion.collection_config import (
    CATEGORIES,
    EXCLUDED_DOCSOURCES,
)

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


def load_document(path: Path) -> dict | None:
    """Load one processed JSON document safely."""

    try:
        with path.open("r", encoding="utf-8") as file:
            return json.load(file)

    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Could not read %s: %s", path.name, exc)
        return None


def classify_court(court: str) -> str | None:
    """
    Convert the API's court source into a supported court type.

    Returns:
        supreme_court
        high_court
        None if the source is not a supported court.
    """

    court = (court or "").strip()

    if court in EXCLUDED_DOCSOURCES:
        return None

    if court == "Supreme Court of India":
        return "supreme_court"

    if court.endswith("High Court"):
        return "high_court"

    return None


def is_valid_judgment(document: dict) -> bool:
    """Check whether a processed document is eligible for the corpus."""

    document_id = document.get("document_id")
    text = document.get("text", "")
    court = document.get("court", "")

    if not document_id:
        return False

    if not isinstance(text, str) or not text.strip():
        return False

    if not classify_court(court):
        return False

    return True


def get_collection_category(document: dict) -> str | None:
    """Read and validate the primary category from collection metadata."""

    metadata = document.get("collection_metadata", {})

    if not isinstance(metadata, dict):
        return None

    category = metadata.get("primary_category")

    if category not in CATEGORIES:
        return None

    return category


def build_inventory() -> dict:
    """
    Scan all processed JSON files and return the corpus inventory.

    Unclassified documents are included in the existing-document ID set,
    but they do not count toward any category target.
    """

    inventory = {
        "total_valid_judgments": 0,
        "classified_judgments": 0,
        "unclassified_judgments": 0,
        "invalid_documents": 0,
        "court_counts": Counter(),
        "category_counts": Counter(),
        "bucket_counts": Counter(),
        "document_ids": set(),
        "classified_document_ids": set(),
        "unclassified_documents": [],
        "documents": [],
    }

    if not PROCESSED_DIR.exists():
        logger.warning("Processed directory does not exist: %s", PROCESSED_DIR)
        return inventory

    seen_ids = set()

    for path in sorted(PROCESSED_DIR.glob("*.json")):
        document = load_document(path)

        if document is None:
            inventory["invalid_documents"] += 1
            continue

        if not is_valid_judgment(document):
            inventory["invalid_documents"] += 1
            continue

        document_id = str(document["document_id"])

        if document_id in seen_ids:
            logger.warning("Duplicate document ID found: %s", document_id)
            continue

        seen_ids.add(document_id)
        inventory["document_ids"].add(document_id)
        inventory["total_valid_judgments"] += 1

        court_type = classify_court(document.get("court", ""))
        category = get_collection_category(document)

        inventory["court_counts"][court_type] += 1

        document_info = {
            "document_id": document_id,
            "title": document.get("title", ""),
            "court": document.get("court", ""),
            "court_type": court_type,
            "primary_category": category,
            "path": str(path),
        }

        inventory["documents"].append(document_info)

        if category:
            inventory["classified_judgments"] += 1
            inventory["classified_document_ids"].add(document_id)
            inventory["category_counts"][category] += 1
            inventory["bucket_counts"][(category, court_type)] += 1

        else:
            inventory["unclassified_judgments"] += 1
            inventory["unclassified_documents"].append(document_info)

    return inventory


def print_inventory(inventory: dict) -> None:
    """Print a readable summary of the existing corpus."""

    print("\n========== LEGALRAG CORPUS INVENTORY ==========")

    print(
        f"Total valid judgments : "
        f"{inventory['total_valid_judgments']}"
    )
    print(
        f"Classified judgments  : "
        f"{inventory['classified_judgments']}"
    )
    print(
        f"Unclassified judgments: "
        f"{inventory['unclassified_judgments']}"
    )
    print(
        f"Invalid documents    : "
        f"{inventory['invalid_documents']}"
    )

    print("\n--- Court Distribution ---")

    for court_type in ("supreme_court", "high_court"):
        print(
            f"{court_type:20} | "
            f"{inventory['court_counts'][court_type]}"
        )

    print("\n--- Category and Court Distribution ---")

    for category, details in CATEGORIES.items():
        print(f"\n{details['label']}")

        for court_type in ("supreme_court", "high_court"):
            count = inventory["bucket_counts"][(category, court_type)]

            print(f"  {court_type:20} | {count}")

    if inventory["unclassified_documents"]:
        print("\n--- Documents Requiring Classification ---")

        for document in inventory["unclassified_documents"]:
            print(
                f"{document['document_id']} | "
                f"{document['court']} | "
                f"{document['title']}"
            )

    print("===============================================\n")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    current_inventory = build_inventory()
    print_inventory(current_inventory)
