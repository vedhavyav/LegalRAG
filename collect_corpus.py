
"""
LegalRAG corpus collection entry point.

Usage:
    python collect_corpus.py
    python collect_corpus.py --target 100
    python collect_corpus.py --target 200
    python collect_corpus.py --inventory
"""

import argparse
import logging

from app.ingestion.collection_config import TARGET_TOTAL
from app.ingestion.collector import collect_corpus
from app.ingestion.corpus_inventory import (
    build_inventory,
    print_inventory,
)


def main():
    parser = argparse.ArgumentParser(
        description="Collect and expand the LegalRAG judgment corpus."
    )

    parser.add_argument(
        "--target",
        type=int,
        default=TARGET_TOTAL,
        help="Total target number of judgments. Must be divisible by 10.",
    )

    parser.add_argument(
        "--inventory",
        action="store_true",
        help="Display the existing corpus inventory without collecting.",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    if args.inventory:
        inventory = build_inventory()
        print_inventory(inventory)
        return

    collect_corpus(target_total=args.target)


if __name__ == "__main__":
    main()
