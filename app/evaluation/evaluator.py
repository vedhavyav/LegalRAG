
import csv
import json
from collections import defaultdict
from pathlib import Path

from app.evaluation.metrics import calculate_metrics


BASE_DIR = Path(__file__).resolve().parent
INPUT_PATH = BASE_DIR / "results" / "auto_labeled_results.csv"
OUTPUT_DIR = BASE_DIR / "results"

METHODS = {
    "BM25": "bm25_rank",
    "Semantic": "semantic_rank",
    "Hybrid": "hybrid_rank",
}

K = 5


def load_results():
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Automatic labels not found: {INPUT_PATH}"
        )

    with INPUT_PATH.open(
        "r", encoding="utf-8-sig", newline=""
    ) as file:
        rows = list(csv.DictReader(file))

    if not rows:
        raise ValueError("No evaluation records found.")

    required = {
        "query_id",
        "category",
        "label_status",
        "bm25_rank",
        "semantic_rank",
        "hybrid_rank",
    }

    missing = required - set(rows[0].keys())

    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    return rows


def mean(values):
    return sum(values) / len(values) if values else 0.0


def summarize(records):
    """Macro-average query-level metrics."""
    metric_keys = [
        f"precision@{K}",
        f"recall@{K}",
        f"mrr@{K}",
        f"ndcg@{K}",
        f"judged_coverage@{K}",
    ]

    summary = {}

    for key in metric_keys:
        summary[key] = mean([
            record[key] for record in records
        ])

    summary["query_count"] = len(records)

    return summary


def evaluate():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    rows = load_results()

    grouped = defaultdict(list)

    for row in rows:
        grouped[row["query_id"]].append(row)

    query_records = []
    category_records = defaultdict(
        lambda: defaultdict(list)
    )

    for query_id, query_rows in sorted(grouped.items()):
        category = query_rows[0]["category"]

        for method, rank_column in METHODS.items():
            metrics = calculate_metrics(
                query_rows,
                rank_column,
                k=K,
            )

            record = {
                "query_id": query_id,
                "category": category,
                "method": method,
                **metrics,
            }

            query_records.append(record)
            category_records[category][method].append(record)

    # Overall macro averages across queries.
    overall = {}

    for method in METHODS:
        method_records = [
            record for record in query_records
            if record["method"] == method
        ]

        overall[method] = summarize(method_records)

    # Category-level macro averages.
    category_summary = {}

    for category, method_data in sorted(category_records.items()):
        category_summary[category] = {
            method: summarize(records)
            for method, records in method_data.items()
        }

    # Save per-query results.
    query_csv = OUTPUT_DIR / "per_query_metrics.csv"

    fieldnames = list(query_records[0].keys())

    with query_csv.open(
        "w", encoding="utf-8-sig", newline=""
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(query_records)

    # Save complete summary.
    summary = {
        "evaluation_type": (
            "Weakly supervised category-level retrieval evaluation"
        ),
        "relevance_source": (
            "Automatically derived from document metadata and text rules"
        ),
        "k": K,
        "query_count": len(grouped),
        "candidate_record_count": len(rows),
        "methods": overall,
        "category_results": category_summary,
        "metric_interpretation": {
            "precision": (
                "Relevant results among judged top-k results"
            ),
            "recall": (
                "Relevant top-k results divided by all known "
                "relevant candidates in the shared candidate pool"
            ),
            "mrr": (
                "Reciprocal rank of the first relevant result "
                "within top-k"
            ),
            "ndcg": (
                "Binary-gain ranking quality using pooled "
                "category-derived labels"
            ),
            "judged_coverage": (
                "Fraction of top-k positions with known labels"
            ),
        },
        "limitations": [
            "Labels measure category alignment, not passage-level legal relevance.",
            "Recall is measured against the pooled candidates, not the complete corpus.",
            "Unknown labels are excluded from precision judgments and relevance counts.",
            "Automatic labels may contain classification errors.",
        ],
    }

    json_path = OUTPUT_DIR / "evaluation_summary.json"

    with json_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=4, ensure_ascii=False)

    print("\n" + "=" * 65)
    print("LEGALRAG RETRIEVAL EVALUATION COMPLETE")
    print("=" * 65)

    print(f"Queries evaluated: {len(grouped)}")
    print(f"Candidate records: {len(rows)}")
    print(f"Metrics: Precision@{K}, Recall@{K}, MRR@{K}, nDCG@{K}")

    for method, metrics in overall.items():
        print(f"\n{method}")

        for key, value in metrics.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value}")

    print("\nOutput files:")
    print(query_csv)
    print(json_path)

    return summary


if __name__ == "__main__":
    evaluate()
