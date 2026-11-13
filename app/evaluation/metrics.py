
import math


def calculate_metrics(rows, rank_column, k=5):
    """Calculate ranking metrics using pooled, metadata-derived labels."""

    ranked = []

    # Denominator is shared across all retrieval methods.
    total_relevant = sum(
        1 for row in rows
        if row.get("label_status") == "relevant"
    )

    known_candidates = sum(
        1 for row in rows
        if row.get("label_status") in ("relevant", "not_relevant")
    )

    for row in rows:
        value = row.get(rank_column, "")

        if value in (None, ""):
            continue

        try:
            rank = int(float(value))
        except (ValueError, TypeError):
            continue

        ranked.append({
            "rank": rank,
            "status": row.get("label_status", "unknown"),
            "relevant": row.get("label_status") == "relevant",
        })

    ranked.sort(key=lambda item: item["rank"])

    top_k = [item for item in ranked if item["rank"] <= k]

    judged_top_k = [
        item for item in top_k
        if item["status"] in ("relevant", "not_relevant")
    ]

    relevant_top_k = [
        item for item in judged_top_k if item["relevant"]
    ]

    # Precision among judged results in the top-k.
    precision = (
        len(relevant_top_k) / len(judged_top_k)
        if judged_top_k else 0.0
    )

    # Recall against all known relevant candidates in the shared pool.
    recall = (
        len(relevant_top_k) / total_relevant
        if total_relevant else 0.0
    )

    # Reciprocal rank of the first known relevant result.
    mrr = 0.0

    for item in top_k:
        if item["relevant"]:
            mrr = 1.0 / item["rank"]
            break

    # DCG: preserve original positions; unknowns contribute zero gain.
    dcg = sum(
        1.0 / math.log2(item["rank"] + 1)
        for item in top_k
        if item["relevant"]
    )

    # Ideal ranking uses all known relevant candidates in the pool.
    ideal_count = min(total_relevant, k)

    idcg = sum(
        1.0 / math.log2(position + 1)
        for position in range(1, ideal_count + 1)
    )

    ndcg = dcg / idcg if idcg else 0.0

    return {
        f"precision@{k}": precision,
        f"recall@{k}": recall,
        f"mrr@{k}": mrr,
        f"ndcg@{k}": ndcg,
        f"judged_coverage@{k}": len(judged_top_k) / k,
        "known_candidates": known_candidates,
        "relevant_candidates": total_relevant,
        f"retrieved@{k}": len(top_k),
    }
