
import csv
import json
import re
from collections import Counter
from pathlib import Path

from app.db.connection import get_connection


BASE_DIR = Path(__file__).resolve().parent

QUERY_PATH = BASE_DIR / "dataset" / "test_queries.json"
POOL_PATH = BASE_DIR / "results" / "candidate_pool.csv"
OUTPUT_PATH = BASE_DIR / "results" / "auto_labeled_results.csv"

# These must match the category names in test_queries.json.
VALID_CATEGORIES = {
    "bail_anticipatory_bail",
    "criminal_procedure_arrest",
    "constitutional_law",
    "pmla_money_laundering",
    "evidence_other_criminal_law",
}

# Fine-grained legal-topic tags mapped to broad evaluation categories.
TAG_RULES = {
    "bail_anticipatory_bail": [
        r"\bbail\b",
        r"anticipatory-bail",
        r"grant-of-bail",
        r"non-bailable-offence",
        r"apprehending-arrest",
    ],
    "criminal_procedure_arrest": [
        r"arrest",
        r"criminal-procedure-code",
        r"\bcrpc\b",
        r"\bbnss\b",
        r"investigation",
        r"police-custody",
        r"remand",
        r"first-information-report",
        r"\bfir\b",
        r"quashing",
        r"cognizance",
    ],
    "constitutional_law": [
        r"equality-before-law",
        r"protection-of-life-and-liberty",
        r"right-to-property",
        r"freedom-with-reasonable-restrictions",
        r"prohibition-of-discrimination",
        r"untouchability",
        r"power-to-issue-writ",
        r"fundamental-right",
        r"constitutional",
    ],
    "pmla_money_laundering": [
        r"\bpmla\b",
        r"prevention-of-money-laundering",
        r"money-laundering",
        r"proceeds-of-crime",
        r"scheduled-offence",
        r"attachment-of-property",
    ],
    "evidence_other_criminal_law": [
        r"admissibility",
        r"confession",
        r"evidence",
        r"accomplice-testimony",
        r"examine-accused",
        r"production-of-documents",
        r"indian-penal-code",
        r"\bipc\b",
        r"murder",
        r"theft",
        r"robbery",
        r"criminal-conspiracy",
        r"abetment",
        r"forgery",
        r"rioting",
        r"unlawful-assembly",
        r"rape",
        r"sexual-assault",
        r"dowry",
    ],
}

# More specific phrase rules for title/full-text fallback.
TEXT_RULES = {
    "bail_anticipatory_bail": [
        r"anticipatory bail",
        r"grant of bail",
        r"regular bail",
        r"bail application",
        r"bail under section 438",
        r"section 438 of the code of criminal procedure",
        r"section 482 of the bharatiya nagarik suraksha sanhita",
    ],
    "criminal_procedure_arrest": [
        r"arrest without warrant",
        r"illegal arrest",
        r"procedure of arrest",
        r"police custody",
        r"judicial custody",
        r"first information report",
        r"investigation under the code of criminal procedure",
        r"criminal procedure code",
        r"bharatiya nagarik suraksha sanhita",
        r"section 41 of the code of criminal procedure",
    ],
    "constitutional_law": [
        r"article 14 of the constitution",
        r"article 19 of the constitution",
        r"article 21 of the constitution",
        r"article 32 of the constitution",
        r"article 226 of the constitution",
        r"fundamental rights",
        r"right to life and personal liberty",
        r"equality before law",
        r"constitutional validity",
    ],
    "pmla_money_laundering": [
        r"prevention of money laundering act",
        r"\bpmla\b",
        r"proceeds of crime",
        r"scheduled offence under the pmla",
        r"attachment under the prevention of money laundering act",
        r"section 3 of the prevention of money laundering act",
        r"section 5 of the prevention of money laundering act",
    ],
    "evidence_other_criminal_law": [
        r"indian evidence act",
        r"bharatiya sakshya adhiniyam",
        r"admissibility of evidence",
        r"confession before police",
        r"confession made by the accused",
        r"burden of proof",
        r"beyond reasonable doubt",
        r"examination of witnesses",
        r"indian penal code",
        r"criminal conspiracy",
        r"offence of murder",
        r"offence of theft",
        r"offence of robbery",
    ],
}


def normalize(value):
    """Normalize text for consistent matching."""
    if value is None:
        return ""
    return re.sub(r"\s+", " ", str(value).lower()).strip()


def parse_categories(value):
    """Handle PostgreSQL JSONB values returned as lists or strings."""
    if value is None:
        return []

    if isinstance(value, list):
        return [normalize(item) for item in value]

    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            if isinstance(parsed, list):
                return [normalize(item) for item in parsed]
        except json.JSONDecodeError:
            return [normalize(value)]

    return []


def load_queries():
    with QUERY_PATH.open("r", encoding="utf-8") as file:
        data = json.load(file)

    queries = data if isinstance(data, list) else data.get("queries", [])

    if not queries:
        raise ValueError("No test queries found.")

    query_categories = {}

    for query in queries:
        query_id = query["query_id"]
        category = normalize(query.get("category", ""))

        if category not in VALID_CATEGORIES:
            raise ValueError(
                f"Invalid or missing category for {query_id}: {category}"
            )

        query_categories[query_id] = category

    return query_categories


def load_candidate_pool():
    if not POOL_PATH.exists():
        raise FileNotFoundError(f"Candidate pool not found: {POOL_PATH}")

    with POOL_PATH.open(
        "r", encoding="utf-8-sig", newline=""
    ) as file:
        rows = list(csv.DictReader(file))

    if not rows:
        raise ValueError("Candidate pool CSV is empty.")

    required = {"query_id", "source_id", "chunk_index"}
    missing = required - set(rows[0].keys())

    if missing:
        raise ValueError(f"Candidate pool is missing columns: {missing}")

    return rows


def load_documents():
    """Load document metadata and text from PostgreSQL."""
    conn = get_connection()

    try:
        rows = conn.execute("""
            SELECT source_id, title, categories, full_text
            FROM legal_documents
        """).fetchall()
    finally:
        conn.close()

    documents = {}

    for source_id, title, categories, full_text in rows:
        documents[str(source_id)] = {
            "title": title or "",
            "categories": parse_categories(categories),
            "full_text": full_text or "",
        }

    return documents


def classify_from_tags(tags):
    """Classify using existing granular metadata tags."""
    text = " ".join(tags)
    matched = set()
    evidence = []

    for category, patterns in TAG_RULES.items():
        for pattern in patterns:
            if re.search(pattern, text):
                matched.add(category)
                evidence.append(f"tag:{pattern}")

    return matched, evidence


def classify_from_text(title, full_text, minimum_hits):
    """
    Apply conservative keyword rules to title or full text.
    A category needs the configured number of distinct phrase matches.
    """
    matched = set()
    evidence = []

    for category, patterns in TEXT_RULES.items():
        hits = [
            pattern
            for pattern in patterns
            if re.search(pattern, title)
        ]

        if len(hits) < minimum_hits and full_text:
            hits.extend(
                pattern
                for pattern in patterns
                if pattern not in hits and re.search(pattern, full_text)
            )

        if len(hits) >= minimum_hits:
            matched.add(category)
            evidence.extend(
                f"text:{category}:{pattern}" for pattern in hits
            )

    return matched, evidence


def classify_document(document):
    """
    Priority:
    1. Existing tags.
    2. Title-based rules.
    3. Full-text fallback.
    4. Unknown.
    """
    tags = document["categories"]
    title = normalize(document["title"])
    full_text = normalize(document["full_text"])

    matched, evidence = classify_from_tags(tags)

    if matched:
        return sorted(matched), "metadata_tags", evidence

    # Title-only matching requires one distinctive phrase.
    matched, evidence = classify_from_text(title, "", minimum_hits=1)

    if matched:
        return sorted(matched), "title_rules", evidence

    # Full-text matching requires at least two distinct phrases.
    matched, evidence = classify_from_text(
        "", full_text, minimum_hits=2
    )

    if matched:
        return sorted(matched), "full_text_rules", evidence

    return [], "unknown", []


def main():
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    print("Loading test queries...")
    query_categories = load_queries()

    print("Loading candidate pool...")
    candidates = load_candidate_pool()

    print("Loading PostgreSQL documents...")
    documents = load_documents()

    print("Classifying documents...")
    classifications = {}

    for source_id, document in documents.items():
        classifications[source_id] = classify_document(document)

    output_rows = []
    label_counts = Counter()
    method_counts = Counter()
    unknown_documents = set()

    for row in candidates:
        query_id = row["query_id"]
        source_id = str(row["source_id"])

        if query_id not in query_categories:
            raise ValueError(f"Unknown query ID in candidate pool: {query_id}")

        target_category = query_categories[query_id]

        if source_id not in classifications:
            categories, method, evidence = [], "missing_document", []
        else:
            categories, method, evidence = classifications[source_id]

        if not categories:
            status = "unknown"
            grade = ""
            unknown_documents.add(source_id)
        elif target_category in categories:
            status = "relevant"
            grade = 2
        else:
            status = "not_relevant"
            grade = 0

        output = dict(row)
        output.update({
            "query_category": target_category,
            "auto_categories": "|".join(categories),
            "auto_relevance": grade,
            "label_status": status,
            "classification_method": method,
            "classification_evidence": " | ".join(evidence),
        })

        output_rows.append(output)
        label_counts[status] += 1
        method_counts[method] += 1

    fieldnames = list(output_rows[0].keys())

    with OUTPUT_PATH.open(
        "w", encoding="utf-8-sig", newline=""
    ) as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)

    print("\n" + "=" * 55)
    print("AUTOMATIC LABELING COMPLETE")
    print("=" * 55)
    print(f"Queries processed: {len(query_categories)}")
    print(f"Candidate records: {len(output_rows)}")
    print(f"Unique candidate documents: {len(set(r['source_id'] for r in candidates))}")
    print(f"Documents in database: {len(documents)}")
    print(f"Unknown candidate documents: {len(unknown_documents)}")

    print("\nLabel distribution:")
    for label, count in sorted(label_counts.items()):
        print(f"{label}: {count}")

    print("\nClassification method distribution:")
    for method, count in sorted(method_counts.items()):
        print(f"{method}: {count}")

    print(f"\nOutput: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
