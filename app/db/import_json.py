
import json
from pathlib import Path

from app.db.repository import upsert_document


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


def main():
    files = list(PROCESSED_DIR.glob("*.json"))

    imported = 0
    failed = 0

    for path in files:
        try:
            with path.open("r", encoding="utf-8") as file:
                document = json.load(file)

            if not document.get("text", "").strip():
                raise ValueError("Judgment text is empty")

            upsert_document(document)
            imported += 1

            print(f"Imported: {document['document_id']}")

        except Exception as exc:
            failed += 1
            print(f"Failed: {path.name}: {exc}")

    print(f"\nImported/updated: {imported}")
    print(f"Failed: {failed}")


if __name__ == "__main__":
    main()
