import logging

from app.ingestion.indiankanoon import IndianKanoonClient
from app.ingestion.pipeline import ingest_query


logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s",
)


def main():
    client = IndianKanoonClient()

    manifest = ingest_query(
        client=client,
        query="anticipatory bail",
        max_documents=10,
        page=0,
    )

    print("\n--- INGESTION SUMMARY ---")
    print(manifest["summary"])

    for result in manifest["results"]:
        print(
            result["document_id"],
            result["status"],
            result.get("title", ""),
        )


if __name__ == "__main__":
    main()


'''
results = client.search(
    "anticipatory bail"
)

for doc in results["docs"]: 
    print("ID:", doc["tid"])
    print("Title: ", doc["title"])
    print("Source: ", doc["docsource"])
    print()

document_id = "117859307"
document = client.get_document(document_id)
print(document.keys())

for key, value in document.items():
    print("\nKEY:", key)
    print("TYPE:", type(value))

    if isinstance(value, str):
        print("Length: ", len(value))
        print("Preview:", value[:500])
    else: 
        print("value: ", value)
'''