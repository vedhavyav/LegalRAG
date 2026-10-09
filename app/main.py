import json
from pathlib import Path
from ingestion.indiankanoon import IndianKanoonClient


client = IndianKanoonClient()
'''
results = client.search(
    "anticipatory bail"
)

for doc in results["docs"]: 
    print("ID:", doc["tid"])
    print("Title: ", doc["title"])
    print("Source: ", doc["docsource"])
    print()
'''
    
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