import re
from bs4 import BeautifulSoup


def clean_judgment_html(html: str) -> str:
    """Convert judgment HTML into readable text."""

    soup = BeautifulSoup(html or "", "html.parser")

    for element in soup(["script", "style"]):
        element.decompose()

    # Preserve structural boundaries.
    for element in soup.find_all([
        "h1", "h2", "h3", "h4",
        "p", "pre", "li", "tr"
    ]):
        element.insert_before("\n")
        element.insert_after("\n")

    text = soup.get_text(separator="", strip=False)

    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def normalize_document(
    document: dict,
    search_result: dict | None = None
) -> dict:
    """Convert an Indian Kanoon response into our schema."""

    search_result = search_result or {}

    document_id = document.get("tid")

    if document_id is None:
        document_id = search_result.get("tid")

    if document_id is None:
        raise ValueError("Document has no document ID (tid)")

    html = document.get("doc", "")
    text = clean_judgment_html(html)

    if not text:
        raise ValueError(
            f"Document {document_id} contains no usable text"
        )

    categories = [
        item["value"]
        for item in document.get("cats", [])
        if item.get("value")
    ]

    related_queries = [
        item["value"]
        for item in document.get("relatedqs", [])
        if item.get("value")
    ]

    return {
        "document_id": str(document_id),
        "title": (
            document.get("title")
            or search_result.get("title", "")
        ),
        "date": document.get("publishdate", ""),
        "court": document.get("docsource", ""),
        "text": text,
        "source_url": (
            f"https://indiankanoon.org/doc/{document_id}/"
        ),
        "num_cites": document.get("numcites", 0),
        "num_cited_by": document.get("numcitedby", 0),
        "categories": categories,
        "related_queries": related_queries,
        "metadata": {
            "division_type": document.get("divtype"),
            "court_copy": document.get("courtcopy"),
            "html_length": len(html),
            "text_length": len(text),
        },
    }
