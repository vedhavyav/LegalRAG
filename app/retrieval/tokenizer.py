import re

def tokenize(text: str) -> list[str]:
    """
        Convert text into lowercase word and number tokens 
    """

    if not text:
        return []

    return re.findall(r"\b\w+\b", text.casefold())
