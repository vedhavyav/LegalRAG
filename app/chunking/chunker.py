def chunk_text(
    text: str,
    chunk_size: int = 1500,
    overlap: int = 200,
) -> list[dict]:
    """
    Split cleaned legal text into overlapping chunks.

    Character offsets refer to the original input text.
    """

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    if overlap < 0 or overlap >= chunk_size:
        raise ValueError(
            "overlap must be non-negative and smaller than chunk_size"
        )

    if not text or not text.strip():
        return []

    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = min(start + chunk_size, text_length)

        # Prefer a paragraph boundary when one is reasonably close.
        if end < text_length:
            boundary = text.rfind(
                "\n",
                start + chunk_size // 2,
                end,
            )

            if boundary > start:
                end = boundary

        raw_chunk = text[start:end]
        chunk = raw_chunk.strip()

        if chunk:
            leading_spaces = len(raw_chunk) - len(raw_chunk.lstrip())
            trailing_end = len(raw_chunk.rstrip())

            actual_start = start + leading_spaces
            actual_end = start + trailing_end

            chunks.append({
                "chunk_index": len(chunks),
                "chunk_text": chunk,
                "char_start": actual_start,
                "char_end": actual_end,
            })

        if end >= text_length:
            break

        # Move forward while retaining some previous context.
        next_start = end - overlap

        # Guarantee forward progress, even for unusual text.
        start = max(next_start, start + 1)

    return chunks