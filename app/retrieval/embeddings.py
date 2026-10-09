import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from app.retrieval.index import load_chunks


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

OUTPUT_DIR = Path("data/embeddings")
EMBEDDINGS_PATH = OUTPUT_DIR / "embeddings.npy"
METADATA_PATH = OUTPUT_DIR / "metadata.json"


def build_embedding_index():
    """Embed all stored chunks and save the local search index."""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    chunks = load_chunks()

    if not chunks:
        raise ValueError(
            "No chunks found. Complete Phase 4 first."
        )

    texts = [chunk["chunk_text"] for chunk in chunks]

    print(f"Loading model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)

    print(f"Generating embeddings for {len(texts)} chunks...")

    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    embeddings = np.asarray(embeddings, dtype=np.float32)

    # Save vectors separately from descriptive metadata.
    np.save(EMBEDDINGS_PATH, embeddings)

    metadata = {
        "model_name": MODEL_NAME,
        "embedding_dimension": int(embeddings.shape[1]),
        "chunk_count": len(chunks),
        "chunks": chunks,
    }

    with METADATA_PATH.open("w", encoding="utf-8") as file:
        json.dump(metadata, file, ensure_ascii=False)

    print("\nEmbedding index created successfully.")
    print(f"Chunks: {len(chunks)}")
    print(f"Vector shape: {embeddings.shape}")
    print(f"Saved vectors: {EMBEDDINGS_PATH}")
    print(f"Saved metadata: {METADATA_PATH}")


if __name__ == "__main__":
    build_embedding_index()