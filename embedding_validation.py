import json
import numpy as np
from pathlib import Path

base = Path("data/embeddings")

vectors = np.load(base / "embeddings.npy")

with open(base / "metadata.json", "r", encoding="utf-8") as f:
    metadata = json.load(f)

chunks = metadata["chunks"]

print("========== EMBEDDING VALIDATION ==========")
print("Vector shape:", vectors.shape)
print("Metadata chunk count:", len(chunks))
print("Declared chunk count:", metadata["chunk_count"])
print("Embedding dimension:", metadata["embedding_dimension"])
print("Model:", metadata["model_name"])
print("All finite:", np.isfinite(vectors).all())
print("All normalized:", np.allclose(
    np.linalg.norm(vectors, axis=1), 1.0, atol=1e-4
))
print("==========================================")