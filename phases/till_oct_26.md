
| Component | Current Implementation | Scaling Concern |
| :--- | :--- | :--- |
| **PostgreSQL** | Stores documents and chunks. | **Good foundation**; no immediate scaling concerns for raw storage. |
| **BM25** | Rebuilds an in-memory index per search. | **Repeated work on every query**; causes high CPU usage and latency spikes as the document count grows. |
| **Semantic search** | Loads `embeddings.npy` and computes similarities with NumPy. | **Loads the embedding matrix for each query**; high I/O overhead and slow matrix multiplication at scale. |
| **Embedding generation** | Re-embeds all loaded chunks in batches of 32. | **Unnecessary recomputation when the corpus changes**; highly inefficient and wastes expensive GPU/CPU resources. |
| **Chunking** | Deletes and recreates chunks for every stored document. | **Full-corpus reprocessing**; creates massive write amplification and locks the database unnecessarily. |
| **Query expansion** | KeyBERT-based expansion exists separately. | **Not integrated into the displayed hybrid-search path**; creates a fragmented pipeline and missed optimization opportunities. |
| **Hybrid search** | BM25 + semantic results combined with RRF. | **Useful foundation; needs evaluation and cleaner logging**; requires standardized benchmarking metrics and unified trace logs. |
