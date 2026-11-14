# LegalRAG

A Retrieval-Augmented Generation (RAG) system for retrieving and answering questions over Indian legal documents using hybrid search and a locally hosted Large Language Model (LLM).

LegalRAG combines **BM25 keyword retrieval**, **semantic vector retrieval**, and **Reciprocal Rank Fusion (RRF)** to identify relevant legal passages. The retrieved context is then supplied to a local LLM through Ollama to generate an answer.

The project focuses on building a practical legal retrieval pipeline and evaluating its retrieval performance.

---

## 1. Project Overview

Searching legal documents is challenging because legal questions may use different terminology from the language used in judgments. A keyword-only search can miss passages that are semantically relevant, while semantic search alone may overlook important legal terms and expressions.

LegalRAG addresses this problem through a hybrid retrieval approach.

### Project objectives

- Collect and organize Indian legal judgments.
- Extract and preprocess legal document text.
- Divide documents into overlapping chunks for retrieval.
- Generate dense vector embeddings for legal passages.
- Implement keyword-based retrieval using BM25.
- Implement semantic retrieval using Sentence Transformers.
- Combine retrieval rankings using Reciprocal Rank Fusion.
- Generate responses using a locally hosted LLM.
- Evaluate retrieval performance using a fixed test-query dataset.

### High-level workflow

```text
Indian Legal Documents
          |
          v
   Document Ingestion
          |
          v
  Text Preprocessing
          |
          v
   Document Chunking
          |
          +----------------------+
          |                      |
          v                      v
     BM25 Index           Embedding Generation
          |                      |
          |                 Vector Storage
          |                      |
          v                      v
    BM25 Retrieval       Semantic Retrieval
          |                      |
          +----------+-----------+
                     |
                     v
          Reciprocal Rank Fusion
                     |
                     v
           Final Retrieved Context
                     |
                     v
             Local LLM (Ollama)
                     |
                     v
              Generated Answer
```

---

## 2. Technology Stack

| Component | Technology |
|---|---|
| Programming language | Python |
| Legal document source | Indian Kanoon API |
| Relational database | PostgreSQL |
| Embedding model | `sentence-transformers/all-MiniLM-L6-v2` |
| Embedding dimension | 384 |
| Embedding storage | NumPy arrays and JSON metadata |
| Keyword retrieval | Custom BM25 implementation |
| Semantic retrieval | Sentence Transformers |
| Hybrid retrieval | Reciprocal Rank Fusion (RRF) |
| Experimental query expansion | KeyBERT |
| Local LLM runtime | Ollama |
| Observed generation model | `llama3.1:8b` |

---

## 3. Dataset

The project uses Indian legal documents collected through the Indian Kanoon API.

The target dataset contains 100 classified judgments, divided across five legal categories.

| Legal category | Supreme Court | High Courts | Total |
|---|---:|---:|---:|
| Bail and anticipatory bail | 10 | 10 | 20 |
| Criminal procedure and arrest | 10 | 10 | 20 |
| Constitutional law | 10 | 10 | 20 |
| PMLA and money laundering | 10 | 10 | 20 |
| Evidence and other criminal law | 10 | 10 | 20 |
| **Total** | **50** | **50** | **100** |

The collected corpus and database contain additional records beyond these 100 classified target judgments.

### Dataset and processing statistics

| Measurement | Count |
|---|---:|
| Classified target judgments | 100 |
| Valid judgments in corpus files | 109 |
| Records in PostgreSQL | 110 |
| Generated document chunks | 8,280 |
| Embedding dimensions | 384 |

The database record count should not be interpreted as the number of classified judgments. It also includes unclassified records.

---

## 4. System Architecture

The implementation is organized into the following major stages.

### 4.1 Document ingestion

Legal documents are collected using the Indian Kanoon API.

The ingestion process stores raw and processed document data separately.

- Raw document data: `data/raw`
- Processed document data: `data/processed`
- Ingestion manifests: `data/manifests`

### 4.2 Text preprocessing

The collected document text is prepared for downstream processing.

Preprocessing supports consistent document representation and prepares the extracted text for chunking, indexing, and embedding generation.

### 4.3 Document chunking

Long legal judgments are divided into smaller overlapping passages.

The current implementation uses paragraph-aware, character-based chunking.

| Parameter | Value |
|---|---:|
| Chunk size | 1,500 characters |
| Chunk overlap | 200 characters |

Overlap helps preserve contextual continuity between adjacent chunks.

The generated chunks retain document identifiers and chunk-level metadata so retrieved passages can be traced back to their source judgments.

### 4.4 Database storage

PostgreSQL stores the legal documents and their corresponding chunks.

The main database tables are:

**`legal_documents`**

Stores document-level information such as:

- Source ID
- Document title
- Court
- Judgment date
- Source URL
- Full text
- Citation information
- Categories
- Related queries
- Additional metadata

**`document_chunks`**

Stores chunk-level information such as:

- Chunk ID
- Source document ID
- Chunk index
- Chunk text
- Character start and end positions
- Chunk metadata

The relationship between the tables is maintained through the document source ID.

### 4.5 Embedding generation

Legal chunks are converted into dense vector representations using:

`sentence-transformers/all-MiniLM-L6-v2`

The model generates 384-dimensional embeddings.

The vectors are L2-normalized before storage.

Current embedding files:

```text
data/
└── embeddings/
    ├── embeddings.npy
    └── metadata.json
```

The NumPy file stores the embedding vectors, while the JSON metadata file maintains the corresponding chunk information.

**Note:** Embeddings are stored separately as NumPy arrays and JSON metadata. They are not stored in PostgreSQL using pgvector.

---

## 5. Retrieval System

LegalRAG uses a hybrid retrieval architecture combining lexical and semantic search.

### 5.1 BM25 retrieval

BM25 is used to retrieve passages based on keyword relevance.

The current custom implementation uses:

| Parameter | Value |
|---|---:|
| `k1` | 1.5 |
| `b` | 0.75 |

BM25 is useful when a query contains specific legal terminology, statutory references, or important keywords.

### 5.2 Semantic retrieval

Semantic retrieval identifies passages based on the similarity between the query embedding and document chunk embeddings.

The same Sentence Transformer model is used for query and document embeddings.

Since the vectors are L2-normalized, their dot product represents cosine similarity.

Semantic retrieval helps identify relevant passages even when the wording of a question differs from the wording in the judgment.

### 5.3 Hybrid retrieval using Reciprocal Rank Fusion

BM25 and semantic retrieval produce separate ranked result lists.

LegalRAG combines these rankings using Reciprocal Rank Fusion.

The RRF score is calculated as:

\[
RRF(d)=\sum_{r \in R}\frac{1}{k+rank_r(d)}
\]

Where:

- \(d\) represents a retrieved chunk.
- \(R\) represents the retrieval systems.
- \(rank_r(d)\) represents the rank of the chunk in a retrieval system.
- \(k\) is the RRF constant.

The implementation uses:

| Parameter | Value |
|---|---:|
| Candidate count (`candidate_k`) | 30 |
| Final result count (`top_k`) | 5 |
| RRF constant (`rrf_k`) | 60 |

The final retrieval stage also applies judgment-level diversification, limiting the final results to a maximum of one chunk per judgment.

This helps reduce repeated results from the same legal document.

---

## 6. Experimental Query Expansion: KeyBERT

KeyBERT was explored as an auxiliary query-expansion technique.

The implementation uses the same Sentence Transformer model to extract candidate keywords and keyphrases from a query.

Current extraction configuration:

| Parameter | Value |
|---|---|
| Embedding model | `all-MiniLM-L6-v2` |
| Keyphrase n-gram range | 1–3 |
| Stop words | English |
| Maximum candidate phrases | 5 |

The experimental implementation includes:

- Keyword and keyphrase extraction.
- Filtering of phrases already represented in the original query.
- Appending selected phrases to form an expanded query.

**Current limitation:** KeyBERT is not integrated into the main BM25, semantic, or hybrid retrieval pipeline. It has only been explored using a small number of test queries, and no reliable retrieval improvement has been established.

Therefore, the main retrieval evaluation focuses on BM25, semantic retrieval, and hybrid retrieval.

---

## 7. Answer Generation

LegalRAG uses Ollama to run a local Large Language Model.

The observed application execution uses:

```text
llama3.1:8b
```

The generation stage receives the user query and the retrieved legal context.

The intended response-generation process is:

```text
User Question
      |
      v
Hybrid Retrieval
      |
      v
Top 5 Legal Judgments
      |
      v
Context Preparation
      |
      v
Ollama LLM
      |
      v
Generated Legal Response
```

Using retrieved context provides the model with relevant legal material instead of relying entirely on its internal knowledge.

However, retrieval-augmented generation does not guarantee that every generated statement is legally correct or fully supported by the retrieved documents. Generated answers and citations require verification.

---

## 8. Evaluation

The project includes a retrieval evaluation workflow using a fixed dataset of 50 test queries.

The test dataset contains 10 queries for each of the five legal categories.

### 8.1 Evaluation pipeline

The evaluation process includes:

1. Loading the test queries.
2. Retrieving candidate chunks using BM25.
3. Retrieving candidate chunks using semantic search.
4. Combining rankings using RRF.
5. Constructing a candidate pool.
6. Assigning category-based relevance labels.
7. Calculating retrieval metrics.
8. Comparing the retrieval methods.

### 8.2 Candidate pool

The candidate pool is constructed from the union of BM25 and semantic retrieval results.

The evaluation produced:

| Measurement | Value |
|---|---:|
| Test queries | 50 |
| Candidate rows | 2,703 |
| Unique candidate judgments | 105 |
| Relevant candidate rows | 1,885 |
| Non-relevant candidate rows | 816 |
| Unknown candidate rows | 2 |

### 8.3 Relevance-labeling approach

The current evaluation uses automatically assigned category-based labels derived from document metadata and classification rules.

A candidate is marked relevant when its document category matches the query category.

This is a weakly supervised evaluation approach. It does not establish whether an individual passage directly answers the legal question.

The results should therefore be interpreted as category-alignment retrieval measurements rather than manually verified legal passage relevance.

### 8.4 Evaluation metrics

The main evaluation uses:

- **Precision@5:** Proportion of judged results in the top five that are relevant.
- **Recall@5:** Relevant results retrieved in the top five relative to the relevant candidates in the shared candidate pool.
- **Mean Reciprocal Rank (MRR@5):** Reciprocal rank of the first relevant result.
- **nDCG@5:** Ranking quality based on the position of relevant results.

Recall@5 is pool-relative and should not be interpreted as recall over the entire legal corpus.

### 8.5 Retrieval results

The following results were obtained from the current 50-query evaluation.

| Metric | BM25 | Semantic | Hybrid |
|---|---:|---:|---:|
| Precision@5 | 0.720 | 0.676 | 0.724 |
| Recall@5 | 0.095 | 0.091 | 0.098 |
| MRR@5 | 0.819 | 0.786 | 0.787 |
| nDCG@5 | 0.716 | 0.673 | 0.711 |

The hybrid method achieved the highest Precision@5 and pool-relative Recall@5 among the three methods. BM25 achieved the highest MRR@5 and nDCG@5.

The results indicate that hybrid retrieval provides a useful combination of lexical and semantic retrieval signals, but it does not outperform BM25 on every metric.

**Evaluation limitation:** These measurements depend on automatically assigned category labels. They should not be treated as proof of legal answer correctness, citation accuracy, or passage-level relevance.

---

## 9. End-to-End Application Testing

The application was also executed using representative legal questions from the five categories.

The tested queries covered:

- Anticipatory bail.
- Police custody and procedural safeguards.
- Article 21 and personal liberty.
- PMLA bail conditions.
- Principles governing circumstantial evidence.

For each query, the application generated BM25 and semantic candidate lists, combined the rankings, and returned five final judgments.

The observed executions demonstrate that the retrieval and generation pipeline runs end to end.

However, successful execution alone does not establish that every generated legal statement is correct. Answer faithfulness and citation accuracy require separate verification.

---

## 10. Running the Project

### 10.1 Prerequisites

- Python
- PostgreSQL
- Ollama
- Required Python dependencies
- Access to the configured legal document dataset and database

### 10.2 Create a virtual environment

**Windows PowerShell**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Verify the Python environment:

```powershell
python -c "import sys; print(sys.executable)"
```

Install the project dependencies:

```powershell
pip install -r requirements.txt
```

### 10.3 Verify PostgreSQL

Ensure that PostgreSQL is running and that the database connection settings used by the application are configured correctly.

### 10.4 Verify Ollama

Check the Ollama installation:

```bash
ollama --version
```

Check installed models:

```bash
ollama list
```

Run the observed application model:

```bash
ollama run llama3.1:8b
```

The model must be available locally, and the Ollama service must be accessible to the application.

### 10.5 Run the application

From the project root, execute:

```bash
python -m app.ask
```

The application accepts a legal question, performs retrieval, and generates a response using the local LLM.

### 10.6 Run document chunking

```bash
python -m app.db.chunks
```

The chunking process uses the configured chunk size and overlap.

### 10.7 Generate embeddings

```bash
python -m app.retrieval.embeddings
```

This generates the embedding array and corresponding metadata for the document chunks.

Run data preparation and indexing steps only when required by the current state of the database and embedding files.

---

## 11. Project Limitations

The current implementation has several limitations.

1. **Dataset size:** The corpus is limited compared with the full volume of Indian legal judgments.
2. **Chunking strategy:** The current character-based chunking configuration has not been established as optimal through a systematic chunk-size experiment.
3. **Embedding model:** The current MiniLM model provides compact embeddings but may not capture all domain-specific legal nuances.
4. **Evaluation labels:** Relevance labels are automatically assigned using category information rather than manual passage-level annotation.
5. **KeyBERT integration:** KeyBERT remains an experimental component and is not part of the main retrieval pipeline.
6. **Generation quality:** Generated legal answers may contain unsupported, incomplete, or inaccurate statements.
7. **Citation verification:** Retrieved source references do not automatically guarantee that every generated claim is supported by the cited passage.
8. **Evaluation scope:** The current quantitative evaluation focuses on retrieval performance rather than a complete human evaluation of legal answer quality.

---

## 12. Future Scope

Possible extensions include:

- Expanding the legal document collection.
- Improving legal-domain text preprocessing.
- Experimenting with alternative chunking strategies.
- Exploring legal-domain embedding models.
- Integrating and systematically evaluating query expansion.
- Developing manually annotated passage-level relevance judgments.
- Evaluating answer faithfulness and citation correctness.
- Improving generation prompts and context selection.
- Comparing retrieval configurations using reproducible experiments.

These are potential extensions and should not be interpreted as already implemented features.

---

## 13. Design Principles

### Evidence before generation

The system prioritizes retrieving relevant legal material before generating an answer.

### Hybrid retrieval

Combining lexical and semantic retrieval helps address different types of query-document matching.

### Reproducible evaluation

Retrieval methods should be compared using the same test-query dataset and evaluation configuration.

### Separation of components

Document processing, indexing, retrieval, and generation are treated as distinct stages so that failures can be investigated at the appropriate layer.

### Transparent limitations

Experimental components and evaluation limitations are documented rather than presented as completed improvements.

---

## 14. Security and Legal Disclaimer

LegalRAG is developed for academic, research, and educational purposes.

It is not a substitute for advice from a qualified legal professional.

Generated responses may be incomplete, inaccurate, or based on retrieved material that does not fully address the legal question. Users should independently verify legal provisions, judgments, and citations before relying on any output.

When handling legal documents, privacy, confidentiality, copyright, licensing, and applicable data-protection requirements should also be considered.

---

## 15. License

Add the project's license information once the repository's licensing decision has been finalized.