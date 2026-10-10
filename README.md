# LegalRAG

A Retrieval-Augmented Generation (RAG) system for answering legal questions using a local LLM, with an emphasis on **grounded answers, retrieval quality, and evaluation**.

The project is being developed as a local-first LegalRAG pipeline. The current LLM runtime is **Ollama**, allowing the model to run locally instead of relying on a hosted API.

---

## Project Goals

LegalRAG is intended to:

- Retrieve relevant passages from a legal document collection.
- Use those passages as evidence for answering legal questions.
- Reduce unsupported or hallucinated answers.
- Evaluate both retrieval and final-answer quality.
- Make the evaluation process reproducible instead of relying only on subjective inspection.
- Keep the core workflow local and practical for modest hardware.

The central idea is:

```text
Legal Documents
      │
      ▼
Document Processing
      │
      ▼
Chunking / Indexing
      │
      ▼
Retriever
      │
      ▼
Relevant Legal Context
      │
      ▼
Local LLM (Ollama)
      │
      ▼
Grounded Answer
      │
      ▼
Evaluation
```

---

## Current Local LLM Setup

The project uses **Ollama** as the local model runtime.

### Recommended model

For the current hardware:

- **RAM:** 16 GB
- **Dedicated GPU:** None
- **Runtime:** Ollama

The selected model is:

```text
qwen2.5:7b
```

Run it with:

```bash
ollama run qwen2.5:7b
```

### Important: the model is downloaded only once

Running:

```bash
ollama run qwen2.5:7b
```

does **not** normally download the model again every time.

Ollama keeps downloaded models locally. If the model is already present, the command starts the existing model.

Check installed models with:

```bash
ollama list
```

If `qwen2.5:7b` appears in the list, it is already installed.

If Ollama reports that it is downloading layers again, verify that:

1. The Ollama installation being used is the expected one.
2. The model name is exactly the same.
3. The Ollama model storage has not been removed or changed.
4. You are not accidentally using a different Ollama environment or machine.

---

## Hardware Considerations

The current setup is intentionally designed around CPU-only execution.

With 16 GB RAM and no dedicated GPU:

- A 7B-class quantized model is a practical starting point.
- Larger models may provide better reasoning but can become significantly slower or more memory-intensive.
- Retrieval quality is especially important because the system should not depend on the LLM alone to find the relevant law.
- Keeping the retrieved context focused helps both answer quality and inference speed.

The project therefore treats **retrieval and evaluation as first-class components**, rather than simply increasing model size.

---

## High-Level Architecture

The intended pipeline consists of the following stages.

### 1. Document ingestion

Legal source documents are loaded into the system.

Typical processing may include:

- Reading source files.
- Extracting text.
- Cleaning unnecessary formatting.
- Preserving useful legal structure where possible.

### 2. Chunking

Documents are divided into smaller passages suitable for retrieval.

Chunking is important because excessively large chunks can dilute retrieval relevance, while excessively small chunks can remove the context needed to understand a legal provision.

### 3. Indexing

The processed chunks are indexed so that relevant passages can be retrieved for a user query.

### 4. Retrieval

For a legal question, the retriever identifies passages that are most relevant to the question.

Conceptually:

```text
Question
   │
   ▼
Query representation
   │
   ▼
Retriever
   │
   ▼
Top-k legal passages
```

### 5. Generation

The retrieved passages are supplied to the local Ollama model.

The model should answer using the retrieved legal evidence rather than relying solely on its internal knowledge.

### 6. Evaluation

The system is evaluated to determine whether it is actually improving.

Evaluation should consider at least:

- Whether the relevant evidence was retrieved.
- Whether the final answer is supported by the evidence.
- Whether the answer addresses the question.
- Whether the system introduces unsupported claims.
- Whether improvements to one component cause regressions elsewhere.

---

## Evaluation Philosophy

A major purpose of this project is to avoid repeatedly changing the system without knowing whether it actually became better.

The evaluation workflow should therefore follow:

```text
Baseline
   │
   ▼
Change one meaningful component
   │
   ▼
Run the same evaluation set
   │
   ▼
Compare metrics / failures
   │
   ▼
Keep or reject the change
   │
   ▼
Repeat
```

The goal is not to make the system appear better on a handful of manually selected examples.

The goal is to establish whether a change produces **measurable improvement on a consistent evaluation set**.

---

## Evaluation Layers

### Retrieval evaluation

Retrieval should be evaluated independently from generation.

Important questions include:

- Was the relevant document retrieved?
- Was the relevant passage retrieved?
- How high did the relevant passage rank?
- How often does the retriever fail before the LLM even receives the correct evidence?

Useful retrieval metrics can include:

- Recall@k
- Precision@k
- Hit Rate
- Mean Reciprocal Rank (MRR)

The exact metrics used by the project should match the available ground-truth evaluation data.

### Generation evaluation

The final answer should be evaluated for:

- **Correctness** — does it answer the legal question correctly?
- **Faithfulness / groundedness** — are claims supported by retrieved evidence?
- **Relevance** — does it stay focused on the question?
- **Completeness** — does it use the necessary evidence?
- **Citation / evidence quality** — where applicable, can claims be traced back to source material?

A high-quality RAG system needs both good retrieval and good generation.

```text
Good Retrieval + Good Generation = Useful LegalRAG

Bad Retrieval + Good Generation = Usually bad answer

Good Retrieval + Bad Generation = Evidence exists, but answer can still fail
```

---

## Development Phases

The project is being developed incrementally.

The later phases focus increasingly on **evaluation and measurable improvement**, rather than repeatedly rebuilding the same pipeline.

The current work includes the transition toward a more systematic evaluation phase and a local Ollama implementation.

When adding a new phase, document:

1. What problem the phase solves.
2. What changed.
3. What files/components changed.
4. How the change is tested.
5. What metric or behavior should improve.
6. Whether the result actually improved compared with the baseline.

---

## Project Structure

The exact directory structure may evolve during development. Keep this section synchronized with the actual repository.

A typical structure is:

```text
LegalRAG/
│
├── data/                 # Legal source documents and datasets
├── evaluation/           # Evaluation datasets, scripts and results
├── src/                  # Application / RAG implementation
│   ├── ingestion/
│   ├── retrieval/
│   ├── generation/
│   └── evaluation/
│
├── tests/                # Automated tests
│
├── .venv/                # Local Python virtual environment (not committed)
├── requirements.txt      # Python dependencies
├── README.md
└── ...
```

Do not commit `.venv/` or other machine-specific generated files to version control.

---

## Python Environment

Create and activate a virtual environment.

### Windows PowerShell

```powershell
python -m venv .venv
```

Activate it:

```powershell
.\.venv\Scripts\Activate.ps1
```

Verify the Python executable:

```powershell
python -c "import sys; print(sys.executable)"
```

The output should point to the project's virtual environment, for example:

```text
...\LegalRAG\.venv\Scripts\python.exe
```

Install dependencies when a `requirements.txt` file is available:

```powershell
pip install -r requirements.txt
```

---

## Ollama Setup

Install Ollama on the development machine and verify that it is available from the terminal:

```bash
ollama --version
```

Check available local models:

```bash
ollama list
```

Run the selected model:

```bash
ollama run qwen2.5:7b
```

For application code that communicates with Ollama, the Ollama service must be available to the application.

Before debugging the LegalRAG pipeline, verify the Ollama installation independently. This separates:

```text
Ollama problem
```

from:

```text
LegalRAG integration problem
```

---

## Typical Development Workflow

A disciplined workflow is:

```text
1. Activate .venv
2. Verify Python
3. Verify Ollama
4. Verify qwen2.5:7b is installed
5. Run the LegalRAG pipeline
6. Run the evaluation set
7. Inspect failures
8. Change one component
9. Re-run the same evaluation
10. Compare against the baseline
```

Avoid changing retrieval, prompting, chunking, model, and evaluation criteria simultaneously. Doing so makes it difficult to determine which change caused an improvement or regression.

---

## Troubleshooting

### `python` points to the wrong environment

Run:

```powershell
python -c "import sys; print(sys.executable)"
```

If it does not point to `.venv`, activate the virtual environment again:

```powershell
.\.venv\Scripts\Activate.ps1
```

---

### Ollama model appears to download again

First check:

```bash
ollama list
```

If the model is listed, run:

```bash
ollama run qwen2.5:7b
```

If the model is not listed, Ollama does not currently have that model available in its local model store.

Do not assume that every `ollama run` command performs a fresh download; normally Ollama reuses an already downloaded model.

---

### Generation is slow

CPU-only inference is expected to be slower than GPU inference.

Possible improvements include:

- Reduce unnecessary context.
- Retrieve fewer but more relevant chunks.
- Avoid excessively large prompts.
- Use an appropriate quantized model.
- Reduce unnecessary generation length.
- Profile retrieval and generation separately.

Do not immediately solve a speed problem by switching to a much larger model. First establish whether the model is actually the bottleneck.

---

### Answers are incorrect despite a capable model

Inspect retrieval before changing the LLM.

A common RAG failure pattern is:

```text
User question
     ↓
Wrong / incomplete evidence retrieved
     ↓
LLM receives bad context
     ↓
Incorrect answer
```

If the correct legal passage never reaches the model, improving the prompt alone cannot reliably fix the underlying retrieval failure.

---

## Reproducibility

Evaluation results should be reproducible.

Record, where relevant:

- Model name and version/tag.
- Retrieval configuration.
- Chunking configuration.
- Top-k value.
- Prompt version.
- Evaluation dataset version.
- Relevant software/dependency versions.
- Evaluation metrics.
- Date of evaluation.

A result without a known configuration is difficult to compare with future experiments.

---

## Design Principles

### 1. Evidence before confidence

The system should prefer an answer grounded in retrieved legal material over an impressive-sounding unsupported answer.

### 2. Retrieval is not a solved problem

A strong LLM does not compensate for consistently poor retrieval.

### 3. Evaluate changes

Every significant modification should have a measurable reason for being kept.

### 4. Keep experiments controlled

Change one major variable at a time whenever possible.

### 5. Separate failures

Distinguish:

```text
Data problem
    ↓
Chunking problem
    ↓
Retrieval problem
    ↓
Prompt/context problem
    ↓
Generation problem
    ↓
Evaluation problem
```

This prevents debugging the wrong layer.

### 6. Prefer reproducible improvements

A change that improves one example but hurts the evaluation set should not be treated as a successful improvement.

---

## Current Status

**Project:** LegalRAG  
**LLM Runtime:** Ollama  
**Current Model:** `qwen2.5:7b`  
**Target Hardware:** 16 GB RAM, CPU-only / no dedicated GPU  
**Development Focus:** RAG pipeline evaluation and measurable improvement

The project is moving from simply making the pipeline work toward determining **whether each component actually improves legal question-answering performance**.

---

## Security and Legal Disclaimer

This project is intended for research, development, and educational purposes.

It is not a substitute for advice from a qualified legal professional.

Legal information can be jurisdiction-specific, time-sensitive, and dependent on facts that may not be present in the retrieved documents. A generated answer should therefore be treated as an aid to research rather than as authoritative legal advice.

When using real legal documents, also consider:

- Privacy.
- Confidentiality.
- Copyright and licensing.
- Personally identifiable information.
- Data retention.
- Applicable legal and organizational requirements.

---

## Contributing / Development Notes

When making a change:

```text
Problem
  ↓
Hypothesis
  ↓
Implementation
  ↓
Evaluation
  ↓
Comparison with baseline
  ↓
Decision
```

Document the result even when the experiment fails.

A failed experiment is useful if it tells us which direction does **not** improve the system.

---

## License

Add the project's license here once the repository's licensing decision has been made.
