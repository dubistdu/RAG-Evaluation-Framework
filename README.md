# RAG Evaluation System

Experimental retrieval optimization framework: parse PDFs, chunk, embed, build vector stores (FAISS), generate synthetic QA, and evaluate retrieval (Recall@K, MRR, Precision@K). Supports optional reranking and multiple chunk/embedding configs.

---

## Evaluation results (comparison of all scores)

All runs use the same corpus and synthetic QA; embedding model is OpenAI unless noted. **Higher is better.**

### Chunk configs (embedding: text-embedding-3-small)

| Config | N | Recall@1 | Recall@5 | Recall@10 | MRR |
|--------|---|----------|----------|-----------|-----|
| recursive_t256_o50 | 130 | 0.262 | 0.569 | 0.731 | 0.385 |
| recursive_t512_o100 | 120 | 0.283 | 0.592 | 0.700 | 0.411 |
| **recursive_t1024_o128** | **119** | **0.311** | **0.613** | — | **0.419** |
| recursive_t1024_o64 | 130 | 0.262 | 0.615 | 0.708 | 0.404 |
| recursive_t1024_o256 | 140 | 0.200 | 0.521 | 0.707 | 0.344 |

**Best chunk config:** `recursive_t1024_o128` (1024 tokens, 128 overlap).

### Embedding model: small vs large (config: recursive_t1024_o128)

| Embedding model | N | Recall@1 | Recall@5 | Recall@10 | MRR |
|-----------------|---|----------|----------|-----------|-----|
| **text-embedding-3-small** | 119 | **0.311** | **0.580** | **0.739** | **0.419** |
| text-embedding-3-large | 119 | 0.218 | 0.580 | 0.739 | 0.370 |

**Best:** `text-embedding-3-small` (and cheaper).

### Reranking (config: recursive_t1024_o128, embedding: text-embedding-3-small)

Retrieve top-30 → rerank with cross-encoder (ms-marco-MiniLM-L-6-v2) → keep top-10.

| Setup | N | Recall@1 | Recall@3 | Recall@5 | MRR |
|-------|---|----------|----------|----------|-----|
| No rerank | 119 | 0.311 | 0.504 | 0.613 | 0.419 |
| **With rerank** | **119** | **0.605** | **0.773** | **0.815** | **0.694** |

Reranking significantly improves Recall@1 and MRR.

---

See **`docs/EVALUATION_NOTES.md`** and **`docs/QUICK_REFERENCE.md`** for pipeline steps and commands.

## Setup

```bash
cd RAG-Evaluation-Framework
pip install -r requirements.txt
```

## Project structure

```
RAG-Evaluation-Framework/
├── data/
│   ├── raw_pdfs/       # Place PDFs here
│   ├── parsed_docs/    # JSON output from parser
│   ├── chunks/
│   └── synthetic_qa/
├── rag/
│   ├── parsing/        # PDF → structured elements (Title, Paragraph, List, etc.)
│   ├── chunking/       # (future)
│   ├── embeddings/     # (future)
│   ├── vectorstore/    # (future)
│   ├── retrieval/     # (future)
│   ├── evaluation/     # (future)
│   └── generation/    # (future)
├── configs/
├── notebooks/
├── scripts/
└── main.py
```

## Step 1: Parse one PDF

From the `RAG-Evaluation-Framework` directory:

```bash
# Using the script (recommended)
python scripts/run_parse_one_pdf.py data/raw_pdfs/lecture.pdf

# Or with custom output dir
python scripts/run_parse_one_pdf.py path/to/file.pdf --output-dir data/parsed_docs

# Or via main.py
python main.py --parse-pdf data/raw_pdfs/lecture.pdf
```

Output is written to `data/parsed_docs/<filename>.json` with a Pydantic-shaped structure: `source_path` and a list of `elements`, each with `type` (Title, Paragraph, List, ListItem, Table, etc.), `text`, `page_number`, and `metadata`.

## Parsing module

- **`rag/parsing/schemas.py`** — Pydantic models: `ElementType`, `ParsedElement`, `ParsedDocument`.
- **`rag/parsing/pdf_parser.py`** — Uses the `unstructured` library to extract elements and return/save JSON.
