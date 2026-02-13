# RAG Evaluation System

Experimental retrieval optimization framework: parse lecture PDFs, chunk, embed, build vector DBs, generate synthetic QA, and evaluate retrieval (Recall@K, MRR@K, Precision@K).

**Step 1 (current):** PDF parsing only.

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
