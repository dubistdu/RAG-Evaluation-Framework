# Evaluation notes

Summary of the RAG evaluation pipeline, experiments run, and results.

---

## Pipeline (high level)

1. **Chunking** — Parsed docs → chunks per config (token-based, boundary-aware). No API calls.
2. **Synthetic QA** — For each chunk config: generate questions from chunks (LLM + optional embedding for multi-hop). Output: `data/synthetic_qa/<config>_qa.json` with `question`, `gold_chunk_ids`.
3. **Build vectorstore** — Embed all chunks for a config, save to `data/vectorstores/<config>_<embedding_model>/`.
4. **Retrieval evaluation** — For each question: embed question, query vectorstore (cosine similarity), compare retrieved chunk IDs to `gold_chunk_ids`. Compute Recall@K, Precision@K, MRR.
5. **Sanity check (optional)** — Self-retrieval: use chunk text as query; same chunk should be rank #1. Validates index/query alignment.

---

## Commands run

- Chunking (all configs):  
  `python scripts/run_chunking.py data/parsed_docs/ --config configs/chunking_configs.yaml`
- QA (per config, with limit for comparable N):  
  `python scripts/generate_synthetic_qa.py --config <CONFIG> --max-chunks 50 --batch-size 10`
- Vectorstore (per config):  
  `python scripts/build_vectorstore.py --config <CONFIG>` (default embedding: `text-embedding-3-small`)
- Retrieval eval (per config):  
  `python scripts/run_retrieval_evaluation.py --config <CONFIG> --output data/eval/<CONFIG>_metrics.json`
- Sanity check:  
  `python scripts/sanity_check_self_retrieval.py --config recursive_t1024_o128 --embedding-model text-embedding-3-small`

---

## Which config won

**Best:** **recursive_t1024_o128** (1024 tokens, 128 overlap) with **text-embedding-3-small**.

| Config              | Recall@1 | Recall@5 | Recall@10 | MRR   | N     |
|---------------------|----------|----------|-----------|-------|-------|
| recursive_t256_o50   | 0.26     | 0.57     | 0.73      | 0.38  | 130   |
| recursive_t512_o100  | 0.28     | 0.59     | 0.70      | 0.41  | 120   |
| **recursive_t1024_o128** | **0.31** | **0.61** | **0.77**  | **0.44** | 119 |

Higher is better for all metrics.

---

## Overlap experiments (1024 tokens)

Same chunk size (1024), different overlap:

| Config           | Recall@1 | Recall@5 | Recall@10 | MRR   |
|------------------|----------|----------|-----------|-------|
| t1024_o64        | 0.26     | 0.62     | 0.71      | 0.40  |
| **t1024_o128**   | **0.31** | **0.61** | **0.77**  | **0.44** |
| t1024_o256       | 0.20     | 0.52     | 0.71      | 0.34  |

**Conclusion:** 128 overlap is the sweet spot for this doc; less (64) or more (256) overlap hurt retrieval.

---

## Embedding model: small vs large

For **recursive_t1024_o128**:

| Model                    | Recall@1 | Recall@5 | Recall@10 | MRR   |
|--------------------------|----------|----------|-----------|-------|
| text-embedding-3-small   | **0.31** | **0.61** | **0.77**  | **0.44** |
| text-embedding-3-large   | 0.22     | 0.58     | 0.74      | 0.37  |

**Conclusion:** Small was better (and cheaper) for this corpus; no need to switch to large.

---

## Sanity check

- **Script:** `scripts/sanity_check_self_retrieval.py`
- **Result:** 20/20 chunks had themselves at rank #1 → **PASS**. Index and query use the same model; no embedding/similarity bug.

---

## Next steps

- **Reranking:** Retrieve top-30, rerank (e.g. cross-encoder), take new top-10, then run the same eval. Can improve Recall@1 and MRR without changing chunking or embedding model.
- **Real questions:** Add 20–50 real user questions with gold chunks and run eval to see if gains hold in practice.
- **More docs:** Run full pipeline on additional PDFs to see if recursive_t1024_o128 stays best.

---

## Paths reference

| Artifact    | Path |
|------------|------|
| Parsed docs | `data/parsed_docs/` |
| Chunks     | `data/chunks/<config_name>/<doc_id>.json` |
| QA         | `data/synthetic_qa/<config_name>_qa.json` |
| Vectorstores | `data/vectorstores/<config_name>_<embedding_model>/` |
| Eval metrics | `data/eval/<config_name>_metrics.json` |
| Chunk configs | `configs/chunking_configs.yaml` |
