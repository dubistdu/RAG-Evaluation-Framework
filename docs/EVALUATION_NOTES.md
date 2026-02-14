# Evaluation notes

Runbook for the RAG evaluation pipeline: steps, commands, and references. **For comparison of all scores and metric definitions, see the [README](../README.md#evaluation-results-comparison-of-all-scores).**

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

**Results summary:** Best config is **recursive_t1024_o128** with **text-embedding-3-small**. Overlap 128 beat 64 and 256. Reranking improves Recall@1 and MRR. Full tables → README.

---

## Sanity check

- **Script:** `scripts/sanity_check_self_retrieval.py`
- **Result:** 20/20 chunks had themselves at rank #1 → **PASS**. Index and query use the same model; no embedding/similarity bug.

---

## Reranking (optional second stage)

Retrieve more candidates (e.g. 30), then rerank with a cross-encoder and take the new top-10. Metrics are computed on the reranked list. Requires `pip install sentence-transformers`.

```bash
python scripts/run_retrieval_evaluation.py --config recursive_t1024_o128 --rerank --retrieve-k 30 --rerank-top-k 10 --output data/eval/recursive_t1024_o128_metrics_rerank.json
```

Compare the output metrics to the same config without `--rerank` to see the gain.

---

## Next steps

- **Reranking:** Implemented; use `--rerank` as above. Compare metrics with and without.
- **Real questions:** Add 20–50 real user questions with gold chunks and run eval to see if gains hold in practice.
- **More docs:** Run full pipeline on additional PDFs to see if recursive_t1024_o128 stays best.

---

## Vector store (FAISS)

The vectorstore uses **FAISS** (`IndexFlatIP` with L2-normalized vectors for cosine similarity). Persisted as `index.faiss` + `metadata.json` per config. Install with: `pip install faiss-cpu`. Legacy directories containing only `vectors.npy` + `metadata.json` are still loaded (FAISS index is built in memory).

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
