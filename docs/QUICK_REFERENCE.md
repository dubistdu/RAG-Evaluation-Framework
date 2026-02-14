# Quick reference — copy-paste commands

Run from **project root** with venv active and `OPENAI_API_KEY` set unless noted.  
Vectorstore uses **FAISS**; ensure `pip install faiss-cpu` (or see `requirements.txt`).

---

## 1. Chunking (all configs)

```bash
python scripts/run_chunking.py data/parsed_docs/ --config configs/chunking_configs.yaml
```

Output: `data/chunks/<config_name>/<doc_id>.json` for each config in the YAML.

---

## 2. Generate synthetic QA (per config)

Use `--max-chunks 50` and `--batch-size 10` for comparable N and to limit API usage.

```bash
# recursive_t256_o50
python scripts/generate_synthetic_qa.py --config recursive_t256_o50 --max-chunks 50 --batch-size 10

# recursive_t512_o100
python scripts/generate_synthetic_qa.py --config recursive_t512_o100 --max-chunks 50 --batch-size 10

# recursive_t1024_o128
python scripts/generate_synthetic_qa.py --config recursive_t1024_o128 --max-chunks 50 --batch-size 10

# recursive_t1024_o256
python scripts/generate_synthetic_qa.py --config recursive_t1024_o256 --max-chunks 50 --batch-size 10

# recursive_t1024_o64
python scripts/generate_synthetic_qa.py --config recursive_t1024_o64 --max-chunks 50 --batch-size 10
```

Output: `data/synthetic_qa/<config_name>_qa.json`.

---

## 3. Build vectorstore (per config)

Default embedding model: `text-embedding-3-small`. Add `--embedding-model text-embedding-3-large` to use large.

```bash
python scripts/build_vectorstore.py --config recursive_t256_o50
python scripts/build_vectorstore.py --config recursive_t512_o100
python scripts/build_vectorstore.py --config recursive_t1024_o128
python scripts/build_vectorstore.py --config recursive_t1024_o256
python scripts/build_vectorstore.py --config recursive_t1024_o64
```

Output: `data/vectorstores/<config_name>_<embedding_model>/`.

---

## 4. Retrieval evaluation (per config)

```bash
python scripts/run_retrieval_evaluation.py --config recursive_t256_o50 --output data/eval/recursive_t256_o50_metrics.json
python scripts/run_retrieval_evaluation.py --config recursive_t512_o100 --output data/eval/recursive_t512_o100_metrics.json
python scripts/run_retrieval_evaluation.py --config recursive_t1024_o128 --output data/eval/recursive_t1024_o128_metrics.json
python scripts/run_retrieval_evaluation.py --config recursive_t1024_o256 --output data/eval/recursive_t1024_o256_metrics.json
python scripts/run_retrieval_evaluation.py --config recursive_t1024_o64 --output data/eval/recursive_t1024_o64_metrics.json
```

With a different embedding model (must match the vectorstore):

```bash
python scripts/run_retrieval_evaluation.py --config recursive_t1024_o128 --embedding-model text-embedding-3-large --output data/eval/recursive_t1024_o128_metrics_large.json
```

Optional: `--max-questions 100` for a quick run. `--k 1 5 10 20` to change K values.

---

## 4b. Retrieval evaluation with reranking

Retrieve 30 candidates, rerank with a cross-encoder, keep top-10, then compute metrics. Requires `pip install sentence-transformers`.

```bash
python scripts/run_retrieval_evaluation.py --config recursive_t1024_o128 --rerank --retrieve-k 30 --rerank-top-k 10 --output data/eval/recursive_t1024_o128_metrics_rerank.json
```

Optional: `--reranker-model cross-encoder/ms-marco-MiniLM-L-6-v2` (default). Compare output to the same config without `--rerank` to measure improvement.

---

## 5. Sanity check (self-retrieval)

Use the **same** `--config` and `--embedding-model` as in your eval.

```bash
python scripts/sanity_check_self_retrieval.py --config recursive_t1024_o128 --embedding-model text-embedding-3-small
```

Optional: `--n 5` for a quicker run. Save output: append `| tee data/eval/sanity_check_result.txt`.

---

## Config names (from chunking_configs.yaml)

| Name                  | Chunk size | Overlap |
|-----------------------|------------|---------|
| recursive_t256_o50    | 256        | 50      |
| recursive_t512_o100   | 512        | 100     |
| recursive_t1024_o128  | 1024       | 128     |
| recursive_t1024_o256  | 1024       | 256     |
| recursive_t1024_o64   | 1024       | 64      |

Recommended: **recursive_t1024_o128** with **text-embedding-3-small**.
