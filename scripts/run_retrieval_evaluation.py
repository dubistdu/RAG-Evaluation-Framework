"""
Run retrieval evaluation: embed each QA question, query vectorstore, compute Recall@K and MRR.

Usage (from project root):
  python scripts/run_retrieval_evaluation.py --config recursive_t512_o100
  python scripts/run_retrieval_evaluation.py --config recursive_t512_o100 --k 5 10
"""

import json
import logging
import os
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from rag.embeddings import OpenAIEmbeddingClient
from rag.evaluation.metrics import compute_metrics
from rag.retrieval import Reranker
from rag.vectorstore import VectorStore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger("run_retrieval_eval")


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser(
        description="Run retrieval evaluation: QA questions -> top-K chunks -> Recall@K, MRR"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        metavar="CONFIG_NAME",
        help="Chunk config name (e.g. recursive_t512_o100)",
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        default="text-embedding-3-small",
        help="Embedding model (must match vectorstore)",
    )
    parser.add_argument(
        "--qa-dir",
        type=Path,
        default=_PROJECT_ROOT / "data" / "synthetic_qa",
        help="Directory containing <config>_qa.json",
    )
    parser.add_argument(
        "--vectorstore-dir",
        type=Path,
        default=_PROJECT_ROOT / "data" / "vectorstores",
        help="Base directory for vectorstores",
    )
    parser.add_argument(
        "--k",
        type=int,
        nargs="+",
        default=[1, 5, 10],
        help="K values for Recall@K and Precision@K (default: 1 5 10)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional: write metrics JSON to this path",
    )
    parser.add_argument(
        "--max-questions",
        type=int,
        default=None,
        help="Optional: cap number of questions (for quick runs)",
    )
    parser.add_argument(
        "--rerank",
        action="store_true",
        help="Retrieve more, then rerank with cross-encoder; use --retrieve-k and --rerank-top-k",
    )
    parser.add_argument(
        "--retrieve-k",
        type=int,
        default=30,
        help="When --rerank: number of candidates to retrieve before reranking (default: 30)",
    )
    parser.add_argument(
        "--rerank-top-k",
        type=int,
        default=10,
        help="When --rerank: number of chunks to keep after reranking (default: 10); should be >= max --k",
    )
    parser.add_argument(
        "--reranker-model",
        type=str,
        default="cross-encoder/ms-marco-MiniLM-L-6-v2",
        help="Cross-encoder model for reranking (default: ms-marco-MiniLM-L-6-v2)",
    )
    args = parser.parse_args()

    qa_path = args.qa_dir / f"{args.config}_qa.json"
    if not qa_path.exists():
        logger.error("QA file not found: %s", qa_path)
        return 1

    store_path = args.vectorstore_dir / f"{args.config}_{args.embedding_model}"
    if not store_path.is_dir():
        logger.error("Vectorstore not found: %s", store_path)
        return 1

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        logger.error("Set OPENAI_API_KEY")
        return 1

    with open(qa_path, encoding="utf-8") as f:
        qa_entries = json.load(f)
    if args.max_questions is not None:
        qa_entries = qa_entries[: args.max_questions]
        logger.info("Capped to %s questions", len(qa_entries))

    store = VectorStore.load(store_path)
    embed_client = OpenAIEmbeddingClient(model=args.embedding_model, api_key=api_key)

    reranker = None
    if args.rerank:
        logger.info("Reranking enabled: retrieve-k=%s, rerank-top-k=%s", args.retrieve_k, args.rerank_top_k)
        reranker = Reranker(model_name=args.reranker_model)

    k_retrieve = args.retrieve_k if args.rerank else max(args.k)

    results = []
    batch_size = 100
    for i in range(0, len(qa_entries), batch_size):
        batch = qa_entries[i : i + batch_size]
        questions = [e["question"] for e in batch]
        query_vectors = embed_client.embed(questions)
        for e, qvec in zip(batch, query_vectors):
            gold = e.get("gold_chunk_ids") or []
            raw = store.query(qvec, k=k_retrieve)
            if reranker is not None:
                candidates = [(cid, text) for cid, text, _score in raw]
                retrieved = reranker.rerank(e["question"], candidates, top_k=args.rerank_top_k)
            else:
                retrieved = [cid for cid, _text, _score in raw]
            results.append((gold, retrieved))
        logger.info("Processed questions %s-%s", i + 1, min(i + batch_size, len(qa_entries)))

    metrics = compute_metrics(results, k_values=args.k)
    metrics["config"] = args.config
    metrics["embedding_model"] = args.embedding_model
    if args.rerank:
        metrics["rerank"] = True
        metrics["retrieve_k"] = args.retrieve_k
        metrics["rerank_top_k"] = args.rerank_top_k
        metrics["reranker_model"] = args.reranker_model

    print("\n--- Retrieval evaluation ---")
    print(
        f"Config: {args.config}  |  "
        f"Embedding: {args.embedding_model}  |  "
        f"N = {metrics['n_questions']}"
    )
    if args.rerank:
        print(f"Rerank: retrieve_k={args.retrieve_k}, rerank_top_k={args.rerank_top_k}")
    for k in args.k:
        recall_key = f"recall@{k}"
        prec_key = f"precision@{k}"
        print(f"  Recall@{k}:    {metrics[recall_key]:.4f}")
        print(f"  Precision@{k}: {metrics[prec_key]:.4f}")
    print(f"  MRR:          {metrics['mrr']:.4f}")
    print("---")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        logger.info("Wrote metrics to %s", args.output)

    return 0


if __name__ == "__main__":
    sys.exit(main())
