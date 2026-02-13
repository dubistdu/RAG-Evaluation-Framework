"""
Sanity check A: self-retrieval. For N random chunks, use chunk text as query;
the same chunk should be rank #1. If not, index/embedding/model mismatch.
"""
import os
import random
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from rag.embeddings import OpenAIEmbeddingClient
from rag.vectorstore import VectorStore


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser(description="Self-retrieval sanity check")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Chunk config name (e.g. recursive_t1024_o128)",
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        default="text-embedding-3-small",
        help="Embedding model (must match vectorstore)",
    )
    parser.add_argument(
        "--vectorstore-dir",
        type=Path,
        default=_PROJECT_ROOT / "data" / "vectorstores",
        help="Base directory for vectorstores",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=20,
        help="Number of random chunks to test",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    args = parser.parse_args()

    store_path = args.vectorstore_dir / f"{args.config}_{args.embedding_model}"
    if not store_path.is_dir():
        print(f"ERROR: Vectorstore not found: {store_path}")
        return 1

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: Set OPENAI_API_KEY")
        return 1

    store = VectorStore.load(store_path)
    embed_client = OpenAIEmbeddingClient(model=args.embedding_model, api_key=api_key)

    n_total = len(store)
    if n_total == 0:
        print("ERROR: Empty store")
        return 1
    n_sample = min(args.n, n_total)
    rng = random.Random(args.seed)
    indices = rng.sample(range(n_total), n_sample)

    rank1_ok = 0
    for i in indices:
        chunk_id = store._chunk_ids[i]
        text = store._texts[i]
        qvec = embed_client.embed([text])[0]
        results = store.query(qvec, k=5)
        if not results:
            print(f"  FAIL: no results for chunk_id={chunk_id}")
            continue
        top_id = results[0][0]
        if top_id == chunk_id:
            rank1_ok += 1
        else:
            print(f"  FAIL: chunk_id={chunk_id} -> top was {top_id}")

    print(f"\nSelf-retrieval check: {rank1_ok}/{n_sample} chunks had themselves at rank #1")
    if rank1_ok >= n_sample - 1:
        print("PASS (at most 1 failure is acceptable).")
    else:
        print("FAIL: index/query embedding mismatch or bug.")
    return 0 if rank1_ok >= n_sample - 1 else 1


if __name__ == "__main__":
    sys.exit(main())
