"""
Build vectorstore from chunks for a given chunk config and embedding model.
Saves to data/vectorstores/<config_name>_<embedding_model>/

Usage (from project root):
  python scripts/build_vectorstore.py --config recursive_t512_o100
  python scripts/build_vectorstore.py --config recursive_t512_o100 --embedding-model text-embedding-3-small
"""

import logging
import os
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from rag.embeddings import OpenAIEmbeddingClient, embed_chunks
from rag.utils.context_constraints import load_model_specs
from rag.vectorstore import VectorStore

# Reuse chunk loading from generation module
from rag.generation.synthetic_qa_generator import load_chunks_for_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger("build_vectorstore")


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser(
        description="Build vectorstore from chunks (embed + persist)"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        metavar="CONFIG_NAME",
        help="Chunk config name (e.g. recursive_t512_o100)",
    )
    parser.add_argument(
        "--chunks-dir",
        type=Path,
        default=_PROJECT_ROOT / "data" / "chunks",
        help="Base directory for chunk configs (default: data/chunks)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_PROJECT_ROOT / "data" / "vectorstores",
        help="Base directory for vectorstores (default: data/vectorstores)",
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        default="text-embedding-3-small",
        help="Embedding model name (default: text-embedding-3-small)",
    )
    parser.add_argument(
        "--model-specs",
        type=Path,
        default=_PROJECT_ROOT / "configs" / "model_specs.yaml",
        help="Path to model_specs.yaml for token validation",
    )
    parser.add_argument(
        "--no-validate",
        action="store_true",
        help="Skip token-limit validation (not recommended)",
    )
    args = parser.parse_args()

    chunks = load_chunks_for_config(args.chunks_dir, args.config)
    if not chunks:
        logger.error("No chunks found for config %s in %s", args.config, args.chunks_dir)
        return 1

    specs = load_model_specs(args.model_specs)
    embedding_spec = None
    for spec in specs.get("embedding_models") or []:
        if spec.name == args.embedding_model:
            embedding_spec = spec
            break
    if not embedding_spec and not args.no_validate:
        logger.warning("Embedding model %s not in model_specs.yaml; skipping validation", args.embedding_model)

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        logger.error("Set OPENAI_API_KEY")
        return 1

    client = OpenAIEmbeddingClient(model=args.embedding_model, api_key=api_key)
    results = embed_chunks(
        chunks,
        client,
        model_spec=embedding_spec,
        validate=not args.no_validate,
        skip_invalid=True,
    )
    if not results:
        logger.error("No chunks embedded (all skipped or failed)")
        return 1

    store = VectorStore()
    store.add_many(
        [r.chunk_id for r in results],
        [r.text for r in results],
        [r.embedding for r in results],
    )

    out_name = f"{args.config}_{args.embedding_model}"
    out_path = args.output_dir / out_name
    store.save(out_path)
    logger.info("Done. Vectorstore: %s (%s chunks)", out_path, len(store))
    return 0


if __name__ == "__main__":
    sys.exit(main())
