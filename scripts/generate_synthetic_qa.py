"""
Generate synthetic questions for retrieval evaluation.
Usage (from project root):
  python scripts/generate_synthetic_qa.py --config recursive_t512_o100
  python scripts/generate_synthetic_qa.py --config recursive_t512_o100 --llm openai --embed
"""

import argparse
import logging
import os
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from rag.generation.synthetic_qa_generator import (
    QAConfig,
    OpenAIEmbeddingClient,
    OpenAILLMClient,
    OpenRouterLLMClient,
    generate_questions_for_config,
)

# Structured logging; no print statements. Silence httpx to avoid log spam per API call.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stdout,
)
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger("generate_synthetic_qa")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate synthetic questions from chunks for retrieval evaluation"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        metavar="CONFIG_NAME",
        help="Chunk config name (e.g. recursive_t512_o100). Chunks loaded from data/chunks/<config>/",
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
        default=_PROJECT_ROOT / "data" / "synthetic_qa",
        help="Output directory for QA JSON (default: data/synthetic_qa)",
    )
    parser.add_argument(
        "--llm",
        type=str,
        choices=("openai", "openrouter"),
        default="openai",
        help="LLM provider (default: openai)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o-mini",
        help="Model name (default: gpt-4o-mini)",
    )
    parser.add_argument(
        "--embed",
        action="store_true",
        help="Enable multi-hop questions (requires embedding API)",
    )
    parser.add_argument(
        "--embed-model",
        type=str,
        default="text-embedding-3-small",
        help="Embedding model for related-chunk search (default: text-embedding-3-small)",
    )
    parser.add_argument(
        "--questions-per-chunk-min",
        type=int,
        default=2,
        help="Min questions per chunk (default: 2)",
    )
    parser.add_argument(
        "--questions-per-chunk-max",
        type=int,
        default=3,
        help="Max questions per chunk (default: 3)",
    )
    parser.add_argument(
        "--multi-hop-ratio",
        type=float,
        default=0.1,
        help="Fraction of questions that are multi-hop (default: 0.1)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=5,
        help="Chunks per LLM call; higher = fewer API calls (default: 5)",
    )
    parser.add_argument(
        "--max-chunks",
        type=int,
        default=None,
        metavar="N",
        help="Cap chunks to process (e.g. 50 for testing); default = all",
    )
    args = parser.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("OPENROUTER_API_KEY")
    if not api_key and args.llm == "openrouter":
        api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        logger.error("Set OPENAI_API_KEY (or OPENROUTER_API_KEY for openrouter)")
        return 1

    if args.llm == "openai":
        llm = OpenAILLMClient(model=args.model, api_key=api_key)
    else:
        llm = OpenRouterLLMClient(model=args.model, api_key=api_key)

    embedding_client = None
    if args.embed:
        embedding_client = OpenAIEmbeddingClient(model=args.embed_model, api_key=os.environ.get("OPENAI_API_KEY"))

    qa_config = QAConfig(
        questions_per_chunk_min=args.questions_per_chunk_min,
        questions_per_chunk_max=args.questions_per_chunk_max,
        batch_size=args.batch_size,
        max_chunks=args.max_chunks,
        multi_hop_ratio=args.multi_hop_ratio,
    )

    out_path = generate_questions_for_config(
        config_name=args.config,
        chunks_dir=args.chunks_dir,
        output_dir=args.output_dir,
        qa_config=qa_config,
        llm=llm,
        embedding_client=embedding_client,
    )
    logger.info("Done. Output: %s", out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
