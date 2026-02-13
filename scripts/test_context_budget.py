"""
Test context budget and embedding constraints (Step 3).
Usage (from project root):
  python scripts/test_context_budget.py
  python scripts/test_context_budget.py --config configs/model_specs.yaml
"""

import argparse
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from rag.utils.context_constraints import (
    ContextBudget,
    ModelSpec,
    compute_safe_retrieval_k,
    count_tokens,
    get_tokenizer_for_spec,
    load_model_specs,
    validate_chunk_for_embedding,
    validate_prompt_assembly,
)


def main():
    parser = argparse.ArgumentParser(description="Test context budget and validation (no hardcoded limits)")
    parser.add_argument(
        "--config",
        type=Path,
        default=_PROJECT_ROOT / "configs" / "model_specs.yaml",
        help="Path to model_specs.yaml",
    )
    args = parser.parse_args()

    if not args.config.exists():
        print(f"Config not found: {args.config}", file=sys.stderr)
        return 1

    specs = load_model_specs(args.config)
    emb = specs.get("embedding_models") or []
    llm = specs.get("llm_models") or []

    print("Loaded model specs (limits from YAML only):")
    for m in emb:
        print(f"  embedding: {m.name} max_input_tokens={m.max_input_tokens} tokenizer={m.tokenizer_name}")
    for m in llm:
        print(f"  llm:       {m.name} max_input_tokens={m.max_input_tokens} max_output_tokens={m.max_output_tokens}")

    tokenizer = get_tokenizer_for_spec(emb[0]) if emb else None
    sample = "This is a sample chunk for embedding. It has several tokens."
    n = count_tokens(sample, tokenizer)
    print(f"\ncount_tokens(sample, tokenizer) = {n}")

    if emb:
        spec = emb[0]
        result = validate_chunk_for_embedding(sample, spec, tokenizer)
        print(f"validate_chunk_for_embedding(sample, {spec.name}) -> valid={result.is_valid} token_count={result.token_count} exceeds_by={result.exceeds_by}")
        big = "word " * (spec.max_input_tokens + 100)
        result_big = validate_chunk_for_embedding(big, spec, tokenizer)
        print(f"validate_chunk_for_embedding(oversized) -> valid={result_big.is_valid} exceeds_by={result_big.exceeds_by}")

    # Context budget and safe K
    budget = ContextBudget(
        model_name="gpt-4o-mini",
        max_context_tokens=128000,
        reserved_system_tokens=500,
        reserved_user_tokens=200,
        reserved_output_tokens=2048,
    )
    print(f"\nContextBudget available_retrieval_tokens = {budget.available_retrieval_tokens}")

    # Chunks with token_count (e.g. from Step 2)
    chunks = [
        {"text": "Chunk one.", "token_count": 100},
        {"text": "Chunk two.", "token_count": 200},
        {"text": "Chunk three.", "token_count": 150},
        {"text": "Chunk four.", "token_count": 500},
    ]
    k = compute_safe_retrieval_k(chunks, budget, tokenizer)
    print(f"compute_safe_retrieval_k({len(chunks)} chunks, budget) = {k}")

    # Validate prompt assembly (must not overflow)
    system = "You are a helpful assistant."
    user = "What is in the document?"
    small_chunks = [{"text": "Doc content.", "token_count": 100}]
    try:
        validate_prompt_assembly(system, user, small_chunks, budget, tokenizer)
        print("validate_prompt_assembly(system, user, small_chunks, budget) -> OK")
    except ValueError as e:
        print(f"validate_prompt_assembly -> {e}")

    # Overflow should raise
    budget_tiny = ContextBudget(
        model_name="test",
        max_context_tokens=100,
        reserved_system_tokens=10,
        reserved_user_tokens=10,
        reserved_output_tokens=20,
    )
    try:
        validate_prompt_assembly(system, user, chunks, budget_tiny, tokenizer)
        print("validate_prompt_assembly(overflow) -> unexpected OK")
        return 1
    except ValueError as e:
        print(f"validate_prompt_assembly(overflow) -> ValueError (expected): {e}")

    print("\nAll context budget checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
