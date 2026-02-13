"""
Model context window & embedding constraints.
Enforces token limits for embedding and LLM models; no silent truncation.
"""

import logging
from pathlib import Path
from typing import Any, List, Optional, Union

import yaml
from pydantic import BaseModel, Field

from rag.chunking.tokenizer import TiktokenTokenizer
from rag.chunking.tokenizer import token_count as _token_count

# Tokenizer protocol: any object with encode(text) -> list of ids
TokenizerLike = Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Models (from config; no hardcoded limits)
# ---------------------------------------------------------------------------


class ModelSpec(BaseModel):
    """Spec for an embedding or LLM model: name, tokenizer, and token limits."""

    name: str = Field(..., description="Model identifier (e.g. text-embedding-3-small)")
    tokenizer_name: str = Field(..., description="Tokenizer to use (e.g. cl100k_base)")
    max_input_tokens: int = Field(..., gt=0, description="Max tokens accepted as input")
    max_output_tokens: Optional[int] = Field(None, description="Max output tokens (LLM only)")
    reserved_prompt_tokens: int = Field(0, ge=0, description="Reserved for system/instruction prompt")


class ContextBudget(BaseModel):
    """Safe token budget for retrieval + prompt assembly (LLM context)."""

    model_name: str = Field(..., description="LLM model name")
    max_context_tokens: int = Field(..., gt=0, description="Full context window size")
    reserved_system_tokens: int = Field(0, ge=0, description="Reserved for system prompt")
    reserved_user_tokens: int = Field(0, ge=0, description="Reserved for user message / query")
    reserved_output_tokens: int = Field(0, ge=0, description="Reserved for model output")

    @property
    def available_retrieval_tokens(self) -> int:
        """Tokens available for retrieved chunks (no overflow, no hidden truncation)."""
        return (
            self.max_context_tokens
            - self.reserved_system_tokens
            - self.reserved_user_tokens
            - self.reserved_output_tokens
        )


class ValidationResult(BaseModel):
    """Result of validating a chunk against an embedding model limit."""

    is_valid: bool = Field(..., description="True if within limit")
    token_count: int = Field(..., ge=0, description="Token count of the chunk")
    exceeds_by: Optional[int] = Field(None, ge=0, description="How many tokens over limit, if invalid")


# ---------------------------------------------------------------------------
# Tokenizer resolution (injectable; no hardcoded token counts)
# ---------------------------------------------------------------------------


def get_tokenizer_for_spec(spec: ModelSpec) -> TokenizerLike:
    """Return a tokenizer matching the model spec (e.g. cl100k_base -> tiktoken)."""
    name = (spec.tokenizer_name or "cl100k_base").strip().lower()
    tiktoken_models = {"cl100k_base": "cl100k_base", "cl100k": "cl100k_base", "o200k_base": "o200k_base"}
    if name in tiktoken_models:
        return TiktokenTokenizer(model=tiktoken_models[name])
    # Extensible: HF, SentencePiece, etc.
    raise ValueError(f"Unknown tokenizer_name for context constraints: {spec.tokenizer_name!r}")


def count_tokens(text: str, tokenizer: Optional[TokenizerLike] = None) -> int:
    """
    Count tokens for text using the given tokenizer.
    Uses Step 2 tokenizer; tokenizer must be injectable (e.g. from get_tokenizer_for_spec).
    """
    return _token_count(text, tokenizer)


# ---------------------------------------------------------------------------
# Validation (do NOT silently trim; log + flag)
# ---------------------------------------------------------------------------


def validate_chunk_for_embedding(
    chunk_text: str,
    model_spec: ModelSpec,
    tokenizer: Optional[TokenizerLike] = None,
) -> ValidationResult:
    """
    Check that a chunk does not exceed the embedding model token limit.
    If exceeded: log and set is_valid=False, exceeds_by=N. Does NOT trim.
    """
    tok = tokenizer if tokenizer is not None else get_tokenizer_for_spec(model_spec)
    n = count_tokens(chunk_text, tok)
    limit = model_spec.max_input_tokens
    if n > limit:
        excess = n - limit
        logger.warning(
            "Chunk exceeds embedding model limit: model=%s limit=%s token_count=%s exceeds_by=%s",
            model_spec.name,
            limit,
            n,
            excess,
        )
        return ValidationResult(is_valid=False, token_count=n, exceeds_by=excess)
    return ValidationResult(is_valid=True, token_count=n, exceeds_by=None)


def compute_safe_retrieval_k(
    chunks: List[Union[dict, Any]],
    context_budget: ContextBudget,
    tokenizer: Optional[TokenizerLike] = None,
) -> int:
    """
    Determine how many top-K chunks fit in the prompt without overflow.
    Chunks are sorted by token length; we add until budget is reached.
    Returns the safe K (number of chunks that fit).
    """
    if tokenizer is None:
        tokenizer = TiktokenTokenizer("cl100k_base")
    budget = context_budget.available_retrieval_tokens
    if budget <= 0:
        return 0

    def token_len(c: Union[dict, Any]) -> int:
        if hasattr(c, "token_count") and c.token_count is not None:
            return int(c.token_count)
        if isinstance(c, dict):
            if "token_count" in c:
                return int(c["token_count"])
            if "text" in c:
                return count_tokens(c["text"], tokenizer)
        if hasattr(c, "text"):
            return count_tokens(getattr(c, "text", ""), tokenizer)
        return 0

    lengths = [token_len(c) for c in chunks]
    lengths.sort()
    total = 0
    k = 0
    for L in lengths:
        if total + L <= budget:
            total += L
            k += 1
        else:
            break
    return k


def validate_prompt_assembly(
    system_prompt: str,
    user_query: str,
    chunks: List[Union[dict, Any]],
    context_budget: ContextBudget,
    tokenizer: Optional[TokenizerLike] = None,
) -> None:
    """
    Check that combined (system + user + chunks) tokens do not exceed model limit.
    Raises ValueError if overflow (no silent truncation).
    """
    if tokenizer is None:
        tokenizer = TiktokenTokenizer("cl100k_base")

    system_tokens = context_budget.reserved_system_tokens or count_tokens(system_prompt, tokenizer)
    user_tokens = context_budget.reserved_user_tokens or count_tokens(user_query, tokenizer)
    chunk_tokens = 0
    for c in chunks:
        if hasattr(c, "token_count") and c.token_count is not None:
            chunk_tokens += int(c.token_count)
        elif isinstance(c, dict):
            chunk_tokens += int(c.get("token_count", count_tokens(c.get("text", ""), tokenizer)))
        else:
            chunk_tokens += count_tokens(getattr(c, "text", ""), tokenizer)

    total = system_tokens + user_tokens + chunk_tokens
    max_allowed = context_budget.max_context_tokens - context_budget.reserved_output_tokens

    if total > max_allowed:
        raise ValueError(
            f"Prompt assembly overflow: total_tokens={total} max_allowed={max_allowed} "
            f"(system={system_tokens} user={user_tokens} chunks={chunk_tokens}); "
            "reduce chunks or reserved_output_tokens to avoid silent truncation."
        )


# ---------------------------------------------------------------------------
# Config loading (limits from YAML only)
# ---------------------------------------------------------------------------


def load_model_specs(path: Union[str, Path]) -> dict[str, Any]:
    """
    Load embedding_models and llm_models from configs/model_specs.yaml.
    Returns dict with keys 'embedding_models' (list of ModelSpec) and 'llm_models' (list of ModelSpec).
    """
    path = Path(path).resolve()
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    out: dict[str, Any] = {
        "embedding_models": [],
        "llm_models": [],
    }
    for raw in data.get("embedding_models") or []:
        out["embedding_models"].append(ModelSpec.model_validate(raw))
    for raw in data.get("llm_models") or []:
        out["llm_models"].append(ModelSpec.model_validate(raw))
    return out
