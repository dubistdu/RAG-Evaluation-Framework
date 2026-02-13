"""
Embedding module: embed chunk text with model-aligned token validation.
"""

import logging
from pathlib import Path
from typing import Any, List, Optional, Protocol

from pydantic import BaseModel, Field

from rag.utils.context_constraints import (
    ModelSpec,
    ValidationResult,
    get_tokenizer_for_spec,
    load_model_specs,
    validate_chunk_for_embedding,
)

logger = logging.getLogger(__name__)


class EmbeddingResult(BaseModel):
    """One chunk's embedding with optional validation info."""

    chunk_id: str = Field(..., description="Chunk identifier")
    text: str = Field(..., description="Chunk text")
    embedding: List[float] = Field(..., description="Vector")
    token_count: Optional[int] = Field(None, description="Token count if validated")
    validation: Optional[ValidationResult] = Field(None, description="Validation result if checked")


class EmbeddingClient(Protocol):
    """Protocol: embed(texts) -> list of vectors."""

    def embed(self, texts: List[str]) -> List[List[float]]:
        ...


class OpenAIEmbeddingClient:
    """OpenAI embeddings with optional API key."""

    def __init__(self, model: str = "text-embedding-3-small", api_key: Optional[str] = None):
        self.model = model
        self._client: Any = None
        self._api_key = api_key

    def _get_client(self) -> Any:
        if self._client is None:
            try:
                from openai import OpenAI
                self._client = OpenAI(api_key=self._api_key)
            except ImportError as e:
                raise ImportError("openai package required. pip install openai") from e
        return self._client

    def embed(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        client = self._get_client()
        resp = client.embeddings.create(model=self.model, input=texts)
        order = sorted(resp.data, key=lambda d: d.index)
        return [order[i].embedding for i in range(len(order))]


def embed_chunks(
    chunks: List[dict],
    embedding_client: EmbeddingClient,
    model_spec: Optional[ModelSpec] = None,
    validate: bool = True,
    skip_invalid: bool = True,
) -> List[EmbeddingResult]:
    """
    Embed a list of chunks (dicts with chunk_id, text).
    If model_spec is provided and validate=True, chunks over the token limit are
    either skipped (skip_invalid=True) or embedded anyway and validation is recorded.
    """
    texts = []
    chunk_ids = []
    indices = []
    for i, c in enumerate(chunks):
        cid = c.get("chunk_id") or ""
        text = c.get("text") or ""
        if not cid or not text:
            continue
        if model_spec and validate:
            res = validate_chunk_for_embedding(text, model_spec)
            if not res.is_valid and skip_invalid:
                logger.warning("Skipping chunk %s (exceeds limit by %s)", cid, res.exceeds_by)
                continue
        texts.append(text)
        chunk_ids.append(cid)
        indices.append((i, c))

    if not texts:
        return []

    # Batch to avoid API limits (e.g. OpenAI 2048 per request)
    batch_size = 200
    all_vectors: List[List[float]] = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        all_vectors.extend(embedding_client.embed(batch))
    vectors = all_vectors

    results: List[EmbeddingResult] = []
    for j, (cid, text) in enumerate(zip(chunk_ids, texts)):
        if j >= len(vectors):
            break
        val: Optional[ValidationResult] = None
        if model_spec and validate:
            val = validate_chunk_for_embedding(text, model_spec)
        results.append(
            EmbeddingResult(
                chunk_id=cid,
                text=text,
                embedding=vectors[j],
                token_count=val.token_count if val else None,
                validation=val,
            )
        )
    return results
