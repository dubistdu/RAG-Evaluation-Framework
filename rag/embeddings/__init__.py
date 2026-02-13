"""Embeddings module: embed chunks with token validation."""

from .embedder import (
    EmbeddingClient,
    EmbeddingResult,
    OpenAIEmbeddingClient,
    embed_chunks,
)

__all__ = [
    "EmbeddingClient",
    "EmbeddingResult",
    "OpenAIEmbeddingClient",
    "embed_chunks",
]
