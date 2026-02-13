"""Generation module: synthetic question generation for retrieval evaluation."""

from .synthetic_qa_generator import (
    QAConfig,
    SyntheticQuestion,
    EmbeddingClient,
    LLMClient,
    OpenAIEmbeddingClient,
    OpenAILLMClient,
    OpenRouterLLMClient,
    generate_questions_for_config,
    load_chunks_for_config,
)

__all__ = [
    "QAConfig",
    "SyntheticQuestion",
    "EmbeddingClient",
    "LLMClient",
    "OpenAIEmbeddingClient",
    "OpenAILLMClient",
    "OpenRouterLLMClient",
    "generate_questions_for_config",
    "load_chunks_for_config",
]
