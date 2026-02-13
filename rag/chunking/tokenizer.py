"""
Token counter and injectable tokenizer for boundary-aware chunking.
Model-aligned tokenizers: tiktoken (OpenAI), sentencepiece (Instructor), transformers (HF).
"""

from typing import Callable, List, Optional, Protocol

# Default: tiktoken (OpenAI-aligned). Lazy import to avoid hard dep if not installed.
_default_encoding: Optional[object] = None


def _get_tiktoken_encoding(model: str = "cl100k_base"):
    """Lazy load tiktoken encoding."""
    global _default_encoding
    if _default_encoding is None:
        try:
            import tiktoken
            _default_encoding = tiktoken.get_encoding(model)
        except ImportError as e:
            raise ImportError(
                "tiktoken is required for default tokenizer. Install with: pip install tiktoken"
            ) from e
    return _default_encoding


class Tokenizer(Protocol):
    """Protocol for injectable tokenizers: encode to ids, decode back to text."""

    def encode(self, text: str) -> List[int]:
        """Encode text to list of token ids."""
        ...

    def decode(self, ids: List[int]) -> str:
        """Decode token ids back to text."""
        ...


def token_count(text: str, tokenizer: Optional[Tokenizer] = None) -> int:
    """
    Return token count for text.
    Uses provided tokenizer or default tiktoken (cl100k_base).
    """
    if tokenizer is not None:
        return len(tokenizer.encode(text))
    enc = _get_tiktoken_encoding()
    return len(enc.encode(text))


def encode_text(text: str, tokenizer: Optional[Tokenizer] = None) -> List[int]:
    """Encode text to token ids. Uses default tiktoken if tokenizer is None."""
    if tokenizer is not None:
        return tokenizer.encode(text)
    enc = _get_tiktoken_encoding()
    return enc.encode(text)


def decode_ids(ids: List[int], tokenizer: Optional[Tokenizer] = None) -> str:
    """Decode token ids to text."""
    if tokenizer is not None:
        return tokenizer.decode(ids)
    enc = _get_tiktoken_encoding()
    return enc.decode(ids)


def text_to_tokens(text: str, tokenizer: Optional[Tokenizer] = None) -> List[int]:
    """Alias for encode_text; returns list of token ids."""
    return encode_text(text, tokenizer)


def tokens_to_text(ids: List[int], tokenizer: Optional[Tokenizer] = None) -> str:
    """Alias for decode_ids."""
    return decode_ids(ids, tokenizer)


class TiktokenTokenizer:
    """Wrapper so tiktoken encoding satisfies Tokenizer protocol."""

    def __init__(self, model: str = "cl100k_base"):
        import tiktoken
        self._enc = tiktoken.get_encoding(model)

    def encode(self, text: str) -> List[int]:
        return self._enc.encode(text)

    def decode(self, ids: List[int]) -> str:
        return self._enc.decode(ids)


def get_default_tokenizer() -> Tokenizer:
    """Return default tokenizer (tiktoken cl100k_base)."""
    return TiktokenTokenizer()
