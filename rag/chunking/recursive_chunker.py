"""
Recursive boundary-aware chunker.
Structure-preserving, token-based chunking with priority fallback:
  Level 1 – Section (Title/Header), Level 2 – Paragraph, Level 3 – \\n, Level 4 – Sentence, Level 5 – Hard token.
"""

import json
from pathlib import Path
from typing import List, Optional, Tuple

from rag.parsing.schemas import ElementType, ParsedDocument, ParsedElement

from .schemas import (
    RecursiveChunk,
    RecursiveChunkingConfig,
    RecursiveChunkMetadata,
)
from .tokenizer import Tokenizer, decode_ids, encode_text, token_count

# Section-boundary types: do not split between these and the following content
SECTION_TYPES = {ElementType.TITLE, ElementType.HEADER}
# Atomic: do not split inside (treat as one segment)
ATOMIC_TYPES = {ElementType.TABLE, ElementType.IMAGE}
# Skip entirely
SKIP_TYPES = {ElementType.PAGE_BREAK}


def _ensure_nltk() -> None:
    try:
        from nltk.tokenize import sent_tokenize  # noqa: F401
    except ImportError:
        raise ImportError("nltk is required for sentence-boundary splitting. Install with: pip install nltk")
    try:
        import nltk
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        import nltk
        nltk.download("punkt", quiet=True)


def _segment_type(el: ParsedElement) -> str:
    t = getattr(el.type, "value", None) or str(el.type)
    return t.split(".")[-1] if "." in str(t) else str(t)


def _split_text_at_boundary(
    text: str,
    max_tokens: int,
    tokenizer: Optional[Tokenizer],
    min_tokens: int,
    max_sentence_tokens: int,
) -> Tuple[str, str]:
    """
    Split text at a natural boundary so the first part has <= max_tokens (and >= min_tokens when possible).
    Priority: Level 3 \\n -> Level 4 sentence -> Level 5 hard token.
    Returns (first_part, rest).
    """
    if not text.strip():
        return text, ""
    ids = encode_text(text, tokenizer)
    if len(ids) <= max_tokens:
        return text, ""

    # Level 5: hard split at token boundary if segment is too long
    if len(ids) > max_sentence_tokens:
        max_tokens = min(max_tokens, max_sentence_tokens)

    target = max(min_tokens, max_tokens)
    first_ids = ids[:target]
    rest_ids = ids[target:]

    # Level 3: try to find last \\n in first part
    first_text = decode_ids(first_ids, tokenizer)
    last_nl = first_text.rfind("\n")
    if last_nl >= 0 and token_count(first_text[: last_nl + 1], tokenizer) >= min_tokens:
        return first_text[: last_nl + 1].rstrip(), first_text[last_nl + 1 :].lstrip() + decode_ids(rest_ids, tokenizer)

    # Level 4: sentence boundary (NLTK)
    _ensure_nltk()
    from nltk.tokenize import sent_tokenize

    sentences = sent_tokenize(first_text)
    if not sentences:
        return decode_ids(first_ids, tokenizer), decode_ids(rest_ids, tokenizer)
    acc = ""
    acc_tokens = 0
    for s in sentences:
        s_tokens = token_count(s, tokenizer)
        if acc_tokens + s_tokens <= max_tokens and (acc_tokens + s_tokens >= min_tokens or acc_tokens == 0):
            acc += (" " if acc else "") + s
            acc_tokens += s_tokens
        else:
            break
    if acc_tokens >= min_tokens:
        rest = first_text[len(acc) :].lstrip() + decode_ids(rest_ids, tokenizer)
        return acc.strip(), rest.strip() if rest else ""

    # Level 5: hard token boundary
    return decode_ids(first_ids, tokenizer), decode_ids(rest_ids, tokenizer)


def _build_segments(elements: List[ParsedElement]) -> List[Tuple[int, ElementType, str]]:
    """(element_index, type, text). Skip PageBreak, Image; skip empty."""
    out: List[Tuple[int, ElementType, str]] = []
    for i, el in enumerate(elements):
        if el.type in SKIP_TYPES:
            continue
        text = (el.text or "").strip()
        if not text and el.type != ElementType.TITLE:
            continue
        out.append((i, el.type, text))
    return out


def _chunk_document_recursive(
    doc: ParsedDocument,
    config: RecursiveChunkingConfig,
    doc_id: str,
    tokenizer: Optional[Tokenizer],
) -> List[RecursiveChunk]:
    """
    Boundary-aware chunking: section -> paragraph -> line -> sentence -> hard token.
    Overlap is taken from end of previous chunk (re-tokenized).
    """
    segments = _build_segments(doc.elements)
    if not segments:
        return []

    chunks: List[RecursiveChunk] = []
    config_name = config.get_name()
    chunk_size = config.chunk_size_tokens
    overlap_tokens = config.overlap_tokens
    min_tokens = config.min_chunk_tokens
    max_sent = config.max_sentence_tokens
    preserve_sections = config.preserve_sections

    overlap_carry = ""
    pending_rest: Optional[Tuple[int, ElementType, str]] = None  # (elem_idx, typ, rest)
    segment_idx = 0
    current_section_title: Optional[str] = None
    chunk_index = 0

    def last_is_section(types: List[str]) -> bool:
        if not types:
            return False
        return types[-1] in ("Title", "Header")

    while segment_idx < len(segments) or pending_rest is not None:
        parts: List[str] = []
        elem_indices: List[int] = []
        types_used: List[str] = []

        if overlap_carry:
            parts.append(overlap_carry)
            tc = token_count(overlap_carry, tokenizer)
        else:
            tc = 0

        # Process pending rest from a previous split
        if pending_rest is not None:
            elem_idx, typ, rest_text = pending_rest
            pending_rest = None
            typ_str = _segment_type(ParsedElement(type=typ, text=rest_text))
            n = token_count(rest_text, tokenizer)
            if tc + n <= chunk_size:
                parts.append(rest_text)
                elem_indices.append(elem_idx)
                types_used.append(typ_str)
                tc += n
                if typ in SECTION_TYPES:
                    current_section_title = rest_text[:200].strip()
            else:
                max_here = chunk_size - tc
                first, rest = _split_text_at_boundary(
                    rest_text, max_here, tokenizer, min_tokens, max_sent
                )
                parts.append(first)
                elem_indices.append(elem_idx)
                types_used.append(typ_str)
                if rest:
                    pending_rest = (elem_idx, typ, rest)
                full_text = "\n\n".join(parts)
                chunk_id = f"{doc_id}_{config_name}_{chunk_index}"
                chunks.append(
                    RecursiveChunk(
                        chunk_id=chunk_id,
                        doc_id=doc_id,
                        chunk_index=chunk_index,
                        text=full_text,
                        token_count=token_count(full_text, tokenizer),
                        source_elements=types_used,
                        metadata=RecursiveChunkMetadata(
                            section_title=current_section_title,
                            start_element_index=min(elem_indices),
                            end_element_index=max(elem_indices),
                        ),
                    )
                )
                chunk_index += 1
                ov_ids = encode_text(full_text, tokenizer)[-overlap_tokens:] if overlap_tokens else []
                overlap_carry = decode_ids(ov_ids, tokenizer) if ov_ids else ""
                continue

        while segment_idx < len(segments):
            elem_idx, typ, text = segments[segment_idx]
            typ_str = _segment_type(ParsedElement(type=typ, text=text))
            sep_tokens = 2 if parts else 0
            add_tokens = token_count(text, tokenizer) + sep_tokens

            if typ in SECTION_TYPES:
                current_section_title = text[:200].strip()

            if tc + add_tokens <= chunk_size:
                parts.append(text)
                elem_indices.append(elem_idx)
                types_used.append(typ_str)
                tc += add_tokens
                segment_idx += 1
                if preserve_sections and last_is_section(types_used):
                    continue
                else:
                    continue
            else:
                # Doesn't fit
                if len(parts) <= (1 if overlap_carry else 0):
                    # Only overlap or nothing: must split this segment
                    max_here = chunk_size - tc
                    if typ in ATOMIC_TYPES:
                        first, rest = text, ""
                        if token_count(first, tokenizer) > max_here:
                            first, rest = _split_text_at_boundary(
                                text, max_here, tokenizer, min_tokens, max_sent
                            )
                    else:
                        first, rest = _split_text_at_boundary(
                            text, max_here, tokenizer, min_tokens, max_sent
                        )
                    parts.append(first)
                    elem_indices.append(elem_idx)
                    types_used.append(typ_str)
                    if rest:
                        pending_rest = (elem_idx, typ, rest)
                    segment_idx += 1
                    full_text = "\n\n".join(parts)
                    tc_full = token_count(full_text, tokenizer)
                    if tc_full >= min_tokens:
                        chunk_id = f"{doc_id}_{config_name}_{chunk_index}"
                        chunks.append(
                            RecursiveChunk(
                                chunk_id=chunk_id,
                                doc_id=doc_id,
                                chunk_index=chunk_index,
                                text=full_text,
                                token_count=tc_full,
                                source_elements=types_used.copy(),
                                metadata=RecursiveChunkMetadata(
                                    section_title=current_section_title,
                                    start_element_index=min(elem_indices),
                                    end_element_index=max(elem_indices),
                                ),
                            )
                        )
                        chunk_index += 1
                    ov_ids = encode_text(full_text, tokenizer)[-overlap_tokens:] if overlap_tokens else []
                    overlap_carry = decode_ids(ov_ids, tokenizer) if ov_ids else ""
                    break
                elif preserve_sections and last_is_section(types_used):
                    # Last is Title/Header: include part of current segment then emit
                    max_here = chunk_size - tc
                    if typ in ATOMIC_TYPES:
                        first, rest = text, ""
                    else:
                        first, rest = _split_text_at_boundary(
                            text, max_here, tokenizer, min_tokens, max_sent
                        )
                    parts.append(first)
                    elem_indices.append(elem_idx)
                    types_used.append(typ_str)
                    if rest:
                        pending_rest = (elem_idx, typ, rest)
                    segment_idx += 1
                    full_text = "\n\n".join(parts)
                    tc_full = token_count(full_text, tokenizer)
                    if tc_full >= min_tokens:
                        chunk_id = f"{doc_id}_{config_name}_{chunk_index}"
                        chunks.append(
                            RecursiveChunk(
                                chunk_id=chunk_id,
                                doc_id=doc_id,
                                chunk_index=chunk_index,
                                text=full_text,
                                token_count=tc_full,
                                source_elements=types_used.copy(),
                                metadata=RecursiveChunkMetadata(
                                    section_title=current_section_title,
                                    start_element_index=min(elem_indices),
                                    end_element_index=max(elem_indices),
                                ),
                            )
                        )
                        chunk_index += 1
                    ov_ids = encode_text(full_text, tokenizer)[-overlap_tokens:] if overlap_tokens else []
                    overlap_carry = decode_ids(ov_ids, tokenizer) if ov_ids else ""
                    break
                else:
                    # Emit chunk; do not consume segment (add next iteration)
                    full_text = "\n\n".join(parts)
                    tc_full = token_count(full_text, tokenizer)
                    if tc_full >= min_tokens:
                        chunk_id = f"{doc_id}_{config_name}_{chunk_index}"
                        chunks.append(
                            RecursiveChunk(
                                chunk_id=chunk_id,
                                doc_id=doc_id,
                                chunk_index=chunk_index,
                                text=full_text,
                                token_count=tc_full,
                                source_elements=types_used.copy(),
                                metadata=RecursiveChunkMetadata(
                                    section_title=current_section_title,
                                    start_element_index=min(elem_indices),
                                    end_element_index=max(elem_indices),
                                ),
                            )
                        )
                        chunk_index += 1
                    ov_ids = encode_text(full_text, tokenizer)[-overlap_tokens:] if overlap_tokens else []
                    overlap_carry = decode_ids(ov_ids, tokenizer) if ov_ids else ""
                    break
        else:
            # No more segments and no pending; emit final chunk
            if parts and (not preserve_sections or not last_is_section(types_used)):
                full_text = "\n\n".join(parts)
                if token_count(full_text, tokenizer) >= min_tokens:
                    chunk_id = f"{doc_id}_{config_name}_{chunk_index}"
                    chunks.append(
                        RecursiveChunk(
                            chunk_id=chunk_id,
                            doc_id=doc_id,
                            chunk_index=chunk_index,
                            text=full_text,
                            token_count=token_count(full_text, tokenizer),
                            source_elements=types_used,
                            metadata=RecursiveChunkMetadata(
                                section_title=current_section_title,
                                start_element_index=min(elem_indices),
                                end_element_index=max(elem_indices),
                            ),
                        )
                    )
                    chunk_index += 1
            break

    return chunks


def chunk_parsed_document(
    doc: ParsedDocument,
    config: RecursiveChunkingConfig,
    doc_id: Optional[str] = None,
    tokenizer: Optional[Tokenizer] = None,
) -> List[RecursiveChunk]:
    """
    Chunk a parsed document with the recursive boundary-aware algorithm.
    doc_id defaults to the stem of doc.source_path.
    """
    doc_id = doc_id or Path(doc.source_path).stem
    return _chunk_document_recursive(doc, config, doc_id, tokenizer)


def load_parsed_document(path: Path) -> ParsedDocument:
    """Load ParsedDocument from JSON."""
    with open(path, encoding="utf-8") as f:
        return ParsedDocument.model_validate(json.load(f))


def save_chunks_to_dir(
    chunks: List[RecursiveChunk],
    output_dir: Path,
    config_name: str,
    doc_id: str,
) -> Path:
    """Save chunks to data/chunks/<config_name>/<doc_id>.json. Creates dirs."""
    output_dir = output_dir.resolve()
    subdir = output_dir / config_name
    subdir.mkdir(parents=True, exist_ok=True)
    out_path = subdir / f"{doc_id}.json"
    payload = [c.model_dump() for c in chunks]
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    return out_path


def run_chunking(
    parsed_path: Path,
    config: RecursiveChunkingConfig,
    output_dir: Path,
    tokenizer: Optional[Tokenizer] = None,
) -> Path:
    """Load parsed JSON, chunk with config, save to output_dir/<config_name>/<doc_id>.json."""
    doc = load_parsed_document(parsed_path)
    doc_id = parsed_path.stem
    chunks = chunk_parsed_document(doc, config, doc_id=doc_id, tokenizer=tokenizer)
    return save_chunks_to_dir(chunks, output_dir, config.get_name(), doc_id)
