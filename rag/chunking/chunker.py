"""
Chunking module: split parsed document text into overlapping chunks.
Uses character-based sliding window; multiple ChunkingConfigs supported.
"""

import json
from pathlib import Path
from typing import List, Union

from rag.parsing.schemas import ParsedDocument, ParsedElement

from .schemas import Chunk, ChunkedDocument, ChunkingConfig


def _build_text_and_ranges(elements: List[ParsedElement], sep: str = "\n\n") -> tuple[str, List[tuple[int, int, int]]]:
    """
    Concatenate element texts and build (start, end, page_number) for each segment.
    Returns (full_text, list of (start, end, page)).
    """
    parts: List[str] = []
    ranges: List[tuple[int, int, int]] = []
    start = 0
    for el in elements:
        text = (el.text or "").strip()
        if not text:
            continue
        page = el.page_number or 0
        parts.append(text)
        end = start + len(text)
        ranges.append((start, end, page))
        start = end + len(sep)
        if sep:
            parts.append(sep)
    full_text = "".join(parts)
    return full_text, ranges


def _pages_for_range(ranges: List[tuple[int, int, int]], c_start: int, c_end: int) -> List[int]:
    """Collect page numbers for segments overlapping [c_start, c_end)."""
    pages: set[int] = set()
    for s, e, p in ranges:
        if p and (s < c_end and e > c_start):
            pages.add(p)
    return sorted(pages)


def chunk_document(
    doc: ParsedDocument,
    config: ChunkingConfig,
    source_stem: str = "",
) -> ChunkedDocument:
    """
    Split a parsed document into chunks using a sliding window.

    Args:
        doc: ParsedDocument (elements with text and page_number).
        config: ChunkingConfig (chunk_size, overlap).
        source_stem: Optional stem for chunk_id (e.g. PDF or JSON filename stem).

    Returns:
        ChunkedDocument with chunks and config.
    """
    if not doc.elements:
        return ChunkedDocument(
            source_path=doc.source_path,
            config=config,
            chunks=[],
        )

    full_text, ranges = _build_text_and_ranges(doc.elements)
    if not full_text.strip():
        return ChunkedDocument(
            source_path=doc.source_path,
            config=config,
            chunks=[],
        )

    config_name = config.get_name()
    stem = source_stem or Path(doc.source_path).stem
    step = config.step
    chunk_size = config.chunk_size
    chunks: List[Chunk] = []
    n = len(full_text)
    index = 0
    start = 0

    while start < n:
        end = min(start + chunk_size, n)
        text = full_text[start:end]
        if not text.strip():
            start += step
            continue
        page_numbers = _pages_for_range(ranges, start, end)
        chunk_id = f"{stem}_{config_name}_{index}"
        chunks.append(
            Chunk(
                text=text,
                chunk_id=chunk_id,
                source_path=doc.source_path,
                config_name=config_name,
                index=index,
                start_char=start,
                end_char=end,
                page_numbers=page_numbers,
                metadata={},
            )
        )
        index += 1
        start += step
        if start >= n:
            break

    return ChunkedDocument(
        source_path=doc.source_path,
        config=config,
        chunks=chunks,
    )


def load_parsed_document(path: Union[str, Path]) -> ParsedDocument:
    """Load a ParsedDocument from JSON."""
    path = Path(path).resolve()
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return ParsedDocument.model_validate(data)


def chunk_parsed_file(
    parsed_path: Union[str, Path],
    config: ChunkingConfig,
    source_stem: str = "",
) -> ChunkedDocument:
    """Load a parsed JSON file and return a ChunkedDocument."""
    doc = load_parsed_document(parsed_path)
    stem = source_stem or Path(parsed_path).stem
    return chunk_document(doc, config, source_stem=stem)


def save_chunked_document(chunked: ChunkedDocument, output_path: Union[str, Path]) -> Path:
    """Save a ChunkedDocument to JSON. Creates parent dirs if needed."""
    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(chunked.model_dump(), f, indent=2, ensure_ascii=False)
    return output_path


def chunk_parsed_file_and_save(
    parsed_path: Union[str, Path],
    config: ChunkingConfig,
    output_dir: Union[str, Path],
    output_filename: str = None,
) -> Path:
    """
    Chunk a parsed document and save to output_dir.
    Filename defaults to <parsed_stem>_<config_name>.json.
    """
    parsed_path = Path(parsed_path).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = parsed_path.stem
    config_name = config.get_name()
    name = output_filename or f"{stem}_{config_name}.json"
    out_path = output_dir / name

    chunked = chunk_parsed_file(parsed_path, config, source_stem=stem)
    return save_chunked_document(chunked, out_path)
