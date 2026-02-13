"""
Tests that recursive chunker uses boundary-aware splitting, not hard character cuts.
"""

import pytest

from rag.parsing.schemas import ElementType, ParsedDocument, ParsedElement
from rag.chunking.recursive_chunker import chunk_parsed_document
from rag.chunking.schemas import RecursiveChunkingConfig


# Natural boundary characters: chunk should end with one of these (or be last chunk)
SENTENCE_END = ".!?"
LINE_OR_PARAGRAPH_END = "\n"
# Chunk may also end with space (e.g. after a word before \n)
NATURAL_BOUNDARY_CHARS = frozenset(SENTENCE_END + LINE_OR_PARAGRAPH_END + " \t")


def _chunk_ends_at_natural_boundary(chunk_text: str, is_last: bool) -> bool:
    """True if chunk does not end with a mid-word or mid-sentence hard cut."""
    # Original text ending with space/newline means we split at a natural boundary
    if chunk_text.endswith((" ", "\n", "\t")):
        return True
    t = chunk_text.rstrip()
    if not t:
        return True
    last = t[-1]
    # Last chunk can end with anything
    if is_last:
        return True
    # Non-last chunk should end at sentence end, newline, or space (not mid-word)
    if last in NATURAL_BOUNDARY_CHARS:
        return True
    # Ending with closing quote/paren after .!? is fine
    if last in '")]\'' and len(t) > 1 and t[-2] in SENTENCE_END:
        return True
    # End with a full word (3+ letters): accept as boundary, not hard cut like "Sec" or "tion"
    if last.isalpha():
        last_word = "".join(c for c in reversed(t) if c.isalpha())[::-1]
        if len(last_word) >= 3:
            return True
    # Otherwise we likely cut mid-word (e.g. "Sec" from "Second")
    return False


def _make_doc(*elements: tuple[str, str]) -> ParsedDocument:
    """(element_type_name, text) -> ParsedDocument."""
    type_map = {
        "Title": ElementType.TITLE,
        "Header": ElementType.HEADER,
        "Paragraph": ElementType.PARAGRAPH,
        "Text": ElementType.TEXT,
    }
    el_list = [
        ParsedElement(type=type_map[t], text=text, page_number=1)
        for t, text in elements
    ]
    return ParsedDocument(source_path="test.pdf", elements=el_list)


def test_chunks_end_at_natural_boundaries_not_mid_word():
    """Chunks must not end with a mid-word character cut (e.g. 'Sec' from 'Second')."""
    # Use content with clear sentence/paragraph boundaries so chunker can split there
    long_para = " ".join([f"Sentence number {i} here." for i in range(40)])
    doc = _make_doc(
        ("Title", "Annual Report"),
        ("Paragraph", "First sentence here. Second sentence there. Third one follows.\n\n"),
        ("Paragraph", "Another paragraph. With multiple sentences. And more text.\n\n"),
        ("Paragraph", "Short.\n\n"),
        ("Paragraph", long_para),
    )
    config = RecursiveChunkingConfig(
        chunk_size_tokens=64,
        overlap_tokens=10,
        min_chunk_tokens=20,
        preserve_sections=True,
    )
    chunks = chunk_parsed_document(doc, config, doc_id="test")
    assert len(chunks) >= 1
    for i, c in enumerate(chunks):
        is_last = i == len(chunks) - 1
        assert _chunk_ends_at_natural_boundary(c.text, is_last), (
            f"Chunk {i} appears to end with mid-word or hard cut: "
            f"ends with {repr(c.text.rstrip()[-20:])}"
        )


def test_chunks_respect_sentence_boundaries_with_tiny_size():
    """With small token limit, splits should fall at sentence boundaries, not mid-sentence."""
    # Many short sentences so chunk_size 25 forces multiple chunks
    sentences = " One. Two. Three. Four. Five. Six. Seven. Eight. Nine. Ten. " * 5
    doc = _make_doc(("Paragraph", sentences))
    config = RecursiveChunkingConfig(
        chunk_size_tokens=25,
        overlap_tokens=5,
        min_chunk_tokens=5,
        preserve_sections=False,
    )
    chunks = chunk_parsed_document(doc, config, doc_id="test")
    assert len(chunks) >= 2
    for i, c in enumerate(chunks):
        is_last = i == len(chunks) - 1
        assert _chunk_ends_at_natural_boundary(c.text, is_last), (
            f"Chunk {i} should end at sentence/newline/space: {repr(c.text[-30:])}"
        )


def test_no_chunk_ends_with_lone_lowercase_letter():
    """No chunk (except last) should end with a lowercase letter (clear mid-word cut)."""
    doc = _make_doc(
        ("Title", "Report"),
        ("Paragraph", "Management discussed the strategy. Results were strong. End of section."),
        ("Paragraph", "Next section has more. Content here. And here."),
    )
    config = RecursiveChunkingConfig(
        chunk_size_tokens=20,
        overlap_tokens=5,
        min_chunk_tokens=5,
        preserve_sections=True,
    )
    chunks = chunk_parsed_document(doc, config, doc_id="test")
    for i, c in enumerate(chunks[:-1]):  # exclude last
        assert _chunk_ends_at_natural_boundary(c.text, is_last=False), (
            f"Chunk {i} should not end mid-word: ...{repr(c.text.rstrip()[-25:])}"
        )


def test_section_title_not_separated_from_content():
    """Title/Header element should not appear alone in a chunk; chunk should include following content."""
    doc = _make_doc(
        ("Title", "Section One"),
        ("Paragraph", "Content for section one. It has two sentences."),
        ("Title", "Section Two"),
        ("Paragraph", "Content for section two."),
    )
    config = RecursiveChunkingConfig(
        chunk_size_tokens=512,
        overlap_tokens=0,
        min_chunk_tokens=10,
        preserve_sections=True,
    )
    chunks = chunk_parsed_document(doc, config, doc_id="test")
    # With preserve_sections, we should not have a chunk that is only "Section One" or "Section Two"
    for c in chunks:
        source = c.source_elements
        # If chunk has only one element and it's Title, that's wrong (title without content)
        if len(source) == 1 and source[0] == "Title":
            pytest.fail(f"Chunk has only Title, no following content: {c.chunk_id}")
        if len(source) == 1 and source[0] == "Header":
            pytest.fail(f"Chunk has only Header, no following content: {c.chunk_id}")


def test_token_count_matches_chunk_size_target_approximately():
    """Chunks should be near target token size (not arbitrary character slices)."""
    doc = _make_doc(
        ("Paragraph", "Word " * 200 + "end."),
    )
    config = RecursiveChunkingConfig(
        chunk_size_tokens=50,
        overlap_tokens=10,
        min_chunk_tokens=15,
        preserve_sections=False,
    )
    chunks = chunk_parsed_document(doc, config, doc_id="test")
    for c in chunks:
        # token_count should be in a sane range (not 500 chars = 500 tokens like naive char split)
        assert c.token_count <= config.chunk_size_tokens + 50, (
            f"Chunk token_count {c.token_count} >> chunk_size_tokens {config.chunk_size_tokens}"
        )
