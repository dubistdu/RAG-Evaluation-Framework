"""
Pydantic schemas for text chunks.
Used by the chunking module to output structured chunks per configuration.
"""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class ChunkingConfig(BaseModel):
    """Configuration for one chunking strategy (chunk_size + overlap)."""

    chunk_size: int = Field(..., gt=0, description="Target size of each chunk (characters)")
    overlap: int = Field(0, ge=0, description="Overlap between consecutive chunks (characters)")
    name: Optional[str] = Field(
        None,
        description="Optional config id, e.g. 'chunk512_overlap64'. Auto-generated if omitted.",
    )

    @property
    def step(self) -> int:
        """Step size between chunk starts (chunk_size - overlap)."""
        return max(1, self.chunk_size - self.overlap)

    def get_name(self) -> str:
        """Config id for filenames and chunk metadata."""
        return self.name or f"chunk{self.chunk_size}_overlap{self.overlap}"


class Chunk(BaseModel):
    """A single text chunk with provenance metadata."""

    text: str = Field(..., description="Chunk text content")
    chunk_id: str = Field(..., description="Unique id, e.g. source_stem + config_name + index")
    source_path: str = Field(..., description="Path to the source parsed document (or original PDF)")
    config_name: str = Field(..., description="Chunking config id this chunk belongs to")
    index: int = Field(..., ge=0, description="0-based index of this chunk within the document")
    start_char: int = Field(0, ge=0, description="Start offset in the concatenated document text")
    end_char: int = Field(0, ge=0, description="End offset in the concatenated document text")
    page_numbers: List[int] = Field(default_factory=list, description="Page numbers spanned by this chunk")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Extra metadata")


class ChunkedDocument(BaseModel):
    """A document chunked under one configuration: config + list of chunks."""

    source_path: str = Field(..., description="Path to the source parsed document")
    config: ChunkingConfig = Field(..., description="Chunking configuration used")
    chunks: List[Chunk] = Field(default_factory=list, description="Ordered list of chunks")


# --- Recursive (boundary-aware, token-based) chunking ---


class RecursiveChunkingConfig(BaseModel):
    """Configuration for boundary-aware, token-based chunking."""

    chunk_size_tokens: int = Field(..., gt=0, description="Target chunk size in tokens (e.g. 512)")
    overlap_tokens: int = Field(0, ge=0, description="Overlap in tokens for context continuity (e.g. 100)")
    min_chunk_tokens: int = Field(50, gt=0, description="Discard chunks smaller than this")
    max_sentence_tokens: int = Field(256, gt=0, description="Prevent single sentence exceeding this (hard split fallback)")
    preserve_sections: bool = Field(True, description="Do not cross Title/Header boundaries")
    name: Optional[str] = Field(None, description="Config id for output dir; auto-generated if omitted.")

    def get_name(self) -> str:
        return self.name or f"recursive_t{self.chunk_size_tokens}_o{self.overlap_tokens}"


class RecursiveChunkMetadata(BaseModel):
    """Metadata for a single recursive chunk."""

    section_title: Optional[str] = Field(None, description="Nearest Title/Header above this chunk")
    start_element_index: int = Field(..., ge=0, description="First element index in doc")
    end_element_index: int = Field(..., ge=0, description="Last element index (inclusive)")


class RecursiveChunk(BaseModel):
    """One chunk produced by the recursive boundary-aware chunker."""

    chunk_id: str = Field(..., description="Unique chunk id")
    doc_id: str = Field(..., description="Document id (e.g. source stem)")
    chunk_index: int = Field(..., ge=0, description="0-based index in document")
    text: str = Field(..., description="Chunk text content")
    token_count: int = Field(..., ge=0, description="Token count of text")
    source_elements: List[str] = Field(default_factory=list, description="Element types included (e.g. Paragraph, List)")
    metadata: RecursiveChunkMetadata = Field(..., description="Section and element indices")
