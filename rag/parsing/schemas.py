"""
Pydantic schemas for parsed document elements.
Used by the PDF parser to output structured JSON.
"""

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class ElementType(str, Enum):
    """Standard element types from parsed documents."""

    TITLE = "Title"
    PARAGRAPH = "Paragraph"
    TEXT = "Text"
    LIST = "List"
    LIST_ITEM = "ListItem"
    TABLE = "Table"
    IMAGE = "Image"
    PAGE_BREAK = "PageBreak"
    HEADER = "Header"
    FOOTER = "Footer"
    UNCATEGORIZED = "UncategorizedText"


class ParsedElement(BaseModel):
    """A single parsed document element (block)."""

    type: ElementType = Field(..., description="Element type (Title, Paragraph, List, etc.)")
    text: str = Field(..., description="Extracted text content")
    page_number: Optional[int] = Field(None, description="1-based page number if available")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Extra metadata (coordinates, etc.)")


class ParsedDocument(BaseModel):
    """Full parsed document: source path + list of elements."""

    source_path: str = Field(..., description="Path to the original PDF")
    elements: List[ParsedElement] = Field(default_factory=list, description="Ordered list of parsed elements")
