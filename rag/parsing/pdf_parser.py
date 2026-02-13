"""
PDF parsing module using the unstructured library.
Extracts structured document elements and outputs JSON (Pydantic schema).
"""

import json
from pathlib import Path
from typing import List, Union

from .schemas import ElementType, ParsedDocument, ParsedElement

# Lazy import so the module loads even if unstructured is not installed
def _partition_pdf(path: Union[str, Path], **kwargs):
    from unstructured.partition.pdf import partition_pdf as _partition_pdf
    return _partition_pdf(filename=str(path), **kwargs)


def _element_type_from_category(category: str) -> ElementType:
    """Map unstructured category to our ElementType."""
    # Handle enum-style values (e.g. "Category.TITLE" or object with .name)
    raw = getattr(category, "name", None) or getattr(category, "value", None) or str(category)
    raw = str(raw).split(".")[-1]  # "Category.TITLE" -> "TITLE"
    mapping = {
        "Title": ElementType.TITLE,
        "NarrativeText": ElementType.PARAGRAPH,
        "Text": ElementType.TEXT,
        "UncategorizedText": ElementType.UNCATEGORIZED,
        "ListItem": ElementType.LIST_ITEM,
        "List": ElementType.LIST,
        "Table": ElementType.TABLE,
        "Image": ElementType.IMAGE,
        "PageBreak": ElementType.PAGE_BREAK,
        "Header": ElementType.HEADER,
        "Footer": ElementType.FOOTER,
    }
    return mapping.get(raw, mapping.get(raw.title(), ElementType.UNCATEGORIZED))


def parse_pdf(path: Union[str, Path], **partition_kwargs) -> ParsedDocument:
    """
    Parse a PDF file into structured elements using unstructured.

    Args:
        path: Path to the PDF file.
        **partition_kwargs: Optional kwargs passed to unstructured.partition (e.g. strategy).

    Returns:
        ParsedDocument with element types and text.
    """
    path = Path(path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {path}")

    elements_raw = _partition_pdf(path, **partition_kwargs)
    elements: List[ParsedElement] = []

    for el in elements_raw:
        category = getattr(el, "category", None) or "UncategorizedText"
        text = getattr(el, "text", "") or ""
        metadata_el = getattr(el, "metadata", None)
        page_number = None
        meta_dict = {}
        if metadata_el is not None:
            page_number = getattr(metadata_el, "page_number", None)
            if hasattr(metadata_el, "to_dict"):
                meta_dict = metadata_el.to_dict()
            else:
                meta_dict = {k: getattr(metadata_el, k, None) for k in ("page_number", "filename") if hasattr(metadata_el, k)}

        elem_type = _element_type_from_category(str(category))
        elements.append(
            ParsedElement(
                type=elem_type,
                text=text.strip(),
                page_number=page_number,
                metadata=meta_dict,
            )
        )

    return ParsedDocument(source_path=str(path), elements=elements)


def save_parsed_doc(doc: ParsedDocument, output_path: Union[str, Path]) -> Path:
    """Save a ParsedDocument to JSON. Creates parent dirs if needed."""
    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(doc.model_dump(), f, indent=2, ensure_ascii=False)
    return output_path


def parse_pdf_and_save(
    pdf_path: Union[str, Path],
    output_dir: Union[str, Path],
    output_filename: str = None,
    **partition_kwargs,
) -> Path:
    """
    Parse a PDF and save the result to output_dir.
    Output filename defaults to the PDF stem + .json.

    Returns:
        Path to the saved JSON file.
    """
    pdf_path = Path(pdf_path).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    name = output_filename or f"{pdf_path.stem}.json"
    out_path = output_dir / name

    doc = parse_pdf(pdf_path, **partition_kwargs)
    return save_parsed_doc(doc, out_path)
