"""
Simple script to run PDF parsing on one PDF.
Usage (from project root):
  python scripts/run_parse_one_pdf.py path/to/file.pdf
  python scripts/run_parse_one_pdf.py path/to/file.pdf --output-dir data/parsed_docs
"""

import argparse
import sys
from pathlib import Path

# Project root = parent of scripts/
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from rag.parsing import parse_pdf_and_save


def main():
    parser = argparse.ArgumentParser(description="Parse one PDF and save structured JSON to data/parsed_docs")
    parser.add_argument("pdf_path", type=Path, help="Path to the PDF file")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_PROJECT_ROOT / "data" / "parsed_docs",
        help="Directory to save parsed JSON (default: data/parsed_docs)",
    )
    parser.add_argument("--output-name", type=str, default=None, help="Output filename (default: <pdf_stem>.json)")
    args = parser.parse_args()

    pdf_path = args.pdf_path.resolve()
    if not pdf_path.exists():
        print(f"Error: PDF not found: {pdf_path}")
        return 1

    output_dir = args.output_dir.resolve()
    out_path = parse_pdf_and_save(
        pdf_path,
        output_dir,
        output_filename=args.output_name,
    )
    print(f"Parsed {pdf_path.name} -> {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
