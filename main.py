"""
RAG evaluation pipeline entry point.
Steps: parsing -> chunking -> (embeddings, vectorstore, retrieval, evaluation) - built incrementally.
"""

import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="RAG evaluation pipeline")
    parser.add_argument("--parse-pdf", type=Path, metavar="PATH", help="Parse a single PDF and save to data/parsed_docs")
    parser.add_argument(
        "--chunk",
        type=Path,
        metavar="PATH",
        help="Chunk parsed JSON file(s); use data/parsed_docs or path to one .json",
    )
    parser.add_argument("--chunk-size", type=int, nargs="+", default=[512], help="Chunk sizes (default: 512)")
    parser.add_argument("--overlap", type=int, nargs="+", default=[64], help="Overlap sizes (default: 64)")
    parser.add_argument("--chunk-output-dir", type=Path, default=Path("data/chunks"), help="Where to save chunked JSON")
    args = parser.parse_args()

    if args.parse_pdf:
        from rag.parsing import parse_pdf_and_save
        out_dir = Path("data/parsed_docs")
        out_path = parse_pdf_and_save(args.parse_pdf, out_dir)
        print(f"Parsed -> {out_path}")
        return 0

    if args.chunk is not None:
        from rag.chunking import ChunkingConfig, chunk_parsed_file_and_save
        inp = args.chunk.resolve()
        paths = [inp] if inp.is_file() else list(inp.glob("*.json"))
        if not paths:
            print("No parsed JSON files found at", args.chunk)
            return 1
        for cs in args.chunk_size:
            for ov in args.overlap:
                if ov >= cs:
                    continue
                config = ChunkingConfig(chunk_size=cs, overlap=ov)
                for p in paths:
                    out_path = chunk_parsed_file_and_save(p, config, args.chunk_output_dir)
                    print(f"Chunked {p.name} [{config.get_name()}] -> {out_path}")
        return 0

    print("Usage:")
    print("  python main.py --parse-pdf path/to/lecture.pdf")
    print("  python main.py --chunk data/parsed_docs [--chunk-size 512 1024] [--overlap 64 128]")
    print("Or use scripts/run_parse_one_pdf.py and scripts/run_chunking.py (see docs/QUICK_REFERENCE.md)")
    return 0


if __name__ == "__main__":
    main()
