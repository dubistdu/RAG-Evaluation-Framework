"""
Legacy: character-based chunking. Prefer scripts/run_chunking.py with
configs/chunking_configs.yaml for token-based boundary-aware chunking.

Chunk parsed documents with one or more (chunk_size, overlap) configurations.
Usage (from project root):
  python scripts/run_chunk.py data/parsed_docs/commercial_realty.json
  python scripts/run_chunk.py data/parsed_docs/commercial_realty.json --chunk-size 512 --overlap 64
  python scripts/run_chunk.py data/parsed_docs/ --chunk-size 512 1024 --overlap 64 128
"""

import argparse
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from rag.chunking import ChunkingConfig, chunk_parsed_file_and_save


def main():
    parser = argparse.ArgumentParser(
        description="Chunk parsed JSON docs with configurable chunk_size and overlap. Saves to data/chunks/."
    )
    parser.add_argument(
        "input",
        type=Path,
        nargs="+",
        help="Paths to parsed JSON file(s) or a directory of parsed docs",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_PROJECT_ROOT / "data" / "chunks",
        help="Directory to save chunked JSON (default: data/chunks)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        nargs="+",
        default=[512],
        metavar="N",
        help="Chunk sizes in characters (default: 512). Can pass multiple: 512 1024",
    )
    parser.add_argument(
        "--overlap",
        type=int,
        nargs="+",
        default=[64],
        metavar="N",
        help="Overlap between chunks in characters (default: 64). Can pass multiple: 64 128",
    )
    args = parser.parse_args()

    # Resolve input paths to parsed JSON files
    input_paths: list[Path] = []
    for p in args.input:
        p = p.resolve()
        if p.is_file():
            if p.suffix.lower() != ".json":
                print(f"Skip (not JSON): {p}", file=sys.stderr)
                continue
            input_paths.append(p)
        elif p.is_dir():
            for f in sorted(p.glob("*.json")):
                input_paths.append(f)
        else:
            print(f"Not found: {p}", file=sys.stderr)

    if not input_paths:
        print("No parsed JSON files found.", file=sys.stderr)
        return 1

    # Build configs: if multiple chunk-sizes or overlaps, pair by index (or Cartesian product)
    # Use Cartesian product: every (chunk_size, overlap) combination
    configs: list[ChunkingConfig] = []
    for cs in args.chunk_size:
        for ov in args.overlap:
            if ov >= cs:
                print(f"Skip config overlap={ov} >= chunk_size={cs}", file=sys.stderr)
                continue
            configs.append(ChunkingConfig(chunk_size=cs, overlap=ov))

    if not configs:
        print("No valid (chunk_size, overlap) configs.", file=sys.stderr)
        return 1

    output_dir = args.output_dir.resolve()
    for parsed_path in input_paths:
        for config in configs:
            out_path = chunk_parsed_file_and_save(
                parsed_path,
                config,
                output_dir,
            )
            print(f"Chunked {parsed_path.name} [{config.get_name()}] -> {out_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
