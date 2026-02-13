"""
Run boundary-aware (recursive) chunking on parsed documents.
Output: data/chunks/<config_name>/<doc_id>.json (list of chunk objects).

Usage (from project root):
  python scripts/run_chunking.py data/parsed_docs/commercial_realty.json
  python scripts/run_chunking.py data/parsed_docs/ --config configs/chunking_configs.yaml
  python scripts/run_chunking.py data/parsed_docs/ --chunk-size 256 512 --overlap 50 100
"""

import argparse
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import yaml

from rag.chunking import RecursiveChunkingConfig
from rag.chunking.recursive_chunker import run_chunking


def load_configs_yaml(path: Path) -> list[RecursiveChunkingConfig]:
    """Load list of RecursiveChunkingConfig from YAML (list of config dicts)."""
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if isinstance(data, list):
        return [RecursiveChunkingConfig.model_validate(c) for c in data]
    return [RecursiveChunkingConfig.model_validate(data)]


def main():
    parser = argparse.ArgumentParser(
        description="Run recursive boundary-aware chunking. Saves to data/chunks/<config_name>/<doc_id>.json"
    )
    parser.add_argument(
        "input",
        type=Path,
        nargs="+",
        help="Parsed JSON file(s) or directory of parsed docs",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_PROJECT_ROOT / "data" / "chunks",
        help="Base directory for chunk output (default: data/chunks)",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="YAML file with list of chunking configs (chunk_size_tokens, overlap_tokens, etc.)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        nargs="+",
        default=None,
        metavar="N",
        help="Override: chunk sizes in tokens (e.g. 256 512 1024)",
    )
    parser.add_argument(
        "--overlap",
        type=int,
        nargs="+",
        default=None,
        metavar="N",
        help="Override: overlap in tokens (e.g. 50 100)",
    )
    parser.add_argument(
        "--min-chunk-tokens",
        type=int,
        default=50,
        help="Min chunk size in tokens (default: 50)",
    )
    parser.add_argument(
        "--preserve-sections",
        action="store_true",
        default=True,
        help="Do not cross Title/Header boundaries (default: True)",
    )
    parser.add_argument(
        "--no-preserve-sections",
        action="store_false",
        dest="preserve_sections",
        help="Allow chunk boundaries to cross sections",
    )
    args = parser.parse_args()

    # Resolve input paths
    input_paths: list[Path] = []
    for p in args.input:
        p = p.resolve()
        if p.is_file():
            if p.suffix.lower() != ".json":
                print(f"Skip (not JSON): {p}", file=sys.stderr)
                continue
            input_paths.append(p)
        elif p.is_dir():
            input_paths.extend(sorted(p.glob("*.json")))
        else:
            print(f"Not found: {p}", file=sys.stderr)

    if not input_paths:
        print("No parsed JSON files found.", file=sys.stderr)
        return 1

    # Build configs
    configs: list[RecursiveChunkingConfig] = []
    if args.config and args.config.exists():
        configs = load_configs_yaml(args.config)
    if args.chunk_size is not None or args.overlap is not None:
        sizes = args.chunk_size or [512]
        overlaps = args.overlap or [100]
        configs = []
        for cs in sizes:
            for ov in overlaps:
                if ov < cs:
                    configs.append(
                        RecursiveChunkingConfig(
                            chunk_size_tokens=cs,
                            overlap_tokens=ov,
                            min_chunk_tokens=args.min_chunk_tokens,
                            preserve_sections=args.preserve_sections,
                        )
                    )
    if not configs:
        # Default single config
        configs = [
            RecursiveChunkingConfig(
                chunk_size_tokens=512,
                overlap_tokens=100,
                min_chunk_tokens=args.min_chunk_tokens,
                preserve_sections=args.preserve_sections,
            )
        ]

    output_dir = args.output_dir.resolve()
    for parsed_path in input_paths:
        for config in configs:
            out_path = run_chunking(parsed_path, config, output_dir)
            print(f"Chunked {parsed_path.name} [{config.get_name()}] -> {out_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
