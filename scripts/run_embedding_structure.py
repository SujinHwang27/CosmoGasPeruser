"""Thin wrapper: S1+S2 of the signed-response embedding track.

Builds the de-entangled DIRECTION embedding and runs the panel-controlled structure
test (TRACK_SPEC §4.0/§4.1 REVISED). No logic here. Run from repo root:

    PYTHONPATH=. uv run python scripts/run_embedding_structure.py
"""

import argparse
import json
from pathlib import Path

from src.core.signed_response_embedding import run_embedding_structure


def main() -> None:
    parser = argparse.ArgumentParser(description="Signed-response embedding S1+S2 structure test.")
    parser.add_argument(
        "--out-dir", type=Path, default=Path("results/signed_response_embedding"),
        help="Output directory (default: results/signed_response_embedding).",
    )
    args = parser.parse_args()
    result = run_embedding_structure(args.out_dir)
    print(json.dumps({k: v for k, v in result.items() if k != "reading"}, indent=2))
    print("\nVERDICT:", result["verdict"])
    print(result["reading"])


if __name__ == "__main__":
    main()
