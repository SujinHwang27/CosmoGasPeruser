"""Thin wrapper: S3 of the signed-response embedding track.

Cross-feature stability + physical interpretation (TRACK_SPEC §4.3 + §4.2). No
logic here. Run from repo root:

    PYTHONPATH=. uv run python scripts/run_s3_stability.py
"""

import argparse
import json
from pathlib import Path

from src.core.signed_response_embedding import run_s3_stability_interpretation


def main() -> None:
    parser = argparse.ArgumentParser(description="Signed-response embedding S3 stability + interpretation.")
    parser.add_argument(
        "--out-dir", type=Path, default=Path("results/signed_response_embedding"),
        help="Output directory (default: results/signed_response_embedding).",
    )
    args = parser.parse_args()
    result = run_s3_stability_interpretation(args.out_dir)
    print(json.dumps({k: v for k, v in result.items() if k != "reading"}, indent=2))
    print("\nVERDICT:", result["verdict"])
    print(result["reading"])


if __name__ == "__main__":
    main()
