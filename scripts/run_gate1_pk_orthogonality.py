"""Thin wrapper: GATE 1 de-risk for the signed-response embedding track.

Tests whether the amplitude-free feedback-response DIRECTION is orthogonal to the
baseline per-sightline flux power spectrum P_F(k) (TRACK_SPEC.md §3.1). No logic
here. Run from repo root:

    PYTHONPATH=. uv run python scripts/run_gate1_pk_orthogonality.py
"""

import argparse
import json
from pathlib import Path

from src.core.signed_response import gate1_pk_orthogonality


def main() -> None:
    parser = argparse.ArgumentParser(description="Gate-1 P_F(k) orthogonality de-risk.")
    parser.add_argument(
        "--out-dir", type=Path, default=Path("results/signed_response_embedding"),
        help="Output directory (default: results/signed_response_embedding).",
    )
    args = parser.parse_args()
    result = gate1_pk_orthogonality(args.out_dir)
    print(json.dumps({k: v for k, v in result.items() if k != "reading"}, indent=2))
    print("\nVERDICT:", result["verdict"])
    print(result["reading"])


if __name__ == "__main__":
    main()
