"""Thin wrapper: adversarial verification of the signed-response [D-01] PASS.

Answers the defense-panel attacks empirically (SR calibration, few-pixel
domination, nonlinear absorption leakage, cross-recipe, magnitude recompute).
No logic here. Run from repo root:

    PYTHONPATH=. uv run python scripts/verify_signed_response.py
"""

import argparse
import json
from pathlib import Path

from src.core.signed_response import verify_signed_response_pass


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Adversarial verification of the signed-response PASS (defense panel)."
    )
    parser.add_argument(
        "--out-dir", type=Path, default=Path("results/signed_response_embedding"),
        help="Output directory (default: results/signed_response_embedding).",
    )
    args = parser.parse_args()
    out = verify_signed_response_pass(args.out_dir)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
