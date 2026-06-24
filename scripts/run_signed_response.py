"""Thin wrapper: run the signed-response-embedding control-first de-risking probe.

No logic here — a single call into
:func:`src.core.signed_response.run_signed_response_probe`. Run from repo root:

    PYTHONPATH=. uv run python scripts/run_signed_response.py
"""

import argparse
import json
from pathlib import Path

from src.core.signed_response import run_signed_response_probe


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the signed-response-embedding de-risking probe (see SCOPING.md)."
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("results/signed_response_embedding"),
        help="Output directory (default: results/signed_response_embedding).",
    )
    args = parser.parse_args()
    result = run_signed_response_probe(args.out_dir)
    print(json.dumps({k: v for k, v in result.items() if k != "reading"}, indent=2))
    print("\nVERDICT:", result["verdict"])
    print(result["reading"])


if __name__ == "__main__":
    main()
