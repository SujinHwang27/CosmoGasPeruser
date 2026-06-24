"""Thin wrapper: run the feedback-response-field control-first de-risking probe.

No logic here — a single call into :func:`src.core.feedback_response.run_response_probe`.
Run from the repo root:

    PYTHONPATH=. uv run python scripts/run_feedback_response.py
"""

import argparse
import json
from pathlib import Path

from src.core.feedback_response import run_response_probe


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the feedback-response-field de-risking probe (see SCOPING.md)."
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("results/feedback_response_field"),
        help="Output directory (default: results/feedback_response_field).",
    )
    args = parser.parse_args()
    result = run_response_probe(args.out_dir)
    print(json.dumps({k: v for k, v in result.items() if k != "reading"}, indent=2))
    print("\nVERDICT:", result["verdict"])
    print(result["reading"])


if __name__ == "__main__":
    main()
