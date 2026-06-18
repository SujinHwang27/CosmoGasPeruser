"""Thin orchestration wrapper: export the four Tier-1 2-D relationship figures for selements-website.

No logic here — argparse + a single call into
:func:`src.core.export.export_exploration_relationships_2d`. Run from the repo root:

    PYTHONPATH=. uv run python scripts/export/exploration_relationships_2d.py
"""

import argparse
from pathlib import Path

from src.core.export import (
    SELEMENTS_WEBSITE_ROOT,
    export_exploration_relationships_2d,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export the four Tier-1 EDA 2-D binned-density relationship CSVs + provenance for selements-website."
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=SELEMENTS_WEBSITE_ROOT / "exploration-relationships-2d",
        help="Landing directory for the export (default: canonical selements-website path).",
    )
    parser.add_argument(
        "--n-edges",
        type=int,
        default=40,
        help="Number of bin edges per axis (default 40 -> 39 bins).",
    )
    args = parser.parse_args()

    out_dir = export_exploration_relationships_2d(
        out_dir=args.out_dir,
        n_edges=args.n_edges,
    )
    print(f"Wrote 4 figure CSVs (+ trends) + provenance to: {out_dir}")


if __name__ == "__main__":
    main()
