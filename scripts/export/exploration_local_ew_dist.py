"""Thin orchestration wrapper: export the per-class local-EW distribution for selements-website.

No logic here — argparse + a single call into
:func:`src.core.export.export_exploration_local_ew_dist_per_class`. Run from the repo root:

    PYTHONPATH=. uv run python scripts/export/exploration_local_ew_dist.py
"""

import argparse
from pathlib import Path

from src.core.export import (
    SELEMENTS_WEBSITE_ROOT,
    export_exploration_local_ew_dist_per_class,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export the per-class local equivalent-width distribution CSVs + provenance for selements-website."
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=SELEMENTS_WEBSITE_ROOT / "exploration-local-ew-dist",
        help="Landing directory for the export (default: canonical selements-website path).",
    )
    parser.add_argument(
        "--n-edges",
        type=int,
        default=40,
        help="Number of log2 bin edges (default 40 -> 39 bins; matches the published figure).",
    )
    args = parser.parse_args()

    csv_path = export_exploration_local_ew_dist_per_class(
        out_dir=args.out_dir,
        n_edges=args.n_edges,
    )
    print(f"Wrote distribution CSV: {csv_path}")
    print(f"Wrote summary CSV: {csv_path.with_name('local-ew-summary-per-class.csv')}")
    print(f"Wrote provenance: {csv_path.with_name('local-ew-dist-per-class.provenance.json')}")


if __name__ == "__main__":
    main()
