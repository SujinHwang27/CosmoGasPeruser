"""Thin orchestration wrapper: export the class-mean P_F(k) table for selements-website.

No logic here — argparse + a single call into
:func:`src.core.export.export_exploration_pk_mean_per_class`. Run from the repo root:

    PYTHONPATH=. uv run python scripts/export/exploration_pk_mean_per_class.py
"""

import argparse
from pathlib import Path

from src.core.export import (
    SELEMENTS_WEBSITE_ROOT,
    export_exploration_pk_mean_per_class,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export the class-mean flux power spectrum P_F(k) CSV + provenance for selements-website."
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=SELEMENTS_WEBSITE_ROOT / "exploration-pk-mean-per-class",
        help="Landing directory for the export (default: canonical selements-website path).",
    )
    parser.add_argument(
        "--n-kbins",
        type=int,
        default=20,
        help="Number of log-spaced k-bins (default 20, project canonical).",
    )
    args = parser.parse_args()

    csv_path = export_exploration_pk_mean_per_class(
        out_dir=args.out_dir,
        n_kbins=args.n_kbins,
    )
    print(f"Wrote CSV: {csv_path}")
    print(f"Wrote provenance: {csv_path.with_name('pk-mean-per-class.provenance.json')}")


if __name__ == "__main__":
    main()
