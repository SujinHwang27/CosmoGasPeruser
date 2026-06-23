"""Thin orchestration wrapper: export v2 separability-vector L2 norms for selements-website.

No logic here — argparse + a single call into
:func:`src.core.export.export_signalclustering_v2_sep_norms`. Run from the repo root:

    PYTHONPATH=. uv run python scripts/export/signalclustering_v2_sep_norms.py
"""

import argparse
from pathlib import Path

from src.core.export import (
    SELEMENTS_WEBSITE_ROOT,
    export_signalclustering_v2_sep_norms,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export per-sightline separability-vector L2 norms (wavelet vs raw, v2) for selements-website."
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=SELEMENTS_WEBSITE_ROOT / "signal-clustering-v2-sep-norms",
        help="Landing directory for the export (default: canonical selements-website path).",
    )
    args = parser.parse_args()

    csv_path = export_signalclustering_v2_sep_norms(out_dir=args.out_dir)
    print(f"Wrote norms CSV: {csv_path}")
    print(f"Wrote summary CSV: {csv_path.with_name('sep-norms-summary.csv')}")
    print(f"Wrote provenance: {csv_path.with_name('sep-norms.provenance.json')}")


if __name__ == "__main__":
    main()
