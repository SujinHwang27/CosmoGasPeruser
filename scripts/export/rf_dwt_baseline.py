"""Thin orchestration wrapper: export the RF + DWT baseline-study data for selements-website.

No logic here — argparse + calls into the two src.core.export functions that need no
re-run (the 9-variant accuracy table, recorded; and the per-level DWT energy, computed
fresh). Confusion matrices are handled separately (they require an RF re-run). Run from
the repo root:

    PYTHONPATH=. uv run python scripts/export/rf_dwt_baseline.py
"""

import argparse
from pathlib import Path

from src.core.export import (
    SELEMENTS_WEBSITE_ROOT,
    export_episode4_mean_energy_per_level,
    export_episode4_rf_baseline_summary,
    export_episode4_rf_confusion_matrices,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export the RF+DWT baseline accuracy table + per-level energy for selements-website."
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=SELEMENTS_WEBSITE_ROOT / "rf-dwt-baseline",
        help="Landing directory for the export (default: canonical selements-website path).",
    )
    parser.add_argument(
        "--confusion",
        action="store_true",
        help="Also re-run RF_Raw/D1/D6 to export numeric confusion matrices (~10-20 min).",
    )
    args = parser.parse_args()

    acc = export_episode4_rf_baseline_summary(out_dir=args.out_dir)
    energy = export_episode4_mean_energy_per_level(out_dir=args.out_dir)
    print(f"Wrote accuracy table: {acc}")
    print(f"Wrote per-level energy: {energy}")
    if args.confusion:
        cm = export_episode4_rf_confusion_matrices(out_dir=args.out_dir)
        print(f"Wrote confusion matrices: {cm}")


if __name__ == "__main__":
    main()
