"""Thin orchestration wrapper: export the primer synthetic spectrum for selements-website.

No logic here — argparse + a single call into
:func:`src.core.export.export_primer_synthetic_spectrum`. Run from the repo root:

    PYTHONPATH=. uv run python scripts/export/primer_synthetic_spectrum.py
"""

import argparse
from pathlib import Path

from src.core.export import (
    SELEMENTS_WEBSITE_ROOT,
    export_primer_synthetic_spectrum,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export the primer synthetic spectrum CSV + provenance for selements-website."
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=SELEMENTS_WEBSITE_ROOT / "primer-synthetic-spectrum",
        help="Landing directory for the export (default: canonical selements-website path).",
    )
    parser.add_argument("--class-id", type=int, default=1, help="Physics class id (default 1=NoFeedback).")
    parser.add_argument("--sightline-idx", type=int, default=0, help="Sightline row index (default 0).")
    args = parser.parse_args()

    csv_path = export_primer_synthetic_spectrum(
        out_dir=args.out_dir,
        class_id=args.class_id,
        sightline_idx=args.sightline_idx,
    )
    print(f"Wrote CSV: {csv_path}")
    print(f"Wrote provenance: {csv_path.with_name('synthetic-spectrum.example.provenance.json')}")


if __name__ == "__main__":
    main()
