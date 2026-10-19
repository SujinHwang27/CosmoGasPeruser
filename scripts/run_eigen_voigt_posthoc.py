"""Stage EV-3 catalog-only post-hoc checks for the eigen-voigt-embedding track.

PYTHONPATH=. uv run python scripts/run_eigen_voigt_posthoc.py
"""

import json
import os
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

from src.core.eigen_voigt import catalog_posthoc

ART = Path("results/eigen_voigt_embedding/artifacts")
OUT = Path("results/eigen_voigt_embedding/ev3_catalog_posthoc.json")


def main() -> None:
    res = catalog_posthoc(ART, mode="full")
    OUT.write_text(json.dumps(res, indent=2))
    print(f"[ev-3] wrote {OUT}")


if __name__ == "__main__":
    main()
