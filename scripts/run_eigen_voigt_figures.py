"""Stage EV-3 figures for the eigen-voigt-embedding track.

PYTHONPATH=. uv run python scripts/run_eigen_voigt_figures.py --mode full
"""

import argparse
from pathlib import Path

from src.core.eigen_viz import fig_b_distributions, fig_le_minus_pca, fig_representation_ba

RES = Path("results/eigen_voigt_embedding")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("smoke", "full"), default="full")
    args = ap.parse_args()
    figs = RES / "figs"
    figs.mkdir(parents=True, exist_ok=True)
    dirs = {v: RES / f"ev2_{v}_{args.mode}" for v in ("fiducial", "taurescaled")}
    for p in (fig_le_minus_pca(dirs, figs / f"fig_le_minus_pca_{args.mode}.png"),
              fig_representation_ba(dirs, figs / f"fig_representation_ba_{args.mode}.png"),
              fig_b_distributions(RES / "artifacts", "full", figs / "fig_voigt_b_by_class.png")):
        print(f"[ev-3] {p}")


if __name__ == "__main__":
    main()
