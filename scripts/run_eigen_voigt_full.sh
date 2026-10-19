#!/usr/bin/env bash
# Full run of the eigen-voigt-embedding track (EV-1 catalogs -> EV-2 evaluation), both
# data variants. Every stage is resumable, so re-running this script continues a run.
# Usage: nohup bash scripts/run_eigen_voigt_full.sh > results/eigen_voigt_embedding/logs/full.log 2>&1 &
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=. MPLBACKEND=Agg
while pgrep -f "run_eigen_voigt_catalog.py --mode full" > /dev/null; do sleep 60; done
for variant in fiducial taurescaled; do
  uv run python scripts/run_eigen_voigt_catalog.py --mode full --variant "$variant"
done
for variant in fiducial taurescaled; do
  uv run python scripts/run_eigen_voigt_eval.py --mode full --variant "$variant" --seeds 10
done
echo "[full] eigen-voigt-embedding full run complete"
