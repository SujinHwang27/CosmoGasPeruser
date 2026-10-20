#!/usr/bin/env bash
# lit-claims-audit EDGE_MAP checks: E2/E3/E4 first (no new catalogs), then the E1 broadened
# forward-modelled catalogs (classes 1 and 2, S/N 30, dndz12, noise seed 0), then E1. Resumable.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=. MPLBACKEND=Agg
uv run python scripts/run_claims_edges.py --checks e2,e3,e4
for bx in 3 6 9 12 15 18 21; do
  uv run python scripts/run_claims_t2.py --name e1_bx${bx} --A-key dndz12 --snr 30 --seed 0 --b-extra ${bx} --classes 1,2
done
uv run python scripts/run_claims_edges.py --checks e1
echo "[claims-edges] complete"
