#!/usr/bin/env bash
# lit-claims-audit defense-panel follow-ups: combined realistic instrument run (attack 8), then the mechanism
# (attack 1a), transfer-function (attack 2), suite-identity (attack 7) and realistic-run analyses.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=. MPLBACKEND=Agg
until grep -q "claims-edges\] complete" results/lit_claims_audit/logs/edges.log 2>/dev/null; do sleep 60; done
uv run python scripts/run_claims_t2.py --name dndz12_realistic_s0 --A-key dndz12 --snr 15 --seed 0 --lsf gauss \
  --lsf-fwhm 19 --noise sqrtF --cont 0.02 --bturb 20
uv run python scripts/run_claims_edges.py --checks x7,x2,m1,x8
echo "[claims-edges2] complete"
