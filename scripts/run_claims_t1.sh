#!/usr/bin/env bash
# lit-claims-audit T1 analysis: waits for the refit queue, then runs scripts/run_claims_t1.py.
set -euo pipefail
cd "$(dirname "$0")/.."
until grep -q "claims-refits\] complete" results/lit_claims_audit/logs/refits.log 2>/dev/null; do sleep 120; done
PYTHONPATH=. MPLBACKEND=Agg nice -n 5 uv run python scripts/run_claims_t1.py
