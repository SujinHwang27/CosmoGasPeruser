#!/usr/bin/env bash
# lit-claims-audit T1 analysis (started once the matched variants are verified; tol sweep awaited inside).
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHONPATH=. MPLBACKEND=Agg nice -n 5 uv run python scripts/run_claims_t1.py
