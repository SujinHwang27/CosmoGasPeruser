#!/usr/bin/env bash
# lit-claims-audit T2 follow-up: rescaling-only null under noise (class 2 at A3, independent noise seed),
# then the T2 analysis. Waits for the main T2 queue.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=. MPLBACKEND=Agg
until grep -q "claims-t2\] complete" results/lit_claims_audit/logs/t2.log 2>/dev/null; do sleep 120; done
T="uv run python scripts/run_claims_t2.py"
$T --name null2_dndz12_snr30_s7 --A-key dndz12 --A-from-class 3 --classes 2 --snr 30 --seed 7
uv run python scripts/run_claims_t2_analysis.py
echo "[claims-t2-followup] complete"
