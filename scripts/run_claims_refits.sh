#!/usr/bin/env bash
# lit-claims-audit refit queue (T1/T3 matched variants, rescaling-only null, tol sweep). Resumable.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=. MPLBACKEND=Agg
R="uv run python scripts/run_claims_catalogs.py"
$R --name dndz12 --factors-key dndz12
$R --name null2_dndz12 --A 1,0.8203515443297114,1,1 --classes 2
$R --name dndz13 --factors-key dndz13
$R --name cddf --factors-key cddf
$R --name dndz13p5 --factors-key dndz13p5
$R --name tol005 --A 1,1,1,1 --classes 2,3 --tol 0.005
$R --name tol002 --A 1,1,1,1 --classes 2,3 --tol 0.002
echo "[claims-refits] complete"
