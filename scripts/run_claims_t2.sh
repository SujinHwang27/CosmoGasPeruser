#!/usr/bin/env bash
# lit-claims-audit T2 queue: waits for the refit queue and the COS LSF, then runs the
# pre-registered forward-model grid (CLAIM_TEST_PLAN amendment 5). Resumable.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=. MPLBACKEND=Agg
until grep -q "claims-refits\] complete" results/lit_claims_audit/logs/refits.log 2>/dev/null; do sleep 120; done
until [ -f data/raw/Sherwood/COS_LSF1_cw1327_75_z0.100.dat ]; do sleep 120; done
T="uv run python scripts/run_claims_t2.py"
[ -f results/lit_claims_audit/t2_injection_cos.json ] || $T --inject --lsf cos
[ -f results/lit_claims_audit/t2_injection_gauss.json ] || $T --inject --lsf gauss --snrs 30
for snr in 30 inf 10 20 50; do $T --name dndz12_snr${snr}_s0 --A-key dndz12 --snr $snr --seed 0; done
for s in 1 2 3 4; do $T --name dndz12_snr30_s$s --A-key dndz12 --snr 30 --seed $s; done
$T --name meanF_snr30_s0 --A-key meanF --snr 30 --seed 0
$T --name dndz12_snr30_s0_gauss --A-key dndz12 --snr 30 --seed 0 --lsf gauss
$T --name dndz12_snr30_s0_sqrtF --A-key dndz12 --snr 30 --seed 0 --noise sqrtF
$T --name dndz12_snr30_s0_aicc10 --A-key dndz12 --snr 30 --seed 0 --daicc -10
$T --name dndz12_snr30_s0_bturb20 --A-key dndz12 --snr 30 --seed 0 --bturb 20
$T --name dndz12_snr30_s0_cont0 --A-key dndz12 --snr 30 --seed 0 --cont 0
echo "[claims-t2] complete"
