#!/bin/bash
#SBATCH --job-name=pk-stage1-smoke
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=1:00:00
#SBATCH --output=cloud_runs/pkstage1smoke-%j.out
#SBATCH --error=cloud_runs/pkstage1smoke-%j.err
# --partition NOT pinned — passed via `sbatch --partition=${JUNO_CPU_PARTITION}` from the wrapper.

# Stage 1 SMOKE-TIMING wrapper — C4 compute-ceiling validation per
# experiments/pk-feedback-classifier/LEDGER.md [D-23] C4. Single 1-fold
# pass at M ∈ {4, 64} single-seed, intended to (a) exercise the producer
# → copy-out → PCV path end-to-end, and (b) measure per-fit wallclock so
# the full-submission 6h ceiling can be sanity-checked against the
# extrapolated ~7940-fit budget BEFORE committing 6 wallclock-hours.
#
# Relies on run_stage1.py exposing a `--smoke` flag (1-fold / 1-seed /
# M∈{4,64} only; per the core-implementer brief). If `--smoke` is not
# supported the run will fail loudly at argparse; surface that back to
# the user — it's not silently shimmable from the wrapper.

set -euo pipefail

# --- 0. Source Juno-side .env (JUNO_* only per [D-05] anti-pattern) ---
if [[ -f "${JUNO_WORK:-$HOME/work/CosmoGasPeruser}/.env" ]]; then
  set -a
  source "${JUNO_WORK:-$HOME/work/CosmoGasPeruser}/.env"
  set +a
fi
: "${JUNO_WORK:=$HOME/work/CosmoGasPeruser}"
: "${JUNO_SCRATCH:=$HOME/scratch}"
: "${JUNO_CPU_PARTITION:=normal}"

RUN_TAG="pkstage1smoke-$(date -u +%Y%m%d-%H%M%S)-$(openssl rand -hex 3)"
RUN_DIR="${JUNO_SCRATCH}/pk-feedback-classifier/${RUN_TAG}"
DEST="${JUNO_WORK}/cloud_runs/${RUN_TAG}"
mkdir -p "${RUN_DIR}" "${DEST}"

cat <<EOF
=== ECONOMIC COMPUTE PLAN (smoke) ===
(a) instance: Juno ${JUNO_CPU_PARTITION} partition, 1 node × 16 cores × 32 GB
(b) hours: 1h wallclock cap (smoke-timing, NOT a full sweep)
(c)/(d) cost: \$0 marginal (Fairshare-aware queueing)
(e) auto-stop: SLURM wallclock + set -euo pipefail cleanup
(f) lifecycle: scratch run dir → MFS cloud_runs/${RUN_TAG}/ → local rsync via scripts/sync_results_from_juno.sh ${RUN_TAG}
(g) purpose: C4 compute-ceiling validation per [D-23]; extrapolates per-fit wallclock to gate full-submission go-decision
====================================
EOF

cd "${RUN_DIR}"

# --- 1. Copy in source + symlink data ---
cp -r "${JUNO_WORK}"/{src,experiments,scripts,pyproject.toml,uv.lock} .
[[ -f "${JUNO_WORK}/.python-version" ]] && cp "${JUNO_WORK}/.python-version" . || true
mkdir -p data/preprocessed
ln -s "${JUNO_SCRATCH}/data/preprocessed/Sherwood_z0.3_inf" data/preprocessed/Sherwood_z0.3_inf
[[ -d "data/preprocessed/Sherwood_z0.3_inf/1" ]] || {
  echo "FATAL: Sherwood mirror missing at ${JUNO_SCRATCH}/data/preprocessed/Sherwood_z0.3_inf/" >&2
  exit 1
}

# --- 2. Environment (conda donates Py3.12; uv resolves deps; uv run honours lockfile) ---
source ~/miniconda3/etc/profile.d/conda.sh
conda activate /work/sxh240010/envs/cosmogasperuser

export PYTHONPATH=.
export PYTHONUNBUFFERED=1
export GIT_PYTHON_REFRESH=quiet
export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16
export OPENBLAS_NUM_THREADS=16

echo "=== run config (smoke) ==="
echo "RUN_TAG=${RUN_TAG}"
echo "Python: $(which python)  $(python --version)"
echo "CPUs available: $(nproc); SLURM allocated: ${SLURM_CPUS_PER_TASK:-?}"

# --- 3. Compute (smoke — 1 fold, 1 seed, M∈{4,64} only) ---
START_TS=$(date +%s)
PYTHONPATH=. uv run python experiments/pk-feedback-classifier/run_stage1.py --run-tag "${RUN_TAG}" --smoke 2>&1 | tee run_stage1_smoke.log
END_TS=$(date +%s)
SMOKE_WALLCLOCK_SEC=$((END_TS - START_TS))

# --- 4. Light PCV — cv_partial.csv exists with >= 1 data row (header + 1 line = 2 lines minimum) ---
# Distinct FATAL exit codes in the 10-15 namespace to stay clear of run_stage1.py's exit 7
# (fold-leakage assert per [D-20] S3 / [D-23] C6).
RESULTS_DIR="results/pk_feedback_classifier/stage1"
[[ -d "${RESULTS_DIR}" ]] || { echo "FATAL[10]: producer wrote no ${RESULTS_DIR} dir" >&2; exit 10; }

CV_PARTIAL="${RESULTS_DIR}/cv_partial.csv"
[[ -s "${CV_PARTIAL}" ]] || { echo "FATAL[10]: ${CV_PARTIAL} missing or empty" >&2; exit 10; }
CV_ROWS=$(wc -l < "${CV_PARTIAL}")
[[ "${CV_ROWS}" -ge 2 ]] || { echo "FATAL[11]: ${CV_PARTIAL} has ${CV_ROWS} lines, expected >= 2 (header + >=1 data row)." >&2; exit 11; }
SMOKE_N_FITS=$((CV_ROWS - 1))

# --- 5. Copy out ---
mkdir -p "${DEST}/results"
cp -r "${RESULTS_DIR}" "${DEST}/results/"
cp run_stage1_smoke.log "${DEST}/"
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  mv "${JUNO_WORK}/cloud_runs/pkstage1smoke-${SLURM_JOB_ID}.out" "${DEST}/" 2>/dev/null || true
  mv "${JUNO_WORK}/cloud_runs/pkstage1smoke-${SLURM_JOB_ID}.err" "${DEST}/" 2>/dev/null || true
fi

# --- 6. Wallclock extrapolation hint ---
# Full sweep: 2 regimes × 7 M × 10 seeds × 5 folds = 700 fits (G1/G3/G4 backbone),
# plus 5000 G2 permutation-null fits = ~5700 backbone+null fits, plus G6 ablation ~1400 fits
# → ~7100 fits in the high-level [D-09] estimate (brief uses ~7940 as the conservative number).
FULL_N_FITS=7940
if [[ "${SMOKE_N_FITS}" -gt 0 ]]; then
  EST_FULL_SEC=$((SMOKE_WALLCLOCK_SEC * FULL_N_FITS / SMOKE_N_FITS))
  EST_FULL_HR=$(awk -v s="${EST_FULL_SEC}" 'BEGIN{ printf "%.2f", s/3600.0 }')
  cat <<EOF
=== SMOKE-TIMING EXTRAPOLATION ===
Smoke wallclock:           ${SMOKE_WALLCLOCK_SEC}s
Smoke fits (cv_partial):   ${SMOKE_N_FITS}
Per-fit:                   $(awk -v s="${SMOKE_WALLCLOCK_SEC}" -v n="${SMOKE_N_FITS}" 'BEGIN{ printf "%.3f", s/n }') s
Full-sweep fit count:      ${FULL_N_FITS}  (2 regimes × 7 M × 10 seeds × 5 folds + G2 null + G6 ablation)
Full-sweep est. wallclock: ~${EST_FULL_HR} h
[D-09] ceiling:            6.00 h
Verdict:                   $(awk -v est="${EST_FULL_SEC}" 'BEGIN{ if (est < 6*3600) print "WITHIN ceiling — full submission OK"; else print "EXCEEDS 6h ceiling — re-spec compute or reduce sweep BEFORE firing full submission" }')

NOTE: smoke is single-fold / single-seed at M∈{4,64}; extrapolation
assumes linear-in-fit-count scaling. RF wallclock is approximately
constant per fit at fixed (n_est=300, n_jobs=-1, n_samples ~ N/M),
so the linear extrapolation is reasonable; M=1 fits are SLOWER per
fit than M=256 (more samples), so the per-fit average across the
M-sweep will exceed the M∈{4,64} smoke average — bake in ~20%
headroom on the extrapolated wallclock before declaring WITHIN.
================================
EOF
fi

echo "=== copied artifacts ==="
find "${DEST}" -maxdepth 4 -type f | head -40

# --- 7. Cleanup ---
cd "${HOME}"
rm -rf "${RUN_DIR}"
echo "=== smoke done, results at ${DEST} ==="
echo "Next: review extrapolation; if WITHIN ceiling + PI APPROVE on C5 + CI green on C6, fire scripts/submit_juno_stage1.sh"
