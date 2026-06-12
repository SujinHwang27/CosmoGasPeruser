#!/bin/bash
#SBATCH --job-name=rs-hardening
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=24G
#SBATCH --time=24:00:00
#SBATCH --output=cloud_runs/rshardening-%j.out
#SBATCH --error=cloud_runs/rshardening-%j.err
# --partition deliberately NOT pinned here — the submit wrapper passes
# `--partition=${JUNO_CPU_PARTITION}` from .env so the partition name is
# overrideable without editing this script. See .claude/skills/juno-hpc/SKILL.md.
#
# reframe-suite HARDENING sprint — ALL THREE compute items, run sequentially in
# ONE job per experiments/reframe-suite/HARDENING_SPEC.md §2/§3/§4/§8:
#   (ii) ⟨F⟩-scalar baseline     — hardening_fbar_baseline.py   (minutes)
#   (i)  binary pair classifiers — hardening_pair_binary.py     (the long pole;
#        M=1 per-sightline tail dominates, cf. Stage-1 [D-25] ~10-15h experience)
#   (iii) injection-recovery     — hardening_injection.py       (minutes)
# Order ii→i→iii matches the spec dependency chain (H3 needs ii+i; iii's α_equiv
# consumes i's C2-C3 number). CPU-only RandomForest, MLflow-free per [D-01].
# 24h wallclock ceiling mirrors the Stage-1 [D-25] decision; `normal` allows 2d.
# All steps idempotent-resume from their *_cv.csv, so a wallclock-kill is
# recoverable by re-firing sbatch.
#
# PCV (§4 below) hard-asserts the full artifact set with DISTINCT FATAL exit
# codes 10-16 — shifted out of the 2-7 namespace to avoid colliding with the
# scripts' own fold-leakage assert (exit 7).
#
# MANUAL RESULT SYNC (from local laptop, after completion):
#   rsync -avzP juno:${JUNO_WORK}/cloud_runs/<RUN_TAG>/results/reframe_suite/hardening/ \
#               results/reframe_suite/hardening/

set -euo pipefail

# --- 0. Source Juno-side .env if present (JUNO_*) ---
if [[ -f "${JUNO_WORK:-$HOME/work/CosmoGasPeruser}/.env" ]]; then
  set -a
  source "${JUNO_WORK:-$HOME/work/CosmoGasPeruser}/.env"
  set +a
fi
: "${JUNO_WORK:=$HOME/work/CosmoGasPeruser}"
: "${JUNO_SCRATCH:=$HOME/scratch}"
: "${JUNO_CPU_PARTITION:=normal}"

RUN_TAG="rshardening-$(date -u +%Y%m%d-%H%M%S)-$(openssl rand -hex 3)"
RUN_DIR="${JUNO_SCRATCH}/reframe-suite-hardening/${RUN_TAG}"
DEST="${JUNO_WORK}/cloud_runs/${RUN_TAG}"
mkdir -p "${RUN_DIR}" "${DEST}"

cat <<EOF
=== ECONOMIC COMPUTE PLAN ===
(a) instance: Juno ${JUNO_CPU_PARTITION} partition, 1 node × 16 cores × 24 GB
(b) hours: 24h wallclock cap (item-i M=1 tail is the long pole)
(c)/(d) cost: \$0 marginal (Fairshare-aware queueing)
(e) auto-stop: SLURM wallclock + set -euo pipefail cleanup
(f) lifecycle: scratch run dir → MFS cloud_runs/${RUN_TAG}/ → local rsync (see header)
============================
EOF

cd "${RUN_DIR}"

# --- 1. Copy in source + symlink data (data mirror already on scratch) ---
cp -r "${JUNO_WORK}"/{src,experiments,scripts,pyproject.toml,uv.lock} .
[[ -f "${JUNO_WORK}/.python-version" ]] && cp "${JUNO_WORK}/.python-version" . || true
mkdir -p data/preprocessed
ln -s "${JUNO_SCRATCH}/data/preprocessed/Sherwood_z0.3_inf" data/preprocessed/Sherwood_z0.3_inf
[[ -d "data/preprocessed/Sherwood_z0.3_inf/1" ]] || {
  echo "FATAL: Sherwood mirror missing at ${JUNO_SCRATCH}/data/preprocessed/Sherwood_z0.3_inf/" >&2
  echo "       run scripts/sync_data_to_juno.sh from the local laptop first." >&2
  exit 1
}

# --- 2. Environment (uv-created .venv at repo root) ---
source "${JUNO_WORK}/.venv/bin/activate"
export PYTHONPATH=.
export PYTHONUNBUFFERED=1
export GIT_PYTHON_REFRESH=quiet
export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16
export OPENBLAS_NUM_THREADS=16

echo "=== run config ==="
echo "RUN_TAG=${RUN_TAG}  RUN_DIR=${RUN_DIR}"
echo "Python: $(which python)  $(python --version)"
echo "CPUs: $(nproc); SLURM allocated: ${SLURM_CPUS_PER_TASK:-?}; OMP/MKL/OPENBLAS=${OMP_NUM_THREADS}"
free -g | head -2

OUT="results/reframe_suite/hardening"

# --- 3. Compute — items ii → i → iii (sequential; spec dependency order) ---
echo; echo "########## ITEM (ii) ⟨F⟩-baseline ##########"
PYTHONPATH=. python -u experiments/reframe-suite/scripts/hardening_fbar_baseline.py \
  --out-dir "${OUT}" 2>&1 | tee hardening_fbar_baseline.log

echo; echo "########## ITEM (i) binary pair classifiers (long pole) ##########"
PYTHONPATH=. python -u experiments/reframe-suite/scripts/hardening_pair_binary.py \
  --out-dir "${OUT}" 2>&1 | tee hardening_pair_binary.log

echo; echo "########## ITEM (iii) injection-recovery ##########"
PYTHONPATH=. python -u experiments/reframe-suite/scripts/hardening_injection.py \
  --out-dir "${OUT}" 2>&1 | tee hardening_injection.log

# --- 4. Producer-Consumer Verification (distinct FATAL exit codes 10-16) ---
[[ -d "${OUT}" ]] || { echo "FATAL[10]: producer wrote no ${OUT} dir" >&2; exit 10; }

# item (ii)
[[ -s "${OUT}/fbar_baseline_cv.csv" ]]      || { echo "FATAL[11]: fbar_baseline_cv.csv missing/empty" >&2; exit 11; }
[[ -s "${OUT}/fbar_baseline_summary.csv" ]] || { echo "FATAL[11]: fbar_baseline_summary.csv missing/empty" >&2; exit 11; }

# item (i) — expected cv rows: 2 regimes × 7 M × 7 tasks × 10 seeds × 5 folds = 4900
#           + saturation (C2-C3, per_sightline, M∈{512,1024}) × 10 seeds × 5 folds = 100 → 5000
PAIR_CV="${OUT}/pair_binary_cv.csv"
[[ -s "${PAIR_CV}" ]] || { echo "FATAL[12]: pair_binary_cv.csv missing/empty" >&2; exit 12; }
PAIR_ROWS=$(wc -l < "${PAIR_CV}")
if [[ "${PAIR_ROWS}" -le 4900 ]]; then
  echo "FATAL[13]: ${PAIR_CV} has ${PAIR_ROWS} lines, expected > 4900 (~5000 data rows + header)." >&2
  echo "           Likely a wallclock-cap-kill mid-sweep; partial CSV survives (idempotent resume)." >&2
  echo "           Re-fire sbatch to continue from completed tuples." >&2
  exit 13
fi
[[ -s "${OUT}/pair_binary_summary.csv" ]] || { echo "FATAL[13]: pair_binary_summary.csv missing/empty" >&2; exit 13; }

# item (iii) — expected cv rows: 8 alphas × 2 M × 10 seeds × 5 folds = 800
INJ_CV="${OUT}/injection_recovery_cv.csv"
[[ -s "${INJ_CV}" ]] || { echo "FATAL[14]: injection_recovery_cv.csv missing/empty" >&2; exit 14; }
INJ_ROWS=$(wc -l < "${INJ_CV}")
[[ "${INJ_ROWS}" -gt 800 ]] || { echo "FATAL[15]: ${INJ_CV} has ${INJ_ROWS} lines, expected > 800." >&2; exit 15; }
[[ -s "${OUT}/injection_recovery_summary.csv" ]] || { echo "FATAL[15]: injection_recovery_summary.csv missing/empty" >&2; exit 15; }
[[ -s "${OUT}/injection_templates.csv" ]]        || { echo "FATAL[16]: injection_templates.csv missing/empty" >&2; exit 16; }

echo "=== PCV PASS — all hardening artifacts present ==="
echo "  fbar_baseline_cv.csv:        present"
echo "  pair_binary_cv.csv:          ${PAIR_ROWS} lines (>4900)"
echo "  injection_recovery_cv.csv:   ${INJ_ROWS} lines (>800)"
echo "  injection_templates.csv:     present"

# --- 5. Copy out to MFS, relocate SLURM .out/.err ---
mkdir -p "${DEST}/results/reframe_suite"
cp -r "${OUT}" "${DEST}/results/reframe_suite/"
cp hardening_*.log "${DEST}/" 2>/dev/null || true
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  mv "${JUNO_WORK}/cloud_runs/rshardening-${SLURM_JOB_ID}.out" "${DEST}/" 2>/dev/null || true
  mv "${JUNO_WORK}/cloud_runs/rshardening-${SLURM_JOB_ID}.err" "${DEST}/" 2>/dev/null || true
fi

echo "=== copied artifacts ==="
find "${DEST}" -maxdepth 5 -type f | head -50

# --- 6. Cleanup ---
cd "${HOME}"
rm -rf "${RUN_DIR}"
echo "=== done, results at ${DEST} ==="
echo "Next (local laptop): rsync -avzP juno:${DEST}/results/reframe_suite/hardening/ results/reframe_suite/hardening/"
