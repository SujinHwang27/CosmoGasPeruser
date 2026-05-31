#!/bin/bash
#SBATCH --job-name=PkProbe
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=24G
#SBATCH --time=1-00:00:00
#SBATCH --output=pkprobe-%j.out
#SBATCH --error=pkprobe-%j.err
# --partition deliberately NOT pinned here — the submit wrapper passes
# `--partition=${JUNO_CPU_PARTITION}` from .env so the partition name is
# overrideable without editing this script. See .claude/skills/juno-hpc/SKILL.md.

# Single CPU job: the experiments/pk-feedback-classifier/run_probe.py
# de-risking probe (P_F(k) -> global 4-class RF, per-sightline + stacked
# M in {4,16,64}, both norm regimes). See experiments/pk-feedback-classifier/LEDGER.md
# for the committed spec ([D-01] through [D-04]). The probe is intentionally
# MLflow-free per [D-01]; outputs are CSVs + PNGs only.
#
# PCV (Producer-Consumer Verification) — see infrastructure-manager agent for
# the framework. This script asserts the probe's expected output set is
# fully present before cleanup wipes the run dir.

set -euo pipefail

# --- 0. Source Juno-side .env if present (JUNO_WORK / JUNO_SCRATCH overrides) ---
if [[ -f "${JUNO_WORK:-$HOME/work/CosmoGasPeruser}/.env" ]]; then
  set -a
  source "${JUNO_WORK:-$HOME/work/CosmoGasPeruser}/.env"
  set +a
fi
: "${JUNO_WORK:=$HOME/work/CosmoGasPeruser}"
: "${JUNO_SCRATCH:=$HOME/scratch}"

RUN_TAG="pkprobe-$(date +%Y%m%d-%H%M%S)-$(uuidgen | cut -c1-6)"
RUN_DIR="${JUNO_SCRATCH}/pk-feedback-classifier/${RUN_TAG}"
mkdir -p "${RUN_DIR}"
cd "${RUN_DIR}"

# --- 1. Copy in source + symlink data (data already mirrored once via sync_data_to_juno.sh) ---
cp -r "${JUNO_WORK}"/{src,experiments,scripts,pyproject.toml,uv.lock} .
# .python-version may not exist on every clone; copy if present, ignore if not.
[[ -f "${JUNO_WORK}/.python-version" ]] && cp "${JUNO_WORK}/.python-version" . || true
mkdir -p data/preprocessed
ln -s "${JUNO_SCRATCH}/data/preprocessed/Sherwood_z0.3_inf" data/preprocessed/Sherwood_z0.3_inf
[[ -d "data/preprocessed/Sherwood_z0.3_inf/1" ]] || {
  echo "FATAL: Sherwood mirror missing at ${JUNO_SCRATCH}/data/preprocessed/Sherwood_z0.3_inf/" >&2
  echo "       run scripts/sync_data_to_juno.sh from the local laptop first." >&2
  exit 1
}

# --- 2. Environment ---
source "${JUNO_WORK}/.venv/bin/activate"
export PYTHONPATH=.
export PYTHONUNBUFFERED=1
# Probe is intentionally MLflow-free per [D-01]; no MLFLOW_TRACKING_URI export.
export GIT_PYTHON_REFRESH=quiet
# Cap thread fan-out so we don't oversubscribe vs SLURM allocation.
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"

echo "=== run config ==="
echo "RUN_TAG=${RUN_TAG}"
echo "RUN_DIR=${RUN_DIR}"
echo "Python: $(which python)  $(python --version)"
echo "CPUs available: $(nproc); SLURM allocated: ${SLURM_CPUS_PER_TASK:-?}; OMP/MKL/OPENBLAS=${OMP_NUM_THREADS}"
free -g | head -2
df -h "${RUN_DIR}" "${JUNO_SCRATCH}" 2>/dev/null || true

# --- 3. Compute ---
python -u experiments/pk-feedback-classifier/run_probe.py 2>&1 | tee run_probe.log

# --- 4. Copy out + Producer-Consumer Verification (PCV) ---
# run_probe.py writes to results/pk_feedback_classifier/ relative to RUN_DIR.
# Hard-assert: balanced_acc_summary.csv, >=1 confusion CSV, >=1 kbin_importance CSV, >=1 PNG.
DEST="${JUNO_WORK}/cloud_runs/${RUN_TAG}"
mkdir -p "${DEST}"

RES_SRC="results/pk_feedback_classifier"
[[ -d "${RES_SRC}" ]] || { echo "FATAL: probe produced no ${RES_SRC} dir" >&2; exit 2; }
[[ -f "${RES_SRC}/balanced_acc_summary.csv" ]] || { echo "FATAL: missing ${RES_SRC}/balanced_acc_summary.csv" >&2; exit 3; }
N_CONF=$(ls -1 "${RES_SRC}"/confusion_*.csv 2>/dev/null | wc -l)
[[ "${N_CONF}" -gt 0 ]] || { echo "FATAL: zero confusion_*.csv produced" >&2; exit 4; }
N_IMP=$(ls -1 "${RES_SRC}"/kbin_importance_*.csv 2>/dev/null | wc -l)
[[ "${N_IMP}" -gt 0 ]] || { echo "FATAL: zero kbin_importance_*.csv produced" >&2; exit 5; }
N_FIG=$(ls -1 "${RES_SRC}"/figs/*.png 2>/dev/null | wc -l)
[[ "${N_FIG}" -gt 0 ]] || { echo "FATAL: zero figures in ${RES_SRC}/figs/" >&2; exit 6; }

cp -r "${RES_SRC}" "${DEST}/"
cp run_probe.log "${DEST}/"

echo "=== copied artifacts ==="
find "${DEST}" -maxdepth 4 -type f | head -40

# --- 5. Cleanup ---
cd "${HOME}"
rm -rf "${RUN_DIR}"
echo "=== done, results at ${DEST} ==="
