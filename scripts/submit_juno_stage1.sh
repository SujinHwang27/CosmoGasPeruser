#!/bin/bash
#SBATCH --job-name=pk-stage1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=6:00:00
#SBATCH --output=cloud_runs/pkstage1-%j.out
#SBATCH --error=cloud_runs/pkstage1-%j.err
# --partition deliberately NOT pinned here — the submit wrapper passes
# `--partition=${JUNO_CPU_PARTITION}` from .env so the partition name is
# overrideable without editing this script. See .claude/skills/juno-hpc/SKILL.md.
#
# NOTE on --output/--error path: SLURM evaluates these BEFORE the body sets
# RUN_TAG, so the per-RUN_TAG subdirectory listed in the brief
# (`cloud_runs/pkstage1-${RUN_TAG}/pkstage1-%j.out`) cannot be expressed in
# the #SBATCH directive itself — RUN_TAG is generated in the body. The .out/.err
# land in `cloud_runs/` at the top level and are moved into the per-RUN_TAG
# dir in §5 alongside the results copy-out.

# Stage 1 sbatch — Stacked-pipeline population-statistics track per
# experiments/pk-feedback-classifier/LEDGER.md [D-09], [D-15]–[D-23].
# Six gates G1–G6, 5-fold CV × 10-seed bootstrap, dense M-sweep
# {1,4,16,32,64,128,256}, both norm regimes, 1000-permutation null on G2,
# K1 ablation gate on G6. ~7940 RF fits estimated; 6h wallclock ceiling per
# [D-09]; checkpointed per [D-20] S5 so a wallclock-cap-kill writes
# `cv_partial.csv` rows up to the kill point. Intentionally MLflow-free per
# [D-01] track-inheritance.
#
# Expanded PCV per [D-23] C5/C6 and infrastructure-manager.md §28–§40 —
# hard-asserts the full G1–G6 artifact set with distinct FATAL exit codes
# 10–15. Exit code namespace is shifted into 10–15 (NOT 2–7) to avoid
# colliding with `run_stage1.py`'s own runtime fold-leakage assert exit 7
# (per [D-20] S3 defense-in-depth, [D-23] C6 namespace coordination).

set -euo pipefail

# --- 0. Source Juno-side .env if present (JUNO_*) ---
# Per [D-05] anti-pattern: Juno-side /work/<netid>/CosmoGasPeruser/.env carries
# the JUNO_* block ONLY (mode 600). AWS keys MUST NOT be propagated here.
if [[ -f "${JUNO_WORK:-$HOME/work/CosmoGasPeruser}/.env" ]]; then
  set -a
  source "${JUNO_WORK:-$HOME/work/CosmoGasPeruser}/.env"
  set +a
fi
: "${JUNO_WORK:=$HOME/work/CosmoGasPeruser}"
: "${JUNO_SCRATCH:=$HOME/scratch}"
: "${JUNO_CPU_PARTITION:=normal}"

RUN_TAG="pkstage1-$(date -u +%Y%m%d-%H%M%S)-$(openssl rand -hex 3)"
RUN_DIR="${JUNO_SCRATCH}/pk-feedback-classifier/${RUN_TAG}"
DEST="${JUNO_WORK}/cloud_runs/${RUN_TAG}"
mkdir -p "${RUN_DIR}" "${DEST}"

# --- Economic-compute plan (per infrastructure-manager.md "Economic compute" checklist) ---
cat <<EOF
=== ECONOMIC COMPUTE PLAN ===
(a) instance: Juno ${JUNO_CPU_PARTITION} partition, 1 node × 16 cores × 32 GB
(b) hours: 6h wallclock cap
(c)/(d) cost: \$0 marginal (Fairshare-aware queueing)
(e) auto-stop: SLURM wallclock + set -euo pipefail cleanup
(f) lifecycle: scratch run dir → MFS cloud_runs/${RUN_TAG}/ → local rsync via scripts/sync_results_from_juno.sh ${RUN_TAG}
============================
EOF

cd "${RUN_DIR}"

# --- 1. Copy in source + symlink data (data mirror already on scratch per [D-05]) ---
cp -r "${JUNO_WORK}"/{src,experiments,scripts,pyproject.toml,uv.lock} .
[[ -f "${JUNO_WORK}/.python-version" ]] && cp "${JUNO_WORK}/.python-version" . || true
mkdir -p data/preprocessed
ln -s "${JUNO_SCRATCH}/data/preprocessed/Sherwood_z0.3_inf" data/preprocessed/Sherwood_z0.3_inf
[[ -d "data/preprocessed/Sherwood_z0.3_inf/1" ]] || {
  echo "FATAL: Sherwood mirror missing at ${JUNO_SCRATCH}/data/preprocessed/Sherwood_z0.3_inf/" >&2
  echo "       run scripts/sync_data_to_juno.sh from the local laptop first." >&2
  exit 1
}

# --- 2. Environment (uv-created .venv at repo root — matches Stage 0 proven pattern) ---
# Original dispatch brief specified conda activation, but Juno has no conda
# installation at the assumed path; Stage 0's proven pattern is the uv .venv.
source "${JUNO_WORK}/.venv/bin/activate"

export PYTHONPATH=.
export PYTHONUNBUFFERED=1
# Stage 1 is intentionally MLflow-free per [D-01] track inheritance; no MLFLOW_TRACKING_URI.
export GIT_PYTHON_REFRESH=quiet
# Cap thread fan-out so we don't oversubscribe vs SLURM allocation.
export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16
export OPENBLAS_NUM_THREADS=16

echo "=== run config ==="
echo "RUN_TAG=${RUN_TAG}"
echo "RUN_DIR=${RUN_DIR}"
echo "Python: $(which python)  $(python --version)"
echo "CPUs available: $(nproc); SLURM allocated: ${SLURM_CPUS_PER_TASK:-?}; OMP/MKL/OPENBLAS=${OMP_NUM_THREADS}"
free -g | head -2
df -h "${RUN_DIR}" "${JUNO_SCRATCH}" 2>/dev/null || true

# --- 3. Compute ---
PYTHONPATH=. python -u experiments/pk-feedback-classifier/run_stage1.py --run-tag "${RUN_TAG}" 2>&1 | tee run_stage1.log

# --- 4. Producer-Consumer Verification (PCV) — [D-23] C5/C6, infrastructure-manager.md §28–§40 ---
# Hard-asserts the full G1–G6 artifact set produced by run_stage1.py.
# Distinct FATAL exit codes 10–15 — shifted out of the 2–7 namespace to avoid
# colliding with run_stage1.py's own runtime fold-leakage assert (exit 7 per
# [D-20] S3 / [D-23] C6 defense-in-depth).
#
# Expected row counts (per [D-09] / [D-22] spec):
#   cv_partial.csv:           2 regimes × 7 M × 10 seeds × 5 folds = 700 data rows
#   g2_permutation_null.csv:  1000 permutation rows
#   g5_routing.csv:           4 class rows (or 3 in G2 control)
#   summary.csv, g1_g4_summary.csv, g6_ablation.csv: shape TBD per implementer; existence-checked only

RESULTS_DIR="results/pk_feedback_classifier/stage1"
[[ -d "${RESULTS_DIR}" ]] || { echo "FATAL: producer wrote no ${RESULTS_DIR} dir" >&2; exit 10; }

# G1/G3/G4 backbone — cv_partial.csv (S5 checkpointed; one row per (regime, M, seed, fold) tuple).
CV_PARTIAL="${RESULTS_DIR}/cv_partial.csv"
[[ -s "${CV_PARTIAL}" ]] || { echo "FATAL[10]: ${CV_PARTIAL} missing or empty" >&2; exit 10; }
# wc -l counts header + data rows; 700 data rows + 1 header = 701; brief specifies `wc -l > 700`.
CV_ROWS=$(wc -l < "${CV_PARTIAL}")
if [[ "${CV_ROWS}" -le 700 ]]; then
  echo "FATAL[11]: ${CV_PARTIAL} has ${CV_ROWS} lines, expected > 700 (header + 700 data rows = 701)." >&2
  echo "           Likely cause: wallclock-cap-kill before sweep completion. The partial CSV survives" >&2
  echo "           per [D-20] S5 / [D-23] C6 checkpointing, but the sweep is INCOMPLETE." >&2
  echo "           Resume path: re-fire sbatch; run_stage1.py reconstructs completed tuples from cv_partial.csv." >&2
  exit 11
fi

# G2 — permutation-test null (1000 permutations per [D-18]).
G2_FILE="${RESULTS_DIR}/g2_permutation_null.csv"
[[ -s "${G2_FILE}" ]] || { echo "FATAL[12]: ${G2_FILE} missing or empty" >&2; exit 12; }
G2_ROWS=$(wc -l < "${G2_FILE}")
[[ "${G2_ROWS}" -gt 1000 ]] || { echo "FATAL[12]: ${G2_FILE} has ${G2_ROWS} lines, expected > 1000 (header + 1000 permutations)." >&2; exit 12; }

# G5 — mean-flux correlation per class (4 rows: C1, C2, C3, C4) per [D-21] / [D-23] C7.
G5_FILE="${RESULTS_DIR}/g5_routing.csv"
[[ -s "${G5_FILE}" ]] || { echo "FATAL[13]: ${G5_FILE} missing or empty" >&2; exit 13; }
G5_ROWS=$(wc -l < "${G5_FILE}")
[[ "${G5_ROWS}" -gt 4 ]] || { echo "FATAL[13]: ${G5_FILE} has ${G5_ROWS} lines, expected > 4 (header + 4 class rows)." >&2; exit 13; }

# G6 — K1 ablation gate (top-3 mid-k bin removal) per [D-16] / [D-23] C1.
G6_FILE="${RESULTS_DIR}/g6_ablation.csv"
[[ -s "${G6_FILE}" ]] || { echo "FATAL[14]: ${G6_FILE} missing or empty" >&2; exit 14; }

# Summary tables.
SUMMARY="${RESULTS_DIR}/summary.csv"
G1G4_SUMMARY="${RESULTS_DIR}/g1_g4_summary.csv"
[[ -s "${SUMMARY}" ]] || { echo "FATAL[15]: ${SUMMARY} missing or empty" >&2; exit 15; }
[[ -s "${G1G4_SUMMARY}" ]] || { echo "FATAL[15]: ${G1G4_SUMMARY} missing or empty" >&2; exit 15; }

echo "=== PCV PASS — all G1–G6 artifacts present with expected row counts ==="
echo "  cv_partial.csv:           ${CV_ROWS} lines (>700)"
echo "  g2_permutation_null.csv:  ${G2_ROWS} lines (>1000)"
echo "  g5_routing.csv:           ${G5_ROWS} lines (>4)"
echo "  g6_ablation.csv:          present, non-empty"
echo "  summary.csv:              present, non-empty"
echo "  g1_g4_summary.csv:        present, non-empty"

# --- 5. Copy out to MFS, then move SLURM .out/.err into per-RUN_TAG dir ---
mkdir -p "${DEST}/results"
cp -r "${RESULTS_DIR}" "${DEST}/results/"
cp run_stage1.log "${DEST}/"

# Best-effort relocate of SLURM .out/.err (cleanup-class operation; 2>/dev/null tolerated here
# per infrastructure-manager.md §28–§40 — NOT on artifact-existence checks).
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  mv "${JUNO_WORK}/cloud_runs/pkstage1-${SLURM_JOB_ID}.out" "${DEST}/" 2>/dev/null || true
  mv "${JUNO_WORK}/cloud_runs/pkstage1-${SLURM_JOB_ID}.err" "${DEST}/" 2>/dev/null || true
fi

echo "=== copied artifacts ==="
find "${DEST}" -maxdepth 4 -type f | head -40

# --- 6. Cleanup ---
cd "${HOME}"
rm -rf "${RUN_DIR}"
echo "=== done, results at ${DEST} ==="
echo "Next: bash scripts/sync_results_from_juno.sh ${RUN_TAG}  (run from local laptop)"
