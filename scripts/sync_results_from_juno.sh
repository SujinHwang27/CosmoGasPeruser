#!/bin/bash
# Pull probe results back from Juno after the sbatch job completes.
# Usage: bash scripts/sync_results_from_juno.sh <RUN_TAG>
# Example: bash scripts/sync_results_from_juno.sh pkprobe-20260530-021345-a3f8c1
# See .claude/skills/juno-hpc/SKILL.md for the contract.

set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 <RUN_TAG>" >&2
  echo "       (find the RUN_TAG in the sbatch .out log line 'RUN_TAG=...' or via ssh juno 'ls ~/work/CosmoGasPeruser/cloud_runs/')" >&2
  exit 1
fi
RUN_TAG="$1"

# Source local .env (must have JUNO_NETID, JUNO_HOST).
if [[ -f "$(dirname "$0")/../.env" ]]; then
  set -a
  source "$(dirname "$0")/../.env"
  set +a
fi
: "${JUNO_NETID:?JUNO_NETID not set; populate .env from .env.example}"
: "${JUNO_HOST:?JUNO_HOST not set; populate .env from .env.example}"
: "${JUNO_WORK:=/work/${JUNO_NETID}/CosmoGasPeruser}"

REMOTE_SRC="${JUNO_WORK}/cloud_runs/${RUN_TAG}/"
LOCAL_DST="cloud_runs/${RUN_TAG}/"

mkdir -p cloud_runs

echo "=== rsync down ==="
echo "  remote: juno:${REMOTE_SRC}"
echo "  local : ${LOCAL_DST}"
rsync -avzP "juno:${REMOTE_SRC}" "${LOCAL_DST}"

echo "=== inventory ==="
find "${LOCAL_DST}" -maxdepth 3 -type f | head -40

# Optional: stage into results/ for PI review. Don't overwrite existing.
RESULTS_DST="results/pk_feedback_classifier"
if [[ ! -d "${RESULTS_DST}" ]] && [[ -d "${LOCAL_DST}/pk_feedback_classifier" ]]; then
  echo "=== staging ${LOCAL_DST}/pk_feedback_classifier -> ${RESULTS_DST} for PI stage-gate review ==="
  cp -r "${LOCAL_DST}/pk_feedback_classifier" "${RESULTS_DST}"
elif [[ -d "${RESULTS_DST}" ]]; then
  echo "NOTE: ${RESULTS_DST} already exists; not overwriting. Move/diff manually if needed."
fi

echo "=== done ==="
echo "Next: dispatch project-architect for stage-gate review against experiments/pk-feedback-classifier/LEDGER.md §1 PASS condition + §5 outcome bands."
