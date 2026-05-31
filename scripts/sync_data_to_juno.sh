#!/bin/bash
# Mirror Sherwood z=0.3 preprocessed data UP from the local laptop to Juno scratch.
# Run once before the first probe dispatch; re-run after a 45-day scratch purge.
# Total ~1 GB (4 x 268 MB flux.npy + small vel/tau/wave/axis.npy).
# See .claude/skills/juno-hpc/SKILL.md for the contract.

set -euo pipefail

# Source local .env (must have JUNO_NETID, JUNO_HOST, JUNO_SCRATCH).
if [[ -f "$(dirname "$0")/../.env" ]]; then
  set -a
  source "$(dirname "$0")/../.env"
  set +a
fi
: "${JUNO_NETID:?JUNO_NETID not set; populate .env from .env.example}"
: "${JUNO_HOST:?JUNO_HOST not set; populate .env from .env.example}"
: "${JUNO_SCRATCH:?JUNO_SCRATCH not set; populate .env from .env.example}"

LOCAL_SRC="data/preprocessed/Sherwood_z0.3_inf/"
REMOTE_DST="${JUNO_SCRATCH}/data/preprocessed/Sherwood_z0.3_inf/"

[[ -d "${LOCAL_SRC}" ]] || {
  echo "FATAL: local data dir ${LOCAL_SRC} not found (run from repo root)." >&2
  exit 1
}

echo "=== ensuring remote dir ==="
ssh juno "mkdir -p '${REMOTE_DST}'"

echo "=== rsync up ==="
echo "  local : ${LOCAL_SRC}"
echo "  remote: juno:${REMOTE_DST}"
rsync -avzP "${LOCAL_SRC}" "juno:${REMOTE_DST}"

echo "=== remote inventory ==="
ssh juno "ls -la '${REMOTE_DST}'/1 | head -10 && echo --- && du -sh '${REMOTE_DST}'"

echo "=== done ==="
echo "Next: ssh juno 'cd ${JUNO_WORK:-/work/${JUNO_NETID}/CosmoGasPeruser} && sbatch --partition=\${JUNO_CPU_PARTITION} scripts/submit_juno_probe.sh'"
