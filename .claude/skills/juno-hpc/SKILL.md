---
name: juno-hpc
description: Wraps a CosmoGasPeruser compute submission to UTD's Juno HPC cluster (SLURM) — login + SSH key setup, storage discipline (home/work/scratch with 45-day purge), CPU partition selection (no GPU needed for the current probe), conda + uv environment bring-up, rsync of local data UP to scratch, the canonical CPU job-script template with Producer-Consumer Verification, and rsync of results back. Trigger when the user says "submit on Juno", "launch on HPC", or "dispatch the probe on Juno"; or when wiring a new `scripts/submit_juno_*.sh` / sbatch script. Do not trigger for SageMaker, EC2, AWS, or anything outside the UTD HPC environment.
---

# Juno HPC submission contract (CosmoGasPeruser)

Juno is the alternate compute target for any CosmoGasPeruser job that exceeds the local laptop budget — currently the `exp/pk-feedback-classifier` P_F(k) 4-class de-risking probe (~1 day on 65,536 sightlines, CPU-only RandomForest). This skill codifies access, storage, partition, data transfer, and the sbatch contract so a Juno dispatch is a 3-script invocation with no scientific drift.

**Two project-specific contrasts vs. the upstream CosmoGasVision Juno skill this was transplanted from:**
1. **Data lives on the local laptop, NOT on S3.** The transfer pattern is `rsync up from local → /scratch/juno/<netid>` (not an S3 mirror). No `cgv-juno-reader` IAM user is needed.
2. **The probe is intentionally MLflow-free** per `experiments/pk-feedback-classifier/LEDGER.md` [D-01]. The round-trip MLflow importer from CosmoGasVision is NOT needed; results are CSVs + PNGs that rsync straight back.

Source of truth for cluster state: <https://hpc.utdallas.edu/systems-resources/juno/> and the UTD HPC orientation deck. Partition names are not published on the public docs — **always verify with `sinfo -s` on first login** before pinning a partition into `.env`.

## Access (one-time per user/machine)

- **Account request**: via UTD Atlas service catalog. PI sponsorship required. Email `hpc@utdallas.edu` if blocked.
- **Login host**: `juno.utdallas.edu` (resolves to one of `juno-l-01/02/03`).
- **Network requirement**: UTD wired Ethernet, CometNet WiFi, or UTD VPN. Off-campus without VPN will not connect.
- **Credentials in `.env`** (gitignored; entries templated in `.env.example`):

  ```bash
  JUNO_NETID=<utd-netid>
  JUNO_HOST=juno.utdallas.edu
  JUNO_WORK=/work/<utd-netid>/CosmoGasPeruser
  JUNO_SCRATCH=/scratch/juno/<utd-netid>
  JUNO_CPU_PARTITION=<verified-via-sinfo>   # e.g. normal, general, cpu, defq
  ```

  Load with `set -a; source .env; set +a` in shell, or `python-dotenv` in Python.

- **SSH key auth (required)** — UTD password login triggers Duo MFA per connection, which makes rsync + sbatch loops unworkable. Set up key auth once per workstation:

  ```bash
  # 1. Generate a Juno-specific Ed25519 key (no passphrase = automation-friendly).
  ssh-keygen -t ed25519 -f ~/.ssh/id_ed25519_juno -N "" -C "cosmogasperuser-juno-$(date +%Y%m%d)"

  # 2. Install the public half on Juno — prompts for UTD password + Duo ONCE.
  ssh-copy-id -i ~/.ssh/id_ed25519_juno.pub -o IdentitiesOnly=no <netid>@juno.utdallas.edu

  # 3. Add a Host alias so `ssh juno` (and `rsync ... juno:...`) just work.
  cat >> ~/.ssh/config <<EOF
  Host juno
      HostName juno.utdallas.edu
      User <netid>
      IdentityFile ~/.ssh/id_ed25519_juno
      IdentitiesOnly yes
      ServerAliveInterval 60
      ServerAliveCountMax 3
  EOF
  chmod 600 ~/.ssh/config ~/.ssh/id_ed25519_juno
  ```

  Verify: `ssh juno "hostname; whoami"` should succeed silently with no prompt.

- **Open OnDemand** (web UI for file browse / VS Code / Jupyter): <https://juno-ood.hpcre.utdallas.edu/>.
- **Support**: `circ-assist@utdallas.edu` (general HPC) or `hpc@utdallas.edu` (Juno-specific).

## Storage layout — pick the right filesystem

| Path | Real location | Filesystem | Quota | Backup | Purge | Use for |
|---|---|---|---|---|---|---|
| `~` (home) | `/home/<netid>` | MFS | 50 GB / 300k inodes | daily | none | login scripts, configs, small inputs — **never** for batch I/O |
| `~/work` → symlink | `/work/<netid>` (NOT `/home/<netid>/work`) | MFS | 1 TB / 3M inodes | daily | none | repo clone, conda envs, `.venv/`, kept results |
| `~/scratch` → symlink | `/scratch/juno/<netid>` (NOT `/scratch/<netid>`) | parallel FS | 30 TB soft | none | **45 days no-access → deleted** | batch I/O — input data, intermediates, run outputs |

Quota check: `mfsgetquota -H <directory>` for `~` and `~/work`. **`~/scratch` is not MFS** — use `df -h ~/scratch`.

**Hard rules**:
1. Compute jobs read/write to `~/scratch`, never to `~` or `~/work`. Scratch is up to 10× faster for large I/O.
2. Job script follows `copy in → compute → copy out → clean up` to avoid the silent 45-day purge wiping in-flight outputs.
3. Anything that must survive 45 days of no-access goes to `~/work` post-job (the probe's `results/pk_feedback_classifier/` CSVs + figs qualify).

Inode-quota note: a conda env contains millions of small files; install under `~/work` (3M inodes), never under `~` (300k blows within minutes).

## Partition selection — verify with `sinfo -s` before pinning

The current CosmoGasPeruser probe is **CPU-only** (scikit-learn RandomForest, no CUDA). Use a CPU partition. UTD HPC does not publish a partition table on public docs, so:

1. First login, run `ssh juno "sinfo -s"` to enumerate available partitions, their nodes, idle/alloc counts, and max wallclock.
2. Pick the standard CPU partition (commonly named `normal`, `general`, `cpu`, or `defq` depending on cluster config). The exact name on UTD Juno must be verified — do not assume.
3. Pin the verified name into `.env` as `JUNO_CPU_PARTITION=<name>`; the dispatch wrapper passes it as `sbatch --partition=...`, so no sbatch-script edit is needed when the name changes.

GPU partitions (A30, H100) exist on Juno and are documented in the source CosmoGasVision repo's skill — those are NOT used by the current probe. If a future sprint needs GPU, port the relevant section from there and add the cu124 torch override and `--gres=gpu:1` discipline.

Partition limits worth remembering (verify per `scontrol show partition <name>`):
- **Max running jobs per user**: typically 4 across GPU partitions; CPU caps may differ.
- **Default memory if `--mem` unset**: typically 64 GB — set explicitly anyway. The probe needs ~16–24 GB for the 65k × 2048 float64 flux load.
- **Max wallclock**: partition-dependent. The probe is ~1 day; a 1-day limit fits exactly. If the partition caps at 12h, split the run or pick a longer-wallclock partition.

Job-priority Fairshare resets monthly; long pauses heal Fairshare to 1.0 in two weeks of disuse.

## Environment bring-up (one-time per project clone)

Juno's default Python is system-provided; use Miniconda to donate a Python 3.12 interpreter, then let `uv` build a `.venv` next to the project. uv installs project deps into `.venv/`, **not** into the conda env — the conda env's only job is to provide the interpreter `.venv` symlinks against.

```bash
# Login node, one-time.
module load miniconda/24.11.1
eval "$(conda shell.bash hook)"
conda create -p /work/<netid>/envs/cosmogasperuser python=3.12 -y
conda activate /work/<netid>/envs/cosmogasperuser

# Project sources in /work (NOT /scratch — scratch purges after 45 days)
git clone git@github.com:SujinHwang27/CosmoGasPeruser.git "${JUNO_WORK}"
cd "${JUNO_WORK}"
pip install uv
export UV_LINK_MODE=copy             # avoids cross-FS hardlink warning
uv sync                              # creates ./.venv with all project deps
.venv/bin/python -c "import sklearn, numpy, matplotlib; print('imports OK')"
```

**No CUDA constraint for the current probe** (CPU-only). The PyTorch in `pyproject.toml` resolves to whatever wheel uv picks — fine because the probe does not call `torch.cuda.*`. If a future GPU job lands, port the cu124 override from CosmoGasVision's skill.

## Data transfer — rsync UP from local (not S3)

Unlike CosmoGasVision (which mirrors Sherwood from S3), CosmoGasPeruser's data lives on the local laptop at `data/preprocessed/Sherwood_z0.3_inf/{1..4}/{flux,vel,tau,wave,axis}.npy`. The probe inputs are `flux.npy` (4 × 268 MB = ~1 GB) + `vel.npy` (4 × 16 KB) only; tau/wave/axis are co-located but unused.

Mirror once before the first dispatch via the wrapper:

```bash
# From the local repo root:
bash scripts/sync_data_to_juno.sh
```

That script sources `.env`, ensures `${JUNO_SCRATCH}/data/preprocessed/Sherwood_z0.3_inf/` exists on Juno, and runs `rsync -avzP` for the entire `Sherwood_z0.3_inf` directory. Single shot ~1–3 min on UTD Ethernet.

Mirror is purged after 45 days of no-access; either re-run the sync script or `touch` the files periodically if a long pause is planned.

## Canonical CPU sbatch script — the P(k) probe

Lives at `scripts/submit_juno_probe.sh`. Single job (not array; the probe is one ~1-day script).

Key contract:
- `--partition` is **not** in the script — passed via `sbatch --partition=${JUNO_CPU_PARTITION} ...` so the partition name is overrideable without editing the script.
- `--cpus-per-task=16` requested; the probe's `train_rf_global` uses `n_jobs=-1`.
- `--mem=24G` to cover the (65536, 2048) float64 flux load + RF working set with headroom.
- `--time=1-00:00:00` (1 day) — matches the PI-spec compute budget. Adjust per partition cap.
- The script: copy source in → activate `.venv` → symlink mirrored data → run probe → **Producer-Consumer Verification** on all expected outputs (balanced_acc_summary.csv, ≥1 confusion CSV, ≥1 kbin_importance CSV, ≥1 figure) → copy out to `${JUNO_WORK}/cloud_runs/<RUN_TAG>/` → clean up scratch run dir.

Submit from `${JUNO_WORK}` on the login node:

```bash
set -a; source "${JUNO_WORK}/.env"; set +a
sbatch --partition="${JUNO_CPU_PARTITION}" scripts/submit_juno_probe.sh
```

Or from the local laptop via the wrapper (does the ssh + sbatch in one):

```bash
bash scripts/dispatch_juno_probe.sh          # if/when written; otherwise inline:
ssh juno "cd ${JUNO_WORK} && set -a && source .env && set +a && sbatch --partition=\${JUNO_CPU_PARTITION} scripts/submit_juno_probe.sh"
```

Monitor:

```bash
ssh juno "squeue --me"                         # see queue state
ssh juno "tail -f ${JUNO_WORK}/pkprobe-<jobid>.out"   # stream stdout
```

## Pull results back to host

After the job completes:

```bash
bash scripts/sync_results_from_juno.sh <RUN_TAG>
```

That script rsyncs `${JUNO_WORK}/cloud_runs/<RUN_TAG>/` → local `cloud_runs/<RUN_TAG>/` and (optionally) copies the `pk_feedback_classifier/` subdir into local `results/` for PI stage-gate review.

## LEDGER write-back

After the run is pulled back and the PI has read the outputs against `experiments/pk-feedback-classifier/LEDGER.md` §1 PASS condition + §5 outcome bands, the PI records the verdict (PASS / Ambiguous / NULL) as a new `[D-XX]` entry and a §7 History entry, per the `ledger-update` skill. The compute provenance (`compute=juno`, `juno_partition=<name>`, `juno_run_tag=<tag>`, `juno_jobid=<id>`) goes in the §7 entry. No MLflow round-trip for the probe ([D-01]).

## Anti-patterns

- **Running compute off `~` or `~/work`** → I/O bottleneck against backup-quality storage.
- **Forgetting to mirror data first** → the sbatch's symlink target won't exist; the probe fails fast at the first np.load.
- **No explicit `--mem`** → defaults to 64 GB on most partitions, often fine but document intent.
- **Submitting from login node without `sbatch`** (i.e. running `python run_probe.py` directly on `juno-l-0X`) → login-node CPU/RAM caps and shared-CPU policy will OOM or get the process killed.
- **Leaving data in `~/scratch` between batches** without a `touch` refresh → 45-day purge wipes it mid-sweep.
- **Skipping the copy-out step** → results die with the scratch purge.
- **Silencing the PCV checks with `2>/dev/null || true`** → see the `infrastructure-manager` agent's Producer-Consumer Verification section; this is the failure mode that lost ~30 GPU-hr in the upstream repo and the reason this skill enforces explicit `exit N` on missing artifacts.

## Triage cheatsheet

| Symptom | Probable cause | Fix |
|---|---|---|
| `sbatch: error: invalid partition specified` | `JUNO_CPU_PARTITION` not set or wrong | `ssh juno "sinfo -s"` → set verified name in `.env` |
| `sbatch: error: Memory specification can not be satisfied` | partition mem cap exceeded | drop `--mem` to fit partition |
| Job sits in `PD` with `(Priority)` | low Fairshare or per-user job cap | `sshare`; wait for sibling jobs |
| Job sits in `PD` with `(Resources)` | partition full | `sinfo` to see capacity |
| Probe FATAL exit 2/3/4/5/6 | PCV check failed (output missing) | inspect `pkprobe-<jobid>.out`; rerun if transient |
| Probe FATAL at first `np.load` | data mirror missing / purged | re-run `scripts/sync_data_to_juno.sh` |
| `~/scratch` data missing | 45-day purge fired | re-run the data sync |
| `module: command not found` | login script regression | `source /etc/profile.d/modules.sh` |

If a Juno-side error persists across 3 attempts, surface to the user with the trial log per the CLAUDE.md failure-handling rule. Email `hpc@utdallas.edu` only when the failure looks like an HPC-infrastructure issue rather than a job-script bug.
