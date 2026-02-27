# MLflow Multi-Machine Synchronization Guide

This project handles MLflow data in two parts to allow seamless experimentation across different machines without a centralized tracking server.

## 1. The Architecture

MLflow requires two storage locations:
- **Artifact Store (S3)**: Stores "heavy" files (plots, models, data). This is already centralized in your project.
- **Backend Store (Local Index)**: Stores "light" metadata (run names, metrics, parameters). This is traditionally local to each machine.

To see the same results on all machines, we use **DVC** to synchronize the **Backend Store (Index)**. Once Machine B has the same index as Machine A, it will automatically know how to fetch the plots from S3.

---

## 2. Manual Workflow (Switching Machines)

Follow these steps every time you finish a batch of experiments or switch workstations.

### Machine A: Saving Progress (After training)
Once you have finished running your scripts (e.g., `train_rf_baseline.py`), sync the new runs:

```bash
# 1. Update the DVC index for mlruns metadata
dvc add mlruns

# 2. Add the tiny pointer file to Git
git add mlruns.dvc
git commit -m "mlflow: sync experiment metadata"

# 3. Push the index to your remote (if applicable) and git push
dvc push
git push
```

### Machine B: Loading Progress (Before reviewing)
When you sit down at another machine to review results in the UI:

```bash
# 1. Pull the latest code and index pointers
git pull

# 2. Pull the actual mlruns metadata
dvc pull mlruns

# 3. Start the UI
mlflow ui
```

---

## 3. Configuration Details

Ensure your `.env` file is consistent across machines:

```bash
# Metadata stays local (and synced via DVC)
MLFLOW_TRACKING_URI=http://localhost:5000

# Experiment name
MLFLOW_EXPERIMENT_NAME=Baseline_RF
```

> [!TIP]
> **Why not just sync S3?**
> The MLflow UI cannot "discover" runs just by looking at S3. It must have the `mlruns` metadata (the identity of the run) in its local database first. Once it knows the run exists, it uses its built-in logic to reach out to S3 and download the plots for you.
