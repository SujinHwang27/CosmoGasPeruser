import os
import numpy as np
import mlflow
from typing import List, Optional

def cleanup_mlflow_local(experiment_name: str, keep_last: int = 5):
    """
    Cleans up local MLflow runs, keeping only the most recent N.
    """
    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    if not experiment:
        return
    
    runs = client.search_runs(experiment_ids=[experiment.experiment_id], order_by=["start_time DESC"])
    for run in runs[keep_last:]:
        print(f"Deleting run: {run.info.run_id}")
        client.delete_run(run.info.run_id)

def check_array_shapes(arrays: List[np.ndarray], expected_dim: Optional[int] = None):
    """
    Standardizes shape verification for numpy arrays.
    """
    for i, arr in enumerate(arrays):
        print(f"Array {i} shape: {arr.shape}")
        if expected_dim and arr.shape[1] != expected_dim:
            print(f"Warning: Array {i} has dim {arr.shape[1]}, expected {expected_dim}")

def get_distribution_stats(data: np.ndarray) -> Dict:
    """
    Computes standard distribution stats for a numpy array.
    """
    return {
        "mean": float(np.mean(data)),
        "std": float(np.std(data)),
        "min": float(np.min(data)),
        "max": float(np.max(data)),
        "sparsity": float(np.mean(np.abs(data) < 1e-4))
    }

def calculate_energy(coeffs: np.ndarray) -> float:
    """
    Mean energy across sightlines: mean_i( sum_j w_j^2 )
    """
    return float(np.mean(np.sum(coeffs ** 2, axis=1)))

def ensure_dir(path: str):
    """
    Ensure directory exists.
    """
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
