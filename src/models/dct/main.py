from load_data import load_sherwood_data
from plot import plot_2d_data
from dct_utils import *

import mlflow
import numpy as np


mlflow.set_tracking_uri("http://localhost:5000")  
mlflow.set_experiment("DCT_Sherwood")

data_base_path = "data/preprocessed/Sherwood_z0.3_inf"
save_path = "data/processed/Sherwood_z0.3_inf/dct_highfreq_2"
run_name = "z0.3_inf_dct_highfreq_2"

if __name__ == "__main__":
    with mlflow.start_run(run_name=run_name) as run:
        # 1. Load data
        X, y = load_sherwood_data(data_base_path)
        mlflow.log_param("dataset_version", "Sherwood_z0.3_inf")
        mlflow.log_param("num_samples", len(y))
        mlflow.log_param("num_features", X.shape[1])
        mlflow.log_param("num_classes", len(np.unique(y)))

        # 2. Run Discrete Cosine Transform
        X_reduced = dct_high_freq(X, 2)
        mlflow.log_param("k", X_reduced.shape[1])

        # Save reduced data per class
        save_reduced_data(X_reduced, y, save_path)

        # 3. Plot and log artifact
        plot_path = plot_2d_data(X_reduced, y, run_name)
        mlflow.log_artifact(plot_path, artifact_path="plots")
        mlflow.log_artifact(save_path, artifact_path="reduced_data")

        print(f"MLflow run completed: {run.info.run_id}")