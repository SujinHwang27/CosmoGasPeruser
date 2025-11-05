from load_data import load_sherwood_data
from plot import *
from dct_utils import *

import mlflow
import numpy as np


mlflow.set_tracking_uri("http://localhost:5000")  
mlflow.set_experiment("DCT_Sherwood")

data_base_path = "data/preprocessed/Sherwood_z0.3_inf"
save_path = "data/processed/Sherwood_z0.3_inf/dct_highfreq_3"
run_name = "z0.3_inf_dct_highfreq_3"

# For highest 2 frequencies
# if __name__ == "__main__":
#     with mlflow.start_run(run_name=run_name) as run:
#         # 1. Load data
#         X, y = load_sherwood_data(data_base_path)
#         mlflow.log_param("dataset_version", "Sherwood_z0.3_inf")
#         mlflow.log_param("num_samples", len(y))
#         mlflow.log_param("num_features", X.shape[1])
#         mlflow.log_param("num_classes", len(np.unique(y)))

#         # 2. Run Discrete Cosine Transform
#         X_reduced = dct_high_freq(X, 2)
#         mlflow.log_param("k", X_reduced.shape[1])

#         # Save reduced data per class
#         save_reduced_data(X_reduced, y, save_path)

#         # 3. Plot and log artifact
#         plot_path = plot_2d_data(X_reduced, y, run_name)
#         mlflow.log_artifact(plot_path, artifact_path="plots")
#         mlflow.log_artifact(save_path, artifact_path="reduced_data")

#         print(f"MLflow run completed: {run.info.run_id}")

# For full dct coefficient samples
# if __name__ == "__main__":
#     with mlflow.start_run(run_name=run_name) as run:
#         # 1. Load data
#         X, y = load_sherwood_data(data_base_path)

#         mlflow.log_param("dataset_version", "Sherwood_z0.3_inf")
#         mlflow.log_param("num_features", X.shape[1])
#         mlflow.log_param("num_classes", len(np.unique(y)))

#         # sample data
#         idx = [0, 1, 2, 3, 4, 5, 6, 7]
#         unique_classes = np.unique(y)
#         class_blocks = np.vsplit(X, len(unique_classes))

#         # Prepare lists for sampled data and labels
#         sampled_X_list = []
#         sampled_y_list = []

#         # Loop over the indices and collect samples from each class
#         for i in idx:
#             for class_idx, block in enumerate(class_blocks):
#                 if i < len(block):
#                     sampled_X_list.append(block[i])
#                     sampled_y_list.append(unique_classes[class_idx])

#         # Convert lists to numpy arrays
#         sampled_X = np.vstack(sampled_X_list)   # shape: (n_samples_total, n_features)
#         sampled_y = np.array(sampled_y_list)    # shape: (n_samples_total,)

#         mlflow.log_param("num_samples", len(sampled_y))

#         # 2. Run Discrete Cosine Transform
#         X_dct = dct_full(sampled_X)

#         # Save reduced dct data per class
#         save_reduced_data(X_dct, sampled_y, save_path)

#         # 3. Plot and log artifact
#         plot_path_list = plot_dct_by_los(X_dct, sampled_y, save_path=save_path, title="DCT_Comparison", n_classes=4)
#         for plot_path in plot_path_list:
#             mlflow.log_artifact(plot_path, artifact_path="plots")
#         mlflow.log_artifact(save_path, artifact_path="reduced_data")

        
#         print(f"MLflow run completed: {run.info.run_id}")

# For dominant/minor 2 or 3 coefficients
if __name__ == "__main__":
    with mlflow.start_run(run_name=run_name) as run:
        # 1. Load data
        X, y = load_sherwood_data(data_base_path)
        mlflow.log_param("dataset_version", "Sherwood_z0.3_inf")
        mlflow.log_param("num_samples", len(y))
        mlflow.log_param("num_features", X.shape[1])
        mlflow.log_param("num_classes", len(np.unique(y)))

        # 2. Run Discrete Cosine Transform
        X_reduced = dct_high_freq(X, 3)
        mlflow.log_param("k", X_reduced.shape[1])

        # Save reduced data per class
        save_reduced_data(X_reduced, y, save_path)

        # 3. Plot and log artifact
        # plot_path = plot_2d_data(X_reduced, y, run_name, save_path=save_path)
        plot_path = plot_3d_interactive(X_reduced, y, run_name, save_path=save_path)
        mlflow.log_artifact(plot_path, artifact_path="plots")
        mlflow.log_artifact(save_path, artifact_path="reduced_data")

        print(f"MLflow run completed: {run.info.run_id}")


