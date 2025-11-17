from load_data import load_sherwood_data
from plot import *
from fisher_utils import *

import mlflow
import numpy as np


mlflow.set_tracking_uri("http://localhost:5000")  
mlflow.set_experiment("Fisher_Sherwood")

data_base_path = "data/processed/Sherwood_z0.3_inf/dct_full"
save_path = "data/feature_analysis/Sherwood_z0.3_inf/dct_fisher"
run_name = "z0.3_inf_dct_fisher"

if __name__ == "__main__":
    with mlflow.start_run(run_name=run_name) as run:
        # 1. Load data
        X, y = load_sherwood_data(data_base_path)
        mlflow.log_param("dataset_version", "Sherwood_z0.3_inf_dct_full")
        mlflow.log_param("num_samples", len(y))
        mlflow.log_param("num_features", X.shape[1])
        mlflow.log_param("num_classes", len(np.unique(y)))

        # reshape data
        class_blocks = np.vsplit(X, 4)
        stacked = np.stack(class_blocks)    # (n_classes, n_samples_per_class, n_features)
        X_LoS_generator = stacked.transpose(1, 0, 2)  # (n_samples_per_class, n_classes, n_features)

        # 2. Calculate Fisher coeff of LoS(c1, c2, c3, c4)
        fisher_scores_matrix = fisher_scores(X_LoS_generator)

        # Save Fisher coeff matrix
        fisher_file_path = save_feature_analysis(fisher_scores_matrix, save_path)
        mlflow.log_artifact(fisher_file_path, artifact_path="score_data_path")
        
        # Plot fisher coeff by feature index
        plot_path = plot_fisher(fisher_scores_matrix, run_name, save_path)
        mlflow.log_artifact(plot_path, artifact_path="plots")


        # 3. Fisher coeff stat analysis
        # Rank feature index by fisher coeff
        mean_fisher = np.mean(fisher_scores_matrix, axis=0)
        mean_plot_path = plot_mean_fisher(mean_fisher, title="Mean Fisher Coefficient", save_path=save_path)
        mlflow.log_artifact(mean_plot_path, artifact_path="plots")
        ranked_idx = np.argsort(mean_fisher)[::-1] 
        np.save(f"{save_path}/feature_ranks.npy", ranked_idx)
        mlflow.log_artifact(f"{save_path}/feature_ranks.npy", artifact_path="feature_ranks")


        print(f"MLflow run completed: {run.info.run_id}")
