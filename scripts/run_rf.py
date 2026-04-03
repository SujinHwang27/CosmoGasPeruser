import argparse
import os
import numpy as np
import mlflow
import pandas as pd
from pathlib import Path

from src.core.data import SignalClusteringData
from src.core.models.rf_classifier import prep_cluster_data, train_rf_for_cluster

def run_rf_pipeline(flavor, k, output_dir):
    """
    Execute the Stage 6 Random Forest training pipeline for one flavor ('wavelet' or 'raw').
    """
    labels_file = f"data/feature_discovery/labels_{flavor}_k{k}.npy"
    if not os.path.exists(labels_file):
        raise FileNotFoundError(f"Missing cluster labels file: {labels_file}")
    
    print(f"\nLoading data for '{flavor}' run...")
    data_loader = SignalClusteringData()
    
    if flavor == 'wavelet':
        X_list, _ = data_loader.load_wavelet_global_normalized()
    else:
        # Load the absorption data per-class
        # Raw flux is F. We need A = 1 - F
        X_flux_list, _ = data_loader.load_flux_per_class()
        X_list = [np.clip(1.0 - X_flux, 0.0, 1.0) for X_flux in X_flux_list]
        
    cluster_labels = np.load(labels_file)
    unique_clusters = np.unique(cluster_labels)
    
    summary_results = []

    # Wrapper MLflow run for the entire flavor execution
    with mlflow.start_run(run_name=f"{flavor}_Stage6_RF"):
        sightline_correct_counts = np.zeros(len(cluster_labels))
        
        for cluster_id in unique_clusters:
            print(f"\n--- Cluster {cluster_id} ---")
            cluster_indices = np.where(cluster_labels == cluster_id)[0]
            
            # Prepare cluster data
            X_cluster, y_cluster = prep_cluster_data(X_list, cluster_indices)
            
            # Train model
            best_estimator, best_params, cv_score, test_score, test_acc = train_rf_for_cluster(
                X=X_cluster, 
                y=y_cluster, 
                cluster_id=cluster_id, 
                run_flavor=flavor
            )
            
            # Record summary
            res = {
                'run_flavor': flavor,
                'cluster_id': cluster_id,
                'cluster_size': len(cluster_indices),
                'train_samples': len(y_cluster) * 0.8,
                'test_f1_score': test_score,
                'test_accuracy': test_acc,
                'cv_f1_score': cv_score,
                **best_params
            }
            summary_results.append(res)
            
            # Evaluate back on the physical sightlines for the accuracy score metric
            for c_idx, X_class in enumerate(X_list):
                true_label = c_idx + 1
                X_comp = X_class[cluster_indices]
                preds = best_estimator.predict(X_comp)
                sightline_correct_counts[cluster_indices] += (preds == true_label).astype(int)
                
        # Calculate resulting score mapping for bar chart visual
        sightline_scores = sightline_correct_counts / 4.0
        
        print("Generating 100% stacked bar chart of RF performance scores...")
        from src.core.viz import plot_cluster_score_distribution
        figs_dir = os.path.join(output_dir, "figs")
        os.makedirs(figs_dir, exist_ok=True)
        rf_png_path = os.path.join(figs_dir, f"fig_score_dist_{flavor}_k{k}.png")
        plot_cluster_score_distribution(cluster_labels, sightline_scores, flavor, rf_png_path)
        mlflow.log_artifact(rf_png_path)
            
    # Save CSV summary
    df_summary = pd.DataFrame(summary_results)
    out_csv = os.path.join(output_dir, f"rf_summary_{flavor}_k{k}.csv")
    df_summary.to_csv(out_csv, index=False)
    print(f"\nSaved RF summary to: {out_csv}")


def main():
    parser = argparse.ArgumentParser(description="Stage 6: Cluster-Specific Random Forest Classifiers")
    parser.add_argument('--run', type=str, choices=['wavelet', 'raw', 'both'], required=True,
                        help="Which feature set to run on")
    parser.add_argument('--k', type=int, default=5,
                        help="Number of clusters (default: 5)")
    parser.add_argument('--output_dir', type=str, default='results/signal_clustering_v2',
                        help="Output directory for CSV summaries")
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    mlflow.set_experiment("SignalClustering_Stage6")
    
    runs = ['wavelet', 'raw'] if args.run == 'both' else [args.run]
    
    print("=" * 60)
    print("Stage 6: Training Cluster-Specific Random Forests")
    print("=" * 60)
    
    for flavor in runs:
        run_rf_pipeline(flavor, args.k, args.output_dir)

if __name__ == "__main__":
    main()
