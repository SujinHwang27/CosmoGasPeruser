"""
Stage 4: Visualization for Signal Clustering Analysis

Generates UMAP plots (2D/3D) and spatial cluster maps for separability vectors.

Outputs:
    - figs/fig_umap_2d_*.png: Static UMAP plots
    - figs/fig_umap_3d_*.html: Interactive 3D UMAP plots
    - figs/fig_spatial_map_*.png: Spatial distribution of clusters
"""

import argparse
import os
import numpy as np
import pandas as pd
from pathlib import Path

from src.core.data import SignalClusteringData
from src.viz import plot_umap_2d, plot_umap_3d_html, plot_spatial_cluster_map


def main():
    parser = argparse.ArgumentParser(description='Stage 4: Visualization')
    parser.add_argument('--run', type=str, choices=['wavelet', 'raw', 'both'], default='both',
                       help='Which run to visualize')
    parser.add_argument('--k', type=int, default=5,
                       help='Target K used for clustering')
    parser.add_argument('--view', type=str, default='all',
                       help='View mode (legacy/orchestrator support)')
    parser.add_argument('--output_dir', type=str, default='data/feature_discovery',
                       help='Directory containing labels and separability vectors')
    parser.add_argument('--figs_dir', type=str, default='figs',
                       help='Output directory for figures')
    args = parser.parse_args()

    # Create output directories
    os.makedirs(args.figs_dir, exist_ok=True)

    print("=" * 60)
    print("Stage 4: Visualization")
    print("=" * 60)

    # Initialize data loader for absorption data (needed for spatial map)
    data = SignalClusteringData()
    X_flux, _ = data.load_flux()
    X_abs = np.clip(1.0 - X_flux, 0.0, 1.0)

    def process_run(run_name):
        """Generate plots for a single run."""
        print(f"\n{'='*40}")
        print(f"Visualizing: {run_name} (K={args.k})")
        print(f"{'='*40}")

        # Load data
        sv_path = os.path.join(args.output_dir, f'separability_vectors_{run_name}.npy')
        labels_path = os.path.join(args.output_dir, f'labels_{run_name}_k{args.k}.npy')

        if not os.path.exists(sv_path) or not os.path.exists(labels_path):
            print(f"Warning: Missing files for {run_name}. Skipping.")
            return

        separability_vectors = np.load(sv_path)
        labels = np.load(labels_path)
        print(f"Loaded separability vectors: {separability_vectors.shape}")
        print(f"Loaded labels: {labels.shape}")

        # 1. Static 2D UMAP
        print("\nGenerating 2D UMAP...")
        save_2d = os.path.join(args.figs_dir, f'fig_umap_2d_{run_name}_k{args.k}.png')
        plot_umap_2d(separability_vectors, labels, run_name, save_2d)

        # 2. Interactive 3D UMAP
        print("Generating 3D UMAP (HTML)...")
        save_3d = os.path.join(args.figs_dir, f'fig_umap_3d_{run_name}_k{args.k}.html')
        plot_umap_3d_html(separability_vectors, labels, run_name, save_3d)

        if args.view:
            import webbrowser
            webbrowser.open('file://' + os.path.realpath(save_3d))

        # 3. Spatial Map
        print("Generating Spatial Cluster Map...")
        save_spatial = os.path.join(args.figs_dir, f'fig_spatial_map_{run_name}_k{args.k}.png')
        plot_spatial_cluster_map(labels, X_abs, run_name, save_spatial)

    # Process runs
    if args.run in ['wavelet', 'both']:
        process_run('wavelet')

    if args.run in ['raw', 'both']:
        process_run('raw')

    # Generate comparison plots if both runs
    if args.run == 'both':
        print("\n" + "="*40)
        print("Generating comparison plots...")
        print("="*40)

        # Generate UMAP comparison
        from src.viz import plot_umap_comparison
        print("\nGenerating UMAP comparison...")
        plot_umap_comparison(args.output_dir, args.k, args.figs_dir)

        # Generate spatial map comparison
        from src.viz import plot_spatial_map_comparison
        print("\nGenerating spatial map comparison...")
        plot_spatial_map_comparison(args.output_dir, args.k, args.figs_dir)

    print("\n" + "=" * 60)
    print("Stage 4 Complete!")
    print("=" * 60)


if __name__ == '__main__':
    main()
