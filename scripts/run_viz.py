"""
Stage 4: Visualization for Signal Clustering Analysis

Generates three complementary views:
- A: UMAP manifold (2D + 3D HTML)
- B: Mean absorption profiles per cluster
- C: Spatial index map

Outputs:
    - figs/fig_umap2d_*.png: 2D UMAP embeddings
    - figs/umap3d_*.html: Interactive 3D UMAP
    - figs/fig_profiles_*.png: Mean absorption profiles
    - figs/fig_spatial_map_*.png: Spatial index maps
"""

import argparse
import os
import numpy as np
import pandas as pd

from src.core.data import SignalClusteringData
from src.viz import plot_umap_2d, plot_umap_3d_html, plot_cluster_profiles, plot_spatial_map


def main():
    parser = argparse.ArgumentParser(description='Stage 4: Visualization')
    parser.add_argument('--run', type=str, choices=['wavelet', 'raw', 'both'], default='both',
                       help='Which clustering to visualize')
    parser.add_argument('--view', type=str, choices=['A', 'B', 'C', 'all'], default='all',
                       help='Which view to generate')
    parser.add_argument('--k', type=int, default=5,
                       help='Number of clusters')
    parser.add_argument('--output_dir', type=str, default='data/feature_discovery',
                       help='Input directory for fingerprints/labels')
    parser.add_argument('--figs_dir', type=str, default='figs',
                       help='Output directory for figures')
    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.figs_dir, exist_ok=True)

    print("=" * 60)
    print("Stage 4: Visualization")
    print("=" * 60)

    # Load data loader
    data = SignalClusteringData()

    def process_run(run_name):
        """Process a single run (wavelet or raw)."""
        print(f"\n{'='*40}")
        print(f"Processing: {run_name}")
        print(f"{'='*40}")

        # Load fingerprints
        fp_path = os.path.join(args.output_dir, f'fingerprints_{run_name}.npy')
        fingerprints = np.load(fp_path)
        print(f"Loaded fingerprints: {fingerprints.shape}")

        # Load labels
        labels_path = os.path.join(args.output_dir, f'labels_{run_name}_k{args.k}.npy')
        labels = np.load(labels_path)
        print(f"Loaded labels: {labels.shape}, {len(np.unique(labels))} clusters")

        # Load absorption per class
        X_abs_per_class, _ = data.load_flux_per_class()
        X_abs_per_class = [np.clip(1.0 - flux, 0.0, 1.0) for flux in X_abs_per_class]

        # Also load concatenated absorption for spatial map
        X_abs = np.vstack(X_abs_per_class)

        # Dummy class_labels (not used in fixed viz function)
        class_labels = None

        # View A: UMAP
        if args.view in ['A', 'all']:
            print("\nGenerating UMAP 2D...")
            embedding = plot_umap_2d(fingerprints, labels, f'{run_name} K={args.k}',
                                    os.path.join(args.figs_dir, f'fig_umap2d_{run_name}_k{args.k}.png'))

            print("Generating UMAP 3D HTML...")
            plot_umap_3d_html(fingerprints, labels, f'{run_name} K={args.k}',
                             os.path.join(args.figs_dir, f'umap3d_{run_name}_k{args.k}.html'))

        # View B: Cluster profiles
        if args.view in ['B', 'all']:
            print("\nGenerating cluster profiles...")
            plot_cluster_profiles(X_abs_per_class, labels, class_labels, f'{run_name} K={args.k}',
                                 os.path.join(args.figs_dir, f'fig_profiles_{run_name}_k{args.k}.png'))

        # View C: Spatial map
        if args.view in ['C', 'all']:
            print("\nGenerating spatial map...")
            plot_spatial_map(labels, X_abs, f'{run_name} K={args.k}',
                            os.path.join(args.figs_dir, f'fig_spatial_map_{run_name}_k{args.k}.png'))

    # Process
    if args.run in ['wavelet', 'both']:
        process_run('wavelet')

    if args.run in ['raw', 'both']:
        process_run('raw')

    # Generate comparison plot if both runs
    if args.run == 'both' and args.view in ['A', 'all']:
        print("\n" + "="*40)
        print("Generating comparison plots...")
        print("="*40)

        # Load both labels
        labels_wavelet = np.load(os.path.join(args.output_dir, f'labels_wavelet_k{args.k}.npy'))
        labels_raw = np.load(os.path.join(args.output_dir, f'labels_raw_k{args.k}.npy'))

        # Generate UMAP comparison (P8)
        from src.viz import plot_umap_comparison
        print("\nGenerating UMAP comparison...")
        plot_umap_comparison(args.output_dir, args.k, args.figs_dir)

    if args.run == 'both' and args.view in ['C', 'all']:
        # Generate spatial map comparison (P15)
        from src.viz import plot_spatial_map_comparison
        print("\nGenerating spatial map comparison...")
        plot_spatial_map_comparison(args.output_dir, args.k, args.figs_dir)

    print("\n" + "=" * 60)
    print("Stage 4 Complete!")
    print("=" * 60)


if __name__ == '__main__':
    main()
