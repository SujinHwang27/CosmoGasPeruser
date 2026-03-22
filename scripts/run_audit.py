"""
Stage 5: Auditing for Signal Clustering Analysis

Performs statistical auditing:
- 5.1: Cross-run cluster agreement (contingency heatmap + Procrustes-aligned 3D UMAP overlay)
- 5.2: Separability vector greyscale visualizations per cluster

Outputs:
    - results/*.csv: Audit statistics
    - figs/*.png, figs/*.html: Audit visualizations
"""

import argparse
import os
import numpy as np
import pandas as pd
from pathlib import Path

from src.audit import (
    cross_run_overlap,
    compute_contingency,
    plot_contingency_heatmap,
    plot_shared_umap_overlay_3d,
    plot_separability_greyscale,
)


def main():
    parser = argparse.ArgumentParser(description='Stage 5: Auditing')
    parser.add_argument('--k', type=int, default=5,
                       help='Target K used for clustering')
    parser.add_argument('--output_dir', type=str, default='data/feature_discovery',
                       help='Directory containing labels and features')
    parser.add_argument('--results_dir', type=str, default='results/signal_clustering_v2',
                       help='Output directory for audit stats')
    parser.add_argument('--figs_dir', type=str, default='results/signal_clustering_v2/figs',
                       help='Output directory for figures')
    args = parser.parse_args()

    # Create output directories
    os.makedirs(args.results_dir, exist_ok=True)
    os.makedirs(args.figs_dir, exist_ok=True)

    print("=" * 60)
    print("Stage 5: Auditing")
    print("=" * 60)

    # Load labels
    labels_wavelet_path = os.path.join(args.output_dir, f'labels_wavelet_k{args.k}.npy')
    labels_raw_path = os.path.join(args.output_dir, f'labels_raw_k{args.k}.npy')

    if not (os.path.exists(labels_wavelet_path) and os.path.exists(labels_raw_path)):
        print("Error: Missing cluster label files. Cannot run audit.")
        return

    labels_wavelet = np.load(labels_wavelet_path)
    labels_raw = np.load(labels_raw_path)

    # ── 5.1: Cross-run Cluster Agreement ──────────────────────────
    print("\n" + "="*40)
    print("5.1: Cross-run Cluster Agreement")
    print("="*40)

    # Overlap summary CSV
    overlap_df, contingency = cross_run_overlap(labels_wavelet, labels_raw, args.k)
    overlap_csv = os.path.join(args.results_dir, 'cross_run_overlap_summary.csv')
    overlap_df.to_csv(overlap_csv, index=False)
    print(f"Saved: {overlap_csv}")

    # Contingency table CSV
    contingency = compute_contingency(labels_wavelet, labels_raw, args.k, args.k)
    cont_df = pd.DataFrame(contingency, index=[f'W_{i}' for i in range(args.k)],
                           columns=[f'R_{i}' for i in range(args.k)])
    cont_csv = os.path.join(args.results_dir, f'cross_run_contingency_k{args.k}.csv')
    cont_df.to_csv(cont_csv)
    print(f"Saved: {cont_csv}")

    # P16a: Contingency heatmap
    heatmap_fig = os.path.join(args.figs_dir, 'fig_contingency_heatmap_k5.png')
    plot_contingency_heatmap(contingency, heatmap_fig, args.k)

    # P16b: Procrustes-aligned 3D UMAP overlay
    fp_wavelet_path = os.path.join(args.output_dir, 'fingerprints_wavelet.npy')
    fp_raw_path = os.path.join(args.output_dir, 'fingerprints_raw.npy')

    if os.path.exists(fp_wavelet_path) and os.path.exists(fp_raw_path):
        print("\nComputing shared 3D UMAP overlay...")
        fp_wavelet = np.load(fp_wavelet_path)
        fp_raw = np.load(fp_raw_path)
        overlay_fig = os.path.join(args.figs_dir, 'fig_shared_umap_overlay_k5.html')
        plot_shared_umap_overlay_3d(fp_wavelet, fp_raw, labels_wavelet, labels_raw,
                                   overlap_df, overlay_fig, args.k)
    else:
        print("Warning: Missing fingerprint files. Skipping Procrustes overlay.")
        fp_wavelet = None
        fp_raw = None

    # ── 5.2: Separability Vector Greyscale Plots ─────────────────
    print("\n" + "="*40)
    print("5.2: Separability Vector Greyscale Plots")
    print("="*40)

    if fp_wavelet is None:
        fp_wavelet = np.load(fp_wavelet_path) if os.path.exists(fp_wavelet_path) else None
    if fp_raw is None:
        fp_raw = np.load(fp_raw_path) if os.path.exists(fp_raw_path) else None

    if fp_wavelet is not None:
        wavelet_gs_fig = os.path.join(args.figs_dir, 'fig_separability_greyscale_wavelet_k5.png')
        plot_separability_greyscale(fp_wavelet, labels_wavelet, 'Wavelet', wavelet_gs_fig, args.k)

    if fp_raw is not None:
        raw_gs_fig = os.path.join(args.figs_dir, 'fig_separability_greyscale_raw_k5.png')
        plot_separability_greyscale(fp_raw, labels_raw, 'Raw', raw_gs_fig, args.k)

    print("\n" + "=" * 60)
    print("Stage 5 Complete!")
    print("=" * 60)


if __name__ == '__main__':
    main()
