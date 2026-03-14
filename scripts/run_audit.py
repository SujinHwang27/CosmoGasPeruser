"""
Stage 5: Auditing for Signal Clustering Analysis

Performs statistical auditing and confusion matrix analysis
for the clustering results.

Outputs:
    - results/audit_stats_*.csv: Cluster-class alignment statistics
    - figs/fig_audit_cm_*.png: Confusion matrices
"""

import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from project_src.core.data import SignalClusteringData
from project_src.audit import perform_cluster_audit, plot_audit_confusion_matrix


def main():
    parser = argparse.ArgumentParser(description='Stage 5: Auditing')
    parser.add_argument('--run', type=str, choices=['wavelet', 'raw', 'both'], default='both',
                       help='Which run to audit')
    parser.add_argument('--k', type=int, default=5,
                       help='Target K used for clustering')
    parser.add_argument('--output_dir', type=str, default='data/feature_discovery',
                       help='Directory containing labels')
    parser.add_argument('--results_dir', type=str, default='results',
                       help='Output directory for audit stats')
    parser.add_argument('--figs_dir', type=str, default='figs',
                       help='Output directory for figures')
    args = parser.parse_args()

    # Create output directories
    os.makedirs(args.results_dir, exist_ok=True)
    os.makedirs(args.figs_dir, exist_ok=True)

    print("=" * 60)
    print("Stage 5: Auditing")
    print("=" * 60)

    print("="*40)

    # Wavelet attribution
    print("\nComputing wavelet attribution...")
    wavelet_attr = wavelet_attribution(args.output_dir, args.k)
    wavelet_attr_path = os.path.join(args.results_dir, 'wavelet_attribution.csv')
    wavelet_attr.to_csv(wavelet_attr_path, index=False)
    print(f"Saved: {wavelet_attr_path}")

    wavelet_fig_path = os.path.join(args.figs_dir, 'fig_wavelet_attribution_heatmap.png')
    plot_wavelet_attribution(wavelet_attr, wavelet_fig_path)

    # Raw attribution
    print("\nComputing raw spectral attribution...")
    raw_attr = raw_attribution(args.output_dir, args.k)
    raw_attr_path = os.path.join(args.results_dir, 'raw_attribution.csv')
    raw_attr.to_csv(raw_attr_path, index=False)
    print(f"Saved: {raw_attr_path}")

    raw_fig_path = os.path.join(args.figs_dir, 'fig_raw_attribution_heatmap.png')
    plot_raw_attribution(raw_attr, raw_fig_path)

    # ============================================================
    # 5.3: Inter-Class Variance Audit
    # ============================================================
    print("\n" + "="*40)
    print("5.3: Inter-Class Variance Audit")
    print("="*40)

    variance_df = variance_audit(labels_wavelet, labels_raw, args.k)

    # Save table
    variance_path = os.path.join(args.results_dir, 'variance_audit.csv')
    variance_df.to_csv(variance_path, index=False)
    print(f"Saved: {variance_path}")

    # Plot
    variance_fig_path = os.path.join(args.figs_dir, 'fig_variance_ratios.png')
    plot_variance_ratios(variance_df, variance_fig_path)

    print("\nVariance audit summary:")
    print(variance_df[['run', 'cluster', 'size', 'inter_class_variance', 'normalized_variance', 'regime']].to_string(index=False))

    # ============================================================
    # Summary
    # ============================================================
    print("\n" + "=" * 60)
    print("Stage 5 Complete!")
    print("=" * 60)
    print(f"\nOutputs generated:")
    print(f"  Tables: {args.results_dir}/")
    print(f"    - cross_run_contingency_k8.csv")
    print(f"    - cross_run_overlap_summary.csv")
    print(f"    - variance_audit.csv")
    print(f"    - wavelet_attribution.csv")
    print(f"    - raw_attribution.csv")
    print(f"  Figures: {args.figs_dir}/")
    print(f"    - fig_cross_run_overlap.png")
    print(f"    - fig_wavelet_attribution_heatmap.png")
    print(f"    - fig_raw_attribution_heatmap.png")
    print(f"    - fig_variance_ratios.png")


if __name__ == '__main__':
    main()
