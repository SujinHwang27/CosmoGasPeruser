"""
Stage 5: Auditing for Signal Clustering Analysis

Runs all three audit components:
- 5.1: Cross-run cluster overlap
- 5.2: Post-hoc feature attribution
- 5.3: Inter-class variance audit
"""

import argparse
import os
import numpy as np
import pandas as pd

from src.audit import (
    cross_run_overlap, plot_cross_run_overlap,
    wavelet_attribution, plot_wavelet_attribution,
    raw_attribution, plot_raw_attribution,
    variance_audit, plot_variance_ratios
)


def main():
    parser = argparse.ArgumentParser(description='Stage 5: Auditing')
    parser.add_argument('--run', type=str, choices=['wavelet', 'raw', 'both'], default='both',
                       help='Which clustering runs to audit')
    parser.add_argument('--k', type=int, default=8,
                       help='Number of clusters')
    parser.add_argument('--output_dir', type=str, default='data/feature_discovery',
                       help='Input directory for fingerprints/labels')
    parser.add_argument('--results_dir', type=str, default='results',
                       help='Output directory for tables')
    parser.add_argument('--figs_dir', type=str, default='figs',
                       help='Output directory for figures')
    args = parser.parse_args()

    # Create output directories
    os.makedirs(args.results_dir, exist_ok=True)
    os.makedirs(args.figs_dir, exist_ok=True)

    print("=" * 60)
    print("Stage 5: Auditing")
    print("=" * 60)

    # Load labels
    labels_wavelet = np.load(os.path.join(args.output_dir, f'labels_wavelet_k{args.k}.npy'))
    labels_raw = np.load(os.path.join(args.output_dir, f'labels_raw_k{args.k}.npy'))

    print(f"Loaded labels: wavelet={labels_wavelet.shape}, raw={labels_raw.shape}")

    # ============================================================
    # 5.1: Cross-Run Cluster Overlap
    # ============================================================
    print("\n" + "="*40)
    print("5.1: Cross-Run Cluster Overlap")
    print("="*40)

    overlap_df, contingency = cross_run_overlap(labels_wavelet, labels_raw, args.k)

    # Save tables
    contingency_path = os.path.join(args.results_dir, 'cross_run_contingency_k8.csv')
    pd.DataFrame(contingency).to_csv(contingency_path, index_label='wavelet_cluster')
    print(f"Saved: {contingency_path}")

    overlap_path = os.path.join(args.results_dir, 'cross_run_overlap_summary.csv')
    overlap_df.to_csv(overlap_path, index=False)
    print(f"Saved: {overlap_path}")

    # Plot
    overlap_plot_path = os.path.join(args.figs_dir, 'fig_cross_run_overlap.png')
    plot_cross_run_overlap(overlap_df, contingency, overlap_plot_path, args.k)

    print("\nCross-run overlap summary:")
    print(overlap_df.to_string(index=False))

    # ============================================================
    # 5.2: Post-Hoc Feature Attribution
    # ============================================================
    print("\n" + "="*40)
    print("5.2: Post-Hoc Feature Attribution")
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
