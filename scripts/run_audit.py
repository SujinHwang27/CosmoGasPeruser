"""
Stage 5: Auditing for Signal Clustering Analysis

Performs statistical auditing:
- Cross-run cluster overlap
- Wavelet level attribution
- Raw spectral attribution
- Inter-class variance audit

Outputs:
    - results/*.csv: Audit statistics
    - figs/*.png: Audit visualizations
"""

import argparse
import os
import numpy as np
import pandas as pd
from pathlib import Path

from src.audit import (
    cross_run_overlap, plot_cross_run_overlap,
    wavelet_attribution, plot_wavelet_attribution,
    raw_attribution, plot_raw_attribution,
    variance_audit, plot_variance_ratios,
    compute_contingency
)


def main():
    parser = argparse.ArgumentParser(description='Stage 5: Auditing')
    parser.add_argument('--run', type=str, choices=['wavelet', 'raw', 'both'], default='both',
                       help='Which run to audit (both required for overlap/variance)')
    parser.add_argument('--k', type=int, default=5,
                       help='Target K used for clustering')
    parser.add_argument('--output_dir', type=str, default='data/feature_discovery',
                       help='Directory containing labels and features')
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

    # 1. Load labels (needed for overlap and variance)
    labels_wavelet_path = os.path.join(args.output_dir, f'labels_wavelet_k{args.k}.npy')
    labels_raw_path = os.path.join(args.output_dir, f'labels_raw_k{args.k}.npy')

    if os.path.exists(labels_wavelet_path) and os.path.exists(labels_raw_path):
        labels_wavelet = np.load(labels_wavelet_path)
        labels_raw = np.load(labels_raw_path)

        # 5.1: Cross-run Overlap
        print("\n" + "="*40)
        print("5.1: Cross-run Overlap Analysis")
        print("="*40)
        
        overlap_df = cross_run_overlap(labels_wavelet, labels_raw, args.k)
        overlap_csv = os.path.join(args.results_dir, 'cross_run_overlap_summary.csv')
        overlap_df.to_csv(overlap_csv, index=False)
        print(f"Saved: {overlap_csv}")

        contingency = compute_contingency(labels_wavelet, labels_raw, args.k, args.k)
        overlap_fig = os.path.join(args.figs_dir, 'fig_cross_run_overlap.png')
        plot_cross_run_overlap(overlap_df, contingency, overlap_fig, args.k)

        # 5.3: Inter-Class Variance Audit
        print("\n" + "="*40)
        print("5.3: Inter-Class Variance Audit")
        print("="*40)
        
        variance_df = variance_audit(labels_wavelet, labels_raw, args.k)
        variance_csv = os.path.join(args.results_dir, 'variance_audit.csv')
        variance_df.to_csv(variance_csv, index=False)
        print(f"Saved: {variance_csv}")

        variance_fig = os.path.join(args.figs_dir, 'fig_variance_ratios.png')
        plot_variance_ratios(variance_df, variance_fig)
    else:
        print("\nWarning: Missing labels for overlap/variance audit. Skipping.")

    # 5.2: Attribution (can be done per run)
    print("\n" + "="*40)
    print("5.2: Post-hoc Attribution")
    print("="*40)

    # Wavelet attribution
    print("\nComputing wavelet attribution...")
    try:
        w_attr = wavelet_attribution(args.output_dir, args.k)
        w_csv = os.path.join(args.results_dir, 'wavelet_attribution.csv')
        w_attr.to_csv(w_csv, index=False)
        w_fig = os.path.join(args.figs_dir, 'fig_wavelet_attribution_heatmap.png')
        plot_wavelet_attribution(w_attr, w_fig)
        print(f"Saved: {w_csv}, {w_fig}")
    except Exception as e:
        print(f"Error in wavelet attribution: {e}")

    # Raw attribution
    print("\nComputing raw spectral attribution...")
    try:
        r_attr = raw_attribution(args.output_dir, args.k)
        r_csv = os.path.join(args.results_dir, 'raw_attribution.csv')
        r_attr.to_csv(r_csv, index=False)
        r_fig = os.path.join(args.figs_dir, 'fig_raw_attribution_heatmap.png')
        plot_raw_attribution(r_attr, r_fig)
        print(f"Saved: {r_csv}, {r_fig}")
    except Exception as e:
        print(f"Error in raw attribution: {e}")

    print("\n" + "=" * 60)
    print("Stage 5 Complete!")
    print("=" * 60)


if __name__ == '__main__':
    main()
