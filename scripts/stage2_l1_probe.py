"""
Stage 2: Per-Level L1-Linear Micro-Classifier Probing
=======================================================
Reads the per-level feature arrays produced by Stage 1.
For each level (D1-D6, A6), trains an L1-regularized LinearSVC 
over all 6 OVO class pairs for every sightline probe.

Configuration:
- L1 penalty (no explicit C -> defaults to C=1.0)
- No feature transform (raw wavelet coefficients as-is)
- OVO: 6 binary classifiers per probe

Output: Per-level coefficient stats table + mechanism fingerprint arrays.
"""
import numpy as np
import os
import csv
import argparse
from sklearn.svm import LinearSVC
from joblib import Parallel, delayed
from tqdm import tqdm

LEVEL_NAMES = ["D1", "D2", "D3", "D4", "D5", "D6", "A6"]

def probe_single(X_probe, C=1.0):
    """
    Trains 6 L1-LinearSVC classifiers (OVO) for a single probe.
    X_probe: shape (4, n_coeffs) — one row per class.
    Returns flattened weight vector across all 6 pairs.
    """
    n_classes = 4
    all_weights = []
    all_intercepts = []
    
    for i in range(n_classes):
        for j in range(i + 1, n_classes):
            X_pair = X_probe[[i, j]]  # shape (2, n_coeffs)
            y_pair = np.array([0, 1])
            
            clf = LinearSVC(penalty='l1', C=C, dual=False, random_state=42, max_iter=10000)
            clf.fit(X_pair, y_pair)
            
            all_weights.append(clf.coef_.flatten())
            all_intercepts.append(clf.intercept_)
    
    return np.concatenate([np.concatenate(all_weights), np.concatenate(all_intercepts)])

def compute_coeff_stats(weights, level_name, coeff_len):
    """
    Computes coefficient stats for a single level's mechanism fingerprints.
    weights: shape (n_probes, 6 * coeff_len + 6)
    """
    w = weights[:, :6 * coeff_len]  # Just the weights, not intercepts
    nnz = np.count_nonzero(w, axis=1)
    return {
        "Level": level_name,
        "Coeff Length": coeff_len,
        "Mechanism Dim": weights.shape[1],
        "Mean |w|": float(np.mean(np.abs(w))),
        "Std |w|": float(np.std(np.abs(w))),
        "Max |w|": float(np.max(np.abs(w))),
        "Sparsity%": float(100.0 * np.mean(w == 0)),
        "Avg Nonzero / Probe": float(np.mean(nnz)),
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default="data/feature_discovery/experiments/wavelet_per_level",
                        help="Directory containing Stage 1 feature arrays (features_D1.npy, etc.)")
    parser.add_argument("--output_dir", type=str, default="data/feature_discovery/experiments/wavelet_per_level",
                        help="Output directory for mechanism fingerprints and stats")
    parser.add_argument("--C", type=float, default=1.0, help="L1 regularization C (default=1.0)")
    parser.add_argument("--n_jobs", type=int, default=-1, help="Number of parallel jobs")
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    all_stats = []
    
    for level_name in LEVEL_NAMES:
        feat_path = os.path.join(args.input_dir, f"features_{level_name}.npy")
        if not os.path.exists(feat_path):
            print(f"[SKIP] {feat_path} not found. Run Stage 1 first.")
            continue
        
        print(f"\n[{level_name}] Loading features from {feat_path}")
        features = np.load(feat_path)  # shape (n_probes, 4 * coeff_len)
        n_probes = features.shape[0]
        coeff_len = features.shape[1] // 4  # Per-class feature count
        
        # Reshape to (n_probes, 4, coeff_len) for per-probe probing
        X = features.reshape(n_probes, 4, coeff_len)
        
        print(f"  Probing {n_probes} sightlines with L1-LinearSVC (C={args.C})...")
        results = Parallel(n_jobs=args.n_jobs)(
            delayed(probe_single)(X[i], C=args.C) for i in tqdm(range(n_probes), desc=f"  {level_name}")
        )
        
        weights = np.array(results)  # (n_probes, 6 * coeff_len + 6)
        
        # Save mechanism fingerprints
        out_path = os.path.join(args.output_dir, f"mechanism_{level_name}.npy")
        np.save(out_path, weights)
        
        stats = compute_coeff_stats(weights, level_name, coeff_len)
        all_stats.append(stats)
    
    # --- Print Stats Table ---
    if not all_stats:
        print("\nNo stats computed. Make sure Stage 1 has been run first.")
        return
    
    header = (f"\n{'Level':<8}{'CoeffLen':>10}{'MechDim':>10}"
              f"{'Mean|w|':>10}{'Std|w|':>10}{'Max|w|':>10}"
              f"{'Sparsity%':>12}{'AvgNNZ':>10}")
    sep = "-" * len(header)
    print(f"\n{'='*70}")
    print("STAGE 2 — L1-Linear Coefficient Stats (Per Level)")
    print(f"{'='*70}")
    print(header)
    print(sep)
    for s in all_stats:
        print(f"{s['Level']:<8}{s['Coeff Length']:>10}{s['Mechanism Dim']:>10}"
              f"{s['Mean |w|']:>10.4f}{s['Std |w|']:>10.4f}{s['Max |w|']:>10.4f}"
              f"{s['Sparsity%']:>11.1f}%{s['Avg Nonzero / Probe']:>10.1f}")
    
    # Save stats as CSV
    csv_path = os.path.join(args.output_dir, "stage2_coeff_stats.csv")
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=all_stats[0].keys())
        writer.writeheader()
        writer.writerows(all_stats)
    print(f"\nSaved coefficient stats to {csv_path}")

if __name__ == "__main__":
    main()
