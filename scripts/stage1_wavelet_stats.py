"""
Stage 1: Per-Level Wavelet Feature Extraction
==============================================
Runs 7 experiments, one per decomposition level (D1-D6, A6).
For each level, extracts the wavelet coefficients for all 4 classes
and prints a data stats table showing the distribution across sightlines.

Output: Console table + saved per-level stats to data/feature_discovery/experiments/wavelet_per_level/
"""
import numpy as np
import os
import pywt
import argparse
from tqdm import tqdm

# --- Configuration ---
LEVELS = {
    "D1": (1, "Detail"),
    "D2": (2, "Detail"),
    "D3": (3, "Detail"),
    "D4": (4, "Detail"),
    "D5": (5, "Detail"),
    "D6": (6, "Detail"),
    "A6": (6, "Approximation"),
}

def extract_level_coeffs(flux, wavelet='db8', level_idx=1, is_approx=False):
    """
    Decomposes flux using DWT and returns the coefficients for one level.
    pywt.wavedec returns: [cA_n, cD_n, cD_{n-1}, ..., cD_1]
    """
    # Compute absorption field
    A = 1.0 - flux
    coeffs = pywt.wavedec(A, wavelet, mode='periodization', level=6)
    # [cA6, cD6, cD5, cD4, cD3, cD2, cD1]  (index 0 = approx, index 1 = D6, ..., index 6 = D1)
    if is_approx:
        return coeffs[0]  # A6
    else:
        # cD1 is at index 6, cD6 is at index 1
        return coeffs[7 - level_idx]

def compute_level_stats(X_all, level_name, level_idx, is_approx, wavelet='db8'):
    """
    For a single level, computes data stats across all sightlines.
    X_all: shape (n_probes, 4, n_pixels) - the 4 classes stacked per probe
    """
    n_probes = X_all.shape[0]
    all_coeffs = []
    
    for i in tqdm(range(n_probes), desc=f"  Extracting {level_name}", leave=False):
        c_per_class = []
        for cls in range(4):
            c = extract_level_coeffs(X_all[i, cls], wavelet=wavelet, level_idx=level_idx, is_approx=is_approx)
            c_per_class.append(c)
        all_coeffs.append(np.concatenate(c_per_class))  # All 4 classes concatenated
    
    data = np.array(all_coeffs)  # (n_probes, 4 * n_level_coeffs)
    
    coeff_len = data.shape[1] // 4  # length per class
    stats = {
        "Level": level_name,
        "Coeff Length": coeff_len,
        "Total Dim": data.shape[1],
        "Mean": float(np.mean(data)),
        "Std": float(np.std(data)),
        "Min": float(np.min(data)),
        "Max": float(np.max(data)),
        "Sparsity (|x|<1e-4)": float(np.mean(np.abs(data) < 1e-4)),
    }
    return stats, data

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/preprocessed/Sherwood_z0.3_inf", 
                        help="Directory with class folders 1-4 containing flux.npy")
    parser.add_argument("--output_dir", type=str, default="data/feature_discovery/experiments/wavelet_per_level",
                        help="Output directory for per-level stats")
    parser.add_argument("--wavelet", type=str, default="db8", help="Wavelet family")
    parser.add_argument("--n_probes", type=int, default=16384, help="Limit probes for testing (None = all)")
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load all flux data
    print("Loading flux data...")
    flux_classes = []
    for c in range(1, 5):
        path = os.path.join(args.data_dir, str(c), "flux.npy")
        flux_classes.append(np.load(path, mmap_mode='r'))
    
    n_probes = flux_classes[0].shape[0]
    if args.n_probes:
        n_probes = args.n_probes
    
    print(f"Running {n_probes} probes x 7 wavelet levels...\n")
    
    # Stack: shape (n_probes, 4, n_pixels)
    X_all = np.stack([fc[:n_probes] for fc in flux_classes], axis=1)
    
    # --- Run 7 experiments ---
    all_stats = []
    for level_name, (level_idx, kind) in LEVELS.items():
        is_approx = (kind == "Approximation")
        print(f"\n[{level_name}] Extracting coefficients...")
        stats, data = compute_level_stats(X_all, level_name, level_idx, is_approx, wavelet=args.wavelet)
        all_stats.append(stats)
        
        # Save per-level feature arrays
        out_path = os.path.join(args.output_dir, f"features_{level_name}.npy")
        np.save(out_path, data)
    
    # --- Print Stats Table ---
    header = f"\n{'Level':<8}{'CoeffLen':>10}{'TotalDim':>10}{'Mean':>10}{'Std':>10}{'Min':>10}{'Max':>10}{'Sparsity%':>12}"
    sep = "-" * len(header)
    print(f"\n{'='*60}")
    print("STAGE 1 — Wavelet Per-Level Data Stats")
    print(f"{'='*60}")
    print(header)
    print(sep)
    for s in all_stats:
        print(f"{s['Level']:<8}{s['Coeff Length']:>10}{s['Total Dim']:>10}"
              f"{s['Mean']:>10.4f}{s['Std']:>10.4f}{s['Min']:>10.4f}{s['Max']:>10.4f}"
              f"{100*s['Sparsity (|x|<1e-4)']:>11.1f}%")
    
    # Save stats as CSV
    import csv
    csv_path = os.path.join(args.output_dir, "stage1_data_stats.csv")
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=all_stats[0].keys())
        writer.writeheader()
        writer.writerows(all_stats)
    print(f"\nSaved stats to {csv_path}")

if __name__ == "__main__":
    main()
