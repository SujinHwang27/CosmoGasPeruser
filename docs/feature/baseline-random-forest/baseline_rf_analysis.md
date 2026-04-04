# Baseline Random Forest Analysis

**Branch:** `feature/baseline-random-forest` (tag: `v0.2-baseline-rf`)

## Objective

Establish a classification baseline using Random Forest on raw spectra and per-level wavelet coefficients across the 4 physics feedback classes. This benchmark validates multi-scale sensitivity and sets the floor for future methods.

## Method

- **Data:** 65,536 spectra (4 classes x 16,384 sightlines x 2,048 pixels)
- **Transform:** Absorption field A = 1 - F, then 6-level DWT (db8, periodization)
- **Model:** `RandomForestClassifier` with 10-fold stratified CV
- **Tracking:** MLflow experiment "Baseline_RF" with nested parent-child runs

For the theoretical basis of the A=1-F transformation and wavelet decomposition, see `docs/feature/signal-clustering-v1/feature_extraction_plan.md`.

## Results — 10-Fold Cross-Validation

| Experiment | Accuracy | Std Dev |
| :--- | :--- | :--- |
| **RF - Raw Spectra** | **0.4514** | 0.0023 |
| RF - Concatenated Wavelet | 0.3139 | 0.0030 |
| RF - Wavelet D1 | 0.3232 | 0.0031 |
| RF - Wavelet D2 | 0.2946 | 0.0031 |
| RF - Wavelet D3 | 0.2745 | 0.0043 |
| RF - Wavelet D4 | 0.2334 | 0.0028 |
| RF - Wavelet D5 | 0.2154 | 0.0021 |
| RF - Wavelet D6 | 0.2094 | 0.0031 |
| RF - Wavelet A6 | 0.2177 | 0.0015 |

## Key Finding: The Resolution Gap

Performance peaks at D1 (~0.32) and drops monotonically to D6 (~0.21). This confirms that the classification signal resides in **high-frequency morphology**. Raw spectra remain superior (0.45), suggesting that fixed-basis wavelets like db8 may lose localized information compared to learnable convolutional filters.

This finding motivated the signal clustering approach: rather than training a global classifier, identify *which spectral positions* carry the discriminative signal, then classify within those regions.

## Artifacts

Results in `results/baseline_rf/`:
- `baseline_summary.png` — Accuracy comparison bar chart (all levels + raw)
- `cm_rf_*.png` — Confusion matrices per wavelet level and raw spectra

## Legacy Notes

The original training script (`scripts/baseline/train_rf_baseline.py`) and model class (`src/core/models.py:BaselineRFClassifier`) were removed during cleanup — they are superseded by the v2 pipeline's per-cluster RF approach in `src/core/models/rf_classifier.py`. The results and confusion matrices remain in `results/baseline_rf/`.
