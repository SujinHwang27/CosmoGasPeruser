# Baseline Random Forest Analysis

This document analyzes the proposed `rf_baseline.py` script and evaluates its alignment with the `CosmoGasPeruser` repository.

## Overview of Implementation

The baseline script implements a robust 10-fold Stratified Cross-Validation pipeline to evaluate Random Forest performance across two main modes:

1.  **Mode 1: Raw Spectra Classification**: Evaluates if the raw 1D signal contains sufficient information for classification without feature engineering.
2.  **Mode 2: Wavelet Coefficient Classification**:
    - **Per-Level**: Evaluated via `run_wavelet_rf()`. It treats each wavelet scale as a completely independent dataset.
    - **Concatenated (Global)**: Evaluated via `run_wavelet_concatenated_rf()`. It combines multiple scales into a single high-dimensional feature set.

### Comparison of Mode 2 Implementations

| Feature | Per-Level (`run_wavelet_rf`) | Concatenated (`run_wavelet_concatenated_rf`) |
| :--- | :--- | :--- |
| **Logic** | Trains $N$ separate models (one per level). | Trains 1 single model on combined data. |
| **Input** | Dict of level arrays: `{'D3': (n, 256), ...}`. | Same dict, but subsets are joined. |
| **Feature Shape** | Varies by level (e.g., 256 for D3, 32 for A6). | Fixed sum (e.g., 512 for combined D3-A6). |
| **Scaling** | Standard scaling per level array. | Mandatory independent scaling *before* concatenation. |
| **Goal** | Identify which physical scale is most discriminative. | Provide a global baseline for the MLP fusion. |
| **Key Function** | `run_cv()` called inside a loop over levels. | `run_cv()` logic integrated with concatenation steps. |

### Key Components
- **Hyperparameter Tuning**: Uses `RandomizedSearchCV` on a single fold to keep computation costs low while ensuring reasonable performance.
- **Scaling Strategy**: To prevent data leakage, the `StandardScaler` is **fitted only on the training fold**. The validation/test data is then **transformed** using that identical scaler. This ensures no information from the validation set influences the feature distributions of the training set.
- **Evaluation of Per-Level Models**: In the "Per-Level" mode, the script treats each wavelet scale as a separate, independent experiment. It produces **$N$ distinct classification reports and accuracy scores** (one for each level). This is intended to highlight which specific scales (e.g., broad structure in D6 vs. narrow lines in D3) carry the most physical information. It is *not* a multi-model voting system; the only "combined" prediction occurs in the **Concatenated** mode.
- **Robustness**: Uses `class_weight='balanced'` and 10-fold CV for statistical significance.

## Overfitting & Underfitting Management

The implementation addresses the bias-variance tradeoff through several layers of protection:

1.  **Stratified 10-Fold Cross-Validation**: Provides a statistically robust estimate of generalization error. Large gaps between training (if logged) and validation accuracy would indicate overfitting.
2.  **Hyperparameter Constraints**: The `RandomizedSearchCV` explores specific regularizing parameters:
    -   `max_depth`: Limits tree complexity by capping growth (range: 10-30 or None).
    -   `min_samples_leaf`: Forces leaves to contain multiple samples, smoothing the decision boundary and preventing the model from fitting individual noisy data points.
    -   `max_features`: Ensures each tree only sees a subset of features, decorrelating the trees in the ensemble.
3.  **Ensemble Averaging**: By its nature, Random Forest reduces variance (overfitting) by averaging multiple decorrelated decision trees. Each tree might overfit, but the collective ensemble typically generalizes well.
4.  **Validation on Unseen Data**: Accuracy is only reported on the validation folds, never on data used for fitting the specific model in that fold.

## Generated Artifacts

The script produces the following visualizations:
- `rf_ablation.png`: Comparative bar chart of accuracy across modes and stability plots across folds.
- `rf_feature_importance.png`: Bar charts showing Gini importance for features (comparable to attention weights).
- `rf_confusion_matrices.png`: Aggregated confusion matrices for each experiment.

## Repository Alignment & Required Modifications

During the review, the following discrepancies were identified:

| Item | Script (Demo) | Repository (Pre-processed) | Action Required |
| :--- | :--- | :--- | :--- |
| **Pixel Count** | 4,096 pixels | 2,048 pixels | Update script constants to match 2,048. |
| **Wavelet Levels** | D1 to A6 (All) | D3 to A6 (Partial) | Filter script to use available levels (D3-A6). |
| **Sample Size** | 16,384 total | 16,384 **per class** | Update loading logic to handle 65,536 total samples. |
| **Classes** | 4 Physical Modes | 4 Physical Modes | Matches. |
| **Data Format** | Dict of arrays | Folders `1/`, `2/`, `3/`, `4/` | Implement loading logic to aggregate `data.npy` results into (65536, 512). |

## Conclusion

The script is conceptually sound and follows the established experiment protocol (10-fold CV, stratified). Once the data loading and dimensionality constants are updated, it will serve as a strong baseline for the more advanced models in the pipeline.
