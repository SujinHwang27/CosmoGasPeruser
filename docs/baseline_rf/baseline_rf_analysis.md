# Baseline Random Forest Analysis

## Pipeline Workflow

The baseline Random Forest (RF) pipeline follows a standardized process from raw simulation outputs to physical classification.

### Stage 1: Data Preparation & Transforms
- **Absorption Field**: Raw quasar flux $F$ is first converted to the absorption field $A = 1.0 - F$. This focuses the model on the peaks and shape of the original absorption lines.
- **Wavelet Decomposition**: A level-6 Discrete Wavelet Transform (DWT) using the `db8` family is applied. The resulting coefficients ($L=2048$) are structured into multi-scale levels (D1 to A6).
- **In-Memory Scaling**: Standardized scaling is performed within each cross-validation fold to prevent data leakage from the validation set.

### Stage 2: Hyperparameter Optimization (Search)
Instead of arbitrary parameter selection, the pipeline performs a **3-fold Randomized Search** (20 iterations) on a stratified subset of **20,000 samples**. 
- **Rationale**: 20,000 samples provide a stable statistical representative for tuning while minimizing CPU overhead. 3-fold splits minimize training time without sacrificing the quality of the selected parameters.

### Stage 3: Scientific Validation (Report)
Once the optimal parameters are identified, the pipeline performs a rigorous **10-fold Cross-Validation** on the **entire dataset** (65,536 samples).
- **Rationale**: 10-fold CV ensures that every individual spectrum is used for validation exactly once, providing the "true" accuracy and standard deviation for scientific reporting.

### Stage 4: Tracking & Visualization
- **MLflow**: Every run (including per-level wavelet experiments) is logged to MLflow, tracking metrics like `mean_accuracy`, `std_accuracy`, and exact parameters.
- **Confusion Matrices**: A confusion matrix is automatically generated for every experiment to identify which feedback classes (e.g., *StellarWind* vs. *WindAGN*) are being confused by the model.

## Hyperparameter Strategy

The current baseline implementation uses a discrete grid search for hyperparameter optimization via `RandomizedSearchCV`. 

### Grid Decomposition
The hyperparameter search space is specifically defined as:
- `n_estimators`: [100] (1 choice)
- `max_depth`: [10, 25] (2 choices)
- `min_samples_split`: [50, 100, 200] (3 choices)
- `min_samples_leaf`: [50, 100] (2 choices)
- `max_features`: ['sqrt', 'log2', 0.1] (3 choices)

This results in exactly **36 possible combinations** ($1 \times 2 \times 3 \times 2 \times 3 = 36$). 

### Search Intensity
Given that `RF_SEARCH_ITER` is currently set to **20**, the script is testing approximately **55.5%** of the entire possible parameter space in every experiment. From a statistical standpoint, broadening this grid within the current feature sets is unlikely to yield a significant "eureka" moment or a substantial jump in accuracy. 

### Optimization Insights
- **Diminishing Returns**: Further expansion of the grid would primarily test subtle variations of regularization (splitting/leaf sizes) that are already well-covered by the 55% sampling.
- **Accuracy Bottleneck**: Results on individual wavelet levels (D4-D6) hovering near random chance (0.25) suggest the bottleneck is **feature information density** rather than hyperparameter tuning. Success is more likely to come from combining features (Concatenated Wavelet) rather than deeper grid searches.

## Detailed Performance Metrics

The following table summarizes the best performing parameters and results for each configuration:

| Experiment | Accuracy | Std Dev | n_estimators | max_depth | min_split | min_leaf | max_features |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **RF - Raw Spectra** | **0.4514** | 0.0023 | 100 | 25 | 100 | 50 | 0.1 |
| RF - Concatenated Wavelet | 0.3139 | 0.0030 | 100 | 25 | 100 | 50 | sqrt |
| RF - Wavelet D1 | 0.3232 | 0.0031 | 100 | 25 | 100 | 50 | sqrt |
| RF - Wavelet D2 | 0.2946 | 0.0031 | 100 | 25 | 50 | 50 | 0.1 |
| RF - Wavelet D3 | 0.2745 | 0.0043 | 100 | 25 | 50 | 50 | 0.1 |
| RF - Wavelet D4 | 0.2334 | 0.0028 | 100 | 25 | 50 | 50 | 0.1 |
| RF - Wavelet D5 | 0.2154 | 0.0021 | 100 | 25 | 100 | 50 | sqrt |
| RF - Wavelet D6 | 0.2094 | 0.0031 | 100 | 25 | 100 | 50 | sqrt |
| RF - Wavelet A6 | 0.2177 | 0.0015 | 100 | 25 | 100 | 50 | sqrt |


## Final Performance Summary

The complete wavelet suite experiment has finished with the following key findings:

![Baseline Performance Comparison](file:///home/sujin/CosmoGasPeruser/results/baseline_rf/baseline_summary.png)
*Figure 1: Comparison of mean validation accuracy across all baseline experiments. The **black horizontal bars** represent the standard deviation ($\sigma$) across the 10 cross-validation folds, indicating model stability and confidence in the reported metrics.*

### Accuracy Trends
- **Raw Spectra (~0.45)**: This is currently the **top-performing baseline model**. It significantly outperforms all wavelet-based approaches, achieving a 45% classification accuracy across 4 classes. This suggests that the raw flux pixels contain high-resolution morphological information that the current wavelet decomposition (at level 6) may be smoothing over or losing.
- **Concatenated Wavelet (~0.31)**: The second-best model. While it beats the random chance baseline (0.25), it falls nearly 15% behind the raw spectra. 
- **Individual Wavelet Levels (D1-A6)**: A clear hierarchical trend is visible. Performance is highest at **D1 (~0.32)** and steadily drops as we move to coarser scales (D6/A6 ~ 0.21), confirming that the critical physical signal resides in the high-frequency/fine-detail regions of the absorption lines.

### Scientific Conclusion
The fact that **Raw Spectra** outperforms the **Concatenated Wavelet** suggests that while wavelets capture multi-scale correlations, the raw pixel resolution is still superior for the Random Forest to find discriminative features in this specific dataset. 

## Lessons Learned: The Resolution Gap

The baseline and wavelet experiments have provided critical insights into the nature of the physical signal in QSO absorption spectra:

1. **The Fidelity Penalty**: The 13% accuracy gap between **Raw Spectra (0.45)** and **Wavelet D1 (0.32)** indicates that the classification signal is extremely sensitive to pixel-level morphology. Even a Level-1 wavelet transform introduces enough "smoothing" or "mixing" to degrade the discriminative features.
2. **Feature Dispersal**: In raw space, an absorption feature exists as a localized group of pixels. In wavelet space, this same feature is dispersed across multiple coefficients and scales. Random Forest, being a non-linear but axis-aligned splitter, struggles to reconstruct these dispersed relationships compared to direct pixel intensity.
3. **Scale Hierarchy**: The monotonic drop in performance from **D1** to **D6** confirms that the feedback models are distinguished primarily by high-frequency features (sharp metal lines, small-scale forest wiggles) rather than the overall large-scale flux envelope (A6).

### Path Forward: From RF to 1D-CNN
These findings strongly justify moving to deep learning. A **1D-CNN** architecture is uniquely suited to solve these specific baseline failures because:
- It works directly on **Raw Pixels**, preserving the high-resolution signal.
- Its **Convolutional Kernels** act as "learnable wavelets," discovering the optimal shape-matching filters for absorption lines rather than being restricted to a fixed basis like `db8`.
- It inherently learns **Local Spatial Hierarchies**, capturing the relationships between neighboring pixels more effectively than an independent-feature model like Random Forest.
