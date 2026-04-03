# Report: Decision Process for the Feature Discovery Pipeline

This report documents the technical rationale and decision-making process behind the transition from a standard classification approach to the **Feature Discovery Pipeline**.

## 1. Context: The "Extreme Scarcity" Constraint
The defining challenge of this project is training on **4 samples with 2048 features**. 
- In this regime, traditional cross-validation is impossible.
- Any standard classifier will "overfit" by finding a random wavelength that perfectly separates 4 points.
- **Decision**: Shift the objective from *Classification Accuracy* to *Structural Signal Identification*.

## 2. Validation of the Input Data (Step 1)
Before approving the new pipeline, I verified the statistics of the `flux.npy` files.
- **Finding**: Raw flux is already scaled to `[0, 1]` (Mean: 0.97, Max: 1.0).
- **Inference**: The data is pre-processed for physical consistency. This confirms that we should move directly to the DCT phase rather than re-normalizing the raw input.

## 3. Rationale for DCT over Raw Flux
The choice of DCT-II (ortho) was evaluated against using raw flux.
- **Decision**: DCT is superior for **Discovery** because it acts as a **Structural Regularizer**.
- **Reasoning**: In raw space, neighbors are correlated. In DCT space, the model must select frequency components. Selecting a frequency component is physically more significant than selecting a single wavelength, as it implies a structured shape difference between classes.

## 4. The Truncation Filter (K=256)
A key decision in the last reply was endorsing the truncation of the frequency spectrum.
- **Reasoning**: High-frequency DCT coefficients often capture instrumental noise or pixel-level artifacts. 
- **Decision**: By keeping only the first 256–512 coefficients, we "force" the L1 penalty to choose from the structural (low-to-mid frequency) features of the spectra, making the resulting probes more robust to noise.

## 5. Standardizing for Weight Interpretability
We previously analyzed why raw weights $w_i$ were not proportional to importance (due to different feature scales).
- **Decision**: Adopt the user's suggestion for **Per-Probe Standardization**.
- **Impact**: By forcing every DCT coefficient to have unit variance within the 4-sample probe, the L1 penalty $(\lambda |w|)$ treats them equally. This makes the weight magnitude $|w_i|$ a direct and unbiased measure of "discriminative power."

## 6. Solver Selection: SAGA vs. Liblinear
While Liblinear is excellent for pure classification, **SAGA (Multinomial)** was chosen for discovery.
- **Reasoning**: Multinomial LogReg (Softmax) handles the 4-class contrast simultaneously rather than in 4 separate OneVsRest passes. This ensures that the discovered features are the ones that best separate the *entire class set* globally.

## Summary Checklist for the Next Phase
- [x] Verified flux scaling is stable ([0,1]).
- [x] Confirmed DCT-II Truncation as a noise filter.
- [x] Validated Z-Score standardization as the path to weight interpretability.
- [x] Shifted to Multinomial SAGA for global contrast discovery.
