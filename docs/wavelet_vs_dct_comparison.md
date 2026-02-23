# Comparative Analysis: Wavelets vs. Windowed DCT-II

## 1. Overview
This report compares two alternative signal transformation approaches for feature discovery: **Multi-Resolution Wavelet Transform** (`db8`) and **Windowed DCT-II**. The goal was to verify if the "Signal Discovery" patterns (the behavioral islands) are consistent across different mathematical representations of the absorption field.

---

## 2. Methodology Alignment
*   **Common Pipeline**: Both experiments used the Absorption Field ($1-F$) and trained 16,384 Micro-SVMs.
*   **Wavelet Specs**: Level 6 `db8`, dropping $D_1/D_2$. (512-dim features)
*   **DCT Specs**: 256-pixel windows, 128-pixel hop, 32 coefficients. (480-dim features)

---

## 3. Findings: The "Stability of Islands"

The most significant finding is that the **Rare Signal Islands are mathematically robust**. The specific indices identified as "discriminative" by Wavelets were also identified as "discriminative" by the Windowed DCT approach, despite the fundamental difference in how they decompose the signal.

### 3.1 Mapping of High-Sensitivity Signals
We calculated the overlap between the two experiments. Below is the mapping of the most critical discovery groups:

| Wavelet Cluster | Best DCT Match | Overlap % | Discovery Context |
| :--- | :--- | :---: | :--- |
| **Wave_G5** (329 pts) | **DCT_G3** (306 pts) | **87.9%** | **Extreme Sensitivity Niche** |
| **Wave_G1** (458 pts) | **DCT_G6** (450 pts) | **90.9%** | **Rare Island A** |
| **Wave_G2** (490 pts) | **DCT_G2** (500 pts) | **85.0%** | **Rare Island B** |

**Observation**: The near-90% overlap for the smallest clusters confirms that these are not artifacts of the Wavelet transform. They represent genuine physical features in the cosmic gas models that any localized transformation will detect.

### 3.2 Mapping of the Bulk Distribution
The large "background" clusters also showed high alignment:

| Wavelet Cluster | Best DCT Match | Overlap % | Interpretation |
| :--- | :--- | :---: | :--- |
| **Wave_G3** (Forest) | **DCT_G4** | 89.7% | Regular Absorption Pattern |
| **Wave_G4** (Void) | **DCT_G1** | 89.5% | Empty Background |
| **Wave_G0** (Bulk) | **DCT_G0** | 82.1% | Transitional Bulk |

---

## 4. Visual Evidence
Static 2D UMAP comparison:
*   **Wavelet 2D**: `data/feature_discovery/clustering_results/classifier_clusters_umap.png`
*   **DCT 2D**: `data/feature_discovery/clustering_results_dct/classifier_clusters_umap.png`

**Interactive 3D Manifolds**:
*   [Wavelet 3D](../data/feature_discovery/clustering_results/umap_3d_k8.html)
*   [Windowed DCT 3D](../data/feature_discovery/clustering_results_dct/umap_3d_dct_k8.html)

---

## 5. Conclusion
The Windowed DCT-II experiment serves as a **cross-validation**. The fact that ~90% of the indices identified as "High Sensitivity" are identical across both methods provides high confidence in our discovery. 

**Next Recommendation**: Since Wavelets inherently provide better localized scale separation (D1-D6), and the results are consistent, we should continue using the **Wavelet-based indices** for the final physical mapping of feedback physics.
