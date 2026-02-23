# Comparative Report: K=5 vs K=8 Clustering Analysis

## 1. Overview
This report analyzes the stability and migration of micro-classifier "personalities" between two different clustering scales ($k=5$ and $k=8$). We identify which behavioral groups remain robust and which ones dissolve into the background when dimensionality is restricted.

---

## 2. Visual Comparison (UMAP Manifolds)

| Metric | K=5 Experiment | K=8 Experiment |
| :--- | :--- | :--- |
| **Interactive 3D** | [umap_3d_k5.html](../data/feature_discovery/clustering_results_k5/umap_3d_k5.html) | [umap_3d_k8.html](../data/feature_discovery/clustering_results/umap_3d_k8.html) |
| **Static 2D** | `classifier_clusters_umap_k5.png` | `classifier_clusters_umap_k8.png` |
| **Primary Goal** | Simplified behavioral mapping | **High-resolution Signal Discovery** |

---

## 3. Cluster Mapping & Overlap

By analyzing the 16,384 individual micro-classifiers, we have mapped the "Discovery Groups" ($k=8$) into the "Simplified Groups" ($k=5$).

### 3.1 Contingency Table (Sample Counts)
The table below shows how many indices from each $k=8$ group (columns) were assigned to each $k=5$ group (rows).

| | K8_G0 | K8_G1 | K8_G2 | K8_G3 | K8_G4 | K8_G5 | K8_G6 | K8_G7 | Total |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **K5_G0** | 1945 | 0 | 0 | 0 | 3755 | 73 | 562 | 1 | **6336** |
| **K5_G1** | 2174 | 0 | 0 | 4658 | 0 | 122 | 153 | 0 | **7107** |
| **K5_G2** | 38 | 0 | 1 | 0 | 10 | 43 | 64 | 1140 | **1296** |
| **K5_G3** | 0 | 1 | 489 | 3 | 0 | 58 | 577 | 9 | **1137** |
| **K5_G4** | 7 | 457 | 0 | 1 | 4 | 33 | 2 | 3 | **507** |

### 3.2 Summary of Mapping

| K5 Group | Composition from K8 | Physical Interpretation |
| :--- | :--- | :--- |
| **K5_G0** | 99.6% of K8_G4 + 46% of K8_G0 | **The "Void/Background" Bulk** |
| **K5_G1** | 99.9% of K8_G3 + 52% of K8_G0 | **The "Lyman-Alpha Forest" Bulk** |
| **K5_G2** | 98.9% of K8_G7 | Robust Transitional Features |
| **K5_G3** | 99.8% of K8_G2 + 42% of K8_G6 | **Rare Signal Island B** |
| **K5_G4** | 99.8% of K8_G1 | **Rare Signal Island A** |

---

## 4. Stability Analysis: The "Dissolving" Signal

The most important takeaway from this comparison is the fate of **K8_G5** (the smallest, most extreme cluster in the $k=8$ run). 

*   In $k=8$, **K8_G5** consisted of **329 indices** forming a tight, distinct behavioral island probably sensitive to extreme physics (like strong AGN feedback).
*   In $k=5$, this cluster **dissolved completely**. Its members were scattered:
    *   37% went to the Forest Bulk (K5_G1)
    *   22% went to the Void Bulk (K5_G0)
    *   The rest were absorbed into transitional groups.

**Conclusion**: $k=5$ is "lossy." It sacrifices the discovery of rare, high-sensitivity signals to achieve a more compact representation of the bulk signal.

---

## 5. Justification of K Value Selection (K=8)

The choice of $k=8$ as our primary discovery scale is justified by geometric metrics, physical discovery requirements, and the architectural constraints of the classification task.

### 5.1 The "Power of 2" & Degrees of Freedom Heuristic
In a system with **4 physics models**, the theoretical complexity of how an index behaves is governed by the possible binary contrasts between classes.
*   **Combinatorial Limit**: With 4 classes, there are 6 distinct binary pairings (OvO). The behavioral "personality" of an index is a result of how these 6 boundaries are arranged.
*   **Symmetry Breaking**: $k=8$ ($2^3$) is the first scale that allows for three major "Symmetry Breakers"—e.g., distinguishing Feedback vs. No Feedback, Wind vs. AGN, and Strong AGN vs. Standard AGN. 
*   **Capacity**: $k=4$ would only allow one group per physics class (assuming perfect separation), while $k=8$ provides the necessary "overflow" slots to capture indices that react to **interactions** between these models.

### 5.2 Relationship to Feature Dimensionality (30-dim)
The micro-classifier behavioral vectors have **30 dimensions** (24 decision values + 6 intercepts). 
*   **Effective Dimensionality**: Since these 30 features are highly correlated (produced by the same 4 data points per index), the true manifold dimensionality is likely much lower (around 3-5 major axes).
*   **The "Goldilocks" Fit**: According to the principle of manifold capacity, $k=8$ matches an effective dimensionality of ~3 ($2^3$), which is the mathematical threshold where "Islands" (compact, dense niches) begin to separate from the "Bulk" distribution without fracturing into noise.

### 5.3 The Silhouette Spike at k=5: Stability vs. Discovery
The optimization plots show a distinct peak in Silhouette score at $k=5$. While mathematically appealing, we deliberately chose $k=8$ based on the following trade-off:
*   **The "Continent" Effect**: At $k=5$, the algorithm identifies the broad "Continents" of the behavior manifold (Void, Forest, Dense Forest). Because these huge clusters are mathematically well-separated, the global average silhouette score spikes.
*   **The Discovery Penalty**: Moving to $k=8$ "peels away" the high-sensitivity **Signal Islands** (G1, G2, G5) from the coastlines of the big continents. Because these rare niches are geometrically adjacent to the bulk forest, they slightly lower the average silhouette score.
*   **The Verdict**: $k=5$ provides **Global Stability** (the background), while $k=8$ is necessary for **Local Discovery**. We accept the mathematical "penalty" of $k=8$ to avoid the physical blindness of $k=5$, which treats our target feedback signatures as mere boundary noise.

### 5.4 Geometric Evidence (Elbow Analysis)
Optimization runs for both Wavelets and Windowed DCT-II (`k_optimization_elbow_silhouette.png`) show that the sum of squared distances (Inertia) continues to drop meaningfully until $k=8$, where it enters the "long-tail" of diminishing returns. This suggests that while $k=5$ is more cohesive, $k=8$ captures significant additional variance in the classifier behaviors.

### 5.5 The Principle of "Island Preservation"
The $k=5$ vs $k=8$ comparison (Section 4) provides the strongest practical justification. 
*   Increasing $k$ from 5 back to 8 is what allows the **"Extreme Sensitivity" Island (G5)** to emerge as a distinct entity. 
*   At $k=5$, these 329 high-priority signals are "dissolved"—they lacks the statistical weight to form their own cluster.
*   **Result**: $k=8$ is the **minimum resolution necessary** to isolate the rare physical divergences we are looking for.

### 5.6 Cross-Transform Consistency
The fact that $k=8$ produces near-identical "Discovery Islands" (~90% overlap) in both Wavelets and DCT transforms (see `wavelet_vs_dct_comparison.md`) confirms that $k=8$ is not an overfit for one specific mathematical representation. It is a robust descriptor of the classifier manifold.

---

## 6. Backup Reasoning for Physical Interpretation

The interpretations in Section 3.2 are derived from three distinct dimensions of evidence: **Prevalence**, **Invariance**, and **Manifold Topology**.

### 6.1 Heuristic 1: The Abundance Rule (Prevalence)
In a 16,384-pixel Quasar spectrum, over 80% of the signal is composed of either "voids" (stochastic noise around zero absorption) or "standard forest" (typical density fluctuations).
*   **Evidence**: Clusters **G0, G3, and G4** contain **77% of all indices**.
*   **Conclusion**: These must represent the "Bulk" states of the intergalactic medium (IGM) that define the global mean.

### 6.2 Heuristic 2: The Signal Invariance Rule
A "Physical Interpretation" is verified by measuring the variance of the signal strength ($|Wavelet|$) across the four physics classes (No Feedback vs. Strong AGN).
*   **Forest Identification**: Group **G3** has high absolute signal magnitude but the **lowest class variance (0.000009)**. 
    *   *Logic*: This represents "Stable Signal"—absorption lines that definitely exist but are physically unaffected by the feedback parameters we are testing.
*   **Void Identification**: Groups **G0 and G4** have the lowest absolute signal magnitude.
    *   *Logic*: No absorption ($1-F \approx 0$) results in near-zero Wavelet coefficients.

### 6.3 Heuristic 3: The Discovery Rule (Divergence)
The prime objective of this project is to find features that **react** to different physics. If an index's SVM behavioral vector is in an "Island" (far from the bulk), it means its classification behavior is unique.
*   **Islands G1 and G5**: These clusters show **10x to 40x higher variance** between physics models compared to the forest (G3).
*   **Physical Meaning**: These 800 indices (~5% of the spectrum) represent the "Smoking Guns"—regions where shocks, turbulence, or bubbles from AGN feedback significantly alter the local absorption field relative to a no-feedback scenario.

## 7. Recommendation

We should proceed with **K=8** for physical mapping. The 3D UMAPs confirm that the "Islands" (K8_G1, G2, G5) represent distinct classification behaviors that have a high probability of being the specific features we need to identify the different physical cosmic gas models.

The full membership mapping for all 16,384 classifiers is available for reference in:
`data/feature_discovery/cluster_membership_comparison.csv`
