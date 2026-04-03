# Clustering Strategy Comparison: Sensitivity vs. Mechanism

This document records the design decision to pivot from **Sensitivity-based** (geometric) clustering to **Mechanism-based** (weight) clustering for sightline classification.

## The Goal
To cluster sightlines based on how physical realizations in different classes relate to one another, prioritized by the **physical reason** for their separation.

---

## Compared Strategies

### 1. Sensitivity-Based (Geometric)
*   **Vector**: 30-dim distance values from SVM decision boundaries (OVO).
*   **Definition**: Measures **"Intensity of Contrast."**
*   **Pros**: Computationally efficient, captures continuous physical "response" magnitude.
*   **Cons**: Groups sightlines that look different physically but have the same magnitude of separability (A "Hydrogen" dip and a "Carbon" dip might be grouped if they are both equally deep).
*   **Intuition**: "How much do the realizations differ?"

### 2. Mechanism-Based (Saliency) - **SELECTED**
*   **Vector**: 3072-dim vector of Linear SVM coefficients ($6 \times 512$ features).
*   **Definition**: Measures **"Physical Saliency."**
*   **Pros**: Groups sightlines that use the **same physical features** (specific wavelet scales) to distinguish classes. Enables one-to-one mapping back to physics.
*   **Cons**: High-dimensional (3072-dim), requires Linear Kernel (curved boundaries are approximated).
*   **Intuition**: "Why do the realizations differ?"

---

## Rationale for Selection (Option B)

The user objective is focused on understanding the **Realizations** themselves across classes. While Approach 1 identifies "where" interesting things happen, Approach 2 identifies "what" those things are.

### 3. Sparsity-Driven Mechanism (L1-Linear) - **SELECTED**
*   **Vector**: 3072-dim vector of sparse coefficients from `LinearSVC`.
*   **Definition**: Measures **"Minimal Physical Support."**
*   **Intuition**: "What is the smallest set of pixels that can separate these classes?"
*   **Advantage**: By using an **L1 Penalty**, we force most coefficients to exactly zero. This creates a "sparse" fingerprint that is incredibly easy to audit, as only the most informative wavelet scales will have non-zero values.

---

## The "Hard Margin" vs Regularization Trade-off

To achieve the "clearest separation" with the "minimal coefficients," we adjust the **C parameter**:
*   **High C (e.g., 100.0)**: Forces a **Hard Margin** on the 4 realizations, ensuring the separation is perfect.
*   **L1 Penalty**: Simultaneously forces **Sparsity**, ensuring that even with a hard margin, the SVM only uses the pixels that truly contribute to the signal.

| Feature | L2 (Standard Linear) | L1 (Sparse Linear) |
| :--- | :--- | :--- |
| **Weights** | Many small, non-zero values | A few high values, many zeros |
| **Auditability** | Difficult (dense signal) | High (sparse signal) |
| **Clustering** | Sensitive to noise | Focuses on physical pillars |

### Transition to Linear Kernel
To implement Mechanism-based clustering, the pipeline will pivot to **Linear SVMs**. While RBF kernels are superior for pure classification accuracy, their "reasoning" is hidden in a non-linear Hilbert space. Linear SVM coefficients provide a direct, auditable path from the cluster label to the spectral feature.

### Dimensionality Management
The 3072-dim space will be handled using **UMAP** for visualization to identify the structure of the "Physical Manifold" before applying K-Means.
