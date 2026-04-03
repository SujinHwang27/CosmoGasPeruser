# Comparison Report: Liblinear+OVR, SVM, and SAGA

This report compares three solvers for training sparse linear probes on high-dimensional data (2048 features) with very few samples per probe (4 samples).

## Theoretical Comparison

| Feature | Liblinear (LogReg) | SVM (LinearSVC) | SAGA (LogReg) |
| :--- | :--- | :--- | :--- |
| **Optimization** | Coordinate Descent | Coordinate Descent | Stochastic Average Gradient |
| **Loss Function** | Logistic Loss | Hinge Loss (Squared) | Logistic Loss |
| **Penalty** | L1 / L2 | L1 / L2 | L1 / L2 / ElasticNet |
| **Scaling** | Good for "n small, p large" | Good for "n small, p large" | Better for "n large, p large" |
| **Sparsity** | High (Exact 0s) | High (Exact 0s) | Moderate (Often needs low tol) |

- **Liblinear**: The gold standard for linear classification on datasets where the number of features $p$ is much larger than the number of samples $n$. It uses coordinate descent, which is extremely efficient at finding sparse solutions for L1 penalties.
- **SVM (LinearSVC)**: When using `LinearSVC(penalty='l1', dual=False)`, it internally uses the `liblinear` library. The primary difference is the objective function (Hinge vs Log loss). For linearly separable data in high dimensions, both usually find similar separators.
- **SAGA**: A faster alternative to traditional GD/SGD for large datasets. However, because it's an iterative stochastic solver, it typically requires more iterations to reach the exact sparse solution that coordinate descent (liblinear) finds almost immediately on small $n$.

---

## Empirical Results (Reality)

Benchmarks were run on **200 probes**, each with **4 samples** and **2048 features**, using an inverse regularization strength **C = 1000**.

### Statistics
| Model | Avg Time / Probe | Overall Sparsity | Consistency |
| :--- | :--- | :--- | :--- |
| **Liblinear (OVR)** | 55.29 ms | **99.30%** | Baseline |
| **SVM (LinearSVC)** | 54.68 ms | **99.16%** | High Overlap |
| **SAGA** | 50.31 ms | 81.12% | Low Sparsity |

### Analysis
1. **Performance Balance**: `liblinear` and `svm` performed almost identically in both speed and sparsity. This confirms that for $n=4, p=2048$, coordinate descent is superior.
2. **SAGA Sparsity Issue**: Despite being slightly faster, `SAGA` achieved significantly lower sparsity (81% vs 99%). In high-dimensional "small-n" problems, the stochastic nature of SAGA is less efficient at "zeroing out" features compared to the greedy coordinate descent.
3. **Data Normalization**: Inspection showed that the DCT data is **not normalized** (Feature 0/DC has mean ~44, while others are ~0.08).
    - `liblinear` and `svm` handled this well, likely due to the robustness of the coordinate descent algorithm.
    - `SAGA` might require better scaled features to converge to a sparse solution more effectively.

## Conclusion

For the current "Probe Training" task (small samples, high features, high sparsity requirement), **Liblinear+OVR** or **LinearSVC** are the best choices. **SAGA is not recommended** as it fails to achieve the desired levels of sparsity without significant tuning.
