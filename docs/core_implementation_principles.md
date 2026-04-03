# Core Implementation Principles: CosmoGasPeruser

These principles guide the development of the CosmoGasPeruser codebase to ensure modularity, scalability, and clarity for both human researchers and AI agents.

## 1. Modular Architecture (`src/core/`)
All reusable logic must reside in the `src/core/` package. Execution scripts in `scripts/` should be thin wrappers that orchestrate these core components.

### 1.1 The Base Patterns
We follow an abstract-base-class (ABC) pattern to enforce consistency:
- **`BaseTransformer`**: For all feature extraction and data processing logic. Must implement `fit_transform(X, y)`.
- **`BaseModel`**: For all predictive models. Must implement `train(...)` and `predict(X)`.

### 1.2 Sub-Modules
- `src/core/data.py`: Centralized ingestion logic (e.g., `DataIngestor`).
- `src/core/models.py`: Classifier implementations (e.g., `BaselineRFClassifier`, `MicroProbingClassifier`).
- `src/core/transforms.py`: Physical transformations (Wavelet, DCT, etc.).
- `src/core/utils.py`: Generic utilities (logging, shape checking, directory management).

## 2. Development Guidelines

### 2.1 "No Logic in Scripts" Policy
- **DO**: Use scripts only for argument parsing, variable configuration, and calling `src.core` modules.
- **DON'T**: Implement complex loops, mathematical transformations, or ML model training directly inside a script.

### 2.2 Branch-Specific Maturity
- **Feature Branches (e.g., `baseline-rf`)**: Once a feature is "Baseline" or "Complete", logic should be fully integrated into `src/core/` and documented with a pipeline diagram.
- **Exploratory Branches (e.g., `signal-clustering`)**: Design should allow for flexible experimentation but follow the `BaseTransformer` pattern where possible to ease eventual migration.

### 2.3 Data Integrity & Validation
- **Shape Verification**: Always use `src.core.utils.check_array_shapes` when passing data between pipeline stages.
- **MLflow Tracking**: Use parent-child run relationships to group related experiments. Ensure confusion matrices and artifact logs are stored for every major run.

## 3. Documentation Ethics
- **Pipeline Diagrams**: Every major feature documentation should include a Mermaid diagram illustrating data flow.
- **Branch Naming**: Documentation folders in `docs/` must match their respective Git branch names.
- **Syncing**: All documentation is tracked via DVC to ensure multi-node synchronization without repo bloating.
