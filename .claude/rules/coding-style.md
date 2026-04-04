---
description: "Python coding style for scientific ML codebase"
globs: ["**/*.py"]
alwaysApply: true
---

# Coding Style

## Architecture
- All reusable logic in `src/core/`. Scripts in `scripts/` are thin orchestration wrappers only.
- Never place modules at `src/` level — only `src/core/` and below. (Learned: `src/viz.py` existed alongside `src/core/viz.py`, causing import failures.)
- Follow the ABC pattern: `BaseTransformer` for transforms, `BaseModel` for models.
- New modules in `src/core/` must have docstrings explaining scientific purpose.

## Python Conventions
- Type hints on all public functions (use `typing` imports: Optional, List, Dict, Tuple, Any).
- Every type used in annotations must be explicitly imported. Python only crashes when the annotated function is *called*, not when the module is imported — so CI import checks alone won't catch missing typing imports.
- NumPy arrays documented with shape comments: `# shape: (n_sightlines, 24)`
- Use `np.float64` for scientific computations (precision matters).
- Random seeds must be explicit parameters, never hardcoded without default.

## Naming
- "Separability vector" is the canonical term for the 24-dim SVM decision distance feature.
- File outputs use `fingerprints_*.npy` for backward compatibility but variable names use `separability_vectors`.
- Physics classes: 1=NoFeedback, 2=StellarWind, 3=WindAGN, 4=WindStrongAGN.

## Data Handling
- Never modify files in `data/` directly — DVC manages them.
- Use `SignalClusteringData` for loading, not raw `np.load()` in scripts.
- Validate data shapes and NaN/Inf before computation.
