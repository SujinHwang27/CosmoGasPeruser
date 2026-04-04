---
description: "Testing rules for scientific ML code"
globs: ["tests/**/*.py", "src/**/*.py"]
alwaysApply: false
---

# Testing

## Framework
- pytest with `uv run pytest tests/`
- Test files: `tests/test_<module>.py` matching `src/core/<module>.py`

## What to Test
- Data loading: correct shapes, dtype, NaN/Inf checks
- Transforms: output shapes, value ranges, determinism with fixed seed
- Clustering: label count matches K, all samples assigned
- Probe: output is 24-dim, finite values, deterministic with seed
- Use small synthetic arrays (e.g., 10 sightlines x 8 features) not real data.

## Conventions
- Every `src/core/` module should have a corresponding test file.
- Tests must not depend on real data in `data/` — use fixtures with `np.random.default_rng(seed)`.
- Mark slow tests with `@pytest.mark.slow` for optional exclusion.
