---
name: verification-loop
description: Post-change verification — lint, test, DVC status, import checks
---

# Verification Loop

## When to Use
After any code change — before committing. Run this checklist to catch broken imports, stale DVC paths, and regressions.

## Verification Steps

### 1. Import Check
```bash
# Verify all src/core/ modules import cleanly
PYTHONPATH=. uv run python -c "
import src.core.data
import src.core.probe
import src.core.cluster
import src.core.viz
import src.core.audit
import src.core.models.rf_classifier
import src.core.base
import src.core.transforms
import src.core.utils
print('All imports OK')
"
```

### 2. Tests
```bash
uv run pytest tests/ -v
```

### 3. DVC Pipeline Integrity
```bash
# Check that all deps in dvc.yaml point to existing files
dvc status
```

### 4. Type Consistency
```bash
# Quick grep for unimported types (common bug pattern in this repo)
grep -rn "Optional\|Any\|Dict\|List\|Tuple" src/core/*.py | grep -v "from typing import" | grep -v "^.*#"
```

## Common Failure Modes in This Repo
- `dvc.yaml` dep path doesn't match actual file location (e.g., `src/cluster.py` vs `src/core/cluster.py`)
- Missing `typing` imports (`Optional`, `Any`, `Dict`) — Python doesn't fail until the function is called
- `run_all.py` stage count out of sync with `dvc.yaml` stage count
- Modules placed at `src/` level instead of `src/core/` — imports from `src.core.X` fail silently

### 5. Dead Code Check
```bash
# Modules in src/core/ that nothing imports
for f in src/core/*.py; do
  name=$(basename "$f" .py)
  if [ "$name" != "__init__" ]; then
    count=$(grep -r "from src.core.$name import\|from src.core import.*$name\|import src.core.$name" scripts/ src/ tests/ 2>/dev/null | wc -l)
    if [ "$count" -eq 0 ]; then
      echo "UNUSED: $f"
    fi
  fi
done
```

### 6. Docs Freshness
- Check that `docs/` doesn't reference deleted files (`scripts/baseline/`, `src/core/models.py`, `src/main.py`)
- If a doc is a strict subset of another doc, delete the subset
