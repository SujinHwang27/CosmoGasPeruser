# CosmoGasPeruser

**CosmoGasPeruser** is a project for analyzing diffuse gas using cosmological simulations and machine learning.


### Project Structure
```
├── configs/        : Experiment YAML configurations
├── data/           : Dataset storage (DVC tracked)
├── reports/        : Generated plots and metrics
├── src/
│   ├── core/       : Shared library (data, transforms, models, viz)
│   └── main.py     : Central experiment orchestrator
```

### Usage
Run any experiment by specifying a configuration file:
```bash
uv run python src/main.py --config configs/example.yaml
```

### Configuration Example
All experiments are defined in YAML:
```yaml
name: "my_experiment"
data: "path/to/data"
transforms:
  - type: "dct"
    params: { mode: "full" }
  - type: "fisher"
    params: { top_k: 20 }
model:
  type: "transformer"
  params: { lr: 0.001, epochs: 10 }
```

## Legacy Architecture (Deprecated)
The previous structure using disconnected scripts in `src/models/` and `src/feature_selection/` is being phased out in favor of the core library.

---
Contact: **Sujin Hwang** (sujinhwang000@gmail.com)

