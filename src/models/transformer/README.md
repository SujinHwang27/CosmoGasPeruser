# Transformer-based Classification Pipeline

## Overview
This project implements a transformer model for a 20-feature, 4-class classification task.  
The repo is structured to support modular training, data ingestion, and model lifecycle workflows.

## File Breakdown
- `load_data.py` – dataset ingestion + preprocessing
- `transformer_utils.py` – transformer model + dataloaders + training loop
- `main.py` – orchestration entry point
- `plot.py` – training visualization utilities
- `__init__.py` – module definition

## Usage
Run training:


## Report
### Experiment 1
ACCURACY: 0.3495
PRECISION: 0.3327
RECALL: 0.3532
F1: 0.2993
Saved plot -> training_plot(1).png

### Experiment 2
Adjustments
in transformerforward():
Replaced this:
x = x.unsqueeze(1)
x = self.input_projection(x)
x = self.encoder(x)
x = x.squeeze(1)
With this:
# treat each feature as a token
x = x.unsqueeze(-1)               # (batch, 20, 1)
x = self.input_projection(x)      # (batch, 20, d_model)
x = self.encoder(x)               # (batch, 20, d_model)
x = x.mean(dim=1)                 # pooled representation
