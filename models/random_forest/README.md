# Random Forest Implementation

A structured implementation of Random Forest for both classification and regression tasks using scikit-learn.

## Project Structure

```
random_forest/
├── config/
│   └── config.yaml         # Configuration parameters
├── data/                   # Data directory
├── models/                 # Saved models directory
├── results/                # Results directory
├── src/
│   ├── model.py           # Random Forest model implementation
│   └── utils/
│       └── data_processor.py  # Data processing utilities
├── tests/                  # Test directory
├── main.py                 # Main execution script
└── requirements.txt        # Project dependencies
```

## Features

- Support for both classification and regression tasks
- Configurable through YAML file
- Data preprocessing utilities
- Model persistence
- Built-in evaluation metrics
- Organized project structure

## Installation

1. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Configuration

Edit `config/config.yaml` to configure:
- Model parameters (task type, number of trees, etc.)
- Data processing parameters
- Directory paths

## Usage

1. Place your data in the `data` directory as `data.csv`

2. Update the configuration in `config/config.yaml`:
   - Set the target column name
   - Specify categorical and numerical columns
   - Adjust model parameters as needed

3. Run the main script:
```bash
python main.py
```

The script will:
- Load and preprocess the data
- Train the model
- Evaluate performance
- Save the model and results

## Model Parameters

The implementation accepts all standard scikit-learn Random Forest parameters:

- `n_estimators`: Number of trees in the forest
- `max_depth`: Maximum depth of the trees
- `min_samples_split`: Minimum number of samples required to split a node
- `min_samples_leaf`: Minimum number of samples required at a leaf node
- And many more...

For a complete list of parameters, refer to the [scikit-learn documentation](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html).

## Evaluation Metrics

For classification tasks:
- Accuracy
- Classification report (precision, recall, F1-score)

For regression tasks:
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE) 