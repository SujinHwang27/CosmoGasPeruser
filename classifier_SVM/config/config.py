"""
Configuration parameters for the project.
"""

# Data parameters
DATA_DIR = "/home/sujin/CosmoGasPeruser/data/Spectra_for_Sujin"
# Physics values will be dynamically determined from the dataset
# The number of physics values can be >= 2 (nofeedback, strongAGN, etc.)
REDSHIFT = 0.1  # Single redshift value to process

# PCA parameters
EXPLAINED_VARIANCE = {
    '95': 0.95,
    '90': 0.90,
    '85': 0.85,
    '80': 0.80,
    '75': 0.75,
    '70': 0.70,
    '65': 0.65,
    '60': 0.60
}

# SVM hyperparameter grids
PARAM_GRID_1 = {
    'C': [1, 10, 100],
    'gamma': [0.1, 1, 10],
    'kernel': ['rbf', 'linear', 'poly']
}

PARAM_GRID_2 = {
    'C': [1, 0.1, 0.01],
    'gamma': [0.1, 1, 10],
    'kernel': ['poly']
}

# Model training parameters
TRAIN_TEST_SPLIT_RATIO = 0.2
RANDOM_STATE = 40
CV_FOLDS = 5

# Local minima parameters
LOCAL_MINIMA_SAMPLE_SIZE = 2000
NUM_MINIMA_SAMPLES = 2000
