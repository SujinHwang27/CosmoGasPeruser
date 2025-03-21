"""
Configuration parameters for the project.
"""

import os
from pathlib import Path

# Get the project root directory (parent of the config directory)
PROJECT_ROOT = Path(__file__).parent.parent.parent

# Set data directory relative to project root
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "TargetedSpecML")

# List of physics values to use
PHYSICS_VALUES = ['no_feedback', 'strongAGN']

# REDSHIFT will be provided by user input

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
