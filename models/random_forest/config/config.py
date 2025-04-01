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

# The number of total spectra per class
CLASS_SIZE = 6000

# The number of spectra per class used for the model training pipeline
DATA_SIZE = 6000

# REDSHIFT
REDSHIFT = 2.4