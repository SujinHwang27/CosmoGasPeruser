"""
Data loading and preprocessing functionality.
"""

import os
import numpy as np
from typing import Dict, Tuple, List
import logging
from ..config.config import DATA_DIR, REDSHIFT

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_physics_values() -> List[str]:
    """
    Get available physics values from the data directory.
    
    Returns:
        List of physics values found in the data directory
    """
    physics_values = []
    redshift_str = str(REDSHIFT)
    
    # List all directories in the data directory
    for item in os.listdir(DATA_DIR):
        if item.startswith(redshift_str + "_") or item.startswith("_" + redshift_str):
            # Extract physics value from directory name
            physics = item.split("_")[0] if item.startswith(redshift_str + "_") else item.split("_")[1]
            if physics not in physics_values:
                physics_values.append(physics)
    
    # Sort physics values to ensure consistent ordering
    physics_values.sort()
    
    if len(physics_values) < 2:
        raise ValueError(f"Dataset must contain at least 2 physics values. Found: {len(physics_values)}")
    
    logger.info(f"Found physics values: {physics_values}")
    return physics_values

def load_data() -> Dict[str, np.ndarray]:
    """
    Load flux data for a single redshift.
    
    Returns:
        Dict[str, np.ndarray]: Dictionary containing flux data organized by physics values
        {physics_value: [[spectrum1],[spectrum2]...]}
    """
    physics_values = get_physics_values()
    data_by_physics = {physics: [] for physics in physics_values}
    redshift_str = str(REDSHIFT)
    
    for physics in physics_values:
        folder_name = f"{physics}_{redshift_str}"
        folder = os.path.join(DATA_DIR, folder_name)
        
        logger.info(f"Processing folder: {folder_name}")

        try:
            flux = np.load(os.path.join(folder, "flux.npy"))
            data_by_physics[physics] = flux
            logger.info(f"Flux data shape for physics {physics}: {flux.shape}")

        except FileNotFoundError as e:
            logger.error(f"Files not found in folder {folder}: {e}")
            del data_by_physics[physics]
            continue
        except Exception as e:
            logger.error(f"An error occurred while processing folder {folder}: {e}")
            continue

    if len(data_by_physics) < 2:
        raise ValueError(f"Dataset must contain at least 2 physics values. Found: {len(data_by_physics)}")

    return data_by_physics

def prepare_dataset(data_by_physics: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Prepare dataset from physics-based data.
    
    Args:
        data_by_physics: Dictionary containing flux data by physics value.
        
    Returns:
        Tuple containing:
        - Array of spectra
        - Array of corresponding labels
    """
    spectra = []
    labels = []

    for physics, fluxes in data_by_physics.items():
        for flux in fluxes:
            spectra.append(flux)
            labels.append(physics)
    
    return np.array(spectra), np.array(labels).astype(int)
