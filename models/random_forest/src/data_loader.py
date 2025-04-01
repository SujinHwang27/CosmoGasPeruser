"""
Data loading and preprocessing functionality.
"""

import os
import numpy as np
from typing import Dict, Tuple, List
import logging
from pathlib import Path
import sys
import glob

# Add the project root to Python path
project_root = str(Path(__file__).parent.parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

# Import config using absolute import
from config.config import DATA_DIR, PHYSICS_VALUES, DATA_SIZE, REDSHIFT

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_physics_values(redshift: float) -> List[str]:
    """
    Get physics values from config.
    
    Args:
        redshift: The redshift value to process (kept for compatibility)
        
    Returns:
        List of physics values from config
    """
    if len(PHYSICS_VALUES) < 2:
        raise ValueError(f"Dataset must contain at least 2 physics values. Found: {len(PHYSICS_VALUES)}")
    
    logger.info(f"Using physics values: {PHYSICS_VALUES}")
    return PHYSICS_VALUES

def load_data(redshift: float = REDSHIFT) -> Dict[str, np.ndarray]:
    """
    Load flux data for a single redshift.
    
    Args:
        redshift: The redshift value to process
        
    Returns:
        Dict[str, np.ndarray]: Dictionary containing flux data organized by physics values
        {physics_value: [[spectrum1],[spectrum2]...]}
    """
    logger.info(f"Loading data for redshift {redshift}...")
    physics_values = get_physics_values(redshift)
    data_by_physics = {physics: [] for physics in physics_values}
    redshift_str = str(redshift)
    
    for physics in physics_values:
        # folder_name = f"{physics}_{redshift_str}"
        folder_name = f"{physics}"
        folder = os.path.join(DATA_DIR, folder_name)
        
        logger.info(f"Processing folder: {folder_name}")

        try:
            flux_files = glob.glob(os.path.join(DATA_DIR, folder, "**/flux.npy"), recursive=True)
            logger.info(f"Found {len(flux_files)} flux.npy files")

            for flux_file in flux_files:
                logger.info(f"Loading flux file {flux_file}")
                flux = np.load(flux_file)
                logger.info(f"{flux.shape[0]} spectra in {flux_file}")
                data_by_physics[physics].extend(flux)
                logger.info(f"Current total spectra for {physics}: {len(data_by_physics[physics])}")

            # Log the total accumulated data shape after all files are processed
            total_fluxes = np.array(data_by_physics[physics])
            logger.info(f"Total accumulated flux data shape for physics {physics}: {total_fluxes.shape}")

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

def prepare_dataset(data_by_physics: Dict[str, np.ndarray], size: int = DATA_SIZE) -> Tuple[np.ndarray, np.ndarray]:
    """
    Prepare dataset from physics-based data.
    
    Args:
        data_by_physics: Dictionary containing flux data by physics value.
        size: Optional int, if provided, randomly sample this many spectra from each physics value.
        
    Returns:
        Tuple containing:
        - Array of spectra
        - Array of corresponding integer labels
    """
    spectra = []
    labels = []

    # Create mapping between physics values and integers
    physics_to_int = {physics: idx for idx, physics in enumerate(sorted(data_by_physics.keys()))}
    logger.info("Physics value to integer mapping:")
    for physics, idx in physics_to_int.items():
        logger.info(f"  {physics} -> {idx}")

    for physics, fluxes in data_by_physics.items():
        # Convert fluxes list to numpy array
        fluxes_array = np.array(fluxes)
        
        # Randomly sample 'size' spectra from each physics value
        if len(fluxes_array) <= size:
            logger.warning(f"Physics {physics} has fewer or equal samples ({len(fluxes_array)}) than requested size ({size}). "
                            "Using all available samples.")
            selected_fluxes = fluxes_array
        else:
            indices = np.random.choice(len(fluxes_array), size=size, replace=False)
            selected_fluxes = fluxes_array[indices]

            
        for flux in selected_fluxes:
            spectra.append(flux)
            labels.append(physics_to_int[physics])
    
    spectra = np.array(spectra)
    labels = np.array(labels)
    
    logger.info(f"Final dataset shape: {spectra.shape}")
    logger.info(f"Number of samples per class: {np.bincount(labels)}")
    
    return spectra, labels
