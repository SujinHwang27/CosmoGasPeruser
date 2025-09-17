from sklearn.model_selection import train_test_split
import numpy as np
from typing import Tuple

import logging

 
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def split_data(labels:np.array) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    logger.info("Splitting data into train and test sets...")
    indices = np.arange(len(labels))

    train_idx, test_idx = train_test_split(
        indices, 
        test_size=0.1,  
        random_state=42,  
        stratify=labels  # maintain class distribution
    )
    logger.info(f"Training set size: {train_idx.shape}")
    logger.info(f"Test set size: {test_idx.shape}")

    return train_idx, test_idx