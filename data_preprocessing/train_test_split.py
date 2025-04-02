from sklearn.model_selection import train_test_split
import numpy as np
from typing import Tuple

import logging

 
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def split_data(spectra:np.array, labels:np.array) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    logger.info("Splitting data into train and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        spectra, 
        labels, 
        test_size=0.1,  
        random_state=42,  
        stratify=labels  # maintain class distribution
    )
    logger.info(f"Training set shape: {X_train.shape}")
    logger.info(f"Test set shape: {X_test.shape}")

    return X_train, X_test, y_train, y_test