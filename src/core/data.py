import numpy as np
import os
from typing import Tuple, List, Optional

class DataIngestor:
    """
    Standardizes data ingestion for Sherwood-style datasets.
    """
    def __init__(self, base_path: str, filename: str = "flux.npy", num_classes: int = 4):
        self.base_path = base_path
        self.filename = filename
        self.num_classes = num_classes

    def load(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Loads and concatenates data from class-specific directories.
        """
        X_list = []
        y_list = []
        
        for i in range(1, self.num_classes + 1):
            file_path = os.path.join(self.base_path, str(i), self.filename)
            if not os.path.exists(file_path):
                # Fallback check if the user provided generic path but file exists as 'data.npy'
                if self.filename == "flux.npy" and os.path.exists(os.path.join(self.base_path, str(i), "data.npy")):
                    file_path = os.path.join(self.base_path, str(i), "data.npy")
                else:
                    raise FileNotFoundError(f"Data file not found at {file_path}")
            
            data = np.load(file_path)
            X_list.append(data)
            y_list.append(np.full(len(data), i))
            
        X = np.vstack(X_list)
        y = np.concatenate(y_list)
        return X, y

class DatasetFactory:
    """
    Factory for retrieving pre-split or raw datasets.
    """
    @staticmethod
    def get_dataset(name: str, base_data_dir: str = "data", filename: str = "flux.npy") -> Tuple[np.ndarray, np.ndarray]:
        # Implementation for specific dataset names if needed
        # For now, default to checking the data directory structure
        dataset_path = os.path.join(base_data_dir, "processed", name)
        if not os.path.exists(dataset_path):
            dataset_path = os.path.join(base_data_dir, name) # check top level data dir
            
        ingestor = DataIngestor(dataset_path, filename=filename)
        return ingestor.load()
