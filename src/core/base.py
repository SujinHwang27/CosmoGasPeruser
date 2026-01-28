from abc import ABC, abstractmethod
import numpy as np
from typing import Optional, Any

class BaseTransformer(ABC):
    """
    Abstract base class for all data transformation modules.
    Follows the Scikit-Learn 'fit_transform' pattern.
    """
    @abstractmethod
    def fit_transform(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Fits the transformer to the data and returns the transformed version.
        """
        pass

class BaseModel(ABC):
    """
    Abstract base class for all predictive model modules.
    """
    @abstractmethod
    def train(self, train_loader: Any, val_loader: Optional[Any] = None, **kwargs) -> Any:
        """
        Trains the model on the provided data loaders.
        """
        pass

    @abstractmethod
    def predict(self, X: Any) -> Any:
        """
        Makes predictions on new data.
        """
        pass
