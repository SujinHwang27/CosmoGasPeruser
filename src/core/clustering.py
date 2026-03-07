import numpy as np
from sklearn.cluster import KMeans
from typing import Dict, List, Tuple
from src.core.base import BaseModel

class SignalClustered(BaseModel):
    """
    Logic for grouping sightlines into k-clusters based on feature similarity.
    """
    def __init__(self, n_clusters: int = 8, random_seed: int = 42):
        self.n_clusters = n_clusters
        self.random_seed = random_seed
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=random_seed, n_init='auto')
        self.labels_ = None

    def train(self, X: np.ndarray, y: Optional[np.ndarray] = None, **kwargs) -> Any:
        """
        Fits k-means to the features.
        """
        self.labels_ = self.kmeans.fit_predict(X)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Assigns new data to existing clusters.
        """
        return self.kmeans.predict(X)

    def get_cluster_stats(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """
        Calculates physics model distribution per cluster.
        """
        if self.labels_ is None:
            self.labels_ = self.kmeans.labels_
            
        stats = {}
        for cluster_id in range(self.n_clusters):
            cluster_mask = (self.labels_ == cluster_id)
            physics_counts = np.bincount(y[cluster_mask], minlength=5)[1:] # 1-4
            stats[cluster_id] = physics_counts
        return stats
