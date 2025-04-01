

import umap
import xgboost as xgb
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from data_loader import load_data, load_wavelength
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


from sklearn.decomposition import PCA
from sklearn.metrics import normalized_mutual_info_score, silhouette_score




import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    spectra, labels = load_data(2.4)
    print(labels)
    
    # 2. Split data into train and test sets
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



    # Reduce dimensions with PCA
    pca = PCA(n_components=0.95)
    X_pca = pca.fit_transform(X_train)
    logger.info(f"X_pca shape: {X_pca.shape}")

    # Apply K-Means
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    y_kmeans = kmeans.fit_predict(X_pca)


    # Evaluate
    silhouette = silhouette_score(X_pca, y_kmeans)
    nmi = normalized_mutual_info_score(y_train, y_kmeans)

    print(f"Silhouette Score (PCA + K-Means): {silhouette:.3f}")
    print(f"NMI (PCA + K-Means): {nmi:.3f}")



    """
    # Reduce dimensions with UMAP
    umap_reducer = umap.UMAP(n_components=3, random_state=42)
    X_umap = umap_reducer.fit_transform(X_train)
    logger.info(f"{X_umap.shape}")

    # Apply K-Means
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    y_kmeans = kmeans.fit_predict(X_umap)

    # Plot results
    plt.scatter(X_umap[:, 0], X_umap[:, 1], c=y_kmeans, cmap='viridis', alpha=0.5)
    plt.title("UMAP + K-Means Clustering")
    plt.show()

    # Evaluate clustering
    silhouette_avg = silhouette_score(X_umap, y_kmeans)
    print(f"Silhouette Score (UMAP + K-Means): {silhouette_avg:.3f}")



    nmi = normalized_mutual_info_score(y_train, y_kmeans)
    print(f"Normalized Mutual Information (NMI): {nmi:.3f}")
    """


if __name__ == "__main__":
    main()