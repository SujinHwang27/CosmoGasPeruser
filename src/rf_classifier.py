import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, RepeatedStratifiedKFold, train_test_split

def prep_cluster_data(X_list, cluster_indices):
    """
    Prepare data for a specific cluster by extracting the corresponding sightline 
    indices from each of the 4 class observation arrays.

    Args:
        X_list: List of 4 np.ndarrays (each shape N x features).
        cluster_indices: 1D np.ndarray of indices that belong to this cluster.

    Returns:
        X_cluster, y_cluster: The combined training data and labels for the cluster.
    """
    X_cluster = []
    y_cluster = []
    for c_idx, X in enumerate(X_list):
        label = c_idx + 1  # Classes 1, 2, 3, 4
        X_comp = X[cluster_indices]
        X_cluster.append(X_comp)
        y_cluster.append(np.full(len(X_comp), label))
    
    return np.vstack(X_cluster), np.concatenate(y_cluster)

def train_rf_for_cluster(X, y, cluster_id, run_flavor):
    """
    Train a Random Forest classifier for a single cluster using RandomizedSearchCV.
    Logs metrics, params, and the model to MLflow.
    
    Args:
        X: Training features for this cluster.
        y: Labels for this cluster.
        cluster_id: The ID of the cluster (for logging).
        run_flavor: 'wavelet' or 'raw' (for logging).
        
    Returns:
        best_estimator, best_params, cv_score, test_score
    """
    # Create an MLflow run for this specific cluster
    with mlflow.start_run(run_name=f"{run_flavor}_cluster_{cluster_id}", nested=True):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )
        
        cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=42)

        param_dist = {
            'n_estimators':      [100, 200, 500],
            'max_depth':         [None, 5, 10, 20],
            'min_samples_leaf':  [1, 2, 5, 10],
            'min_samples_split': [2, 5, 10],
            'max_features':      ['sqrt', 'log2', 0.3, 0.5],
        }

        # Scale down n_iter if dataset is small, largely as a safeguard (though 100 is generally fine)
        n_iter = 100
        
        search = RandomizedSearchCV(
            RandomForestClassifier(random_state=42),
            param_distributions=param_dist,
            n_iter=n_iter,
            scoring='f1_weighted',
            cv=cv,
            n_jobs=-1,
            random_state=42
        )

        print(f"  -> Fitting RF for cluster {cluster_id} (train size: {len(y_train)}, test size: {len(y_test)})...")
        search.fit(X_train, y_train)

        best_score = search.best_score_
        test_score = search.score(X_test, y_test)
        
        print(f"     Best params: {search.best_params_}")
        print(f"     CV score (f1_weighted): {best_score:.4f}")
        print(f"     Test score: {test_score:.4f}")

        # Log params and metrics to MLflow
        mlflow.log_params(search.best_params_)
        mlflow.log_param("cluster_id", cluster_id)
        mlflow.log_param("run_flavor", run_flavor)
        mlflow.log_param("train_samples", len(y_train))
        mlflow.log_metric("cv_f1_weighted", best_score)
        mlflow.log_metric("test_score", test_score)
        
        # Log the scikit-learn model
        mlflow.sklearn.log_model(search.best_estimator_, f"model_cluster_{cluster_id}")
        
        return search.best_estimator_, search.best_params_, best_score, test_score
