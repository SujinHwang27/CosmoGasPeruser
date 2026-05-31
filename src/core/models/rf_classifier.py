import numpy as np
import mlflow
import mlflow.sklearn
from typing import Dict, List, Any
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


def train_rf_global(
    X: np.ndarray,
    y: np.ndarray,
    *,
    seed: int = 42,
    n_estimators: int = 300,
    return_importances: bool = True,
) -> Dict[str, Any]:
    """
    Train a single global Random Forest 4-class classifier on (X, y) with a
    stratified 80/20 hold-out split. NO GridSearchCV. NO MLflow. Thin sibling
    to `train_rf_for_cluster`, used by the pk-feedback-classifier de-risking
    probe (experiments/pk-feedback-classifier).

    Parameters
    ----------
    X : np.ndarray, shape (n_samples, n_features), float64
    y : np.ndarray, shape (n_samples,), integer class labels (any contiguous
        encoding; this routine uses sklearn's ordering).
    seed : int, default 42. Controls both the RF and the split.
    n_estimators : int, default 300.
    return_importances : bool, default True. If False, the 'feature_importances'
        dict entry is an empty np.ndarray (skips the attribute read).

    Returns
    -------
    dict with keys:
        balanced_acc_test : float — sklearn.metrics.balanced_accuracy_score on
            the 20% hold-out.
        confusion_test : np.ndarray, shape (n_classes, n_classes),
            row-normalized (true-class rows sum to 1).
        feature_importances : np.ndarray, shape (n_features,) or empty.
        n_train : int
        n_test : int
        class_labels : list — sklearn class ordering used for the confusion rows/cols.
    """
    X = np.asarray(X)
    y = np.asarray(y)
    if X.ndim != 2:
        raise ValueError(f"X must be 2-D, got shape {X.shape}.")
    if y.shape[0] != X.shape[0]:
        raise ValueError(
            f"X and y first-dim mismatch: X={X.shape[0]}, y={y.shape[0]}."
        )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=seed
    )

    rf = RandomForestClassifier(
        n_estimators=n_estimators, random_state=seed, n_jobs=-1
    )
    rf.fit(X_train, y_train)

    y_pred = rf.predict(X_test)
    bal_acc = float(balanced_accuracy_score(y_test, y_pred))

    class_labels: List[Any] = rf.classes_.tolist()
    cm = confusion_matrix(y_test, y_pred, labels=rf.classes_).astype(np.float64)
    row_sums = cm.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums > 0, row_sums, 1.0)
    cm_norm = cm / row_sums

    if return_importances:
        importances = np.asarray(rf.feature_importances_, dtype=np.float64)
    else:
        importances = np.empty(0, dtype=np.float64)

    return {
        "balanced_acc_test": bal_acc,
        "confusion_test": cm_norm,
        "feature_importances": importances,
        "n_train": int(X_train.shape[0]),
        "n_test": int(X_test.shape[0]),
        "class_labels": class_labels,
    }

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
    Train a Random Forest classifier for a single cluster using GridSearchCV.
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
        
        from sklearn.model_selection import StratifiedKFold
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

        param_dist = {
            'n_estimators':      [100],
            'max_depth':         [25],
            'min_samples_split': [100, 200],
            'min_samples_leaf':  [50, 100],
            'max_features':      ['sqrt', 0.1]
        }

        search = GridSearchCV(
            RandomForestClassifier(random_state=42),
            param_grid=param_dist,
            scoring='f1_weighted',
            cv=cv,
            n_jobs=4
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
        
        # Calculate Accuracy and Confusion Matrix on the hold-out test set
        y_pred = search.best_estimator_.predict(X_test)
        test_acc = accuracy_score(y_test, y_pred)
        
        # Plot and log confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['Class 1', 'Class 2', 'Class 3', 'Class 4'],
                    yticklabels=['Class 1', 'Class 2', 'Class 3', 'Class 4'])
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title(f'Confusion Matrix - Cluster {cluster_id}')
        plt.tight_layout()
        cm_path = f"confusion_matrix_cluster_{cluster_id}.png"
        plt.savefig(cm_path)
        plt.close()
        
        # Log to MLflow
        mlflow.log_metric("test_accuracy", test_acc)
        mlflow.log_artifact(cm_path)
        
        # Log the scikit-learn model with signature
        from mlflow.models.signature import infer_signature
        signature = infer_signature(X_train[:5], search.predict(X_train[:5]))
        mlflow.sklearn.log_model(
            search.best_estimator_, 
            f"model_cluster_{cluster_id}",
            signature=signature,
            input_example=X_train[:5]
        )
        
        import os
        if os.path.exists(cm_path): os.remove(cm_path)
        
        return search.best_estimator_, search.best_params_, best_score, test_score, test_acc
