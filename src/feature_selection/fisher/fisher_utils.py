def _fisher_score_(X_los):
    """
    Computes Fisher coefficient per feature for one LoS (n_classes x n_features)
    X_los: array of shape (n_classes, n_features)
    Returns: 1D array of Fisher coefficients (length = n_features)
    """
    mean_total = np.mean(X_los, axis=0)
    n_classes = X_los.shape[0]

    # Between-class scatter (variance of class means)
    mean_per_class = X_los  # since each row = class mean already (one sample per class)
    S_b = np.var(mean_per_class, axis=0)  

    # Within-class scatter (assume 1 sample per class → small constant to avoid division by zero)
    S_w = np.ones_like(S_b) * 1e-6

    F = S_b / S_w
    return F

def fisher_scores(X_LoS_generator):
    fisher_scores_list = []
    for los_idx in range(X_LoS_generator.shape[0]):
        los = X_LoS_generator[los_idx]     # shape: (n_classes, n_features)
        F = _fisher_score_(los)
        fisher_scores_list.append(F)

    fisher_scores_matrix = np.vstack(fisher_scores_list)  # shape: (n_samples_per_class, n_features)
    return fisher_scores_matrix

def save_feature_analysis(score_matrix, out_base):
    """
    Saves processed data by class
    """
    save_path = os.path.join(out_base, fisher_scores.npy)
    os.makedirs(save_path, exist_ok=True)
    np.save(save_path, score_matrix)
    print(f"Saved {len(score_matrix)} rows to {save_path}")
    return save_path



