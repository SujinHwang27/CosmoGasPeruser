"""
Baseline Random Forest Classifier — CosmoGasPeruser
===================================================

This script implements a baseline Random Forest classifier to classify 
synthetic quasar absorption spectra into 4 physical feedback modes.

Two Modes of Operation:
1. Mode 1: Raw Spectra Signal (Flux)
2. Mode 2: Wavelet Coefficients (Per-level and Concatenated)

Integrated with MLflow for experiment tracking.
"""

from dotenv import load_dotenv

# --- CONFIGURATION ---
load_dotenv()
CLASS_NAMES = ['NoFeedback', 'StellarWind', 'WindAGN', 'WindStrongAGN']
CLASS_DIRS  = ['1', '2', '3', '4']
LEVELS      = ['D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'A6']  # All levels included
N_CLASSES   = 4
N_FOLDS     = 10
N_JOBS      = 8  # Reduced to avoid memory issues
RANDOM_SEED = 42

RF_PARAM_DIST = {
    'n_estimators':      [100],
    'max_depth':         [10, 25],
    'min_samples_split': [50, 100, 200],
    'min_samples_leaf':  [50, 100],
    'max_features':      ['sqrt', 'log2', 0.1]
}
RF_SEARCH_ITER = 20

# --- DATA LOADING ---

def load_data(data_root="data", mode="wavelet", subset=None):
    """
    Loads data for all 4 classes.
    - wavelet: data/processed/wavelet_db8_l6_d12/[1-4]/data.npy (Shape: (16384, 2048))
    - raw: data/preprocessed/Sherwood_z0.3_inf/[1-4]/flux.npy (Shape: (16384, 2048))
    """
    X_list = []
    y_list = []
    
    print(f"Loading {mode} data...")
    pbar = tqdm(CLASS_DIRS, desc="Classes")
    for i, class_dir in enumerate(pbar):
        pbar.set_postfix(cls=CLASS_NAMES[i])
        if mode == "wavelet":
            path = os.path.join(data_root, "processed", "wavelet_db8_l6_d12", class_dir, "data.npy")
        else:
            path = os.path.join(data_root, "preprocessed", "Sherwood_z0.3_inf", class_dir, "flux.npy")
            
        if not os.path.exists(path):
            raise FileNotFoundError(f"Data not found at {path}")
            
        data = np.load(path)
        if subset:
            data = data[:subset]
            
        X_list.append(data)
        y_list.append(np.full(data.shape[0], i))
        
        print(f"  Class {CLASS_NAMES[i]}: {data.shape}")
        
    X = np.concatenate(X_list, axis=0)
    y = np.concatenate(y_list, axis=0)
    
    return X, y

def split_wavelet_levels(X_combined):
    """
    Splits the 2048-dim concatenated wavelet vector back into levels.
    D1: 1024, D2: 512, D3: 256, D4: 128, D5: 64, D6: 32, A6: 32 (Total 2048)
    """
    levels_dict = {}
    cursor = 0
    dims = {'D1': 1024, 'D2': 512, 'D3': 256, 'D4': 128, 'D5': 64, 'D6': 32, 'A6': 32}
    
    for lv in LEVELS:
        d = dims[lv]
        levels_dict[lv] = X_combined[:, cursor:cursor+d]
        cursor += d
        
    return levels_dict

# --- UTILITIES ---

def scale_fold(X_train_raw, X_val_raw):
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train_raw.astype(np.float64)).astype(np.float32)
    X_val_sc = scaler.transform(X_val_raw.astype(np.float64)).astype(np.float32)
    return X_train_sc, X_val_sc

def find_best_params(X_train, y_train):
    search = RandomizedSearchCV(
        RandomForestClassifier(random_state=RANDOM_SEED, n_jobs=N_JOBS, class_weight='balanced'),
        param_distributions=RF_PARAM_DIST,
        n_iter=RF_SEARCH_ITER,
        cv=3,
        scoring='accuracy',
        random_state=RANDOM_SEED,
        n_jobs=N_JOBS,
        verbose=1
    )
    search.fit(X_train, y_train)
    return search.best_params_

def run_cv(label, X, y, best_params, experiment_name=None):
    """Core CV loop with MLflow logging."""
    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    fold_accs = []
    all_preds = np.zeros(len(y), dtype=int)
    all_true = np.zeros(len(y), dtype=int)
    
    print(f"\nRunning CV for: {label}")
    
    pbar = tqdm(total=N_FOLDS, desc=f"CV: {label[:20]}...")
    with mlflow.start_run(run_name=label, nested=True):
        mlflow.log_params(best_params)
        mlflow.log_param("input_dim", X.shape[1])
        
        for fold, (train_idx, val_idx) in enumerate(cv.split(X, y)):
            X_train_sc, X_val_sc = scale_fold(X[train_idx], X[val_idx])
            y_train, y_val = y[train_idx], y[val_idx]
            
            rf = RandomForestClassifier(**best_params, random_state=RANDOM_SEED, 
                                        n_jobs=N_JOBS, class_weight='balanced')
            rf.fit(X_train_sc, y_train)
            
            preds = rf.predict(X_val_sc)
            acc = accuracy_score(y_val, preds)
            fold_accs.append(acc)
            all_preds[val_idx] = preds
            all_true[val_idx] = y_val
            
            mlflow.log_metric(f"fold_{fold}_acc", acc)
            pbar.set_postfix(acc=f"{acc:.4f}")
            pbar.update(1)
            
        pbar.close()
        mean_acc = np.mean(fold_accs)
        std_acc = np.std(fold_accs)
        
        mlflow.log_metric("mean_accuracy", mean_acc)
        mlflow.log_metric("std_accuracy", std_acc)
        
        print(f"Result: {mean_acc:.4f} ± {std_acc:.4f}")
        
        # Log confusion matrix plot
        fig, ax = plt.subplots(figsize=(8, 6))
        ConfusionMatrixDisplay.from_predictions(all_true, all_preds, display_labels=CLASS_NAMES, 
                                              cmap='Blues', ax=ax, xticks_rotation=45)
        ax.set_title(f"CM: {label} (Acc: {mean_acc:.3f})")
        plt.tight_layout()
        plot_path = f"results/baseline_rf/cm_{label.replace(' ', '_').lower()}.png"
        os.makedirs("results/baseline_rf", exist_ok=True)
        plt.savefig(plot_path)
        mlflow.log_artifact(plot_path)
        plt.close()
        
    return {
        'label': label,
        'mean_acc': mean_acc,
        'std_acc': std_acc,
        'fold_accs': fold_accs
    }

# --- MAIN ---

def main():
    parser = argparse.ArgumentParser(description="Train Baseline Random Forest Experiment")
    parser.add_argument("--subset", type=int, default=None, help="Use a subset of data for testing")
    parser.add_argument("--mode", type=str, choices=['raw', 'wavelet', 'both'], default='both', help="Experiment mode")
    parser.add_argument("--levels", type=str, nargs='*', default=None, help="Specific wavelet levels to run (e.g. D3 D4)")
    args = parser.parse_args()
    
    mlflow.set_experiment("Baseline_RF")
    
    # Start a Parent Run to group all sub-experiments
    with mlflow.start_run(run_name=f"Execution Session: {args.mode}"):
        results = []
        
        # --- Mode 1: Raw Spectra ---
        if args.mode in ['raw', 'both']:
            try:
                X_raw, y_raw = load_data(mode="raw", subset=args.subset)
                
                # Find params on a subset of the first fold for speed
                cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)
                tr_idx, _ = next(iter(cv.split(X_raw, y_raw)))
                X_tr_sc, _ = scale_fold(X_raw[tr_idx], X_raw[tr_idx])
                
                print("Searching hyperparameters for Raw Spectra...")
                best_params_raw = find_best_params(X_tr_sc[:20000] if len(X_tr_sc) > 20000 else X_tr_sc, 
                                                y_raw[tr_idx][:20000] if len(X_tr_sc) > 20000 else y_raw[tr_idx])
                
                res_raw = run_cv("RF - Raw Spectra", X_raw, y_raw, best_params_raw)
                results.append(res_raw)
            except FileNotFoundError as e:
                print(f"Skipping Mode 1: {e}")

        # --- Mode 2: Wavelet ---
        if args.mode in ['wavelet', 'both']:
            X_wav_cat, y_wav = load_data(mode="wavelet", subset=args.subset)
            levels_dict = split_wavelet_levels(X_wav_cat)
            
            # 2a: Per-Level
            print("\nStarting Per-Level Wavelet Experiments...")
            target_levels = args.levels if args.levels else LEVELS
            for lv in tqdm(target_levels, desc="Wavelet Levels"):
                X_lv = levels_dict[lv]
                cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)
                tr_idx, _ = next(iter(cv.split(X_lv, y_wav)))
                X_tr_sc, _ = scale_fold(X_lv[tr_idx], X_lv[tr_idx])
                
                print(f"Searching hyperparameters for Wavelet {lv}...")
                best_params_lv = find_best_params(X_tr_sc[:20000] if len(X_tr_sc) > 20000 else X_tr_sc,
                                                y_wav[tr_idx][:20000] if len(X_tr_sc) > 20000 else y_wav[tr_idx])
                
                res_lv = run_cv(f"RF - Wavelet {lv}", X_lv, y_wav, best_params_lv)
                results.append(res_lv)
                
            # 2b: Concatenated
            cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)
            tr_idx, _ = next(iter(cv.split(X_wav_cat, y_wav)))
            X_tr_sc, _ = scale_fold(X_wav_cat[tr_idx], X_wav_cat[tr_idx])
            
            print("Searching hyperparameters for Concatenated Wavelet...")
            best_params_cat = find_best_params(X_tr_sc[:20000] if len(X_tr_sc) > 20000 else X_tr_sc,
                                            y_wav[tr_idx][:20000] if len(X_tr_sc) > 20000 else y_wav[tr_idx])
            
            res_cat = run_cv("RF - Concatenated Wavelet", X_wav_cat, y_wav, best_params_cat)
            results.append(res_cat)

        # Summary Plot
        if results:
            plt.figure(figsize=(12, 6))
            labels = [r['label'] for r in results]
            accs = [r['mean_acc'] for r in results]
            stds = [r['std_acc'] for r in results]
            
            plt.barh(labels, accs, xerr=stds, capsize=5, color='skyblue')
            plt.axvline(0.25, color='red', linestyle='--', label='Chance (0.25)')
            plt.xlim(0, 1.0)
            plt.xlabel("Accuracy")
            plt.title("Baseline Random Forest Performance Comparison")
            plt.legend()
            plt.tight_layout()
            summary_path = "results/baseline_rf/baseline_summary.png"
            os.makedirs("results/baseline_rf", exist_ok=True)
            plt.savefig(summary_path)
            
            # Log the summary plot to the PARENT run
            mlflow.log_artifact(summary_path)
            print(f"\nSummary plot saved to {summary_path} and logged to MLflow.")
        
if __name__ == "__main__":
    main()
