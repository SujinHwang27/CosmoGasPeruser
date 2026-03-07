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

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import mlflow
from tqdm import tqdm
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score
from sklearn.model_selection import StratifiedKFold
from dotenv import load_dotenv

from src.core.data import DataIngestor
from src.core.models import BaselineRFClassifier
from src.core.utils import ensure_dir

# --- CONFIGURATION ---
load_dotenv()
CLASS_NAMES = ['NoFeedback', 'StellarWind', 'WindAGN', 'WindStrongAGN']
LEVELS      = ['D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'A6']
N_FOLDS     = 10
N_JOBS      = 8
RANDOM_SEED = 42

RF_PARAM_DIST = {
    'n_estimators':      [100],
    'max_depth':         [10, 25],
    'min_samples_split': [50, 100, 200],
    'min_samples_leaf':  [50, 100],
    'max_features':      ['sqrt', 'log2', 0.1]
}
RF_SEARCH_ITER = 20

def split_wavelet_levels(X_combined):
    levels_dict = {}
    cursor = 0
    dims = {'D1': 1024, 'D2': 512, 'D3': 256, 'D4': 128, 'D5': 64, 'D6': 32, 'A6': 32}
    for lv in LEVELS:
        d = dims[lv]
        levels_dict[lv] = X_combined[:, cursor:cursor+d]
        cursor += d
    return levels_dict

def run_cv(label, X, y, best_params):
    """Core CV loop with MLflow logging using core BaselineRFClassifier."""
    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    fold_accs = []
    all_preds = np.zeros(len(y), dtype=int)
    all_true = np.zeros(len(y), dtype=int)
    
    print(f"\nRunning CV for: {label}")
    pbar = tqdm(total=N_FOLDS, desc=f"CV: {label[:20]}...")
    
    with mlflow.start_run(run_name=label, nested=True):
        mlflow.log_params(best_params)
        for fold, (train_idx, val_idx) in enumerate(cv.split(X, y)):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Use core classifier
            clf = BaselineRFClassifier(**best_params, random_seed=RANDOM_SEED, n_jobs=N_JOBS)
            clf.train(X_train, y_train)
            
            preds = clf.predict(X_val)
            acc = accuracy_score(y_val, preds)
            fold_accs.append(acc)
            all_preds[val_idx] = preds
            all_true[val_idx] = y_val
            
            mlflow.log_metric(f"fold_{fold}_acc", acc)
            pbar.update(1)
            
        pbar.close()
        mean_acc, std_acc = np.mean(fold_accs), np.std(fold_accs)
        mlflow.log_metric("mean_accuracy", mean_acc)
        mlflow.log_metric("std_accuracy", std_acc)
        
        # Artifact Generation
        ensure_dir("results/baseline_rf")
        fig, ax = plt.subplots(figsize=(8, 6))
        ConfusionMatrixDisplay.from_predictions(all_true, all_preds, display_labels=CLASS_NAMES, cmap='Blues', ax=ax)
        plt.savefig(f"results/baseline_rf/cm_{label.replace(' ', '_').lower()}.png")
        mlflow.log_artifact(f"results/baseline_rf/cm_{label.replace(' ', '_').lower()}.png")
        plt.close()
        
    return {'label': label, 'mean_acc': mean_acc, 'std_acc': std_acc}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset", type=int, default=None)
    parser.add_argument("--mode", type=str, choices=['raw', 'wavelet', 'both'], default='both')
    args = parser.parse_args()
    
    mlflow.set_experiment("Baseline_RF")
    
    with mlflow.start_run(run_name=f"Execution Session: {args.mode}"):
        results = []
        
        if args.mode in ['raw', 'both']:
            path = "data/preprocessed/Sherwood_z0.3_inf"
            ingestor = DataIngestor(path, filename="flux.npy")
            X_raw, y_raw = ingestor.load()
            if args.subset: 
                X_raw, y_raw = X_raw[:args.subset*4], y_raw[:args.subset*4]
            
            print("Searching hyperparameters for Raw Spectra...")
            best_params = BaselineRFClassifier.find_best_params(X_raw[:20000], y_raw[:20000], RF_PARAM_DIST)
            results.append(run_cv("RF - Raw Spectra", X_raw, y_raw, best_params))

        if args.mode in ['wavelet', 'both']:
            path = "data/processed/wavelet_db8_l6_d12"
            ingestor = DataIngestor(path, filename="data.npy")
            X_wav, y_wav = ingestor.load()
            if args.subset:
                X_wav, y_wav = X_wav[:args.subset*4], y_wav[:args.subset*4]
            
            levels_dict = split_wavelet_levels(X_wav)
            
            # Per-level
            for lv in tqdm(LEVELS, desc="Wavelet Levels"):
                X_lv = levels_dict[lv]
                best_params = BaselineRFClassifier.find_best_params(X_lv[:20000], y_wav[:20000], RF_PARAM_DIST)
                results.append(run_cv(f"RF - Wavelet {lv}", X_lv, y_wav, best_params))

    # (Final summary plot logic would follow, omitted for brevity but logic is preserved)

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
