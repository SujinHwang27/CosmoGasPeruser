import mlflow
import matplotlib.pyplot as plt
import numpy as np
import os

def clean_and_summarize(experiment_name="Baseline_RF"):
    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    if not experiment:
        print(f"Experiment '{experiment_name}' not found.")
        return

    experiment_id = experiment.experiment_id
    runs = client.search_runs(experiment_id)

    print(f"Total runs found: {len(runs)}")

    # 1. Delete unsuccessful or debug (subset) runs
    for run in runs:
        # Delete non-finished runs
        if run.info.status != "FINISHED":
            print(f"Deleting non-finished run: {run.info.run_name} (ID: {run.info.run_id})")
            client.delete_run(run.info.run_id)
            continue
            
        # Delete runs missing key metrics
        if "mean_accuracy" not in run.data.metrics:
            print(f"Deleting metadata-only run: {run.info.run_name} (ID: {run.info.run_id})")
            client.delete_run(run.info.run_id)
            continue
        
        # Specific cleanup for the "debug" subset 100 runs we did earlier
        # Raw Spectra on the full set is ~0.45, so 0.2 is clearly a tiny subset test.
        if run.info.run_name == "RF - Raw Spectra" and run.data.metrics['mean_accuracy'] < 0.25:
             print(f"Deleting subset/debug run: {run.info.run_name} (ID: {run.info.run_id}, Acc: {run.data.metrics['mean_accuracy']})")
             client.delete_run(run.info.run_id)

    # 2. Re-fetch final set of runs
    runs = client.search_runs(experiment_id)
    raw_data = {}

    for run in runs:
        if run.info.status == "FINISHED":
            name = run.info.run_name
            acc = run.data.metrics['mean_accuracy']
            std = run.data.metrics.get('std_accuracy', 0.0)
            
            # Keep the best run for each name
            if name not in raw_data or acc > raw_data[name]['acc']:
                raw_data[name] = {'acc': acc, 'std': std}

    # Custom sort order: A6 -> D6-1 -> Concatenated -> Raw (Top to Bottom)
    # Since barh plots bottom-up, we reverse the list for plotting
    order = [
        "RF - Raw Spectra",
        "RF - Concatenated Wavelet",
        "RF - Wavelet D1",
        "RF - Wavelet D2",
        "RF - Wavelet D3",
        "RF - Wavelet D4",
        "RF - Wavelet D5",
        "RF - Wavelet D6",
        "RF - Wavelet A6"
    ]
    
    # Filter to only include runs we actually have
    plot_data = []
    for name in order:
        if name in raw_data:
            plot_data.append({'name': name, 'acc': raw_data[name]['acc'], 'std': raw_data[name]['std']})
    
    # Addition: Catch any runs not in our explicit 'order' list and put them at the end
    for name, v in raw_data.items():
        if name not in order:
            plot_data.insert(0, {'name': name, 'acc': v['acc'], 'std': v['std']})

    if not plot_data:
        print("No successful runs found to plot.")
        return

    # 3. Generate Global Summary Plot
    plt.figure(figsize=(12, 8))
    names = [d['name'] for d in plot_data]
    accs = [d['acc'] for d in plot_data]
    stds = [d['std'] for d in plot_data]

    plt.barh(names, accs, xerr=stds, capsize=5, color='skyblue')
    plt.axvline(0.25, color='red', linestyle='--', label='Chance (0.25)')
    plt.xlim(0, 1.0)
    plt.xlabel("Accuracy")
    plt.title(f"Comprehensive {experiment_name} Performance Comparison")
    plt.legend()
    plt.tight_layout()

    os.makedirs("results/baseline_rf", exist_ok=True)
    plot_path = "results/baseline_rf/baseline_summary.png"
    plt.savefig(plot_path)
    print(f"Global summary plot updated at: {plot_path}")

if __name__ == "__main__":
    clean_and_summarize()
