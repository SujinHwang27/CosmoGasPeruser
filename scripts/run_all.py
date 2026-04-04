"""
Signal Clustering Analysis - End-to-End Pipeline Orchestrator
Executes the 6 stages defined in docs/feature/signal-clustering-analysis/signal_clustering_project_plan_v4.md

Stages:
1. Input Preparation (run_prep.py)
2. Micro-Probing (run_probe.py)
3. Clustering (run_cluster.py)
4. Visualization (run_viz.py)
5. Auditing (run_audit.py)
6. Random Forest Classification (run_rf.py)
"""

import argparse
import subprocess
import sys
import os
import time
from pathlib import Path

def run_command(command, stage_name):
    """Executes a shell command and handles errors."""
    print(f"\n" + "="*60)
    print(f"STARTING STAGE: {stage_name}")
    print(f"COMMAND: {' '.join(command)}")
    print("="*60 + "\n")
    
    start_time = time.time()
    try:
        # Use uv run to ensure the environment is correct
        full_command = ["uv", "run"] + command
        
        # Add current directory to PYTHONPATH so 'src' is found in sub-processes
        env = os.environ.copy()
        current_pythonpath = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = f".:{current_pythonpath}" if current_pythonpath else "."
        
        result = subprocess.run(full_command, check=True, env=env)
        elapsed = time.time() - start_time
        print(f"\n" + "-"*60)
        print(f"COMPLETED STAGE: {stage_name} (Time: {elapsed:.2f}s)")
        print("-"*60 + "\n")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n" + "!"*60)
        print(f"ERROR IN STAGE: {stage_name}")
        print(f"Command failed with return code {e.returncode}")
        print("!"*60 + "\n")
        return False

def main():
    parser = argparse.ArgumentParser(description="Orchestrator for Signal Clustering Analysis Pipeline")
    parser.add_argument("--stages", type=str, default="1,2,3,4,5,6",
                        help="Comma-separated list of stages to run (e.g., '1,2,3') or 'all'")
    parser.add_argument("--run", type=str, choices=['wavelet', 'raw', 'both'], default='both',
                        help="Which data representation to use (Stage 2, 3, 4, 5)")
    parser.add_argument("--k", type=int, default=5,
                        help="Number of clusters (Stage 3, 4, 5)")
    parser.add_argument("--n_jobs", type=int, default=-1,
                        help="Number of parallel jobs for Stage 2")
    parser.add_argument("--view", type=str, choices=['A', 'B', 'C', 'all'], default='all',
                        help="Which visualization view to generate (Stage 4)")
    
    args = parser.parse_args()
    
    if args.stages == 'all':
        stages_to_run = [1, 2, 3, 4, 5, 6]
    else:
        stages_to_run = [int(s.strip()) for s in args.stages.split(',')]
        
    print("="*60)
    print("SIGNAL CLUSTERING PIPELINE ORCHESTRATOR")
    print(f"Stages to execute: {stages_to_run}")
    print(f"Run mode: {args.run}")
    print(f"K value: {args.k}")
    print("="*60)

    # Stage 1: Input Preparation
    if 1 in stages_to_run:
        cmd = ["python", "scripts/run_prep.py"]
        if not run_command(cmd, "Stage 1: Input Preparation"):
            sys.exit(1)

    # Stage 2: Micro-Probing
    if 2 in stages_to_run:
        cmd = ["python", "scripts/run_probe.py", "--run", args.run, "--n_jobs", str(args.n_jobs)]
        if not run_command(cmd, "Stage 2: Micro-Probing"):
            sys.exit(1)

    # Stage 3: Clustering
    if 3 in stages_to_run:
        cmd = ["python", "scripts/run_cluster.py", "--run", args.run, "--k", str(args.k)]
        if not run_command(cmd, "Stage 3: Clustering"):
            sys.exit(1)

    # Stage 4: Visualization
    if 4 in stages_to_run:
        cmd = ["python", "scripts/run_viz.py", "--run", args.run, "--view", args.view, "--k", str(args.k)]
        if not run_command(cmd, "Stage 4: Visualization"):
            sys.exit(1)

    # Stage 5: Auditing
    if 5 in stages_to_run:
        cmd = ["python", "scripts/run_audit.py", "--run", args.run, "--k", str(args.k)]
        if not run_command(cmd, "Stage 5: Auditing"):
            sys.exit(1)

    # Stage 6: Random Forest Classification
    if 6 in stages_to_run:
        cmd = ["python", "scripts/run_rf.py", "--run", args.run, "--k", str(args.k)]
        if not run_command(cmd, "Stage 6: Random Forest Classification"):
            sys.exit(1)

    print("\n" + "#"*60)
    print("FULL PIPELINE EXECUTION COMPLETED SUCCESSFULLY")
    print("#"*60 + "\n")

if __name__ == "__main__":
    main()
