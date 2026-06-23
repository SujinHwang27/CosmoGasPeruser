"""Thin wrapper: run the cluster-environment-gradient probe.

No logic here — a single call into :func:`src.core.cluster_env_probe.run_env_gradient_probe`.
Run from the repo root:

    PYTHONPATH=. uv run python scripts/run_cluster_env_gradient.py
"""

import argparse
import json
from pathlib import Path

from src.core.cluster_env_probe import run_env_gradient_probe


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the cluster-environment-gradient probe (PI-scoped; see SCOPING.md)."
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("results/cluster_environment_gradient"),
        help="Output directory (default: results/cluster_environment_gradient).",
    )
    args = parser.parse_args()

    result = run_env_gradient_probe(args.out_dir)
    print(json.dumps({
        "verdict": result["verdict"],
        "wavelet_arm_gradient": result["wavelet_arm"]["gradient"],
        "raw_summary_arm_gradient": result["raw_summary_arm"]["gradient"],
        "cross_representation": result["cross_representation"],
    }, indent=2))
    print("\nVERDICT:", result["verdict"])
    print(result["reading"])


if __name__ == "__main__":
    main()
