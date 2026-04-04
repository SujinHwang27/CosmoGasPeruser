"""
Provenance tracking — stamps pipeline outputs with git and environment metadata.

Ensures every result CSV/artifact can be traced back to the exact code state,
branch, and parameters that produced it.
"""

import subprocess
import datetime
from typing import Dict, Any


def get_git_info() -> Dict[str, str]:
    """
    Capture current git state for provenance tracking.

    Returns:
        Dict with keys: commit, branch, dirty, timestamp
    """
    def _run(cmd: list[str]) -> str:
        try:
            return subprocess.check_output(cmd, stderr=subprocess.DEVNULL).decode().strip()
        except (subprocess.CalledProcessError, FileNotFoundError):
            return "unknown"

    commit = _run(["git", "rev-parse", "--short", "HEAD"])
    branch = _run(["git", "rev-parse", "--abbrev-ref", "HEAD"])
    dirty = _run(["git", "status", "--porcelain"]) != ""

    return {
        "commit": commit,
        "branch": branch,
        "dirty": str(dirty),
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
    }


def provenance_header(params: Dict[str, Any] = None) -> Dict[str, str]:
    """
    Build a provenance dict suitable for saving alongside results.

    Args:
        params: Optional experiment parameters (k, run_type, seed, etc.)

    Returns:
        Dict combining git info + experiment params
    """
    info = get_git_info()
    if params:
        info.update({f"param_{k}": str(v) for k, v in params.items()})
    return info


def mlflow_experiment_name(base_name: str = "CosmoGasPeruser") -> str:
    """
    Generate a branch-aware MLflow experiment name.

    Examples:
        "CosmoGasPeruser/feature/signal-clustering-v2"
        "CosmoGasPeruser/main"

    Args:
        base_name: Project prefix

    Returns:
        Experiment name string
    """
    git_info = get_git_info()
    branch = git_info["branch"]
    return f"{base_name}/{branch}"
