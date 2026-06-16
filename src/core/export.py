"""
Curated outbound data exports to external consumers.

Scientific purpose: this module is the single, audited boundary through which
CosmoGasPeruser data leaves the project for consumption by external systems
(e.g. the selements-website primer panels). Every export here carries a
no-bit-lost integrity guarantee: data is loaded through the canonical loaders
(never raw ``np.load`` in scripts), validated for shape / finiteness / physical
bounds, written at full float64 precision (no silent rounding), and shipped
alongside a git-stamped provenance sidecar so the exact source array, producing
function, and code state are recoverable.

This is the [D-31] (selements-website data-export contract) implementation. The
data-engineer agent is the primary owner of this workflow.
"""

import csv
import datetime
import json
from pathlib import Path
from typing import Dict, List

import numpy as np

from src.core.data import SignalClusteringData, load_velocity_axis
from src.core.provenance import get_git_info

# Canonical landing root for selements-website exports. All website-bound
# artifacts live under here so the consumer has one stable mount point.
SELEMENTS_WEBSITE_ROOT = Path("results/exports/selements-website")

# Class-id -> human label, mirroring the project-wide physics-class convention
# (1=NoFeedback, 2=StellarWind, 3=WindAGN, 4=WindStrongAGN).
_CLASS_LABELS: Dict[int, str] = {
    1: "NoFeedback",
    2: "StellarWind",
    3: "WindAGN",
    4: "WindStrongAGN",
}

# Canonical sightline length for the Sherwood z=0.3 snapshot.
_N_PIXELS = 2048


def _validate_spectrum(flux: np.ndarray, name: str = "flux") -> None:
    """Validate a single-sightline flux vector before export.

    Enforces the project's domain conventions for transmitted flux:
    canonical shape ``(2048,)``, all-finite (no silent NaN/Inf passthrough),
    and physical bounds ``flux in [0, 1]``.

    Args:
        flux: Candidate single-sightline transmitted-flux array.
        name: Human label used in error messages.

    Raises:
        ValueError: if the shape is not ``(2048,)``, any value is non-finite,
            or any value falls outside ``[0, 1]``.
    """
    if flux.shape != (_N_PIXELS,):
        raise ValueError(
            f"{name} has shape {flux.shape}, expected ({_N_PIXELS},)"
        )
    if not np.all(np.isfinite(flux)):
        raise ValueError(f"{name} contains non-finite values (NaN/Inf)")
    fmin = float(np.min(flux))
    fmax = float(np.max(flux))
    if fmin < 0.0 or fmax > 1.0:
        raise ValueError(
            f"{name} out of physical range: min={fmin}, max={fmax}, "
            "expected transmitted flux in [0, 1]"
        )


def _load_wavelength_axis(
    class_id: int,
    flux_base: str = SignalClusteringData.FLUX_BASE,
) -> np.ndarray:
    """Load the per-pixel wavelength axis ``wave.npy`` (Angstrom) for one class.

    Args:
        class_id: Physics class id in ``{1, 2, 3, 4}``.
        flux_base: Base directory under which ``{class_id}/wave.npy`` lives.

    Returns:
        ``(2048,) float64`` wavelength-axis array in Angstrom.

    Raises:
        FileNotFoundError: if ``wave.npy`` is absent for the requested class.
        ValueError: if the loaded axis is not 1D length 2048, or non-finite.
    """
    if class_id not in (1, 2, 3, 4):
        raise ValueError(f"class_id must be in {{1,2,3,4}}, got {class_id}")
    path = Path(flux_base) / str(class_id) / "wave.npy"
    if not path.exists():
        raise FileNotFoundError(f"wave.npy not found at {path}")
    wave = np.load(path).astype(np.float64, copy=False)
    if wave.ndim != 1 or wave.shape[0] != _N_PIXELS:
        raise ValueError(
            f"wave.npy for class {class_id} has shape {wave.shape}, "
            f"expected ({_N_PIXELS},)"
        )
    if not np.all(np.isfinite(wave)):
        raise ValueError(
            f"wave.npy for class {class_id} contains non-finite values"
        )
    return wave


def write_spectrum_csv(
    csv_path: Path,
    pixel: np.ndarray,
    wavelength_angstrom: np.ndarray,
    velocity_kms: np.ndarray,
    flux: np.ndarray,
) -> Path:
    """Write a clean 4-column spectrum CSV (header + data only, no comments).

    The website parser is comment-unaware, so this writer emits ONLY a header
    row followed by data rows. Floats are written via ``repr`` (full float64
    precision) so no information is lost on round-trip. Validates the flux
    vector and array-length consistency before writing.

    Columns: ``pixel,wavelength_angstrom,velocity_kms,flux``.

    Args:
        csv_path: Destination CSV path (parent dirs created if absent).
        pixel: Integer pixel indices, shape ``(2048,)``.
        wavelength_angstrom: Wavelength axis (Angstrom), shape ``(2048,)``.
        velocity_kms: Velocity axis (km/s), shape ``(2048,)``.
        flux: Transmitted flux, shape ``(2048,)``, validated to ``[0, 1]``.

    Returns:
        The ``csv_path`` written.

    Raises:
        ValueError: if any input array length differs from ``(2048,)`` or the
            flux fails :func:`_validate_spectrum`.
    """
    _validate_spectrum(np.asarray(flux, dtype=np.float64), name="flux")
    for arr, nm in (
        (pixel, "pixel"),
        (wavelength_angstrom, "wavelength_angstrom"),
        (velocity_kms, "velocity_kms"),
    ):
        if np.asarray(arr).shape != (_N_PIXELS,):
            raise ValueError(
                f"{nm} has shape {np.asarray(arr).shape}, expected ({_N_PIXELS},)"
            )

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    header: List[str] = ["pixel", "wavelength_angstrom", "velocity_kms", "flux"]
    with open(csv_path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        for i in range(_N_PIXELS):
            writer.writerow([
                int(pixel[i]),
                repr(float(wavelength_angstrom[i])),
                repr(float(velocity_kms[i])),
                repr(float(flux[i])),
            ])
    return csv_path


def export_primer_synthetic_spectrum(
    out_dir: Path,
    class_id: int = 1,
    sightline_idx: int = 0,
) -> Path:
    """Export the primer single-sightline synthetic spectrum for selements-website.

    Loads the requested class flux through the canonical loader
    (:meth:`SignalClusteringData.load_flux_per_class`), pulls one sightline,
    validates it (shape ``(2048,)``, finite, ``in [0, 1]``), joins it with the
    pixel / wavelength (``wave.npy``) / velocity (``vel.npy``) axes, and writes a
    clean CSV plus a git-stamped provenance JSON sidecar.

    Determinism: re-running with the same ``(class_id, sightline_idx)`` against
    the same source data produces byte-identical CSV content (the provenance
    JSON differs only in its ``export_timestamp`` and git state).

    Args:
        out_dir: Landing directory for the two output files (created if absent).
        class_id: Physics class id in ``{1, 2, 3, 4}``. Default 1 (NoFeedback).
        sightline_idx: Row index into the per-class flux block. Default 0.

    Returns:
        Path to the written CSV (``synthetic-spectrum.example.csv``).

    Raises:
        ValueError: on bad ``class_id``, out-of-range ``sightline_idx``, or a
            flux vector that violates shape/finiteness/range.
    """
    if class_id not in _CLASS_LABELS:
        raise ValueError(f"class_id must be in {sorted(_CLASS_LABELS)}, got {class_id}")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- load via canonical loader (never raw np.load in scripts) ---
    loader = SignalClusteringData()
    flux_per_class, _ = loader.load_flux_per_class()
    class_block = flux_per_class[class_id - 1]  # mmap'd (N, 2048)
    if not (0 <= sightline_idx < class_block.shape[0]):
        raise ValueError(
            f"sightline_idx {sightline_idx} out of range "
            f"[0, {class_block.shape[0]}) for class {class_id}"
        )
    flux = np.asarray(class_block[sightline_idx], dtype=np.float64)
    _validate_spectrum(flux, name=f"class{class_id}.sightline{sightline_idx}.flux")

    # --- axes ---
    velocity = load_velocity_axis(class_id)  # (2048,) km/s
    wavelength = _load_wavelength_axis(class_id)  # (2048,) Angstrom
    pixel = np.arange(_N_PIXELS, dtype=np.int64)

    # Δv from the uniform velocity spacing (derived from disk, not hardcoded).
    delta_v = float(np.diff(velocity).mean())

    # --- write CSV ---
    csv_path = out_dir / "synthetic-spectrum.example.csv"
    write_spectrum_csv(csv_path, pixel, wavelength, velocity, flux)

    # --- write provenance sidecar ---
    git_info = get_git_info()
    source_path = str(
        Path(SignalClusteringData.FLUX_BASE) / str(class_id) / "flux.npy"
    )
    provenance = {
        "export_request_slug": "primer-synthetic-spectrum",
        "consumer": "selements-website",
        "producing_function": "src.core.export.export_primer_synthetic_spectrum",
        "consumer_facing_filename": csv_path.name,
        "export_timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        "git": git_info,
        "source_data_path": source_path,
        "class_id": class_id,
        "class_label": _CLASS_LABELS[class_id],
        "sightline_idx": sightline_idx,
        "n_pixels": _N_PIXELS,
        "delta_v_kms_per_px": delta_v,
        "flux_units": "transmitted flux (normalized), dimensionless in [0, 1]",
        "wavelength_units": "Angstrom",
        "velocity_units": "km/s",
        "flux_stats": {
            "min": float(np.min(flux)),
            "max": float(np.max(flux)),
            "mean": float(np.mean(flux)),
        },
        "source_lineage": (
            "Sherwood simulation suite (Bolton+2017), z=0.3 snapshot, "
            "60 cMpc/h box"
        ),
        "selection_caveat": (
            "selection = first stored sightline of class 1 (matches "
            "results/eda/tier1_representative_spectra_2x2.png top-left panel), "
            "NOT a statistically-central/'typical' spectrum."
        ),
    }
    provenance_path = out_dir / "synthetic-spectrum.example.provenance.json"
    with open(provenance_path, "w") as fh:
        json.dump(provenance, fh, indent=2)

    return csv_path
