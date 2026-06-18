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

This is the selements-website data-export contract implementation, governed by
the ``data-export`` skill. It is cross-cutting infrastructure (NOT part of any
experiment track or its LEDGER decision-numbering) and lives on the dedicated
``service/data-export`` branch. The data-engineer agent is the primary owner.
"""

import csv
import datetime
import json
from pathlib import Path
from typing import Dict, List

import numpy as np

from src.core.data import (
    SignalClusteringData,
    assert_velocity_axes_uniform,
    load_velocity_axis,
)
from src.core.provenance import get_git_info
from src.core.transforms import FluxPowerSpectrum

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

# Canonical number of log-spaced k-bins for P_F(k), matching the project's
# pk-feedback-classifier convention (run_probe.py / run_stage1.py: N_KBINS=20).
_N_KBINS = 20


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


def _write_pk_tidy_csv(csv_path: Path, rows: List[Dict[str, object]]) -> Path:
    """Write the tidy long-form P_F(k) table (header + data only, no comments).

    The website parser is comment-unaware, so units are carried in the column
    names (``k_center_s_per_km``) and the full convention block lives in the
    provenance sidecar rather than in a fragile ``#`` header line. Floats are
    written via ``repr`` for full float64 precision.

    Columns: ``feedback_class,class_name,k_bin_idx,k_center_s_per_km,pk_mean,
    pk_sem,n_modes_in_bin,n_sightlines``. ``n_modes_in_bin`` is the number of raw
    FFT modes that fell in that log-bin; where it is 0 the bin is EMPTY and
    ``pk_mean``/``pk_sem`` are 0.0 by convention (undefined, not a physical zero)
    — the consumer should drop those rows on a log axis.

    Args:
        csv_path: Destination CSV path (parent dirs created if absent).
        rows: One dict per (class, k-bin) cell, carrying the column keys above.

    Returns:
        The ``csv_path`` written.
    """
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    header: List[str] = [
        "feedback_class",
        "class_name",
        "k_bin_idx",
        "k_center_s_per_km",
        "pk_mean",
        "pk_sem",
        "n_modes_in_bin",
        "n_sightlines",
    ]
    with open(csv_path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        for r in rows:
            writer.writerow([
                int(r["feedback_class"]),
                str(r["class_name"]),
                int(r["k_bin_idx"]),
                repr(float(r["k_center_s_per_km"])),
                repr(float(r["pk_mean"])),
                repr(float(r["pk_sem"])),
                int(r["n_modes_in_bin"]),
                int(r["n_sightlines"]),
            ])
    return csv_path


def export_exploration_pk_mean_per_class(
    out_dir: Path,
    n_kbins: int = _N_KBINS,
) -> Path:
    """Export the class-mean flux power spectrum P_F(k) for all 4 classes.

    Single-sources the physics from the project's canonical transform
    :class:`src.core.transforms.FluxPowerSpectrum` with ``norm="per_sightline"``
    (delta_F = F / <F>_los - 1, the same regime the pk-feedback-classifier and
    reframe-suite use). For each class it computes per-sightline P_F(k), then
    stacks to the class mean and the standard error on the mean
    (std / sqrt(N)) across all N=16,384 sightlines, and writes one tidy
    long-form CSV plus a git-stamped provenance sidecar that records the full
    P(k) convention (window, normalization, k-binning, units) and the honest
    verb-ceiling caveat for the ratio-to-NoFeedback signal.

    Convention (inherited verbatim from FluxPowerSpectrum, documented in the
    sidecar so the consumer's axis labels are exact):
      - delta_F = F / <F>_los - 1 (per-sightline mean-flux normalization).
      - Hann window applied before a one-sided numpy rfft.
      - P(k) = |rfft(Hann . delta_F)|^2 * (delta_v / N); units (delta_F)^2 . (km/s).
        No 1/sum(w^2) Hann-power correction (a fixed multiplicative offset on the
        absolute amplitude; irrelevant to class comparison / the ratio panel).
      - k = 2*pi*rfftfreq(N, d=delta_v), units s/km; delta_v derived from vel.npy.
      - log-binned to ``n_kbins`` geometric-center bins.

    Determinism: re-running against the same source data produces byte-identical
    CSV content (the provenance JSON differs only in timestamp/git state).

    Args:
        out_dir: Landing directory for the two output files (created if absent).
        n_kbins: Number of log-spaced k-bins. Default 20 (project canonical).

    Returns:
        Path to the written CSV (``pk-mean-per-class.csv``).

    Raises:
        ValueError: if any class produces a non-finite P_F(k) stack.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Canonical, uniform-across-classes velocity spacing (asserts identity of
    # all 4 vel.npy axes; derived from disk, never hardcoded).
    delta_v = float(assert_velocity_axes_uniform())

    # --- load all 4 class flux blocks via the canonical loader ---
    loader = SignalClusteringData()
    flux_per_class, _ = loader.load_flux_per_class()  # list of 4 x (N, 2048)

    rows: List[Dict[str, object]] = []
    k_centers: np.ndarray = np.empty(0, dtype=np.float64)
    for class_id in (1, 2, 3, 4):
        flux = np.asarray(flux_per_class[class_id - 1], dtype=np.float64)
        transform = FluxPowerSpectrum(
            norm="per_sightline", n_kbins=n_kbins, delta_v=delta_v
        )
        pk = transform.fit_transform(flux)  # shape: (N, n_kbins)
        if not np.all(np.isfinite(pk)):
            raise ValueError(
                f"non-finite P_F(k) for class {class_id}; check source flux."
            )
        k_centers = np.asarray(transform.k_bin_centers_, dtype=np.float64)
        n_sightlines = int(pk.shape[0])
        pk_mean = pk.mean(axis=0)  # shape: (n_kbins,) class mean over N sightlines
        pk_sem = pk.std(axis=0, ddof=1) / np.sqrt(n_sightlines)  # SEM on the mean
        # Count raw FFT modes per log-bin (same digitize+clip convention as the
        # transform) so empty bins (pk_mean==0) are self-documenting. The k-grid
        # is identical across classes, so this is the same vector each iteration.
        k_raw = np.asarray(transform.k_raw_, dtype=np.float64)
        edges = np.asarray(transform.k_bin_edges_, dtype=np.float64)
        bin_of_mode = np.clip(
            np.digitize(k_raw, edges, right=False), 0, n_kbins
        )
        n_modes = np.array(
            [int(np.sum(bin_of_mode == b)) for b in range(1, n_kbins + 1)],
            dtype=np.int64,
        )  # shape: (n_kbins,)
        for j in range(k_centers.shape[0]):
            rows.append({
                "feedback_class": class_id,
                "class_name": _CLASS_LABELS[class_id],
                "k_bin_idx": j,
                "k_center_s_per_km": float(k_centers[j]),
                "pk_mean": float(pk_mean[j]),
                "pk_sem": float(pk_sem[j]),
                "n_modes_in_bin": int(n_modes[j]),
                "n_sightlines": n_sightlines,
            })

    # --- write tidy CSV ---
    csv_path = out_dir / "pk-mean-per-class.csv"
    _write_pk_tidy_csv(csv_path, rows)

    # --- write provenance sidecar ---
    git_info = get_git_info()
    source_paths = {
        _CLASS_LABELS[c]: str(
            Path(SignalClusteringData.FLUX_BASE) / str(c) / "flux.npy"
        )
        for c in (1, 2, 3, 4)
    }
    provenance = {
        "export_request_slug": "exploration-pk-mean-per-class",
        "consumer": "selements-website",
        "producing_function": (
            "src.core.export.export_exploration_pk_mean_per_class"
        ),
        "consumer_facing_filename": csv_path.name,
        "export_timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        "git": git_info,
        "source_data_paths": source_paths,
        "canonical_transform": "src.core.transforms.FluxPowerSpectrum",
        "n_classes": 4,
        "n_kbins": int(n_kbins),
        "n_sightlines_per_class": 16384,
        "delta_v_kms_per_px": delta_v,
        "k_units": "s/km (velocity-space, conventional Lya basis)",
        "pk_units": "(delta_F)^2 * (km/s)",
        "delta_F_definition": "delta_F = F / <F>_los - 1 (per-sightline mean-flux)",
        "fft_convention": (
            "Hann window applied, then one-sided numpy rfft; "
            "P(k) = |rfft(Hann . delta_F)|^2 * (delta_v / N), N=2048. "
            "No 1/sum(w^2) Hann-power correction (fixed multiplicative offset "
            "on absolute amplitude; cancels in class comparison and the ratio "
            "panel)."
        ),
        "k_binning": (
            f"{n_kbins} log-spaced bins, geometric centers, "
            "edges from k_raw[1]..k_raw[-1]; identical k-grid across classes."
        ),
        "pk_sem_definition": "std(P_F(k)) / sqrt(N) across the N sightlines",
        "empty_bin_convention": (
            "Where n_modes_in_bin == 0 the log-bin caught no raw FFT modes; "
            "pk_mean and pk_sem are 0.0 by convention (UNDEFINED, not a physical "
            "zero). Drop these rows on a log axis. With n_kbins=20 over 1025 raw "
            "modes this affects one or more of the lowest-k bins."
        ),
        "source_lineage": (
            "Sherwood simulation suite (Bolton+2017), z=0.3 snapshot, "
            "60 cMpc/h box; one realization, fixed cosmology / UVB / thermal "
            "history"
        ),
        # Honest-reporting verb-ceiling, PI-framing-checked against
        # reframe-suite HARDENING_OUTCOME §H1/H2/H3 + CLOSE_OUT §1 R9 / §2
        # (2026-06-13 box-from-disk resolution promoted the verb to "weak
        # feedback signal"). Do not weaken or strengthen without a fresh check.
        "verb_ceiling": (
            "The per-class P_F(k) differences (and the ratio-to-NoFeedback "
            "panel computed from this table) are a WEAK, gate-marginal feedback "
            "signal in the 1D flux power spectrum — NOT a per-sightline "
            "detection and NOT a feedback classifier. Do not caption as "
            "'feedback detected' / 'feedback classifier' / 'we recover feedback'."
        ),
        "provenance_note": (
            "The four classes are the same density realization with only the "
            "feedback recipe varied: box size (60 cMpc/h), cosmology, redshift, "
            "sampling, and initial-condition skewer geometry are byte-identical "
            "across all four classes (confirmed from the native Sherwood LOS "
            "headers, 2026-06-13), so cross-class differences on this field are "
            "pure feedback. One open item is external only: 60 cMpc/h is "
            "non-canonical for Bolton+2017's 40/80/160 boxes, so the exact run "
            "within the Sherwood suite is not yet pinned to a published table "
            "entry — this does not affect the feedback signal."
        ),
        "strength_and_scope_caveats": [
            "The C2(StellarWind)-C3(WindAGN) pair is the weakest — gate-marginal "
            "(p16 = 0.603 at deep stacking, bootstrap CI straddles the 0.60 "
            "line) and emerges only with within-class stacking. The other "
            "feedback-recipe pairs separate more clearly (p16 >= 0.75). Read the "
            "C2-C3 ratio difference as 'consistent with a weak signal,' never as "
            "a clean separation.",
            "Mean flux and P(k) shape are partially entangled. The per-class "
            "lattice ordering correlates with per-class mean transmitted flux "
            "(Spearman ~0.94); P_F(k) shape carries a genuine residual on top, "
            "but the ratio-to-NoFeedback differences for the C1/C4-involving "
            "pairs blend a real mean-flux offset with a P(k)-shape difference. "
            "(For the C2-C3 pair specifically the difference is carried entirely "
            "by shape; its mean-flux-only baseline stays at chance.)",
            "Single z=0.3 snapshot, single realization, fixed cosmology and UV "
            "background; no nuisance-parameter marginalization was performed.",
            "The lowest-k bin can carry residual mean-flux-normalization / "
            "windowing leakage (per reframe-suite [D-06]) and may be "
            "de-emphasized.",
        ],
    }
    provenance_path = out_dir / "pk-mean-per-class.provenance.json"
    with open(provenance_path, "w") as fh:
        json.dump(provenance, fh, indent=2)

    return csv_path
