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
from typing import Dict, List, Tuple

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

# Local-EW distribution grid: 40 log2 edges (-> 39 bins), matching the published
# EDA figure tier1_local_ew_dist_2x2.png (scripts/eda_sherwood.py:335). EPS is the
# plot-axis lower floor from eda_sherwood.py:14 (a floor on the log grid, NOT a
# physical EW threshold).
_EW_N_EDGES = 40
_EW_EPS = 1e-12

# Per-sightline line-density distribution grid: linear (line density ~0-3
# lines/Angstrom), 40 edges -> 39 bins, data-driven bounds across all classes.
_LD_N_EDGES = 40

# PI-framing-checked, caption-binding strings for the four relationship figures.
# Two of the four requested premises did NOT survive full-data computation; these
# captions are the honest, PI-approved framings. Do not weaken/strengthen without
# a fresh framing-check. (Guardrail: never imply C4 has weaker PER-LINE
# absorption — that is a retired overstatement; C4's lower TOTAL EW is driven by
# fewer lines, not shallower lines.)
_FIG_CAPTIONS: Dict[str, str] = {
    "ew-vs-density": (
        "Total equivalent width vs. line density (log-log), 16,384 sightlines "
        "per class. The positive EW-density trend is weak and class-dependent: "
        "Pearson r = +0.13 (C1), +0.16 (C2), +0.11 (C3), and -0.03 (C4) — the "
        "trend reverses sign for WindStrongAGN, it does not merely weaken. The "
        "robust separation is that C4 occupies the lower-left (median density "
        "1.29 vs ~1.78 lines/A; median total EW 0.36 vs ~0.55 A); C1-C3 overlap. "
        "Line density counts every local flux minimum (find_peaks(-flux), no "
        "prominence threshold), so it includes shallow, low-prominence minima — "
        "the spectra are noiseless (S/N=inf), so these are genuine sub-threshold "
        "features of the simulated flux field, not measurement noise — not only "
        "the prominent, resolved absorption lines."
    ),
    "gap-vs-density": (
        "Mean gap vs. line density (log-log). Slopes -0.99 to -1.01 (r ~ -0.99, "
        "all classes). This is a near-exact algebraic identity — mean gap = "
        "wavelength span / (n_lines - 1) — not an empirical finding; it serves as "
        "a consistency check on the line-detection and density computation. Any "
        "physical clustering-vs-Poisson signal lives in the gap-distribution "
        "shape (see the gap-distribution figure), not in this mean relation, "
        "which is definitionally fixed."
    ),
    "ew-vs-depth": (
        "Per-line local EW (log) vs. depth 1-F (linear). Depth is saturation-"
        "bounded in [0,1] while local EW spans ~2 orders of magnitude (max ~2.9 / "
        "10.6 / 18.4 / 10.4 A for C1-C4) — wide feature-width dynamic range at "
        "capped depth. Descriptive; one z=0.3 snapshot."
    ),
    "activity-vs-density": (
        "Per-bin absorption activity (integrated 1-F) vs. per-bin line count, "
        "linear axes, 50 wavelength bins x 16,384 sightlines. The intuitive 'more "
        "lines -> more activity' relation does not hold: Pearson r is near zero "
        "(-0.06, +0.15, +0.08, +0.14 for C1-C4) and mean activity is non-"
        "monotonic, peaking at ~1 line/bin then decreasing (k=1: ~0.013-0.017; "
        "k=2: ~0.008-0.011; k=3: ~0.005-0.007, all classes). Because line "
        "detection uses find_peaks(-flux) with no prominence threshold, bins with "
        "more detected 'lines' contain proportionally more shallow, low-prominence "
        "minima (the spectra are noiseless, so these are sub-threshold flux "
        "features, not measurement noise) that carry little EW, which lowers mean "
        "per-line activity — the decrease is consistent with a detection-threshold "
        "effect, not a physical anti-correlation. Reported as a descriptive null "
        "relative to the naive expectation."
    ),
}


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


def _write_ew_dist_csv(csv_path: Path, rows: List[Dict[str, object]]) -> Path:
    """Write the tidy per-class local-EW distribution CSV (header + data only).

    Columns: ``feedback_class,class_name,ew_center_angstrom,density,count``.
    ``density`` is the fraction of the class's (positive) lines in that bin
    (sums to 1 across the shared grid); ``count`` is the raw line count, so the
    consumer can grey out under-populated tails. Floats via ``repr`` for full
    precision.

    Args:
        csv_path: Destination CSV path (parent dirs created if absent).
        rows: One dict per (class, EW-bin) cell, carrying the column keys above.

    Returns:
        The ``csv_path`` written.
    """
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    header: List[str] = [
        "feedback_class",
        "class_name",
        "ew_center_angstrom",
        "density",
        "count",
    ]
    with open(csv_path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        for r in rows:
            writer.writerow([
                int(r["feedback_class"]),
                str(r["class_name"]),
                repr(float(r["ew_center_angstrom"])),
                repr(float(r["density"])),
                int(r["count"]),
            ])
    return csv_path


def _write_ew_summary_csv(csv_path: Path, rows: List[Dict[str, object]]) -> Path:
    """Write the companion per-class local-EW summary CSV (median / IQR / n).

    Columns: ``feedback_class,class_name,median_ew_angstrom,p25_ew_angstrom,
    p75_ew_angstrom,n_lines``. Quantiles are over the positive EW values that
    enter the distribution; ``n_lines`` is the total absorption lines extracted
    for the class.

    Args:
        csv_path: Destination CSV path (parent dirs created if absent).
        rows: One dict per class, carrying the column keys above.

    Returns:
        The ``csv_path`` written.
    """
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    header: List[str] = [
        "feedback_class",
        "class_name",
        "median_ew_angstrom",
        "p25_ew_angstrom",
        "p75_ew_angstrom",
        "n_lines",
    ]
    with open(csv_path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        for r in rows:
            writer.writerow([
                int(r["feedback_class"]),
                str(r["class_name"]),
                repr(float(r["median_ew_angstrom"])),
                repr(float(r["p25_ew_angstrom"])),
                repr(float(r["p75_ew_angstrom"])),
                int(r["n_lines"]),
            ])
    return csv_path


def export_exploration_local_ew_dist_per_class(
    out_dir: Path,
    n_edges: int = _EW_N_EDGES,
) -> Path:
    """Export the per-class local equivalent-width distribution for selements-website.

    Single-sources the physics from the project's canonical EDA extractor
    ``scripts.eda_sherwood.local_equivalent_widths`` (saddle-point-splitting
    deblend: per-line EW = integral of (1 - F) dlambda between neighbouring
    saddle points), with line centres from ``detect_local_minima`` (local minima
    of F via ``find_peaks(-flux)``; no detection threshold, no integrand clamp).
    The extractor lives in a script (the only canonical implementation), so it is
    imported lazily here to keep the export library import light.

    For each of the 4 classes it concatenates the per-line EW over all 16,384
    sightlines, then bins to a SHARED log2 grid (data-driven bounds, matching the
    published figure ``tier1_local_ew_dist_2x2.png``, eda_sherwood.py:335) and
    writes:
      - ``local-ew-dist-per-class.csv`` (tidy: class, ew_center, density, count),
      - ``local-ew-summary-per-class.csv`` (median / IQR / n_lines per class),
      - a git-stamped provenance sidecar with the full convention + honesty caveat.

    ``density`` is the fraction of the class's positive lines per bin (sums to 1),
    so the four shapes overlay regardless of how many lines each class has.

    Conventions (documented in the sidecar so the consumer's axis labels are
    exact):
      - EW in Angstrom; integrand 1 - F (NO clamp; a saddle with F>1 can subtract
        slightly), matching eda_sherwood.py:89.
      - Lines = every local minimum of F (no prominence / smoothing / depth gate).
      - Shared grid: 40 log2 edges (39 bins) over [max(global_min_pos, EPS),
        global_max], geometric bin centres. EPS=1e-12 is a log-axis floor.
      - Non-positive EW values (rare degenerate saddles) are dropped before the
        log histogram and counted in the sidecar.

    Determinism: re-running against the same source data produces byte-identical
    CSV content (provenance JSON differs only in timestamp/git state).

    Args:
        out_dir: Landing directory for the output files (created if absent).
        n_edges: Number of log2 bin EDGES (default 40 -> 39 bins; matches the
            published figure).

    Returns:
        Path to the written distribution CSV (``local-ew-dist-per-class.csv``).

    Raises:
        ValueError: if a class yields no positive EW values (no lines extracted).
    """
    # Lazy import of the canonical extractor (script-level; pulls matplotlib +
    # load_dotenv on import, so we defer it to call time rather than module load).
    from scripts.eda_sherwood import detect_local_minima, local_equivalent_widths

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    wave = _load_wavelength_axis(1)  # (2048,) Angstrom; identical across classes
    loader = SignalClusteringData()
    flux_per_class, _ = loader.load_flux_per_class()  # list of 4 x (N, 2048)

    # --- pass 1: extract all per-line EW per class ---
    ew_by_class: Dict[int, np.ndarray] = {}
    for class_id in (1, 2, 3, 4):
        block = np.asarray(flux_per_class[class_id - 1], dtype=np.float64)
        per_sightline: List[np.ndarray] = []
        for i in range(block.shape[0]):
            fl = block[i]  # shape: (2048,)
            minima_idx = detect_local_minima(fl)
            ews = local_equivalent_widths(wave, fl, minima_idx)  # per-line EW (A)
            if ews.size:
                per_sightline.append(ews)
        ew_by_class[class_id] = (
            np.concatenate(per_sightline) if per_sightline else np.array([])
        )

    # --- shared log2 grid from the global positive-EW range across all classes ---
    pos = {
        c: ew_by_class[c][ew_by_class[c] > 0.0] for c in (1, 2, 3, 4)
    }
    for c in (1, 2, 3, 4):
        if pos[c].size == 0:
            raise ValueError(f"class {c} produced no positive EW values")
    global_min_pos = min(float(pos[c].min()) for c in (1, 2, 3, 4))
    global_max = max(float(pos[c].max()) for c in (1, 2, 3, 4))
    lo = max(global_min_pos, _EW_EPS)
    edges = np.logspace(
        np.log2(lo), np.log2(global_max), n_edges, base=2.0
    )  # shape: (n_edges,) -> n_edges-1 bins
    centers = np.sqrt(edges[:-1] * edges[1:])  # geometric bin centres

    # --- per class: histogram on the shared grid + summary stats ---
    dist_rows: List[Dict[str, object]] = []
    summary_rows: List[Dict[str, object]] = []
    n_nonpositive: Dict[str, int] = {}
    density_by_class: Dict[int, np.ndarray] = {}
    n_lines_by_label: Dict[str, int] = {}
    modal_ew_by_label: Dict[str, float] = {}
    for class_id in (1, 2, 3, 4):
        all_ew = ew_by_class[class_id]
        ew_pos = pos[class_id]
        n_pos = int(ew_pos.size)
        n_nonpositive[_CLASS_LABELS[class_id]] = int(all_ew.size - n_pos)
        counts, _ = np.histogram(ew_pos, bins=edges)  # shape: (n_edges-1,)
        density = counts.astype(np.float64) / float(n_pos)  # fraction per bin
        density_by_class[class_id] = density
        n_lines_by_label[_CLASS_LABELS[class_id]] = int(all_ew.size)
        modal_ew_by_label[_CLASS_LABELS[class_id]] = float(centers[int(np.argmax(density))])
        for j in range(centers.shape[0]):
            dist_rows.append({
                "feedback_class": class_id,
                "class_name": _CLASS_LABELS[class_id],
                "ew_center_angstrom": float(centers[j]),
                "density": float(density[j]),
                "count": int(counts[j]),
            })
        q25, q50, q75 = (float(v) for v in np.quantile(ew_pos, [0.25, 0.5, 0.75]))
        summary_rows.append({
            "feedback_class": class_id,
            "class_name": _CLASS_LABELS[class_id],
            "median_ew_angstrom": q50,
            "p25_ew_angstrom": q25,
            "p75_ew_angstrom": q75,
            "n_lines": int(all_ew.size),
        })

    # --- distribution-overlap metric (grounds the honesty caveat in numbers) ---
    # Total-variation distance of each class's normalized EW distribution vs
    # NoFeedback: 0.5 * sum|p_c - p_1|. 0 = identical, 1 = disjoint.
    ref = density_by_class[1]
    tv_vs_nofeedback: Dict[str, float] = {
        _CLASS_LABELS[c]: float(0.5 * np.sum(np.abs(density_by_class[c] - ref)))
        for c in (1, 2, 3, 4)
    }

    # --- write the two CSVs ---
    csv_path = out_dir / "local-ew-dist-per-class.csv"
    _write_ew_dist_csv(csv_path, dist_rows)
    summary_path = out_dir / "local-ew-summary-per-class.csv"
    _write_ew_summary_csv(summary_path, summary_rows)

    # --- provenance sidecar ---
    git_info = get_git_info()
    source_paths = {
        _CLASS_LABELS[c]: str(
            Path(SignalClusteringData.FLUX_BASE) / str(c) / "flux.npy"
        )
        for c in (1, 2, 3, 4)
    }
    provenance = {
        "export_request_slug": "exploration-local-ew-dist",
        "consumer": "selements-website",
        "producing_function": (
            "src.core.export.export_exploration_local_ew_dist_per_class"
        ),
        "consumer_facing_filenames": [csv_path.name, summary_path.name],
        "export_timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        "git": git_info,
        "source_data_paths": source_paths,
        "canonical_extractor": (
            "scripts.eda_sherwood.local_equivalent_widths + detect_local_minima"
        ),
        "n_classes": 4,
        "n_sightlines_per_class": 16384,
        "ew_units": "Angstrom",
        "ew_definition": (
            "EW_local = integral of (1 - F) dlambda between neighbouring saddle "
            "points (Saddle-Point Splitting deblend); lines = local minima of F "
            "(find_peaks(-flux), no prominence/smoothing/depth threshold)."
        ),
        "integrand_clamp": (
            "none — integrand is 1 - F as-is (a saddle with F>1 can subtract "
            "slightly), per eda_sherwood.py:89."
        ),
        "grid": (
            f"shared log2 grid, {n_edges} edges ({n_edges - 1} bins), geometric "
            "centres, data-driven bounds [max(global_min_positive, EPS), "
            "global_max] across all 4 classes; matches tier1_local_ew_dist_2x2.png "
            "(eda_sherwood.py:335). EPS=1e-12 is a log-axis floor, not a physical "
            "EW threshold."
        ),
        "grid_bounds_angstrom": {"lo": lo, "hi": global_max},
        "density_definition": (
            "count / n_positive_lines (fraction of the class's positive lines per "
            "bin; sums to 1 across the shared grid). Bins are uniform in log2(EW)."
        ),
        "n_nonpositive_ew_dropped": n_nonpositive,
        "note_on_requested_range": (
            "The request's '~2^-6..2^0 A' is loose LEDGER §6 prose; the actual "
            "per-line EW median is ~2^-9.8 A, so that range would clip the bulk of "
            "the distribution. This export uses the published figure's data-driven "
            "range instead (much wider), so the curve is faithful."
        ),
        "source_lineage": (
            "Sherwood simulation suite (Bolton+2017), z=0.3 snapshot, 60 cMpc/h "
            "box; one realization, fixed cosmology / UVB / thermal history"
        ),
        # Auditable evidence behind the honesty caveat.
        "tv_distance_vs_nofeedback": tv_vs_nofeedback,
        "modal_ew_angstrom_per_class": modal_ew_by_label,
        "n_lines_per_class": n_lines_by_label,
        # Honesty caveat — PI-adjudicated (DO-NOT-SHIP-as-drafted -> reframe).
        # The originally requested framing ("Class 4 clearly separates,
        # narrowest/lowest") is CONTRADICTED by this data and was an overstated
        # LEDGER §6 caption (a KDE/log-axis artifact). See PI ruling.
        "key_finding": (
            "The per-line local-EW distributions are NEARLY IDENTICAL across all "
            "four feedback classes: same modal EW (~8e-4 A), >=97% distributional "
            "overlap (total-variation distance <= 0.026 vs NoFeedback). The real "
            "class separator is LINE DENSITY, not per-line EW: WindStrongAGN "
            "(Class 4) produces ~28% fewer absorption lines (612k vs 823-847k), "
            "consistent with a sparse, void-like environment."
        ),
        "honest_reporting_caveat": (
            "The per-line EW distributions overlap almost completely across all "
            "four classes (total-variation distance <= 0.026 vs NoFeedback; "
            "identical modal EW). Class 4 does NOT have the lowest per-line EW — "
            "its median is marginally the highest. A small (~4-10%) progressive "
            "median right-shift runs C1<C2<C3~=C4. The genuine class separator is "
            "line density, not per-line EW: Class 4 carries ~28% fewer lines. Do "
            "NOT caption as 'Class 4 separates' / 'narrowest/lowest EW' or imply a "
            "classifier. Single z=0.3 snapshot, one realization, fixed cosmology."
        ),
    }
    provenance_path = out_dir / "local-ew-dist-per-class.provenance.json"
    with open(provenance_path, "w") as fh:
        json.dump(provenance, fh, indent=2)

    return csv_path


def _write_line_density_dist_csv(
    csv_path: Path, rows: List[Dict[str, object]]
) -> Path:
    """Write the tidy per-class line-density distribution CSV (header + data only).

    Columns: ``feedback_class,class_name,line_density_center_lines_per_angstrom,
    density,count``. ``density`` = fraction of the class's sightlines per bin
    (sums to 1); ``count`` = raw sightline count. Floats via ``repr``.
    """
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    header: List[str] = [
        "feedback_class",
        "class_name",
        "line_density_center_lines_per_angstrom",
        "density",
        "count",
    ]
    with open(csv_path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        for r in rows:
            writer.writerow([
                int(r["feedback_class"]),
                str(r["class_name"]),
                repr(float(r["line_density_center_lines_per_angstrom"])),
                repr(float(r["density"])),
                int(r["count"]),
            ])
    return csv_path


def _write_line_density_summary_csv(
    csv_path: Path, rows: List[Dict[str, object]]
) -> Path:
    """Write the companion per-class line-density summary CSV.

    Columns: ``feedback_class,class_name,median_lines_per_angstrom,
    p25_lines_per_angstrom,p75_lines_per_angstrom,mean_lines_per_angstrom,
    n_sightlines``.
    """
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    header: List[str] = [
        "feedback_class",
        "class_name",
        "median_lines_per_angstrom",
        "p25_lines_per_angstrom",
        "p75_lines_per_angstrom",
        "mean_lines_per_angstrom",
        "n_sightlines",
    ]
    with open(csv_path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        for r in rows:
            writer.writerow([
                int(r["feedback_class"]),
                str(r["class_name"]),
                repr(float(r["median_lines_per_angstrom"])),
                repr(float(r["p25_lines_per_angstrom"])),
                repr(float(r["p75_lines_per_angstrom"])),
                repr(float(r["mean_lines_per_angstrom"])),
                int(r["n_sightlines"]),
            ])
    return csv_path


def export_exploration_line_density_per_class(
    out_dir: Path,
    n_edges: int = _LD_N_EDGES,
) -> Path:
    """Export the per-class absorption-line-density distribution for selements-website.

    This is the quantity that ACTUALLY separates the feedback classes (the
    per-line EW distribution does not — see
    :func:`export_exploration_local_ew_dist_per_class`). Single-sourced from the
    canonical ``scripts.eda_sherwood.line_density`` (lines per Angstrom =
    n_local_minima / wavelength_span) with line centres from
    ``detect_local_minima``.

    For each class it computes the per-sightline line density over all 16,384
    sightlines, bins to a SHARED linear grid (data-driven bounds), and writes a
    tidy distribution CSV + a summary CSV (median / IQR / mean / n_sightlines) +
    a git-stamped provenance sidecar. ``density`` is the fraction of the class's
    sightlines per bin (sums to 1), so the four ridgelines overlay regardless of
    count.

    Determinism: re-running against the same source data is byte-identical
    (provenance JSON differs only in timestamp/git state).

    Args:
        out_dir: Landing directory for the output files (created if absent).
        n_edges: Number of linear bin EDGES (default 40 -> 39 bins).

    Returns:
        Path to the written distribution CSV (``line-density-dist-per-class.csv``).

    Raises:
        ValueError: if a class yields no sightlines.
    """
    from scripts.eda_sherwood import detect_local_minima, line_density

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    wave = _load_wavelength_axis(1)  # (2048,) Angstrom; identical across classes
    loader = SignalClusteringData()
    flux_per_class, _ = loader.load_flux_per_class()  # list of 4 x (N, 2048)

    # --- per-sightline line density per class ---
    ld_by_class: Dict[int, np.ndarray] = {}
    for class_id in (1, 2, 3, 4):
        block = np.asarray(flux_per_class[class_id - 1], dtype=np.float64)
        vals = np.empty(block.shape[0], dtype=np.float64)  # shape: (N,)
        for i in range(block.shape[0]):
            minima_idx = detect_local_minima(block[i])
            vals[i] = float(line_density(minima_idx, wave))  # lines / Angstrom
        if vals.size == 0:
            raise ValueError(f"class {class_id} produced no sightlines")
        ld_by_class[class_id] = vals

    # --- shared linear grid from the global range across all classes ---
    global_min = min(float(ld_by_class[c].min()) for c in (1, 2, 3, 4))
    global_max = max(float(ld_by_class[c].max()) for c in (1, 2, 3, 4))
    edges = np.linspace(global_min, global_max, n_edges)  # shape: (n_edges,)
    centers = 0.5 * (edges[:-1] + edges[1:])

    dist_rows: List[Dict[str, object]] = []
    summary_rows: List[Dict[str, object]] = []
    for class_id in (1, 2, 3, 4):
        vals = ld_by_class[class_id]
        n = int(vals.size)
        counts, _ = np.histogram(vals, bins=edges)  # shape: (n_edges-1,)
        density = counts.astype(np.float64) / float(n)  # fraction per bin
        for j in range(centers.shape[0]):
            dist_rows.append({
                "feedback_class": class_id,
                "class_name": _CLASS_LABELS[class_id],
                "line_density_center_lines_per_angstrom": float(centers[j]),
                "density": float(density[j]),
                "count": int(counts[j]),
            })
        q25, q50, q75 = (float(v) for v in np.quantile(vals, [0.25, 0.5, 0.75]))
        summary_rows.append({
            "feedback_class": class_id,
            "class_name": _CLASS_LABELS[class_id],
            "median_lines_per_angstrom": q50,
            "p25_lines_per_angstrom": q25,
            "p75_lines_per_angstrom": q75,
            "mean_lines_per_angstrom": float(vals.mean()),
            "n_sightlines": n,
        })

    csv_path = out_dir / "line-density-dist-per-class.csv"
    _write_line_density_dist_csv(csv_path, dist_rows)
    summary_path = out_dir / "line-density-summary-per-class.csv"
    _write_line_density_summary_csv(summary_path, summary_rows)

    git_info = get_git_info()
    source_paths = {
        _CLASS_LABELS[c]: str(
            Path(SignalClusteringData.FLUX_BASE) / str(c) / "flux.npy"
        )
        for c in (1, 2, 3, 4)
    }
    mean_ld = {
        _CLASS_LABELS[c]: float(ld_by_class[c].mean()) for c in (1, 2, 3, 4)
    }
    provenance = {
        "export_request_slug": "exploration-line-density",
        "consumer": "selements-website",
        "producing_function": (
            "src.core.export.export_exploration_line_density_per_class"
        ),
        "consumer_facing_filenames": [csv_path.name, summary_path.name],
        "export_timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        "git": git_info,
        "source_data_paths": source_paths,
        "canonical_extractor": (
            "scripts.eda_sherwood.line_density + detect_local_minima"
        ),
        "n_classes": 4,
        "n_sightlines_per_class": 16384,
        "line_density_units": "lines per Angstrom (n_local_minima / wavelength_span)",
        "line_density_definition": (
            "per sightline: count of local minima of F (find_peaks(-flux)) divided "
            "by the wavelength span (lambda[-1]-lambda[0] ~= 28.71 A), per "
            "eda_sherwood.py:100-102."
        ),
        "grid": (
            f"shared LINEAR grid, {n_edges} edges ({n_edges - 1} bins), midpoint "
            "centres, data-driven bounds across all 4 classes."
        ),
        "density_definition": (
            "count / n_sightlines (fraction of the class's sightlines per bin; "
            "sums to 1)."
        ),
        "mean_lines_per_angstrom_per_class": mean_ld,
        "source_lineage": (
            "Sherwood simulation suite (Bolton+2017), z=0.3 snapshot, 60 cMpc/h "
            "box; one realization, fixed cosmology / UVB / thermal history"
        ),
        # This is the honest separator (per PI ruling + eda-sherwood LEDGER §5,
        # which the PI re-verified as correct, unlike the §6 per-line-EW caption).
        "honest_reporting_caveat": (
            "Descriptive distribution. This is where the feedback classes DO "
            "differ: WindStrongAGN (Class 4) has clearly lower line density "
            "(~1.30 vs ~1.75-1.80 lines/A; ~28% fewer lines), consistent with a "
            "sparse, void-like environment. Classes 1-2 are nearly identical, "
            "Class 3 slightly lower. This is line ABUNDANCE, not line strength "
            "(per-line EW distributions overlap across all 4 classes — see the "
            "local-ew-dist export). Still descriptive, NOT a classifier: one "
            "z=0.3 snapshot, one realization, fixed cosmology; the C1/C2/C3 "
            "distributions overlap heavily, only C4 stands out."
        ),
    }
    provenance_path = out_dir / "line-density-dist-per-class.provenance.json"
    with open(provenance_path, "w") as fh:
        json.dump(provenance, fh, indent=2)

    return csv_path


def _edges_1d(pooled: np.ndarray, n_edges: int, log: bool) -> np.ndarray:
    """Shared bin edges over a pooled value array (log10 or linear)."""
    vals = pooled[np.isfinite(pooled)]
    lo, hi = float(vals.min()), float(vals.max())
    if log:
        return np.logspace(np.log10(lo), np.log10(hi), n_edges)
    return np.linspace(lo, hi, n_edges)


def _centers_1d(edges: np.ndarray, log: bool) -> np.ndarray:
    """Bin centres: geometric for log edges, midpoint for linear."""
    if log:
        return np.sqrt(edges[:-1] * edges[1:])
    return 0.5 * (edges[:-1] + edges[1:])


def _bin2d_rows(
    xv: np.ndarray,
    yv: np.ndarray,
    x_edges: np.ndarray,
    y_edges: np.ndarray,
    x_log: bool,
    y_log: bool,
    class_id: int,
) -> List[Dict[str, object]]:
    """2-D histogram one class onto the shared grid; emit NON-EMPTY bins only.

    ``density`` is per-class normalized (counts / class total) so it sums to 1
    across the grid and shapes compare regardless of N.
    """
    counts, _, _ = np.histogram2d(xv, yv, bins=[x_edges, y_edges])
    total = float(counts.sum())
    xc = _centers_1d(x_edges, x_log)
    yc = _centers_1d(y_edges, y_log)
    rows: List[Dict[str, object]] = []
    nz = np.argwhere(counts > 0)
    for ix, iy in nz:
        rows.append({
            "feedback_class": class_id,
            "class_name": _CLASS_LABELS[class_id],
            "x": float(xc[ix]),
            "y": float(yc[iy]),
            "count": int(counts[ix, iy]),
            "density": float(counts[ix, iy] / total),
        })
    return rows


def _trend_rows(
    xv: np.ndarray,
    yv: np.ndarray,
    x_edges: np.ndarray,
    x_log: bool,
    class_id: int,
    min_count: int = 20,
) -> List[Dict[str, object]]:
    """Per-x-bin y median / p25 / p75 for one class (bins with >= min_count)."""
    xc = _centers_1d(x_edges, x_log)
    idx = np.digitize(xv, x_edges) - 1  # 0..len(xc)-1 for in-range
    rows: List[Dict[str, object]] = []
    for ix in range(xc.shape[0]):
        m = idx == ix
        if int(m.sum()) >= min_count:
            q25, q50, q75 = (float(v) for v in np.quantile(yv[m], [0.25, 0.5, 0.75]))
            rows.append({
                "feedback_class": class_id,
                "class_name": _CLASS_LABELS[class_id],
                "x": float(xc[ix]),
                "y_median": q50,
                "y_p25": q25,
                "y_p75": q75,
            })
    return rows


def _write_2d_csv(
    csv_path: Path, rows: List[Dict[str, object]], x_col: str, y_col: str
) -> Path:
    """Write a tidy 2-D density CSV: class, <x_col>, <y_col>, count, density."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    header = ["feedback_class", "class_name", x_col, y_col, "count", "density"]
    with open(csv_path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        for r in rows:
            writer.writerow([
                int(r["feedback_class"]),
                str(r["class_name"]),
                repr(float(r["x"])),
                repr(float(r["y"])),
                int(r["count"]),
                repr(float(r["density"])),
            ])
    return csv_path


def _write_trend_csv(
    csv_path: Path, rows: List[Dict[str, object]], x_col: str
) -> Path:
    """Write a tidy per-x-bin trend CSV: class, <x_col>, y_median, y_p25, y_p75."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    header = ["feedback_class", "class_name", x_col, "y_median", "y_p25", "y_p75"]
    with open(csv_path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        for r in rows:
            writer.writerow([
                int(r["feedback_class"]),
                str(r["class_name"]),
                repr(float(r["x"])),
                repr(float(r["y_median"])),
                repr(float(r["y_p25"])),
                repr(float(r["y_p75"])),
            ])
    return csv_path


def _loglog_corr(xv: np.ndarray, yv: np.ndarray) -> Tuple[float, float]:
    """Pearson r and slope of log10(y) vs log10(x) (for the honesty stats)."""
    lx, ly = np.log10(xv), np.log10(yv)
    r = float(np.corrcoef(lx, ly)[0, 1])
    slope = float(np.polyfit(lx, ly, 1)[0])
    return r, slope


def export_exploration_relationships_2d(
    out_dir: Path,
    n_edges: int = 40,
) -> Path:
    """Export four Tier-1 EDA relationship figures as 2-D binned-density CSVs.

    Single-sources every metric from the canonical EDA code in
    ``scripts/eda_sherwood.py`` (``total_equivalent_width``, ``line_density``,
    ``gap_statistics``, ``absorption_depths``, ``local_equivalent_widths``,
    ``absorption_activity_profile``). One pass over all 16,384 sightlines x 4
    classes produces, per figure, a shared-grid 2-D density CSV (+ a companion
    per-x-bin trend CSV) and a combined provenance sidecar.

    Figures (file stems):
      - ``ew-vs-density``    : per-sightline total EW vs line density (log-log)
      - ``gap-vs-density``   : per-sightline mean gap vs line density (log-log)
      - ``ew-vs-depth``      : per-line local EW (log) vs depth 1-F (linear)
      - ``activity-vs-density``: per-bin activity vs per-bin line density (linear)

    ``density`` is per-class normalized (sums to 1 across the grid). Only
    non-empty bins are emitted. The provenance carries per-class correlation
    statistics so the honesty caveats are grounded in numbers (two of the four
    requested findings do NOT hold as stated — see the caveat).

    Determinism: re-running against the same source data is byte-identical
    (provenance JSON differs only in timestamp/git state).

    Args:
        out_dir: Landing directory for the output files (created if absent).
        n_edges: Number of bin edges per axis (default 40 -> 39 bins).

    Returns:
        Path to the landing directory.
    """
    from scripts.eda_sherwood import (
        absorption_activity_profile,
        absorption_depths,
        detect_local_minima,
        gap_statistics,
        line_density,
        local_equivalent_widths,
        total_equivalent_width,
    )

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    wave = _load_wavelength_axis(1)  # (2048,) Angstrom; identical across classes
    binedges = np.linspace(wave.min(), wave.max(), 51)  # canonical 50-bin grid
    bin_width = float(np.diff(binedges).mean())
    loader = SignalClusteringData()
    flux_per_class, _ = loader.load_flux_per_class()

    # data[fig][class_id] = (x_array, y_array)
    figs = ["ew-vs-density", "gap-vs-density", "ew-vs-depth", "activity-vs-density"]
    data: Dict[str, Dict[int, Tuple[np.ndarray, np.ndarray]]] = {f: {} for f in figs}

    for class_id in (1, 2, 3, 4):
        block = np.asarray(flux_per_class[class_id - 1], dtype=np.float64)
        n = block.shape[0]
        dens = np.empty(n, dtype=np.float64)
        tew = np.empty(n, dtype=np.float64)
        gap = np.empty(n, dtype=np.float64)
        depth_chunks: List[np.ndarray] = []
        ew_chunks: List[np.ndarray] = []
        act_chunks: List[np.ndarray] = []
        bden_chunks: List[np.ndarray] = []
        for i in range(n):
            f = block[i]
            mi = detect_local_minima(f)
            dens[i] = float(line_density(mi, wave))
            tew[i] = float(total_equivalent_width(wave, f))
            gap[i] = float(gap_statistics(wave, mi)["gap_mean"])
            depth_chunks.append(absorption_depths(f, mi))
            ew_chunks.append(local_equivalent_widths(wave, f, mi))
            act, bden = absorption_activity_profile(wave, f, binedges)
            act_chunks.append(np.asarray(act, dtype=np.float64))
            bden_chunks.append(np.asarray(bden, dtype=np.float64))
        depth = np.concatenate(depth_chunks)
        ew = np.concatenate(ew_chunks)
        act = np.concatenate(act_chunks)
        bden = np.concatenate(bden_chunks)
        data["ew-vs-density"][class_id] = (dens, tew)
        data["gap-vs-density"][class_id] = (dens, gap)
        data["ew-vs-depth"][class_id] = (depth, ew)
        data["activity-vs-density"][class_id] = (bden, act)

    # Per-figure: scales, column names, filter, shared edges, bin, write, stats.
    fig_cfg = {
        "ew-vs-density": dict(
            xcol="line_density_lines_per_angstrom", ycol="total_ew_angstrom",
            xlog=True, ylog=True,
        ),
        "gap-vs-density": dict(
            xcol="line_density_lines_per_angstrom", ycol="mean_gap_angstrom",
            xlog=True, ylog=True,
        ),
        "ew-vs-depth": dict(
            xcol="depth_1_minus_f", ycol="local_ew_angstrom",
            xlog=False, ylog=True,
        ),
        "activity-vs-density": dict(
            xcol="bin_line_density_lines_per_angstrom",
            ycol="bin_activity_ew_angstrom", xlog=False, ylog=False,
        ),
    }

    def _filter(fig: str, xv: np.ndarray, yv: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        cfg = fig_cfg[fig]
        mask = np.isfinite(xv) & np.isfinite(yv)
        if cfg["xlog"]:
            mask &= xv > 0.0
        elif fig == "ew-vs-depth":
            mask &= xv >= 0.0  # depth linear axis; drop rare F>1 negatives
        else:  # activity-vs-density: keep non-empty (density>0) bins
            mask &= xv > 0.0
        if cfg["ylog"]:
            mask &= yv > 0.0
        return xv[mask], yv[mask]

    fig_stats: Dict[str, Dict[str, object]] = {}
    for fig in figs:
        cfg = fig_cfg[fig]
        filt = {c: _filter(fig, *data[fig][c]) for c in (1, 2, 3, 4)}
        px = np.concatenate([filt[c][0] for c in (1, 2, 3, 4)])
        py = np.concatenate([filt[c][1] for c in (1, 2, 3, 4)])
        x_edges = _edges_1d(px, n_edges, cfg["xlog"])
        y_edges = _edges_1d(py, n_edges, cfg["ylog"])

        dist_rows: List[Dict[str, object]] = []
        trend_rows: List[Dict[str, object]] = []
        per_class_stat: Dict[str, object] = {}
        for c in (1, 2, 3, 4):
            xv, yv = filt[c]
            dist_rows += _bin2d_rows(
                xv, yv, x_edges, y_edges, cfg["xlog"], cfg["ylog"], c
            )
            trend_rows += _trend_rows(xv, yv, x_edges, cfg["xlog"], c)
            # honesty statistic per class
            if cfg["xlog"] and cfg["ylog"]:
                r, slope = _loglog_corr(xv, yv)
                per_class_stat[_CLASS_LABELS[c]] = {
                    "loglog_pearson_r": round(r, 4), "loglog_slope": round(slope, 4)
                }
            elif fig == "ew-vs-depth":
                per_class_stat[_CLASS_LABELS[c]] = {
                    "depth_min": round(float(xv.min()), 4),
                    "depth_max": round(float(xv.max()), 4),
                    "ew_max_angstrom": round(float(yv.max()), 4),
                }
            else:  # activity-vs-density: r + activity at integer line-count bins
                r = float(np.corrcoef(xv, yv)[0, 1])
                k = np.rint(xv * bin_width).astype(int)
                act_by_k = {
                    str(kk): round(float(yv[k == kk].mean()), 5)
                    for kk in (1, 2, 3) if np.any(k == kk)
                }
                per_class_stat[_CLASS_LABELS[c]] = {
                    "pearson_r": round(r, 4), "mean_activity_by_linecount": act_by_k
                }

        _write_2d_csv(out_dir / f"{fig}-2d.csv", dist_rows, cfg["xcol"], cfg["ycol"])
        _write_trend_csv(out_dir / f"{fig}-trend.csv", trend_rows, cfg["xcol"])
        fig_stats[fig] = {
            "x_col": cfg["xcol"], "y_col": cfg["ycol"],
            "x_scale": "log10" if cfg["xlog"] else "linear",
            "y_scale": "log10" if cfg["ylog"] else "linear",
            "n_bins_per_axis": n_edges - 1,
            "per_class": per_class_stat,
        }

    # --- combined provenance sidecar ---
    git_info = get_git_info()
    source_paths = {
        _CLASS_LABELS[c]: str(
            Path(SignalClusteringData.FLUX_BASE) / str(c) / "flux.npy"
        )
        for c in (1, 2, 3, 4)
    }
    provenance = {
        "export_request_slug": "exploration-relationships-2d",
        "consumer": "selements-website",
        "producing_function": (
            "src.core.export.export_exploration_relationships_2d"
        ),
        "figures": {
            f: {
                "density_csv": f"{f}-2d.csv",
                "trend_csv": f"{f}-trend.csv",
                "caption": _FIG_CAPTIONS[f],
                **fig_stats[f],
            }
            for f in figs
        },
        "export_timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        "git": git_info,
        "source_data_paths": source_paths,
        "canonical_metrics": "scripts/eda_sherwood.py",
        "n_classes": 4,
        "n_sightlines_per_class": 16384,
        "activity_bin_grid": "50 linear wavelength bins (np.linspace(min,max,51))",
        "density_definition": (
            "count / class_total (fraction per 2-D bin; sums to 1 per class). "
            "Bin centres: geometric for log axes, midpoint for linear. Only "
            "non-empty bins are emitted."
        ),
        "axis_note": (
            "x_center/y_center are in LINEAR physical units; apply the stated "
            "x_scale/y_scale (log10 or linear) on the axis. EW/gap in Angstrom, "
            "line density in lines/Angstrom, depth = 1-F (dimensionless), "
            "activity = EW (Angstrom) integrated within a wavelength bin."
        ),
        "source_lineage": (
            "Sherwood simulation suite (Bolton+2017), z=0.3 snapshot, 60 cMpc/h "
            "box; one realization, fixed cosmology / UVB / thermal history"
        ),
        # Honesty caveat — PI-framing-checked (ship all 4; FIG3 approved as-is,
        # FIG1/2/4 approved-with-edits, FIG4 ships as a descriptive null). Use the
        # per-figure 'caption' strings above; this is the cross-figure summary.
        "honest_reporting_caveat": (
            "Descriptive Tier-1 relationships, computed fresh over all 16,384x4 = "
            "65,536 sightlines, one z=0.3 snapshot, NOT a classifier. Only ONE of "
            "the four requested premises survives full-data computation: "
            "ew-vs-depth (FIG3) holds; ew-vs-density (FIG1) is weak and SIGN-FLIPS "
            "negative for C4; gap-vs-density (FIG2) is near-definitional (mean gap "
            "= span/(n_lines-1)); activity-vs-density (FIG4) is OVERTURNED (no "
            "monotone relation; activity peaks at ~1 line/bin then falls). "
            "DISCLOSURE (binds FIG1/FIG2/FIG4): line detection is find_peaks(-flux) "
            "with NO prominence threshold, so 'line density' counts shallow, "
            "low-prominence flux minima — the spectra are noiseless (S/N=inf), so "
            "these are genuine sub-threshold features of the simulated flux field, "
            "not measurement noise — not only resolved absorption lines; this is "
            "also the likely mechanism behind FIG4's decrease. GUARDRAIL: do NOT say or "
            "imply Class 4 has weaker/shallower PER-LINE absorption (a retired "
            "overstatement); C4's lower TOTAL EW is driven by fewer lines, not "
            "shallower lines. Across all figures: Class 4 stands apart (sparser), "
            "Classes 1-3 overlap heavily."
        ),
    }
    provenance_path = out_dir / "relationships-2d.provenance.json"
    with open(provenance_path, "w") as fh:
        json.dump(provenance, fh, indent=2)

    return out_dir


# Recorded 10-fold stratified-CV accuracy from the baseline-RF study (raw spectra
# vs db8-level-6 DWT features, 4-class feedback classification). Source of truth:
# docs/feature/baseline-random-forest/baseline_rf_analysis.md:20-30 +
# experiments/baseline-random-forest/LEDGER.md:144-154 (D1 cross-checked against
# MLflow run 82d7421e .../metrics/mean_accuracy = 0.3232). Mean + 10-fold std;
# NO percentile CI was recorded. The original 10-fold training code was removed,
# so these recorded values are the authoritative source (exported, not re-derived).
# (variant, accuracy, std_10fold)
_RF_BASELINE_RECORDED: List[Tuple[str, float, float]] = [
    ("RF_Raw", 0.4514, 0.0023),
    ("RF_D1", 0.3232, 0.0031),
    ("RF_D2", 0.2946, 0.0031),
    ("RF_D3", 0.2745, 0.0043),
    ("RF_D4", 0.2334, 0.0028),
    ("RF_D5", 0.2154, 0.0021),
    ("RF_D6", 0.2094, 0.0031),
    ("RF_A6", 0.2177, 0.0015),
    ("RF_Concat", 0.3139, 0.0030),
]
_RF_RANDOM_BASELINE_4CLASS = 0.25  # reference line for a balanced 4-class draw

# Canonical DWT convention, single-sourced from src/core/transforms.py
# WaveletTransform (db8, level 6, periodization).
_DWT_WAVELET = "db8"
_DWT_LEVEL = 6
_DWT_MODE = "periodization"

# Recorded baseline-RF hyperparameters (MLflow run 82d7421e params). Used to
# reproduce the confusion matrices faithfully; the original 10-fold training code
# was removed, so this is a single 80/20 holdout with the same hyperparameters.
_RF_RECORDED_HP = dict(
    n_estimators=100,
    max_depth=25,
    max_features="sqrt",
    min_samples_leaf=50,
    min_samples_split=100,
    random_state=42,
    n_jobs=-1,
)
# Recorded 10-fold accuracy per variant, for the holdout-vs-record sanity check.
_RF_RECORDED_ACC = {
    "RF_Raw": 0.4514, "RF_D1": 0.3232, "RF_D2": 0.2946, "RF_D3": 0.2745,
    "RF_D4": 0.2334, "RF_D5": 0.2154, "RF_D6": 0.2094, "RF_A6": 0.2177,
    "RF_Concat": 0.3139,
}


def export_episode4_rf_baseline_summary(out_dir: Path) -> Path:
    """Export the 9-variant RF accuracy table (raw vs db8-L6 DWT), recorded values.

    The numbers are the recorded 10-fold CV results (the original training code was
    removed), transcribed verbatim from
    ``docs/feature/baseline-random-forest/baseline_rf_analysis.md`` + the
    baseline-RF LEDGER and cross-checked against MLflow. This is a retrieval, not a
    re-derivation. Writes a clean CSV + a git-stamped provenance sidecar.

    Columns: ``variant,accuracy,acc_std,acc_lo_1sd,acc_hi_1sd``. ``acc_lo/hi_1sd``
    are mean +/- one 10-fold std (a band), NOT bootstrap percentiles — no p16/p84
    CI was recorded; the column names say so to avoid implying percentiles.

    Args:
        out_dir: Landing directory (created if absent).

    Returns:
        Path to the written CSV (``baseline_summary.csv``).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / "baseline_summary.csv"
    header = ["variant", "accuracy", "acc_std", "acc_lo_1sd", "acc_hi_1sd"]
    with open(csv_path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        for variant, acc, std in _RF_BASELINE_RECORDED:
            writer.writerow([
                variant, repr(float(acc)), repr(float(std)),
                repr(float(acc - std)), repr(float(acc + std)),
            ])

    git_info = get_git_info()
    provenance = {
        "export_request_slug": "rf-dwt-baseline",
        "consumer": "selements-website",
        "producing_function": "src.core.export.export_episode4_rf_baseline_summary",
        "consumer_facing_filename": csv_path.name,
        "export_timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        "git": git_info,
        "recorded_source": (
            "docs/feature/baseline-random-forest/baseline_rf_analysis.md:20-30 ; "
            "experiments/baseline-random-forest/LEDGER.md:144-154 ; D1 cross-"
            "checked vs MLflow run 82d7421e (mean_accuracy=0.3232)"
        ),
        "methodology": (
            "10-fold stratified CV, 4-class (1=NoFeedback..4=WindStrongAGN). "
            "RandomForest hyperparameters per MLflow run 82d7421e (n_estimators="
            "100, max_depth=25, max_features=sqrt, min_samples_leaf=50, "
            "min_samples_split=100); exact random seed / fold assignment and the "
            "training code are NOT recorded (code removed) — values are exported "
            "from the record, not recomputed."
        ),
        "random_baseline_4class": _RF_RANDOM_BASELINE_4CLASS,
        "ci_note": (
            "acc_std is the 10-fold CV std dev; acc_lo/hi_1sd = mean +/- 1 std "
            "(a band). NO bootstrap p16/p84 percentile CI was recorded."
        ),
        "source_lineage": (
            "Sherwood simulation suite (Bolton+2017), z=0.3 snapshot, 60 cMpc/h "
            "box; one realization, fixed cosmology"
        ),
        # PI-framing-checked honesty caveat (verbatim PI-approved string).
        "honest_reporting_caveat": (
            "Accuracies are 10-fold stratified CV mean +/- std, exported VERBATIM "
            "from the recorded run (baseline_rf_analysis.md:20-30, LEDGER section "
            "5, MLflow-cross-checked for D1). The original training code "
            "(BaselineRFClassifier, train_rf_baseline.py) was removed during "
            "cleanup and is absent on disk — these numbers are NOT re-derivable "
            "from the current repo, so the record is reproduced rather than "
            "recomputed. Hyperparameters are in MLflow; random seed and fold "
            "assignment are not recorded. Raw wins (0.4514 = the 45% 4-class "
            "ceiling); db8 DWT does NOT help; accuracy drops monotonically across "
            "the detail bands D1->D6 (0.3232->0.2094), and A6 (0.2177) sits just "
            "ABOVE the D6 floor — so 'monotonic D1->A6' would overstate it (the "
            "source itself scopes monotonicity to D6). Random 4-class baseline = "
            "0.25; D4/D5/D6/A6 are at/near chance. The lo/hi columns are mean +/- "
            "1 SD bands, NOT percentile CIs (no p16/p84 recorded). This table is "
            "accuracy ONLY — it carries NO per-class recall, and does NOT itself "
            "substantiate any 'only Class 4 is identified' claim; that is a "
            "confusion-matrix reading (PNG-only, no recorded numbers) pending a "
            "separate re-run. Not a deployment benchmark."
        ),
    }
    with open(out_dir / "baseline_summary.provenance.json", "w") as fh:
        json.dump(provenance, fh, indent=2)

    return csv_path


def export_episode4_mean_energy_per_level(out_dir: Path) -> Path:
    """Export the mean DWT energy per level (D1..D6, A6) over all sightlines.

    Computed fresh with ``pywt.wavedec(db8, level 6, periodization)`` (the same
    convention as ``src/core/transforms.py`` WaveletTransform), pooled over all
    16,384 x 4 sightlines: per band, mean of squared coefficients. Spans ~7 orders
    of magnitude (justifies "RF is scale-invariant, no z-score scaling needed").

    Columns: ``level,mean_energy`` with ``level`` in ``{D1,D2,D3,D4,D5,D6,A6}``.

    Args:
        out_dir: Landing directory (created if absent).

    Returns:
        Path to the written CSV (``mean_energy_per_level.csv``).
    """
    import pywt

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    loader = SignalClusteringData()
    flux_per_class, _ = loader.load_flux_per_class()

    # Accumulate sum-of-squares and element count per band across classes.
    bands = ["D1", "D2", "D3", "D4", "D5", "D6", "A6"]
    sq_sum: Dict[str, float] = {b: 0.0 for b in bands}
    n_elem: Dict[str, int] = {b: 0 for b in bands}
    for class_id in (1, 2, 3, 4):
        X = np.asarray(flux_per_class[class_id - 1], dtype=np.float64)
        # coeffs = [cA6, cD6, cD5, cD4, cD3, cD2, cD1]
        coeffs = pywt.wavedec(X, _DWT_WAVELET, level=_DWT_LEVEL, mode=_DWT_MODE, axis=1)
        band_of = {
            "A6": coeffs[0], "D6": coeffs[1], "D5": coeffs[2], "D4": coeffs[3],
            "D3": coeffs[4], "D2": coeffs[5], "D1": coeffs[6],
        }
        for b in bands:
            c = band_of[b]
            sq_sum[b] += float(np.sum(c.astype(np.float64) ** 2))
            n_elem[b] += int(c.size)

    mean_energy = {b: sq_sum[b] / n_elem[b] for b in bands}

    csv_path = out_dir / "mean_energy_per_level.csv"
    with open(csv_path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["level", "mean_energy"])
        for b in bands:
            writer.writerow([b, repr(float(mean_energy[b]))])

    git_info = get_git_info()
    provenance = {
        "export_request_slug": "rf-dwt-baseline",
        "consumer": "selements-website",
        "producing_function": "src.core.export.export_episode4_mean_energy_per_level",
        "consumer_facing_filename": csv_path.name,
        "export_timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        "git": git_info,
        "source_data_paths": {
            _CLASS_LABELS[c]: str(
                Path(SignalClusteringData.FLUX_BASE) / str(c) / "flux.npy"
            )
            for c in (1, 2, 3, 4)
        },
        "method": (
            f"pywt.wavedec(wavelet={_DWT_WAVELET}, level={_DWT_LEVEL}, "
            f"mode={_DWT_MODE}, axis=1); per band mean of squared coefficients, "
            "pooled over all 16,384 x 4 sightlines. Matches src/core/transforms.py "
            "WaveletTransform convention."
        ),
        "mean_energy_per_level": mean_energy,
        # PI-framing-checked honesty caveat (verbatim PI-approved string).
        "honest_reporting_caveat": (
            "Per-level DWT energy, computed FRESH (pywt.wavedec, db8, L6, "
            "periodization), pooled over all 65,536 sightlines as the mean of "
            "squared coefficients per band. Energy spans ~10 orders of magnitude "
            "and rises monotonically D1->A6 (1.04e-08 -> 6.15e+01); A6 dominates. "
            "These are RAW (non-z-scored) coefficient energies. The baseline RF "
            "needs no per-feature scaling because tree splits are threshold-based "
            "and scale-invariant PER FEATURE — this is a property of RF, not a "
            "general 'no scaling needed.' Note the DOWNSTREAM clustering pipeline "
            "DOES apply a per-level z-score (CLAUDE.md domain convention) precisely "
            "so A6/D5/D6 don't dominate the distance metric; this energy table is "
            "the direct evidence for why that z-score is needed there. D1's small "
            "coefficients are LOW-energy fine-scale detail, NOT noise (the Sherwood "
            "data is noiseless)."
        ),
    }
    with open(out_dir / "mean_energy_per_level.provenance.json", "w") as fh:
        json.dump(provenance, fh, indent=2)

    return csv_path


def export_episode4_rf_confusion_matrices(out_dir: Path) -> Path:
    """Re-run all 9 RF variants and export row-normalized 4x4 confusion matrices.

    The original 10-fold training code was removed, so this is a fresh SINGLE
    80/20 stratified holdout (seed 42) with the RECORDED hyperparameters
    (:data:`_RF_RECORDED_HP`) — a faithful reproduction, NOT the exact original
    run. The 9 input representations (matching the recorded baseline study):
      - RF_Raw    : raw flux (2048-dim).
      - RF_D1..D6 : db8-L6 detail bands cD1 (1024) .. cD6 (32).
      - RF_A6     : db8-L6 approximation band cA6 (32-dim).
      - RF_Concat : hstack of all seven bands [D1..D6, A6] (2048-dim).

    Each variant's holdout accuracy is checked against the recorded 10-fold value
    (:data:`_RF_RECORDED_ACC`); the delta is recorded in the provenance and a large
    mismatch is flagged (honest-reporting), but matrices are still written.

    Writes one tidy CSV (all nine variants) + a provenance sidecar carrying the
    per-variant holdout accuracy, the recorded-vs-holdout delta, and the Class-4
    recall (which substantiates the "Class 4 is the most reliably identified
    class" reading — absent from disk until now).

    CSV columns: ``variant,true_class,true_label,pred_class,pred_label,fraction,count``.

    Args:
        out_dir: Landing directory (created if absent).

    Returns:
        Path to the written CSV (``confusion_matrices.csv``).
    """
    import pywt
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import confusion_matrix
    from sklearn.model_selection import train_test_split

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    loader = SignalClusteringData()
    flux_per_class, _ = loader.load_flux_per_class()
    X_raw = np.vstack(
        [np.asarray(flux_per_class[c - 1], dtype=np.float64) for c in (1, 2, 3, 4)]
    )  # shape: (65536, 2048)
    y = np.concatenate([
        np.full(flux_per_class[c - 1].shape[0], c, dtype=np.int64) for c in (1, 2, 3, 4)
    ])  # shape: (65536,)

    # db8-L6 bands: coeffs = [cA6, cD6, cD5, cD4, cD3, cD2, cD1].
    coeffs = pywt.wavedec(X_raw, _DWT_WAVELET, level=_DWT_LEVEL, mode=_DWT_MODE, axis=1)
    # All 9 RF input representations (raw, per-band detail D1..D6, approximation
    # A6, and the full concatenation), matching the recorded baseline study and
    # scripts/baseline/train_rf_baseline.py band dims. RF_Concat = hstack of all
    # seven bands in [D1..D6, A6] order (2048-dim).
    concat = np.ascontiguousarray(np.hstack([
        coeffs[6], coeffs[5], coeffs[4], coeffs[3], coeffs[2], coeffs[1], coeffs[0]
    ]))  # shape: (65536, 2048)
    features = {
        "RF_Raw": X_raw,
        "RF_D1": np.ascontiguousarray(coeffs[6]),  # cD1, (65536, 1024)
        "RF_D2": np.ascontiguousarray(coeffs[5]),  # cD2, (65536, 512)
        "RF_D3": np.ascontiguousarray(coeffs[4]),  # cD3, (65536, 256)
        "RF_D4": np.ascontiguousarray(coeffs[3]),  # cD4, (65536, 128)
        "RF_D5": np.ascontiguousarray(coeffs[2]),  # cD5, (65536, 64)
        "RF_D6": np.ascontiguousarray(coeffs[1]),  # cD6, (65536, 32)
        "RF_A6": np.ascontiguousarray(coeffs[0]),  # cA6, (65536, 32)
        "RF_Concat": concat,
    }

    labels = [1, 2, 3, 4]
    rows: List[Dict[str, object]] = []
    holdout_acc: Dict[str, float] = {}
    acc_delta: Dict[str, float] = {}
    class4_recall: Dict[str, float] = {}
    for variant, X in features.items():
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        clf = RandomForestClassifier(**_RF_RECORDED_HP)
        clf.fit(X_tr, y_tr)
        y_pred = clf.predict(X_te)
        acc = float((y_pred == y_te).mean())
        holdout_acc[variant] = acc
        acc_delta[variant] = round(acc - _RF_RECORDED_ACC[variant], 4)
        cm = confusion_matrix(y_te, y_pred, labels=labels).astype(np.float64)
        row_sums = cm.sum(axis=1, keepdims=True)
        cm_norm = np.divide(cm, row_sums, out=np.zeros_like(cm), where=row_sums > 0)
        class4_recall[variant] = round(float(cm_norm[3, 3]), 4)
        for i, tc in enumerate(labels):
            for j, pc in enumerate(labels):
                rows.append({
                    "variant": variant,
                    "true_class": tc,
                    "true_label": _CLASS_LABELS[tc],
                    "pred_class": pc,
                    "pred_label": _CLASS_LABELS[pc],
                    "fraction": float(cm_norm[i, j]),
                    "count": int(cm[i, j]),
                })

    csv_path = out_dir / "confusion_matrices.csv"
    header = [
        "variant", "true_class", "true_label", "pred_class", "pred_label",
        "fraction", "count",
    ]
    with open(csv_path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        for r in rows:
            writer.writerow([
                str(r["variant"]), int(r["true_class"]), str(r["true_label"]),
                int(r["pred_class"]), str(r["pred_label"]),
                repr(float(r["fraction"])), int(r["count"]),
            ])

    # Honest sanity: flag any holdout-vs-recorded gap > 0.05.
    large_gaps = {v: d for v, d in acc_delta.items() if abs(d) > 0.05}

    git_info = get_git_info()
    provenance = {
        "export_request_slug": "rf-dwt-baseline",
        "consumer": "selements-website",
        "producing_function": "src.core.export.export_episode4_rf_confusion_matrices",
        "consumer_facing_filename": csv_path.name,
        "export_timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        "git": git_info,
        "source_data_paths": {
            _CLASS_LABELS[c]: str(
                Path(SignalClusteringData.FLUX_BASE) / str(c) / "flux.npy"
            )
            for c in (1, 2, 3, 4)
        },
        "methodology": (
            "FRESH single 80/20 stratified holdout (random_state=42) with the "
            f"recorded hyperparameters {dict(_RF_RECORDED_HP)}. The original "
            "10-fold training code was removed, so this is a faithful "
            "reproduction, NOT the exact original run. Features: RF_Raw=raw flux "
            "(2048-dim); RF_D1=db8-L6 cD1 (1024-dim); RF_D6=db8-L6 cD6 (32-dim). "
            "Confusion matrices are row-normalized (each true-class row sums to 1)."
        ),
        "holdout_accuracy": {v: round(a, 4) for v, a in holdout_acc.items()},
        "recorded_10fold_accuracy": _RF_RECORDED_ACC,
        "holdout_minus_recorded": acc_delta,
        "holdout_vs_record_flag": (
            "all variants within 0.05 of the recorded 10-fold accuracy"
            if not large_gaps
            else f"WARNING: large holdout-vs-recorded gap(s): {large_gaps} — "
            "feature construction or hyperparameters may not match the original"
        ),
        "class4_recall": class4_recall,
        # PI-framing-checked (verbatim PI-approved 9-variant strings). The earlier
        # "1<->2 swap" wording was an over-claim — true-1/true-3 are two-way splits.
        "figure_caption": (
            "Confusion-matrix drift across all 9 RF input representations (single "
            "80/20 holdout, seed 42; reproduces recorded 10-fold within 0.021). "
            "Class 4 (WindStrongAGN) is the only consistently self-identified class "
            "(recall 0.63-0.94, peaking at db8-D1/D2/Concat ~0.93-0.94). Classes "
            "1/2/3 collapse in every variant: into a Class-1 sink for raw flux "
            "(true-2/true-3 -> predicted-1), and into the predicted-1/predicted-2 "
            "columns for the wavelet bands (clean edge: true-2->predicted-1 ~0.5; "
            "true-1 and true-3 split two ways - not a pairwise swap). For D4-A6 the "
            "1/2/3 block sits at the 0.25 random baseline (argmax destinations are "
            "near-ties). Wavelets do not improve overall separability - every "
            "variant is below raw-flux accuracy (0.45) - though D1/D2 sharpen "
            "Class-4 self-recognition specifically."
        ),
        "honest_reporting_caveat": (
            "Row-normalized 4-class confusion matrices for all 9 RF input "
            "representations (raw flux, db8-L6 detail bands D1..D6, approximation "
            "A6, and the full concatenation), from a FRESH single 80/20 stratified "
            "holdout (seed 42) with the recorded hyperparameters; the original "
            "10-fold training code was removed, so this is a faithful reproduction, "
            "not the exact original run (holdout accuracy within 0.021 of the "
            "recorded 10-fold for all 9; per-variant deltas in the provenance "
            "JSON). Class 4 (WindStrongAGN) is the only true class that "
            "self-identifies in every variant (recall 0.63-0.94; strongest for "
            "RF_D1 ~ RF_Concat ~0.94 and RF_D2 0.93, weakest for RF_D6 0.63) and "
            "the only true class whose argmax prediction is itself in all nine. "
            "Classes 1/2/3 never self-recognize (diagonal mostly <0.16, falling to "
            "~0.06-0.11 for D4/D5/D6/A6); they collapse, and the collapse geometry "
            "is VARIANT-SPECIFIC. In RF_Raw the collapse is a Class-1 sink: true-2 "
            "(86%) and true-3 (78%) are predicted as Class 1, so Class 1's high "
            "0.84 recall is a majority-attractor artifact, not a detection - "
            "Classes 2/3 are unrecovered (recall 0.04/0.05). In the wavelet-band "
            "variants the collapse is NOT a clean pairwise 1<->2 swap: the "
            "true-1/2/3 mass piles into the predicted-1 and predicted-2 columns "
            "while predicted-3/predicted-4 go nearly empty for those rows. The one "
            "consistent clean edge is true-2->predicted-1 (~0.50-0.53 in "
            "D1/D2/Concat); true-1 splits across predicted-2 (~0.42-0.50) AND "
            "predicted-3 (~0.33-0.37), and true-3 dissolves across predicted-1 and "
            "predicted-2. Do NOT re-narrate the RF_Raw Class-1 sink onto these "
            "variants, and do NOT describe them as a 1<->2 swap - the true-1 and "
            "true-3 rows are two-way splits, not a swap. For RF_D4/D5/D6/A6 overall "
            "accuracy sits at/near the random 4-class baseline (0.25), so the 1/2/3 "
            "structure there is near-chance and diffuse - the per-row argmax "
            "destinations are near-ties (e.g. RF_A6 true-1 pred-2 0.326 vs pred-3 "
            "0.322; true-3 destinations split within ~1 point) and must NOT be read "
            "as stable arrows; do NOT overclaim 1/2/3 separation in those rows. "
            "This 1/2/3 indistinguishability is the supervised shadow of the "
            "unsupervised EW-distribution overlap: per-line EW overlaps >=97% "
            "across all 4 classes and the real separator is line density (Class 4 "
            "~28% fewer lines; eda-sherwood LEDGER section 5). db8 detail-band "
            "coefficients (esp. D1) are low-energy, not noise - the data is "
            "noiseless. Descriptive, one z=0.3 snapshot; NOT a deployed classifier "
            "and not a per-sightline detection claim."
        ),
    }
    with open(out_dir / "confusion_matrices.provenance.json", "w") as fh:
        json.dump(provenance, fh, indent=2)

    return csv_path


def export_episode4_sample_sightline(
    out_dir: Path,
    class_id: int = 1,
    sightline_idx: int = 0,
) -> Path:
    """Export one sample sightline's flux F and absorption A=1-F (frame 12-2).

    Illustrative before/after panels for the DWT pipeline. Descriptive single
    sightline (NOT claim-bearing). Reuses the same representative sightline as the
    primer export (class 1, idx 0 = the EDA tier1_representative_spectra_2x2.png
    top-left panel) for visual consistency across episodes.

    Columns: ``pixel,wavelength_angstrom,flux,absorption`` (absorption = 1 - flux),
    full float64.

    Args:
        out_dir: Landing directory (created if absent).
        class_id: Physics class id in {1,2,3,4}. Default 1 (NoFeedback).
        sightline_idx: Row index into the per-class flux block. Default 0.

    Returns:
        Path to the written CSV (``sample_sightline.csv``).
    """
    if class_id not in _CLASS_LABELS:
        raise ValueError(f"class_id must be in {sorted(_CLASS_LABELS)}, got {class_id}")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    loader = SignalClusteringData()
    flux_per_class, _ = loader.load_flux_per_class()
    block = flux_per_class[class_id - 1]
    if not (0 <= sightline_idx < block.shape[0]):
        raise ValueError(
            f"sightline_idx {sightline_idx} out of range [0, {block.shape[0]})"
        )
    flux = np.asarray(block[sightline_idx], dtype=np.float64)
    _validate_spectrum(flux, name=f"class{class_id}.sightline{sightline_idx}.flux")
    absorption = 1.0 - flux
    wavelength = _load_wavelength_axis(class_id)
    pixel = np.arange(_N_PIXELS, dtype=np.int64)

    csv_path = out_dir / "sample_sightline.csv"
    with open(csv_path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["pixel", "wavelength_angstrom", "flux", "absorption"])
        for i in range(_N_PIXELS):
            writer.writerow([
                int(pixel[i]), repr(float(wavelength[i])),
                repr(float(flux[i])), repr(float(absorption[i])),
            ])

    git_info = get_git_info()
    provenance = {
        "export_request_slug": "rf-dwt-baseline",
        "consumer": "selements-website",
        "producing_function": "src.core.export.export_episode4_sample_sightline",
        "consumer_facing_filename": csv_path.name,
        "export_timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        "git": git_info,
        "source_data_path": str(
            Path(SignalClusteringData.FLUX_BASE) / str(class_id) / "flux.npy"
        ),
        "class_id": class_id,
        "class_label": _CLASS_LABELS[class_id],
        "sightline_idx": sightline_idx,
        "columns": (
            "pixel; wavelength_angstrom; flux F (normalized transmitted, [0,1]); "
            "absorption A = 1 - F"
        ),
        "source_lineage": (
            "Sherwood simulation suite (Bolton+2017), z=0.3 snapshot, 60 cMpc/h "
            "box; noiseless (S/N=inf)"
        ),
        "honest_reporting_caveat": (
            "Illustrative single sightline for the 12-2 before/after panels. "
            "Selection = first stored sightline of class 1 (matches "
            "results/eda/tier1_representative_spectra_2x2.png top-left and the "
            "primer-synthetic-spectrum export), NOT a statistically-central / "
            "'typical' spectrum. Noiseless data."
        ),
    }
    with open(out_dir / "sample_sightline.provenance.json", "w") as fh:
        json.dump(provenance, fh, indent=2)

    return csv_path
