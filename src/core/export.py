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
        "prominence threshold), so it includes shallow continuum-noise dips, not "
        "only physical absorbers."
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
        "more detected 'lines' contain proportionally more shallow noise dips, "
        "which lowers mean per-line activity — the decrease is consistent with a "
        "detection-threshold effect, not a physical anti-correlation. Reported as "
        "a descriptive null relative to the naive expectation."
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
            "with NO prominence threshold, so 'line density' counts shallow "
            "continuum-noise dips, not only physical absorbers — this is also the "
            "likely mechanism behind FIG4's decrease. GUARDRAIL: do NOT say or "
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
