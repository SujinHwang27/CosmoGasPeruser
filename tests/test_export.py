"""Tests mirroring src/core/export.py.

Synthetic-fixture tests cover the CSV-writing / validation logic on a
``(2048,)`` array. The single real-data end-to-end test is marked ``slow``.
"""

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from src.core.export import (
    _N_KBINS,
    _N_PIXELS,
    _validate_spectrum,
    _write_pk_tidy_csv,
    export_episode4_mean_energy_per_level,
    export_episode4_rf_baseline_summary,
    export_episode4_rf_confusion_matrices,
    export_episode4_sample_sightline,
    export_exploration_line_density_per_class,
    export_exploration_local_ew_dist_per_class,
    export_exploration_pk_mean_per_class,
    export_exploration_relationships_2d,
    export_primer_synthetic_spectrum,
    export_signalclustering_v2_sep_norms,
    write_spectrum_csv,
)


def _synthetic_axes(seed: int = 7):
    rng = np.random.default_rng(seed)
    flux = rng.uniform(0.0, 1.0, size=_N_PIXELS).astype(np.float64)
    pixel = np.arange(_N_PIXELS, dtype=np.int64)
    wavelength = np.linspace(1580.0, 1609.0, _N_PIXELS).astype(np.float64)
    velocity = np.linspace(0.0, 2.6365 * (_N_PIXELS - 1), _N_PIXELS).astype(np.float64)
    return pixel, wavelength, velocity, flux


def test_validate_spectrum_accepts_in_range():
    _, _, _, flux = _synthetic_axes()
    _validate_spectrum(flux)  # should not raise


def test_validate_spectrum_rejects_wrong_shape():
    with pytest.raises(ValueError):
        _validate_spectrum(np.zeros(10, dtype=np.float64))


def test_validate_spectrum_rejects_nan():
    _, _, _, flux = _synthetic_axes()
    flux[0] = np.nan
    with pytest.raises(ValueError):
        _validate_spectrum(flux)


def test_validate_spectrum_rejects_out_of_range_high():
    _, _, _, flux = _synthetic_axes()
    flux[5] = 1.5
    with pytest.raises(ValueError):
        _validate_spectrum(flux)


def test_validate_spectrum_rejects_out_of_range_low():
    _, _, _, flux = _synthetic_axes()
    flux[5] = -0.1
    with pytest.raises(ValueError):
        _validate_spectrum(flux)


def test_write_spectrum_csv_shape_and_columns(tmp_path: Path):
    pixel, wavelength, velocity, flux = _synthetic_axes()
    out = tmp_path / "synthetic-spectrum.example.csv"
    written = write_spectrum_csv(out, pixel, wavelength, velocity, flux)
    assert written == out
    with open(out, newline="") as fh:
        rows = list(csv.reader(fh))
    assert rows[0] == ["pixel", "wavelength_angstrom", "velocity_kms", "flux"]
    assert len(rows) == _N_PIXELS + 1  # header + data
    # No comment lines (every data row parses to 4 fields).
    assert all(len(r) == 4 for r in rows)


def test_write_spectrum_csv_full_precision_roundtrip(tmp_path: Path):
    pixel, wavelength, velocity, flux = _synthetic_axes()
    out = tmp_path / "s.csv"
    write_spectrum_csv(out, pixel, wavelength, velocity, flux)
    arr = np.genfromtxt(out, delimiter=",", names=True)
    np.testing.assert_array_equal(arr["flux"], flux)


def test_write_spectrum_csv_rejects_out_of_range_flux(tmp_path: Path):
    pixel, wavelength, velocity, flux = _synthetic_axes()
    flux[0] = 2.0
    with pytest.raises(ValueError):
        write_spectrum_csv(tmp_path / "bad.csv", pixel, wavelength, velocity, flux)


def test_write_spectrum_csv_rejects_axis_length_mismatch(tmp_path: Path):
    pixel, wavelength, velocity, flux = _synthetic_axes()
    short_vel = velocity[:-1]
    with pytest.raises(ValueError):
        write_spectrum_csv(tmp_path / "bad.csv", pixel, wavelength, short_vel, flux)


def _pk_rows(n_classes: int = 4, n_kbins: int = _N_KBINS):
    """Synthetic tidy P_F(k) rows for the writer test (no real data)."""
    labels = {1: "NoFeedback", 2: "StellarWind", 3: "WindAGN", 4: "WindStrongAGN"}
    rng = np.random.default_rng(11)
    rows = []
    for c in range(1, n_classes + 1):
        for j in range(n_kbins):
            empty = j == 1  # mimic an empty low-k bin
            rows.append({
                "feedback_class": c,
                "class_name": labels[c],
                "k_bin_idx": j,
                "k_center_s_per_km": float(10 ** (-3 + j * 0.1)),
                "pk_mean": 0.0 if empty else float(rng.uniform(0.05, 0.4)),
                "pk_sem": 0.0 if empty else float(rng.uniform(1e-3, 1e-2)),
                "n_modes_in_bin": 0 if empty else int(rng.integers(1, 80)),
                "n_sightlines": 16384,
            })
    return rows


def test_write_pk_tidy_csv_columns_and_rows(tmp_path: Path):
    rows = _pk_rows()
    out = tmp_path / "pk-mean-per-class.csv"
    written = _write_pk_tidy_csv(out, rows)
    assert written == out
    with open(out, newline="") as fh:
        parsed = list(csv.reader(fh))
    assert parsed[0] == [
        "feedback_class", "class_name", "k_bin_idx", "k_center_s_per_km",
        "pk_mean", "pk_sem", "n_modes_in_bin", "n_sightlines",
    ]
    assert len(parsed) == 4 * _N_KBINS + 1  # header + 4 classes x kbins
    # No comment lines: every row parses to exactly 8 fields.
    assert all(len(r) == 8 for r in parsed)


def test_write_pk_tidy_csv_full_precision_roundtrip(tmp_path: Path):
    rows = _pk_rows()
    out = tmp_path / "pk.csv"
    _write_pk_tidy_csv(out, rows)
    arr = np.genfromtxt(out, delimiter=",", names=True)
    expected = np.array([r["pk_mean"] for r in rows])
    np.testing.assert_array_equal(arr["pk_mean"], expected)


def test_pk_empty_bins_have_zero_and_flagged(tmp_path: Path):
    rows = _pk_rows()
    out = tmp_path / "pk.csv"
    _write_pk_tidy_csv(out, rows)
    parsed = list(csv.DictReader(open(out)))
    for r in parsed:
        if int(r["n_modes_in_bin"]) == 0:
            assert float(r["pk_mean"]) == 0.0
            assert float(r["pk_sem"]) == 0.0


@pytest.mark.slow
def test_export_exploration_pk_mean_per_class_real_data(tmp_path: Path):
    csv_path = export_exploration_pk_mean_per_class(tmp_path)
    assert csv_path.exists()
    prov_path = csv_path.with_name("pk-mean-per-class.provenance.json")
    assert prov_path.exists()

    parsed = list(csv.DictReader(open(csv_path)))
    assert len(parsed) == 4 * _N_KBINS  # 4 classes x 20 k-bins
    # k-grid identical across classes; non-empty bins strictly positive.
    for r in parsed:
        if int(r["n_modes_in_bin"]) > 0:
            assert float(r["pk_mean"]) > 0.0
        else:
            assert float(r["pk_mean"]) == 0.0

    with open(prov_path) as fh:
        prov = json.load(fh)
    assert prov["git"]["commit"] != "unknown"
    assert prov["canonical_transform"] == "src.core.transforms.FluxPowerSpectrum"
    # Verb-ceiling must be present and must NOT over-claim.
    vc = prov["verb_ceiling"].lower()
    assert "weak" in vc and "not a feedback classifier" in vc


@pytest.mark.slow
def test_export_exploration_local_ew_dist_real_data(tmp_path: Path):
    csv_path = export_exploration_local_ew_dist_per_class(tmp_path)
    assert csv_path.exists()
    summary_path = csv_path.with_name("local-ew-summary-per-class.csv")
    prov_path = csv_path.with_name("local-ew-dist-per-class.provenance.json")
    assert summary_path.exists() and prov_path.exists()

    dist = list(csv.DictReader(open(csv_path)))
    # 4 classes x 39 bins (40 log2 edges).
    assert len(dist) == 4 * 39
    # Per-class density sums to ~1 (fraction-per-bin).
    sums = {}
    for r in dist:
        sums[r["class_name"]] = sums.get(r["class_name"], 0.0) + float(r["density"])
    for v in sums.values():
        assert v == pytest.approx(1.0, abs=1e-4)

    summ = list(csv.DictReader(open(summary_path)))
    assert len(summ) == 4
    # Line-density separation is the real signal: C4 has clearly fewer lines.
    n = {r["class_name"]: int(r["n_lines"]) for r in summ}
    assert n["WindStrongAGN"] < 0.85 * n["NoFeedback"]

    with open(prov_path) as fh:
        prov = json.load(fh)
    # The honesty caveat must NOT carry the overstated "Class 4 separates" framing.
    cav = prov["honest_reporting_caveat"].lower()
    assert "overlap almost completely" in cav
    assert "line density" in cav
    # All four TV-distances vs NoFeedback are small (distributions overlap).
    assert all(v <= 0.05 for v in prov["tv_distance_vs_nofeedback"].values())


@pytest.mark.slow
def test_export_exploration_line_density_real_data(tmp_path: Path):
    csv_path = export_exploration_line_density_per_class(tmp_path)
    assert csv_path.exists()
    summary_path = csv_path.with_name("line-density-summary-per-class.csv")
    assert summary_path.exists()

    summ = {r["class_name"]: r for r in csv.DictReader(open(summary_path))}
    assert len(summ) == 4
    # The honest separation: C4 line density clearly below C1/C2 (~1.30 vs ~1.80).
    c4 = float(summ["WindStrongAGN"]["mean_lines_per_angstrom"])
    c1 = float(summ["NoFeedback"]["mean_lines_per_angstrom"])
    assert c4 == pytest.approx(1.301, abs=0.02)
    assert c1 == pytest.approx(1.80, abs=0.02)
    assert c4 < 0.8 * c1

    dist = list(csv.DictReader(open(csv_path)))
    sums = {}
    for r in dist:
        sums[r["class_name"]] = sums.get(r["class_name"], 0.0) + float(r["density"])
    for v in sums.values():
        assert v == pytest.approx(1.0, abs=1e-4)


def test_export_episode4_rf_baseline_summary(tmp_path: Path):
    # Recorded values — no real data needed.
    csv_path = export_episode4_rf_baseline_summary(tmp_path)
    rows = {r["variant"]: r for r in csv.DictReader(open(csv_path))}
    assert len(rows) == 9
    assert float(rows["RF_Raw"]["accuracy"]) == pytest.approx(0.4514)
    # A6 sits ABOVE the D6 floor (the "monotonic D1->A6" overstatement guard).
    assert float(rows["RF_A6"]["accuracy"]) > float(rows["RF_D6"]["accuracy"])
    # Detail bands monotone decreasing D1..D6.
    d = [float(rows[f"RF_D{i}"]["accuracy"]) for i in range(1, 7)]
    assert all(d[i] > d[i + 1] for i in range(len(d) - 1))
    with open(csv_path.with_name("baseline_summary.provenance.json")) as fh:
        prov = json.load(fh)
    cav = prov["honest_reporting_caveat"].lower()
    assert "no per-class recall" in cav  # disowns the "only Class 4" claim
    assert "reproduced rather than recomputed" in cav


@pytest.mark.slow
def test_export_signalclustering_v2_sep_norms_real_data(tmp_path: Path):
    csv_path = export_signalclustering_v2_sep_norms(tmp_path)
    rows = list(csv.DictReader(open(csv_path)))
    assert len(rows) == 16384
    assert list(rows[0].keys()) == ["sightline_idx", "wavelet_l2_norm", "raw_l2_norm"]
    # All norms finite and non-negative.
    for r in rows[:50]:
        assert float(r["wavelet_l2_norm"]) >= 0.0
        assert float(r["raw_l2_norm"]) >= 0.0

    summ = {r["run"]: r for r in csv.DictReader(open(csv_path.with_name("sep-norms-summary.csv")))}
    assert set(summ) == {"wavelet", "raw"}
    # Sanity anchor: wavelet median ~2.04 > raw median ~1.62.
    assert float(summ["wavelet"]["median"]) == pytest.approx(2.04, abs=0.02)
    assert float(summ["raw"]["median"]) == pytest.approx(1.62, abs=0.02)

    with open(csv_path.with_name("sep-norms.provenance.json")) as fh:
        prov = json.load(fh)
    # md5 pin to v0.4-clustering-v2 is verified by the export (would raise otherwise).
    assert prov["source_md5_verified"]["wavelet"] == "b7160641e12650c0e536cb510690f12d"
    # Honesty: caveat carries the scale-artifact + decoupling, not "wavelet better".
    cav = prov["honest_reporting_caveat"].lower()
    assert "scale caveat" in cav and "z-scored" in cav
    assert "decouple" in cav


@pytest.mark.slow
def test_export_episode4_sample_sightline_real_data(tmp_path: Path):
    csv_path = export_episode4_sample_sightline(tmp_path)
    rows = list(csv.DictReader(open(csv_path)))
    assert len(rows) == _N_PIXELS
    assert list(rows[0].keys()) == ["pixel", "wavelength_angstrom", "flux", "absorption"]
    for r in rows:
        f = float(r["flux"])
        assert 0.0 <= f <= 1.0
        # absorption = 1 - flux exactly.
        assert float(r["absorption"]) == pytest.approx(1.0 - f, abs=1e-12)


@pytest.mark.slow
def test_export_episode4_rf_confusion_matrices_real_data(tmp_path: Path):
    csv_path = export_episode4_rf_confusion_matrices(tmp_path)
    rows = list(csv.DictReader(open(csv_path)))
    # All 9 variants x 4x4 cells.
    assert len({r["variant"] for r in rows}) == 9
    assert len(rows) == 9 * 16
    # Each true-class row is normalized (fractions sum to ~1 per variant,true_class).
    sums = {}
    for r in rows:
        k = (r["variant"], r["true_class"])
        sums[k] = sums.get(k, 0.0) + float(r["fraction"])
    for v in sums.values():
        assert v == pytest.approx(1.0, abs=1e-6)

    with open(csv_path.with_name("confusion_matrices.provenance.json")) as fh:
        prov = json.load(fh)
    # Faithful reproduction: holdout within 0.05 of recorded for all 9 variants.
    assert len(prov["holdout_minus_recorded"]) == 9
    for v, d in prov["holdout_minus_recorded"].items():
        assert abs(d) < 0.05
    # Class 4 is the consistent standout across all 9 variants (recall > chance).
    assert all(r > 0.6 for r in prov["class4_recall"].values())
    assert prov["class4_recall"]["RF_D1"] > 0.9
    # Honesty: caveat keeps the Class-1 sink AND the swap-overclaim correction.
    cav = prov["honest_reporting_caveat"].lower()
    assert "sink" in cav and "not a swap" in cav
    assert "confuse heavily with each other" not in cav


@pytest.mark.slow
def test_export_episode4_mean_energy_per_level_real_data(tmp_path: Path):
    csv_path = export_episode4_mean_energy_per_level(tmp_path)
    rows = {r["level"]: float(r["mean_energy"]) for r in csv.DictReader(open(csv_path))}
    assert set(rows) == {"D1", "D2", "D3", "D4", "D5", "D6", "A6"}
    # Energy rises monotonically D1 -> A6, spanning many orders of magnitude.
    order = ["D1", "D2", "D3", "D4", "D5", "D6", "A6"]
    vals = [rows[k] for k in order]
    assert all(vals[i] < vals[i + 1] for i in range(len(vals) - 1))
    assert rows["A6"] / rows["D1"] > 1e6  # ~10 orders of magnitude
    with open(csv_path.with_name("mean_energy_per_level.provenance.json")) as fh:
        prov = json.load(fh)
    # Must NOT carry the retired "no scaling needed" overclaim unqualified.
    assert "per feature" in prov["honest_reporting_caveat"].lower()


@pytest.mark.slow
def test_export_exploration_relationships_2d_real_data(tmp_path: Path):
    out_dir = export_exploration_relationships_2d(tmp_path)
    figs = ["ew-vs-density", "gap-vs-density", "ew-vs-depth", "activity-vs-density"]
    for f in figs:
        assert (out_dir / f"{f}-2d.csv").exists()
        assert (out_dir / f"{f}-trend.csv").exists()
        rows = list(csv.DictReader(open(out_dir / f"{f}-2d.csv")))
        # per-class density sums to ~1
        sums = {}
        for r in rows:
            sums[r["class_name"]] = sums.get(r["class_name"], 0.0) + float(r["density"])
        assert len(sums) == 4
        for v in sums.values():
            assert v == pytest.approx(1.0, abs=1e-4)

    with open(out_dir / "relationships-2d.provenance.json") as fh:
        prov = json.load(fh)
    # Every figure ships a PI-approved caption.
    for f in figs:
        assert prov["figures"][f]["caption"]
    # FIG1: C4 log-log correlation is sign-flipped negative (not positive).
    assert prov["figures"]["ew-vs-density"]["per_class"]["WindStrongAGN"]["loglog_pearson_r"] < 0
    # FIG4 overturned: mean activity decreases from k=1 to k=3 (non-monotone).
    a = prov["figures"]["activity-vs-density"]["per_class"]["NoFeedback"]["mean_activity_by_linecount"]
    assert a["1"] > a["2"] > a["3"]


@pytest.mark.slow
def test_export_primer_synthetic_spectrum_real_data(tmp_path: Path):
    csv_path = export_primer_synthetic_spectrum(tmp_path, class_id=1, sightline_idx=0)
    assert csv_path.exists()
    prov_path = csv_path.with_name("synthetic-spectrum.example.provenance.json")
    assert prov_path.exists()

    with open(csv_path, newline="") as fh:
        rows = list(csv.reader(fh))
    assert len(rows) == _N_PIXELS + 1
    flux = np.array([float(r[3]) for r in rows[1:]])
    assert np.all((flux >= 0.0) & (flux <= 1.0))
    assert float(flux.min()) == pytest.approx(0.5454615502295661, abs=1e-9)
    assert float(flux.max()) == pytest.approx(0.999823923791532, abs=1e-9)
    assert float(flux.mean()) == pytest.approx(0.9831231593231671, abs=1e-9)

    with open(prov_path) as fh:
        prov = json.load(fh)
    assert prov["class_label"] == "NoFeedback"
    assert prov["git"]["commit"] != "unknown"
