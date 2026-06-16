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
    _N_PIXELS,
    _validate_spectrum,
    export_primer_synthetic_spectrum,
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
