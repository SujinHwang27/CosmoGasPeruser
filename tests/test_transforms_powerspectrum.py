"""Tests for src.core.transforms.FluxPowerSpectrum."""

import numpy as np
import pytest

from src.core.transforms import FluxPowerSpectrum


N_PIXELS = 2048
N_KBINS = 20
DELTA_V = 2.6365  # Sherwood z=0.3 nominal pixel velocity, km/s


@pytest.fixture
def synthetic_flux():
    """10 sightlines, 2048 pixels, F in [0, 1]-ish, float64. Seeded for determinism."""
    rng = np.random.default_rng(123)
    # Build a transmitted-flux-like array: noisy continuum near 1.0 with weak troughs.
    base = 0.85 + 0.10 * rng.standard_normal((10, N_PIXELS))
    # Inject a few absorption dips so delta_F is not pure noise.
    base[:, 100:120] *= 0.3
    base[:, 1000:1010] *= 0.2
    return np.clip(base, 0.0, 1.0).astype(np.float64)


class TestFluxPowerSpectrumShape:
    def test_per_sightline_output_shape(self, synthetic_flux):
        t = FluxPowerSpectrum(norm="per_sightline", n_kbins=N_KBINS, delta_v=DELTA_V)
        Z = t.fit_transform(synthetic_flux)
        assert Z.shape == (10, N_KBINS)

    def test_global_output_shape_drops_k0(self, synthetic_flux):
        t = FluxPowerSpectrum(norm="global", n_kbins=N_KBINS, delta_v=DELTA_V)
        Z = t.fit_transform(synthetic_flux)
        # 'global' drops the lowest log-bin per docstring -> n_kbins - 1.
        assert Z.shape == (10, N_KBINS - 1)
        assert t.dropped_k0_ is True

    def test_dtype_float64(self, synthetic_flux):
        t = FluxPowerSpectrum(norm="per_sightline", n_kbins=N_KBINS, delta_v=DELTA_V)
        Z = t.fit_transform(synthetic_flux)
        assert Z.dtype == np.float64


class TestFluxPowerSpectrumNumerics:
    def test_no_nan_or_inf_per_sightline(self, synthetic_flux):
        t = FluxPowerSpectrum(norm="per_sightline", n_kbins=N_KBINS, delta_v=DELTA_V)
        Z = t.fit_transform(synthetic_flux)
        assert np.all(np.isfinite(Z))

    def test_no_nan_or_inf_global(self, synthetic_flux):
        t = FluxPowerSpectrum(norm="global", n_kbins=N_KBINS, delta_v=DELTA_V)
        Z = t.fit_transform(synthetic_flux)
        assert np.all(np.isfinite(Z))

    def test_clip_underflow_handled(self):
        rng = np.random.default_rng(0)
        X = rng.uniform(0.5, 1.0, size=(5, N_PIXELS)).astype(np.float64)
        # Inject ~1e-16 underflow negatives.
        X[:, 50] = -1e-16
        X[:, 1000] = -5e-17
        t = FluxPowerSpectrum(norm="per_sightline", n_kbins=N_KBINS, delta_v=DELTA_V)
        Z = t.fit_transform(X)
        assert np.all(np.isfinite(Z))

    def test_deterministic_two_calls_identical(self, synthetic_flux):
        t1 = FluxPowerSpectrum(norm="per_sightline", n_kbins=N_KBINS, delta_v=DELTA_V)
        t2 = FluxPowerSpectrum(norm="per_sightline", n_kbins=N_KBINS, delta_v=DELTA_V)
        Z1 = t1.fit_transform(synthetic_flux)
        Z2 = t2.fit_transform(synthetic_flux)
        np.testing.assert_array_equal(Z1, Z2)


class TestFluxPowerSpectrumPhysicalK:
    def test_k_axis_spans_expected_range(self, synthetic_flux):
        t = FluxPowerSpectrum(norm="per_sightline", n_kbins=N_KBINS, delta_v=DELTA_V)
        _ = t.fit_transform(synthetic_flux)
        # Nyquist ~ pi / DELTA_V s/km.
        nyquist = np.pi / DELTA_V
        assert t.k_raw_[-1] == pytest.approx(nyquist, rel=1e-6)
        assert t.k_raw_[0] == 0.0
        # Bin centers all positive and within (k_min_pos, nyquist).
        assert (t.k_bin_centers_ > 0).all()
        assert (t.k_bin_centers_ <= nyquist).all()

    def test_sine_input_lights_up_correct_kbin(self):
        """A pure sine of known wavenumber should put dominant power at the bin
        whose log-edge interval contains that k."""
        # Choose a k in the middle of the log axis, away from Nyquist and DC.
        # Sine angular frequency: x positions are pixel * DELTA_V (km/s).
        # delta_F = sin(k_true * v). Wrap into F = mean*(1 + small*delta_F) so the
        # 'per_sightline' normalization recovers the same sine for delta_F.
        N = N_PIXELS
        v = np.arange(N) * DELTA_V
        k_true = 0.1  # s/km, sits comfortably mid-range (< Nyquist ~ 1.19).
        delta = 0.05 * np.sin(k_true * v)
        F = 0.5 * (1.0 + delta)
        F = F[None, :].repeat(3, axis=0).astype(np.float64)  # (3, N)
        t = FluxPowerSpectrum(norm="per_sightline", n_kbins=N_KBINS, delta_v=DELTA_V)
        Z = t.fit_transform(F)
        # Find which bin contains k_true.
        edges = t.k_bin_edges_
        target_bin = int(np.searchsorted(edges, k_true, side="right") - 1)
        # 'per_sightline' returns n_kbins columns indexed 0..n_kbins-1 aligned to bins 1..n_kbins.
        assert 0 <= target_bin < N_KBINS
        # The target bin should hold the maximum power.
        assert int(np.argmax(Z[0])) == target_bin
