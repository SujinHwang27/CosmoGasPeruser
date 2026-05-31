"""Tests for src/core/data.py — data ingestion and normalization."""

import numpy as np
import os
import pytest
import tempfile
from src.core.data import (
    DataIngestor,
    SignalClusteringData,
    DataSummary,
    stack_flux_within_class,
    group_indices_within_class,
    load_velocity_axis,
    assert_velocity_axes_uniform,
)


@pytest.fixture
def mock_data_dir():
    """Create a temporary directory with mock class data."""
    with tempfile.TemporaryDirectory() as tmpdir:
        rng = np.random.default_rng(42)
        for i in range(1, 5):
            class_dir = os.path.join(tmpdir, str(i))
            os.makedirs(class_dir)
            data = rng.standard_normal((20, 2048)).astype(np.float64)
            np.save(os.path.join(class_dir, "flux.npy"), data)
            np.save(os.path.join(class_dir, "data.npy"), data)
        yield tmpdir


class TestDataIngestor:
    def test_load_shapes(self, mock_data_dir):
        ingestor = DataIngestor(mock_data_dir, filename="flux.npy")
        X, y = ingestor.load()
        assert X.shape == (80, 2048)  # 4 classes x 20 samples
        assert y.shape == (80,)

    def test_class_labels(self, mock_data_dir):
        ingestor = DataIngestor(mock_data_dir, filename="flux.npy")
        _, y = ingestor.load()
        unique = np.unique(y)
        np.testing.assert_array_equal(unique, [1, 2, 3, 4])

    def test_load_per_class(self, mock_data_dir):
        ingestor = DataIngestor(mock_data_dir, filename="flux.npy")
        X_list, y = ingestor.load_per_class()
        assert len(X_list) == 4
        assert all(x.shape == (20, 2048) for x in X_list)

    def test_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            ingestor = DataIngestor("/nonexistent/path", filename="flux.npy")
            ingestor.load()

    def test_fallback_to_data_npy(self, mock_data_dir):
        """Test fallback when flux.npy doesn't exist but data.npy does."""
        # Remove flux.npy, keep data.npy
        for i in range(1, 5):
            os.remove(os.path.join(mock_data_dir, str(i), "flux.npy"))
        ingestor = DataIngestor(mock_data_dir, filename="flux.npy")
        X, y = ingestor.load()
        assert X.shape == (80, 2048)


class TestSignalClusteringData:
    def test_normalize_wavelet_per_level_shape(self):
        rng = np.random.default_rng(42)
        X = rng.standard_normal((50, 2048))
        result = SignalClusteringData.normalize_wavelet_per_level(X)
        assert result.shape == X.shape

    def test_normalize_wavelet_per_level_zero_mean(self):
        rng = np.random.default_rng(42)
        X = rng.standard_normal((50, 2048))
        result = SignalClusteringData.normalize_wavelet_per_level(X)
        # Each wavelet level should have approximately zero mean
        level_boundaries = {
            'D1': (0, 1024), 'D2': (1024, 1536), 'D3': (1536, 1792),
            'D4': (1792, 1920), 'D5': (1920, 1984), 'D6': (1984, 2016),
            'A6': (2016, 2048),
        }
        for name, (start, end) in level_boundaries.items():
            level_mean = np.mean(result[:, start:end])
            assert abs(level_mean) < 1e-10, f"Level {name} mean={level_mean}"

    def test_normalize_preserves_no_nan(self):
        rng = np.random.default_rng(42)
        X = rng.standard_normal((50, 2048))
        result = SignalClusteringData.normalize_wavelet_per_level(X)
        assert not np.any(np.isnan(result))


class TestStackFluxWithinClass:
    def test_stack_flux_within_class_shape_and_seed(self):
        rng = np.random.default_rng(0)
        flux = rng.standard_normal((1024, 2048)).astype(np.float64)

        out1 = stack_flux_within_class(flux, M=16, seed=42)
        assert out1.shape == (64, 2048)
        assert out1.dtype == np.float64

        # Determinism on fixed seed.
        out2 = stack_flux_within_class(flux, M=16, seed=42)
        np.testing.assert_array_equal(out1, out2)

        # Different seed → different grouping → different (with overwhelming probability) output.
        out3 = stack_flux_within_class(flux, M=16, seed=43)
        assert not np.array_equal(out1, out3)

    def test_stack_flux_within_class_averaging_is_correct(self):
        # If all sightlines in a group are identical, the mean should equal that row.
        flux = np.tile(np.arange(2048, dtype=np.float64), (32, 1))
        out = stack_flux_within_class(flux, M=4, seed=7)
        assert out.shape == (8, 2048)
        for row in out:
            np.testing.assert_array_equal(row, np.arange(2048, dtype=np.float64))

    def test_stack_flux_within_class_drops_tail(self):
        rng = np.random.default_rng(0)
        flux = rng.standard_normal((10, 2048)).astype(np.float64)  # 10 // 4 = 2, drops 2
        out = stack_flux_within_class(flux, M=4, seed=42)
        assert out.shape == (2, 2048)

    def test_stack_flux_within_class_rejects_bad_inputs(self):
        rng = np.random.default_rng(0)
        flux = rng.standard_normal((32, 2048)).astype(np.float64)
        with pytest.raises(ValueError):
            stack_flux_within_class(flux, M=0)
        with pytest.raises(ValueError):
            stack_flux_within_class(flux.ravel(), M=4)  # not 2D
        with pytest.raises(NotImplementedError):
            stack_flux_within_class(flux, M=4, mode="pk_passthrough")
        with pytest.raises(ValueError):
            stack_flux_within_class(flux, M=4, mode="bogus")  # type: ignore[arg-type]


class TestGroupIndicesWithinClass:
    def test_group_indices_within_class_disjoint(self):
        groups = group_indices_within_class(1024, M=16, seed=42)
        assert len(groups) == 64
        for g in groups:
            assert g.shape == (16,)
        flat = np.concatenate(groups)
        # No duplicates, complete partition of range(1024) (M divides N exactly here).
        assert flat.shape == (1024,)
        assert len(np.unique(flat)) == 1024
        assert set(flat.tolist()) == set(range(1024))

    def test_group_indices_within_class_determinism(self):
        a = group_indices_within_class(256, M=8, seed=123)
        b = group_indices_within_class(256, M=8, seed=123)
        for ga, gb in zip(a, b):
            np.testing.assert_array_equal(ga, gb)

    def test_group_indices_within_class_drops_tail(self):
        groups = group_indices_within_class(10, M=4, seed=42)
        assert len(groups) == 2  # 10 // 4 = 2
        flat = np.concatenate(groups)
        assert flat.shape == (8,)
        assert len(np.unique(flat)) == 8


class TestVelocityAxis:
    def test_assert_velocity_axes_uniform_synthetic(self, tmp_path):
        """Synthetic-only sanity check on assert_velocity_axes_uniform — no real-data dep."""
        dv = 2.6365
        vel = np.arange(2048, dtype=np.float64) * dv
        for c in (1, 2, 3, 4):
            cdir = tmp_path / str(c)
            cdir.mkdir()
            np.save(cdir / "vel.npy", vel)
        out_dv = assert_velocity_axes_uniform(tol=1e-9, flux_base=str(tmp_path))
        assert abs(out_dv - dv) < 1e-9

    def test_assert_velocity_axes_uniform_detects_mismatch(self, tmp_path):
        dv = 2.6365
        vel = np.arange(2048, dtype=np.float64) * dv
        for c in (1, 2, 3, 4):
            cdir = tmp_path / str(c)
            cdir.mkdir()
            np.save(cdir / "vel.npy", vel.copy())
        # Perturb class 3's axis.
        bad = vel.copy()
        bad[100] += 1.0
        np.save(tmp_path / "3" / "vel.npy", bad)
        with pytest.raises(AssertionError):
            assert_velocity_axes_uniform(tol=1e-9, flux_base=str(tmp_path))

    def test_assert_velocity_axes_uniform_detects_nonuniform_spacing(self, tmp_path):
        dv = 2.6365
        vel = np.arange(2048, dtype=np.float64) * dv
        vel[1000] += 0.5  # break uniform spacing while keeping all 4 identical
        for c in (1, 2, 3, 4):
            cdir = tmp_path / str(c)
            cdir.mkdir()
            np.save(cdir / "vel.npy", vel.copy())
        with pytest.raises(AssertionError):
            assert_velocity_axes_uniform(tol=1e-9, flux_base=str(tmp_path))

    def test_load_velocity_axis_rejects_bad_class(self):
        with pytest.raises(ValueError):
            load_velocity_axis(5)

    @pytest.mark.slow
    def test_assert_velocity_axes_uniform_real_data(self):
        """Real on-disk vel.npy — Δv ≈ 2.6365 km/s, uniform across all 4 classes ([D-04])."""
        dv = assert_velocity_axes_uniform(tol=1e-6)
        assert abs(dv - 2.6365) < 1e-3, f"Δv = {dv:.6f}, expected ≈ 2.6365 km/s"


class TestDataSummary:
    def test_summary_fields(self):
        summary = DataSummary(
            shape=(100, 24), dtype="float64",
            min=-3.0, max=3.0, mean=0.0, std=1.0,
            has_nan=False, has_inf=False
        )
        assert summary.shape == (100, 24)
        assert not summary.has_nan
