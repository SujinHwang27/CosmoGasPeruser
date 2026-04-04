"""Tests for src/core/probe.py — micro-probing module."""

import numpy as np
import pytest
from src.core.probe import probe_sightline, run_probe, OVO_PAIRS


@pytest.fixture
def synthetic_classes():
    """Create 4 synthetic class arrays: 10 sightlines x 8 features each."""
    rng = np.random.default_rng(42)
    return [rng.standard_normal((10, 8)) for _ in range(4)]


class TestOVOPairs:
    def test_pair_count(self):
        assert len(OVO_PAIRS) == 6  # C(4,2) = 6

    def test_pairs_are_unique(self):
        assert len(set(OVO_PAIRS)) == 6

    def test_all_classes_covered(self):
        classes_seen = set()
        for p, q in OVO_PAIRS:
            classes_seen.add(p)
            classes_seen.add(q)
        assert classes_seen == {0, 1, 2, 3}


class TestProbeSightline:
    def test_output_shape(self, synthetic_classes):
        result = probe_sightline(0, synthetic_classes)
        assert result.shape == (24,)  # 6 pairs x 4 distances

    def test_output_dtype(self, synthetic_classes):
        result = probe_sightline(0, synthetic_classes)
        assert result.dtype == np.float64

    def test_output_finite(self, synthetic_classes):
        result = probe_sightline(0, synthetic_classes)
        assert np.all(np.isfinite(result))

    def test_different_sightlines_differ(self, synthetic_classes):
        r0 = probe_sightline(0, synthetic_classes)
        r1 = probe_sightline(1, synthetic_classes)
        assert not np.allclose(r0, r1)


class TestRunProbe:
    def test_output_shape(self, synthetic_classes):
        result = run_probe(synthetic_classes, n_jobs=1)
        assert result.shape == (10, 24)  # 10 sightlines x 24-dim

    def test_output_finite(self, synthetic_classes):
        result = run_probe(synthetic_classes, n_jobs=1)
        assert np.all(np.isfinite(result))

    def test_deterministic(self, synthetic_classes):
        r1 = run_probe(synthetic_classes, n_jobs=1)
        r2 = run_probe(synthetic_classes, n_jobs=1)
        np.testing.assert_array_equal(r1, r2)
