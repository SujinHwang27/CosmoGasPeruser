"""Tests for src.core.models.rf_classifier.train_rf_global."""

import numpy as np
import pytest

from src.core.models.rf_classifier import train_rf_global


@pytest.fixture
def synthetic_separable_4class():
    """4-class, well-separated synthetic data: balanced acc near 1.0."""
    rng = np.random.default_rng(7)
    n_per_class = 80
    n_features = 8
    centers = np.array([
        [5, 0, 0, 0, 0, 0, 0, 0],
        [0, 5, 0, 0, 0, 0, 0, 0],
        [0, 0, 5, 0, 0, 0, 0, 0],
        [0, 0, 0, 5, 0, 0, 0, 0],
    ], dtype=np.float64)
    X_list = []
    y_list = []
    for c, mu in enumerate(centers, start=1):
        X_list.append(mu + 0.3 * rng.standard_normal((n_per_class, n_features)))
        y_list.append(np.full(n_per_class, c, dtype=np.int64))
    X = np.vstack(X_list).astype(np.float64)
    y = np.concatenate(y_list)
    return X, y


class TestTrainRFGlobalShapesAndKeys:
    def test_returns_required_keys(self, synthetic_separable_4class):
        X, y = synthetic_separable_4class
        out = train_rf_global(X, y, seed=42, n_estimators=50)
        for k in ("balanced_acc_test", "confusion_test", "feature_importances",
                  "n_train", "n_test", "class_labels"):
            assert k in out, f"missing key {k!r}"

    def test_confusion_is_4x4_row_normalized(self, synthetic_separable_4class):
        X, y = synthetic_separable_4class
        out = train_rf_global(X, y, seed=42, n_estimators=50)
        cm = out["confusion_test"]
        assert cm.shape == (4, 4)
        # Every row sums to 1 (or 0 for an empty true-row, which won't happen here).
        row_sums = cm.sum(axis=1)
        np.testing.assert_allclose(row_sums, np.ones(4), atol=1e-9)

    def test_feature_importances_shape(self, synthetic_separable_4class):
        X, y = synthetic_separable_4class
        out = train_rf_global(X, y, seed=42, n_estimators=50)
        assert out["feature_importances"].shape == (X.shape[1],)

    def test_return_importances_false(self, synthetic_separable_4class):
        X, y = synthetic_separable_4class
        out = train_rf_global(X, y, seed=42, n_estimators=50, return_importances=False)
        assert out["feature_importances"].size == 0

    def test_train_test_sizes_sum_correctly(self, synthetic_separable_4class):
        X, y = synthetic_separable_4class
        out = train_rf_global(X, y, seed=42, n_estimators=50)
        assert out["n_train"] + out["n_test"] == X.shape[0]
        # 80/20 split.
        assert out["n_test"] == X.shape[0] // 5


class TestTrainRFGlobalAccuracy:
    def test_balanced_acc_near_one_on_separable_data(self, synthetic_separable_4class):
        X, y = synthetic_separable_4class
        out = train_rf_global(X, y, seed=42, n_estimators=100)
        assert out["balanced_acc_test"] > 0.95, \
            f"expected >0.95 on separable data, got {out['balanced_acc_test']}"
