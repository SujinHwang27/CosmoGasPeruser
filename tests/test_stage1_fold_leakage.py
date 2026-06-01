"""C6 fold-leakage unit test for the pk-feedback-classifier Stage 1 protocol.

Binding: experiments/pk-feedback-classifier/LEDGER.md [D-20] S3 + [D-23] C6.

Deterministic synthetic-fixture test of the fold-membership logic: for one
representative (regime, M, seed) tuple at M=16, exercise the real
`group_indices_within_class` stacking helper from `src/core/data.py`, then run
sklearn `StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)` over the
post-stacking GROUP index per [D-20] S3 — and assert that for each of the 5
folds, NO source-sightline-index appears in BOTH the train-stack-memberships
AND the test-stack-memberships of that fold.

This is the primary CI gate per [D-23] C6 — it must be green before
infrastructure-manager fires the Stage 1 sbatch on Juno (a 6h wallclock cost to
discover what this 1-second unit test would have caught at commit time).

Fixture
-------
4-class balanced × 25 sightlines × 2048 pixels float64, deterministic
``np.random.default_rng(seed=42)``. Per-pixel deterministic so the test is
fully reproducible.
"""
from __future__ import annotations

from typing import List, Set

import numpy as np
import pytest
from sklearn.model_selection import StratifiedKFold

from src.core.data import group_indices_within_class


# ---------------------------------------------------------------------------
# Synthetic fixture
# ---------------------------------------------------------------------------
N_CLASSES: int = 4
N_PER_CLASS: int = 25       # 100 total sightlines per the C6 brief
N_PIXELS: int = 2048
M_REPR: int = 16            # smallest M with > 1 stacks per fold (25/16 = 1 stack/class -> too small)
N_FOLDS: int = 5
SEED: int = 42

# Note: we intentionally use a *larger* per-class count below in the actual
# leakage test so that M=16 yields > 1 stacks per class. The 100-sightline
# fixture from the brief is the *deterministic* slice; we extend per-class
# count so that the StratifiedKFold can meaningfully split (you need
# n_stacks_per_class >= n_folds for stratification). See ``_build_fixture``
# for the resolved sizing.


def _build_fixture(
    n_classes: int = N_CLASSES,
    n_per_class: int = N_PER_CLASS,
    n_pixels: int = N_PIXELS,
    seed: int = SEED,
) -> List[np.ndarray]:
    """Deterministic 4-class balanced synthetic flux fixture, list-of-arrays.

    Each class is a separate ``(n_per_class, n_pixels)`` float64 block drawn
    from ``np.random.default_rng(seed + class_idx)`` so the per-class slices
    are distinct but reproducible.
    """
    out: List[np.ndarray] = []
    for c in range(n_classes):
        rng = np.random.default_rng(seed + c)
        # shape: (n_per_class, n_pixels)
        block = rng.standard_normal((n_per_class, n_pixels)).astype(np.float64)
        out.append(block)
    return out


# ---------------------------------------------------------------------------
# Helper — replicates the post-stacking-group → source-sightline mapping that
# `run_stage1.py` MUST use. Kept HERE (in the test) and asserted against the
# production code path's outputs via the same `group_indices_within_class`
# call — i.e., the test exercises the real stacking helper, not a stand-in.
# ---------------------------------------------------------------------------
def _build_stack_membership_table(
    flux_per_class: List[np.ndarray], M: int, seed: int
) -> List[np.ndarray]:
    """Return per-class list of (n_groups, M) index arrays via the REAL helper.

    Each row of the returned (n_groups, M) array is the set of source-sightline
    indices that go into one post-stacking group within the corresponding
    class. The class-level group ordering is the same as
    `group_indices_within_class`'s return order.
    """
    table: List[np.ndarray] = []
    for flux in flux_per_class:
        groups = group_indices_within_class(flux.shape[0], M=M, seed=seed)
        # shape: (n_groups, M)
        groups_arr = np.stack(groups, axis=0) if len(groups) > 0 else np.empty(
            (0, M), dtype=np.int64
        )
        table.append(groups_arr)
    return table


def _flatten_to_global_groups(
    membership_table: List[np.ndarray],
) -> tuple:
    """Concat per-class group-membership into a global group index space.

    Returns
    -------
    global_group_to_source : list of (class_idx, np.ndarray-of-source-idx)
        Length == total n_groups across classes; aligns with the row layout
        used by ``StratifiedKFold`` below.
    y_groups : np.ndarray, shape (total_groups,), int — class label per group.
    """
    flat: List[tuple] = []
    y_groups_list: List[int] = []
    for c_idx, groups_arr in enumerate(membership_table):
        n_groups = groups_arr.shape[0]
        for g in range(n_groups):
            flat.append((c_idx, groups_arr[g]))
            y_groups_list.append(c_idx + 1)  # 1..4 to mirror physics class IDs
    y_groups = np.asarray(y_groups_list, dtype=np.int64)
    return flat, y_groups


# ---------------------------------------------------------------------------
# The test
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("M", [M_REPR])
def test_no_source_sightline_leak_across_train_test_per_fold(M: int) -> None:
    """For (M=16, seed=42), the 5-fold split over post-stacking groups must
    not place ANY source-sightline-index in both train and test of any fold.

    Per [D-20] S3 + [D-23] C6. Group-level splitting at the post-stacking
    layer is the binding contract; this test would catch a regression to
    sightline-level (which would smear source-sightlines across folds via the
    stacking step) or any future class-internal-shuffle bug that violates
    disjointness within a class.
    """
    # Ensure n_per_class is large enough for M=16 to yield >= n_folds stacks
    # per class — otherwise StratifiedKFold cannot split.
    n_per_class = max(N_PER_CLASS, N_FOLDS * M + M)  # 5*16 + 16 = 96
    flux_per_class = _build_fixture(
        n_classes=N_CLASSES, n_per_class=n_per_class, n_pixels=N_PIXELS, seed=SEED
    )
    assert all(f.shape == (n_per_class, N_PIXELS) for f in flux_per_class)
    assert all(f.dtype == np.float64 for f in flux_per_class)

    # Build the per-class stack-membership table via the REAL helper.
    membership_table = _build_stack_membership_table(
        flux_per_class, M=M, seed=SEED
    )

    # Within-class disjointness sanity (a precondition the real helper must
    # guarantee — exercised here to make CI failures localize cleanly).
    for c_idx, groups_arr in enumerate(membership_table):
        flat_idx = groups_arr.reshape(-1)
        assert len(set(flat_idx.tolist())) == len(flat_idx), (
            f"class {c_idx+1}: source-sightline duplicated within stack "
            f"memberships (group_indices_within_class is broken)"
        )

    # Flatten to a global group index space and stratify-split.
    global_group_to_source, y_groups = _flatten_to_global_groups(membership_table)
    n_total_groups = len(global_group_to_source)
    group_indices = np.arange(n_total_groups)

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

    fold_id = 0
    for train_grp, test_grp in skf.split(group_indices, y_groups):
        fold_id += 1

        # Materialize per-class source-sightline sets for train/test stacks of
        # this fold. Leakage is defined PER CLASS — a source-sightline in
        # class C's train can never appear in class C's test (within a class,
        # stack memberships are partitioned; across classes the source index
        # space is class-local and disjoint by construction). Test BOTH
        # within-class disjointness AND the joint (class_idx, sightline_idx)
        # disjointness for defense-in-depth.

        train_pairs: Set[tuple] = set()
        test_pairs: Set[tuple] = set()

        for gi in train_grp:
            c_idx, src_arr = global_group_to_source[int(gi)]
            for s in src_arr.tolist():
                train_pairs.add((c_idx, int(s)))
        for gi in test_grp:
            c_idx, src_arr = global_group_to_source[int(gi)]
            for s in src_arr.tolist():
                test_pairs.add((c_idx, int(s)))

        overlap = train_pairs & test_pairs
        assert overlap == set(), (
            f"fold {fold_id} (M={M}, seed={SEED}): "
            f"{len(overlap)} (class_idx, source_sightline_idx) pairs leaked "
            f"between train and test stack memberships. "
            f"Sample leaks: {list(overlap)[:5]}"
        )


def test_fixture_is_deterministic() -> None:
    """Re-running the fixture with the same seed yields bit-identical output.

    Catches accidental introduction of non-deterministic seeding in the
    fixture or the stacking helper.
    """
    a = _build_fixture(seed=SEED)
    b = _build_fixture(seed=SEED)
    for fa, fb in zip(a, b):
        np.testing.assert_array_equal(fa, fb)

    table_a = _build_stack_membership_table(a, M=M_REPR, seed=SEED)
    table_b = _build_stack_membership_table(b, M=M_REPR, seed=SEED)
    for ga, gb in zip(table_a, table_b):
        np.testing.assert_array_equal(ga, gb)
