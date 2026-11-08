"""
Unit tests for the pure metric helpers in scripts/seed_consistency.py (no GPU,
checkpoints or data needed):
    pytest tests/test_seed_consistency.py
"""

import numpy as np

from scripts.seed_consistency import (
    binarize, jaccard, precision_recall, roi_patch_labels, spearman, top_fraction_set,
)


def test_spearman_identical_reversed_and_constant():
    a = np.array([0.0, 1.0, 2.0, 3.0])
    assert spearman(a, a) == 1.0
    assert spearman(a, -a) == -1.0
    assert np.isnan(spearman(a, np.zeros(4)))


def test_jaccard_overlap_and_empty():
    assert jaccard(np.array([1, 2, 3]), np.array([2, 3, 4])) == 0.5
    assert np.isnan(jaccard(np.array([], dtype=int), np.array([], dtype=int)))


def test_top_fraction_ignores_zero_mass_patches():
    w = np.array([0.0, 0.0, 5.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 3.0])
    # 10% of 10 patches -> 1 patch: the largest active one.
    assert list(top_fraction_set(w, 0.10)) == [2]
    # 50% -> k=5 but only 3 patches are active, so the set is just those 3.
    assert list(top_fraction_set(w, 0.50)) == [2, 3, 9]
    assert len(top_fraction_set(np.zeros(10), 0.10)) == 0


def test_binarize_rules():
    s = np.array([0.0, 2.0, 4.0, 10.0])
    assert list(binarize(s, 'minmax:0.5')) == [False, False, False, True]
    assert list(binarize(s, 'top:0.5')) == [False, False, True, True]
    assert not binarize(np.ones(4), 'minmax:0.5').any()  # constant heatmap -> empty


def test_precision_recall():
    pred = np.array([True, True, False, False])
    roi = np.array([True, False, True, False])
    assert precision_recall(pred, roi) == (0.5, 0.5)
    p, r = precision_recall(np.zeros(4, dtype=bool), roi)
    assert np.isnan(p) and r == 0.0


def test_roi_patch_labels_grid_coordinates():
    # 100x100 mask at 1/10 resolution of a 1000x1000 slide; RoI = left half.
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[:, :50] = 255
    # Patches of 100 px (10 mask px) on a 100-px grid: grid x = 0 is inside, 9 outside.
    coords = np.array([[0, 0], [4, 5], [9, 0]], dtype=float)
    labels = roi_patch_labels(mask, coords, coord_unit_px=100, patch_px=100, mask_downsample=10)
    assert list(labels) == [True, True, False]
