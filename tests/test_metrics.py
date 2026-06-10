"""Unit tests for the tactical and statistical code (run with: pytest -q)."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from badmintonvision.tactical.metrics import (
    path_efficiency, direction_changes, center_court_time,
)
from badmintonvision.tactical.homography import estimate_homography, apply_homography, HALF_COURT_CORNERS_M
from badmintonvision.tactical.markov import transition_counts, transition_matrix
from badmintonvision.tactical.lag_sequential import lag_sequential_residuals
from badmintonvision.stats.analysis import cohens_d, cohens_d_ci


def test_path_efficiency_straight_line_is_one():
    xy = np.array([[0, 0], [1, 0], [2, 0], [3, 0]], dtype=float)
    assert abs(path_efficiency(xy) - 1.0) < 1e-9


def test_path_efficiency_backtrack_less_than_one():
    xy = np.array([[0, 0], [2, 0], [1, 0]], dtype=float)  # overshoot then return
    assert 0.0 < path_efficiency(xy) < 1.0


def test_direction_changes_counts_reversals():
    # zig-zag: every step reverses direction (>90 deg)
    xy = np.array([[0, 0], [1, 0], [0, 0], [1, 0], [0, 0]], dtype=float)
    assert direction_changes(xy, angle_threshold_deg=90) == 3


def test_center_court_time_fraction():
    # all points at court centre -> fraction 1.0
    cx, cy = 5.18 / 2, 6.70 / 2
    xy = np.array([[cx, cy]] * 10, dtype=float)
    assert abs(center_court_time(xy) - 1.0) < 1e-9


def test_homography_roundtrip_recovers_corners():
    # square image corners -> known court corners; mapping the image corners back
    img = np.array([[100, 500], [900, 500], [900, 100], [100, 100]], dtype=float)
    H = estimate_homography(img, HALF_COURT_CORNERS_M)
    mapped = apply_homography(H, img)
    assert np.allclose(mapped, HALF_COURT_CORNERS_M, atol=1e-6)


def test_markov_rows_sum_to_one():
    seqs = [["Forehand", "Backhand", "Forehand", "Jump_Smash"],
            ["Backhand", "Backhand", "Forehand"]]
    P = transition_matrix(transition_counts(seqs))
    row_sums = P.sum(axis=1)
    # rows with at least one outgoing transition sum to 1
    for r in row_sums:
        assert abs(r - 1.0) < 1e-9 or abs(r) < 1e-9


def test_lag_sequential_shapes_and_significance_flag():
    seqs = [["Forehand", "Jump_Smash"] * 20]
    out = lag_sequential_residuals(seqs, lag=1)
    z = out["adjusted_residuals"]
    assert z.shape == (3, 3)
    assert "significant" in out and out["significant"].shape == (3, 3)


def test_cohens_d_sign_and_ci_brackets_estimate():
    rng = np.random.default_rng(0)
    a = rng.normal(1.0, 1.0, 200)
    b = rng.normal(0.0, 1.0, 200)
    d = cohens_d(a, b)
    assert d > 0
    ci = cohens_d_ci(a, b)
    assert ci["ci_low"] < ci["d"] < ci["ci_high"]
