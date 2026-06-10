"""Trajectory processing and movement speed (paper Sections 2.6-2.7).

Player trajectories are obtained from detected bounding-box centroids, mapped to
metric court coordinates by the per-match homography, and smoothed with a
Savitzky-Golay filter (window = 5, polynomial order = 2) that preserves
directional changes. In all movement/speed equations the coordinates x and y
denote post-homography court positions in metres (not raw pixels), so speeds and
path lengths are reported directly in metric court units.

Instantaneous speed between consecutive frames i-1, i:
    v_i = sqrt((x_i - x_{i-1})^2 + (y_i - y_{i-1})^2) / dt,      dt = 1 / fps
"""
from __future__ import annotations

import numpy as np

try:
    from scipy.signal import savgol_filter
except ImportError:  # pragma: no cover
    savgol_filter = None


def smooth_trajectory(xy_m: np.ndarray, window: int = 5, polyorder: int = 2) -> np.ndarray:
    """Savitzky-Golay smoothing of an (N, 2) metric trajectory."""
    xy = np.asarray(xy_m, dtype=np.float64)
    n = len(xy)
    if n < window or savgol_filter is None:
        return xy
    win = window if window % 2 == 1 else window + 1     # must be odd
    win = min(win, n if n % 2 == 1 else n - 1)
    if win <= polyorder:
        return xy
    out = np.empty_like(xy)
    out[:, 0] = savgol_filter(xy[:, 0], win, polyorder)
    out[:, 1] = savgol_filter(xy[:, 1], win, polyorder)
    return out


def instantaneous_speed(xy_m: np.ndarray, fps: float = 25.0) -> np.ndarray:
    """Per-step speed (m/s) for an (N, 2) metric trajectory."""
    xy = np.asarray(xy_m, dtype=np.float64)
    if len(xy) < 2:
        return np.zeros(0)
    steps = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    return steps * fps


def average_speed(xy_m: np.ndarray, fps: float = 25.0) -> float:
    """Mean movement speed over the trajectory (m/s)."""
    v = instantaneous_speed(xy_m, fps)
    return float(v.mean()) if len(v) else 0.0


def total_distance(xy_m: np.ndarray) -> float:
    """Total path length travelled (m)."""
    xy = np.asarray(xy_m, dtype=np.float64)
    if len(xy) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(xy, axis=0), axis=1).sum())


def straight_line_displacement(xy_m: np.ndarray) -> float:
    """Straight-line distance between first and last position (m)."""
    xy = np.asarray(xy_m, dtype=np.float64)
    if len(xy) < 2:
        return 0.0
    return float(np.linalg.norm(xy[-1] - xy[0]))


__all__ = [
    "smooth_trajectory",
    "instantaneous_speed",
    "average_speed",
    "total_distance",
    "straight_line_displacement",
]
