"""Per-rally movement metrics (paper Sections 2.6, 2.10).

All metrics operate on a smoothed, metric (post-homography) trajectory.

- path_efficiency      : straight-line displacement / total distance travelled.
- center_court_time    : fraction of frames spent in the central court region.
- base_recovery_time   : time (s) to return within a radius of the base position
                         after the trajectory's farthest excursion.
- direction_changes    : number of court-direction reversals per rally, defined
                         as successive changes in the smoothed movement-vector
                         orientation exceeding a threshold angle (default 90 deg).

The base position is taken as the trajectory centroid (a stable, rally-specific
reference); the central region is a configurable rectangle centred on the court.
"""
from __future__ import annotations

import numpy as np

from .trajectory import straight_line_displacement, total_distance

# Half-court dimensions in metres (singles).
COURT_WIDTH_M = 5.18
COURT_LENGTH_HALF_M = 6.70


def path_efficiency(xy_m: np.ndarray) -> float:
    """Straight-line displacement divided by total distance travelled (0-1)."""
    dist = total_distance(xy_m)
    if dist <= 1e-9:
        return 0.0
    return float(straight_line_displacement(xy_m) / dist)


def center_court_time(
    xy_m: np.ndarray,
    width_m: float = COURT_WIDTH_M,
    length_m: float = COURT_LENGTH_HALF_M,
    center_frac: float = 0.5,
) -> float:
    """Fraction of frames inside the central rectangle (default middle 50%)."""
    xy = np.asarray(xy_m, dtype=np.float64)
    if len(xy) == 0:
        return 0.0
    cx, cy = width_m / 2, length_m / 2
    half_w = width_m * center_frac / 2
    half_l = length_m * center_frac / 2
    inside = (np.abs(xy[:, 0] - cx) <= half_w) & (np.abs(xy[:, 1] - cy) <= half_l)
    return float(inside.mean())


def base_recovery_time(xy_m: np.ndarray, fps: float = 25.0,
                       radius_m: float = 1.0) -> float:
    """Seconds from the farthest excursion back to within ``radius_m`` of base.

    The base is the trajectory centroid. Returns 0 if the player is already
    within the radius at/after the farthest point for the remainder of the rally.
    """
    xy = np.asarray(xy_m, dtype=np.float64)
    if len(xy) < 3:
        return 0.0
    base = xy.mean(axis=0)
    dist_to_base = np.linalg.norm(xy - base, axis=1)
    far_idx = int(np.argmax(dist_to_base))
    for i in range(far_idx, len(xy)):
        if dist_to_base[i] <= radius_m:
            return (i - far_idx) / fps
    return (len(xy) - 1 - far_idx) / fps


def direction_changes(xy_m: np.ndarray, angle_threshold_deg: float = 90.0) -> int:
    """Count successive movement-vector orientation changes exceeding a threshold.

    A direction change is registered when the angle between consecutive
    displacement vectors exceeds ``angle_threshold_deg``.
    """
    xy = np.asarray(xy_m, dtype=np.float64)
    if len(xy) < 3:
        return 0
    vecs = np.diff(xy, axis=0)
    norms = np.linalg.norm(vecs, axis=1)
    valid = norms > 1e-6
    vecs, norms = vecs[valid], norms[valid]
    if len(vecs) < 2:
        return 0
    unit = vecs / norms[:, None]
    cos = np.clip((unit[:-1] * unit[1:]).sum(axis=1), -1.0, 1.0)
    angles = np.degrees(np.arccos(cos))
    return int((angles > angle_threshold_deg).sum())


def rally_metrics(xy_m: np.ndarray, fps: float = 25.0) -> dict:
    """Compute all per-rally movement metrics for one player trajectory."""
    from .trajectory import average_speed

    return {
        "avg_speed_mps": average_speed(xy_m, fps),
        "path_efficiency": path_efficiency(xy_m),
        "center_court_time_pct": 100.0 * center_court_time(xy_m),
        "base_recovery_time_s": base_recovery_time(xy_m, fps),
        "direction_changes": direction_changes(xy_m),
    }


__all__ = [
    "path_efficiency",
    "center_court_time",
    "base_recovery_time",
    "direction_changes",
    "rally_metrics",
]
