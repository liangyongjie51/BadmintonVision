"""Court homography (paper Section 2.6).

Image (pixel) coordinates of detected player positions are mapped to real-world
court coordinates (metres) via a perspective transform estimated from four
manually identified court corner points. Following BWF specifications the
singles court is 13.40 m long and 5.18 m wide (6.10 m is the *doubles* width);
the modelled region is one half-court, 6.70 m (net-to-baseline) by 5.18 m
(singles width). A **separate** homography is estimated per match (different
camera placement / zoom), rather than assuming a single constant transform.

The homography H satisfies, in homogeneous coordinates,
    s * [x_court, y_court, 1]^T = H * [u_pixel, v_pixel, 1]^T,
where (x_court, y_court) are in metres and s is a scale factor.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None

# BWF singles half-court corners in metres (x = width, y = length).
HALF_COURT_CORNERS_M = np.array(
    [[0.00, 0.00], [5.18, 0.00], [5.18, 6.70], [0.00, 6.70]], dtype=np.float64
)


def estimate_homography(image_corners: np.ndarray,
                        court_corners_m: np.ndarray = HALF_COURT_CORNERS_M) -> np.ndarray:
    """Estimate the 3x3 homography mapping pixels -> court metres.

    Parameters
    ----------
    image_corners : (4, 2) pixel coordinates of the half-court corners, in the
        same clockwise order as ``court_corners_m`` (near-left, near-right,
        far-right, far-left).
    """
    image_corners = np.asarray(image_corners, dtype=np.float64)
    assert image_corners.shape == (4, 2), "need exactly four corner points"
    if cv2 is not None:
        H, _ = cv2.findHomography(image_corners, court_corners_m, method=0)
        return H
    return _dlt_homography(image_corners, court_corners_m)


def apply_homography(H: np.ndarray, points_px: np.ndarray) -> np.ndarray:
    """Map an (N, 2) array of pixel points to court metres."""
    pts = np.asarray(points_px, dtype=np.float64).reshape(-1, 2)
    ones = np.ones((pts.shape[0], 1))
    hom = np.hstack([pts, ones]) @ H.T
    court = hom[:, :2] / hom[:, 2:3]
    return court


def load_match_homography(court_keypoints_dir: str, match_id: str) -> np.ndarray:
    """Load per-match corner points (JSON) and return the homography.

    Expected JSON: ``{"image_corners": [[u,v], [u,v], [u,v], [u,v]]}``.
    """
    path = Path(court_keypoints_dir) / f"{match_id}.json"
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    return estimate_homography(np.array(data["image_corners"], dtype=np.float64))


def _dlt_homography(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Minimal Direct-Linear-Transform fallback when OpenCV is unavailable."""
    A = []
    for (x, y), (X, Y) in zip(src, dst):
        A.append([-x, -y, -1, 0, 0, 0, x * X, y * X, X])
        A.append([0, 0, 0, -x, -y, -1, x * Y, y * Y, Y])
    _, _, vh = np.linalg.svd(np.asarray(A))
    H = vh[-1].reshape(3, 3)
    return H / H[2, 2]


__all__ = [
    "estimate_homography",
    "apply_homography",
    "load_match_homography",
    "HALF_COURT_CORNERS_M",
]
