# Court key-points (per-match homography)

The tactical analysis maps image (pixel) coordinates to metric court coordinates
via a homography estimated from four manually identified **half-court** corner
points, **one homography per match** (Section 2.6).

For each match, create `court_keypoints/<match_id>.json`:

```json
{
  "match_id": "2025_paris_WS_final",
  "image_corners": [[u_nl, v_nl], [u_nr, v_nr], [u_fr, v_fr], [u_fl, v_fl]]
}
```

`image_corners` are the four corners of the modelled half-court in **clockwise**
order: near-left, near-right, far-right, far-left. They map onto the BWF singles
half-court rectangle (width 5.18 m, net-to-baseline length 6.70 m); note 6.10 m
is the *doubles* width and is not used here.

See `badmintonvision/tactical/homography.py`.
