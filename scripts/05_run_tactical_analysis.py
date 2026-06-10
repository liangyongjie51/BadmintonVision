#!/usr/bin/env python3
"""Run the tactical analysis over detected/recognised rallies (Sections 2.6-2.8).

For each rally this script: (1) maps player centroids to metric court
coordinates via the per-match homography, (2) smooths the trajectory, (3)
computes movement metrics (speed, path efficiency, centre-court time, recovery
time, direction changes), and (4) aggregates stroke sequences into Markov
transition matrices and lag-1 sequential adjusted residuals. Results are written
as CSV/JSON for the statistical-analysis and figure scripts.

Input
-----
A tracks file (``data/tracks.json``) with, per rally:
    {"rally_id", "match_id", "gender", "scorer",
     "near_player_centroids_px": [[u,v], ...],
     "stroke_sequence": ["Forehand", "Backhand", ...]}

Example
-------
    python scripts/05_run_tactical_analysis.py --config configs/default.yaml \
        --tracks data/tracks.json
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from badmintonvision.utils.config import Config  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--tracks", default="data/tracks.json")
    args = ap.parse_args()
    cfg = Config.load(args.config)

    import numpy as np

    from badmintonvision.tactical.homography import load_match_homography, apply_homography
    from badmintonvision.tactical.trajectory import smooth_trajectory
    from badmintonvision.tactical.metrics import rally_metrics
    from badmintonvision.tactical.markov import transition_counts, transition_matrix
    from badmintonvision.tactical.lag_sequential import lag_sequential_residuals

    with open(args.tracks, "r", encoding="utf-8") as fh:
        rallies = json.load(fh)

    fps = cfg.temporal["fps"]
    rows, seqs_by_cond = [], {"scoring": [], "conceding": []}
    homographies: dict[str, object] = {}

    for r in rallies:
        mid = r["match_id"]
        if cfg.tactical["homography_per_match"] and mid not in homographies:
            try:
                homographies[mid] = load_match_homography(cfg.paths["court_keypoints"], mid)
            except FileNotFoundError:
                homographies[mid] = None
        H = homographies.get(mid)

        px = np.asarray(r["near_player_centroids_px"], dtype=float)
        court_m = apply_homography(H, px) if H is not None else px
        court_m = smooth_trajectory(court_m, cfg.tactical["savgol_window"],
                                    cfg.tactical["savgol_polyorder"])
        m = rally_metrics(court_m, fps=fps)
        m.update({"rally_id": r["rally_id"], "match_id": mid,
                  "gender": r.get("gender", "NA"), "scorer": r.get("scorer", "NA")})
        rows.append(m)

        cond = "scoring" if r.get("is_scoring", True) else "conceding"
        seqs_by_cond[cond].append(r.get("stroke_sequence", []))

    os.makedirs(cfg.paths["outputs"], exist_ok=True)
    # movement metrics CSV
    metrics_csv = os.path.join(cfg.paths["outputs"], "rally_metrics.csv")
    if rows:
        keys = ["rally_id", "match_id", "gender", "scorer", "avg_speed_mps",
                "path_efficiency", "center_court_time_pct",
                "base_recovery_time_s", "direction_changes"]
        with open(metrics_csv, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=keys)
            w.writeheader()
            w.writerows([{k: row.get(k) for k in keys} for row in rows])

    # Markov + lag-sequential per condition
    states = cfg.tactical["markov_states"]
    tactical = {}
    for cond, seqs in seqs_by_cond.items():
        counts = transition_counts(seqs, states)
        tactical[cond] = {
            "transition_counts": counts.tolist(),
            "transition_matrix": transition_matrix(counts).round(4).tolist(),
            "lag_sequential_z": lag_sequential_residuals(
                seqs, states, cfg.tactical["lag"])["adjusted_residuals"].round(3).tolist(),
            "states": list(states),
        }
    with open(os.path.join(cfg.paths["outputs"], "tactical_summary.json"),
              "w", encoding="utf-8") as fh:
        json.dump(tactical, fh, indent=2)

    print(f"Wrote {metrics_csv} ({len(rows)} rallies) and tactical_summary.json")


if __name__ == "__main__":
    main()
