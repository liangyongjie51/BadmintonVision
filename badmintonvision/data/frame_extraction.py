"""Frame extraction and quality filtering (paper Section 2.2).

Videos are segmented into individual rally clips (service contact to point
conclusion). Frames are extracted at a fixed stride and a variance-of-Laplacian
blur filter removes motion-blurred frames. In the paper this yielded 208,891
candidate frames, of which 169,501 (81.1%) passed the quality filter.

Usage
-----
    python -m badmintonvision.data.frame_extraction \
        --videos data/raw_videos --out data/frames \
        --stride 10 --laplacian-threshold 100
"""
from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None


def variance_of_laplacian(image) -> float:
    """Return the focus measure (higher == sharper)."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def extract_video(
    video_path: str,
    out_dir: str,
    match_id: str,
    stride: int = 10,
    laplacian_threshold: float = 100.0,
) -> dict:
    """Extract and quality-filter frames from a single match video.

    Returns a small dict of statistics (candidate / kept counts).
    """
    if cv2 is None:
        raise ImportError("OpenCV (cv2) is required for frame extraction.")
    os.makedirs(out_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Could not open video: {video_path}")

    frame_idx, candidate, kept = 0, 0, 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx % stride == 0:
            candidate += 1
            if variance_of_laplacian(frame) >= laplacian_threshold:
                fname = f"{match_id}_{frame_idx:07d}.jpg"
                cv2.imwrite(os.path.join(out_dir, fname), frame)
                kept += 1
        frame_idx += 1
    cap.release()

    retention = kept / candidate if candidate else 0.0
    return {
        "match_id": match_id,
        "candidate_frames": candidate,
        "kept_frames": kept,
        "retention_rate": round(retention, 4),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Extract & quality-filter frames.")
    ap.add_argument("--videos", required=True, help="directory of match .mp4 files")
    ap.add_argument("--out", required=True, help="output frame directory")
    ap.add_argument("--stride", type=int, default=10)
    ap.add_argument("--laplacian-threshold", type=float, default=100.0)
    args = ap.parse_args()

    rows = []
    for video in sorted(Path(args.videos).glob("*.mp4")):
        match_id = video.stem
        stats = extract_video(
            str(video),
            os.path.join(args.out, match_id),
            match_id=match_id,
            stride=args.stride,
            laplacian_threshold=args.laplacian_threshold,
        )
        print(stats)
        rows.append(stats)

    if rows:
        summary = os.path.join(args.out, "extraction_summary.csv")
        with open(summary, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        tot_c = sum(r["candidate_frames"] for r in rows)
        tot_k = sum(r["kept_frames"] for r in rows)
        print(f"\nTotal: {tot_k}/{tot_c} kept "
              f"({100 * tot_k / max(tot_c, 1):.1f}% retention) -> {summary}")


if __name__ == "__main__":
    main()
