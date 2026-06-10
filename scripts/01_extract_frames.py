#!/usr/bin/env python3
"""Stage 1: extract and quality-filter frames (thin CLI over the data module).

Reads paths from the config and calls the frame-extraction routine. See
``badmintonvision/data/frame_extraction.py`` for the implementation.

Example
-------
    python scripts/01_extract_frames.py --config configs/default.yaml
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from badmintonvision.utils.config import Config
from badmintonvision.data.frame_extraction import extract_video
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("overrides", nargs="*")
    args = ap.parse_args()
    cfg = Config.load(args.config).override(args.overrides)

    videos = sorted(Path(cfg.paths["raw_videos"]).glob("*.mp4"))
    if not videos:
        raise SystemExit(f"No .mp4 files in {cfg.paths['raw_videos']}")
    for v in videos:
        stats = extract_video(
            str(v), os.path.join(cfg.paths["frames"], v.stem), match_id=v.stem,
            stride=cfg.frames["stride"],
            laplacian_threshold=cfg.frames["laplacian_threshold"],
        )
        print(stats)


if __name__ == "__main__":
    main()
