#!/usr/bin/env python3
"""Train and evaluate the YOLOv12 detector (paper Sections 2.2, 2.9, 3.1).

The backbone is initialised from the SSL-pretrained encoder (feature fusion).
Evaluation reports AP@0.5 per class and mAP@0.5 at the paper's inference
settings (conf 0.25, NMS IoU 0.70). Splitting must be match-level (see
``configs/default.yaml``: split.unit = match).

Example
-------
    python scripts/03_train_detector.py --config configs/default.yaml \
        detection.epochs=100 detection.ssl_init=true
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from badmintonvision.utils.config import Config, set_seed  # noqa: E402


def write_data_yaml(cfg: Config) -> str:
    """Emit an Ultralytics data.yaml referencing the match-level splits."""
    import yaml

    data = {
        "path": os.path.abspath(cfg.paths["frames"]),
        "train": "../splits/train.txt",
        "val": "../splits/val.txt",
        "test": "../splits/test.txt",
        "names": {i: c for i, c in enumerate(cfg.classes)},
    }
    out = os.path.join(cfg.paths["splits"], "data.yaml")
    os.makedirs(cfg.paths["splits"], exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        yaml.safe_dump(data, fh, sort_keys=False)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--eval-only", action="store_true")
    ap.add_argument("overrides", nargs="*")
    args = ap.parse_args()

    cfg = Config.load(args.config).override(args.overrides)
    set_seed(cfg.project["seed"])

    if cfg.split["unit"] != "match":
        raise SystemExit("split.unit must be 'match' to avoid data leakage "
                         "(Reviewer 2, Major Comment 1).")

    from badmintonvision.detection.yolo_convnext import BadmintonDetector

    data_yaml = write_data_yaml(cfg)
    ssl_ckpt = os.path.join(cfg.paths["weights"], "ssl_convnextv2_mae_encoder.pt")
    detector = BadmintonDetector(
        model=cfg.detection["model"],
        ssl_checkpoint=ssl_ckpt if cfg.detection["ssl_init"] and os.path.isfile(ssl_ckpt) else None,
    )

    if not args.eval_only:
        detector.train(
            data_yaml,
            epochs=cfg.detection["epochs"],
            imgsz=cfg.detection["imgsz"],
            batch=cfg.detection["batch_size"],
        )

    metrics = detector.evaluate(
        data_yaml,
        conf=cfg.detection["conf_threshold"],
        iou=cfg.detection["nms_iou"],
        imgsz=cfg.detection["imgsz"],
    )
    print("Evaluation (match-level split):")
    print(metrics)


if __name__ == "__main__":
    main()
