"""YOLOv12 detector with an SSL-augmented backbone (paper Sections 2.2, 2.9).

The detector is YOLOv12m. "Feature fusion" in the paper means the
SSL-pretrained ConvNeXt V2 weights are transferred into / used to initialise the
detector backbone (backbone *augmentation*), and the enriched backbone features
are propagated through the standard YOLO feature-pyramid neck to the detection
head. The backbone is **not** replaced.

This module is a thin, documented wrapper around the Ultralytics YOLO API so the
pipeline stays reproducible while delegating the heavy lifting to a maintained
detector implementation. Inference uses a confidence threshold of 0.25 and NMS
IoU 0.70 (paper-reported settings).
"""
from __future__ import annotations

import os
from typing import Optional


def load_ssl_backbone_into_yolo(yolo_model, ssl_checkpoint: str) -> None:
    """Copy matching SSL-pretrained backbone tensors into the YOLO backbone.

    Only parameters whose names and shapes match are transferred; the YOLO neck
    and head are left at their default initialisation. This realises the
    feature-fusion / backbone-augmentation described in the paper.
    """
    import torch

    if not os.path.isfile(ssl_checkpoint):
        raise FileNotFoundError(ssl_checkpoint)
    state = torch.load(ssl_checkpoint, map_location="cpu")
    state = state.get("encoder", state.get("model", state))
    target = yolo_model.model.state_dict()
    transferred = 0
    for name, tensor in state.items():
        # map common encoder prefixes onto the YOLO backbone namespace
        for cand in (name, f"model.0.{name}", name.replace("encoder.", "")):
            if cand in target and target[cand].shape == tensor.shape:
                target[cand].copy_(tensor)
                transferred += 1
                break
    yolo_model.model.load_state_dict(target, strict=False)
    print(f"[ssl-init] transferred {transferred} backbone tensors "
          f"from {ssl_checkpoint}")


class BadmintonDetector:
    """Train / evaluate / predict wrapper around Ultralytics YOLO."""

    def __init__(self, model: str = "yolov12m", ssl_checkpoint: Optional[str] = None):
        from ultralytics import YOLO

        # YOLOv12 ships as 'yolov12m.yaml' / 'yolov12m.pt' in recent ultralytics.
        weight = model if model.endswith((".pt", ".yaml")) else f"{model}.yaml"
        self.model = YOLO(weight)
        if ssl_checkpoint:
            load_ssl_backbone_into_yolo(self.model, ssl_checkpoint)

    def train(self, data_yaml: str, epochs: int = 100, imgsz: int = 1280,
              batch: int = 16, **kw):
        return self.model.train(data=data_yaml, epochs=epochs, imgsz=imgsz,
                                batch=batch, **kw)

    def evaluate(self, data_yaml: str, conf: float = 0.25, iou: float = 0.70,
                 imgsz: int = 1280):
        """Evaluate; reports AP@0.5 per class and mAP@0.5."""
        metrics = self.model.val(data=data_yaml, conf=conf, iou=iou, imgsz=imgsz)
        return metrics

    def predict(self, source, conf: float = 0.25, iou: float = 0.70):
        return self.model.predict(source=source, conf=conf, iou=iou)


__all__ = ["BadmintonDetector", "load_ssl_backbone_into_yolo"]
