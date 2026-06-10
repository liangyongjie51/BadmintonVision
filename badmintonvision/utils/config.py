"""Configuration loading and global seeding utilities."""
from __future__ import annotations

import os
import random
from dataclasses import dataclass
from typing import Any

import yaml


class Config(dict):
    """Dict with attribute access and dotted-key overrides.

    Example
    -------
    >>> cfg = Config.load("configs/default.yaml")
    >>> cfg.detection["epochs"]
    100
    >>> cfg.override(["detection.epochs=150", "project.seed=0"])
    """

    def __getattr__(self, item: str) -> Any:
        try:
            return self[item]
        except KeyError as exc:  # pragma: no cover
            raise AttributeError(item) from exc

    @classmethod
    def load(cls, path: str) -> "Config":
        with open(path, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
        return cls(data)

    def override(self, pairs: list[str]) -> "Config":
        """Apply ``a.b.c=value`` overrides parsed from the command line."""
        for pair in pairs:
            if "=" not in pair:
                continue
            key, value = pair.split("=", 1)
            node: Any = self
            parts = key.split(".")
            for part in parts[:-1]:
                node = node[part]
            node[parts[-1]] = yaml.safe_load(value)  # type: ignore[index]
        return self


def set_seed(seed: int = 42, deterministic: bool = True) -> None:
    """Seed Python, NumPy and (if available) PyTorch for reproducibility."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:  # pragma: no cover
        pass
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:  # pragma: no cover
        pass


@dataclass
class CourtGeometry:
    """BWF singles court geometry used for homography (Section 2.6).

    The singles court is 13.40 m long and 5.18 m wide; 6.10 m is the *doubles*
    width. The modelled region is one half-court (net to baseline).
    """

    length_half_m: float = 6.70   # net-to-baseline (half court)
    width_singles_m: float = 5.18  # singles width

    @property
    def corners_m(self):
        """Return the four half-court corners in metres (x=width, y=length).

        Order: near-left, near-right, far-right, far-left (clockwise),
        suitable for ``cv2.findHomography`` against the matching image points.
        """
        w, l = self.width_singles_m, self.length_half_m
        return [(0.0, 0.0), (w, 0.0), (w, l), (0.0, l)]
