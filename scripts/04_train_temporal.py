#!/usr/bin/env python3
"""Train MotionFormer for temporal stroke recognition (paper Sections 2.5, 3.2).

Operates on frozen per-frame SSL detection features over 10-frame windows
(400 ms at 25 fps). Expects pre-extracted feature windows; see
``docs/reproducing_results.md`` for the expected ``.npz`` layout
(``features``: (N, W, D), ``labels``: (N,), ``rally_id``: (N,)).

Example
-------
    python scripts/04_train_temporal.py --config configs/default.yaml \
        temporal.window=10 temporal.epochs=80
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from badmintonvision.utils.config import Config, set_seed  # noqa: E402


def load_feature_windows(npz_path):
    import numpy as np

    data = np.load(npz_path)
    return data["features"], data["labels"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--train-npz", default="data/temporal/train_windows.npz")
    ap.add_argument("--val-npz", default="data/temporal/val_windows.npz")
    ap.add_argument("overrides", nargs="*")
    args = ap.parse_args()

    cfg = Config.load(args.config).override(args.overrides)
    set_seed(cfg.project["seed"])

    import numpy as np
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    from badmintonvision.temporal.motionformer import MotionFormer

    device = cfg.project["device"] if torch.cuda.is_available() else "cpu"
    Xtr, ytr = load_feature_windows(args.train_npz)
    feat_dim = Xtr.shape[-1]

    model = MotionFormer(
        feat_dim=feat_dim,
        num_classes=len(cfg.classes),
        window=cfg.temporal["window"],
        d_model=cfg.temporal["d_model"],
        n_heads=cfg.temporal["n_heads"],
        n_layers=cfg.temporal["n_layers"],
        mlp_ratio=cfg.temporal["mlp_ratio"],
        dropout=cfg.temporal["dropout"],
    ).to(device)

    loader = DataLoader(
        TensorDataset(torch.tensor(Xtr, dtype=torch.float32),
                      torch.tensor(ytr, dtype=torch.long)),
        batch_size=cfg.temporal["batch_size"], shuffle=True, drop_last=True,
    )
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.temporal["lr"],
                            weight_decay=cfg.temporal["weight_decay"])
    crit = torch.nn.CrossEntropyLoss()

    for epoch in range(cfg.temporal["epochs"]):
        model.train()
        running = 0.0
        for xb, yb in loader:
            opt.zero_grad()
            loss = crit(model(xb.to(device)), yb.to(device))
            loss.backward()
            opt.step()
            running += loss.item()
        print(f"[motionformer] epoch {epoch + 1}/{cfg.temporal['epochs']} "
              f"loss={running / len(loader):.4f}")

    os.makedirs(cfg.paths["weights"], exist_ok=True)
    out = os.path.join(cfg.paths["weights"], "motionformer.pt")
    torch.save({"model": model.state_dict(), "config": dict(cfg.temporal)}, out)
    print(f"Saved MotionFormer weights -> {out}")


if __name__ == "__main__":
    main()
