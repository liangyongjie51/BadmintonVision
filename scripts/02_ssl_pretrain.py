#!/usr/bin/env python3
"""Self-supervised pre-training of the detection backbone (paper Section 2.3).

Trains either ConvNeXt V2 MAE (default) or SimSiam on the unlabeled frame pool
(139,501 images in the paper) and saves the encoder weights for the detector.

Example
-------
    python scripts/02_ssl_pretrain.py --config configs/default.yaml \
        ssl.method=convnextv2_mae ssl.epochs=200
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from badmintonvision.utils.config import Config, set_seed  # noqa: E402


def build_image_loader(frames_dir, image_size, batch_size, num_workers, two_views):
    import torch
    from torch.utils.data import DataLoader, Dataset
    from torchvision import transforms
    from PIL import Image
    import glob

    aug = transforms.Compose([
        transforms.RandomResizedCrop(image_size, scale=(0.2, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    class FrameDataset(Dataset):
        def __init__(self):
            self.paths = sorted(glob.glob(os.path.join(frames_dir, "**", "*.jpg"),
                                          recursive=True))
            if not self.paths:
                raise RuntimeError(f"No .jpg frames found under {frames_dir}")

        def __len__(self):
            return len(self.paths)

        def __getitem__(self, i):
            img = Image.open(self.paths[i]).convert("RGB")
            return (aug(img), aug(img)) if two_views else aug(img)

    return DataLoader(FrameDataset(), batch_size=batch_size, shuffle=True,
                      num_workers=num_workers, drop_last=True, pin_memory=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("overrides", nargs="*")
    args = ap.parse_args()

    cfg = Config.load(args.config).override(args.overrides)
    set_seed(cfg.project["seed"])

    import torch
    from torch.optim import AdamW

    device = cfg.project["device"] if torch.cuda.is_available() else "cpu"
    method = cfg.ssl["method"]
    two_views = method == "simsiam"
    loader = build_image_loader(
        cfg.paths["frames"], cfg.ssl["image_size"], cfg.ssl["batch_size"],
        cfg.project["num_workers"], two_views,
    )

    if method == "simsiam":
        from badmintonvision.ssl.convnextv2_mae import _build_encoder
        from badmintonvision.ssl.simsiam import SimSiam, simsiam_loss
        encoder, feat_dim = _build_encoder(cfg.ssl["backbone"])
        model = SimSiam(encoder, feat_dim, cfg.ssl["proj_dim"]).to(device)
        forward_loss = lambda batch: simsiam_loss(*model(batch[0].to(device),
                                                          batch[1].to(device)))
    else:
        from badmintonvision.ssl.convnextv2_mae import ConvNeXtV2MAE
        model = ConvNeXtV2MAE(cfg.ssl["backbone"], cfg.ssl["image_size"],
                              mask_ratio=cfg.ssl["mask_ratio"]).to(device)
        forward_loss = lambda batch: model(batch.to(device))[0]

    opt = AdamW(model.parameters(), lr=cfg.ssl["lr"],
                weight_decay=cfg.ssl["weight_decay"])
    os.makedirs(cfg.paths["weights"], exist_ok=True)

    for epoch in range(cfg.ssl["epochs"]):
        model.train()
        running = 0.0
        for batch in loader:
            opt.zero_grad()
            loss = forward_loss(batch)
            loss.backward()
            opt.step()
            running += loss.item()
        print(f"[ssl:{method}] epoch {epoch + 1}/{cfg.ssl['epochs']} "
              f"loss={running / len(loader):.4f}")

    encoder = model.encoder if hasattr(model, "encoder") else model
    out = os.path.join(cfg.paths["weights"], f"ssl_{method}_encoder.pt")
    torch.save({"encoder": encoder.state_dict(), "config": dict(cfg.ssl)}, out)
    print(f"Saved SSL encoder weights -> {out}")


if __name__ == "__main__":
    main()
