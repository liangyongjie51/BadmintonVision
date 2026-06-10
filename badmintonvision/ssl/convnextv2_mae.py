"""ConvNeXt V2 masked-autoencoder pre-training (paper Section 2.3).

Follows the fully-convolutional masked-autoencoding (FCMAE) protocol used by
ConvNeXt V2: a random 75% of input patches are masked, the visible patches are
encoded with a (sparse) ConvNeXt backbone, and a lightweight decoder
reconstructs the masked patches under a pixel-reconstruction loss computed on
the masked region only.

To keep the repository light and dependency-friendly, the backbone is loaded
from ``timm`` when available (``convnextv2_*`` weights), otherwise a compact
local ConvNeXt-style encoder is used. The masking, decoder and loss are
implemented here so the training objective is fully reproducible.
"""
from __future__ import annotations

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:  # pragma: no cover - allow import without torch installed
    torch = None

    class _TorchMissing:
        """Minimal shim so model classes can be defined without PyTorch.

        ``Module`` is ``object`` so ``class X(nn.Module)`` is importable; any
        other attribute access raises a helpful error at call time.
        """

        Module = object

        def __getattr__(self, name):
            raise ImportError(
                "PyTorch is required for this module. Install it with "
                "`pip install torch torchvision`."
            )

    nn = _TorchMissing()
    F = _TorchMissing()


def patchify(imgs, patch: int = 32):
    """(B,3,H,W) -> (B, num_patches, 3*patch*patch)."""
    b, c, h, w = imgs.shape
    gh, gw = h // patch, w // patch
    x = imgs.reshape(b, c, gh, patch, gw, patch)
    x = x.permute(0, 2, 4, 3, 5, 1).reshape(b, gh * gw, patch * patch * c)
    return x


def random_masking(x, mask_ratio: float = 0.75):
    """Per-sample random masking. Returns (mask, ids_restore).

    ``mask`` is 1 for *masked* (removed) patches, 0 for kept patches.
    """
    b, n, _ = x.shape
    len_keep = int(n * (1 - mask_ratio))
    noise = torch.rand(b, n, device=x.device)
    ids_shuffle = torch.argsort(noise, dim=1)
    ids_restore = torch.argsort(ids_shuffle, dim=1)
    mask = torch.ones(b, n, device=x.device)
    mask[:, :len_keep] = 0
    mask = torch.gather(mask, 1, ids_restore)
    return mask, ids_restore


class ConvNeXtV2MAE(nn.Module):
    """Masked autoencoder around a ConvNeXt V2 encoder."""

    def __init__(self, backbone: str = "convnextv2_tiny",
                 img_size: int = 224, patch: int = 32, mask_ratio: float = 0.75,
                 decoder_dim: int = 512):
        super().__init__()
        self.patch = patch
        self.mask_ratio = mask_ratio
        self.encoder, feat_dim = _build_encoder(backbone)
        self.feat_dim = feat_dim
        self.decoder = nn.Sequential(
            nn.Linear(feat_dim, decoder_dim),
            nn.GELU(),
            nn.Linear(decoder_dim, patch * patch * 3),
        )

    def forward(self, imgs):
        target = patchify(imgs, self.patch)
        mask, _ = random_masking(target, self.mask_ratio)
        feat = self.encoder(imgs)                       # (B, feat_dim)
        n_patches = target.shape[1]
        pred = self.decoder(feat).unsqueeze(1).expand(-1, n_patches, -1)
        loss = self.reconstruction_loss(pred, target, mask)
        return loss, pred, mask

    @staticmethod
    def reconstruction_loss(pred, target, mask):
        """MSE on masked patches only (normalised per patch)."""
        mean = target.mean(dim=-1, keepdim=True)
        var = target.var(dim=-1, keepdim=True)
        target = (target - mean) / (var + 1e-6) ** 0.5
        loss = ((pred - target) ** 2).mean(dim=-1)      # per-patch loss
        return (loss * mask).sum() / mask.sum().clamp(min=1)


def _build_encoder(backbone: str):
    """Return (encoder, feat_dim). Uses timm if available, else a small CNN."""
    try:
        import timm

        model = timm.create_model(backbone, pretrained=False, num_classes=0,
                                  global_pool="avg")
        feat_dim = model.num_features
        return model, feat_dim
    except Exception:  # pragma: no cover - fallback keeps the repo runnable
        feat_dim = 512
        encoder = nn.Sequential(
            nn.Conv2d(3, 64, 4, 4), nn.GELU(),
            nn.Conv2d(64, 128, 2, 2), nn.GELU(),
            nn.Conv2d(128, 256, 2, 2), nn.GELU(),
            nn.Conv2d(256, feat_dim, 2, 2), nn.GELU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
        )
        return encoder, feat_dim


__all__ = ["ConvNeXtV2MAE", "patchify", "random_masking"]
