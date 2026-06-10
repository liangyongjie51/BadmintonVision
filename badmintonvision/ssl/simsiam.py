"""SimSiam self-supervised pre-training (paper Section 2.3, Eq. 1).

The objective is the *negative cosine similarity* between the two augmented
views, with a stop-gradient on the target branch:

    D(p, z) = - (p . z) / (||p||_2 * ||z||_2)          # Eq. 1 denominator
    L = 0.5 * D(p1, stopgrad(z2)) + 0.5 * D(p2, stopgrad(z1))

Feature vectors are L2-normalised before the dot product, so the denominator is
exactly the product of the two Euclidean (L2) norms ||z1|| * ||z2|| (this is the
clarification added during revision).

This is a faithful, runnable re-implementation; the encoder defaults to a
ConvNeXt V2 backbone but any feature extractor returning a pooled embedding can
be supplied.
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


def _mlp(in_dim: int, hid_dim: int, out_dim: int, last_bn: bool = True) -> "nn.Module":
    layers = [
        nn.Linear(in_dim, hid_dim, bias=False),
        nn.BatchNorm1d(hid_dim),
        nn.ReLU(inplace=True),
        nn.Linear(hid_dim, out_dim, bias=not last_bn),
    ]
    if last_bn:
        layers.append(nn.BatchNorm1d(out_dim, affine=False))
    return nn.Sequential(*layers)


class SimSiam(nn.Module):
    """SimSiam model: encoder f, projector g (-> z), predictor h (-> p)."""

    def __init__(self, encoder: "nn.Module", feat_dim: int, proj_dim: int = 2048):
        super().__init__()
        self.encoder = encoder
        self.projector = _mlp(feat_dim, proj_dim, proj_dim, last_bn=True)
        # bottleneck predictor (proj_dim -> proj_dim/4 -> proj_dim)
        self.predictor = nn.Sequential(
            nn.Linear(proj_dim, proj_dim // 4, bias=False),
            nn.BatchNorm1d(proj_dim // 4),
            nn.ReLU(inplace=True),
            nn.Linear(proj_dim // 4, proj_dim),
        )

    def forward(self, x1, x2):
        z1 = self.projector(self.encoder(x1))
        z2 = self.projector(self.encoder(x2))
        p1 = self.predictor(z1)
        p2 = self.predictor(z2)
        # stop-gradient on the target branch (z)
        return p1, p2, z1.detach(), z2.detach()


def negative_cosine_similarity(p, z):
    """D(p, z) = - <p_hat, z_hat>; z is already detached (stop-gradient)."""
    p = F.normalize(p, dim=1)          # L2-normalise -> denominator = ||p||*||z||
    z = F.normalize(z, dim=1)
    return -(p * z).sum(dim=1).mean()


def simsiam_loss(p1, p2, z1, z2):
    """Symmetric SimSiam loss (Eq. 1)."""
    return 0.5 * negative_cosine_similarity(p1, z2) + \
           0.5 * negative_cosine_similarity(p2, z1)


__all__ = ["SimSiam", "simsiam_loss", "negative_cosine_similarity"]
