"""MotionFormer: temporal action recognition (paper Section 2.5).

MotionFormer is **not** a new attention operator; it is a lightweight temporal
-aggregation module adapted from the standard Transformer encoder (Vaswani et
al.), applied on top of frozen self-supervised detection features. It ingests a
short window of ``W`` per-frame feature vectors (W = 10 frames at 25 fps =
400 ms, chosen to span a full stroke cycle: preparation, execution, recovery)
and predicts the stroke class for the window.

Components: a linear input projection, learnable positional embeddings, a
``CLS`` token, ``n_layers`` standard Transformer-encoder blocks, and a linear
classification head on the pooled ``CLS`` representation.
"""
from __future__ import annotations

try:
    import torch
    import torch.nn as nn
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


class MotionFormer(nn.Module):
    def __init__(
        self,
        feat_dim: int,            # dim of per-frame backbone features
        num_classes: int,
        window: int = 10,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 4,
        mlp_ratio: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.window = window
        self.input_proj = nn.Linear(feat_dim, d_model)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos_embed = nn.Parameter(torch.zeros(1, window + 1, d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * mlp_ratio,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, num_classes)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, frame_feats):
        """frame_feats: (B, W, feat_dim) -> logits (B, num_classes)."""
        b = frame_feats.shape[0]
        x = self.input_proj(frame_feats)                       # (B, W, d)
        cls = self.cls_token.expand(b, -1, -1)                 # (B, 1, d)
        x = torch.cat([cls, x], dim=1) + self.pos_embed        # (B, W+1, d)
        x = self.encoder(x)
        cls_out = self.norm(x[:, 0])                           # pooled CLS
        return self.head(cls_out)


__all__ = ["MotionFormer"]
