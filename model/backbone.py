"""Frozen per-frame DINOv3 embedder.

Wraps `torch.hub.load(<local dinov3 repo>, 'dinov3_vitb16', source='local')`
so we can call `embedder(frames)` where `frames` is (N, 3, H, W) and get
back (N, 768) CLS embeddings.

DINOv3 ViT-B/16 was trained at 224 with patch 16 (-> 14x14 = 196 patches).
We feed cropped LH/RH/Face regions at this resolution.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn


DEFAULT_DINOV3_REPO = "/path/to/dinov3"
DEFAULT_INPUT_SIZE = 224


class FrozenDinov3FrameEmbedder(nn.Module):
    """Wraps a DINOv3 ViT-B/16 in eval mode with all parameters frozen.

    forward(x):
        x: (N, 3, H, W) -- already normalised with ImageNet mean/std and
           resized to `input_size`x`input_size`.
        returns: (N, embed_dim) CLS embeddings.

    Call .embed_dim for the output dim (768 for ViT-B/16).
    """

    def __init__(
        self,
        repo_path: str = DEFAULT_DINOV3_REPO,
        model_name: str = "dinov3_vitb16",
        weights_path: Optional[str] = None,
        input_size: int = DEFAULT_INPUT_SIZE,
        pretrained: bool = True,
    ):
        super().__init__()
        self.input_size = input_size
        self.repo_path = str(Path(repo_path).resolve())
        self.model_name = model_name

        kwargs = {"pretrained": pretrained}
        if weights_path is not None:
            kwargs["weights"] = weights_path
        self.backbone = torch.hub.load(
            self.repo_path, self.model_name, source="local", **kwargs
        )
        self.backbone.eval()
        for p in self.backbone.parameters():
            p.requires_grad_(False)

        # dinov3 ViT exposes .embed_dim on the model object
        self.embed_dim = int(getattr(self.backbone, "embed_dim", 768))

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (N, 3, H, W) -> (N, embed_dim)."""
        feats = self.backbone.forward_features(x)
        # dict with 'x_norm_clstoken', 'x_norm_patchtokens', etc.
        return feats["x_norm_clstoken"]

    def train(self, mode: bool = True):  # keep backbone in eval no matter what
        super().train(mode)
        self.backbone.eval()
        return self
