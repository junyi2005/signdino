"""Multi-stream fusion + layer-weighted sum.

Two complementary modules:

1. `LayerWeightedSum`: for ONE stream, learn a softmax over its L+1
   Transformer-encoder layers and emit a single (T, D) per-frame tensor.
   This is the SHuBERT § 4.2 "weighted sum of all layers" trick that
   Tab. 6 shows is worth ~6 BLEU vs taking the last layer.

2. `StreamFusion`: across the 3 streams, layer-norm + linear-project each
   to a common dim, concat per-frame, then project once more to the
   downstream-task input dim (e.g. ByT5-Base d_model=1472). Mirrors
   SHuBERT § 3.2 fusion (concat → linear).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class FusionConfig:
    streams: List[str] = field(default_factory=lambda: ["lh", "rh", "face"])
    in_dim: int = 384                # per-stream feature dim (= temporal encoder embed_dim)
    proj_dim: int = 256              # per-stream projected dim
    out_dim: int = 1472              # downstream d_model (ByT5-Base = 1472)
    use_layer_weighted_sum: bool = True
    n_layers: int = 7                # depth + 1 (post-norm) layers exposed by extract_features.py


class LayerWeightedSum(nn.Module):
    """Learned softmax over L layers -> single (B, T, D) output."""

    def __init__(self, n_layers: int):
        super().__init__()
        self.weights = nn.Parameter(torch.zeros(n_layers))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (L, ..., D)  ->  (..., D). Softmax over dim 0."""
        w = F.softmax(self.weights, dim=0)
        # broadcast w along (L, 1, 1, ..., 1)
        shape = (-1,) + (1,) * (x.ndim - 2) + (1,)
        return (x * w.view(*shape)).sum(dim=0)


class StreamFusion(nn.Module):
    """Concat-then-project fusion across streams, with optional per-stream
    layer-weighted sum applied first."""

    def __init__(self, cfg: FusionConfig):
        super().__init__()
        self.cfg = cfg
        if cfg.use_layer_weighted_sum:
            self.layer_pool = nn.ModuleDict({s: LayerWeightedSum(cfg.n_layers) for s in cfg.streams})
        else:
            self.layer_pool = None
        self.layer_norm = nn.ModuleDict({s: nn.LayerNorm(cfg.in_dim) for s in cfg.streams})
        self.stream_proj = nn.ModuleDict({s: nn.Linear(cfg.in_dim, cfg.proj_dim) for s in cfg.streams})
        self.post_proj = nn.Linear(cfg.proj_dim * len(cfg.streams), cfg.out_dim)

    def forward(
        self,
        per_stream: Dict[str, torch.Tensor],          # stream -> (B, T, D)   if no layer pool
                                                      # stream -> (L, B, T, D) otherwise
    ) -> torch.Tensor:
        """Returns (B, T, out_dim) per-frame fused features."""
        outs = []
        for s in self.cfg.streams:
            x = per_stream[s]
            if self.layer_pool is not None:
                if x.dim() != 4:
                    raise ValueError(
                        f"stream {s!r}: layer-weighted fusion needs (L,B,T,D); got {tuple(x.shape)}"
                    )
                x = self.layer_pool[s](x)              # (B, T, D)
            x = self.layer_norm[s](x)
            x = self.stream_proj[s](x)                 # (B, T, proj_dim)
            outs.append(x)
        cat = torch.cat(outs, dim=-1)                  # (B, T, n*proj_dim)
        return self.post_proj(cat)                     # (B, T, out_dim)
