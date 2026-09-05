"""SignDinoISLR -- isolated sign language recognition head.

Pipeline (mirrors SHuBERT § 4.3):

    per_stream features ──► StreamFusion ──► time-average over frames ──► BatchNorm ──► Linear ──► class logits

The "average across the time dimension" is done with attention-masking so
padded frames are excluded.

Optional LoRA mode: the upstream SignDINO temporal encoders are loaded
live (via a separate path -- see `live_upstream`), wrapped with rank-1
LoRA, while the base weights are frozen. Implemented in `lora.py`. The
trainer code distinguishes mode='features' (default) vs mode='live'.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..common.fusion import FusionConfig, StreamFusion


@dataclass
class ISLRConfig:
    num_classes: int = 1000
    fusion: FusionConfig = field(default_factory=lambda: FusionConfig(out_dim=768))
    label_smoothing: float = 0.1


class SignDinoISLR(nn.Module):
    def __init__(self, cfg: ISLRConfig):
        super().__init__()
        self.cfg = cfg
        self.fusion = StreamFusion(cfg.fusion)
        self.bn = nn.BatchNorm1d(cfg.fusion.out_dim)
        self.classifier = nn.Linear(cfg.fusion.out_dim, cfg.num_classes)

    def forward(
        self,
        per_stream: Dict[str, torch.Tensor],          # stream -> (B,T,D) or (L,B,T,D)
        attention_mask: torch.Tensor,                  # (B, T) 1/0
        labels: Optional[torch.Tensor] = None,         # (B,)
    ):
        fused = self.fusion(per_stream)                # (B, T, out_dim)
        m = attention_mask.unsqueeze(-1).float()
        pooled = (fused * m).sum(dim=1) / m.sum(dim=1).clamp(min=1.0)   # (B, out_dim)
        pooled = self.bn(pooled)
        logits = self.classifier(pooled)               # (B, num_classes)

        if labels is None:
            return logits
        loss = F.cross_entropy(logits, labels, label_smoothing=self.cfg.label_smoothing)
        return logits, loss


class LiveSignDinoISLR(nn.Module):
    """Wraps SignDinoISLR with a LiveUpstream (encoders wrapped in rank-1
    LoRA via `apply_lora_to`) so the full SHuBERT § 4.3 protocol works:
    DINOv3 frozen, SignDINO encoder weights frozen, LoRA-rank-1 trained.
    The base SignDinoISLR head (fusion + bn + classifier) trains normally.
    """

    def __init__(self, base: 'SignDinoISLR', live_upstream):
        super().__init__()
        self.base = base
        self.live_upstream = live_upstream

    def _encode_streams(self, batch_live):
        return self.live_upstream(
            batch_live["per_stream_crops"], batch_live["valid_mask"],
            batch_live["time_indices"],
            return_all_layers=self.base.fusion.cfg.use_layer_weighted_sum,
        )

    def forward(self, batch_live, labels=None):
        per_stream = self._encode_streams(batch_live)
        return self.base(per_stream, batch_live["attention_mask"], labels)
