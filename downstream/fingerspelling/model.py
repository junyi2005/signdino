"""SignDinoFsDetector -- per-frame binary classifier for fingerspelling.

Pipeline:
    per_stream features ──► StreamFusion ──► small TransformerEncoder ──► Linear(d, 1) ──► σ

Loss: per-frame binary cross-entropy with a positional weight to balance
the rare-positive class.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..common.fusion import FusionConfig, StreamFusion


@dataclass
class FsConfig:
    fusion: FusionConfig = field(default_factory=lambda: FusionConfig(out_dim=512))
    hidden_depth: int = 2            # extra contextual layers on top of fusion
    hidden_heads: int = 8
    hidden_dropout: float = 0.1
    pos_weight: float = 5.0          # BCE positive-class weight (fingerspelling is rare)


class SignDinoFsDetector(nn.Module):
    def __init__(self, cfg: FsConfig):
        super().__init__()
        self.cfg = cfg
        self.fusion = StreamFusion(cfg.fusion)
        D = cfg.fusion.out_dim
        layer = nn.TransformerEncoderLayer(
            d_model=D, nhead=cfg.hidden_heads, dim_feedforward=4 * D,
            dropout=cfg.hidden_dropout, activation="gelu",
            batch_first=True, norm_first=True,
        )
        self.context = nn.TransformerEncoder(layer, num_layers=cfg.hidden_depth)
        self.norm = nn.LayerNorm(D)
        self.head = nn.Linear(D, 1)

    def forward(
        self,
        per_stream: Dict[str, torch.Tensor],
        attention_mask: torch.Tensor,                 # (B, T)
        frame_labels: Optional[torch.Tensor] = None,   # (B, T) 0/1
    ):
        fused = self.fusion(per_stream)               # (B, T, D)
        x = self.context(fused, src_key_padding_mask=~attention_mask.bool())
        x = self.norm(x)
        logits = self.head(x).squeeze(-1)             # (B, T)
        if frame_labels is None:
            return logits
        pw = logits.new_tensor(self.cfg.pos_weight)
        loss = F.binary_cross_entropy_with_logits(
            logits, frame_labels.float(), pos_weight=pw, reduction="none"
        )
        mask = attention_mask.float()
        return logits, (loss * mask).sum() / mask.sum().clamp(min=1.0)


class LiveSignDinoFsDetector(nn.Module):
    """Wraps SignDinoFsDetector with a LiveUpstream for full fine-tune of
    upstream encoders (DINOv3 stays frozen; no LoRA per SHuBERT § 4.4)."""

    def __init__(self, base: 'SignDinoFsDetector', live_upstream):
        super().__init__()
        self.base = base
        self.live_upstream = live_upstream

    def _encode_streams(self, batch_live):
        return self.live_upstream(
            batch_live["per_stream_crops"], batch_live["valid_mask"],
            batch_live["time_indices"],
            return_all_layers=self.base.fusion.cfg.use_layer_weighted_sum,
        )

    def forward(self, batch_live, frame_labels=None):
        per_stream = self._encode_streams(batch_live)
        return self.base(per_stream, batch_live["attention_mask"], frame_labels)
