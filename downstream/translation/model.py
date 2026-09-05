"""SignDinoTranslator -- ByT5-Base decoder over fused 3-stream per-frame features.

Architecture (mirrors SHuBERT § 4.2):

    per_stream features  ──► StreamFusion (LayerWeightedSum per stream → LN → Linear(384→256) ×3 → concat → Linear(768→1472))
                                                                 │
                                                                 ▼
                                              inputs_embeds to ByT5-Base ENCODER
                                                                 │
                                                                 ▼
                                                ByT5 cross-attention DECODER ──► UTF-8 byte logits

ByT5-Base d_model = 1472 (note: byte vocabulary, 384 tokens + 125 specials).
Loss: cross-entropy with label-smoothing 0.2 (SHuBERT § 4.2).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..common.fusion import FusionConfig, StreamFusion


@dataclass
class TranslatorConfig:
    backbone_id: str = "google/byt5-base"
    fusion: FusionConfig = field(default_factory=lambda: FusionConfig(out_dim=1472))
    label_smoothing: float = 0.2
    max_gen_length: int = 384
    num_beams: int = 5
    length_penalty: float = 0.6


class SignDinoTranslator(nn.Module):
    def __init__(self, cfg: TranslatorConfig):
        super().__init__()
        from transformers import AutoTokenizer, T5ForConditionalGeneration
        self.cfg = cfg
        self.byt5 = T5ForConditionalGeneration.from_pretrained(cfg.backbone_id)
        d_model = int(self.byt5.config.d_model)
        cfg.fusion.out_dim = d_model
        self.fusion = StreamFusion(cfg.fusion)
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.backbone_id)

    def encode(self, per_stream: Dict[str, torch.Tensor], attention_mask: torch.Tensor):
        fused = self.fusion(per_stream)                         # (B, T, d_model)
        # ByT5 expects (B, T, d_model) as inputs_embeds on the encoder
        return self.byt5.encoder(
            inputs_embeds=fused,
            attention_mask=attention_mask,
            return_dict=True,
        )

    def forward(
        self,
        per_stream: Dict[str, torch.Tensor],
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        encoder_outputs = self.encode(per_stream, attention_mask)
        out = self.byt5(
            encoder_outputs=encoder_outputs,
            attention_mask=attention_mask,
            labels=labels,
        )
        # T5's default loss is CE without smoothing. Recompute with smoothing
        # when self.training only -- otherwise use the val-mode unsmoothed CE
        # for comparable validation numbers.
        if self.training and self.cfg.label_smoothing > 0:
            logits = out.logits.float()
            loss = _label_smoothed_nll(logits, labels, eps=self.cfg.label_smoothing,
                                       ignore_index=-100)
            return loss
        return out.loss

    @torch.no_grad()
    def generate(
        self,
        per_stream: Dict[str, torch.Tensor],
        attention_mask: torch.Tensor,
        max_length: Optional[int] = None,
        num_beams: Optional[int] = None,
    ) -> torch.Tensor:
        encoder_outputs = self.encode(per_stream, attention_mask)
        return self.byt5.generate(
            encoder_outputs=encoder_outputs,
            attention_mask=attention_mask,
            max_length=max_length or self.cfg.max_gen_length,
            num_beams=num_beams or self.cfg.num_beams,
            length_penalty=self.cfg.length_penalty,
            early_stopping=True,
        )


class LiveSignDinoTranslator(nn.Module):
    """Wraps SignDinoTranslator with a LiveUpstream so the entire stack
    (DINOv3 frozen + SignDINO temporal encoders + fusion + ByT5) can be
    fine-tuned end-to-end. The trainer assigns lr/10 to the SignDINO
    encoder parameters (SHuBERT § 4.2 protocol)."""

    def __init__(self, base: 'SignDinoTranslator', live_upstream):
        super().__init__()
        self.base = base
        self.live_upstream = live_upstream

    def _encode_streams(self, batch_live):
        return self.live_upstream(
            batch_live["per_stream_crops"], batch_live["valid_mask"],
            batch_live["time_indices"],
            return_all_layers=self.base.fusion.cfg.use_layer_weighted_sum,
        )

    def forward(self, batch_live):
        per_stream = self._encode_streams(batch_live)
        return self.base(per_stream, batch_live["attention_mask"], batch_live["labels"])

    @torch.no_grad()
    def generate(self, batch_live, max_length=None, num_beams=None):
        per_stream = self._encode_streams(batch_live)
        return self.base.generate(per_stream, batch_live["attention_mask"],
                                  max_length=max_length, num_beams=num_beams)


def _label_smoothed_nll(logits: torch.Tensor, labels: torch.Tensor, eps: float, ignore_index: int = -100) -> torch.Tensor:
    """Standard label-smoothed cross-entropy. logits: (B, T, V); labels: (B, T)."""
    V = logits.shape[-1]
    logp = F.log_softmax(logits, dim=-1)
    nll = -logp.gather(-1, labels.clamp(min=0).unsqueeze(-1)).squeeze(-1)
    smooth = -logp.mean(dim=-1)
    loss = (1.0 - eps) * nll + eps * smooth
    mask = (labels != ignore_index).float()
    return (loss * mask).sum() / mask.sum().clamp(min=1)
