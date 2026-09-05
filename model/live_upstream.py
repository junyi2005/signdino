"""LiveUpstream -- wraps frozen DINOv3 + 3 trained SignDINO temporal encoders
so that downstream tasks can fine-tune the encoders end-to-end during
training (the SHuBERT Table 7 / DINOv3 dinotxt "fine-tune vision tower"
mode).

Inputs are per-stream crops (B, T, 3, H, W); the DINOv3 backbone is in
eval mode with `no_grad`, so its forward pass is cheap and produces no
graph. The temporal encoders run with grad enabled (or wrapped in LoRA
by `apply_lora_to`) and gradients flow back through them to the fusion +
downstream head.

The encoders are initialised from a stage-2 (Gram-refined) SSL checkpoint
of each stream's `SignTemporalDinoModel`; we pull the **teacher** encoder
weights, which are the EMA-averaged target representations.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn as nn

from .backbone import FrozenDinov3FrameEmbedder
from .temporal_encoder import TemporalTransformer


@dataclass
class LiveUpstreamConfig:
    streams: List[str]                                # ['lh', 'rh', 'face']
    ckpts: Dict[str, str]                             # stream -> path to SignDINO SSL ckpt
    dinov3_repo: str = "/path/to/dinov3"
    dinov3_model_name: str = "dinov3_vitb16"
    dinov3_weights_path: Optional[str] = None
    dinov3_pretrained: bool = True
    dinov3_input_size: int = 224
    share_backbone: bool = True                       # only one DINOv3 instance shared across streams


class LiveUpstream(nn.Module):
    def __init__(self, cfg: LiveUpstreamConfig):
        super().__init__()
        self.cfg = cfg
        self.frame_embedder = FrozenDinov3FrameEmbedder(
            repo_path=cfg.dinov3_repo, model_name=cfg.dinov3_model_name,
            weights_path=cfg.dinov3_weights_path, input_size=cfg.dinov3_input_size,
            pretrained=cfg.dinov3_pretrained,
        )
        # per-stream trainable temporal encoders
        self.encoders = nn.ModuleDict()
        self.encoder_dim: Optional[int] = None
        for stream in cfg.streams:
            ckpt_path = cfg.ckpts.get(stream)
            if ckpt_path is None:
                raise ValueError(f"missing ckpt for stream {stream!r} in LiveUpstreamConfig.ckpts")
            sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            mcfg = sd["cfg"]["model"]
            enc = TemporalTransformer(
                in_dim=mcfg["in_dim"], embed_dim=mcfg["embed_dim"],
                depth=mcfg["depth"], num_heads=mcfg["num_heads"],
                mlp_ratio=mcfg["mlp_ratio"], dropout=0.0,
                max_len=mcfg["max_len"],
            )
            # pull the teacher_encoder.* subset from the SSL state_dict
            teacher_sd = {
                k[len("teacher_encoder."):]: v
                for k, v in sd["model"].items() if k.startswith("teacher_encoder.")
            }
            missing, unexpected = enc.load_state_dict(teacher_sd, strict=False)
            if len(missing) > 5 or len(unexpected) > 5:
                # heuristic: more than a couple of mismatches probably means wrong ckpt
                raise RuntimeError(
                    f"loading teacher_encoder for stream {stream!r}: "
                    f"{len(missing)} missing, {len(unexpected)} unexpected"
                )
            self.encoders[stream] = enc
            if self.encoder_dim is None:
                self.encoder_dim = mcfg["embed_dim"]

    def forward(
        self,
        per_stream_crops: Dict[str, torch.Tensor],          # stream -> (B, T, 3, H, W)
        valid_mask: Dict[str, torch.Tensor],                 # stream -> (B, T)
        time_indices: Optional[Dict[str, torch.Tensor]] = None,  # stream -> (B, T)
        return_all_layers: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """Forward the streams through DINOv3 (frozen) then their temporal encoders.

        Returns:
          stream -> (L+1, B, T, D)   if return_all_layers (for layer-weighted fusion)
          stream -> (B, T, D)        otherwise.
        """
        out: Dict[str, torch.Tensor] = {}
        for stream in self.cfg.streams:
            crops = per_stream_crops[stream]                 # (B, T, 3, H, W)
            B, T, C, H, W = crops.shape
            with torch.no_grad():
                frames = self.frame_embedder(crops.reshape(B * T, C, H, W))  # (B*T, 768)
            frames = frames.reshape(B, T, -1).detach()
            vm = valid_mask[stream]
            tidx = time_indices[stream] if time_indices is not None else None
            enc = self.encoders[stream]
            if return_all_layers:
                _, _, all_layers = enc(frames, vm, None, tidx, return_all_layers=True)
                # all_layers: (L+1, B, T+1, D) -> drop CLS at index 0 -> (L+1, B, T, D)
                out[stream] = all_layers[:, :, 1:]
            else:
                _, patches = enc(frames, vm, None, tidx)
                out[stream] = patches                        # (B, T, D)
        return out
