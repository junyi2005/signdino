"""Per-video, per-stream cache of frozen DINOv3 frame embeddings.

Layout:
    <crop_root>/.embedding_cache/<backbone_tag>/<video_id>/<stream>.pt

Each .pt holds a {'embeds': float16 [T, D], 'valid': bool [T]} dict. T is the
number of frames in the manifest for that video; invalid frames hold zeros.
Reading is a single torch.load() per video/stream.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch


class EmbeddingCache:
    def __init__(self, crop_root: str | Path, backbone_tag: str = "dinov3_vitb16_224"):
        self.crop_root = Path(crop_root)
        self.backbone_tag = backbone_tag
        self.base = self.crop_root / ".embedding_cache" / backbone_tag
        self.base.mkdir(parents=True, exist_ok=True)

    def path(self, video_id: str, stream: str) -> Path:
        return self.base / video_id / f"{stream}.pt"

    def exists(self, video_id: str, stream: str) -> bool:
        return self.path(video_id, stream).exists()

    def save(self, video_id: str, stream: str, embeds: torch.Tensor, valid: np.ndarray) -> None:
        out_path = self.path(video_id, stream)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "embeds": embeds.detach().to(torch.float16).cpu().contiguous(),
                "valid": torch.from_numpy(valid.astype(np.bool_)),
            },
            out_path,
        )

    def load(self, video_id: str, stream: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (embeds [T, D] float32, valid [T] bool)."""
        d = torch.load(self.path(video_id, stream), map_location="cpu", weights_only=True)
        return d["embeds"].float(), d["valid"]
