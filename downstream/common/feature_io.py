"""Read per-stream features dumped by `extract_features.py`.

Each video has up to 3 stream files: `<feature_root>/<video_id>/{lh,rh,face}.pt`.
A file contains:
    cls:               (D,)   float16
    patches:           (T, D) float16          -- final-layer per-frame tokens
    valid:             (T,)   bool
    patches_per_layer: (L+1, T, D) float16     -- optional (mode=all_layers)

`PerStreamFeatures` is one video's bundle of streams. `load_video_features`
returns it given a feature_root and a video_id.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import torch


STREAMS = ("lh", "rh", "face")


@dataclass
class PerStreamFeatures:
    video_id: str
    cls:    Dict[str, torch.Tensor]                       # stream -> (D,)
    patches: Dict[str, torch.Tensor]                      # stream -> (T_s, D)
    valid:   Dict[str, torch.Tensor]                      # stream -> (T_s,) bool
    per_layer: Optional[Dict[str, torch.Tensor]] = None   # stream -> (L+1, T_s, D)

    def num_frames(self) -> int:
        # streams may differ slightly in T due to padding; take the max
        return max(v.shape[0] for v in self.valid.values())

    def stream_dim(self, stream: str = "lh") -> int:
        return self.patches[stream].shape[-1]

    def num_layers(self) -> Optional[int]:
        if self.per_layer is None:
            return None
        return next(iter(self.per_layer.values())).shape[0]


def load_video_features(
    feature_root: str | Path,
    video_id: str,
    streams: List[str] = list(STREAMS),
    load_per_layer: bool = False,
) -> PerStreamFeatures:
    feature_root = Path(feature_root)
    cls: Dict[str, torch.Tensor] = {}
    patches: Dict[str, torch.Tensor] = {}
    valid: Dict[str, torch.Tensor] = {}
    per_layer: Dict[str, torch.Tensor] = {}
    for s in streams:
        path = feature_root / video_id / f"{s}.pt"
        if not path.exists():
            raise FileNotFoundError(f"missing feature file {path}")
        d = torch.load(path, map_location="cpu", weights_only=True)
        cls[s] = d["cls"].float()
        patches[s] = d["patches"].float()
        valid[s] = d["valid"].bool()
        if load_per_layer:
            if "patches_per_layer" not in d:
                raise ValueError(f"{path} has no per-layer features; re-extract with --mode all_layers")
            per_layer[s] = d["patches_per_layer"].float()
    return PerStreamFeatures(
        video_id=video_id, cls=cls, patches=patches, valid=valid,
        per_layer=per_layer if load_per_layer else None,
    )
