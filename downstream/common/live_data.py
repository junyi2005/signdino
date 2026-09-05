"""Live-mode dataset: load raw per-frame crops from disk for a video.

For downstream tasks in `live` mode we forward crops through the frozen
DINOv3 backbone + the (possibly fine-tuned) SignDINO temporal encoders
during training. The dataset returns:

    'video_id':         str
    'per_stream_crops': {stream: (T, 3, H, W) float32 -- ImageNet-normalised}
    'valid_mask':       {stream: (T,) bool}
    'time_indices':     {stream: (T,) int64}                # absolute frame idx
    'attention_mask':   (T,) int    (union of stream validity, length = T)

Used by all 3 downstream tasks; each task wraps it with its own
task-specific labels (text / class / per-frame label).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

import sys
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT))
from data.augment import build_image_eval                          # noqa: E402
from data.crop_io import STREAMS, VideoCropEntry, VideoCropIndex   # noqa: E402


@dataclass
class LiveStreamConfig:
    crop_root: str
    streams: List[str] = None
    input_size: int = 224
    max_frames: int = 512
    frame_stride: int = 1            # subsample to keep per-video frame count manageable
    split: str = "train"


class LiveStreamLoader:
    """Pure functional: given a `VideoCropEntry`, return the dict above."""

    def __init__(self, cfg: LiveStreamConfig):
        self.cfg = cfg
        if cfg.streams is None:
            cfg.streams = list(STREAMS)
        self.transform = build_image_eval(cfg.input_size)

    def load(self, entry: VideoCropEntry) -> Dict:
        T_total = entry.num_frames()
        idx = np.arange(0, T_total, self.cfg.frame_stride)[: self.cfg.max_frames]
        T = len(idx)
        per_stream_crops: Dict[str, torch.Tensor] = {}
        valid_mask: Dict[str, torch.Tensor] = {}
        time_indices: Dict[str, torch.Tensor] = {}
        attn = np.zeros(T, dtype=np.int64)
        for s in self.cfg.streams:
            v_full = entry.valid_mask(s)
            v = v_full[idx]
            crops = []
            for ii, t in enumerate(idx):
                if v[ii]:
                    img = entry.load_frame(s, int(t))
                else:
                    img = Image.new("RGB", tuple(entry.crop_size(s)), (0, 0, 0))
                crops.append(self.transform(img))
            per_stream_crops[s] = torch.stack(crops, dim=0)                  # (T, 3, H, W)
            valid_mask[s] = torch.from_numpy(v.copy())
            time_indices[s] = torch.from_numpy(idx.astype(np.int64))
            attn |= v.astype(np.int64)
        return {
            "video_id": entry.video_id,
            "per_stream_crops": per_stream_crops,
            "valid_mask": valid_mask,
            "time_indices": time_indices,
            "attention_mask": torch.from_numpy(attn),
        }


class LiveStreamDataset(Dataset):
    """Generic live dataset. Wrap with a task-specific subclass / adapter
    that adds labels (text, class, intervals)."""

    def __init__(self, cfg: LiveStreamConfig, index: Optional[VideoCropIndex] = None):
        self.cfg = cfg
        self.index = index or VideoCropIndex.from_root(cfg.crop_root, splits=[cfg.split])
        self.entries: List[VideoCropEntry] = self.index.by_split.get(cfg.split, [])
        self.loader = LiveStreamLoader(cfg)

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, i: int):
        return self.loader.load(self.entries[i])


def live_collate(batch: List[Dict]) -> Dict:
    """Pad per-stream crops to max T in batch. All streams in one sample share T."""
    streams = list(batch[0]["per_stream_crops"].keys())
    Tmax = max(b["attention_mask"].numel() for b in batch)
    B = len(batch)
    per_stream_crops: Dict[str, torch.Tensor] = {}
    valid_mask: Dict[str, torch.Tensor] = {}
    time_indices: Dict[str, torch.Tensor] = {}
    for s in streams:
        c0 = batch[0]["per_stream_crops"][s]
        _, C, H, W = c0.shape
        crops_buf = torch.zeros(B, Tmax, C, H, W, dtype=c0.dtype)
        valid_buf = torch.zeros(B, Tmax, dtype=torch.bool)
        time_buf = torch.zeros(B, Tmax, dtype=torch.long)
        for i, b in enumerate(batch):
            t = b["per_stream_crops"][s].shape[0]
            crops_buf[i, :t] = b["per_stream_crops"][s]
            valid_buf[i, :t] = b["valid_mask"][s]
            time_buf[i, :t] = b["time_indices"][s]
        per_stream_crops[s] = crops_buf
        valid_mask[s] = valid_buf
        time_indices[s] = time_buf
    attn = torch.zeros(B, Tmax, dtype=torch.long)
    for i, b in enumerate(batch):
        attn[i, : b["attention_mask"].numel()] = b["attention_mask"]
    return {
        "video_id": [b["video_id"] for b in batch],
        "per_stream_crops": per_stream_crops,
        "valid_mask": valid_mask,
        "time_indices": time_indices,
        "attention_mask": attn,
    }
