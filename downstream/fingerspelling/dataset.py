"""Fingerspelling dataset + collate.

A sample is (video_id, intervals, num_frames). The dataset builds a
per-frame binary label vector from the intervals. Collate pads features
+ labels to the max T in the batch.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import Dataset

from ..common.feature_io import STREAMS, load_video_features
from ..common.live_data import LiveStreamConfig, LiveStreamLoader, live_collate as _live_collate
from ..common.manifests import FingerspellingManifest, load_fs_jsonl

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent.parent))
from data.crop_io import VideoCropIndex                                  # noqa: E402


@dataclass
class FsDatasetConfig:
    feature_root: str
    manifest_jsonl: str
    split: str = "train"
    streams: List[str] = None
    load_per_layer: bool = True
    max_frames: int = 1024


class FsDataset(Dataset):
    def __init__(self, cfg: FsDatasetConfig):
        self.cfg = cfg
        if cfg.streams is None:
            cfg.streams = list(STREAMS)
        self.entries: List[FingerspellingManifest] = load_fs_jsonl(cfg.manifest_jsonl, splits=[cfg.split])

    def __len__(self) -> int:
        return len(self.entries)

    def _build_labels(self, num_frames: int, intervals) -> np.ndarray:
        y = np.zeros(num_frames, dtype=np.int64)
        for s, e in intervals:
            s = max(0, int(s)); e = min(num_frames - 1, int(e))
            y[s : e + 1] = 1
        return y

    def __getitem__(self, i: int):
        m = self.entries[i]
        feats = load_video_features(
            self.cfg.feature_root, m.video_id, streams=self.cfg.streams,
            load_per_layer=self.cfg.load_per_layer,
        )
        per_stream = {}
        for s in self.cfg.streams:
            x = feats.per_layer[s] if feats.per_layer is not None else feats.patches[s]
            x = x[..., : self.cfg.max_frames, :] if x.dim() == 2 else x[:, : self.cfg.max_frames, :]
            per_stream[s] = x
        valid = feats.valid[self.cfg.streams[0]][: self.cfg.max_frames]
        labels = self._build_labels(m.num_frames, m.intervals)
        labels = torch.from_numpy(labels[: self.cfg.max_frames])
        # gold intervals (clipped to max_frames) for IoU eval
        gold = [(s, min(e, self.cfg.max_frames - 1)) for s, e in m.intervals if s < self.cfg.max_frames]
        return {
            "video_id": m.video_id, "per_stream": per_stream, "valid": valid,
            "frame_labels": labels, "gold_intervals": gold,
        }


@dataclass
class LiveFsDatasetConfig:
    crop_root: str
    manifest_jsonl: str
    split: str = "train"
    streams: List[str] = None
    input_size: int = 224
    max_frames: int = 1024
    frame_stride: int = 1


class LiveFsDataset(Dataset):
    def __init__(self, cfg: LiveFsDatasetConfig):
        self.cfg = cfg
        if cfg.streams is None:
            cfg.streams = list(STREAMS)
        self.entries: List[FingerspellingManifest] = load_fs_jsonl(cfg.manifest_jsonl, splits=[cfg.split])
        self.index = VideoCropIndex.from_root(cfg.crop_root, splits=[cfg.split])
        self._by_id = {e.video_id: e for e in self.index.entries}
        self.loader = LiveStreamLoader(LiveStreamConfig(
            crop_root=cfg.crop_root, streams=cfg.streams,
            input_size=cfg.input_size, max_frames=cfg.max_frames,
            frame_stride=cfg.frame_stride, split=cfg.split,
        ))

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, i: int):
        m = self.entries[i]
        entry = self._by_id[m.video_id]
        live = self.loader.load(entry)
        T = live["attention_mask"].numel()
        y = np.zeros(T, dtype=np.int64)
        # `time_indices` give the absolute frame indices we kept; mark labels at those positions
        first_stream = list(live["time_indices"].keys())[0]
        idx = live["time_indices"][first_stream].numpy()
        for s, e in m.intervals:
            mask = (idx >= s) & (idx <= e)
            y[mask] = 1
        live["frame_labels"] = torch.from_numpy(y)
        live["gold_intervals"] = list(m.intervals)
        return live


def live_fs_collate(batch: List[Dict]) -> Dict:
    base = _live_collate([{k: v for k, v in b.items() if k not in ("frame_labels", "gold_intervals")} for b in batch])
    Tmax = base["attention_mask"].shape[1]
    B = len(batch)
    frame_labels = torch.zeros(B, Tmax, dtype=torch.long)
    for i, b in enumerate(batch):
        frame_labels[i, : b["frame_labels"].numel()] = b["frame_labels"]
    base["frame_labels"] = frame_labels
    base["gold_intervals"] = [b["gold_intervals"] for b in batch]
    return base


def fs_collate(batch: List[Dict]) -> Dict:
    streams = list(batch[0]["per_stream"].keys())
    has_layers = batch[0]["per_stream"][streams[0]].dim() == 3
    Tmax = max(b["valid"].numel() for b in batch)
    B = len(batch)
    per_stream_padded: Dict[str, torch.Tensor] = {}
    for s in streams:
        sample = batch[0]["per_stream"][s]
        if has_layers:
            L, _, D = sample.shape
            buf = torch.zeros(L, B, Tmax, D, dtype=sample.dtype)
            for i, b in enumerate(batch):
                t = b["per_stream"][s].shape[1]
                buf[:, i, :t] = b["per_stream"][s]
        else:
            D = sample.shape[-1]
            buf = torch.zeros(B, Tmax, D, dtype=sample.dtype)
            for i, b in enumerate(batch):
                t = b["per_stream"][s].shape[0]
                buf[i, :t] = b["per_stream"][s]
        per_stream_padded[s] = buf
    attn = torch.zeros(B, Tmax, dtype=torch.long)
    frame_labels = torch.zeros(B, Tmax, dtype=torch.long)
    for i, b in enumerate(batch):
        t = b["valid"].numel()
        attn[i, :t] = 1
        frame_labels[i, : b["frame_labels"].numel()] = b["frame_labels"]
    return {
        "video_id": [b["video_id"] for b in batch],
        "per_stream": per_stream_padded,
        "attention_mask": attn,
        "frame_labels": frame_labels,
        "gold_intervals": [b["gold_intervals"] for b in batch],
    }
