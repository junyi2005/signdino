"""ISLR dataset + collate.

A sample is (video_id, class_label). Reads per-stream features from the
extract_features.py output directory; collate pads per-stream tensors to
max T in batch and produces attention_mask.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import torch
from torch.utils.data import Dataset

from ..common.feature_io import STREAMS, load_video_features
from ..common.live_data import LiveStreamConfig, LiveStreamLoader, live_collate as _live_collate
from ..common.manifests import ISLRManifest, build_label_vocab, load_islr_jsonl

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent.parent))
from data.crop_io import VideoCropIndex                                  # noqa: E402


@dataclass
class ISLRDatasetConfig:
    feature_root: str
    manifest_jsonl: str
    split: str = "train"
    streams: List[str] = None
    load_per_layer: bool = True
    max_frames: int = 256
    # if provided, must be the same vocab across train/val/test; otherwise built from manifest
    label_vocab: Dict[str, int] = None


class ISLRDataset(Dataset):
    def __init__(self, cfg: ISLRDatasetConfig):
        self.cfg = cfg
        if cfg.streams is None:
            cfg.streams = list(STREAMS)
        self.entries: List[ISLRManifest] = load_islr_jsonl(cfg.manifest_jsonl, splits=[cfg.split])
        if cfg.label_vocab is None:
            # build from this split only -- usually you should pass a shared vocab
            self.vocab: Dict[str, int] = build_label_vocab(self.entries)
        else:
            self.vocab = cfg.label_vocab

    def __len__(self) -> int:
        return len(self.entries)

    def num_classes(self) -> int:
        return len(self.vocab)

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
        return {
            "video_id": m.video_id,
            "per_stream": per_stream,
            "valid": valid,
            "label": torch.tensor(self.vocab[m.label], dtype=torch.long),
        }


@dataclass
class LiveISLRDatasetConfig:
    crop_root: str
    manifest_jsonl: str
    split: str = "train"
    streams: List[str] = None
    input_size: int = 224
    max_frames: int = 128
    frame_stride: int = 1
    label_vocab: Dict[str, int] = None


class LiveISLRDataset(Dataset):
    """Live mode for ISLR: returns raw crops + class label."""

    def __init__(self, cfg: LiveISLRDatasetConfig):
        self.cfg = cfg
        if cfg.streams is None:
            cfg.streams = list(STREAMS)
        self.entries: List[ISLRManifest] = load_islr_jsonl(cfg.manifest_jsonl, splits=[cfg.split])
        if cfg.label_vocab is None:
            self.vocab: Dict[str, int] = build_label_vocab(self.entries)
        else:
            self.vocab = cfg.label_vocab
        self.index = VideoCropIndex.from_root(cfg.crop_root, splits=[cfg.split])
        self._by_id = {e.video_id: e for e in self.index.entries}
        self.loader = LiveStreamLoader(LiveStreamConfig(
            crop_root=cfg.crop_root, streams=cfg.streams,
            input_size=cfg.input_size, max_frames=cfg.max_frames,
            frame_stride=cfg.frame_stride, split=cfg.split,
        ))

    def __len__(self) -> int:
        return len(self.entries)

    def num_classes(self) -> int:
        return len(self.vocab)

    def __getitem__(self, i: int):
        m = self.entries[i]
        entry = self._by_id[m.video_id]
        live = self.loader.load(entry)
        live["label"] = torch.tensor(self.vocab[m.label], dtype=torch.long)
        return live


def live_islr_collate(batch: List[Dict]) -> Dict:
    base = _live_collate([{k: v for k, v in b.items() if k != "label"} for b in batch])
    base["labels"] = torch.stack([b["label"] for b in batch], dim=0)
    return base


def islr_collate(batch: List[Dict]) -> Dict:
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
    for i, b in enumerate(batch):
        attn[i, : b["valid"].numel()] = 1
    labels = torch.stack([b["label"] for b in batch], dim=0)
    return {
        "video_id": [b["video_id"] for b in batch],
        "per_stream": per_stream_padded,
        "attention_mask": attn,
        "labels": labels,
    }
