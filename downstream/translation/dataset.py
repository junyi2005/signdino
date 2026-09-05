"""Translation dataset + collate.

A dataset item is one (video_id, text) pair; the dataset returns the
per-stream features (loaded from `extract_features.py` output) and the
ByT5-tokenised target text. The collate pads per-stream features to the
max T in the batch and produces an attention_mask.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import torch
from torch.utils.data import Dataset

from ..common.feature_io import STREAMS, load_video_features
from ..common.live_data import LiveStreamConfig, LiveStreamLoader, live_collate as _live_collate
from ..common.manifests import TranslationManifest, load_translation_jsonl

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent.parent))
from data.crop_io import VideoCropIndex                                  # noqa: E402


@dataclass
class TranslationDatasetConfig:
    feature_root: str
    manifest_jsonl: str
    split: str = "train"
    streams: List[str] = None
    load_per_layer: bool = True
    max_frames: int = 1024
    tokenizer_id: str = "google/byt5-base"
    max_text_length: int = 384


class TranslationDataset(Dataset):
    def __init__(self, cfg: TranslationDatasetConfig):
        from transformers import AutoTokenizer
        self.cfg = cfg
        if cfg.streams is None:
            cfg.streams = list(STREAMS)
        self.entries: List[TranslationManifest] = load_translation_jsonl(
            cfg.manifest_jsonl, splits=[cfg.split]
        )
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer_id)

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, i: int):
        m = self.entries[i]
        feats = load_video_features(
            self.cfg.feature_root, m.video_id, streams=self.cfg.streams,
            load_per_layer=self.cfg.load_per_layer,
        )
        # take per-stream per-layer tokens if available, else single-layer patches
        per_stream = {}
        for s in self.cfg.streams:
            if feats.per_layer is not None:
                x = feats.per_layer[s]              # (L+1, T, D)
            else:
                x = feats.patches[s]                # (T, D)
            x = x[..., : self.cfg.max_frames, :] if x.dim() == 2 else x[:, : self.cfg.max_frames, :]
            per_stream[s] = x
        valid_any = feats.valid[self.cfg.streams[0]][: self.cfg.max_frames]
        # tokenise text
        tok = self.tokenizer(
            m.text, max_length=self.cfg.max_text_length, truncation=True,
            return_tensors="pt", add_special_tokens=True,
        )
        return {
            "video_id": m.video_id,
            "per_stream": per_stream,
            "valid": valid_any,
            "labels": tok["input_ids"].squeeze(0),
        }


@dataclass
class LiveTranslationDatasetConfig:
    crop_root: str
    manifest_jsonl: str
    split: str = "train"
    streams: List[str] = None
    input_size: int = 224
    max_frames: int = 512
    frame_stride: int = 2                # subsample for speed (paper does this)
    tokenizer_id: str = "google/byt5-base"
    max_text_length: int = 384


class LiveTranslationDataset(Dataset):
    """Live mode: returns raw crops + tokenised text. The upstream + downstream
    fine-tune jointly; cf. SHuBERT Table 7."""

    def __init__(self, cfg: LiveTranslationDatasetConfig):
        from transformers import AutoTokenizer
        self.cfg = cfg
        if cfg.streams is None:
            cfg.streams = list(STREAMS)
        self.entries: List[TranslationManifest] = load_translation_jsonl(
            cfg.manifest_jsonl, splits=[cfg.split]
        )
        self.index = VideoCropIndex.from_root(cfg.crop_root, splits=[cfg.split])
        self._by_id = {e.video_id: e for e in self.index.entries}
        self.loader = LiveStreamLoader(LiveStreamConfig(
            crop_root=cfg.crop_root, streams=cfg.streams,
            input_size=cfg.input_size, max_frames=cfg.max_frames,
            frame_stride=cfg.frame_stride, split=cfg.split,
        ))
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer_id)

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, i: int):
        m = self.entries[i]
        entry = self._by_id[m.video_id]
        live = self.loader.load(entry)
        tok = self.tokenizer(
            m.text, max_length=self.cfg.max_text_length, truncation=True,
            return_tensors="pt", add_special_tokens=True,
        )
        live["labels"] = tok["input_ids"].squeeze(0)
        return live


def live_translation_collate(batch: List[Dict]) -> Dict:
    base = _live_collate([{k: v for k, v in b.items() if k != "labels"} for b in batch])
    # pad labels
    Lmax = max(b["labels"].numel() for b in batch)
    labels = torch.full((len(batch), Lmax), -100, dtype=torch.long)
    for i, b in enumerate(batch):
        labels[i, : b["labels"].numel()] = b["labels"]
    base["labels"] = labels
    return base


def translation_collate(batch: List[Dict]) -> Dict:
    """Pad per-stream features to max T in batch, tokens to max length."""
    streams = list(batch[0]["per_stream"].keys())
    has_layers = batch[0]["per_stream"][streams[0]].dim() == 3
    # max T per batch
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
    # labels: pad with -100
    Lmax = max(b["labels"].numel() for b in batch)
    labels = torch.full((B, Lmax), -100, dtype=torch.long)
    for i, b in enumerate(batch):
        labels[i, : b["labels"].numel()] = b["labels"]
    return {
        "video_id": [b["video_id"] for b in batch],
        "per_stream": per_stream_padded,
        "attention_mask": attn,
        "labels": labels,
    }
