"""Read the on-disk crop layout.

A `VideoCropIndex` holds the global list of (video_id, split) pairs read
from `<crop_root>/index.jsonl`. A `VideoCropEntry` is the per-video
manifest + helpers to load specific stream frames as PIL images.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from PIL import Image


STREAMS = ("lh", "rh", "face")


def _parse_mask(s) -> np.ndarray:
    if isinstance(s, str):
        return np.array([int(c) for c in s.split(",") if c != ""], dtype=np.bool_)
    return np.asarray(s, dtype=np.bool_)


@dataclass
class StreamSpec:
    crop_size: List[int]              # [h, w]
    valid_mask: np.ndarray            # (T,) bool


@dataclass
class VideoManifest:
    video_id: str
    source_video: str
    augmentation_tag: str
    num_frames: int
    fps: float
    streams: Dict[str, StreamSpec]    # 'lh', 'rh', 'face'


def load_manifest(manifest_path: Path) -> VideoManifest:
    with open(manifest_path, "r") as f:
        raw = json.load(f)
    streams = {
        s: StreamSpec(
            crop_size=list(v["crop_size"]),
            valid_mask=_parse_mask(v["valid_mask"]),
        )
        for s, v in raw["streams"].items()
    }
    return VideoManifest(
        video_id=raw["video_id"],
        source_video=raw.get("source_video", raw["video_id"]),
        augmentation_tag=raw.get("augmentation_tag", ""),
        num_frames=int(raw["num_frames"]),
        fps=float(raw.get("fps", 24.0)),
        streams=streams,
    )


@dataclass
class VideoCropEntry:
    video_id: str
    split: str
    root: Path                        # <crop_root>/<video_id>
    manifest: VideoManifest = field(repr=False)

    def stream_dir(self, stream: str) -> Path:
        assert stream in STREAMS, f"unknown stream {stream!r}"
        return self.root / stream

    def num_frames(self) -> int:
        return self.manifest.num_frames

    def valid_mask(self, stream: str) -> np.ndarray:
        return self.manifest.streams[stream].valid_mask

    def crop_size(self, stream: str) -> List[int]:
        return self.manifest.streams[stream].crop_size

    def load_frame(self, stream: str, frame_idx: int) -> Image.Image:
        p = self.stream_dir(stream) / f"{frame_idx:06d}.jpg"
        return Image.open(p).convert("RGB")


@dataclass
class VideoCropIndex:
    crop_root: Path
    entries: List[VideoCropEntry]
    by_split: Dict[str, List[VideoCropEntry]] = field(default_factory=dict)

    @classmethod
    def from_root(cls, crop_root: str | Path, splits: Optional[List[str]] = None) -> "VideoCropIndex":
        crop_root = Path(crop_root)
        index_path = crop_root / "index.jsonl"
        if not index_path.exists():
            raise FileNotFoundError(f"missing {index_path}")
        entries: List[VideoCropEntry] = []
        with open(index_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                vid = rec["video_id"]
                split = rec.get("split", "train")
                if splits is not None and split not in splits:
                    continue
                vroot = crop_root / vid
                manifest = load_manifest(vroot / "manifest.json")
                entries.append(VideoCropEntry(video_id=vid, split=split, root=vroot, manifest=manifest))
        by_split: Dict[str, List[VideoCropEntry]] = {}
        for e in entries:
            by_split.setdefault(e.split, []).append(e)
        return cls(crop_root=crop_root, entries=entries, by_split=by_split)
