"""Downstream-task manifests.

A manifest is a JSONL file with one record per (video_id, split). Each task
defines its own additional fields:

- TranslationManifest: {"video_id", "text", "split"}
- ISLRManifest:        {"video_id", "label", "split"}      # label is gloss / class string
- FingerspellingManifest: {"video_id", "intervals": [[s,e],...], "num_frames", "split"}

Manifests are split-aware: a single jsonl can hold train/val/test rows, and
the loaders return only the rows matching a requested split.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# --------- Translation ---------------------------------------------------

@dataclass
class TranslationManifest:
    video_id: str
    text: str
    split: str


def load_translation_jsonl(path: str | Path, splits: Optional[List[str]] = None) -> List[TranslationManifest]:
    out: List[TranslationManifest] = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if splits is not None and rec["split"] not in splits:
                continue
            out.append(TranslationManifest(
                video_id=rec["video_id"], text=rec["text"], split=rec["split"]
            ))
    return out


# --------- ISLR ----------------------------------------------------------

@dataclass
class ISLRManifest:
    video_id: str
    label: str
    split: str


def load_islr_jsonl(path: str | Path, splits: Optional[List[str]] = None) -> List[ISLRManifest]:
    out: List[ISLRManifest] = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if splits is not None and rec["split"] not in splits:
                continue
            out.append(ISLRManifest(video_id=rec["video_id"], label=rec["label"], split=rec["split"]))
    return out


def build_label_vocab(manifests: List[ISLRManifest]) -> Dict[str, int]:
    labels = sorted({m.label for m in manifests})
    return {lab: i for i, lab in enumerate(labels)}


# --------- Fingerspelling -----------------------------------------------

@dataclass
class FingerspellingManifest:
    video_id: str
    intervals: List[Tuple[int, int]]      # list of (start_frame, end_frame), inclusive
    num_frames: int
    split: str


def load_fs_jsonl(path: str | Path, splits: Optional[List[str]] = None) -> List[FingerspellingManifest]:
    out: List[FingerspellingManifest] = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if splits is not None and rec["split"] not in splits:
                continue
            out.append(FingerspellingManifest(
                video_id=rec["video_id"],
                intervals=[tuple(iv) for iv in rec["intervals"]],
                num_frames=int(rec["num_frames"]),
                split=rec["split"],
            ))
    return out
