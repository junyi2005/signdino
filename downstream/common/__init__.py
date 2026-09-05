"""Shared downstream pieces: feature I/O, multi-stream fusion (incl. layer-
weighted sum), manifest readers, metric helpers."""

from .feature_io import PerStreamFeatures, load_video_features
from .fusion import StreamFusion, LayerWeightedSum, FusionConfig
from .live_data import LiveStreamDataset, LiveStreamLoader, LiveStreamConfig, live_collate
from .manifests import (
    TranslationManifest, ISLRManifest, FingerspellingManifest,
    load_translation_jsonl, load_islr_jsonl, load_fs_jsonl,
)
from .metrics import compute_bleu, compute_chrf, compute_recall_at_k, interval_iou

__all__ = [
    "PerStreamFeatures", "load_video_features",
    "StreamFusion", "LayerWeightedSum", "FusionConfig",
    "LiveStreamDataset", "LiveStreamLoader", "LiveStreamConfig", "live_collate",
    "TranslationManifest", "ISLRManifest", "FingerspellingManifest",
    "load_translation_jsonl", "load_islr_jsonl", "load_fs_jsonl",
    "compute_bleu", "compute_chrf", "compute_recall_at_k", "interval_iou",
]
