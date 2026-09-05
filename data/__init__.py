"""Data pipeline for SignDINO-v2.

On-disk layout (the contract documented in DESIGN.md § 4.2):
    <crop_root>/<video_id>/{lh,rh,face}/<frame:06d>.jpg
    <crop_root>/<video_id>/manifest.json
    <crop_root>/index.jsonl

The dataset returns multi-temporal-crop samples for one stream at a time;
the trainer instantiates one StreamClipDataset per stream.
"""

from .crop_io import VideoCropIndex, VideoCropEntry, load_manifest
from .embedding_cache import EmbeddingCache
from .temporal_crop import TemporalMultiCropSampler, MultiCropConfig
from .augment import build_image_aug, IMAGENET_MEAN, IMAGENET_STD
from .dataset import StreamClipDataset
from .collate import multi_crop_collate

__all__ = [
    "VideoCropIndex",
    "VideoCropEntry",
    "load_manifest",
    "EmbeddingCache",
    "TemporalMultiCropSampler",
    "MultiCropConfig",
    "build_image_aug",
    "IMAGENET_MEAN",
    "IMAGENET_STD",
    "StreamClipDataset",
    "multi_crop_collate",
]
