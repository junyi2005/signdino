"""Fingerspelling detection (ASL-Stem Wiki, IoU eval).

Per-frame binary classifier predicting whether each frame is part of a
fingerspelling interval. Inference applies median smoothing + contiguous
interval extraction, then computes mean IoU vs gold intervals. Mirrors
SHuBERT § 4.4 setup.
"""

from .model import SignDinoFsDetector, FsConfig, LiveSignDinoFsDetector
from .dataset import (
    FsDataset, fs_collate,
    LiveFsDataset, LiveFsDatasetConfig, live_fs_collate,
)
from .trainer import FsTrainer

__all__ = [
    "SignDinoFsDetector", "FsConfig", "LiveSignDinoFsDetector",
    "FsDataset", "fs_collate",
    "LiveFsDataset", "LiveFsDatasetConfig", "live_fs_collate",
    "FsTrainer",
]
