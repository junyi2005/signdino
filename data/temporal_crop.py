"""Multi-temporal-crop sampler.

Given a per-video sequence of length T and a validity mask, sample
`n_global` long windows + `n_local` short windows. Each window is a
contiguous block of frames: a random length is drawn per window from
[len_min, len_max] and a random start in [0, T - length]. Global and
local windows are sampled **independently** over the whole video,
mirroring DINOv2/DINOv3 spatial multi-crop, where each crop (global or
local) is an independent `RandomResizedCrop` of the source image. The
only distinction between globals and locals here is the length range.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np


@dataclass
class MultiCropConfig:
    n_global: int = 2
    n_local: int = 8
    # Window length is drawn uniformly from [min, max] per window.
    global_len_min: int = 64
    global_len_max: int = 96
    local_len_min: int = 10
    local_len_max: int = 32
    min_valid_frames: int = 4
    max_attempts: int = 8


class TemporalMultiCropSampler:
    def __init__(self, cfg: MultiCropConfig, rng: Optional[np.random.Generator] = None):
        self.cfg = cfg
        self.rng = rng or np.random.default_rng()

    def _sample_window(self, T: int, valid: np.ndarray, len_min: int, len_max: int) -> np.ndarray:
        """Sample one window of contiguous frames. Returns absolute indices [length]."""
        for _ in range(self.cfg.max_attempts):
            length = int(self.rng.integers(len_min, len_max + 1))
            if T < length:
                # video shorter than the drawn length: take all frames and
                # right-pad by repeating the last index.
                idx = np.arange(T)
                idx = np.concatenate([idx, np.full(length - T, idx[-1])])
                return idx
            start = int(self.rng.integers(0, T - length + 1))
            idx = np.arange(start, start + length)
            if int(valid[idx].sum()) >= self.cfg.min_valid_frames:
                return idx
        # give up; return the last attempt
        return idx

    def sample(self, T: int, valid: np.ndarray) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """Returns (global_windows, local_windows) -- each a list of np arrays of
        absolute frame indices into the source video sequence. Per-window length
        and start are sampled independently."""
        globals_ = [
            self._sample_window(T, valid, self.cfg.global_len_min, self.cfg.global_len_max)
            for _ in range(self.cfg.n_global)
        ]
        locals_ = [
            self._sample_window(T, valid, self.cfg.local_len_min, self.cfg.local_len_max)
            for _ in range(self.cfg.n_local)
        ]
        return globals_, locals_
