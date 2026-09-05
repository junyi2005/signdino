"""StreamClipDataset -- one sample per video, all crops for one stream.

A sample is a dict:
    'video_id':       str
    'global_frames':  list[Ng] of torch.float32 (Tg, D)    -- per-frame embeddings
    'global_valid':   list[Ng] of torch.bool    (Tg,)
    'global_time':    list[Ng] of torch.int64   (Tg,)      -- absolute frame indices
    'local_frames':   list[Nl] of torch.float32 (Tl, D)
    'local_valid':    list[Nl] of torch.bool    (Tl,)
    'local_time':     list[Nl] of torch.int64   (Tl,)

If `embedding_source == 'cache'`, D = 768 and embeddings are loaded from
the per-video cache file. If 'live', D = 768 again but we forward the
cropped frames through a frozen DINOv3 inside the trainer's collate (the
collate hook in the trainer handles this, since DINOv3 must run on GPU).
For 'live' mode, frames are returned as preprocessed image tensors (3, H, W)
in place of embeddings and the trainer batches them.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from .augment import build_image_aug, build_image_eval
from .crop_io import VideoCropEntry, VideoCropIndex
from .embedding_cache import EmbeddingCache
from .temporal_crop import MultiCropConfig, TemporalMultiCropSampler


@dataclass
class StreamDatasetConfig:
    crop_root: str
    stream: str                          # 'lh' | 'rh' | 'face'
    split: str = "train"
    embedding_source: str = "cache"      # 'cache' | 'live'
    backbone_tag: str = "dinov3_vitb16_224"
    input_size: int = 224
    multi_crop: MultiCropConfig = None
    augment: bool = True


class StreamClipDataset(Dataset):
    def __init__(self, cfg: StreamDatasetConfig, index: Optional[VideoCropIndex] = None):
        self.cfg = cfg
        self.index = index or VideoCropIndex.from_root(cfg.crop_root, splits=[cfg.split])
        self.entries: List[VideoCropEntry] = self.index.by_split.get(cfg.split, [])
        if cfg.multi_crop is None:
            cfg.multi_crop = MultiCropConfig()
        self.sampler = TemporalMultiCropSampler(cfg.multi_crop)
        if cfg.embedding_source == "cache":
            self.cache = EmbeddingCache(cfg.crop_root, backbone_tag=cfg.backbone_tag)
            self.transform = None
        else:
            self.cache = None
            self.transform = build_image_aug(cfg.input_size) if cfg.augment else build_image_eval(cfg.input_size)

    def __len__(self) -> int:
        return len(self.entries)

    def _load_full_embeddings(self, entry: VideoCropEntry) -> torch.Tensor:
        embeds, _ = self.cache.load(entry.video_id, self.cfg.stream)
        return embeds                                # (T, 768)

    def _load_frame_images(self, entry: VideoCropEntry, indices: np.ndarray) -> torch.Tensor:
        imgs = []
        for i in indices:
            img = entry.load_frame(self.cfg.stream, int(i))
            imgs.append(self.transform(img))
        return torch.stack(imgs, dim=0)              # (T, 3, H, W)

    def __getitem__(self, i: int):
        entry = self.entries[i]
        valid = entry.valid_mask(self.cfg.stream)
        T = len(valid)
        globals_, locals_ = self.sampler.sample(T, valid)

        if self.cfg.embedding_source == "cache":
            full = self._load_full_embeddings(entry)               # (T, 768)
            def take(idx):  # noqa: E306
                return (
                    full[idx],
                    torch.from_numpy(valid[idx].copy()),
                    torch.from_numpy(idx.astype(np.int64)),
                )
        else:
            def take(idx):  # noqa: E306
                return (
                    self._load_frame_images(entry, idx),           # (Twin, 3, H, W) -- not embeds
                    torch.from_numpy(valid[idx].copy()),
                    torch.from_numpy(idx.astype(np.int64)),
                )

        g_pack = [take(idx) for idx in globals_]
        l_pack = [take(idx) for idx in locals_]
        return {
            "video_id": entry.video_id,
            "global_frames": [g[0] for g in g_pack],
            "global_valid": [g[1] for g in g_pack],
            "global_time": [g[2] for g in g_pack],
            "local_frames": [l[0] for l in l_pack],
            "local_valid": [l[1] for l in l_pack],
            "local_time": [l[2] for l in l_pack],
        }
