"""One-off pass: build the frozen DINOv3 per-frame embedding cache.

For every video in `<crop_root>/index.jsonl`, for every stream:
1. Load all frames from disk as preprocessed image tensors.
2. Forward through the frozen DINOv3 ViT-B/16 in batches.
3. Save (T, 768) float16 embeddings + (T,) bool valid mask to
   `<crop_root>/.embedding_cache/<backbone_tag>/<video_id>/<stream>.pt`.

Run:
    python preprocess/cache_embeddings.py \
        --crop-root /path/to/CROPS \
        --dinov3-repo /path/to/dinov3 \
        --model-name dinov3_vitb16 \
        --batch-size 256 \
        --num-workers 4
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
from data.augment import build_image_eval                # noqa: E402
from data.crop_io import STREAMS, VideoCropIndex          # noqa: E402
from data.embedding_cache import EmbeddingCache          # noqa: E402
from model.backbone import FrozenDinov3FrameEmbedder    # noqa: E402


class _VideoStreamFrames(Dataset):
    """Yields one (frame_idx, image_tensor) per __getitem__ for one stream of one video."""

    def __init__(self, entry, stream: str, transform):
        self.entry = entry
        self.stream = stream
        self.T = entry.num_frames()
        self.transform = transform
        self.valid = entry.valid_mask(stream)

    def __len__(self) -> int:
        return self.T

    def __getitem__(self, i: int):
        if self.valid[i]:
            img = self.entry.load_frame(self.stream, i)
        else:
            img = Image.new("RGB", tuple(self.entry.crop_size(self.stream)), (0, 0, 0))
        return i, self.transform(img)


def cache_one_stream(entry, stream, embedder, cache, batch_size, num_workers, device):
    if cache.exists(entry.video_id, stream):
        return False
    T = entry.num_frames()
    transform = build_image_eval(embedder.input_size)
    ds = _VideoStreamFrames(entry, stream, transform)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                        pin_memory=True, drop_last=False)
    embeds = torch.zeros(T, embedder.embed_dim, dtype=torch.float32)
    for idxs, imgs in loader:
        imgs = imgs.to(device, non_blocking=True)
        with torch.no_grad():
            cls = embedder(imgs)                          # (B, D)
        embeds[idxs] = cls.float().cpu()
    cache.save(entry.video_id, stream, embeds, entry.valid_mask(stream))
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--crop-root", required=True, type=Path)
    ap.add_argument("--dinov3-repo", default="/path/to/dinov3")
    ap.add_argument("--model-name", default="dinov3_vitb16")
    ap.add_argument("--backbone-tag", default="dinov3_vitb16_224")
    ap.add_argument("--input-size", type=int, default=224)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--splits", nargs="*", default=None,
                    help="If given, only cache these splits (e.g. train val).")
    args = ap.parse_args()

    device = torch.device(args.device)
    embedder = FrozenDinov3FrameEmbedder(
        repo_path=args.dinov3_repo,
        model_name=args.model_name,
        input_size=args.input_size,
    ).to(device)
    embedder.eval()

    index = VideoCropIndex.from_root(args.crop_root, splits=args.splits)
    cache = EmbeddingCache(args.crop_root, backbone_tag=args.backbone_tag)
    print(f"[cache] {len(index.entries)} videos, streams = {STREAMS}")

    t0 = time.time()
    n_done = 0
    n_skip = 0
    pbar = tqdm(index.entries, desc="videos")
    for entry in pbar:
        for stream in STREAMS:
            wrote = cache_one_stream(entry, stream, embedder, cache,
                                     args.batch_size, args.num_workers, device)
            n_done += int(wrote)
            n_skip += int(not wrote)
        pbar.set_postfix(done=n_done, skip=n_skip)
    print(f"[cache] wrote {n_done} stream caches, skipped {n_skip} existing, elapsed {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
