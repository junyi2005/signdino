"""Dump per-stream teacher features for every video in the index.

Used by the downstream translation / recognition / fingerspelling trainer
(see `downstream/`).

Two output modes:

`--mode final` (default): one file per (video, stream) with the last-
layer outputs only:
    <out-dir>/<video_id>/<stream>.pt
    {'cls': (D,) float16, 'patches': (T, D) float16, 'valid': (T,) bool}

`--mode all_layers`: also dump the per-Transformer-layer per-frame tokens
so the downstream can do the layer-weighted-sum trick (SHuBERT Tab. 6):
    {'cls': (D,), 'patches': (T, D), 'patches_per_layer': (L+1, T, D), 'valid': (T,)}
where L is the temporal-encoder depth (6 by default) and the +1 includes
the post-norm output.

Run:
    python extract_features.py \
        --ckpt runs/refine_lh/ckpt_e30.pt \
        --crop-root /path/to/CROPS --stream lh \
        --out-dir runs/refine_lh/features
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

from data.crop_io import VideoCropIndex                       # noqa: E402
from data.embedding_cache import EmbeddingCache               # noqa: E402
from model.meta_arch import SignDinoConfig, SignTemporalDinoModel  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, type=Path)
    ap.add_argument("--crop-root", required=True, type=Path)
    ap.add_argument("--stream", required=True, choices=["lh", "rh", "face"])
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--backbone-tag", default="dinov3_vitb16_224")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--splits", nargs="*", default=None)
    ap.add_argument("--mode", choices=["final", "all_layers"], default="final")
    args = ap.parse_args()

    device = torch.device(args.device)
    sd = torch.load(args.ckpt, map_location="cpu", weights_only=True)
    saved_cfg = sd["cfg"]
    m = saved_cfg["model"]
    cfg = SignDinoConfig(
        in_dim=m["in_dim"], embed_dim=m["embed_dim"], depth=m["depth"],
        num_heads=m["num_heads"], mlp_ratio=m["mlp_ratio"], dropout=m["dropout"],
        max_len=m["max_len"],
        head_out_dim=m["head_out_dim"], head_hidden=m["head_hidden"],
        head_bottleneck=m["head_bottleneck"], head_layers=m["head_layers"],
        head_norm_last=m["head_norm_last"],
        student_temp=m["student_temp"], teacher_temp=m["teacher_temp"],
        teacher_temp_start=m["teacher_temp_start"], sinkhorn_iters=m["sinkhorn_iters"],
        dino_weight=m["dino_weight"], ibot_weight=m["ibot_weight"],
        dkoleo_weight=m["dkoleo_weight"], gram_weight=m["gram_weight"],
    )
    model = SignTemporalDinoModel(cfg).to(device)
    model.load_state_dict(sd["model"], strict=False)
    model.eval()

    index = VideoCropIndex.from_root(args.crop_root, splits=args.splits)
    cache = EmbeddingCache(args.crop_root, backbone_tag=args.backbone_tag)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    max_len = cfg.max_len
    for entry in tqdm(index.entries, desc=f"extract({args.stream})"):
        if not cache.exists(entry.video_id, args.stream):
            continue
        embeds, valid = cache.load(entry.video_id, args.stream)
        T = embeds.shape[0]
        if T == 0:
            continue
        chunks = []
        cls_chunks = []
        per_layer_chunks = []   # only if mode == all_layers
        for s in range(0, T, max_len):
            e = min(s + max_len, T)
            chunk = embeds[s:e].unsqueeze(0).to(device)
            v = valid[s:e].unsqueeze(0).to(device)
            t_idx = torch.arange(s, e, device=device).unsqueeze(0)
            with torch.no_grad():
                if args.mode == "all_layers":
                    cls, patches, all_layers = model.teacher_encoder(
                        chunk, v, None, t_idx, return_all_layers=True
                    )
                    # all_layers: (L+1, B, T+1, D); take frame tokens [:, :, 1:, :] -> (L+1, T, D)
                    per_layer_chunks.append(all_layers[:, 0, 1:].cpu())
                else:
                    cls, patches = model.teacher_encoder(chunk, v, None, t_idx)
            cls_chunks.append(cls.cpu())
            chunks.append(patches.squeeze(0).cpu())
        full_patches = torch.cat(chunks, dim=0)
        full_cls = torch.stack(cls_chunks).mean(dim=0).squeeze(0)
        out = {"cls": full_cls.half(), "patches": full_patches.half(), "valid": valid}
        if args.mode == "all_layers":
            out["patches_per_layer"] = torch.cat(per_layer_chunks, dim=1).half()  # (L+1, T, D)
        out_path = args.out_dir / entry.video_id / f"{args.stream}.pt"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(out, out_path)
    print("[extract] done")


if __name__ == "__main__":
    main()
