"""Fingerspelling test-time evaluation. Writes per-video predicted
intervals and prints mean IoU."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from downstream.common.fusion import FusionConfig                                # noqa: E402
from downstream.common.metrics import interval_iou                                # noqa: E402
from downstream.fingerspelling.dataset import FsDataset, FsDatasetConfig, fs_collate  # noqa: E402
from downstream.fingerspelling.model import SignDinoFsDetector, FsConfig          # noqa: E402
from downstream.fingerspelling.trainer import probs_to_intervals                  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--feature-root", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--out-jsonl", required=True)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--min-len", type=int, default=3)
    ap.add_argument("--smooth-kernel", type=int, default=3)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sd = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = sd["cfg"]
    fusion_cfg = FusionConfig(**cfg["fusion"])
    model_cfg = FsConfig(
        fusion=fusion_cfg, hidden_depth=cfg["model"]["hidden_depth"],
        hidden_heads=cfg["model"]["hidden_heads"], hidden_dropout=cfg["model"]["hidden_dropout"],
        pos_weight=cfg["model"]["pos_weight"],
    )
    model = SignDinoFsDetector(model_cfg).to(device)
    model.load_state_dict(sd["model"], strict=False)
    model.eval()

    ds = FsDataset(FsDatasetConfig(
        feature_root=args.feature_root, manifest_jsonl=args.manifest,
        split=args.split, load_per_layer=cfg["fusion"]["use_layer_weighted_sum"],
        max_frames=cfg["data"]["max_frames"],
    ))
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=4, collate_fn=fs_collate, pin_memory=True)

    ious = []
    with open(args.out_jsonl, "w") as fout, torch.no_grad():
        for batch in loader:
            per = {s: t.to(device) for s, t in batch["per_stream"].items()}
            attn = batch["attention_mask"].to(device)
            logits = model(per, attn)
            probs = torch.sigmoid(logits).cpu().numpy()
            for i, gold in enumerate(batch["gold_intervals"]):
                t = int(attn[i].sum())
                pred = probs_to_intervals(probs[i, :t], threshold=args.threshold,
                                          min_len=args.min_len, smooth_kernel=args.smooth_kernel)
                ious.append(interval_iou(pred, gold))
                fout.write(json.dumps({"video_id": batch["video_id"][i],
                                       "pred_intervals": pred, "gold_intervals": gold}) + "\n")
    print(json.dumps({"mean_iou": float(np.mean(ious)) if ious else 0.0}, indent=2))


if __name__ == "__main__":
    main()
