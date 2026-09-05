"""ISLR test-time evaluation.

Loads `ckpt_best.pt`, runs the test split, prints R@1/5/10 (and per-
class accuracy if requested).
"""
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
from downstream.common.metrics import compute_recall_at_k                        # noqa: E402
from downstream.islr.dataset import ISLRDataset, ISLRDatasetConfig, islr_collate # noqa: E402
from downstream.islr.model import SignDinoISLR, ISLRConfig                       # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--feature-root", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--batch-size", type=int, default=128)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sd = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = sd["cfg"]
    fusion_cfg = FusionConfig(**cfg["fusion"])
    vocab = sd["vocab"]
    model_cfg = ISLRConfig(num_classes=len(vocab), fusion=fusion_cfg,
                           label_smoothing=cfg["model"]["label_smoothing"])
    model = SignDinoISLR(model_cfg).to(device)
    model.load_state_dict(sd["model"], strict=False)
    model.eval()

    ds_cfg = ISLRDatasetConfig(
        feature_root=args.feature_root, manifest_jsonl=args.manifest,
        split=args.split, load_per_layer=cfg["fusion"]["use_layer_weighted_sum"],
        max_frames=cfg["data"]["max_frames"], label_vocab=vocab,
    )
    ds = ISLRDataset(ds_cfg)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=4, collate_fn=islr_collate, pin_memory=True)

    scores, labels = [], []
    with torch.no_grad():
        for batch in loader:
            per = {s: t.to(device) for s, t in batch["per_stream"].items()}
            attn = batch["attention_mask"].to(device)
            logits = model(per, attn)
            scores.append(logits.cpu().numpy())
            labels.append(batch["labels"].numpy())
    scores = np.concatenate(scores, axis=0); labels = np.concatenate(labels, axis=0)
    print(json.dumps(compute_recall_at_k(scores, labels, ks=(1, 5, 10)), indent=2))


if __name__ == "__main__":
    main()
