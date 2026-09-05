"""Fingerspelling-detection trainer.

Loss: per-frame BCE. Metric: mean interval IoU vs gold (greedy match).
Hyperparameters mirror SHuBERT § 4.4 setup (close to the ISLR recipe).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from downstream.common.fusion import FusionConfig                                  # noqa: E402
from downstream.common.metrics import interval_iou                                  # noqa: E402
from downstream.fingerspelling.dataset import (                                      # noqa: E402
    FsDataset, FsDatasetConfig, fs_collate,
    LiveFsDataset, LiveFsDatasetConfig, live_fs_collate,
)
from downstream.fingerspelling.model import (                                        # noqa: E402
    SignDinoFsDetector, FsConfig, LiveSignDinoFsDetector,
)
from model.live_upstream import LiveUpstream, LiveUpstreamConfig                    # noqa: E402


def probs_to_intervals(probs: np.ndarray, threshold: float, min_len: int, smooth_kernel: int = 3):
    """Threshold + median smoothing + contiguous-interval extraction.
    Returns list of (start, end_inclusive)."""
    if smooth_kernel and smooth_kernel > 1:
        pad = smooth_kernel // 2
        padded = np.pad(probs, pad, mode="edge")
        smoothed = np.array([np.median(padded[i : i + smooth_kernel]) for i in range(len(probs))])
        probs = smoothed
    binary = (probs >= threshold).astype(np.int32)
    intervals = []
    i = 0
    n = len(binary)
    while i < n:
        if binary[i] == 1:
            j = i
            while j < n and binary[j] == 1:
                j += 1
            if j - i >= min_len:
                intervals.append((i, j - 1))
            i = j
        else:
            i += 1
    return intervals


class FsTrainer:
    def __init__(self, cfg: dict, output_dir: Path):
        self.cfg = cfg
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.mode = cfg.get("mode", "features")
        if self.mode == "live":
            self.train_ds = LiveFsDataset(LiveFsDatasetConfig(
                crop_root=cfg["data"]["crop_root"],
                manifest_jsonl=cfg["data"]["manifest_jsonl"], split="train",
                streams=cfg["fusion"]["streams"],
                input_size=cfg["live"].get("dinov3_input_size", 224),
                max_frames=cfg["data"]["max_frames"],
                frame_stride=cfg["data"].get("frame_stride", 1),
            ))
            self.val_ds = LiveFsDataset(LiveFsDatasetConfig(
                crop_root=cfg["data"]["crop_root"],
                manifest_jsonl=cfg["data"]["manifest_jsonl"], split="val",
                streams=cfg["fusion"]["streams"],
                input_size=cfg["live"].get("dinov3_input_size", 224),
                max_frames=cfg["data"]["max_frames"],
                frame_stride=cfg["data"].get("frame_stride", 1),
            ))
            self.collate = live_fs_collate
        else:
            self.train_ds = FsDataset(FsDatasetConfig(
                feature_root=cfg["data"]["feature_root"],
                manifest_jsonl=cfg["data"]["manifest_jsonl"], split="train",
                load_per_layer=cfg["fusion"]["use_layer_weighted_sum"],
                max_frames=cfg["data"]["max_frames"],
            ))
            self.val_ds = FsDataset(FsDatasetConfig(
                feature_root=cfg["data"]["feature_root"],
                manifest_jsonl=cfg["data"]["manifest_jsonl"], split="val",
                load_per_layer=cfg["fusion"]["use_layer_weighted_sum"],
                max_frames=cfg["data"]["max_frames"],
            ))
            self.collate = fs_collate

        self.train_loader = DataLoader(self.train_ds, batch_size=cfg["train"]["batch_size"],
                                       shuffle=True, num_workers=cfg["data"]["num_workers"],
                                       collate_fn=self.collate, pin_memory=True, drop_last=True)
        self.val_loader = DataLoader(self.val_ds, batch_size=cfg["train"]["batch_size"],
                                     shuffle=False, num_workers=cfg["data"]["num_workers"],
                                     collate_fn=self.collate, pin_memory=True)

        fusion_cfg = FusionConfig(**cfg["fusion"])
        model_cfg = FsConfig(
            fusion=fusion_cfg, hidden_depth=cfg["model"]["hidden_depth"],
            hidden_heads=cfg["model"]["hidden_heads"], hidden_dropout=cfg["model"]["hidden_dropout"],
            pos_weight=cfg["model"]["pos_weight"],
        )
        base_model = SignDinoFsDetector(model_cfg).to(self.device)

        if self.mode == "live":
            up_cfg = LiveUpstreamConfig(
                streams=cfg["fusion"]["streams"], ckpts=cfg["live"]["upstream_ckpts"],
                dinov3_repo=cfg["live"]["dinov3_repo"],
                dinov3_model_name=cfg["live"]["dinov3_model_name"],
                dinov3_weights_path=cfg["live"].get("dinov3_weights_path"),
                dinov3_pretrained=cfg["live"].get("dinov3_pretrained", True),
                dinov3_input_size=cfg["live"].get("dinov3_input_size", 224),
            )
            up = LiveUpstream(up_cfg).to(self.device)
            # full fine-tune: leave encoder params trainable (no LoRA, no freeze)
            self.model = LiveSignDinoFsDetector(base_model, up).to(self.device)
        else:
            self.model = base_model

        # Param groups: head (lr_head, wd), upstream (lr/10) in live mode
        head_params = [p for n, p in self.model.named_parameters()
                       if p.requires_grad and "live_upstream.encoders" not in n]
        groups = [{"params": head_params, "lr": cfg["train"]["lr"],
                   "weight_decay": cfg["train"]["weight_decay"]}]
        if self.mode == "live":
            up_params = [p for n, p in self.model.named_parameters()
                         if p.requires_grad and "live_upstream.encoders" in n]
            groups.append({"params": up_params,
                           "lr": cfg["train"].get("lr_upstream", cfg["train"]["lr"] / 10),
                           "weight_decay": cfg["train"]["weight_decay"]})
        self.optimizer = torch.optim.Adam(groups)
        self.best_iou = -1.0
        self.bad = 0
        self.patience = cfg["train"]["early_stop_patience"]
        self.log_f = open(self.output_dir / "train.jsonl", "a")

    def _move(self, b):
        out = {"video_id": b["video_id"],
               "attention_mask": b["attention_mask"].to(self.device),
               "frame_labels": b["frame_labels"].to(self.device),
               "gold_intervals": b["gold_intervals"]}
        if "per_stream" in b:
            out["per_stream"] = {s: t.to(self.device) for s, t in b["per_stream"].items()}
        if "per_stream_crops" in b:
            out["per_stream_crops"] = {s: t.to(self.device) for s, t in b["per_stream_crops"].items()}
            out["valid_mask"] = {s: t.to(self.device) for s, t in b["valid_mask"].items()}
            out["time_indices"] = {s: t.to(self.device) for s, t in b["time_indices"].items()}
        return out

    def run(self):
        for epoch in range(self.cfg["train"]["epochs"]):
            self.model.train()
            losses = []
            t0 = time.time()
            for batch in self.train_loader:
                batch = self._move(batch)
                if self.mode == "live":
                    _, loss = self.model(batch, frame_labels=batch["frame_labels"])
                else:
                    _, loss = self.model(batch["per_stream"], batch["attention_mask"], batch["frame_labels"])
                self.optimizer.zero_grad(set_to_none=True); loss.backward(); self.optimizer.step()
                losses.append(float(loss))
            iou = self._eval()
            rec = {"epoch": epoch, "train_loss": float(np.mean(losses)),
                   "val_iou": iou, "elapsed": time.time() - t0}
            print(json.dumps(rec)); self.log_f.write(json.dumps(rec) + "\n"); self.log_f.flush()
            if iou > self.best_iou:
                self.best_iou = iou; self.bad = 0
                torch.save({"epoch": epoch, "model": self.model.state_dict(), "cfg": self.cfg},
                           self.output_dir / "ckpt_best.pt")
            else:
                self.bad += 1
                if self.bad >= self.patience:
                    print("[early stop]"); break

    @torch.no_grad()
    def _eval(self) -> float:
        self.model.eval()
        ious = []
        for batch in self.val_loader:
            batch = self._move(batch)
            if self.mode == "live":
                logits = self.model(batch)
            else:
                logits = self.model(batch["per_stream"], batch["attention_mask"])
            probs = torch.sigmoid(logits).cpu().numpy()
            for i, gold in enumerate(batch["gold_intervals"]):
                t = int(batch["attention_mask"][i].sum())
                pred = probs_to_intervals(
                    probs[i, :t],
                    threshold=self.cfg["infer"]["threshold"],
                    min_len=self.cfg["infer"]["min_len"],
                    smooth_kernel=self.cfg["infer"]["smooth_kernel"],
                )
                ious.append(interval_iou(pred, gold))
        return float(np.mean(ious)) if ious else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    FsTrainer(cfg, args.output_dir).run()


if __name__ == "__main__":
    main()
