"""ISLR trainer.

SHuBERT § 4.3 recipe: 125 epochs, batch 128, Adam, lr 1e-4, weight-decay
1e-4, early stop on Recall@1. With LoRA: LoRA lr = 1/10 of head lr.
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

from downstream.common.fusion import FusionConfig                                # noqa: E402
from downstream.common.metrics import compute_recall_at_k                        # noqa: E402
from downstream.islr.dataset import (                                             # noqa: E402
    ISLRDataset, ISLRDatasetConfig, islr_collate,
    LiveISLRDataset, LiveISLRDatasetConfig, live_islr_collate,
)
from downstream.islr.model import SignDinoISLR, ISLRConfig, LiveSignDinoISLR     # noqa: E402
from downstream.islr.lora import apply_lora_to                                    # noqa: E402
from model.live_upstream import LiveUpstream, LiveUpstreamConfig                 # noqa: E402


class ISLRTrainer:
    def __init__(self, cfg: dict, output_dir: Path):
        self.cfg = cfg
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.mode = cfg.get("mode", "features")
        if self.mode == "live":
            self.train_ds = LiveISLRDataset(LiveISLRDatasetConfig(
                crop_root=cfg["data"]["crop_root"],
                manifest_jsonl=cfg["data"]["manifest_jsonl"], split="train",
                streams=cfg["fusion"]["streams"],
                input_size=cfg["live"].get("dinov3_input_size", 224),
                max_frames=cfg["data"]["max_frames"],
                frame_stride=cfg["data"].get("frame_stride", 1),
            ))
            self.val_ds = LiveISLRDataset(LiveISLRDatasetConfig(
                crop_root=cfg["data"]["crop_root"],
                manifest_jsonl=cfg["data"]["manifest_jsonl"], split="val",
                streams=cfg["fusion"]["streams"],
                input_size=cfg["live"].get("dinov3_input_size", 224),
                max_frames=cfg["data"]["max_frames"],
                frame_stride=cfg["data"].get("frame_stride", 1),
                label_vocab=self.train_ds.vocab,
            ))
            self.collate = live_islr_collate
        else:
            self.train_ds = ISLRDataset(ISLRDatasetConfig(
                feature_root=cfg["data"]["feature_root"],
                manifest_jsonl=cfg["data"]["manifest_jsonl"],
                split="train", load_per_layer=cfg["fusion"]["use_layer_weighted_sum"],
                max_frames=cfg["data"]["max_frames"],
            ))
            self.val_ds = ISLRDataset(ISLRDatasetConfig(
                feature_root=cfg["data"]["feature_root"],
                manifest_jsonl=cfg["data"]["manifest_jsonl"],
                split="val", load_per_layer=cfg["fusion"]["use_layer_weighted_sum"],
                max_frames=cfg["data"]["max_frames"],
                label_vocab=self.train_ds.vocab,
            ))
            self.collate = islr_collate

        self.train_loader = DataLoader(self.train_ds, batch_size=cfg["train"]["batch_size"],
                                       shuffle=True, num_workers=cfg["data"]["num_workers"],
                                       collate_fn=self.collate, drop_last=True, pin_memory=True)
        self.val_loader = DataLoader(self.val_ds, batch_size=cfg["train"]["batch_size"],
                                     shuffle=False, num_workers=cfg["data"]["num_workers"],
                                     collate_fn=self.collate, pin_memory=True)

        fusion_cfg = FusionConfig(**cfg["fusion"])
        model_cfg = ISLRConfig(
            num_classes=self.train_ds.num_classes(),
            fusion=fusion_cfg,
            label_smoothing=cfg["model"]["label_smoothing"],
        )
        base_model = SignDinoISLR(model_cfg).to(self.device)

        lora_params = []
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
            # rank-1 LoRA on every nn.Linear in the temporal encoders (DINOv3 untouched)
            lora_params = apply_lora_to(up.encoders,
                                        rank=cfg["live"].get("lora_rank", 1),
                                        alpha=cfg["live"].get("lora_alpha", 1.0))
            self.model = LiveSignDinoISLR(base_model, up).to(self.device)
        else:
            self.model = base_model

        # Param groups: head/fusion (lr_head, wd), LoRA (lr_head/10, no wd)
        head_params = [p for n, p in self.model.named_parameters()
                       if p.requires_grad and "lora_A" not in n and "lora_B" not in n]
        groups = [{"params": head_params, "lr": cfg["train"]["lr_head"],
                   "weight_decay": cfg["train"]["weight_decay"]}]
        if lora_params:
            groups.append({"params": lora_params,
                           "lr": cfg["train"].get("lr_lora", cfg["train"]["lr_head"] / 10),
                           "weight_decay": 0.0})
        self.optimizer = torch.optim.Adam(groups)

        self.best_r1 = -1.0
        self.patience = cfg["train"]["early_stop_patience"]
        self.bad = 0
        self.log_f = open(self.output_dir / "train.jsonl", "a")

    def _move(self, b):
        out = {"video_id": b["video_id"], "labels": b["labels"].to(self.device),
               "attention_mask": b["attention_mask"].to(self.device)}
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
            t0 = time.time()
            losses = []
            for batch in self.train_loader:
                batch = self._move(batch)
                if self.mode == "live":
                    logits, loss = self.model(batch, labels=batch["labels"])
                else:
                    logits, loss = self.model(batch["per_stream"], batch["attention_mask"], batch["labels"])
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                self.optimizer.step()
                losses.append(float(loss))
            train_loss = float(np.mean(losses))
            metrics = self._eval()
            rec = {"epoch": epoch, "train_loss": train_loss, **metrics, "elapsed": time.time() - t0}
            print(json.dumps(rec))
            self.log_f.write(json.dumps(rec) + "\n"); self.log_f.flush()
            if metrics["r@1"] > self.best_r1:
                self.best_r1 = metrics["r@1"]; self.bad = 0
                torch.save({"epoch": epoch, "model": self.model.state_dict(),
                            "cfg": self.cfg, "vocab": self.train_ds.vocab},
                           self.output_dir / "ckpt_best.pt")
            else:
                self.bad += 1
                if self.bad >= self.patience:
                    print(f"[early stop] {self.bad} epochs without r@1 improvement.")
                    break

    @torch.no_grad()
    def _eval(self):
        self.model.eval()
        all_scores, all_labels = [], []
        for batch in self.val_loader:
            batch = self._move(batch)
            if self.mode == "live":
                logits = self.model(batch)
            else:
                logits = self.model(batch["per_stream"], batch["attention_mask"])
            all_scores.append(logits.cpu().numpy())
            all_labels.append(batch["labels"].cpu().numpy())
        scores = np.concatenate(all_scores, axis=0)
        labels = np.concatenate(all_labels, axis=0)
        return compute_recall_at_k(scores, labels, ks=(1, 5, 10))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    ISLRTrainer(cfg, args.output_dir).run()


if __name__ == "__main__":
    main()
