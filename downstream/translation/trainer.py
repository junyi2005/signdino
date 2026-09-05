"""Two-phase translation trainer.

Phase 1: large-corpus pre-training (e.g. YouTube-ASL), 250K steps,
   AdamW lr 5e-4 for ByT5, 5e-5 for SignDINO (when fine-tuned), 10K warmup
   cosine, batch 2 utts/GPU × grad-accum 8, weight decay 0.1.

Phase 2: benchmark fine-tuning (How2Sign / OpenASL), 50K steps,
   lr 1e-4 with 5K warmup cosine.

Both phases use cross-entropy with label-smoothing 0.2 (handled inside
SignDinoTranslator).
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from downstream.common.fusion import FusionConfig                            # noqa: E402
from downstream.translation.dataset import (                                   # noqa: E402
    TranslationDataset, TranslationDatasetConfig, translation_collate,
    LiveTranslationDataset, LiveTranslationDatasetConfig, live_translation_collate,
)
from downstream.translation.model import (                                     # noqa: E402
    SignDinoTranslator, TranslatorConfig, LiveSignDinoTranslator,
)
from model.live_upstream import LiveUpstream, LiveUpstreamConfig              # noqa: E402


def cosine_warmup(base_lr: float, warmup_steps: int, total_steps: int):
    def lr_at(step: int) -> float:
        if step < warmup_steps:
            return base_lr * (step + 1) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))
    return lr_at


class TranslationTrainer:
    def __init__(self, cfg: dict, output_dir: Path):
        self.cfg = cfg
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        fusion_cfg = FusionConfig(
            streams=cfg["fusion"]["streams"],
            in_dim=cfg["fusion"]["in_dim"],
            proj_dim=cfg["fusion"]["proj_dim"],
            out_dim=cfg["fusion"]["out_dim"],
            use_layer_weighted_sum=cfg["fusion"]["use_layer_weighted_sum"],
            n_layers=cfg["fusion"]["n_layers"],
        )
        model_cfg = TranslatorConfig(
            backbone_id=cfg["model"]["backbone_id"],
            fusion=fusion_cfg,
            label_smoothing=cfg["model"]["label_smoothing"],
            max_gen_length=cfg["model"]["max_gen_length"],
            num_beams=cfg["model"]["num_beams"],
            length_penalty=cfg["model"]["length_penalty"],
        )
        base_model = SignDinoTranslator(model_cfg).to(self.device)
        if cfg.get("init_ckpt"):
            sd = torch.load(cfg["init_ckpt"], map_location="cpu", weights_only=True)
            base_model.load_state_dict(sd["model"], strict=False)

        # ----- mode dispatch ---------------------------------------------
        self.mode = cfg.get("mode", "features")
        if self.mode == "live":
            up_cfg = LiveUpstreamConfig(
                streams=cfg["fusion"]["streams"],
                ckpts=cfg["live"]["upstream_ckpts"],          # {'lh': '..', 'rh': '..', 'face': '..'}
                dinov3_repo=cfg["live"]["dinov3_repo"],
                dinov3_model_name=cfg["live"]["dinov3_model_name"],
                dinov3_weights_path=cfg["live"].get("dinov3_weights_path"),
                dinov3_pretrained=cfg["live"].get("dinov3_pretrained", True),
                dinov3_input_size=cfg["live"].get("dinov3_input_size", 224),
            )
            up = LiveUpstream(up_cfg).to(self.device)
            self.model = LiveSignDinoTranslator(base_model, up).to(self.device)
            ds_cfg = LiveTranslationDatasetConfig(
                crop_root=cfg["data"]["crop_root"],
                manifest_jsonl=cfg["data"]["manifest_jsonl"],
                split=cfg["data"]["split"],
                streams=cfg["fusion"]["streams"],
                input_size=cfg["live"].get("dinov3_input_size", 224),
                max_frames=cfg["data"]["max_frames"],
                frame_stride=cfg["data"].get("frame_stride", 2),
                tokenizer_id=cfg["model"]["backbone_id"],
                max_text_length=cfg["data"]["max_text_length"],
            )
            self.dataset = LiveTranslationDataset(ds_cfg)
            self.collate = live_translation_collate
        else:
            self.model = base_model
            ds_cfg = TranslationDatasetConfig(
                feature_root=cfg["data"]["feature_root"],
                manifest_jsonl=cfg["data"]["manifest_jsonl"],
                split=cfg["data"]["split"],
                streams=cfg["fusion"]["streams"],
                load_per_layer=cfg["fusion"]["use_layer_weighted_sum"],
                max_frames=cfg["data"]["max_frames"],
                tokenizer_id=cfg["model"]["backbone_id"],
                max_text_length=cfg["data"]["max_text_length"],
            )
            self.dataset = TranslationDataset(ds_cfg)
            self.collate = translation_collate

        self.loader = DataLoader(
            self.dataset, batch_size=cfg["train"]["per_gpu_batch_size"],
            shuffle=True, num_workers=cfg["data"]["num_workers"],
            pin_memory=True, collate_fn=self.collate, drop_last=True,
        )

        # ----- parameter groups (3-tier when in live mode) ----------------
        byt5_params, head_params, upstream_params = [], [], []
        for n, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            if "live_upstream.frame_embedder" in n:
                continue                                # DINOv3 is always frozen
            if "live_upstream.encoders" in n:
                upstream_params.append(p)
            elif ".byt5." in n or n.startswith("byt5."):
                byt5_params.append(p)
            else:
                head_params.append(p)
        lr_byt5 = cfg["train"]["lr_byt5"]
        lr_head = cfg["train"].get("lr_head", lr_byt5)
        lr_up   = cfg["train"].get("lr_upstream", lr_byt5 / 10)
        groups = [
            {"params": byt5_params, "lr": lr_byt5},
            {"params": head_params, "lr": lr_head},
        ]
        if upstream_params:
            groups.append({"params": upstream_params, "lr": lr_up})
        self.optimizer = torch.optim.AdamW(groups, weight_decay=cfg["train"]["weight_decay"])

        self.total_steps = cfg["train"]["total_steps"]
        self.warmup_steps = cfg["train"]["warmup_steps"]
        self.grad_accum = cfg["train"]["grad_accum"]
        self.lr_byt5_sched = cosine_warmup(lr_byt5, self.warmup_steps, self.total_steps)
        self.lr_head_sched = cosine_warmup(lr_head, self.warmup_steps, self.total_steps)
        self.lr_up_sched   = cosine_warmup(lr_up,   self.warmup_steps, self.total_steps)
        self.amp = cfg["train"].get("amp", True)
        self.log_every = cfg["train"].get("log_every", 50)
        self.ckpt_every = cfg["train"].get("ckpt_every", 5000)

        self.log_f = open(self.output_dir / "train.jsonl", "a")

    def run(self):
        step = 0
        accum = 0
        t0 = time.time()
        self.model.train()
        amp_dtype = torch.bfloat16 if self.amp else torch.float32
        while step < self.total_steps:
            for batch in self.loader:
                batch = self._move(batch)
                with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=self.amp and self.device.type == "cuda"):
                    if self.mode == "live":
                        loss = self.model(batch)
                    else:
                        loss = self.model(batch["per_stream"], batch["attention_mask"], batch["labels"])
                    loss = loss / self.grad_accum
                loss.backward()
                accum += 1
                if accum % self.grad_accum == 0:
                    self.optimizer.param_groups[0]["lr"] = self.lr_byt5_sched(step)
                    self.optimizer.param_groups[1]["lr"] = self.lr_head_sched(step)
                    if len(self.optimizer.param_groups) > 2:
                        self.optimizer.param_groups[2]["lr"] = self.lr_up_sched(step)
                    torch.nn.utils.clip_grad_norm_([p for g in self.optimizer.param_groups for p in g["params"]], 1.0)
                    self.optimizer.step()
                    self.optimizer.zero_grad(set_to_none=True)
                    if step % self.log_every == 0:
                        rec = {"step": step, "loss": float(loss) * self.grad_accum,
                               "lr_byt5": self.optimizer.param_groups[0]["lr"],
                               "lr_head": self.optimizer.param_groups[1]["lr"],
                               "elapsed": time.time() - t0}
                        print(json.dumps(rec))
                        self.log_f.write(json.dumps(rec) + "\n"); self.log_f.flush()
                    if step > 0 and step % self.ckpt_every == 0:
                        self._save(step)
                    step += 1
                    if step >= self.total_steps:
                        break
        self._save(step)

    def _move(self, batch):
        out = {"video_id": batch["video_id"],
               "attention_mask": batch["attention_mask"].to(self.device),
               "labels": batch["labels"].to(self.device)}
        if "per_stream" in batch:
            out["per_stream"] = {s: t.to(self.device) for s, t in batch["per_stream"].items()}
        if "per_stream_crops" in batch:
            out["per_stream_crops"] = {s: t.to(self.device) for s, t in batch["per_stream_crops"].items()}
            out["valid_mask"] = {s: t.to(self.device) for s, t in batch["valid_mask"].items()}
            out["time_indices"] = {s: t.to(self.device) for s, t in batch["time_indices"].items()}
        return out

    def _save(self, step: int):
        path = self.output_dir / f"ckpt_s{step}.pt"
        torch.save({"step": step, "model": self.model.state_dict(), "cfg": self.cfg}, path)
        print(f"[ckpt] {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    TranslationTrainer(cfg, args.output_dir).run()


if __name__ == "__main__":
    main()
