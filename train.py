"""SignDINO-v2 SSL training -- one stream per run.

Usage:
    python train.py --config configs/pretrain.yaml --stream lh --exp-name pretrain_lh
    python train.py --config configs/refine.yaml --stream lh --exp-name refine_lh \
                    --init-ckpt runs/pretrain_lh/ckpt_e100.pt

Stage 1 (pretrain): DINO + iBOT + 0.1 * DKoleo. Stage 2 (refine): adds
Gram anchoring with a frozen Gram teacher snapshotted from the stage-1
EMA teacher.

The trainer is single-GPU. The frozen DINOv3 backbone is only used when
`data.embedding_source == "live"`; with `cache` mode the embeddings are
read from disk and the backbone is not loaded.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

from data import (  # noqa: E402
    StreamClipDataset,
    multi_crop_collate,
    MultiCropConfig,
)
from data.dataset import StreamDatasetConfig  # noqa: E402
from dino import cosine_schedule, linear_warmup_cosine  # noqa: E402
from model import SignTemporalDinoModel, FrozenDinov3FrameEmbedder  # noqa: E402
from model.meta_arch import SignDinoConfig  # noqa: E402
from model.temporal_encoder import TemporalTransformer  # noqa: E402


# --------- helpers ---------------------------------------------------------

def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_dataset(cfg: dict, stream: str) -> StreamClipDataset:
    mc = MultiCropConfig(
        n_global=cfg["multi_crop"]["n_global"],
        n_local=cfg["multi_crop"]["n_local"],
        global_len_min=cfg["multi_crop"]["global_len_min"],
        global_len_max=cfg["multi_crop"]["global_len_max"],
        local_len_min=cfg["multi_crop"]["local_len_min"],
        local_len_max=cfg["multi_crop"]["local_len_max"],
        min_valid_frames=cfg["multi_crop"]["min_valid_frames"],
    )
    ds_cfg = StreamDatasetConfig(
        crop_root=cfg["data"]["crop_root"],
        stream=stream,
        split=cfg["data"]["split"],
        embedding_source=cfg["data"]["embedding_source"],
        backbone_tag=cfg["data"]["backbone_tag"],
        input_size=cfg["data"]["input_size"],
        multi_crop=mc,
        augment=cfg["data"].get("augment", True),
    )
    return StreamClipDataset(ds_cfg)


def build_model(cfg: dict, gram_weight: float | None = None) -> SignTemporalDinoModel:
    m = cfg["model"]
    model_cfg = SignDinoConfig(
        in_dim=m["in_dim"], embed_dim=m["embed_dim"],
        depth=m["depth"], num_heads=m["num_heads"],
        mlp_ratio=m["mlp_ratio"], dropout=m["dropout"],
        max_len=m["max_len"],
        head_out_dim=m["head_out_dim"], head_hidden=m["head_hidden"],
        head_bottleneck=m["head_bottleneck"], head_layers=m["head_layers"],
        head_norm_last=m["head_norm_last"],
        student_temp=m["student_temp"],
        teacher_temp=m["teacher_temp"],
        teacher_temp_start=m["teacher_temp_start"],
        sinkhorn_iters=m["sinkhorn_iters"],
        dino_weight=m["dino_weight"], ibot_weight=m["ibot_weight"],
        dkoleo_weight=m["dkoleo_weight"],
        gram_weight=m["gram_weight"] if gram_weight is None else gram_weight,
    )
    return SignTemporalDinoModel(model_cfg)


def maybe_embed_live(batch, embedder: FrozenDinov3FrameEmbedder, device):
    """If batch tensors are images (5-D: Nv, B, Tw, 3, H, W), forward each
    frame through the frozen DINOv3 backbone to get embeddings (Nv, B, Tw, D)."""
    for key in ("global_frames", "local_frames"):
        t = batch[key].to(device, non_blocking=True)
        if t.ndim == 6:
            Nv, B, Tw, C, H, W = t.shape
            with torch.no_grad():
                cls = embedder(t.reshape(Nv * B * Tw, C, H, W))    # (Nv*B*Tw, D)
            batch[key] = cls.reshape(Nv, B, Tw, -1)
        else:
            batch[key] = t
    for key in ("global_valid", "local_valid", "global_time", "local_time", "ibot_mask"):
        batch[key] = batch[key].to(device, non_blocking=True)
    return batch


# --------- main -----------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, type=str)
    ap.add_argument("--stream", required=False, default=None,
                    choices=["lh", "rh", "face"])
    ap.add_argument("--exp-name", required=False, default=None)
    ap.add_argument("--init-ckpt", default=None, type=str,
                    help="Path to a stage-1 checkpoint. Required for stage 2.")
    ap.add_argument("--resume", default=None, type=str)
    ap.add_argument("--synthetic-smoke", action="store_true",
                    help="Skip the real dataset and run on synthetic tensors (smoke test).")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.stream is not None:
        cfg["stream"] = args.stream
    if args.exp_name is not None:
        cfg["exp_name"] = args.exp_name

    set_seed(cfg.get("seed", 0))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ----- build dataset (or synthetic) -------------------------------
    if args.synthetic_smoke:
        from data.dataset import StreamDatasetConfig                                 # noqa: F811
        from torch.utils.data import Dataset                                          # noqa: F811

        class _Synth(Dataset):
            def __init__(self, n=32, D=cfg["model"]["in_dim"]):
                self.n, self.D = n, D
                self.Ng = cfg["multi_crop"]["n_global"]
                self.Nl = cfg["multi_crop"]["n_local"]
                self.gmin = cfg["multi_crop"]["global_len_min"]
                self.gmax = cfg["multi_crop"]["global_len_max"]
                self.lmin = cfg["multi_crop"]["local_len_min"]
                self.lmax = cfg["multi_crop"]["local_len_max"]

            def __len__(self):
                return self.n

            def __getitem__(self, i):
                def _draw(lmin, lmax):
                    return int(torch.randint(lmin, lmax + 1, (1,)).item())
                g = [(torch.randn(t := _draw(self.gmin, self.gmax), self.D),
                      torch.ones(t, dtype=torch.bool),
                      torch.arange(t)) for _ in range(self.Ng)]
                l = [(torch.randn(t := _draw(self.lmin, self.lmax), self.D),
                      torch.ones(t, dtype=torch.bool),
                      torch.arange(t)) for _ in range(self.Nl)]
                return {
                    "video_id": f"synth_{i}",
                    "global_frames": [x[0] for x in g], "global_valid": [x[1] for x in g],
                    "global_time": [x[2] for x in g],
                    "local_frames": [x[0] for x in l], "local_valid": [x[1] for x in l],
                    "local_time": [x[2] for x in l],
                }
        ds = _Synth()
    else:
        ds = build_dataset(cfg, cfg["stream"])

    loader = DataLoader(
        ds, batch_size=cfg["train"]["batch_size"], shuffle=True,
        num_workers=cfg["data"]["num_workers"], pin_memory=cfg["data"]["pin_memory"],
        drop_last=True,
        collate_fn=lambda b: multi_crop_collate(
            b, ibot_mask_min=cfg["ibot"]["mask_min"], ibot_mask_max=cfg["ibot"]["mask_max"]
        ),
    )

    # ----- backbone (live-mode only) ---------------------------------
    live_mode = (not args.synthetic_smoke) and cfg["data"]["embedding_source"] == "live"
    embedder = None
    if live_mode:
        embedder = FrozenDinov3FrameEmbedder(
            repo_path=cfg["backbone"]["dinov3_repo"],
            model_name=cfg["backbone"]["model_name"],
            weights_path=cfg["backbone"].get("weights_path"),
            input_size=cfg["data"]["input_size"],
            pretrained=cfg["backbone"]["pretrained"],
        ).to(device)
        embedder.eval()

    # ----- model -----------------------------------------------------
    model = build_model(cfg).to(device)

    if args.init_ckpt:
        sd = torch.load(args.init_ckpt, map_location="cpu", weights_only=True)
        missing, unexpected = model.load_state_dict(sd["model"], strict=False)
        print(f"[init] loaded {args.init_ckpt}; missing={len(missing)} unexpected={len(unexpected)}")
        if cfg["stage"] == "refine":
            # snapshot the EMA teacher encoder as the Gram teacher
            gram_teacher = TemporalTransformer(
                in_dim=cfg["model"]["in_dim"], embed_dim=cfg["model"]["embed_dim"],
                depth=cfg["model"]["depth"], num_heads=cfg["model"]["num_heads"],
                mlp_ratio=cfg["model"]["mlp_ratio"], dropout=cfg["model"]["dropout"],
                max_len=cfg["model"]["max_len"],
            ).to(device)
            gram_teacher.load_state_dict(model.teacher_encoder.state_dict())
            model.attach_gram_teacher(gram_teacher)

    # ----- optimizer + schedules -------------------------------------
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=cfg["train"]["base_lr"],
                                  weight_decay=cfg["train"]["weight_decay"])

    steps_per_epoch = max(1, len(loader))
    total_steps = steps_per_epoch * cfg["train"]["epochs"]
    warmup_steps = steps_per_epoch * cfg["train"]["warmup_epochs"]

    lr_sched = linear_warmup_cosine(
        start=cfg["train"]["base_lr"] * 1e-3,
        peak=cfg["train"]["base_lr"],
        end=cfg["train"]["min_lr"],
        total_steps=total_steps,
        warmup_steps=warmup_steps,
    )
    wd_sched = cosine_schedule(
        base=cfg["train"]["weight_decay"], final=cfg["train"]["weight_decay_end"],
        total_steps=total_steps,
    )
    momentum_sched = cosine_schedule(
        base=cfg["train"]["ema_momentum_start"], final=cfg["train"]["ema_momentum_end"],
        total_steps=total_steps,
    )
    teacher_temp_warmup_steps = steps_per_epoch * cfg["train"]["teacher_temp_warmup_epochs"]
    teacher_temp_sched = cosine_schedule(
        base=cfg["model"]["teacher_temp_start"], final=cfg["model"]["teacher_temp"],
        total_steps=max(1, teacher_temp_warmup_steps),
    ) if teacher_temp_warmup_steps > 0 else None

    # ----- output dir -------------------------------------------------
    output_dir = Path(cfg["train"]["output_root"]) / cfg["exp_name"]
    output_dir.mkdir(parents=True, exist_ok=True)
    log_jsonl = open(output_dir / "train.jsonl", "a")
    with open(output_dir / "config.yaml", "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    print(f"[train] outputs -> {output_dir}")

    # ----- train loop -------------------------------------------------
    amp_dtype = torch.bfloat16 if cfg["train"]["amp"] else torch.float32

    global_step = 0
    t0 = time.time()
    for epoch in range(cfg["train"]["epochs"]):
        model.train()
        for batch in loader:
            # move + (optionally) live embedding pass
            if live_mode:
                batch = maybe_embed_live(batch, embedder, device)
            else:
                for k in ("global_frames", "local_frames",
                          "global_valid", "local_valid",
                          "global_time", "local_time", "ibot_mask"):
                    batch[k] = batch[k].to(device, non_blocking=True)

            student_views = {
                "global_frames": batch["global_frames"], "global_valid": batch["global_valid"],
                "global_time":   batch["global_time"],
                "local_frames":  batch["local_frames"],  "local_valid":  batch["local_valid"],
                "local_time":    batch["local_time"],
                "ibot_mask":     batch["ibot_mask"],
            }
            teacher_views = {
                "global_frames": batch["global_frames"], "global_valid": batch["global_valid"],
                "global_time":   batch["global_time"],
            }

            # current schedules
            for g in optimizer.param_groups:
                g["lr"] = float(lr_sched[min(global_step, total_steps - 1)])
                g["weight_decay"] = float(wd_sched[min(global_step, total_steps - 1)])
            if teacher_temp_sched is not None and global_step < teacher_temp_warmup_steps:
                teacher_temp = float(teacher_temp_sched[global_step])
            else:
                teacher_temp = float(cfg["model"]["teacher_temp"])
            momentum = float(momentum_sched[min(global_step, total_steps - 1)])

            with torch.autocast(device_type="cuda", dtype=amp_dtype,
                                enabled=cfg["train"]["amp"] and device.type == "cuda"):
                losses = model.compute_losses(student_views, teacher_views, teacher_temp)
                loss = losses["loss_total"]

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if cfg["train"]["grad_clip"]:
                torch.nn.utils.clip_grad_norm_(params, cfg["train"]["grad_clip"])
            optimizer.step()
            model.ema_step(momentum)

            if global_step % cfg["train"]["log_every"] == 0:
                rec = {
                    "step": global_step, "epoch": epoch,
                    "lr": float(optimizer.param_groups[0]["lr"]),
                    "wd": float(optimizer.param_groups[0]["weight_decay"]),
                    "ema_m": momentum, "teacher_temp": teacher_temp,
                    **{k: float(v.detach()) for k, v in losses.items()},
                    "elapsed": time.time() - t0,
                }
                print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in rec.items()}))
                log_jsonl.write(json.dumps(rec) + "\n")
                log_jsonl.flush()
            global_step += 1

        if (epoch + 1) % cfg["train"]["ckpt_every"] == 0:
            ckpt_path = output_dir / f"ckpt_e{epoch + 1}.pt"
            torch.save(
                {"epoch": epoch + 1, "step": global_step, "model": model.state_dict(),
                 "cfg": cfg, "stage": cfg["stage"]},
                ckpt_path,
            )
            print(f"[ckpt] saved {ckpt_path}")

    log_jsonl.close()
    print("[train] done")


if __name__ == "__main__":
    main()
