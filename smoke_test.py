"""End-to-end smoke test on synthetic embeddings.

Verifies that the data collate, multi-crop view assembly, model forward,
all four losses (DINO/iBOT/DKoleo/optional Gram), backprop, and EMA
update all work together. Does NOT need any real data or the frozen
DINOv3 backbone.

Run:
    python smoke_test.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

from data import multi_crop_collate, MultiCropConfig                                # noqa: E402
from model import SignTemporalDinoModel                                              # noqa: E402
from model.meta_arch import SignDinoConfig                                          # noqa: E402
from model.temporal_encoder import TemporalTransformer                              # noqa: E402


def _synth_sample(i, Ng=2, Nl=8, g_range=(24, 32), l_range=(6, 12), D=768):
    """Each window draws its own length to exercise the padding collate."""
    def _t():
        return None
    def _draw(rng):
        lo, hi = rng
        return int(torch.randint(lo, hi + 1, (1,)).item())
    g_lens = [_draw(g_range) for _ in range(Ng)]
    l_lens = [_draw(l_range) for _ in range(Nl)]
    g = [(torch.randn(T, D), torch.ones(T, dtype=torch.bool), torch.arange(T)) for T in g_lens]
    l = [(torch.randn(T, D), torch.ones(T, dtype=torch.bool), torch.arange(T)) for T in l_lens]
    return {
        "video_id": f"synth_{i}",
        "global_frames": [x[0] for x in g], "global_valid": [x[1] for x in g], "global_time": [x[2] for x in g],
        "local_frames":  [x[0] for x in l], "local_valid":  [x[1] for x in l], "local_time":  [x[2] for x in l],
    }


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[smoke] device={device}")

    batch_size = 4
    D = 768
    batch = multi_crop_collate([_synth_sample(i, D=D) for i in range(batch_size)])
    print(f"[smoke] padded shapes  globals={tuple(batch['global_frames'].shape)} "
          f"locals={tuple(batch['local_frames'].shape)}")
    for k in ("global_frames", "local_frames", "global_valid", "local_valid",
              "global_time", "local_time", "ibot_mask"):
        batch[k] = batch[k].to(device)

    # build a small model (still ViT-B input dim so configs match)
    cfg = SignDinoConfig(
        in_dim=D, embed_dim=192, depth=2, num_heads=4, mlp_ratio=2.0,
        max_len=64, head_out_dim=2048, head_hidden=512, head_bottleneck=128,
        head_layers=2,
    )
    model = SignTemporalDinoModel(cfg).to(device)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=5e-4)

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

    # ----- stage 1: pre-training losses ------------------------------
    print("[smoke] stage 1 step ...")
    t0 = time.time()
    losses = model.compute_losses(student_views, teacher_views, teacher_temp=0.07)
    loss = losses["loss_total"]
    loss.backward()
    opt.step()
    opt.zero_grad()
    model.ema_step(0.99)
    print(f"  loss_total={float(loss):.4f}  dino={float(losses['loss_dino']):.4f}  "
          f"ibot={float(losses['loss_ibot']):.4f}  dkoleo={float(losses['loss_dkoleo']):.4f}  "
          f"({time.time() - t0:.2f}s)")
    assert torch.isfinite(loss), "stage 1 loss is NaN/inf"

    # ----- stage 2: Gram anchoring step ------------------------------
    print("[smoke] stage 2 step (Gram on) ...")
    gram_teacher = TemporalTransformer(
        in_dim=cfg.in_dim, embed_dim=cfg.embed_dim, depth=cfg.depth,
        num_heads=cfg.num_heads, mlp_ratio=cfg.mlp_ratio,
        max_len=cfg.max_len,
    ).to(device)
    gram_teacher.load_state_dict(model.teacher_encoder.state_dict())
    model.attach_gram_teacher(gram_teacher)
    model.cfg.gram_weight = 2.0

    t0 = time.time()
    losses = model.compute_losses(student_views, teacher_views, teacher_temp=0.07)
    loss = losses["loss_total"]
    loss.backward()
    opt.step()
    opt.zero_grad()
    model.ema_step(0.99)
    print(f"  loss_total={float(loss):.4f}  gram={float(losses['loss_gram']):.4f}  ({time.time() - t0:.2f}s)")
    assert "loss_gram" in losses
    assert torch.isfinite(loss), "stage 2 loss is NaN/inf"

    print("[smoke] OK")


if __name__ == "__main__":
    main()
