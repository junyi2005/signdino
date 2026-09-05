"""Beam-search translation inference + BLEU/chrF eval.

Usage:
    python downstream/translation/infer.py \
        --ckpt runs/translation_how2sign/ckpt_s50000.pt \
        --manifest data/how2sign/test.jsonl \
        --feature-root features/how2sign \
        --out-hyps runs/translation_how2sign/test_hyps.txt
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import yaml
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from downstream.common.fusion import FusionConfig                                    # noqa: E402
from downstream.common.metrics import compute_bleu, compute_chrf                     # noqa: E402
from downstream.translation.dataset import (                                          # noqa: E402
    TranslationDataset, TranslationDatasetConfig, translation_collate,
)
from downstream.translation.model import SignDinoTranslator, TranslatorConfig        # noqa: E402
from torch.utils.data import DataLoader                                              # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--feature-root", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--num-beams", type=int, default=5)
    ap.add_argument("--max-length", type=int, default=384)
    ap.add_argument("--out-hyps", required=True)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sd = torch.load(args.ckpt, map_location="cpu", weights_only=True)
    cfg = sd["cfg"]
    fusion_cfg = FusionConfig(**cfg["fusion"])
    model_cfg = TranslatorConfig(
        backbone_id=cfg["model"]["backbone_id"], fusion=fusion_cfg,
        label_smoothing=cfg["model"]["label_smoothing"],
        max_gen_length=args.max_length, num_beams=args.num_beams,
        length_penalty=cfg["model"]["length_penalty"],
    )
    model = SignDinoTranslator(model_cfg).to(device)
    model.load_state_dict(sd["model"], strict=False)
    model.eval()

    ds_cfg = TranslationDatasetConfig(
        feature_root=args.feature_root, manifest_jsonl=args.manifest,
        split=args.split, streams=cfg["fusion"]["streams"],
        load_per_layer=cfg["fusion"]["use_layer_weighted_sum"],
        max_frames=cfg["data"]["max_frames"],
        tokenizer_id=cfg["model"]["backbone_id"],
        max_text_length=cfg["data"]["max_text_length"],
    )
    ds = TranslationDataset(ds_cfg)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=4, pin_memory=True, collate_fn=translation_collate)

    hyps, refs, video_ids = [], [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc="infer"):
            attn = batch["attention_mask"].to(device)
            per_stream = {s: t.to(device) for s, t in batch["per_stream"].items()}
            ids = model.generate(per_stream, attn)
            decoded = model.tokenizer.batch_decode(ids, skip_special_tokens=True)
            hyps.extend(decoded)
            video_ids.extend(batch["video_id"])
            # decode references from labels (replace -100 with pad)
            ref_labels = batch["labels"].clone()
            ref_labels[ref_labels == -100] = model.tokenizer.pad_token_id
            refs.extend(model.tokenizer.batch_decode(ref_labels, skip_special_tokens=True))

    with open(args.out_hyps, "w") as f:
        for vid, hyp in zip(video_ids, hyps):
            f.write(json.dumps({"video_id": vid, "hyp": hyp}) + "\n")

    bleu = compute_bleu(hyps, refs)
    chrf = compute_chrf(hyps, refs)
    print(json.dumps({**bleu, **chrf}, indent=2))


if __name__ == "__main__":
    main()
