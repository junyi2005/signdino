"""Evaluation metrics for the three downstream tasks.

- Translation: sacreBLEU (BLEU + chrF). BLEURT is optional and only
  imported on demand because the bleurt package is heavy.
- ISLR: Recall@1/5/10 (the SHuBERT § 4.3 standard).
- Fingerspelling: interval-level intersection-over-union (mean IoU per
  video, then averaged across the test set), following ASL-Stem Wiki.
"""
from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple

import numpy as np


# --------- Translation ---------------------------------------------------

def compute_bleu(hyps: Sequence[str], refs: Sequence[str]) -> dict:
    """BLEU via sacrebleu. Returns {'bleu': float, 'bleu_1': float, 'signature': str}."""
    try:
        import sacrebleu
    except ImportError as e:
        raise ImportError("install sacrebleu (pip install sacrebleu)") from e
    bleu = sacrebleu.corpus_bleu(list(hyps), [list(refs)])
    return {"bleu": float(bleu.score), "bleu_1": float(bleu.precisions[0]), "signature": str(bleu.get_signature())}


def compute_chrf(hyps: Sequence[str], refs: Sequence[str]) -> dict:
    try:
        import sacrebleu
    except ImportError as e:
        raise ImportError("install sacrebleu") from e
    chrf = sacrebleu.corpus_chrf(list(hyps), [list(refs)])
    return {"chrf": float(chrf.score)}


def compute_bleurt(hyps: Sequence[str], refs: Sequence[str], checkpoint: str | None = None) -> dict:
    """Optional: requires the bleurt-pytorch package and a downloaded
    checkpoint. We expose it but do not import unless called."""
    try:
        from bleurt_pytorch import BleurtConfig, BleurtForSequenceClassification, BleurtTokenizer
    except ImportError as e:
        raise ImportError("install bleurt-pytorch") from e
    import torch
    ckpt = checkpoint or "lucadiliello/BLEURT-20"
    tok = BleurtTokenizer.from_pretrained(ckpt)
    model = BleurtForSequenceClassification.from_pretrained(ckpt).eval()
    scores = []
    with torch.no_grad():
        for h, r in zip(hyps, refs):
            inputs = tok(r, h, return_tensors="pt", padding=True, truncation=True)
            scores.append(float(model(**inputs).logits.squeeze()))
    return {"bleurt": float(np.mean(scores))}


# --------- ISLR ----------------------------------------------------------

def compute_recall_at_k(scores: np.ndarray, labels: np.ndarray, ks: Sequence[int] = (1, 5, 10)) -> dict:
    """scores: (N, num_classes); labels: (N,) int. Returns {'r@1': .., 'r@5': .., ...}."""
    out = {}
    top = np.argsort(-scores, axis=-1)
    for k in ks:
        hits = (top[:, :k] == labels[:, None]).any(axis=-1).mean()
        out[f"r@{k}"] = float(hits)
    return out


# --------- Fingerspelling -----------------------------------------------

def _iv_iou(a: Tuple[int, int], b: Tuple[int, int]) -> float:
    s1, e1 = a; s2, e2 = b
    inter = max(0, min(e1, e2) - max(s1, s2) + 1)
    union = (e1 - s1 + 1) + (e2 - s2 + 1) - inter
    return inter / union if union > 0 else 0.0


def interval_iou(pred: List[Tuple[int, int]], gold: List[Tuple[int, int]]) -> float:
    """Average IoU after greedy 1-1 matching (mirrors ASL-Stem Wiki eval).

    For each gold interval pick its best-IoU prediction (unused); average
    across all gold intervals. Missing matches contribute 0.
    """
    if not gold:
        return 1.0 if not pred else 0.0
    used = set()
    ious = []
    for g in gold:
        best_i, best = -1, 0.0
        for i, p in enumerate(pred):
            if i in used:
                continue
            v = _iv_iou(p, g)
            if v > best:
                best, best_i = v, i
        if best_i >= 0:
            used.add(best_i)
        ious.append(best)
    return float(np.mean(ious))
