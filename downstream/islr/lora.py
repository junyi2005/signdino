"""Minimal rank-r LoRA wrapper for nn.Linear.

For each target `nn.Linear(in, out)` we add a low-rank residual:
    y = x @ W_base^T + (alpha / r) * x @ A^T @ B^T,
where A: (r, in) is initialised as N(0, sigma^2) and B: (out, r) starts at
zero so the LoRA contribution is initially nil. W_base is frozen.

`apply_lora_to(module, rank=1)` walks `module`, replacing every nn.Linear
with a LoRALinear and freezing the base weight. Returns the list of
newly created LoRA-parameter handles for the optimizer (useful for
setting their LR separately, per SHuBERT § 4.3: LoRA lr = 1/10 head lr).
"""
from __future__ import annotations

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int = 1, alpha: float = 1.0):
        super().__init__()
        assert isinstance(base, nn.Linear)
        self.in_features = base.in_features
        self.out_features = base.out_features
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        # frozen base
        self.weight = nn.Parameter(base.weight.detach().clone(), requires_grad=False)
        self.bias = None
        if base.bias is not None:
            self.bias = nn.Parameter(base.bias.detach().clone(), requires_grad=False)

        # trainable low-rank update
        self.lora_A = nn.Parameter(torch.zeros(rank, self.in_features))
        self.lora_B = nn.Parameter(torch.zeros(self.out_features, rank))
        nn.init.normal_(self.lora_A, std=1.0 / max(self.in_features, 1) ** 0.5)
        # lora_B stays at zero so the initial output equals the frozen base

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = F.linear(x, self.weight, self.bias)
        update = F.linear(F.linear(x, self.lora_A), self.lora_B) * self.scaling
        return base + update

    def lora_parameters(self) -> List[nn.Parameter]:
        return [self.lora_A, self.lora_B]


def apply_lora_to(root: nn.Module, rank: int = 1, alpha: float = 1.0) -> List[nn.Parameter]:
    """In-place: replace every nn.Linear submodule of `root` with LoRALinear.
    Returns the flat list of newly-trainable LoRA parameters."""
    out: List[nn.Parameter] = []
    for name, m in list(root.named_modules()):
        # walk children of m; can't replace m itself if it IS a Linear
        for child_name, child in list(m.named_children()):
            if isinstance(child, nn.Linear):
                wrapped = LoRALinear(child, rank=rank, alpha=alpha)
                setattr(m, child_name, wrapped)
                out.extend(wrapped.lora_parameters())
    # Freeze every non-LoRA parameter under `root` (LoRA params are fresh leaves
    # not registered under the original linears any more, so they remain trainable).
    for n, p in root.named_parameters():
        if "lora_A" not in n and "lora_B" not in n:
            p.requires_grad_(False)
    return out
