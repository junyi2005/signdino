"""Gram-anchoring loss (stage 2). Ported from dinov3.loss.gram_loss with no
distributed code paths -- works on plain tensors.

Constrains the patch-patch (here: frame-frame) similarity structure of the
student to match that of a frozen Gram teacher, rather than constraining
the patch features themselves. Preserves the temporal grammar of a clip
once stage 1 has driven the per-frame tokens towards the CLS.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class GramLoss(nn.Module):
    def __init__(self, apply_norm: bool = True, remove_neg: bool = True):
        super().__init__()
        self.apply_norm = apply_norm
        self.remove_neg = remove_neg
        self.mse = nn.MSELoss()

    def forward(self, student_tokens: torch.Tensor, gram_teacher_tokens: torch.Tensor) -> torch.Tensor:
        """student_tokens, gram_teacher_tokens: (B, T, D) per-frame token tensors.

        Returns the mean squared Frobenius distance between the per-clip
        normalised gram matrices.
        """
        s = student_tokens.float()
        g = gram_teacher_tokens.float()
        if self.apply_norm:
            s = F.normalize(s, dim=-1)
            g = F.normalize(g, dim=-1)
        s_sim = torch.matmul(s, s.transpose(-1, -2))
        g_sim = torch.matmul(g, g.transpose(-1, -2))
        if self.remove_neg:
            s_sim = s_sim.clamp(min=0.0)
            g_sim = g_sim.clamp(min=0.0)
        return self.mse(s_sim, g_sim)
