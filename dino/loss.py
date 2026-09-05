"""DINO / iBOT / KoLeo losses -- single-GPU, modality-agnostic.

Ported from dinov3/loss/* with the distributed all-reduce paths removed.
Everything here consumes plain tensors, so the same code drives the
image DINO and the sign-pose DINO unchanged.
"""
import math

import torch
import torch.nn.functional as F
from torch import nn


class DINOLoss(nn.Module):
    """Cross-entropy between sharpened student logits and a Sinkhorn-balanced
    teacher distribution, over the CLS / clip-level token."""

    def __init__(self, out_dim, student_temp=0.1):
        super().__init__()
        self.student_temp = student_temp
        self.out_dim = out_dim

    @torch.no_grad()
    def sinkhorn_knopp_teacher(self, teacher_output, teacher_temp, n_iterations=3):
        """Balanced assignment of `teacher_output` [N, K] -> probs [N, K].

        Sinkhorn keeps the K prototypes uniformly used across the batch,
        which is the main thing that stops the student-teacher pair from
        collapsing to a constant.
        """
        Q = torch.exp(teacher_output.float() / teacher_temp).t()  # K x N
        K, N = Q.shape
        Q /= Q.sum()
        for _ in range(n_iterations):
            Q /= Q.sum(dim=1, keepdim=True)
            Q /= K
            Q /= Q.sum(dim=0, keepdim=True)
            Q /= N
        Q *= N
        return Q.t()  # N x K, rows sum to 1

    def forward(self, student_logits, teacher_probs, ignore_diagonal=False):
        """student_logits [S, B, K], teacher_probs [T, B, K] (rows sum to 1).

        Averages cross-entropy over every (student-view, teacher-view) pair.
        With ignore_diagonal the s==t pairs are skipped (used for the
        global-vs-global term so a view is never its own target).
        """
        S, B, K = student_logits.shape
        T = teacher_probs.shape[0]
        student_lsm = F.log_softmax(student_logits.float() / self.student_temp, dim=-1)
        if not ignore_diagonal:
            loss = -torch.einsum("sbk,tbk->", student_lsm, teacher_probs)
            return loss / (B * S * T)
        loss = -torch.einsum("sbk,tbk->st", student_lsm, teacher_probs)
        m = min(S, T)
        loss = torch.diagonal_scatter(loss, loss.new_zeros(m))
        return loss.sum() / (B * S * T - B * m)


class iBOTFrameLoss(nn.Module):
    """Masked-frame modelling: student must predict the teacher's per-frame
    token distribution at frames the student saw masked.

    This is the temporal analogue of iBOT's masked-patch loss.
    """

    def __init__(self, out_dim, student_temp=0.1):
        super().__init__()
        self.student_temp = student_temp
        self.out_dim = out_dim

    @torch.no_grad()
    def sinkhorn_knopp_teacher(self, teacher_output, teacher_temp, n_iterations=3):
        Q = torch.exp(teacher_output.float() / teacher_temp).t()
        K, N = Q.shape
        if N == 0:
            return teacher_output
        Q /= Q.sum()
        for _ in range(n_iterations):
            Q /= Q.sum(dim=1, keepdim=True)
            Q /= K
            Q /= Q.sum(dim=0, keepdim=True)
            Q /= N
        Q *= N
        return Q.t()

    def forward(self, student_masked, teacher_masked_probs):
        """student_masked [Nmasked, K], teacher_masked_probs [Nmasked, K].

        Both tensors are the post-head outputs gathered over exactly the
        masked frames of the batch, in the same order.
        """
        if student_masked.shape[0] == 0:
            return student_masked.new_zeros(())
        student_lsm = F.log_softmax(student_masked.float() / self.student_temp, dim=-1)
        return -(teacher_masked_probs * student_lsm).sum(dim=-1).mean()


class KoLeoLoss(nn.Module):
    """Kozachenko-Leonenko differential-entropy regularizer: pushes each
    sample away from its nearest neighbour so features spread out and do
    not collapse. Ported from dinov3/loss/koleo_loss.py."""

    def __init__(self):
        super().__init__()
        self.pdist = nn.PairwiseDistance(2, eps=1e-8)

    def pairwise_NNs_inner(self, x):
        dots = x @ x.t()
        n = x.shape[0]
        dots.view(-1)[:: (n + 1)].fill_(-1)  # mask the diagonal
        return dots.max(dim=1).indices

    def forward(self, student_output, eps=1e-8):
        x = F.normalize(student_output, eps=eps, p=2, dim=-1)
        idx = self.pairwise_NNs_inner(x)
        dist = self.pdist(x, x[idx])
        return -torch.log(dist + eps).mean()
