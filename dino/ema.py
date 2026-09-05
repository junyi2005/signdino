"""EMA teacher update + cosine schedules (momentum / temperature / lr)."""
import math

import numpy as np
import torch


class EMAUpdater:
    """teacher <- m * teacher + (1 - m) * student, for every matching param
    and buffer. The teacher receives no gradient; this is its only update.

    Caches the param lists on first call so the per-step cost is one fused
    `_foreach` multiply-add.
    """

    def __init__(self, student, teacher):
        self.student = student
        self.teacher = teacher
        self._lists = None
        # snapshot: teacher starts as an exact copy of the student
        teacher.load_state_dict(student.state_dict())
        for p in teacher.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, m):
        if self._lists is None:
            s = [p for p in self.student.parameters()]
            t = [p for p in self.teacher.parameters()]
            assert len(s) == len(t), "student/teacher param count mismatch"
            self._lists = (s, t)
        s, t = self._lists
        torch._foreach_mul_(t, m)
        torch._foreach_add_(t, s, alpha=1 - m)
        # buffers (e.g. norm running stats, if any) copied hard
        for bt, bs in zip(self.teacher.buffers(), self.student.buffers()):
            bt.copy_(bs)


def cosine_schedule(base, final, total_steps, warmup_steps=0, start_warmup=0.0):
    """Linear warmup from `start_warmup` to `base`, then cosine decay to
    `final`. Returns a numpy array of length `total_steps`."""
    warmup = np.array([])
    if warmup_steps > 0:
        warmup = np.linspace(start_warmup, base, warmup_steps)
    iters = np.arange(total_steps - warmup_steps)
    cos = final + 0.5 * (base - final) * (1 + np.cos(np.pi * iters / max(len(iters), 1)))
    sched = np.concatenate([warmup, cos])
    assert len(sched) == total_steps
    return sched


def linear_warmup_cosine(start, peak, end, total_steps, warmup_steps):
    """Variant used for the lr schedule: warm up start->peak, decay peak->end."""
    return cosine_schedule(peak, end, total_steps, warmup_steps, start_warmup=start)
