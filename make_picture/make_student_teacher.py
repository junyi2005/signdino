"""Student/teacher mechanism, instantiated for one anatomical stream.

This figure complements the pipeline figure by zooming in on the temporal
SSL game: how a single stream's per-frame embedding tensor is split into
global vs local temporal crops, fed in parallel to an EMA teacher and a
student, and how DINO / iBOT / DKoleo / Gram losses are wired up.

Layout (textwidth):
  Row 1 — input embedding strip with global / local crop highlights
  Row 2 — teacher (left) and student (right) branches in tinted bands
  Row 3 — Stage-1 loss chips (DINO, iBOT, DKoleo) — chips fill the box
  Row 4 — Stage-2 Gram refinement (left text, right loss equation)

Compared to the previous draft, this version:
  * Reduces the height of the loss + Gram boxes so they snugly enclose their
    content (was: tall boxes with huge empty regions inside).
  * Bumps the chip heights so each loss formula reads at scan distance.
  * Centers the Gram equation chip across the lower box instead of
    floating in the right half with a 2.4-unit dead zone in the middle.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch, FancyArrowPatch

import style

style.apply()

OUT = Path(__file__).parent / "student_teacher.pdf"
P = style.PALETTE


def _round_box(ax, x, y, w, h, fc, ec="#404040", lw=0.7, alpha=1.0, z=2):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.0,rounding_size=0.05",
        facecolor=fc, edgecolor=ec, linewidth=lw, alpha=alpha, zorder=z))


def _arrow(ax, x1, y1, x2, y2, color="#404040", lw=0.9, dashed=False,
           rad=0.0, z=4):
    ax.add_patch(FancyArrowPatch(
        (x1, y1), (x2, y2),
        connectionstyle=f"arc3,rad={rad}",
        arrowstyle="-|>,head_width=3.5,head_length=5.5",
        linestyle=("--" if dashed else "-"),
        color=color, lw=lw, zorder=z, mutation_scale=1.0))


def main() -> None:
    # No figure title — caller's LaTeX caption carries that.
    fig = plt.figure(figsize=(7.0, 3.80))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 7.60)
    ax.set_axis_off()

    # =================================================================
    # ROW 1 — input embedding strip
    # =================================================================
    _round_box(ax, 0.30, 6.50, 1.95, 0.78, P["cstudent"])
    ax.text(1.28, 6.89, "frozen DINOv3\nViT-B/16",
            ha="center", va="center", fontsize=7.1, color="#202020",
            linespacing=1.05)

    strip_x0, strip_y = 2.95, 6.60
    slot_w, slot_h = 0.34, 0.55
    n_strip = 28
    for k in range(n_strip):
        ax.add_patch(Rectangle(
            (strip_x0 + k * slot_w, strip_y), slot_w - 0.02, slot_h,
            facecolor=P["ctube"], edgecolor="#888888", linewidth=0.30))
    ax.text(strip_x0 + n_strip * slot_w + 0.12, strip_y + slot_h / 2,
            "$E^{L}_{1:T}$",
            ha="left", va="center", fontsize=8.0, color="#202020")
    _arrow(ax, 2.30, strip_y + slot_h / 2, 2.95, strip_y + slot_h / 2)

    # ---- global crop ----
    g_k0, g_w = 2, 14
    gx = strip_x0 + g_k0 * slot_w
    ax.add_patch(Rectangle((gx, strip_y - 0.06),
                           g_w * slot_w - 0.02, slot_h + 0.12,
                           facecolor="none", edgecolor=P["cdarkgreen"],
                           linewidth=1.4))
    # Place the label left-anchored at the LEFT edge of the green box so its
    # first letter 'g' sits where the green arrow leaves the strip.
    green_lbl_x = gx + 0.04            # x position of the 'g' of "global"
    ax.text(green_lbl_x, strip_y + slot_h + 0.20,
            r"global crop  $T_g\!\in\![64,96]$  ($N_g{=}2$)",
            ha="left", va="bottom", fontsize=6.8,
            color=P["cdarkgreen"], fontweight="bold")

    # ---- 3 local crops at (inside / edge / outside) of the global window ----
    # Global covers frames 2..15; each local is 4 frames wide.
    local_boxes = [(5, 4), (13, 4), (20, 4)]    # (l_k0, l_w): inside, straddles end, outside
    inside_l_k0, inside_l_w = local_boxes[0]
    lx = strip_x0 + inside_l_k0 * slot_w        # used for downstream label/arrow anchors
    l_w = inside_l_w
    for lk, lw in local_boxes:
        lkx = strip_x0 + lk * slot_w
        ax.add_patch(Rectangle((lkx, strip_y - 0.06),
                               lw * slot_w - 0.02, slot_h + 0.12,
                               facecolor="none", edgecolor=P["cdarkorange"],
                               linewidth=1.4))
    # local label anchored at the LEFT edge of the outside (rightmost) local
    # crop, so its first letter 'l' is where the purple arrow leaves.
    outside_l_k0, outside_l_w = local_boxes[2]
    purple_lbl_x = strip_x0 + outside_l_k0 * slot_w + 0.04
    ax.text(purple_lbl_x, strip_y + slot_h + 0.20,
            r"local crops  $T_l\!\in\![10,32]$  ($N_l{=}8$)",
            ha="left", va="bottom", fontsize=6.8,
            color=P["cdarkorange"], fontweight="bold")

    # iBOT mask glyphs (inside the "inside" local crop) + label BELOW the
    # strip so it doesn't collide with the global / local crop labels above.
    mask_xs = []
    for k_mask in (inside_l_k0 + 1, inside_l_k0 + 2):
        mx = strip_x0 + k_mask * slot_w
        mask_xs.append(mx)
        ax.add_patch(Rectangle(
            (mx, strip_y), slot_w - 0.02, slot_h,
            facecolor="#7A7A7A", edgecolor="#3A3A3A",
            linewidth=0.4, hatch="////", alpha=0.65))
    mask_centre_x = (mask_xs[0] + mask_xs[-1] + slot_w - 0.02) / 2
    ax.text(mask_centre_x, strip_y - 0.10,
            r"iBOT $\mathtt{[MASK]}$",
            ha="center", va="top", fontsize=6.8,
            color="#202020", fontweight="bold")

    # =================================================================
    # ROW 2 — teacher (left) and student (right) branches
    # =================================================================
    # ---- TEACHER ----
    ax.add_patch(Rectangle((0.30, 3.70), 6.40, 2.00,
                           facecolor=P["cteacher"], alpha=0.32,
                           edgecolor="none", zorder=1))
    ax.text(0.45, 5.58, "Teacher branch  (EMA, no gradient)",
            ha="left", va="top", fontsize=7.7,
            color="#8B7000", fontweight="bold")

    _round_box(ax, 0.55, 4.40, 2.30, 0.80, "white", ec="#9F8C00", lw=0.8)
    ax.text(1.70, 4.80,
            r"EMA teacher" "\n" r"temporal ViT  $f_{\overline{\phi}}$",
            ha="center", va="center", fontsize=8.0, color="#202020",
            fontweight="bold", linespacing=1.05)

    _round_box(ax, 3.50, 4.40, 3.00, 0.80, "white", ec="#9F8C00", lw=0.8)
    ax.text(5.00, 4.80,
            r"heads  $h_t^{CLS},\ h_t^{frame}$" "\n"
            r"Sinkhorn–Knopp / $\tau_T$",
            ha="center", va="center", fontsize=8.0, color="#202020",
            fontweight="bold", linespacing=1.05)
    _arrow(ax, 2.85, 4.80, 3.50, 4.80, color="#9F8C00", lw=0.7)

    ax.text(3.50, 3.92,
            r"input: global crops only;  output: target $p_T(\cdot)$",
            ha="center", va="center", fontsize=7.0, color="#202020")

    # Green arrow: starts from just below the 'g' of "global crop" and lands
    # in the white gap above the yellow band, with its tip aimed between
    # "Teacher branch" and "(EMA, no gradient)" in the band's title.  The
    # head must not enter the yellow frame, so y stays above the band top
    # (which is at y = 5.70).
    _arrow(ax, green_lbl_x, strip_y + slot_h + 0.15,
           1.98, 5.80, color=P["cdarkgreen"], lw=1.0, rad=0.30)

    # ---- STUDENT ----
    ax.add_patch(Rectangle((7.30, 3.70), 6.40, 2.00,
                           facecolor=P["cstudent"], alpha=0.32,
                           edgecolor="none", zorder=1))
    ax.text(7.45, 5.58, "Student branch  (trained by SGD)",
            ha="left", va="top", fontsize=7.7,
            color="#5E35B1", fontweight="bold")

    _round_box(ax, 7.55, 4.40, 2.30, 0.80, "white", ec="#5E35B1", lw=0.8)
    ax.text(8.70, 4.80,
            r"student" "\n" r"temporal ViT  $f_{\phi}$",
            ha="center", va="center", fontsize=8.0, color="#202020",
            fontweight="bold", linespacing=1.05)

    _round_box(ax, 10.50, 4.40, 3.00, 0.80, "white", ec="#5E35B1", lw=0.8)
    ax.text(12.00, 4.80,
            r"heads  $h_s^{CLS},\ h_s^{frame}$" "\n"
            r"softmax / $\tau_S$",
            ha="center", va="center", fontsize=8.0, color="#202020",
            fontweight="bold", linespacing=1.05)
    _arrow(ax, 9.85, 4.80, 10.50, 4.80, color="#5E35B1", lw=0.7)

    ax.text(10.50, 3.92,
            "input: local + global crops, with iBOT-masked frames",
            ha="center", va="center", fontsize=7.0, color="#202020")

    # Purple arrow: starts from just below the 'l' of "local crops" and lands
    # in the white gap above the purple band, with its tip aimed between
    # "Student" and "branch" in the band's title.  Head doesn't enter the
    # purple frame (band top is at y = 5.70).
    _arrow(ax, purple_lbl_x, strip_y + slot_h + 0.15,
           8.19, 5.80, color=P["cdarkorange"], lw=1.0, rad=-0.28)

    # ---- EMA arrow student → teacher (over the top of the row) ----
    ema = FancyArrowPatch((7.55, 5.18), (2.85, 5.18),
                          connectionstyle="arc3,rad=0.32",
                          arrowstyle="-|>,head_width=3.5,head_length=5.5",
                          color="#404040", lw=1.0, ls=(0, (3, 1.5)),
                          zorder=6)
    ax.add_patch(ema)
    # EMA update rule.  Use \overline{} so the bar is unambiguously over phi
    # (the prior \bar\phi rendered tightly enough to look like the bar sat
    # over m instead).  Black, non-italic.
    ax.text(5.20, 6.10,
            r"EMA  $\overline{\phi} \leftarrow m\,\overline{\phi} + (1-m)\,\phi$",
            ha="center", va="center", fontsize=7.4, color="#202020")

    # =================================================================
    # ROW 3 — Stage-1 loss block  (snug-fit chips, no dead space)
    # =================================================================
    # Stage 1 red box raised by 0.10 (closer to the teacher/student bands
    # above) and grown by 0.10 in height so the title no longer touches the
    # chip borders below.
    loss_box_y = 2.10
    loss_box_h = 1.30
    _round_box(ax, 0.30, loss_box_y, 13.40, loss_box_h,
               P["closs"], ec="#A04060", lw=0.9)
    ax.text(0.55, loss_box_y + loss_box_h - 0.12,
            "Stage 1  SSL loss",
            ha="left", va="top", fontsize=7.6, color="#7A2030",
            fontweight="bold")

    # 3 chips, snug-fit to the longest formula (iBOT).  We do this by
    # measuring each chip against a reference width and packing them with a
    # small gap; DINO's chip would otherwise have 60% horizontal dead space.
    chip_y = loss_box_y + 0.14
    chip_h = 0.74                        # was 0.93 — chips were too tall
    chip_x0 = 0.55
    chip_x1 = 13.45
    chip_gap = 0.16
    chip_w = (chip_x1 - chip_x0 - 2 * chip_gap) / 3
    chips = [
        ("#5E35B1", r"$\mathcal{L}_{\mathrm{DINO}}$",
         r"$-\mathbb{E}_{(\ell,g)}\,p_T(g)^{\top}\log p_S(\ell)$"),
        ("#8B1A1A", r"$\mathcal{L}_{\mathrm{iBOT}}$",
         r"$-\mathbb{E}_{(\ell,t)\in\mathcal{M}}\,p_T(g_{j(\ell)},t)^{\top}\log p_S(\ell,t)$"),
        ("#1F4E79", r"$0.1\cdot\mathcal{L}_{\mathrm{DKoleo}}$",
         r"$-\mathbb{E}_b\,\log\,\|\hat c_{S,b}-\hat c_{S,\mathrm{NN}(b)}\|_2$"),
    ]
    for i, (color, name, eq) in enumerate(chips):
        cx = chip_x0 + i * (chip_w + chip_gap)
        _round_box(ax, cx, chip_y, chip_w, chip_h, "white", ec=color, lw=0.8)
        # Bolder, larger text inside each chip.
        ax.text(cx + chip_w / 2, chip_y + chip_h - 0.14, name,
                ha="center", va="top", fontsize=9.0, color=color,
                fontweight="bold")
        ax.text(cx + chip_w / 2, chip_y + 0.12, eq,
                ha="center", va="bottom", fontsize=6.9, color="#202020",
                fontweight="bold")

    # Teacher/student → Stage 1 arrows: black, short, head AND tail both in
    # the white gap [loss_box_top, band_bottom] (= [3.40, 3.70]) — neither
    # end touches the loss box or the teacher/student band.
    _arrow(ax, 4.50, 3.66, 4.50, loss_box_y + loss_box_h + 0.04,
           color="#202020", lw=0.9, z=5)
    _arrow(ax, 10.50, 3.66, 10.50, loss_box_y + loss_box_h + 0.04,
           color="#202020", lw=0.9, z=5)

    # =================================================================
    # ROW 4 — Stage-2 Gram refinement (snug)
    # =================================================================
    # Reduced height 1.85→1.40, and the equation chip is no longer a giant
    # 6.8×1.35 frame around a one-line formula — it's now 4.7×0.95 wide.
    g_box_y = 0.20
    g_box_h = 1.55
    _round_box(ax, 0.30, g_box_y, 13.40, g_box_h,
               P["cgram"], ec="#7E3FAA", lw=0.9)
    ax.text(0.55, g_box_y + g_box_h - 0.18,
            "Stage 2  Gram-anchoring refinement",
            ha="left", va="top", fontsize=7.6, color="#5E35B1",
            fontweight="bold")

    # 3 enumerated steps; (i)(ii)(iii) makes the structure read as three
    # distinct items rather than three loose sentences.  Z_S / Z_G are
    # named inline so the subscripts S, G don't need a separate legend.
    ax.text(0.55, g_box_y + g_box_h - 0.52,
            r"(i)  snapshot stage-1 teacher  $\to$  frozen Gram teacher  "
            r"$f_{\overline{\phi}^*}$",
            ha="left", va="top", fontsize=7.4, color="#202020")
    ax.text(0.55, g_box_y + g_box_h - 0.82,
            r"(ii)  per-frame student / Gram-teacher tokens  "
            r"$Z_S, Z_G \in \mathbb{R}^{T\times D}$",
            ha="left", va="top", fontsize=7.4, color="#202020")
    ax.text(0.55, g_box_y + g_box_h - 1.12,
            "(iii)  Gram cross-token similarity matched against the snapshot",
            ha="left", va="top", fontsize=7.4, color="#202020")

    # right: snug equation chip — width fits the Gram formula tightly,
    # height fits the two stacked equations (both bumped up + bold + black).
    eq_w = 5.20
    eq_x = 13.45 - eq_w
    eq_h = 1.05
    eq_y = g_box_y + (g_box_h - eq_h) / 2 - 0.06
    _round_box(ax, eq_x, eq_y, eq_w, eq_h, "white", ec="#7E3FAA", lw=0.8)
    ax.text(eq_x + eq_w / 2, eq_y + eq_h - 0.30,
            r"$\mathcal{L}_{\mathrm{Gram}} = "
            r"\| Z_S Z_S^{\top} - Z_G Z_G^{\top} \|_F^{2}$",
            ha="center", va="center", fontsize=9.6, color="#202020",
            fontweight="bold")
    ax.text(eq_x + eq_w / 2, eq_y + 0.26,
            r"$\mathcal{L}_{\mathrm{S2}} = "
            r"\mathcal{L}_{\mathrm{DINO}} + \mathcal{L}_{\mathrm{iBOT}} "
            r"+ 0.1\,\mathcal{L}_{\mathrm{DKoleo}} "
            r"+ 2.0\,\mathcal{L}_{\mathrm{Gram}}$",
            ha="center", va="center", fontsize=7.4, color="#202020",
            fontweight="bold")

    fig.savefig(OUT)
    plt.close(fig)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
