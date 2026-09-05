"""Intro insight figure: spatial-axis DINOv3 → temporal-axis SignDino.

Columnwidth (~3.4in) vertical layout. The figure's single point: the
DINOv3 student–teacher game is defined over crops of an image (spatial
axes x, y); SignDino is the literal axis-transpose of that game, with
crops over time on a per-frame embedding row from one anatomical stream.

Design choices vs. the previous draft:
  * Panel backgrounds are now neutral (very pale grey) so the coloured
    title bar + the visual content carry the panel identity, instead of
    a saturated tint that washes the inner patch grid out.
  * The 6×6 image grid uses a *filled* light-blue patch colour against
    the neutral background, so it's actually visible.
  * The single-row temporal strip uses orange-tinted slots against the
    same neutral background; same trick.
  * The axis arrows are pushed away from the bottom captions to remove
    overlap (the old draft had "spatial axis x" land on top of
    "student/teacher see crops of one image").
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch, FancyArrowPatch

import style

style.apply()

OUT = Path(__file__).parent / "insight.pdf"
P = style.PALETTE

# Neutral panel background — lets the coloured cells inside pop
PANEL_BG = "#F8F8FB"
PATCH_FACE  = "#C8DCF2"       # light blue
PATCH_EDGE  = "#5A7CA8"
STRIP_FACE  = "#FFDDBC"       # light orange
STRIP_EDGE  = "#A05C18"


def _round_box(ax, x, y, w, h, fc, ec="#404040", lw=0.6):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.0,rounding_size=0.05",
        facecolor=fc, edgecolor=ec, linewidth=lw, zorder=2))


def main() -> None:
    fig = plt.figure(figsize=(3.45, 4.50))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 7.0)
    ax.set_ylim(0, 9.0)
    ax.set_axis_off()

    PANEL_W = 6.4
    PANEL_X = 0.30

    # =================================================================
    # PANEL 1 — DINOv3 spatial crops
    # =================================================================
    p1_y, p1_h = 5.10, 3.80
    _round_box(ax, PANEL_X, p1_y, PANEL_W, p1_h, PANEL_BG, ec="#B8C0CC", lw=0.7)

    # title strip (filled, blue)
    title_h = 0.55
    _round_box(ax, PANEL_X, p1_y + p1_h - title_h, PANEL_W, title_h,
               P["cvideo"], ec="#5A7CA8", lw=0.7)
    ax.text(PANEL_X + PANEL_W / 2, p1_y + p1_h - title_h / 2,
            "DINOv3  —  spatial crops in one image",
            ha="center", va="center", fontsize=8.4,
            color=P["cdarkblue"], fontweight="bold")

    # 6×6 patch grid
    n = 6
    cell = 0.30
    grid_w = n * cell
    # leave room on the right for the global/local labels
    grid_x0 = PANEL_X + 0.55
    grid_y0 = p1_y + 0.95
    for i in range(n):
        for j in range(n):
            ax.add_patch(Rectangle(
                (grid_x0 + j * cell, grid_y0 + i * cell),
                cell - 0.04, cell - 0.04,
                facecolor=PATCH_FACE, edgecolor=PATCH_EDGE, linewidth=0.45,
                zorder=3))

    # global spatial crop (outer 5×5)
    gx, gy, gw, gh = grid_x0, grid_y0 + 0.5 * cell, 5.0 * cell, 5.0 * cell
    ax.add_patch(Rectangle((gx, gy), gw, gh, facecolor="none",
                           edgecolor=P["cdarkgreen"], linewidth=1.7,
                           zorder=4))

    # 3 local spatial crops at (inside / edge / outside) relative to the global box.
    # Aspect ratios respect DINOv2/v3's RandomResizedCrop range of [3/4, 4/3].
    # Each tuple is (lx_cell, ly_cell, lw_cell, lh_cell) in cell units.
    spatial_locals = [
        (1.0, 1.7, 2.0, 2.0),    # fully inside global  (ratio 1:1)
        (4.0, 0.9, 2.0, 2.0),    # straddles right edge of global  (ratio 1:1)
        (4.9, 3.2, 1.5, 2.0),    # fully outside global (right column)  (ratio 3:4, at DINO's lower bound)
    ]
    for lf in spatial_locals:
        ax.add_patch(Rectangle(
            (grid_x0 + lf[0] * cell, grid_y0 + lf[1] * cell),
            lf[2] * cell, lf[3] * cell,
            facecolor="none", edgecolor=P["cdarkorange"], linewidth=1.7,
            zorder=4))

    # ---- iBOT mask: 2 patches inside the "inside" local crop ----
    # Real DINOv3 applies iBOT-style token masking to a random fraction of
    # patches inside each local crop.  We hatch two adjacent cells fully
    # contained in the inside local crop, then label them above the grid.
    mask_patches = [(1, 2), (2, 2)]                # (j, i) cell indices
    for jm, im in mask_patches:
        ax.add_patch(Rectangle(
            (grid_x0 + jm * cell, grid_y0 + im * cell),
            cell - 0.04, cell - 0.04,
            facecolor="#7A7A7A", edgecolor="#3A3A3A",
            linewidth=0.4, hatch="////", alpha=0.70, zorder=5))
    mask_centre_x = grid_x0 + 2.0 * cell           # midline between the two masked cells
    mask_top_y    = grid_y0 + 3 * cell             # top of row i=2
    grid_top_y    = grid_y0 + n * cell             # top of full grid
    # Thin connector + label parked just above the grid, below the title bar.
    ax.annotate("", xy=(mask_centre_x, mask_top_y + 0.02),
                xytext=(mask_centre_x, grid_top_y + 0.06),
                arrowprops=dict(arrowstyle="-", lw=0.55, color="#404040"),
                zorder=6)
    ax.text(mask_centre_x, grid_top_y + 0.08, "iBOT [MASK]",
            ha="center", va="bottom", fontsize=6.3, color="#404040",
            fontweight="bold", zorder=7)

    # labels on the right (real DINOv3 ViT-B/16 patch counts; grid drawn coarser for legibility)
    lbl_x = grid_x0 + grid_w + 0.20
    ax.text(lbl_x, gy + gh - 0.10,
            r"global crop" + "\n" + r"$T_g{=}14{\times}14$ patches  ($N_g{=}2$)",
            ha="left", va="top", fontsize=6.8,
            color=P["cdarkgreen"], fontweight="bold", linespacing=1.15)
    ax.text(lbl_x, gy + 0.30,
            r"local crop" + "\n" + r"$T_l{=}6{\times}6$ patches  ($N_l{=}8$)",
            ha="left", va="bottom", fontsize=6.8,
            color=P["cdarkorange"], fontweight="bold", linespacing=1.15)

    # axis arrows — placed close to the grid, NOT in the caption zone
    arr_off = 0.16
    ax.annotate("", xy=(grid_x0 + grid_w + 0.04, grid_y0 - arr_off),
                xytext=(grid_x0 - 0.04, grid_y0 - arr_off),
                arrowprops=dict(arrowstyle="->", lw=0.75, color="#404040"),
                zorder=5)
    ax.text(grid_x0 + grid_w / 2, grid_y0 - arr_off - 0.13,
            r"spatial axis  $x$",
            ha="center", va="top", fontsize=6.4, color="#404040")

    ax.annotate("", xy=(grid_x0 - arr_off, grid_y0 + grid_w + 0.04),
                xytext=(grid_x0 - arr_off, grid_y0 - 0.04),
                arrowprops=dict(arrowstyle="->", lw=0.75, color="#404040"),
                zorder=5)
    ax.text(grid_x0 - arr_off - 0.10, grid_y0 + grid_w / 2,
            r"$y$",
            ha="right", va="center", fontsize=6.4, color="#404040")

    # panel-bottom caption
    ax.text(PANEL_X + PANEL_W / 2, p1_y + 0.30,
            r"student / teacher see crops of $\mathit{one\ image}$",
            ha="center", va="center", fontsize=6.9, color="#404040")

    # =================================================================
    # HINGE — axis transpose
    # =================================================================
    hinge_y = 4.70
    arrow = FancyArrowPatch(
        (3.50, hinge_y + 0.22), (3.50, hinge_y - 0.22),
        connectionstyle="arc3,rad=0",
        arrowstyle="-|>,head_width=5,head_length=7",
        lw=1.4, color="#404040", zorder=6,
    )
    ax.add_patch(arrow)
    ax.text(3.50 + 0.32, hinge_y + 0.05, "axis transpose",
            ha="left", va="center", fontsize=7.4,
            color="#202020", fontweight="bold")
    ax.text(3.50 + 0.32, hinge_y - 0.18,
            r"$(x, y) \;\to\; t$",
            ha="left", va="center", fontsize=7.0,
            color="#5A5A5A", fontstyle="italic")

    # =================================================================
    # PANEL 2 — SignDino temporal crops
    # =================================================================
    p2_y, p2_h = 0.20, 3.95
    _round_box(ax, PANEL_X, p2_y, PANEL_W, p2_h, PANEL_BG, ec="#C8AAAA", lw=0.7)

    title_h2 = 0.55
    _round_box(ax, PANEL_X, p2_y + p2_h - title_h2, PANEL_W, title_h2,
               P["closs"], ec="#A04060", lw=0.7)
    ax.text(PANEL_X + PANEL_W / 2, p2_y + p2_h - title_h2 / 2,
            "SignDino  —  temporal crops on a frame stream",
            ha="center", va="center", fontsize=8.4,
            color=P["cdarkred"], fontweight="bold")

    # Single-row temporal strip
    n_frames = 22
    slot_w = 0.24
    strip_w = n_frames * slot_w
    strip_x = PANEL_X + (PANEL_W - strip_w) / 2 + 0.18  # nudge right, leave room for $L_t$ label
    strip_y = p2_y + 1.95
    strip_h = 0.55
    for k in range(n_frames):
        ax.add_patch(Rectangle(
            (strip_x + k * slot_w, strip_y), slot_w - 0.025, strip_h,
            facecolor=STRIP_FACE, edgecolor=STRIP_EDGE, linewidth=0.45,
            zorder=3))

    # ---- global temporal crop ----
    g_k0, g_w = 1, 13
    g_x = strip_x + g_k0 * slot_w - 0.04
    g_y = strip_y - 0.09
    g_h = strip_h + 0.18
    ax.add_patch(Rectangle((g_x, g_y), g_w * slot_w, g_h,
                           facecolor="none",
                           edgecolor=P["cdarkgreen"], linewidth=1.7,
                           zorder=4))
    ax.text(g_x + g_w * slot_w / 2, g_y + g_h + 0.12,
            r"global crop  $T_g\!\in\![64,96]$  ($N_g{=}2$)",
            ha="center", va="bottom", fontsize=6.8,
            color=P["cdarkgreen"], fontweight="bold")

    # ---- 3 local temporal crops at (inside / edge / outside) of the global window ----
    # Global covers frames 1..13; each local is 3 frames wide.
    # Outside crop is positioned 2 frames after the edge crop so the two purple
    # boxes are visually separated (one empty frame slot in between).
    temporal_locals = [(3, 3), (12, 3), (16, 3)]   # (l_k0, l_w): inside, straddles end, outside
    inside_l_k0, inside_l_w = temporal_locals[0]
    for l_k0, l_w in temporal_locals:
        l_x = strip_x + l_k0 * slot_w - 0.04
        l_y = strip_y - 0.05
        l_h = strip_h + 0.10
        ax.add_patch(Rectangle((l_x, l_y), l_w * slot_w, l_h,
                               facecolor="none",
                               edgecolor=P["cdarkorange"], linewidth=1.7,
                               zorder=4))
    # Label below the leftmost (inside) local; "× 8" reminds the reader of N_l.
    lbl_l_x = strip_x + inside_l_k0 * slot_w - 0.04 + (inside_l_w * slot_w) / 2
    lbl_l_y = strip_y - 0.05 - 0.12
    ax.text(lbl_l_x, lbl_l_y,
            r"local crops  $T_l\!\in\![10,32]$  ($N_l{=}8$)",
            ha="center", va="top", fontsize=6.8,
            color=P["cdarkorange"], fontweight="bold")

    # ---- iBOT mask hatching on two frames inside the "inside" local crop ----
    for k_mask in (inside_l_k0 + 1, inside_l_k0 + 2):
        mx = strip_x + k_mask * slot_w
        ax.add_patch(Rectangle(
            (mx, strip_y), slot_w - 0.025, strip_h,
            facecolor="#7A7A7A", edgecolor="#3A3A3A",
            linewidth=0.4, hatch="////", alpha=0.70, zorder=5))
    # mask call-out
    ib_x = strip_x + (inside_l_k0 + 1.5) * slot_w
    ib_top = strip_y + strip_h
    ax.annotate("", xy=(ib_x, ib_top + 0.03),
                xytext=(ib_x, ib_top + 0.55),
                arrowprops=dict(arrowstyle="-", lw=0.55, color="#404040"),
                zorder=5)
    ax.text(ib_x, ib_top + 0.62,
            "iBOT [MASK]",
            ha="center", va="bottom", fontsize=6.3, color="#404040")

    # ---- time-axis arrow, padded above the caption ----
    t_y = strip_y - 0.60
    ax.annotate("", xy=(strip_x + strip_w + 0.04, t_y),
                xytext=(strip_x - 0.04, t_y),
                arrowprops=dict(arrowstyle="->", lw=0.75, color="#404040"),
                zorder=5)
    ax.text(strip_x + strip_w / 2, t_y - 0.16,
            r"time axis  $t \to$",
            ha="center", va="top", fontsize=6.4, color="#404040")

    # panel-bottom caption
    ax.text(PANEL_X + PANEL_W / 2, p2_y + 0.32,
            r"student / teacher see crops of "
            r"$\mathit{one\ anatomical\ stream}$",
            ha="center", va="center", fontsize=6.9, color="#404040")

    fig.savefig(OUT)
    plt.close(fig)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
