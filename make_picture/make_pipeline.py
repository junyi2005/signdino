"""SignDino pipeline figure — textwidth (7.0in), 2-row layout (no title).

Typography rules applied throughout:
  * all text is black;
  * no italic;
  * the small text inside each frame uses the SAME font size as that
    frame's title (titles are bold; small text is not).

Layout:
  Row A (3 equal-height boxes, equal-length arrows between them; images
         are inset from the box borders by a small margin):
    A1 unlabeled sign video (wide landscape raw frame; V={I_t} above image)
    A2 YOLOv8n + ByteTrack (single TRACK+PRED panel, no header)
    A3 per-frame anatomical L/R/F crop strip (yellow box hugs the strip)

  Row 2 (flat):
    Left   — output band (text + 3 horizontal chips), text horizontally
             centered.
    Middle — pink temporal-SSL box.
    Right  — purple DINOv3 (top) over green per-frame embeddings (bottom),
             same x-extent.  Green's bottom edge aligns with pink and output
             bottoms.  The 'time axis' arrowhead sits entirely in the white
             gap between purple and green.

  All inter-block arrows have their head AND tail inside the white gap so
  no arrowhead touches a box.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch, FancyArrowPatch

import style

style.apply()

OUT = Path(__file__).parent / "pipeline.pdf"
HERE = Path(__file__).resolve().parent
FIGS = HERE.parent / "SignDino" / "figs"
P = style.PALETTE

RAW_FRAME   = FIGS / "pipeline_raw_frame.jpg"
YOLO_TRACK  = HERE / "yolo_track_pipeline.png"   # single TRACK+PRED panel

# Single font size for titles AND subtitle text inside boxes; titles use
# fontweight='bold', subtitles do not.
FS_TITLE    = 7.6
FS_SUBTITLE = 7.6
FS_LABEL    = 6.7   # arrow labels ("frozen per-frame CLS embedding", "time axis")
TXT_COLOR   = "#202020"


def _round(ax, x, y, w, h, fc, ec="#404040", lw=0.8, alpha=1.0, z=2):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.0,rounding_size=0.07",
        facecolor=fc, edgecolor=ec, linewidth=lw, alpha=alpha, zorder=z))


def _arrow(ax, x1, y1, x2, y2, color="#404040", lw=0.9,
           dashed=False, rad=0.0, z=4, mut=1.0,
           head_w=4, head_l=6):
    ax.add_patch(FancyArrowPatch(
        (x1, y1), (x2, y2),
        connectionstyle=f"arc3,rad={rad}",
        arrowstyle=f"-|>,head_width={head_w},head_length={head_l}",
        linestyle=("--" if dashed else "-"),
        color=color, lw=lw, zorder=z, mutation_scale=mut))


def _img(ax, path, x, y, w, h, z=3):
    if not path.exists():
        ax.add_patch(Rectangle((x, y), w, h,
                               facecolor="#EEEEEE",
                               edgecolor="#888888", linewidth=0.5, zorder=z))
        return
    img = mpimg.imread(str(path))
    ax.imshow(img, extent=(x, x + w, y, y + h),
              aspect="auto", zorder=z, interpolation="bilinear")


def main() -> None:
    fig = plt.figure(figsize=(7.0, 3.10))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 6.20)
    ax.set_axis_off()

    # =================================================================
    # ROW A — 3 equal-height boxes, equal-length arrows between them
    # =================================================================
    ay = 3.40
    ah = 2.60

    # A3 hugs the L/R/F strip — moderate L/R padding around the strip.
    n_frames = 8
    img_w = img_h = 0.62
    img_gap = 0.04
    strip_w = n_frames * img_w + (n_frames - 1) * img_gap     # ≈ 5.24
    label_pad = 0.28
    box_h_pad = 0.24
    a3w = label_pad + strip_w + box_h_pad                     # ≈ 5.76

    a1w = a2w = 3.35
    edge_pad = 0.20
    g = (14.0 - 2 * edge_pad - a1w - a2w - a3w) / 2
    a1x = edge_pad
    a2x = a1x + a1w + g
    a3x = a2x + a2w + g

    img_margin_h = 0.12
    img_margin_v = 0.07

    # ---- A1  unlabeled sign video ----
    _round(ax, a1x, ay, a1w, ah, P["cvideo"])
    ax.text(a1x + a1w / 2, ay + ah - 0.15, "unlabeled sign video",
            ha="center", va="top", fontsize=FS_TITLE, color=TXT_COLOR,
            fontweight="bold")
    fw = a1w - 2 * img_margin_h
    fh = fw / (1280 / 720)
    fx = a1x + img_margin_h
    fy = ay + img_margin_v
    _img(ax, RAW_FRAME, fx, fy, fw, fh)
    img_top = fy + fh
    title_baseline = ay + ah - 0.30
    ax.text(a1x + a1w / 2, (img_top + title_baseline) / 2 + 0.04,
            r"$V=\{I_t\}_{t=1}^{T}$",
            ha="center", va="center", fontsize=FS_SUBTITLE, color=TXT_COLOR)

    # ---- A2  YOLOv8n + ByteTrack ----
    _round(ax, a2x, ay, a2w, ah, P["ctube"])
    ax.text(a2x + a2w / 2, ay + ah - 0.15,
            "YOLOv8n + ByteTrack",
            ha="center", va="top", fontsize=FS_TITLE, color=TXT_COLOR,
            fontweight="bold")
    yw = a2w - 2 * img_margin_h
    yh = yw / (600 / 338)
    yx = a2x + img_margin_h
    yy = ay + img_margin_v
    _img(ax, YOLO_TRACK, yx, yy, yw, yh)

    # ---- A3  per-frame anatomical streams ----
    _round(ax, a3x, ay, a3w, ah, P["cteacher"])
    ax.text(a3x + a3w / 2, ay + ah - 0.15,
            r"per-frame anatomical streams  "
            r"$x^{L}_t, x^{R}_t, x^{F}_t \in \mathbb{R}^{3\times112\times112}$",
            ha="center", va="top", fontsize=6.8, color=TXT_COLOR,
            fontweight="bold")
    stream_specs = [
        ("$F$", P["cdarkorange"], "pipeline_face"),
        ("$R$", P["cdarkred"],    "pipeline_rh"),
        ("$L$", P["cdarkpurple"], "pipeline_lh"),
    ]
    stream_gap = 0.06
    strip_h = 3 * img_h + 2 * stream_gap
    bottom = ay + 0.08
    img_x0 = a3x + label_pad
    for s_i, (label, color, stem) in enumerate(stream_specs):
        y = bottom + s_i * (img_h + stream_gap)
        # L/R/F label colour is kept (semantic per-stream marker).
        ax.text(img_x0 - 0.06, y + img_h / 2, label,
                ha="right", va="center", fontsize=8.0, color=color,
                fontweight="bold")
        for k in range(n_frames):
            _img(ax, FIGS / f"{stem}_{k+1}.jpg",
                 img_x0 + k * (img_w + img_gap), y, img_w, img_h)

    # Row-A horizontal arrows
    gap_pad = 0.10
    _arrow(ax, a1x + a1w + gap_pad, ay + ah / 2,
                 a2x - gap_pad,       ay + ah / 2, lw=1.0)
    _arrow(ax, a2x + a2w + gap_pad, ay + ah / 2,
                 a3x - gap_pad,       ay + ah / 2, lw=1.0)

    # =================================================================
    # ROW 2 — flat
    # =================================================================
    bh = 2.50
    snake_gap = 0.40
    row2_top = ay - snake_gap                  # = 3.00
    by = row2_top - bh                         # = 0.50

    # Layout: out↔pink gap 0.45, pink↔right gap 0.70 (room for diagonal).
    out_x, out_w = 0.20, 4.30
    pink_x, pink_w = 4.95, 3.75
    right_x, right_w = 9.40, 4.40

    green_y, green_h = by, 0.55
    ta_gap = 0.30
    purple_y = green_y + green_h + ta_gap
    purple_h = row2_top - purple_y

    # =================================================================
    # SNAKE  A3 → purple top
    # =================================================================
    snake_x = right_x + right_w / 2
    _arrow(ax, snake_x, ay - 0.08,
                  snake_x, row2_top + 0.08,
                  color="#404040", lw=1.0)
    ax.text(snake_x - 0.18, (ay + row2_top) / 2,
            "frozen per-frame CLS embedding",
            ha="right", va="center", fontsize=FS_LABEL, color=TXT_COLOR)

    # =================================================================
    # PURPLE  frozen DINOv3
    # =================================================================
    _round(ax, right_x, purple_y, right_w, purple_h, P["cstudent"])
    cx_p = right_x + right_w / 2
    # 4 lines, all FS_SUBTITLE = title size; first line bold.
    line_y = lambda i: purple_y + purple_h - 0.16 - i * 0.34
    ax.text(cx_p, line_y(0), "frozen DINOv3  ViT-B/16",
            ha="center", va="top", fontsize=FS_TITLE, color=TXT_COLOR,
            fontweight="bold")
    ax.text(cx_p, line_y(1),
            r"$e^{(r)}_t = \mathrm{CLS}(x^{(r)}_t) \in \mathbb{R}^{768}$",
            ha="center", va="top", fontsize=FS_SUBTITLE, color=TXT_COLOR)
    ax.text(cx_p, line_y(2),
            r"shared over $r \in \{L, R, F\}$",
            ha="center", va="top", fontsize=FS_SUBTITLE, color=TXT_COLOR)
    ax.text(cx_p, line_y(3),
            "no SSL gradient through DINOv3",
            ha="center", va="top", fontsize=FS_SUBTITLE, color=TXT_COLOR)

    # =================================================================
    # GREEN  per-frame embeddings
    # =================================================================
    _round(ax, right_x, green_y, right_w, green_h, P["cproto"])
    ax.text(right_x + right_w / 2, green_y + green_h / 2,
            r"per-frame embeddings  "
            r"$E^{(r)} \in \mathbb{R}^{T \times 768}$",
            ha="center", va="center", fontsize=FS_TITLE, color=TXT_COLOR,
            fontweight="bold")

    # time-axis arrowhead inside the white gap [green_top, purple_bottom]
    ta_x = right_x + right_w / 2
    ta_top    = purple_y - 0.05
    ta_bottom = green_y + green_h + 0.05
    _arrow(ax, ta_x, ta_top, ta_x, ta_bottom,
           color="#404040", lw=0.9,
           mut=0.7, head_w=4, head_l=5)
    ax.text(ta_x + 0.18, (purple_y + green_y + green_h) / 2,
            "time axis", ha="left", va="center",
            fontsize=FS_LABEL, color=TXT_COLOR)

    # =================================================================
    # PINK  temporal SSL
    # =================================================================
    _round(ax, pink_x, by, pink_w, bh, P["closs"])
    ax.text(pink_x + pink_w / 2, by + bh - 0.18,
            "temporal SSL  (per stream)",
            ha="center", va="top", fontsize=FS_TITLE, color=TXT_COLOR,
            fontweight="bold")
    tch_w, tch_h = 1.45, 0.60
    tch_y = by + bh - 1.30
    _round(ax, pink_x + 0.20, tch_y, tch_w, tch_h,
           P["cteacher"], ec="#9F8C00", lw=0.7)
    ax.text(pink_x + 0.20 + tch_w / 2, tch_y + tch_h / 2,
            "EMA teacher", ha="center", va="center",
            fontsize=FS_SUBTITLE, color=TXT_COLOR)
    _round(ax, pink_x + pink_w - 0.20 - tch_w, tch_y, tch_w, tch_h,
           P["cstudent"], ec="#5E35B1", lw=0.7)
    ax.text(pink_x + pink_w - 0.20 - tch_w / 2, tch_y + tch_h / 2,
            "student ViT", ha="center", va="center",
            fontsize=FS_SUBTITLE, color=TXT_COLOR)
    ax.add_patch(FancyArrowPatch(
        (pink_x + pink_w - 0.20 - tch_w, tch_y + tch_h / 2),
        (pink_x + 0.20 + tch_w, tch_y + tch_h / 2),
        connectionstyle="arc3,rad=0",
        arrowstyle="-|>,head_width=3,head_length=4.5",
        ls=(0, (3, 1.5)), color="#404040", lw=0.7, zorder=5))
    ax.text(pink_x + pink_w / 2, by + 0.74,
            r"DINO + iBOT + DKoleo + Gram",
            ha="center", va="center", fontsize=FS_SUBTITLE, color=TXT_COLOR)
    ax.text(pink_x + pink_w / 2, by + 0.34,
            "details: student/teacher diagram",
            ha="center", va="center", fontsize=FS_SUBTITLE, color=TXT_COLOR)

    # =================================================================
    # GREEN  → PINK  (diagonal up-left; head AND tail well inside the
    # white gap between pink's right edge and green's left edge)
    # =================================================================
    _arrow(ax,
           right_x         - 0.16, green_y + green_h / 2,
           pink_x + pink_w + 0.16, by + bh / 2 - 0.25,
           color="#404040", lw=1.0, rad=0.18)

    # =================================================================
    # OUTPUT band
    # =================================================================
    _round(ax, out_x, by, out_w, bh, P["cgram"])
    cx_out = out_x + out_w / 2
    ax.text(cx_out, by + bh - 0.16,
            "output:  per-stream sign encoders",
            ha="center", va="top", fontsize=FS_TITLE, color=TXT_COLOR,
            fontweight="bold")
    ax.text(cx_out, by + bh - 0.50,
            r"$f_\phi^{(L)},\ f_\phi^{(R)},\ f_\phi^{(F)} \;\to\; "
            r"z^{(r)}_t \in \mathbb{R}^{384}$",
            ha="center", va="top", fontsize=FS_SUBTITLE, color=TXT_COLOR)
    ax.text(cx_out, by + bh - 0.86,
            "three streams independent in SSL",
            ha="center", va="top", fontsize=FS_SUBTITLE, color=TXT_COLOR)
    ax.text(cx_out, by + bh - 1.16,
            "fusion only at the downstream task head",
            ha="center", va="top", fontsize=FS_SUBTITLE, color=TXT_COLOR)

    # 3 horizontal chips, centered as a block
    chip_h_box = 0.58
    chip_y = by + 0.22
    chip_gap = 0.14
    chip_count = 3
    chip_w = 1.20
    chips_total = chip_count * chip_w + (chip_count - 1) * chip_gap
    chip_x0 = out_x + (out_w - chips_total) / 2
    for i, name in enumerate(["ByT5 translation", "ISLR + LoRA",
                              "fingerspelling"]):
        x = chip_x0 + i * (chip_w + chip_gap)
        _round(ax, x, chip_y, chip_w, chip_h_box, "white",
               ec="#404040", lw=0.7)
        ax.text(x + chip_w / 2, chip_y + chip_h_box / 2, name,
                ha="center", va="center", fontsize=6.4, color=TXT_COLOR)

    # PINK → OUTPUT  (head and tail inside the white gap)
    _arrow(ax, pink_x         - 0.12, by + bh / 2,
                 out_x + out_w + 0.12, by + bh / 2,
                 color="#404040", lw=1.0)

    fig.savefig(OUT)
    plt.close(fig)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
