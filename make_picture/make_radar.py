"""Radar chart: per-benchmark coverage of SignDino vs. Previous SOTA.

8-axis radar styled after SHuBERT (Gueuwou et al. 2025) Fig. 1: one inner
polygon for the strongest previously-published task-specific number per
axis, one outer polygon for our model.

This revision (per user request) moves the long two-line benchmark labels
out of the radar — spoke tips now carry just a numeric chip (1..8), and a
2-column key strip below the radar maps each number to its benchmark +
metric. Per-vertex numeric values still sit next to each vertex.

Axes (clockwise from top, matching SHuBERT's reading order)
    1. SLT-How2Sign BLEU      (Tab. 3 column)
    2. SLT-How2Sign BLEURT    (Tab. 3 column)
    3. SLT-OpenASL BLEU       (Tab. 3 column)
    4. SLT-FLEURS BLEU        (Tab. 3 column)
    5. ISLR-ASL_Citizen R@1   (Tab. 4 column)
    6. ISLR-Sem-Lex R@1       (Tab. 4 column)
    7. ISLR-WLASL P-C         (Tab. 4 column)
    8. FS-StemWiki mIoU       (Tab. 4 column)
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch

import style

style.apply()

OUT = Path(__file__).parent / "radar.pdf"

# axis_label, metric_label, prev_sota_value, axis_cap
AXES = [
    ("SLT-How2Sign",     "BLEU",   16.2,  22.0),
    ("SLT-How2Sign",     "BLEURT", 49.9,  60.0),
    ("SLT-OpenASL",      "BLEU",   23.2,  30.0),
    ("SLT-FLEURS",       "BLEU",    4.7,   8.0),
    ("ISLR-ASL Citizen", "R@1",    65.0,  90.0),
    ("ISLR-Sem-Lex",     "R@1",    54.0,  80.0),
    ("ISLR-WLASL",       "P-C",    61.32, 80.0),
    ("FS-StemWiki",      "mIoU",   40.0,  55.0),
]

SIGNDINO_OVERRIDE: list[float] | None = [17.9, 51.3, 24.5, 5.3, 70.4, 59.3, 65.92, 43.0]
P = style.PALETTE

SD_COLOR   = "#3D7AB8"
SOTA_COLOR = "#E69138"


def _vals() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    prev = np.array([row[2] for row in AXES])
    cap  = np.array([row[3] for row in AXES])
    sd = (np.array(SIGNDINO_OVERRIDE) if SIGNDINO_OVERRIDE
          else prev * 1.10)
    return prev / cap, sd / cap, prev, sd


def main() -> None:
    n = len(AXES)
    theta = np.linspace(0, 2 * np.pi, n, endpoint=False)
    theta_closed = np.concatenate([theta, [theta[0]]])

    prev_norm, sd_norm, prev_raw, sd_raw = _vals()

    # Figure: radar on top, key strip on bottom, then mini legend.
    fig = plt.figure(figsize=(3.55, 4.10))

    # Radar axes
    ax = fig.add_axes([0.06, 0.32, 0.88, 0.64], projection="polar")
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    ax.set_ylim(0, 1.05)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels([])
    ax.grid(which="major", color="#C5C5C5", linewidth=0.45, alpha=0.85)
    theta_circle = np.linspace(0, 2 * np.pi, 256)
    ax.plot(theta_circle, np.ones_like(theta_circle), color="#404040",
            lw=0.6, zorder=10)
    ax.spines["polar"].set_visible(False)

    # ---- SignDino outer polygon (blue) ----
    # User-requested visual override: stretch every blue vertex out to the
    # outermost black ring (r=1.0), regardless of the underlying value.
    # The orange (Previous-SOTA) polygon is NOT touched.  Per-vertex numeric
    # values still print the real SignDino numbers, just at the ring radius.
    sd_polygon = np.ones_like(sd_norm)
    sd_closed = np.concatenate([sd_polygon, [sd_polygon[0]]])
    ax.fill(theta_closed, sd_closed, color=SD_COLOR, alpha=0.30, lw=0, zorder=4)
    ax.plot(theta_closed, sd_closed, color=SD_COLOR, lw=1.4, zorder=6,
            marker="o", markersize=3.0, markeredgewidth=0)

    # ---- Previous-SOTA inner polygon (orange) ----
    prev_closed = np.concatenate([prev_norm, [prev_norm[0]]])
    ax.fill(theta_closed, prev_closed, color=SOTA_COLOR, alpha=0.30, lw=0, zorder=5)
    ax.plot(theta_closed, prev_closed, color=SOTA_COLOR, lw=1.2, zorder=7,
            marker="o", markersize=2.6, markeredgewidth=0)

    # ---- numeric chip at each spoke tip (1..8) ----
    ax.set_xticks(theta)
    ax.set_xticklabels([])
    for i, ang in enumerate(theta, start=1):
        ax.text(ang, 1.18, f"{i}", ha="center", va="center",
                fontsize=7.6, color="#202020", fontweight="bold",
                bbox=dict(boxstyle="circle,pad=0.30",
                          facecolor="white", edgecolor="#202020",
                          linewidth=0.7),
                zorder=12)

    # ---- per-vertex numeric values ----
    # Blue labels are now placed just outside the expanded polygon (which
    # sits on the outer ring), in the white space between the ring and the
    # numeric chip at r=1.18.  Orange labels stay where they were
    # (positioned relative to the real SOTA polygon).
    for ang, (_n, _m, _pv, cap), sdv, pv in zip(theta, AXES, sd_raw, prev_raw):
        r_pv  = pv  / cap
        ax.text(ang, 1.06,
                f"{sdv:.1f}",
                ha="center", va="center", fontsize=6.2, color=SD_COLOR,
                bbox=dict(facecolor="white", edgecolor="none",
                          pad=0.4, alpha=0.85),
                zorder=11)
        ax.text(ang, max(r_pv - 0.10, 0.05),
                f"{pv:.1f}",
                ha="center", va="center", fontsize=5.9,
                color=SOTA_COLOR, zorder=11)

    # =================================================================
    # KEY STRIP — 2-column layout below the radar, mapping 1..8 → name
    # =================================================================
    # Coordinates in figure fraction.
    key_ax = fig.add_axes([0.02, 0.08, 0.96, 0.18])
    key_ax.set_xlim(0, 1)
    key_ax.set_ylim(0, 1)
    key_ax.set_axis_off()

    # 4 rows × 2 columns (items 1..4 in left column, 5..8 in right)
    rows = 4
    col_x = [0.04, 0.52]            # left column x, right column x
    row_y = [0.88, 0.62, 0.36, 0.10]  # top → bottom (4 rows)
    for idx, (name, metric, _pv, _cap) in enumerate(AXES, start=1):
        col = (idx - 1) // rows
        row = (idx - 1) %  rows
        x = col_x[col]
        y = row_y[row]
        # numeric chip
        key_ax.text(x + 0.014, y, f"{idx}",
                    ha="center", va="center",
                    fontsize=6.6, color="#202020", fontweight="bold",
                    bbox=dict(boxstyle="circle,pad=0.28",
                              facecolor="white", edgecolor="#202020",
                              linewidth=0.6))
        # name + metric on one line
        key_ax.text(x + 0.05, y, f"{name} · {metric}",
                    ha="left", va="center", fontsize=6.8,
                    color="#202020")

    # =================================================================
    # LEGEND STRIP — colours / models, at the very bottom
    # =================================================================
    leg_ax = fig.add_axes([0.02, 0.005, 0.96, 0.065])
    leg_ax.set_xlim(0, 1)
    leg_ax.set_ylim(0, 1)
    leg_ax.set_axis_off()

    # SignDino swatch + label
    leg_ax.add_patch(FancyBboxPatch(
        (0.05, 0.30), 0.04, 0.42,
        boxstyle="round,pad=0.0,rounding_size=0.02",
        facecolor=SD_COLOR, edgecolor=SD_COLOR, alpha=0.55, linewidth=0))
    leg_ax.text(0.105, 0.51, "SignDino (Ours)",
                ha="left", va="center", fontsize=6.8, color="#202020")

    # Prev SOTA swatch + label
    leg_ax.add_patch(FancyBboxPatch(
        (0.51, 0.30), 0.04, 0.42,
        boxstyle="round,pad=0.0,rounding_size=0.02",
        facecolor=SOTA_COLOR, edgecolor=SOTA_COLOR, alpha=0.55, linewidth=0))
    leg_ax.text(0.565, 0.51, "Previous SOTA  (task-specific)",
                ha="left", va="center", fontsize=6.8, color="#202020")

    fig.savefig(OUT)
    plt.close(fig)
    print(f"wrote {OUT}")
    print(f"placeholder SignDino values:")
    for (name, metric, pv, _cap), sdv in zip(AXES, sd_raw):
        print(f"  {name:20s} {metric:8s}  prev_sota={pv:>6.2f}  sd≈{sdv:>6.2f}")


if __name__ == "__main__":
    main()
