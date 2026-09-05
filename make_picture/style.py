"""Shared style for SignDino paper figures.

Sets the matplotlib rcParams so every PDF in this directory has a coherent
academic look: STIX serif font, thin axes, embedded TrueType fonts (Type 42
so EMNLP/ACL submission preflight passes), and a shared color palette that
matches the LaTeX `\\definecolor` block in main.tex.
"""
from __future__ import annotations

import matplotlib as mpl


PALETTE = {
    "cvideo":      "#DCEBFF",
    "cregion":     "#FFE6CC",
    "ctube":       "#E5F7E5",
    "cstudent":    "#E7DFFF",
    "cteacher":    "#FFF2B2",
    "closs":       "#FFD6DE",
    "cproto":      "#DDF7F4",
    "cgram":       "#F3D9FA",
    "cdarkblue":   "#1F4E79",
    "cdarkgreen":  "#2E7D32",
    "cdarkred":    "#8B1A1A",
    "cdarkpurple": "#5E35B1",
    "cdarkorange": "#C2580E",
    "cgray":       "#F2F2F2",
    "cinkblack":   "#202020",
}


def apply():
    mpl.rcParams.update({
        "font.family":       "serif",
        "font.serif":        ["STIXGeneral", "DejaVu Serif", "Times New Roman"],
        "mathtext.fontset":  "stix",
        "pdf.fonttype":       42,
        "ps.fonttype":        42,
        "axes.linewidth":     0.6,
        "axes.edgecolor":    "#404040",
        "axes.labelcolor":   "#202020",
        "xtick.color":       "#202020",
        "ytick.color":       "#202020",
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size":  2.5,
        "ytick.major.size":  2.5,
        "legend.frameon":    False,
        "legend.fontsize":   7.5,
        "axes.titlesize":    9,
        "axes.labelsize":    8,
        "xtick.labelsize":   7.5,
        "ytick.labelsize":   7.5,
        "figure.dpi":        150,
        "savefig.dpi":       300,
        "savefig.bbox":      "tight",
        "savefig.pad_inches": 0.02,
    })
