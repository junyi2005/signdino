"""Compose a publication-quality 'YOLO misses hands vs YOLO+ByteTrack catches
them' gallery from the existing per-case PNGs.

We hand-pick 8 of the 13 'both-hand-recovered' dramatic cases (the ones where
YOLO sees ONLY the face and the tracker recovers BOTH hands via Kalman pred).
The result is 1 composite figure: 4 columns × 2 rows, each cell showing the
official YOLO | TRACK+PRED pair side-by-side with a one-line caption.

Output:  make_picture/yolo_recovery_gallery.png  (also .pdf if --pdf is on)
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]

# (manifest_dir, case_id, dataset_tag) — hand-picked from the 13 both-hand cases
PICKS = [
    ("runs/yolov8_hf/openasl_pairs",       94, "OpenASL"),
    ("runs/yolov8_hf/openasl_pairs",      171, "OpenASL"),
    ("runs/yolov8_hf/openasl_pairs",      125, "OpenASL"),
    ("runs/yolov8_hf/openasl_pairs",      212, "OpenASL"),
    ("runs/yolov8_hf/openasl_pairs",      200, "OpenASL"),
    ("runs/yolov8_hf/openasl_pairs",      260, "OpenASL"),
    ("runs/yolov8_hf/h2s_pairs_more",      74, "How2Sign"),
    ("runs/yolov8_hf/openasl_pairs",      130, "OpenASL"),
]


def _find_png(manifest_dir: Path, case_id: int) -> Path:
    matches = sorted(manifest_dir.glob(f"case_{case_id:03d}_*.png"))
    if not matches:
        raise FileNotFoundError(f"no PNG for case {case_id} in {manifest_dir}")
    return matches[0]


def _strip_header(img: np.ndarray, header_h: int = 92) -> np.ndarray:
    """Cells were rendered with a 92-px header strip carrying the case caption +
    pred summary. We drop that strip in the gallery; the gallery uses its own
    short caption underneath each cell."""
    return img[header_h:]


def _draw_cell_caption(panel: np.ndarray, tag: str, lost: tuple[int, int]) -> np.ndarray:
    """Add a thin caption bar under the YOLO|TRACK pair."""
    h, w = panel.shape[:2]
    bar_h = 40
    cap = np.full((bar_h, w, 3), 250, dtype=np.uint8)
    # ASCII only: cv2.putText does not draw the unicode middot.
    text = f"{tag}    hand#1 +{lost[0]}f,  hand#2 +{lost[1]}f"
    cv2.putText(cap, text, (12, 27),
                cv2.FONT_HERSHEY_SIMPLEX, 0.62, (15, 15, 15), 1, cv2.LINE_AA)
    out = np.vstack([panel, cap])
    cv2.rectangle(out, (0, 0), (w - 1, out.shape[0] - 1), (215, 215, 215), 1)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-stem", default="make_picture/yolo_recovery_gallery")
    ap.add_argument("--pdf", action="store_true")
    ap.add_argument("--cell-width", type=int, default=560)
    args = ap.parse_args()

    import json
    cells: list[np.ndarray] = []
    for manifest_dir, case_id, tag in PICKS:
        d = ROOT / manifest_dir
        # parse pred summary from manifest
        mani = json.loads((d / "manifest.json").read_text())
        entry = next(e for e in mani if e["case_id"] == case_id)
        lost = [0, 0]
        for p in (entry.get("pred_summary") or []):
            # 'hand#1 +4f'
            tok, age = p.split()
            slot = int(tok.split("#")[1]) - 1
            lost[slot] = int(age.lstrip("+").rstrip("f"))

        png = _find_png(d, case_id)
        full = cv2.imread(str(png))
        if full is None:
            raise FileNotFoundError(png)
        panel = _strip_header(full)
        # resize to target cell width while keeping aspect
        scale = args.cell_width / panel.shape[1]
        panel = cv2.resize(panel,
                            (args.cell_width,
                             int(round(panel.shape[0] * scale))),
                            interpolation=cv2.INTER_AREA)
        cell = _draw_cell_caption(panel, tag, tuple(lost))
        cells.append(cell)

    # pad cells to the same height (some H2S panels are wider-aspect)
    maxh = max(c.shape[0] for c in cells)
    cells = [cv2.copyMakeBorder(c, 0, maxh - c.shape[0], 0, 0,
                                  cv2.BORDER_CONSTANT, value=(255, 255, 255))
             for c in cells]

    cols = 4
    rows: list[np.ndarray] = []
    pad = 12
    for r in range((len(cells) + cols - 1) // cols):
        row = cells[r*cols:(r+1)*cols]
        while len(row) < cols:
            row.append(np.full_like(cells[0], 255))
        sep = np.full((row[0].shape[0], pad, 3), 255, dtype=np.uint8)
        line = []
        for i, c in enumerate(row):
            if i:
                line.append(sep)
            line.append(c)
        rows.append(np.hstack(line))

    sep_v = np.full((pad, rows[0].shape[1], 3), 255, dtype=np.uint8)
    grid_lines: list[np.ndarray] = []
    for i, r in enumerate(rows):
        if i:
            grid_lines.append(sep_v)
        grid_lines.append(r)
    grid = np.vstack(grid_lines)

    # title strip — ASCII only
    title_h = 64
    title = np.full((title_h, grid.shape[1], 3), 255, dtype=np.uint8)
    cv2.putText(title,
                "YOLOv8n alone  vs  YOLOv8n + ByteTrack (Kalman fill):  both hands recovered",
                (20, 42),
                cv2.FONT_HERSHEY_SIMPLEX, 0.95, (15, 15, 15), 2, cv2.LINE_AA)
    out = np.vstack([title, grid])
    cv2.rectangle(out, (0, 0), (out.shape[1] - 1, out.shape[0] - 1),
                  (200, 200, 200), 1)

    out_png = ROOT / f"{args.out_stem}.png"
    cv2.imwrite(str(out_png), out)
    print(f"[wrote] {out_png}  ({out.shape[1]}×{out.shape[0]})")

    if args.pdf:
        # Embed the PNG into a matplotlib PDF at textwidth.
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7.0, 7.0 * out.shape[0] / out.shape[1]))
        ax.imshow(out[:, :, ::-1])         # BGR → RGB
        ax.set_axis_off()
        fig.tight_layout(pad=0)
        out_pdf = ROOT / f"{args.out_stem}.pdf"
        fig.savefig(out_pdf, dpi=200, bbox_inches="tight", pad_inches=0)
        plt.close(fig)
        print(f"[wrote] {out_pdf}")


if __name__ == "__main__":
    main()
