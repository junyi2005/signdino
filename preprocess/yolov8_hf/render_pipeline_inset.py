"""Render a single YOLO / TRACK+PRED pair for use inside the pipeline figure.

Differences vs. extract_pairs.render_tile():

  * NO 'Case NNN | <stem> | fXXXX' header — the parent figure already labels
    the box, so the inset just shows the imagery.
  * A visible white gap between the YOLO and TRACK panels (the default
    `hstack` makes them look squished).
  * Thin "YOLO" / "TRACK + PRED" captions baked just above each panel
    (small, centered, optional via --no-caption).
  * Tight outer border so the image fits flush inside the pipeline box.

Usage:
    python preprocess/yolov8_hf/render_pipeline_inset.py \\
        --src /path/to/clip.mp4 \\
        --frame-idx 12 \\
        --out make_picture/yolo_track_pipeline.png \\
        --panel-width 460 --gap 14
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from preprocess.yolov8_hf.yolo_bytetrack import (  # noqa: E402
    Box, HandFaceDetector, fps_of, run_on_video,
)
from preprocess.yolov8_hf.extract_pairs import (  # noqa: E402
    _color_for, _label_text, _clip_box,
)


def _draw_box(img: np.ndarray, b: Box, stroke: int = 2) -> None:
    h, w = img.shape[:2]
    x1, y1, x2, y2 = _clip_box(b.xyxy, w, h)
    col = _color_for(b)
    cv2.rectangle(img, (x1, y1), (x2, y2), col, stroke)

    text = _label_text(b)
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)
    ty = max(th + 3, y1)
    cv2.rectangle(img, (x1, ty - th - 4),
                  (min(w - 1, x1 + tw + 4), ty + 2), col, -1)
    cv2.putText(img, text, (x1 + 2, ty - 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 0, 0), 1, cv2.LINE_AA)


def _panel(src_img: np.ndarray, boxes, panel_w: int) -> np.ndarray:
    h, w = src_img.shape[:2]
    scale = panel_w / w
    panel_h = int(round(h * scale))
    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
    panel = cv2.resize(src_img, (panel_w, panel_h), interpolation=interp)
    for b in boxes:
        x1, y1, x2, y2 = b.xyxy
        scaled = Box(
            xyxy=(x1 * scale, y1 * scale, x2 * scale, y2 * scale),
            score=b.score, label=b.label, track_id=b.track_id,
            state=b.state, lost_age=b.lost_age,
        )
        _draw_box(panel, scaled)
    return panel


def run_to_frame(src: Path, frame_idx: int):
    """Run the full detector+tracker up to the requested frame, return
    (raw_img, yolo_boxes, track_boxes)."""
    detector = HandFaceDetector(device=0, fps=fps_of(src))
    saved = None
    for img, res in run_on_video(detector, src, batch=32):
        if res.frame_idx == frame_idx:
            saved = (img.copy(), list(res.yolo_boxes), list(res.track_boxes))
            break
    if saved is None:
        raise RuntimeError(f"frame_idx {frame_idx} not reached in {src}")
    return saved


def render(src: Path, frame_idx: int, panel_w: int, gap_px: int,
           caption: bool, out: Path) -> None:
    img, yolo_boxes, track_boxes = run_to_frame(src, frame_idx)
    left = _panel(img, yolo_boxes, panel_w)
    right = _panel(img, track_boxes, panel_w)
    h = left.shape[0]
    gap = np.full((h, gap_px, 3), 255, dtype=np.uint8)
    pair = np.hstack([left, gap, right])

    if caption:
        cap_h = 22
        full = np.full((pair.shape[0] + cap_h, pair.shape[1], 3),
                       255, dtype=np.uint8)
        full[cap_h:] = pair
        font = cv2.FONT_HERSHEY_SIMPLEX
        for x_center, text in [
            (panel_w // 2, "YOLO"),
            (panel_w + gap_px + panel_w // 2, "TRACK + PRED"),
        ]:
            (tw, th), _ = cv2.getTextSize(text, font, 0.52, 1)
            cv2.putText(full, text,
                        (x_center - tw // 2, cap_h - 6),
                        font, 0.52, (30, 30, 30), 1, cv2.LINE_AA)
        pair = full

    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), pair)
    print(f"[wrote] {out}  ({pair.shape[1]}x{pair.shape[0]})")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", type=Path, required=True,
                    help="source video file")
    ap.add_argument("--frame-idx", type=int, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--panel-width", type=int, default=460)
    ap.add_argument("--gap", type=int, default=14,
                    help="white gap between the two panels (px)")
    ap.add_argument("--no-caption", action="store_true",
                    help="omit the YOLO / TRACK+PRED captions above the panels")
    args = ap.parse_args()
    render(args.src, args.frame_idx, args.panel_width, args.gap,
           caption=not args.no_caption, out=args.out)


if __name__ == "__main__":
    main()
