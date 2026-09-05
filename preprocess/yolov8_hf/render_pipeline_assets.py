"""Render all Row-A assets for the pipeline figure from a single H2S video.

Outputs (paths fixed so make_pipeline.py picks them up automatically):

  SignDino/figs/pipeline_raw_frame.jpg       — full RGB frame, middle of window
  make_picture/yolo_track_pipeline.png       — YOLO vs TRACK+PRED inset for that frame
  SignDino/figs/pipeline_face_{1..8}.jpg     — 112×112 face crops, 8 frames
  SignDino/figs/pipeline_rh_{1..8}.jpg       — 112×112 right-hand crops, 8 frames
  SignDino/figs/pipeline_lh_{1..8}.jpg       — 112×112 left-hand crops, 8 frames

Crops use a square half-size = max(box_w, box_h)/2 + padding, so the box is
loosely centered with breathing room.  Anything that falls off-frame is left
blank only if --strict-edge is OFF; --strict-edge raises if so we never produce
black-bottom crops by accident.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path("/path/to/signdino")
sys.path.insert(0, str(ROOT))

from preprocess.yolov8_hf.yolo_bytetrack import (  # noqa: E402
    HandFaceDetector, fps_of, run_on_video, Box,
)

FIGS = ROOT / "SignDino" / "figs"
MAKE = ROOT / "make_picture"


# --- drawing parameters mirror extract_pairs.py -----------------------------

COL_FACE = (80, 220, 80)             # green
COL_HAND_ACTIVE = (50, 170, 255)     # orange-amber
COL_PRED = (255, 0, 255)             # magenta
COL_BORDER = (214, 214, 214)


def _color_for(b: Box) -> tuple[int, int, int]:
    if b.state == "predicted":
        return COL_PRED
    if b.label == "face":
        return COL_FACE
    return COL_HAND_ACTIVE


def _label_text(b: Box) -> str:
    name = f"{b.label}#{b.track_id}"
    suffix = "pred" if b.state == "predicted" else f"{b.score:.2f}"
    return f"{name} {suffix}"


def _clip_box(box, w, h):
    x1, y1, x2, y2 = [int(round(v)) for v in box]
    return (max(0, min(w - 1, x1)), max(0, min(h - 1, y1)),
            max(0, min(w - 1, x2)), max(0, min(h - 1, y2)))


def _draw_box(img: np.ndarray, b: Box) -> None:
    h, w = img.shape[:2]
    x1, y1, x2, y2 = _clip_box(b.xyxy, w, h)
    col = _color_for(b)
    cv2.rectangle(img, (x1, y1), (x2, y2), col, 2)
    text = _label_text(b)
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)
    ty = max(th + 3, y1)
    cv2.rectangle(img, (x1, ty - th - 4), (min(w - 1, x1 + tw + 4), ty + 2), col, -1)
    cv2.putText(img, text, (x1 + 2, ty - 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 0, 0), 1, cv2.LINE_AA)


# --- crop helper ------------------------------------------------------------


def square_crop(img: np.ndarray, box, pad_frac: float = 0.15,
                out_size: int = 112,
                strict_edge: bool = False) -> np.ndarray:
    """Crop a square region around `box` (xyxy), pad by `pad_frac`, resize to
    out_size.  If the box goes off-frame, raise (in strict mode) or pad with the
    edge color."""
    H, W = img.shape[:2]
    x1, y1, x2, y2 = box
    cx, cy = 0.5 * (x1 + x2), 0.5 * (y1 + y2)
    half = max(x2 - x1, y2 - y1) * (0.5 + pad_frac)
    half = max(half, 24.0)
    cx1 = int(round(cx - half))
    cy1 = int(round(cy - half))
    side = 2 * int(round(half))
    cx2 = cx1 + side
    cy2 = cy1 + side

    if strict_edge and (cx1 < 0 or cy1 < 0 or cx2 > W or cy2 > H):
        raise RuntimeError(f"crop off-frame ({cx1},{cy1})-({cx2},{cy2}) "
                            f"vs image {W}x{H}")

    crop = np.full((side, side, 3), 60, dtype=np.uint8)
    sx1, sy1 = max(0, cx1), max(0, cy1)
    sx2, sy2 = min(W, cx2), min(H, cy2)
    if sx2 > sx1 and sy2 > sy1:
        dx1, dy1 = sx1 - cx1, sy1 - cy1
        dx2, dy2 = dx1 + (sx2 - sx1), dy1 + (sy2 - sy1)
        crop[dy1:dy2, dx1:dx2] = img[sy1:sy2, sx1:sx2]
    return cv2.resize(crop, (out_size, out_size), interpolation=cv2.INTER_AREA)


# --- render pipeline inset --------------------------------------------------


def render_inset(frame: np.ndarray, yolo_boxes, track_boxes,
                  panel_width: int = 600,
                  mode: str = "pair") -> np.ndarray:
    """Render the A2 inset.

    mode='pair'   — side-by-side YOLO + TRACK+PRED panels with small headers
    mode='single' — single TRACK+PRED panel, no header (clean image for the
                    pipeline figure where the parent box already carries the
                    title 'YOLOv8n + ByteTrack').
    """
    h, w = frame.shape[:2]
    scale = panel_width / w
    panel_h = int(round(h * scale))
    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC

    def _panel(boxes):
        p = cv2.resize(frame, (panel_width, panel_h), interpolation=interp)
        for b in boxes:
            x1, y1, x2, y2 = b.xyxy
            sb = Box(xyxy=(x1*scale, y1*scale, x2*scale, y2*scale),
                     score=b.score, label=b.label, track_id=b.track_id,
                     state=b.state, lost_age=b.lost_age)
            _draw_box(p, sb)
        return p

    if mode == "single":
        # Single TRACK+PRED panel, no header / no caption — the parent figure
        # supplies the title.
        track_panel = _panel(track_boxes)
        cv2.rectangle(track_panel, (0, 0),
                       (track_panel.shape[1] - 1, track_panel.shape[0] - 1),
                       COL_BORDER, 1)
        return track_panel

    yolo_panel = _panel(yolo_boxes)
    track_panel = _panel(track_boxes)

    # short captions baked into a small header above each panel (no big
    # case-id strip — the parent figure already labels the box).
    cap_h = 24
    pair_w = 2 * panel_width + 14         # 14 px gap between panels
    out = np.full((panel_h + cap_h, pair_w, 3), 255, dtype=np.uint8)
    cv2.putText(out, "YOLO", (panel_width // 2 - 22, 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(out, "TRACK + PRED",
                (panel_width + 14 + panel_width // 2 - 72, 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 0, 0), 2, cv2.LINE_AA)
    out[cap_h:, :panel_width] = yolo_panel
    out[cap_h:, panel_width + 14:] = track_panel
    cv2.rectangle(out, (0, cap_h), (panel_width - 1, panel_h + cap_h - 1),
                  COL_BORDER, 1)
    cv2.rectangle(out, (panel_width + 14, cap_h),
                  (pair_w - 1, panel_h + cap_h - 1), COL_BORDER, 1)
    return out


# --- driver -----------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True,
                    help="path to source video")
    ap.add_argument("--start-frame", type=int, required=True,
                    help="first frame of the 8-frame window")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--center-frame", type=int, default=None,
                    help="frame to use for the raw + inset (default = middle "
                          "of the window)")
    ap.add_argument("--inset-panel-width", type=int, default=600)
    ap.add_argument("--inset-mode", choices=["pair", "single"], default="pair")
    ap.add_argument("--portrait-crop-frame", action="store_true",
                    help="if set, crop raw frame to ~0.9 portrait aspect "
                          "around the signer instead of keeping the full "
                          "1280x720 landscape")
    ap.add_argument("--out-prefix", default="pipeline",
                    help="output basename prefix (figs/<prefix>_lh_1.jpg etc)")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--strict-edge", action="store_true")
    args = ap.parse_args()

    src = Path(args.src)
    detector = HandFaceDetector(device=args.device, fps=fps_of(src))

    frames, results = [], []
    for img, res in run_on_video(detector, src, batch=32):
        frames.append(img)
        results.append(res)
    print(f"[info] {src.name}: {len(frames)} frames at {fps_of(src):.1f} fps")

    s, n = args.start_frame, args.n
    assert s + n <= len(frames), f"window {s}..{s+n} exceeds {len(frames)} frames"

    center_idx = args.center_frame if args.center_frame is not None else (s + n // 2)
    assert s <= center_idx < s + n
    print(f"[info] window=frames[{s}..{s+n-1}], center={center_idx}")

    # ----- raw frame ----------------------------------------------------
    raw_path = FIGS / f"{args.out_prefix}_raw_frame.jpg"
    cv2.imwrite(str(raw_path), frames[center_idx],
                [cv2.IMWRITE_JPEG_QUALITY, 92])
    print(f"[wrote] {raw_path}  ({frames[center_idx].shape[1]}x{frames[center_idx].shape[0]})")

    # ----- YOLO/TRACK inset --------------------------------------------
    res = results[center_idx]
    inset = render_inset(frames[center_idx],
                          res.yolo_boxes, res.track_boxes,
                          panel_width=args.inset_panel_width)
    inset_path = MAKE / "yolo_track_pipeline.png"
    cv2.imwrite(str(inset_path), inset)
    print(f"[wrote] {inset_path}  ({inset.shape[1]}x{inset.shape[0]})")

    # ----- L/R/F crop strip --------------------------------------------
    for k in range(n):
        idx = s + k
        img = frames[idx]
        res = results[idx]
        face = next((b for b in res.track_boxes if b.label == "face"), None)
        h1   = next((b for b in res.track_boxes
                       if b.label == "hand" and b.track_id == 1), None)
        h2   = next((b for b in res.track_boxes
                       if b.label == "hand" and b.track_id == 2), None)
        assert face is not None and h1 is not None and h2 is not None, \
                f"missing tracks at frame {idx}"

        for tag, b in (("face", face), ("rh", h2), ("lh", h1)):
            crop = square_crop(img, b.xyxy, pad_frac=0.18,
                                out_size=112,
                                strict_edge=args.strict_edge)
            out_path = FIGS / f"{args.out_prefix}_{tag}_{k+1}.jpg"
            cv2.imwrite(str(out_path), crop,
                        [cv2.IMWRITE_JPEG_QUALITY, 90])
            if k == 0:
                print(f"[wrote] {out_path}  ({crop.shape[1]}x{crop.shape[0]})")
    print(f"[done] wrote 8 frames × 3 streams under {FIGS}/")


if __name__ == "__main__":
    main()
