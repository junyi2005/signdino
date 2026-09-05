"""Render N qualitative YOLO-vs-ByteTrack tiles, each saved as its own image.

Mirrors the layout of the official 118 grids
(`yolov8n-hf-trt/assets/grid_*_yolo_vs_track_pred.jpg`):

  +-----------------------------------------------------+
  | Case 003 | ts=42 | <clip>@0042                       |
  | Pred: hand#2 +1f                                     |
  | YOLO                | TRACK + PRED                   |
  +---------------------+-------------------------------+
  | YOLOv8 boxes        | ByteTrack active + Kalman     |
  | (face green,        | preds (face green, hand       |
  | hand orange)        | orange, preds magenta)        |
  +---------------------+-------------------------------+

Unlike `make_demo_grid.py` (deprecated), this script writes each tile as a
standalone PNG into the chosen `--out-dir`. Up to one tile per source clip is
emitted, and only frames where the tracker recovered at least one hand that
YOLO missed are kept (i.e. only "interesting" frames). Pass `--all-pred-frames`
to keep every recovered frame instead of one per clip.

Usage:
    python preprocess/yolov8_hf/extract_pairs.py \\
        --sources-glob '/path/to/videos/*.mp4' \\
        --out-dir runs/yolov8_hf/h2s_pairs \\
        --num 200 --device cuda:0
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from preprocess.yolov8_hf.yolo_bytetrack import (  # noqa: E402
    Box, FrameResult, HandFaceDetector, fps_of, run_on_video,
)


# --- drawing parameters (mirror track_hf_phoenix.draw_track) -----------------

COL_FACE = (80, 220, 80)             # green
COL_HAND_ACTIVE = (50, 170, 255)     # orange-amber
COL_PRED = (255, 0, 255)             # magenta
COL_HEADER_BG = (246, 246, 246)
COL_BORDER = (214, 214, 214)
COL_TEXT_DARK = (15, 15, 15)
COL_TEXT_MID = (55, 55, 55)


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


def _clip_box(box: tuple[float, float, float, float], w: int, h: int) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = [int(round(v)) for v in box]
    return (
        max(0, min(w - 1, x1)),
        max(0, min(h - 1, y1)),
        max(0, min(w - 1, x2)),
        max(0, min(h - 1, y2)),
    )


def _draw_box(img: np.ndarray, b: Box) -> None:
    h, w = img.shape[:2]
    x1, y1, x2, y2 = _clip_box(b.xyxy, w, h)
    col = _color_for(b)
    cv2.rectangle(img, (x1, y1), (x2, y2), col, 2)

    text = _label_text(b)
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)
    ty = max(th + 3, y1)
    cv2.rectangle(img, (x1, ty - th - 4), (min(w - 1, x1 + tw + 4), ty + 2), col, -1)
    cv2.putText(img, text, (x1 + 2, ty - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 0, 0), 1, cv2.LINE_AA)


def _put_text(img: np.ndarray, org, text: str, *,
              color=COL_TEXT_DARK, scale=0.72, thick=2) -> None:
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


# --- candidate selection -----------------------------------------------------


@dataclass
class Candidate:
    src: Path
    frame_idx: int
    img: np.ndarray
    yolo_boxes: list[Box]
    track_boxes: list[Box]      # active + predicted
    pred_summary: list[str]     # ["hand#2 +1f", ...]


def find_candidates(src: Path, batch: int, max_per_src: int, device) -> list[Candidate]:
    detector = HandFaceDetector(device=device, fps=fps_of(src))
    keep: list[Candidate] = []
    for img, res in run_on_video(detector, src, batch=batch):
        preds = [b for b in res.track_boxes if b.state == "predicted" and b.label == "hand"]
        if not preds:
            continue
        if not res.yolo_boxes:
            # all-blank YOLO frames are less interesting
            continue
        pred_summary = [f"hand#{b.track_id} +{b.lost_age}f" for b in preds]
        keep.append(Candidate(
            src=src,
            frame_idx=res.frame_idx,
            img=img.copy(),
            yolo_boxes=list(res.yolo_boxes),
            track_boxes=list(res.track_boxes),
            pred_summary=pred_summary,
        ))
        if len(keep) >= max_per_src * 8:  # over-sample, prune later
            break
    return keep


# --- tile rendering ----------------------------------------------------------


def render_tile(case_id: int, cand: Candidate, panel_width: int) -> np.ndarray:
    """Render a single side-by-side tile.

    Each panel is resized so that the WIDER side is `panel_width` (boxes scaled
    in lock-step), then boxes are drawn at panel resolution. This keeps the
    on-screen stroke + font visible (matches the official 118 grids) regardless
    of source video resolution.
    """
    src_img = cand.img
    h, w = src_img.shape[:2]
    scale_x = panel_width / w
    panel_h = int(round(h * scale_x))
    interp = cv2.INTER_AREA if scale_x < 1.0 else cv2.INTER_CUBIC

    def _panel(boxes: Iterable[Box]) -> np.ndarray:
        panel = cv2.resize(src_img, (panel_width, panel_h), interpolation=interp)
        for b in boxes:
            x1, y1, x2, y2 = b.xyxy
            scaled = Box(
                xyxy=(x1 * scale_x, y1 * scale_x, x2 * scale_x, y2 * scale_x),
                score=b.score, label=b.label, track_id=b.track_id,
                state=b.state, lost_age=b.lost_age,
            )
            _draw_box(panel, scaled)
        return panel

    yolo_panel = _panel(cand.yolo_boxes)
    track_panel = _panel(cand.track_boxes)

    pair = np.hstack([yolo_panel, track_panel])

    header_h = 92
    cell = np.full((pair.shape[0] + header_h, pair.shape[1], 3), 246, dtype=np.uint8)
    cell[header_h:] = pair

    pred_text = ", ".join(cand.pred_summary) if cand.pred_summary else "—"
    src_short = cand.src.stem[:48]
    _put_text(cell, (12, 30),
              f"Case {case_id:03d} | {src_short} | f{cand.frame_idx:04d}",
              color=COL_TEXT_DARK, scale=0.72, thick=2)
    _put_text(cell, (12, 62), f"Pred: {pred_text}", color=COL_TEXT_MID, scale=0.66, thick=2)
    _put_text(cell, (12, 86), "YOLO", color=(0, 0, 0), scale=0.66, thick=2)
    _put_text(cell, (panel_width + 12, 86), "TRACK + PRED", color=(0, 0, 0), scale=0.66, thick=2)
    cv2.line(cell, (panel_width, header_h), (panel_width, cell.shape[0] - 1), (255, 255, 255), 4)
    cv2.rectangle(cell, (0, 0), (cell.shape[1] - 1, cell.shape[0] - 1), COL_BORDER, 2)
    return cell


# --- driver ------------------------------------------------------------------


def gather_sources(sources_glob: str | None, sources_root: str | None, ext: str,
                   shuffle_seed: int, limit: int) -> list[Path]:
    paths: list[Path] = []
    if sources_glob:
        paths = sorted(Path().glob(sources_glob)) if not Path(sources_glob).is_absolute() else \
            sorted(Path("/").glob(sources_glob.lstrip("/")))
    elif sources_root:
        root = Path(sources_root)
        paths = sorted(root.rglob(f"*{ext}"))
    else:
        raise SystemExit("must pass --sources-glob or --sources-root")
    rng = random.Random(shuffle_seed)
    rng.shuffle(paths)
    return paths[:limit] if limit > 0 else paths


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources-glob", default=None,
                    help="glob for video files / frame dirs (e.g. '/.../*.mp4')")
    ap.add_argument("--sources-root", default=None,
                    help="root dir to recursively scan for *.mp4 (or --ext)")
    ap.add_argument("--ext", default=".mp4")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--num", type=int, default=200,
                    help="number of individual tiles to emit")
    ap.add_argument("--max-per-src", type=int, default=1,
                    help="max candidate frames kept per source video")
    ap.add_argument("--max-sources", type=int, default=800,
                    help="cap on number of source clips scanned")
    ap.add_argument("--batch", type=int, default=32,
                    help="YOLO batch size for predict_batch")
    ap.add_argument("--panel-width", type=int, default=480,
                    help="width of each panel in pixels — boxes are drawn at this resolution")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--manifest", type=Path, default=None,
                    help="optional JSON manifest of chosen cases")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    sources = gather_sources(
        args.sources_glob, args.sources_root, args.ext, args.seed, args.max_sources,
    )
    if not sources:
        sys.exit("[fatal] no source clips matched")
    print(f"[info] {len(sources)} source clips to scan")

    all_cands: list[Candidate] = []
    for i, src in enumerate(sources, start=1):
        try:
            cands = find_candidates(src, batch=args.batch,
                                    max_per_src=args.max_per_src, device=args.device)
        except Exception as e:
            print(f"[warn] {src.name}: {e}")
            continue
        random.Random(args.seed + i).shuffle(cands)
        cands = cands[: args.max_per_src]
        all_cands.extend(cands)
        print(f"[{i}/{len(sources)}] {src.name[:48]:48s}  +{len(cands)}  "
              f"(total {len(all_cands)})", flush=True)
        if len(all_cands) >= args.num:
            break

    if not all_cands:
        sys.exit("[fatal] no recovery candidates found")

    rng = random.Random(args.seed + 99)
    rng.shuffle(all_cands)
    chosen = all_cands[: args.num]
    chosen.sort(key=lambda c: (str(c.src), c.frame_idx))

    print(f"[info] rendering {len(chosen)} tiles to {args.out_dir}")
    manifest: list[dict] = []
    for case_id, cand in enumerate(chosen, start=1):
        tile = render_tile(case_id, cand, panel_width=args.panel_width)
        out_name = f"case_{case_id:03d}_{cand.src.stem[:32]}_f{cand.frame_idx:04d}.png"
        out_path = args.out_dir / out_name
        cv2.imwrite(str(out_path), tile)
        manifest.append({
            "case_id": case_id,
            "src": str(cand.src),
            "frame_idx": cand.frame_idx,
            "out": str(out_path),
            "pred_summary": cand.pred_summary,
        })
    print(f"[done] wrote {len(chosen)} images → {args.out_dir}")

    if args.manifest is None:
        args.manifest = args.out_dir / "manifest.json"
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2))
    print(f"[done] manifest → {args.manifest}")


if __name__ == "__main__":
    main()
