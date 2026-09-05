"""Extract face / LH / RH crops at scale for cluster-visualisation figures.

For each video in the provided sources, we:
  1. Sample frames at a fixed stride.
  2. Run the YOLOv8n + ByteTrack pipeline (same params as extract_pairs.py).
  3. From each frame's track_boxes, pick frames where ALL THREE of face#1,
     hand#1, hand#2 are 'active' (not predicted) with score >= the per-class
     thresholds. Boxes must also be sufficiently large.
  4. Pad each box (face x1.2, hand x1.4, square) and crop the source frame.
  5. Resize to 112x112 and save as PNG into:
        cluster_crops/face/<corpus>_<idx>.png
        cluster_crops/lh/<corpus>_<idx>.png
        cluster_crops/rh/<corpus>_<idx>.png
  6. Append a record to manifest.jsonl with (corpus, src, frame, scores, ...).

Designed to be re-runnable: caches its own progress.jsonl so consecutive runs
on the same out-dir continue scanning new sources without re-processing
previously-seen ones.

Usage:
    python preprocess/yolov8_hf/extract_cluster_crops.py \\
        --sources-glob '/path/to/*.mp4' --corpus h2s \\
        --out-dir runs/yolov8_hf/cluster_crops \\
        --max-sources 200 --stride 30 --device cuda:0
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from preprocess.yolov8_hf.yolo_bytetrack import (  # noqa: E402
    Box, HandFaceDetector, fps_of,
)


# Conservative thresholds: we want CLEAR crops, not borderline detections.
# Overridable via CLI for low-detection corpora (--lenient).
FACE_MIN_SCORE = 0.55
HAND_MIN_SCORE = 0.40
FACE_MIN_SIDE_PX = 40        # min face box side at native resolution
HAND_MIN_SIDE_PX = 32        # min hand box side at native resolution
FACE_PAD = 1.20
HAND_PAD = 1.40
CROP_SIZE = 112


def _square_pad(xyxy, pad: float, W: int, H: int) -> tuple[int, int, int, int] | None:
    """Return a square crop in original coords, expanded by `pad` around the box
    centre and clipped to the frame.  Returns None if the resulting box is degenerate."""
    x1, y1, x2, y2 = xyxy
    cx = (x1 + x2) * 0.5
    cy = (y1 + y2) * 0.5
    side = max(x2 - x1, y2 - y1) * pad
    half = side * 0.5
    sx1 = int(round(cx - half))
    sy1 = int(round(cy - half))
    sx2 = int(round(cx + half))
    sy2 = int(round(cy + half))
    # Clip and require non-empty
    sx1 = max(0, sx1)
    sy1 = max(0, sy1)
    sx2 = min(W, sx2)
    sy2 = min(H, sy2)
    if sx2 - sx1 < 8 or sy2 - sy1 < 8:
        return None
    return sx1, sy1, sx2, sy2


def _crop_and_resize(img: np.ndarray, box) -> np.ndarray:
    x1, y1, x2, y2 = box
    crop = img[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    out = cv2.resize(crop, (CROP_SIZE, CROP_SIZE), interpolation=cv2.INTER_AREA)
    return out


def _box_ok(b: Box, min_score: float, min_side_px: float) -> bool:
    if b.state != "active":
        return False
    if b.score < min_score:
        return False
    x1, y1, x2, y2 = b.xyxy
    side_min = min(x2 - x1, y2 - y1)
    if side_min < min_side_px:
        return False
    return True


def _pick(boxes: list[Box], label: str, track_id: int) -> Box | None:
    for b in boxes:
        if b.label == label and b.track_id == track_id:
            return b
    return None


@dataclass
class FrameSample:
    frame_idx: int
    img: np.ndarray              # BGR uint8 HxWx3


def iter_video_strided(src: Path, stride: int) -> Iterator[FrameSample]:
    """Yield every Nth frame of a video, with frame_idx attached."""
    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        return
    idx = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if idx % stride == 0:
                yield FrameSample(frame_idx=idx, img=frame)
            idx += 1
    finally:
        cap.release()


def process_video(
    detector: HandFaceDetector,
    src: Path,
    corpus: str,
    out_dirs: dict,
    counter: dict,
    manifest_f,
    stride: int,
    batch: int,
    max_per_src: int,
) -> int:
    """Stream frames from src, run detection in batches, save crops. Returns the
    number of triplets successfully extracted."""
    # collect strided frames first (so we can run detector in big batches)
    samples = list(iter_video_strided(src, stride))
    if not samples:
        return 0
    detector.reset()

    saved = 0
    for i in range(0, len(samples), batch):
        chunk = samples[i:i + batch]
        imgs = [s.img for s in chunk]
        results = detector.predict_batch(imgs)
        for s, res in zip(chunk, results):
            if saved >= max_per_src:
                return saved
            face = _pick(res.track_boxes, "face", 1)
            lh   = _pick(res.track_boxes, "hand", 1)
            rh   = _pick(res.track_boxes, "hand", 2)
            if face is None or lh is None or rh is None:
                continue
            if not _box_ok(face, FACE_MIN_SCORE, FACE_MIN_SIDE_PX):
                continue
            if not _box_ok(lh, HAND_MIN_SCORE, HAND_MIN_SIDE_PX):
                continue
            if not _box_ok(rh, HAND_MIN_SCORE, HAND_MIN_SIDE_PX):
                continue
            H, W = s.img.shape[:2]
            face_box = _square_pad(face.xyxy, FACE_PAD, W, H)
            lh_box   = _square_pad(lh.xyxy,   HAND_PAD, W, H)
            rh_box   = _square_pad(rh.xyxy,   HAND_PAD, W, H)
            if face_box is None or lh_box is None or rh_box is None:
                continue
            face_crop = _crop_and_resize(s.img, face_box)
            lh_crop   = _crop_and_resize(s.img, lh_box)
            rh_crop   = _crop_and_resize(s.img, rh_box)
            if face_crop is None or lh_crop is None or rh_crop is None:
                continue
            idx = counter["next"]
            stem = f"{corpus}_{idx:06d}"
            cv2.imwrite(str(out_dirs["face"] / f"{stem}.png"), face_crop)
            cv2.imwrite(str(out_dirs["lh"]   / f"{stem}.png"), lh_crop)
            cv2.imwrite(str(out_dirs["rh"]   / f"{stem}.png"), rh_crop)
            manifest_f.write(json.dumps({
                "idx": idx,
                "stem": stem,
                "corpus": corpus,
                "src": str(src),
                "frame": s.frame_idx,
                "face_score": float(face.score),
                "lh_score": float(lh.score),
                "rh_score": float(rh.score),
            }) + "\n")
            counter["next"] = idx + 1
            saved += 1
    return saved


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources-glob", default=None,
                    help="glob for video files (recursive supported)")
    ap.add_argument("--sources-root", default=None,
                    help="directory to recursively scan for *.mp4 (or --ext)")
    ap.add_argument("--ext", default=".mp4")
    ap.add_argument("--corpus", required=True,
                    help="short tag prepended to stems (h2s / openasl / ...)")
    ap.add_argument("--out-dir", type=Path,
                    default=Path("runs/yolov8_hf/cluster_crops"))
    ap.add_argument("--max-sources", type=int, default=200)
    ap.add_argument("--stride", type=int, default=30,
                    help="sample every Nth frame of each video")
    ap.add_argument("--batch", type=int, default=32,
                    help="frames per YOLO predict_batch call")
    ap.add_argument("--max-per-src", type=int, default=8,
                    help="cap on saved triplets per source video")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=0,
                    help="RNG seed for source shuffle")
    ap.add_argument("--lenient", action="store_true",
                    help="lower face/hand detection thresholds for noisier corpora")
    args = ap.parse_args()

    if args.lenient:
        global FACE_MIN_SCORE, HAND_MIN_SCORE, FACE_MIN_SIDE_PX, HAND_MIN_SIDE_PX
        FACE_MIN_SCORE = 0.40
        HAND_MIN_SCORE = 0.25
        FACE_MIN_SIDE_PX = 28
        HAND_MIN_SIDE_PX = 22

    # gather sources
    if args.sources_glob:
        sources = sorted(Path("/").glob(args.sources_glob.lstrip("/")))
    elif args.sources_root:
        root = Path(args.sources_root)
        sources = sorted(root.rglob(f"*{args.ext}"))
    else:
        raise SystemExit("must pass --sources-glob or --sources-root")
    rng = random.Random(args.seed)
    rng.shuffle(sources)
    sources = sources[:args.max_sources]
    print(f"[cluster-crops] scanning {len(sources)} sources; corpus={args.corpus}")

    # output dirs
    out_dirs = {
        k: args.out_dir / k for k in ("face", "lh", "rh")
    }
    for d in out_dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    manifest_path = args.out_dir / "manifest.jsonl"
    progress_path = args.out_dir / f"progress_{args.corpus}.txt"

    # resume support: read previously-seen src paths
    seen = set()
    if progress_path.exists():
        seen = set(progress_path.read_text().splitlines())
    # counter: how many triplets already saved (global, across corpora)
    counter = {"next": 0}
    if manifest_path.exists():
        with open(manifest_path) as f:
            for line in f:
                rec = json.loads(line)
                counter["next"] = max(counter["next"], rec["idx"] + 1)

    detector = HandFaceDetector(device=args.device, fps=25.0)

    t0 = time.time()
    with open(manifest_path, "a") as manifest_f, open(progress_path, "a") as prog_f:
        for i, src in enumerate(sources, start=1):
            if str(src) in seen:
                continue
            n_saved = 0
            try:
                # update detector FPS for this video (track_buffer responds to fps)
                fps = fps_of(src)
                if fps > 0:
                    detector.fps = fps
                n_saved = process_video(
                    detector, src, args.corpus, out_dirs, counter,
                    manifest_f, stride=args.stride, batch=args.batch,
                    max_per_src=args.max_per_src,
                )
            except Exception as e:                                              # noqa: BLE001
                print(f"  [{i:3d}/{len(sources)}] {src.name}: ERROR {type(e).__name__}: {e}",
                      flush=True)
                continue
            prog_f.write(str(src) + "\n")
            prog_f.flush()
            elapsed = time.time() - t0
            print(f"  [{i:3d}/{len(sources)}] {src.name:60.60s}  "
                  f"+{n_saved} triplets  total={counter['next']}  "
                  f"{elapsed:.0f}s", flush=True)

    print(f"[cluster-crops] done. total triplets = {counter['next']}, "
          f"elapsed {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
