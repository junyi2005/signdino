"""Post-process existing face crops to suppress identity signal.

The original face crops are 112x112 of the whole face padded x1.2 around the
YOLO box, which keeps hair / ears / chin / collar -- all of which are highly
identity-discriminative and dominate DINOv2 features.

This script crops each face image to its inner ~70% (a tight mid-face strip
covering eyes + nose + mouth + upper cheeks only), then resizes back to
112x112. The result emphasises expression / mouthing / gaze cues.

Outputs go to <root>/face_mid/, leaving <root>/face/ untouched.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path,
                    default=Path("runs/yolov8_hf/cluster_crops"))
    ap.add_argument("--stream", default="face",
                    help="source subfolder (face/lh/rh)")
    ap.add_argument("--dest", default=None,
                    help="dest subfolder; defaults to <stream>_tight")
    ap.add_argument("--inner", type=float, default=0.72,
                    help="fraction of side length to keep (centred crop)")
    args = ap.parse_args()

    src_dir = args.root / args.stream
    dst_name = args.dest or (
        "face_mid" if args.stream == "face" and args.inner <= 0.72 else f"{args.stream}_tight"
    )
    dst_dir = args.root / dst_name
    dst_dir.mkdir(parents=True, exist_ok=True)

    paths = sorted(src_dir.glob("*.png"))
    n_in = 112
    margin = int(round((1.0 - args.inner) * n_in * 0.5))
    print(f"[mid-face] keeping inner {args.inner*100:.0f}% "
          f"(margin {margin}px on each side); {len(paths)} crops")

    for i, p in enumerate(paths):
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if img is None:
            continue
        H, W = img.shape[:2]
        m_y = int(round(margin * H / n_in))
        m_x = int(round(margin * W / n_in))
        crop = img[m_y:H - m_y, m_x:W - m_x]
        out = cv2.resize(crop, (n_in, n_in), interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(dst_dir / p.name), out)
        if (i + 1) % 2000 == 0:
            print(f"  [mid-face] {i + 1}/{len(paths)}", flush=True)
    print(f"[mid-face] wrote {dst_dir}")


if __name__ == "__main__":
    main()
