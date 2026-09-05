"""Adapter: convert the remote yolov8n-hf-trt pipeline's output to the on-disk
crop layout documented in preprocess/README.md.

TODO: the remote pipeline's exact output schema is owned by the team on
<detector-host> (`<detector-project-root>/`). Once the
schema is fixed, fill in `_convert_one_video` below. For now this script
exits non-zero with a message.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _convert_one_video(remote_video_dir: Path, out_root: Path) -> None:  # noqa: D401
    """Convert one remote video's crops to the on-disk contract.

    Expected remote layout (TODO: confirm with <detector-host> owner):
        <remote_root>/<video_id>/<stream>/frame_<idx>.jpg
        <remote_root>/<video_id>/track_meta.json  # bbox + tracker id history

    Target layout:
        <out_root>/<video_id>/{lh,rh,face}/<idx:06d>.jpg
        <out_root>/<video_id>/manifest.json
    """
    raise NotImplementedError(
        "adapt_remote_crops is a stub. Fill in _convert_one_video once the "
        "yolov8n-hf-trt output schema is finalised."
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--remote-root", required=True, type=Path)
    ap.add_argument("--out-root", required=True, type=Path)
    ap.add_argument("--split-manifest", required=False, type=Path,
                    help="CSV / json mapping video_id -> split; defaults to 'train' for all.")
    args = ap.parse_args()

    out_root = args.out_root
    out_root.mkdir(parents=True, exist_ok=True)
    index_path = out_root / "index.jsonl"

    print(f"[adapt] remote_root={args.remote_root} out_root={out_root}")
    print("[adapt] TODO: this script is a stub; see comment in _convert_one_video.")
    print("[adapt] No files were written.")
    sys.exit(1)


if __name__ == "__main__":
    main()
