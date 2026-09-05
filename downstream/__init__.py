"""Downstream tasks: translation, isolated sign recognition, fingerspelling.

All three consume the same on-disk artefacts:
- the crop layout (`<crop_root>/<video_id>/{lh,rh,face}/...`) produced by the
  upstream YOLOv8n + ByteTrack pipeline (see `preprocess/README.md`); and
- per-stream contextualised features dumped by `extract_features.py` for
  each trained SignDINO stream encoder.

Each task subpackage exposes a `model`, `dataset`, `trainer`, `infer`
module + a `configs/` directory.
"""

__all__ = ["common", "translation", "islr", "fingerspelling"]
