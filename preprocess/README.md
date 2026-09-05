# Preprocessing pipeline (upstream of SSL training)

This README is the **single source of truth** for what the SSL trainer in
the parent directory expects on disk. Anything that produces files in the
crop layout below — by hand, by an in-repo script, or by an external
machine — is compatible.

## Stage P1 — Video augmentation (480× per source video)

Driver: an in-house GUAVA + EHM-Tracker renderer (not part of this release).

For each input video (How2Sign, PHOENIX-2014T, ...), produce 240
augmented variants (3D viewpoints via GUAVA, identity reenactment, 2D
CV augmentation, temporal speed/FPS), then left/right mirror each at the
video level → 480 augmented videos per source.

Each output mp4 is named:

```
<source_video_id>__<aug_tag>__<mirror_tag>.mp4
```

`mirror_tag ∈ {"orig", "mirror"}`; `aug_tag` is a stable identifier of the
augmentation parameters (e.g. `cv13`, `3d_yaw+0.2_pitch-0.1`,
`id07_yaw+0.2_zoom1.0`).

## Stage P2 — Per-frame patch extraction (LH / RH / Face)

Driver: `yolov8n-hf-trt` (remote machine `<detector-host>`, project root
`<detector-project-root>/`, conda env at
`<conda-env>`, activate with
`source env.sh`).

The pipeline runs **YOLOv8n** for face + hand detection on every frame
and **ByteTrack** to associate hand identities across frames (so left
and right hand assignments are temporally consistent and short detector
drops are interpolated). For each augmented video it emits the on-disk
crop layout below.

Crops:
- **Face**: square crop around the YOLOv8n face box, padded × 1.2.
- **LH / RH**: square crop around the tracked hand box, padded × 1.4
  (hand fast motion blurs the box → larger context).

Crop size at write time: 112 × 112 (chosen to give 49 patches under
DINOv3 ViT-B/16; can be re-rendered larger and downsampled at training).

When the tracker fails for ≥ K consecutive frames the crop is left
empty and `valid_mask[t] = 0`.

## On-disk crop layout (the contract)

```
<CROP_ROOT>/
  index.jsonl                                  # one line per video, with split tag
  <video_id>/                                  # mirrors P1 video naming
    manifest.json
    lh/   000000.jpg ... 000NNN.jpg
    rh/   000000.jpg ... 000NNN.jpg
    face/ 000000.jpg ... 000NNN.jpg
```

`index.jsonl`:

```json
{"video_id": "<id>", "split": "train", "source_video": "<src>", "augmentation_tag": "<aug>__<mirror>"}
```

`manifest.json` schema:

```json
{
  "video_id": "BrinTPjveZY_6-3-rgb_front__cv13__mirror",
  "source_video": "BrinTPjveZY_6-3-rgb_front.mp4",
  "augmentation_tag": "cv13__mirror",
  "num_frames": 122,
  "fps": 24.0,
  "streams": {
    "lh":   {"crop_size": [112, 112], "valid_mask": "0,1,1,...,1"},
    "rh":   {"crop_size": [112, 112], "valid_mask": "1,1,1,...,1"},
    "face": {"crop_size": [112, 112], "valid_mask": "1,1,1,...,1"}
  }
}
```

`valid_mask` is a comma-separated 0/1 string of length `num_frames`.

## Adapter for the remote pipeline

If the remote `yolov8n-hf-trt` pipeline writes its output in a different
on-disk layout, run `python preprocess/adapt_remote_crops.py
--remote-root <path> --out-root <CROP_ROOT>` to convert it to the
contract above. (As of v1 this script is a stub; see TODO inside.)

## Stage P3 — Frozen-embedding cache (optional, recommended for v1)

Driver: `preprocess/cache_embeddings.py`.

One-off pass that loads each video's crops, normalises to ImageNet
mean/std, forwards through the frozen DINOv3 ViT-B/16, and saves the
per-frame 768-d CLS embedding to
`<CROP_ROOT>/.embedding_cache/dinov3_vitb16_224/<video_id>/<stream>.pt`.

Trade-off (covered in `DESIGN.md` § 4.3): caching skips per-iter image
augmentation and gives a ~50× training speedup. v1 uses the cache by
default; we will report an ablation with `embedding_source: live`.
