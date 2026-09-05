# YOLOv8 face+hand + ByteTrack visualization pipeline

Local port of the canonical **`yolov8n-hf-trt`** pipeline (<detector-host>
`<detector-project-root>/`). A clone of that repo
lives alongside this dir at `preprocess/yolov8n-hf-trt/` for cross-reference;
the source of truth for tracker logic is `src/phoenix/track_hf_phoenix.py`
there.

Produces the "YOLO Misses Recovered by Tracking Predictions" qualitative
visualizations — same colour scheme, label format, and tracker params as the
remote implementation — for whatever sign-language video corpus we point it at
(How2Sign, OpenASL, …).

---

## Files

```
preprocess/yolov8_hf/
  README.md                 # this file
  yolo_bytetrack.py         # HandFaceDetector — full port of track_hf_phoenix
                            # (HandSlotMapper + face tracker + batched predict)
  extract_pairs.py          # CLI: emit N individual YOLO-vs-TRACK+PRED tiles
  run_h2s.py                # wrapper: extract_pairs over How2Sign val
  run_openasl.py            # wrapper: extract_pairs over our 2000-video sample
  models/
    face_yolov8n.pt         # Bingsu/adetailer face_yolov8n.pt
    hand_yolov8n.pt         # Bingsu/adetailer hand_yolov8n.pt
                            # (identical to remote weights/hand_yolov8n.pt)
```

Outputs land in `runs/yolov8_hf/`:

```
runs/yolov8_hf/
  h2s_pairs/                # 200 PNGs, one per case, How2Sign val
    case_001_*.png
    manifest.json
  openasl_pairs/            # 200 PNGs, one per case, OpenASL sample
    case_001_*.png
    manifest.json
```

The 2000-video OpenASL sample materialised by `run_openasl.py` lives at
`data/openasl_2k/`:

```
data/openasl_2k/
  videos/<2000 *.mp4>           # rsync'd from <detector-host>:.../video_org/
  data/openasl-v1.0.tsv         # canonical annotations (split, gloss, text)
  file_list.txt                 # shuf seed='sign_dino_seed_2026' on 118
```

---

## Parity with remote 118

`HandFaceDetector` (`yolo_bytetrack.py`) mirrors
`track_hf_phoenix.py` frame-for-frame:

| Param | Remote 118 | Here |
| --- | --- | --- |
| `conf` (YOLO score thresh) | **0.10** | **0.10** (was 0.30 — fixed) |
| `iou` (YOLO NMS thresh)    | **0.50** | **0.50** (was 0.7 default — fixed) |
| `imgsz`                    | 640 | 640 |
| `track_high_thresh`        | 0.25 | 0.25 |
| `track_low_thresh`         | 0.05 | 0.05 |
| `new_track_thresh`         | 0.20 | 0.20 |
| `match_thresh`             | 0.80 | 0.80 |
| `track_buffer`             | 75 | 75 |
| `fuse_score`               | True | True |
| `max_pred_draw`            | 4 | 4 |
| `keep_face / keep_hands`   | 1 / 2 | 1 / 2 |
| face tracker present       | **yes** | **yes** (was missing — fixed) |
| hand slot mapper           | **`HandSlotMapper`** (combinatorial cost) | **same class, ported** (was top-2-by-x naive — fixed) |
| pred colour                | `(255,0,255)` magenta solid | same |
| active hand colour         | `(50,170,255)` orange | same |
| face colour                | `(80,220,80)` green | same |
| label format               | `face#1 0.90`, `hand#1 0.74`, `hand#1 pred` | same |

The only intentional difference is the face model checkpoint — local uses
`Bingsu/adetailer face_yolov8n.pt`, remote uses
`yolov8n-face-lindevs.pt`. Detection quality is comparable; if you need bit
parity, drop `weights/yolov8n-face-lindevs.pt` from 118 into `models/`
and re-point `FACE_CKPT` in `yolo_bytetrack.py`.

### Things that fixed the bad How2Sign output

The first version (replaced) extracted weak hand-recovery cases because:

1. `conf=0.30 / 0.35` — too high; the official uses 0.10. Many low-score hand
   boxes the tracker would normally chain into a track were filtered out, so
   the tracker had no history to predict from.
2. NMS `iou=0.7` (default) instead of 0.5 — slightly more redundant boxes,
   but harmless on its own; matched anyway for parity.
3. **No face tracker** — face was YOLO-only, so face#1 flickered.
4. **Naive hand slot mapper** — `top-2 by score → sort left/right` does not
   keep stable hand#1 / hand#2 IDs across crossings or occlusions. The
   official `HandSlotMapper` uses a combinatorial assignment over (cost =
   center distance + IoU + raw_id history + pred penalty) which is what
   makes `hand#1 pred` show up in the same logical slot a few frames after
   the last detection.

---

## Run on a new corpus

Pick a wrapper based on the corpus. Each wrapper just stuffs sensible defaults
into `extract_pairs.py`'s argv:

```bash
cd /path/to/signdino

# How2Sign val: 1739 videos at /path/to/how2sign/val/rgb_front/raw_videos
python preprocess/yolov8_hf/run_h2s.py        # 200 tiles → runs/yolov8_hf/h2s_pairs/

# OpenASL: 2000-video sample (see data/openasl_2k/)
python preprocess/yolov8_hf/run_openasl.py    # 200 tiles → runs/yolov8_hf/openasl_pairs/

# Probe-only (no detection)
python preprocess/yolov8_hf/run_h2s.py --probe
```

Override defaults with the same flags `extract_pairs.py` accepts:

```bash
# quick 5-tile smoke test on a different out-dir
python preprocess/yolov8_hf/run_h2s.py --num 5 --max-sources 20 --out-dir runs/yolov8_hf/h2s_smoke
```

Full flag list (`extract_pairs.py --help`):

| Flag | Default | Notes |
| --- | --- | --- |
| `--sources-glob`   | — | shell glob, e.g. `'/path/**/*.mp4'` |
| `--sources-root`   | — | dir scanned recursively for `*--ext` |
| `--ext`            | `.mp4` | extension when `--sources-root` is used |
| `--out-dir PATH`   | required | individual PNGs go here, one per case |
| `--num N`          | 200 | number of tiles to emit |
| `--max-per-src N`  | 1 | cap on candidate frames kept per source clip |
| `--max-sources N`  | 800 | cap on number of source clips scanned |
| `--batch N`        | 32 | YOLO batch size for `predict_batch` |
| `--panel-width N`  | 480 | per-panel width in px (boxes drawn at this res) |
| `--device DEV`     | `cuda:0` | YOLO device |
| `--seed N`         | 0 | RNG seed (clip + tile shuffle) |
| `--manifest PATH`  | `<out-dir>/manifest.json` | json log of (case_id, src, frame_idx, pred) |

---

## Add a new corpus wrapper

Two-file change:

1. `preprocess/yolov8_hf/run_<corpus>.py` — copy `run_openasl.py`, swap
   `OPENASL_DIR` for your videos dir and `OUT_DIR` for the desired output
   path. The wrapper is just a thin argv builder around `extract_pairs.main()`.
2. **(optional)** Materialize your corpus locally first. For data on 118,
   `rsync -av --files-from=<list.txt> <detector-host>:<remote-dir>/ data/<corpus>/videos/`
   with a deterministic `shuf` seed (see `data/openasl_2k/file_list.txt`).

Things to watch when re-targeting:

1. **Aspect ratio.** `extract_pairs.render_tile()` resizes by width
   (`panel_width`) and preserves source aspect, so non-16:9 corpora (e.g.
   PHOENIX 210x260) Just Work — no white-padding letterbox issue.
2. **FPS.** `HandFaceDetector(..., fps=…)` controls
   `BYTETracker.max_time_lost`. `extract_pairs.find_candidates()` reads it
   per-clip via `fps_of()`, which uses `cv2.CAP_PROP_FPS` (folder-of-PNGs
   sources fall back to 25.0). Pass `--fps` explicitly if your source FPS
   is unreliable.
3. **Frame-name caption.** `extract_pairs._panel()` formats `f{frame_idx:04d}`
   uniformly; change `render_tile()` if you want per-corpus stem decoration.
4. **Pred filter.** `find_candidates()` only emits a frame when at least one
   `state == 'predicted'` hand is present. To get a "every YOLO box" grid,
   loosen that filter to keep frames without preds.
5. **Diverse selection.** Current code picks tiles by random shuffle then
   prunes to `--num`. The remote `make_pred_case_grids.py` uses a
   feature-space diversity greedy (`feature = [cx/W, cy/H, t/T, seq_i]`) —
   port that if you ever want a curated 9-tile "best of" rather than a
   broad 200-tile sample.

---

## Remote-118 reference paths

```
<detector-host>:<detector-project-root>/
  src/phoenix/track_hf_phoenix.py     # detect + ByteTrack on PHOENIX frames
  src/phoenix/make_pred_case_grids.py # 3x3 / 2x3 grid composite (example.png)
  src/phoenix/make_compare_videos.py  # side-by-side comparison videos
  weights/yolov8n-face-lindevs.pt
  weights/hand_yolov8n.pt

<detector-host>:<detector-data-root>/OpenASL/
  video_org/          # 50,007 sentence-cut mp4s
  data/openasl-v1.0.tsv
  data/bbox-v1.0.json
```

Conda env on 118: `<conda-env>`
(activate via `source env.sh` from the project root, sets
`PYTHONNOUSERSITE=1` + adds the env bin to PATH).
