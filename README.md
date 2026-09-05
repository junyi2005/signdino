# SignDino

**Self-Supervised Sign Language Representation Learning via Temporal-Axis Self-Distillation**

Junyi Hu, Zhewen He, Haomian Huang, Zhenghua Li, Zhifei Li, Yi Fang<sup>†</sup>
New York University Abu Dhabi · <sup>†</sup>corresponding author

[**Project page**](https://junyi2005.github.io/signdino/) · [**Paper (arXiv)**](https://junyi2005.github.io/signdino/) · [**BibTeX**](#citation)

---

SignDino moves the DINOv3 student–teacher recipe from the **spatial** domain of image
crops to the **temporal** domain of tracked sign streams. Each video is decomposed into
left-hand, right-hand and face streams by a detector-first YOLOv8n + ByteTrack pipeline; a
**frozen** DINOv3 ViT-B/16 embeds every per-frame anatomical crop; and lightweight temporal
Transformers — not the image backbone — form the student and EMA teacher, trained with
temporal DINO, frame-level iBOT, DKoleo and Gram anchoring.

Only ~30M parameters are self-supervised; the image backbone never receives a gradient.

## Results

Under the SHuBERT public-data protocol (984 source hours of YouTube-ASL):

| Task | Benchmark | Metric | SHuBERT | **SignDino** |
| --- | --- | --- | --- | --- |
| Translation | How2Sign | BLEU-4 | 16.2 | **17.9** |
| Translation | OpenASL | BLEU-4 | 23.2 | **24.5** |
| Translation | FLEURS-ASL (zero-shot) | BLEU-4 | 4.7 | **5.3** |
| ISLR | ASL Citizen | R@1 | 0.65 | **0.704** |
| ISLR | Sem-Lex | R@1 | 0.54 | **0.593** |
| ISLR | WLASL2000 | P-I | 60.90 | **68.5** |
| Fingerspelling | ASL-STEM-Wiki | mIoU | 0.40 | **0.43** |
| Phonology (16 features) | Sem-Lex / ASL Citizen | R@1 | 79.38 / 85.02 | **87.64 / 91.85** |

See the paper for the full tables, seeds, and ablations.

## Installation

```bash
git clone https://github.com/junyi2005/signdino.git
cd signdino
pip install -r requirements.txt
```

Tested with Python 3.10, PyTorch 2.6 and CUDA 12.4.

### External weights

None are bundled. Download them from their official releases and point the configs at
the local paths:

| Weight | Used by | Config key |
| --- | --- | --- |
| DINOv3 ViT-B/16 | `model/backbone.py` per-frame CLS embedder | `backbone.*` |
| ByT5-Base | `downstream/translation/model.py` decoder | `model.backbone_id` |
| BLEURT-20 | `downstream/common/metrics.py` | evaluation only |
| YOLOv8n hand / face detectors | `preprocess/yolov8_hf/yolo_bytetrack.py` | `--models-dir` |

The visual-backbone ablation additionally uses DINOv2 ViT-S/14, DINOv2 ViT-L/14,
DINOv3 ViT-S/16 and DINOv3 ViT-L/16 in the same `backbone.model_name` slot.

### External datasets

Obtain each dataset from its authors under its own terms of use.

| Dataset | Used for |
| --- | --- |
| YouTube-ASL | SSL pre-training source corpus; source-stage SLT |
| YouTube-SL-25 (ASL) | replaces the OpenASL-decontaminated portion of YouTube-ASL |
| How2Sign / OpenASL | target-stage SLT fine-tuning and test |
| FLEURS-ASL | zero-shot SLT evaluation |
| ASL Citizen / Sem-Lex / WLASL2000 | isolated sign recognition |
| ASL-STEM-Wiki | fingerspelling detection |

Following the protocol of the paper, clips intersecting the OpenASL evaluation set are
removed from YouTube-ASL and the removed duration is replaced with non-overlapping ASL
videos from YouTube-SL-25, giving ≈984 hours of unique source video.

## Quickstart

### 0. Smoke test (no data needed)

```bash
python smoke_test.py
```

Runs the full student/teacher step on synthetic tensors.

### 1. Produce the anatomical crop streams (YOLOv8n + ByteTrack)

`preprocess/yolov8_hf/yolo_bytetrack.py` is the detector-tracker used in the paper:
YOLOv8n emits per-frame hand and face boxes, ByteTrack associates hand identities
across frames and Kalman-fills detector drop-outs of at most `K` frames. In our runs it
was driven as a separate batch service; anything that writes the layout below — that
script, your own driver, or an external pipeline — is a valid input to the SSL trainer.
`preprocess/README.md` is the full contract, and `preprocess/adapt_remote_crops.py` is a
stub for converting a differently-shaped external output.

Expected on-disk layout:

```
<crop_root>/<video_id>/lh/<frame:06d>.jpg
<crop_root>/<video_id>/rh/<frame:06d>.jpg
<crop_root>/<video_id>/face/<frame:06d>.jpg
<crop_root>/<video_id>/manifest.json    per-frame validity mask + bbox per stream
<crop_root>/index.jsonl                 one line per video, with split tag
```

### 2. Build the frozen-DINOv3 embedding cache (one-off)

```bash
python preprocess/cache_embeddings.py \
    --crop-root /path/to/CROPS \
    --dinov3-repo /path/to/dinov3 \
    --batch-size 256 --num-workers 4
```

This is what makes SSL training ≈50× faster per step: the image backbone is never
evaluated inside the training loop.

### 3. Stage 1 — DINO + iBOT + DKoleo, one run per stream

```bash
python train.py --config configs/pretrain.yaml --stream lh   --exp-name pretrain_lh
python train.py --config configs/pretrain.yaml --stream rh   --exp-name pretrain_rh
python train.py --config configs/pretrain.yaml --stream face --exp-name pretrain_face
```

### 4. Stage 2 — Gram-anchoring refinement

```bash
python train.py --config configs/refine.yaml --stream lh --exp-name refine_lh \
    --init-ckpt runs/pretrain_lh/ckpt_e100.pt
```

### 5. Dump teacher features for downstream tasks

```bash
python extract_features.py --ckpt runs/refine_lh/ckpt_e30.pt \
    --crop-root /path/to/CROPS --stream lh --out-dir runs/refine_lh/features
```

### 6. Downstream

```bash
# translation: source stage on YouTube-ASL, then target stage on How2Sign
python -m downstream.translation.trainer \
    --config downstream/translation/configs/phase1_yasl.yaml   --output-dir runs/slt_src
python -m downstream.translation.trainer \
    --config downstream/translation/configs/phase2_how2sign.yaml --output-dir runs/slt_h2s
python -m downstream.translation.infer \
    --ckpt runs/slt_h2s/best.pt --manifest /path/to/how2sign_manifest.jsonl \
    --feature-root runs/refine_lh/features --split test --out-hyps runs/slt_h2s/hyps.txt

# isolated sign recognition
python -m downstream.islr.trainer \
    --config downstream/islr/configs/asl_citizen.yaml --output-dir runs/islr_aslc

# fingerspelling detection
python -m downstream.fingerspelling.trainer \
    --config downstream/fingerspelling/configs/asl_stem_wiki.yaml --output-dir runs/fs_stemwiki
```

`downstream/translation/configs/phase2_how2sign_live.yaml` is the live-fine-tune variant
(temporal encoders updated at 1/10 the decoder LR, DINOv3 still frozen);
`eval_fleurs.yaml` holds the zero-shot FLEURS-ASL evaluation settings.

## Repository layout

| Path | Purpose |
| --- | --- |
| `dino/` | modality-agnostic SSL plumbing: DINO / iBOT / DKoleo / Gram losses, EMA, schedules |
| `model/` | frozen DINOv3 frame embedder, temporal Transformer, student/teacher wiring |
| `data/` | crop I/O, embedding cache, multi-temporal-crop sampler, dataset, collate |
| `configs/` | `pretrain.yaml` (stage 1), `refine.yaml` (stage 2) |
| `preprocess/` | detector-first crop pipeline and embedding-cache builder |
| `downstream/` | translation (ByT5), ISLR (+ LoRA), fingerspelling, shared fusion + metrics |
| `make_picture/` | scripts that render the paper figures |
| `train.py` | SSL training entry point (one stream per run) |
| `extract_features.py` | dump stage-2 teacher features |
| `smoke_test.py` | synthetic end-to-end test |
| `DESIGN.md` | long-form design document |

Hyperparameters in the YAML files match the paper's hyperparameter card line by line.

## Hardware

All SSL and SLT runs in the paper use 8 × NVIDIA A100 80GB. The three per-stream encoders
are trained sequentially on the same pool: ≈240 GPU-hours per stream for stage 1 plus ≈40
for stage 2, and a one-off ≈300 GPU-hours for the embedding cache.

## Citation

```bibtex
@article{hu2026signdino,
  title   = {SignDino: Self-Supervised Sign Language Representation Learning
             via Temporal-Axis Self-Distillation},
  author  = {Hu, Junyi and He, Zhewen and Huang, Haomian and Li, Zhenghua and
             Li, Zhifei and Fang, Yi},
  journal = {arXiv preprint},
  year    = {2026}
}
```

## License

[MIT](LICENSE). The datasets and third-party checkpoints referenced above keep their own
licences.
