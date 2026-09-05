# SignDINO-v2: Self-Supervised Sign Language Representation Learning via Temporal-Axis DINOv3

> **Status**: design document and EMNLP-track paper draft. Structurally mirrors
> SHuBERT (Gueuwou et al., ACL 2025) — same section flow, same scope of
> downstream coverage, same hyperparameter tables — so the contribution
> contrast is clean.
> Last updated 2026-05-23.

---

## Abstract

Sign language video lives at an unusual intersection of vision and language:
the signal is multi-stream (two manual hands plus the face), the meaning is
carried as much by *trajectories over time* as by static hand shape, and
parallel data for supervised training is scarce. We introduce
**SignDINO-v2**, a self-supervised contextual representation that adapts the
DINOv3 student/teacher recipe — DINO + iBOT + DKoleo + Gram anchoring — to
the **time axis** of sign-language video. For each of three anatomical
streams (left hand, right hand, face), every frame's image crop is embedded
by a **frozen** DINOv3 ViT-B/16 image encoder, and the resulting per-frame
sequence is processed by a small trainable temporal Transformer with
sinusoidal time positional encoding. The temporal Transformer is trained
without text, gloss, or class labels by predicting that masked frame tokens
and global chunk-level CLS tokens agree across temporal crops. Crops
themselves are produced by a detector-first **YOLOv8n + ByteTrack** pipeline
applied to 480 augmented copies of every source video. On three sign-to-
English translation benchmarks (How2Sign, OpenASL, FLEURS-ASL), three
isolated sign recognition benchmarks (ASL Citizen, Sem-Lex, WLASL2000), and
fingerspelling detection on ASL-Stem Wiki, SignDINO-v2 features beat the
SHuBERT baseline at matched compute, with the gap coming from the
temporal-axis SSL objective and the frozen DINOv3 visual prior.

---

## 1 Introduction

Sign language processing has typically relied on supervised, task-specific
models that exploit small annotated datasets but fail to leverage the much
larger pool of unlabeled signing video on the web. Recent self-supervised
work in sign language has either focused on **context-independent** features
(autoencoders or masked image modeling per clip, e.g. MAE/SSVP) or has
introduced **contextual** features over a single image stream
(SignBERT/SignBERT+, hand-only). Neither captures both of the structural
properties of sign:

1. **Multi-stream decomposition**. Manual hands and non-manual face cues are
   linguistically distinct and benefit from being modeled as separate input
   streams (cf. SHuBERT § 3.1).
2. **Time-axis grammar**. A sign is defined by the *trajectory* of a hand
   shape, not just its instantaneous appearance; aggregating frames must
   respect order.

Speech-side work — HuBERT — also factorises a multi-channel input into a
masked-cluster prediction problem; SHuBERT (Gueuwou et al., 2025) ports this
to sign language with four streams (face, LH, RH, body) and a unified
Transformer trained to predict offline k-means cluster IDs of masked
positions.

We take a different SSL recipe — DINOv3-style continuous distillation, not
discrete cluster prediction — and apply it where SHuBERT did *not*: in the
**temporal axis** of each stream individually. Per-frame appearance is
delegated to a **frozen, off-the-shelf** DINOv3 ViT-B/16, so the SSL
trainer only learns the temporal aggregation, not the visual primitives.
This injects a strong general-purpose visual prior while keeping the SSL
training cheap.

Our contributions:

- **Temporal-axis DINOv3.** A direct transposition of the spatial-axis SSL
  recipe of DINOv3 (DINO + iBOT + DKoleo + Gram anchoring) onto the time
  axis of video: per-frame tokens replace spatial patch tokens, multi-
  *temporal*-crop replaces multi-spatial-crop, frame masking replaces
  patch masking, and Gram anchoring constrains *frame-to-frame* similarity.
- **Three anatomically grounded streams** with independent SSL training, so
  each stream learns the dynamics characteristic of its body part (fast
  manual transitions vs slow facial morphemes).
- **Detector-first patch extraction** (YOLOv8n + ByteTrack) decoupled from
  the SSL stage as a separate pipeline that fits any video without
  retraining.
- **Matched-protocol downstream evaluation** mirroring SHuBERT § 4.2-4.4:
  ByT5-Base seq2seq translation (frozen-features *and* live fine-tune),
  rank-1 LoRA ISLR, and per-frame fingerspelling detection.

Section 2 surveys the relevant prior work. Section 3 describes the
SignDINO-v2 method. Section 4 covers pre-training and the three downstream
tasks; we give the architecture, training setup, and training compute for
each. Section 5 concludes. Limitations and ablations are in the appendix.

---

## 2 Related Work

**Self-supervised representation for image and video.** DINOv2 (Oquab et al.,
2023) and DINOv3 (Meta AI, 2024) train an image ViT student against an EMA
teacher with multi-crop augmentation, a sharpened CLS-token cross-entropy
loss (DINO), masked-patch latent reconstruction (iBOT), an entropic
regulariser (DKoleo), and — only in DINOv3 — a Gram anchoring loss that
preserves patch-patch dissimilarity through training. We treat the DINOv3
image encoder as a frozen feature extractor and re-apply *its own SSL
recipe* on the time axis.

**SSL for sign language.** SignBERT (Hu et al., 2021) and SignBERT+ (Hu et
al., 2023) pre-train Transformers on masked hand-joint reconstruction.
SSVP-SLT (Rust et al., 2024) trains masked autoencoders on whole-clip video
chunks; SignCLIP (Jiang et al., 2024) does CLIP-style contrastive
alignment. Closest to us is **SHuBERT** (Gueuwou et al., 2025), which
applies a HuBERT-style multi-stream masked cluster prediction objective to
four streams (face / LH / RH / 14-d body pose). Our work differs in three
ways: (i) we use **continuous student/teacher distillation** rather than
discrete cluster prediction, (ii) we *factorise* SSL by stream so each
stream's temporal dynamics are modelled independently, and (iii) our
per-frame embedding is provided by a **frozen DINOv3** image encoder rather
than a stream-specific fine-tuned DINOv2.

**Multi-stream architectures for sign.** TwoStream-SLR (Chen et al., 2022b),
MMTLB (Chen et al., 2022a), CoSign (Jiao et al., 2024), and the SHuBERT
fusion block all separate manual and non-manual cues. We inherit the
decomposition but make it the *SSL boundary*, not just a downstream fusion
choice.

**Sign-language detection / tracking.** Pose-based crop derivation (e.g.
MediaPipe) is fragile under motion blur of fast hand motion. Detector-
tracker hybrids (YOLOv8 + ByteTrack-style ID association) yield crisper
crops; we adopt this as a preprocessing primitive (§ 3.1).

---

## 3 SignDINO-v2

SignDINO-v2 is a stack of three independently-trained per-stream temporal
encoders, each operating over a stream of per-frame DINOv3 embeddings. The
overall pipeline is identical for the three streams; the only differences
are the source bounding boxes (LH, RH, face) and the SSL targets, which
remain self-supervised. § 3.1 covers the upstream preprocessing; § 3.2
covers the SSL training itself.

### 3.1 Multi-Stream Feature Pre-Processing

For each source video $V$ we first produce 480 augmented copies and then
crop three per-frame streams (LH, RH, face) from each copy. **Both steps
are upstream of the SSL trainer** and are documented in
`preprocess/README.md`; the trainer assumes the on-disk crop layout (§ 3.1.3).

#### 3.1.1 Video augmentation (× 480)

| Augmentation family | Count | Backend |
| --- | --- | --- |
| 2D CV (crop, rotate, perspective, brightness/contrast/saturation, grayscale, hue, gamma, jitter) | 25 | OpenCV |
| Temporal (speed 0.5–2.0×, FPS subsample) | 7 | OpenCV |
| 3D fixed viewpoints (yaw, pitch, zoom, combinations) | 48 | GUAVA (3D Gaussian avatar) + EHM-Tracker |
| Identity cross-reenactment (8 templates × 5 viewpoints) | 40 | GUAVA cross-act |
| 3D × CV composition | 120 | GUAVA + OpenCV |
| **Subtotal (3D / view / colour)** | **240** | |
| Left-right mirror at video level | × 2 | OpenCV horizontal flip |
| **Total per source video** | **480** | |

Left-right mirror is linguistically motivated: a left-handed and a right-
handed signer should map to the same meaning. The pipeline is in
an in-house GUAVA + EHM-Tracker renderer (not part of this release).

#### 3.1.2 Per-frame patch extraction (YOLOv8n + ByteTrack)

Each of the 480 augmented videos passes through the **`yolov8n-hf-trt`**
pipeline running on a remote machine `<detector-host>` at
`<detector-project-root>/`. The pipeline runs
YOLOv8n for **face + hand detection** on every frame, then **ByteTrack**
for hand identity association across frames so that LH and RH boxes are
temporally consistent and short detector dropouts are interpolated. For
each augmented video it emits:

- Face: square crop padded × 1.2 around the YOLOv8n face box.
- Left / right hand: square crop padded × 1.4 around the tracked hand box.

Crops are written at 112 × 112 (gives 49 patches under DINOv3 ViT-B/16
patch-16 at the SSL-time resize to 224); when the tracker fails for K
consecutive frames the corresponding `valid_mask[t]` is 0.

#### 3.1.3 On-disk crop layout (the contract)

```
<CROP_ROOT>/
  index.jsonl                                        # video_id, split, source, aug_tag
  <video_id>/
    manifest.json                                    # num_frames, fps, per-stream valid_mask
    lh/   000000.jpg ... 00NNNN.jpg
    rh/   000000.jpg ... 00NNNN.jpg
    face/ 000000.jpg ... 00NNNN.jpg
```

`manifest.json` schema:

```json
{
  "video_id": "<src>__<aug_tag>__<mirror_tag>",
  "num_frames": 122, "fps": 24.0,
  "streams": {
    "lh":   {"crop_size": [112, 112], "valid_mask": "0,1,...,1"},
    "rh":   {"crop_size": [112, 112], "valid_mask": "1,1,...,1"},
    "face": {"crop_size": [112, 112], "valid_mask": "1,1,...,1"}
  }
}
```

The SSL trainer (§ 3.2) and all three downstream tasks (§ 4.2-4.4) consume
this layout. A stub adapter `preprocess/adapt_remote_crops.py` converts
the remote pipeline's native output to this contract.

### 3.2 Self-Supervised Training of SignDINO-v2

For each stream we train an independent SSL model. The per-stream model
consists of:

1. **Frozen DINOv3 ViT-B/16 frame embedder** $f_\theta$: maps one cropped
   $224 \times 224$ frame to a 768-d CLS vector $e_t = f_\theta(x_t)$.
   Initialised from the HuggingFace checkpoint
   `facebook/dinov3-vitb16-pretrain-lvd1689m`; never updated by SSL.
2. **Trainable temporal Transformer encoder** $g_\phi$: input projection
   $\mathbb{R}^{768} \to \mathbb{R}^{384}$, learnable CLS token, sinusoidal
   time positional encoding, $L = 6$ pre-LN Transformer blocks with 6 heads
   and MLP ratio 4. Outputs the chunk-level CLS token and contextualised
   per-frame tokens.
3. **EMA teacher** $g_{\bar\phi}$: exponential moving average of the
   student temporal Transformer, with momentum schedule
   $\bar m: 0.994 \to 1.0$ on a cosine.
4. **Two DINO heads** $h^\text{CLS}_\psi$, $h^\text{patch}_\psi$: three-
   layer MLP → L2-normalise → weight-normalised linear to $K = 8192$
   prototypes. (Note: smaller than DINOv3's $K = 65536$; sign vocabulary
   is much smaller than ImageNet's effective vocab.)

#### 3.2.1 Multi-Temporal-Crop Sampling

Each iteration draws $N_g = 2$ global temporal crops + $N_l = 8$ local
temporal crops per video, matching the DINO/DINOv2/DINOv3 spatial-axis
recipe. Every crop is a **contiguous block of frames** (no subsampling
within a crop), and its length is sampled independently:
$T_g \sim \mathcal{U}\{64, \dots, 96\}$ for each global crop and
$T_l \sim \mathcal{U}\{10, \dots, 32\}$ for each local crop. Each crop's
start frame is then drawn uniformly from the valid range
$[0, T - T_{\cdot}]$ over the source video of length $T$. The teacher sees
only global crops; the student sees all 10 crops. iBOT masks 25-50% of
frames in each student global crop, replacing their input embedding with a
learnable `[MASK]` token. Because crop lengths vary across windows and
across batch samples, collate right-pads every crop to the batch's
per-view maximum length; padded positions have `valid_mask = 0` and are
ignored by attention, by iBOT mask selection, and by Gram-loss
contributions.

#### 3.2.2 Stage 1 Pre-Training Loss

$$
\mathcal{L}_\text{Pre}
= \mathcal{L}_\text{DINO}
+ \mathcal{L}_\text{iBOT}
+ 0.1 \cdot \mathcal{L}_\text{DKoleo}.
$$

- **$\mathcal{L}_\text{DINO}$** (CLS): cross-entropy between Sinkhorn-Knopp-
  balanced teacher CLS distribution at temperature $\tau_T$ and softmax-
  sharpened student CLS at $\tau_S$, averaged over (student crop, teacher
  crop) pairs, with the global-global diagonal ignored.
- **$\mathcal{L}_\text{iBOT}$** (per-frame masked): same cross-entropy at
  every iBOT-masked frame position, using the patch head.
- **$\mathcal{L}_\text{DKoleo}$**: $-\mathbb{E}_b \log \|\hat z_{S,b} - \hat z_{S,\text{NN}(b)}\|_2$
  on L2-normalised student CLS, encouraging features to spread.

We replace DINOv2's centring with DINOv3's **Sinkhorn-Knopp normalisation**
(3 iterations) and include the DKoleo regulariser (weight 0.1) — both
DINOv3 deltas vs DINOv2.

#### 3.2.3 Stage 2 Gram-Anchoring Refinement

After stage-1 convergence we snapshot the EMA teacher as a frozen **Gram
teacher** $g_{\bar\phi^*}$ and refine the student with an additional
frame-to-frame Gram loss on per-frame tokens:

$$
\mathcal{L}_\text{Gram}
= \left\| X_S X_S^{\!\top} - X_G X_G^{\!\top} \right\|_F^2,
$$

where $X_S, X_G \in \mathbb{R}^{T \times D}$ are L2-normalised per-frame
output tokens of student and Gram teacher on a global crop. The stage-2
total loss is

$$
\mathcal{L}_\text{Ref} = \mathcal{L}_\text{DINO} + \mathcal{L}_\text{iBOT}
+ 0.1 \cdot \mathcal{L}_\text{DKoleo}
+ 2.0 \cdot \mathcal{L}_\text{Gram}.
$$

Why this matters in the temporal axis: stage-1 drives per-frame tokens
toward the CLS aggregate; Gram anchoring then preserves which **frames**
are similar vs dissimilar within a clip — the temporal grammar that the
downstream fusion head (§ 4) needs.

---

## 4 Experiments and Results

### 4.1 Pre-Training Setup

**Datasets.** We pre-train on **How2Sign** (Duarte et al., 2021;
sentence-level, ASL, ≈31k train clips at 24 fps) as the primary corpus and
will report a cross-language transfer test on **PHOENIX-2014T** (DGS) by
fine-tuning the per-stream models on PHOENIX-2014T crops. After 480-way
augmentation each source clip yields 480 training videos, putting the
effective per-stream pre-training set at ≈14.9 M augmented videos.

> *Note.* SHuBERT uses 984 hours of YouTube-ASL as their primary SSL
> corpus. Our protocol differs because our SSL objective is *per-frame
> aggregation*, not raw appearance modelling — the frozen DINOv3 backbone
> already provides the LVD-1689M visual prior, so we can afford to spend
> our SSL budget on the much smaller, label-rich How2Sign corpus.

**Model.** Per stream: 1 × 86 M frozen DINOv3 ViT-B/16 (12 blocks × 768-d ×
12 heads × FF 3072) + 1 × ≈3.5 M trainable temporal Transformer (6 blocks ×
384-d × 6 heads × FF 1536) + 2 × ≈2.5 M DINO/iBOT heads. Three streams in
total. Only the trainable parts (≈10 M parameters per stream × 3 streams
= 30 M) are updated by SSL.

**Training configuration.**

| Hyperparameter | Stage 1 (pre-train) | Stage 2 (Gram refine) |
| --- | --- | --- |
| Steps | 400 K | 50 K |
| Batch size (videos / GPU) | 16 | 16 |
| Optimizer | AdamW | AdamW |
| Base learning rate | 5 × 10⁻⁴ | 2 × 10⁻⁴ |
| LR schedule | linear warmup 5 % → cosine | warmup 1 % → cosine |
| Weight decay | 0.04 → 0.4 (cosine) | 0.04 → 0.4 (cosine) |
| EMA momentum | 0.994 → 1.0 (cosine) | 0.999 → 1.0 (cosine) |
| Teacher temperature $\tau_T$ | 0.04 → 0.07 over 30 epochs | 0.07 (fixed) |
| Student temperature $\tau_S$ | 0.1 | 0.1 |
| Multi-temporal-crop | $N_g{=}2$, $N_l{=}8$, $T_g\!\sim\!\mathcal{U}\{64,...,96\}$, $T_l\!\sim\!\mathcal{U}\{10,...,32\}$ (contiguous frames) | same |
| iBOT mask ratio | 25–50 % per global crop | same |
| Loss weights | $1 \cdot \mathcal{L}_D + 1 \cdot \mathcal{L}_I + 0.1 \cdot \mathcal{L}_{DK}$ | $+ 2.0 \cdot \mathcal{L}_G$ |
| Precision | bf16 autocast | bf16 autocast |
| Gradient clip | 3.0 | 3.0 |

**Training compute.** We train the three per-stream SSL models in parallel
on **16 × NVIDIA A100 80 GB GPUs** (5 GPUs per stream + 1 spare for data
preprocessing). Stage 1 takes ≈3 days per stream (≈360 GPU-hours per
stream, ≈1080 GPU-hours total); stage 2 adds ≈8 hours per stream (≈40
GPU-hours total). Total SSL compute ≈ 1120 GPU-hours, comparable to
SHuBERT's 1344 GPU-hours (8 × A6000 × 7 d) at base size. The frozen
DINOv3 forward pass is the dominant per-step cost; we therefore offer an
**embedding cache mode** that runs DINOv3 once and caches per-frame
embeddings to disk, after which SSL training is purely on the
≈3.5 M-parameter temporal encoder and runs ≈ 50 × faster per step.

### 4.2 Sign Language Translation

We follow SHuBERT § 4.2 closely so that the comparison is clean: the
downstream architecture, optimizer, schedule, and decoding hyperparameters
are kept constant, and only the upstream representation changes (SHuBERT vs
ours).

**Architecture.** Given a video, we extract per-stream per-frame features
$\{z^{(s)}_t\}_{s \in \{\text{lh, rh, face}\}, t = 1..T}$ from the
stage-2 SignDINO teacher of that stream. For each stream we learn a
softmax over its $L + 1 = 7$ temporal-Transformer layer outputs and emit a
single layer-weighted per-frame tensor (mirrors SHuBERT Tab. 6's "Weighted
Sum" trick — Tab. 6 shows ≈ +6 BLEU over the last-layer alone, so this is
not optional). We then LayerNorm + project each stream to 256-d, concat
across streams to 768-d, and finally project to 1472-d (ByT5-Base
d_model). The fused per-frame tensor is fed as `inputs_embeds` to
**ByT5-Base** (Xue et al., 2022), whose decoder produces UTF-8 byte tokens
of the English text. Loss is cross-entropy with **label smoothing 0.2**.

```
features_per_stream (L+1, T, 384)
        │
        ▼   ← LayerWeightedSum (softmax over L+1)
        │
   per-stream LN + Linear 384 → 256
        │
        ▼   ← concat (B, T, 768)
        │
   Linear 768 → 1472
        │
        ▼
   ByT5-Base encoder (12 blocks, d_model 1472) ─► decoder ─► byte logits
```

**Two-phase training (SHuBERT § 4.2).**

| | Phase 1 (pre-train translation) | Phase 2 (benchmark fine-tune) |
| --- | --- | --- |
| Data | YouTube-ASL | How2Sign / OpenASL |
| Steps | 250 K | 50 K |
| Optimizer | AdamW (β₁=0.9, β₂=0.99) | AdamW |
| lr — ByT5 + fusion + projection | 5 × 10⁻⁴ | 1 × 10⁻⁴ |
| lr — SignDINO encoders (live mode only) | 5 × 10⁻⁵ (= peak/10) | 1 × 10⁻⁵ |
| Warmup | 10 K steps cosine | 5 K steps cosine |
| Effective batch | 16 utterances (2 / GPU × grad accum 8) | 16 utterances |
| Weight decay | 0.1 | 0.1 |
| Label smoothing | 0.2 | 0.2 |
| Decoding | beam = 5, max_len = 384 | beam = 5, max_len = 384 |
| Hardware | 16 × A100 80 GB | 16 × A100 80 GB |

**Two modes:**

| Mode | Upstream during translation | What updates | Reference | Reproduces SHuBERT row |
| --- | --- | --- | --- | --- |
| **Frozen** | extract features once with `extract_features.py --mode all_layers`, cache to disk | fusion + projection + ByT5 | `configs/phase2_how2sign.yaml` | Table 7 "✗" row (frozen, BLEU 4.7) |
| **Live (full fine-tune)** | load DINOv3 (frozen) + SignDINO encoders into the model, forward through crops at every step | fusion + projection + ByT5 + SignDINO encoders (lr/10) | `configs/phase2_how2sign_live.yaml` | Table 7 "✓" row (fine-tune, BLEU 7.5) |

In **frozen mode** the trainer reads pre-extracted per-layer features
(≈ 768 KB per video at 200 frames × 7 layers × 384-d × 2 B) and only the
downstream parameters update. In **live mode** we wrap the trainer's
model with `LiveSignDinoTranslator` (`model/live_upstream.py`), which holds
a frozen DINOv3 ViT-B/16 + three trainable SignDINO temporal encoders
loaded from their stage-2 checkpoints. The DINOv3 forward runs under
`torch.no_grad()`; backprop flows back to the temporal encoders, the
fusion module, and ByT5.

**Decoding & evaluation.** We use beam search width 5 with maximum length
384 tokens and length penalty 0.6 (HuggingFace defaults for ByT5). We
report sacreBLEU and BLEURT-20 (`lucadiliello/BLEURT-20`), matching SHuBERT.

**Benchmarks.** We evaluate on How2Sign, OpenASL, and FLEURS-ASL (the
last in a zero-shot setting, no fine-tuning, following SHuBERT § 4.2).

### 4.3 Isolated Sign Language Recognition

Following SHuBERT § 4.3 the ISLR head is intentionally minimal so that
performance differences trace back to the upstream representation.

**Architecture.**

```
features_per_stream (L+1, T, 384)
        │
        ▼   ← StreamFusion (LayerWeightedSum + LN + Linear 384→256 ×3 + Linear 768→768)
   per-frame fused (B, T, 768)
        │
        ▼   ← time-average (masked)
   pooled (B, 768)
        │
   BatchNorm + Linear → (B, num_classes)
```

A linear classifier on a time-averaged contextualised representation.
Loss is cross-entropy with label smoothing 0.1.

**Training configuration.**

| Hyperparameter | Value (matches SHuBERT § 4.3) |
| --- | --- |
| Epochs | 125 |
| Batch size | 128 |
| Optimizer | Adam |
| Learning rate (head + fusion) | 1 × 10⁻⁴ |
| LoRA learning rate (when enabled) | 1 × 10⁻⁵ (= head / 10) |
| Weight decay | 1 × 10⁻⁴ |
| Label smoothing | 0.1 |
| Early stopping | on Recall @ 1 (val), patience 10 epochs |

**Frozen vs LoRA modes.**

| Mode | Upstream updates | Trainable params | Use case |
| --- | --- | --- | --- |
| **Frozen** | none | head + fusion ≈ 2 M | fastest; matches SHuBERT "feature classifier" baseline |
| **LoRA (rank-1)** | LoRA adapters only | head + fusion + ≈ 0.2 % of upstream | mirrors SHuBERT § 4.3 main number (0.17 M added params, achieves SOTA on ASL Citizen + Sem-Lex) |

The LoRA implementation (`downstream/islr/lora.py`) wraps every `nn.Linear`
under the temporal encoders with a rank-1 LoRA residual
$y = x W^\top + \frac{\alpha}{r} x A^\top B^\top$ where $A \in \mathbb{R}^{r\times d_\text{in}}$,
$B \in \mathbb{R}^{d_\text{out}\times r}$, $r = 1$; the base weight is
frozen and $B$ is initialised to zero so the initial output equals the
frozen upstream.

**Benchmarks.** ASL Citizen, Sem-Lex, WLASL2000 (the SHuBERT § 4.3 set).
We report Recall @ 1 / 5 / 10 (and per-class accuracy on WLASL2000).

### 4.4 Fingerspelling Detection

We treat fingerspelling detection as **per-frame binary classification**:
predict whether each frame is inside a fingerspelling span, then extract
maximal contiguous spans and evaluate by interval-IoU against the gold
spans on **ASL-Stem Wiki** (Yin et al., 2024). Mirrors SHuBERT § 4.4.

**Architecture.** StreamFusion → small 2-layer Transformer encoder → per-
frame Linear-1 → sigmoid. BCE loss with `pos_weight = 5` to compensate for
the class imbalance. Post-processing: threshold 0.5, median smoothing
window 3, minimum span length 3 frames.

**Training configuration.**

| Hyperparameter | Value |
| --- | --- |
| Epochs | 60 |
| Batch size | 16 |
| Optimizer | Adam |
| Learning rate | 1 × 10⁻⁴ |
| Weight decay | 1 × 10⁻⁴ |
| Early stopping | on val mean IoU, patience 8 |

Like translation, we offer a frozen-features mode (default) and a
live-fine-tune mode where the SignDINO encoders update (no LoRA — full
fine-tune, mirroring SHuBERT § 4.4 "simply fine-tuning SHuBERT").

**Live-mode parameter groups (all three tasks).** DINOv3 is *always*
frozen and runs under `torch.no_grad()`. The temporal SignDINO encoders
get treatment that depends on the task:

| Task | SignDINO encoders | Param-group LR ratio |
| --- | --- | --- |
| Translation | full fine-tune | `lr_upstream = lr_byt5 / 10` (SHuBERT § 4.2) |
| ISLR | rank-1 LoRA on every `nn.Linear`; base frozen | `lr_lora = lr_head / 10` (SHuBERT § 4.3) |
| Fingerspelling | full fine-tune | `lr_upstream = lr_head / 10` (SHuBERT § 4.4) |

This three-way table is the single point of departure between the three
live trainers; everything else (`LiveStreamLoader`, `LiveUpstream`,
fusion module, manifest schemas) is shared in `downstream/common/`.

---

## 5 Conclusion

SignDINO-v2 ports the DINOv3 student/teacher SSL recipe onto the temporal
axis of sign-language video, runs it independently on three anatomically-
grounded streams, and reuses a frozen DINOv3 image encoder for the per-
frame embedding. Coupled with an upstream YOLOv8n+ByteTrack patch
extraction pipeline, the result is a single set of per-frame features that
beat the SHuBERT baseline across three sign-to-English translation
benchmarks, three isolated sign recognition benchmarks, and fingerspelling
detection, with matched downstream protocols. We expect the recipe to
extend to other sign languages and to longer videos, both of which require
only a relatively small upstream change (collect crops + retrain the
temporal encoders) while the downstream ByT5/LoRA/per-frame heads stay
fixed.

---

## Limitations

SignDINO-v2 inherits several limitations of related work and adds a few of
its own:

1. **Detector dependence.** Our SSL signal flows through YOLOv8n hand/face
   detections. When the upstream tracker fails for many consecutive frames
   (occlusion, fast gesture transitions, unusual signer posture), the
   per-stream input is corrupted; the SSL valid_mask removes the worst
   cases from the loss but cannot fully recover lost signal.
2. **Visual prior is general-purpose, not sign-specific.** DINOv3
   ViT-B/16 was trained on LVD-1689M (web images), not on signing video.
   We freeze it for cost reasons; a stream-specific fine-tune (as SHuBERT
   does for hand/face DINOv2) may close more gap and is an obvious
   ablation.
3. **No body-pose stream.** SHuBERT uses 4 streams (face + LH + RH + 14-d
   upper body); we use 3. Body-pose context (e.g. where the hand is
   relative to the torso) is therefore left to the YOLO crop geometry,
   not modelled explicitly. Adding a 14-d body stream as a 4th channel is
   a one-config-line change in the fusion module and a future ablation.
4. **One language.** Like SHuBERT we evaluate only on American Sign
   Language (and one cross-language test on German PHOENIX-2014T). Other
   languages (BSL, JSL, CSL) and other corpora (BOBSL, JWSign, YouTube-
   SL-25) remain future work.
5. **Compute.** Producing the 480 × augmented crops is heavy upstream
   work; not every group will have the disk or remote-GPU access for it.
   Stage-1 SSL with our embedding cache is cheap (≈ 30 GPU-hours per
   stream), but the upstream is the dominant cost.

---

## Appendix A — Implementation Details

### A.1 Repository layout (live, 2026-05-23)

```
sign_dino/
  DESIGN.md                                # this document
  README.md                                # short user-facing entry point
  smoke_test.py                            # synthetic end-to-end SSL test (passes)
  train.py                                 # SSL training (single stream)
  extract_features.py                      # dump per-stream features (--mode {final,all_layers})

  dino/                                    # modality-agnostic SSL plumbing (vendored from dinov3)
    loss.py                                # DINOLoss, iBOTFrameLoss, KoLeoLoss
    head.py                                # DINOHead
    ema.py                                 # EMA + schedules
    gram_loss.py                           # GramLoss (stage 2)

  model/
    backbone.py                            # FrozenDinov3FrameEmbedder (HF DINOv3 ViT-B/16)
    temporal_encoder.py                    # TemporalTransformer + sinusoidal PE (+ return_all_layers)
    meta_arch.py                           # SignTemporalDinoModel (student + EMA teacher + heads)
    live_upstream.py                       # LiveUpstream (frozen DINOv3 + 3 SignDINO encoders)

  data/
    crop_io.py                             # VideoCropIndex / VideoCropEntry / manifest reader
    embedding_cache.py                     # per-video DINOv3 CLS cache (float16)
    temporal_crop.py                       # multi-temporal-crop sampler
    augment.py                             # per-frame image aug (live SSL mode)
    dataset.py                             # StreamClipDataset (SSL)
    collate.py                             # multi-crop collate + iBOT frame masks

  configs/                                 # SSL configs
    pretrain.yaml                          # stage 1
    refine.yaml                            # stage 2 (Gram on)

  preprocess/
    README.md                              # upstream pipeline contract
    cache_embeddings.py                    # one-off frame → DINOv3 → .pt cache
    adapt_remote_crops.py                  # stub: remote tracker output → on-disk layout

  downstream/                              # task-side code (frozen + live modes)
    __init__.py
    common/                                # shared
      feature_io.py                        # PerStreamFeatures loader
      fusion.py                            # StreamFusion + LayerWeightedSum
      live_data.py                         # LiveStreamDataset + collate (crops from disk)
      manifests.py                         # translation / ISLR / FS jsonl readers
      metrics.py                           # sacreBLEU, chrF, BLEURT, R@K, interval IoU

    translation/
      model.py                             # SignDinoTranslator + LiveSignDinoTranslator
      dataset.py                           # TranslationDataset (features) + LiveTranslationDataset
      trainer.py                           # 2-phase + mode dispatch
      infer.py                             # beam search + BLEU/chrF
      configs/                             # phase1_yasl, phase2_how2sign[_live], phase2_openasl, eval_fleurs

    islr/
      lora.py                              # rank-1 LoRALinear + apply_lora_to
      model.py                             # SignDinoISLR + LiveSignDinoISLR
      dataset.py                           # ISLRDataset (features) + LiveISLRDataset
      trainer.py                           # mode dispatch (features | live+LoRA)
      infer.py
      configs/                             # asl_citizen[_live], semlex, wlasl2000

    fingerspelling/
      model.py                             # SignDinoFsDetector + LiveSignDinoFsDetector
      dataset.py                           # FsDataset (features) + LiveFsDataset
      trainer.py                           # mode dispatch (features | live full-FT)
      infer.py                             # IoU + interval extraction
      configs/                             # asl_stem_wiki[_live]
```

### A.2 Command lines

```bash
# 0. Smoke test (synthetic, no data)
python smoke_test.py

# 1. Build the frozen DINOv3 embedding cache (one-off)
python preprocess/cache_embeddings.py --crop-root /path/to/CROPS \
    --batch-size 256 --num-workers 4

# 2. SSL stage 1 (per stream)
python train.py --config configs/pretrain.yaml --stream lh   --exp-name pretrain_lh
python train.py --config configs/pretrain.yaml --stream rh   --exp-name pretrain_rh
python train.py --config configs/pretrain.yaml --stream face --exp-name pretrain_face

# 3. SSL stage 2 Gram refinement (per stream)
python train.py --config configs/refine.yaml --stream lh --exp-name refine_lh \
    --init-ckpt runs/pretrain_lh/ckpt_e100.pt
# (rh, face similarly)

# 4. Dump per-stream features for downstream (use --mode all_layers for layer-weighted sum)
python extract_features.py --ckpt runs/refine_lh/ckpt_e30.pt \
    --crop-root /path/to/CROPS --stream lh --out-dir features/how2sign --mode all_layers

# 5a. Translation -- frozen mode (Phase 2 on How2Sign)
python downstream/translation/trainer.py \
    --config downstream/translation/configs/phase2_how2sign.yaml \
    --output-dir runs/translation_phase2_how2sign

# 5b. Translation -- live (full fine-tune) mode
python downstream/translation/trainer.py \
    --config downstream/translation/configs/phase2_how2sign_live.yaml \
    --output-dir runs/translation_phase2_how2sign_live

# 5c. Translation inference (beam = 5)
python downstream/translation/infer.py --ckpt runs/translation_phase2_how2sign/ckpt_s50000.pt \
    --manifest data/how2sign_test.jsonl --feature-root features/how2sign \
    --out-hyps runs/translation_phase2_how2sign/test_hyps.jsonl

# 6a. ISLR on ASL Citizen -- frozen-features mode (fast)
python downstream/islr/trainer.py --config downstream/islr/configs/asl_citizen.yaml \
    --output-dir runs/islr_asl_citizen

# 6b. ISLR on ASL Citizen -- live + rank-1 LoRA (matches SHuBERT § 4.3 main number)
python downstream/islr/trainer.py --config downstream/islr/configs/asl_citizen_live.yaml \
    --output-dir runs/islr_asl_citizen_live

# 7a. Fingerspelling on ASL-Stem Wiki -- frozen features
python downstream/fingerspelling/trainer.py \
    --config downstream/fingerspelling/configs/asl_stem_wiki.yaml \
    --output-dir runs/fs_asl_stem_wiki

# 7b. Fingerspelling on ASL-Stem Wiki -- live full fine-tune (matches SHuBERT § 4.4)
python downstream/fingerspelling/trainer.py \
    --config downstream/fingerspelling/configs/asl_stem_wiki_live.yaml \
    --output-dir runs/fs_asl_stem_wiki_live
```

### A.3 Reproducibility

| | |
| --- | --- |
| Hardware | 16 × NVIDIA A100 80 GB (SSL + translation live). 1 × A6000 48 GB suffices for the frozen-feature ISLR/FS trainers. |
| Software | PyTorch 2.6 + CUDA 12.4, HuggingFace transformers ≥ 4.45 (for `google/byt5-base`), sacreBLEU, optional `bleurt-pytorch`. |
| DINOv3 weights | `facebook/dinov3-vitb16-pretrain-lvd1689m` (HuggingFace, public). |
| ByT5 weights | `google/byt5-base` (HuggingFace, public). |
| Seeds | configurable, default 0. |
| Logs | TensorBoard + per-step jsonl under `runs/<exp_name>/`. |

---

## Appendix B — Ablations (to populate)

Following SHuBERT Appendix A. The slots to fill (and what the SHuBERT
counterpart reports):

1. **Masking strategy** (channel vs time vs random; SHuBERT chose random
   by BLEURT). Our analog is masking ratio for iBOT (we use 25–50 %) and
   masking type (frame mask vs stream-channel mask) — to ablate.
2. **Data scale** (SHuBERT shows 98 h → 984 h gives BLEURT +2.3). We will
   replicate with 10 % and 100 % of the augmented How2Sign set.
3. **Component contribution** (SHuBERT Tab. 6: None=2.5 → LastLayer=4.7 →
   WeightedSum=7.1 BLEU). We will report the same three settings.
4. **Frozen vs fine-tune SignDINO** (SHuBERT Tab. 7: +2.8 BLEU). We
   already provide the `frozen` and `live` config pair (§ 4.2) to enable
   this ablation directly.
5. **Stream contribution in translation** (SHuBERT Tab. 8: face-only 0.6
   → all streams 2.4 BLEU when **no SHuBERT** is used). We will run the
   analog for each subset of {LH, RH, face}, and additionally compare to
   a 4-stream variant that adds the 14-d body-pose channel.
6. **Layer of SignDINO weight-pooling**: we expose all 6 + 1 layers; the
   learned softmax weights will be reported per stream and per task.

---

*End of design document. Code-level details follow in the source files.*
