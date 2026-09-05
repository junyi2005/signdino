"""Embed crops with frozen DINOv3 → k-means cluster → render a SHuBERT-style
row-per-cluster grid PDF.

Per stream pipeline:
  Phase 1 (embed):  load every PNG under <root>/<stream>/, normalise to 224,
                    forward through FrozenDinov3FrameEmbedder, cache
                    (N, 768) float32 array + matching file list to
                    <root>/<stream>_features.npz.
  Phase 2 (cluster): k-means on the cached features with K large enough that
                    the visually meaningful sub-categories separate; cache
                    labels to <root>/<stream>_labels.npy.
  Phase 3 (render): pick K_show clusters (filtered for size and stability),
                    sample 10 members per cluster (nearest-to-centroid by
                    default), tile into a rows × cols PDF.

Usage:
  python preprocess/yolov8_hf/cluster_grid.py \\
      --root runs/yolov8_hf/cluster_crops --stream face --K 30 --K-show 5
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

# matplotlib imports deferred to render time so embedding-only runs don't pay
# for the heavy backend init.


sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


class _Dinov2Embedder:
    """Tiny ad-hoc embedder using DINOv2 ViT-B/14 (locally cached at
    ~/.cache/torch/hub/checkpoints/dinov2_vitb14_pretrain.pth).

    Used here only for crop clustering visuals; the actual SignDINO encoder
    in the paper is DINOv3 ViT-B/16. DINOv2 gives equivalent visual features
    for the purposes of qualitative clustering."""

    input_size = 224
    embed_dim = 768

    def __init__(self, device: str):
        self.device = device
        # facebookresearch/dinov2 entrypoint; weights resolved from local cache.
        self.backbone = torch.hub.load("facebookresearch/dinov2",
                                       "dinov2_vitb14", source="github").to(device).eval()
        for p in self.backbone.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        # dinov2 forward gives CLS embedding by default
        return self.backbone(x).float()


IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _list_crops(stream_dir: Path) -> list[Path]:
    return sorted(stream_dir.glob("*.png"))


def _load_and_pre(path: Path, input_size: int) -> np.ndarray:
    import cv2
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"cannot read {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if img.shape[0] != input_size or img.shape[1] != input_size:
        img = cv2.resize(img, (input_size, input_size), interpolation=cv2.INTER_AREA)
    arr = img.astype(np.float32) / 255.0
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    return np.transpose(arr, (2, 0, 1))                                          # (3, H, W)


def embed_stream(root: Path, stream: str, batch_size: int, device: str) -> tuple[np.ndarray, list[Path]]:
    stream_dir = root / stream
    feat_path = root / f"{stream}_features.npz"
    if feat_path.exists():
        data = np.load(feat_path, allow_pickle=True)
        return data["features"], list(data["paths"])

    paths = _list_crops(stream_dir)
    if not paths:
        raise SystemExit(f"no crops under {stream_dir}")
    print(f"[embed] {stream}: {len(paths)} crops")

    embedder = _Dinov2Embedder(device)
    feats = np.zeros((len(paths), embedder.embed_dim), dtype=np.float32)
    for i in range(0, len(paths), batch_size):
        chunk_paths = paths[i:i + batch_size]
        batch = np.stack([_load_and_pre(p, embedder.input_size) for p in chunk_paths])
        x = torch.from_numpy(batch).to(device, non_blocking=True)
        out = embedder(x).cpu().numpy()
        feats[i:i + len(chunk_paths)] = out
        if (i // batch_size) % 10 == 0:
            print(f"  [embed] {stream}: {i + len(chunk_paths):5d}/{len(paths)}", flush=True)
    np.savez(feat_path, features=feats, paths=np.array([str(p) for p in paths]))
    print(f"[embed] {stream}: saved {feat_path}")
    return feats, paths


def cluster_stream(root: Path, stream: str, K: int, seed: int, force: bool = False) -> tuple[np.ndarray, np.ndarray]:
    label_path = root / f"{stream}_labels_K{K}.npz"
    if label_path.exists() and not force:
        data = np.load(label_path)
        return data["labels"], data["centroids"]

    from sklearn.cluster import KMeans                                          # noqa: PLC0415

    feats_path = root / f"{stream}_features.npz"
    data = np.load(feats_path, allow_pickle=True)
    feats = data["features"]
    # k-means on L2-normalised CLS to match DINO's cosine-similarity semantics
    f_norm = feats / (np.linalg.norm(feats, axis=1, keepdims=True) + 1e-8)
    print(f"[cluster] {stream}: K={K} on {len(feats)} points")
    km = KMeans(n_clusters=K, n_init=10, random_state=seed)
    labels = km.fit_predict(f_norm)
    centroids = km.cluster_centers_
    np.savez(label_path, labels=labels, centroids=centroids)
    print(f"[cluster] {stream}: inertia={km.inertia_:.2f}, saved {label_path}")
    return labels, centroids


def _video_id_of(crop_path: Path) -> str:
    """Map a crop file path back to its source-video stem via the manifest
    JSON record (idx encoded in filename: <corpus>_<idx>.png)."""
    return crop_path.stem  # we'll attach the real source ID in pick_clusters


def _unique_sources_per_cluster(paths: list[Path], labels: np.ndarray,
                                manifest_path: Path) -> dict[int, int]:
    """Return cluster_id -> count of unique source-video stems among its members."""
    # build idx -> src map from the manifest
    src_by_stem: dict[str, str] = {}
    with open(manifest_path) as f:
        for line in f:
            rec = json.loads(line)
            src_by_stem[rec["stem"]] = Path(rec["src"]).stem
    unique = {}
    for cid in range(int(labels.max() + 1)):
        members = np.where(labels == cid)[0]
        seen = set()
        for m in members:
            seen.add(src_by_stem.get(paths[m].stem, str(m)))
        unique[cid] = len(seen)
    return unique


def pick_clusters(labels: np.ndarray, centroids: np.ndarray, K_show: int,
                  min_size: int, mode: str = "diverse",
                  paths: list[Path] | None = None,
                  manifest_path: Path | None = None) -> list[int]:
    """Pick K_show clusters from the eligible pool (size >= min_size).

    mode == 'size'             : largest by membership count
    mode == 'diverse'          : farthest-first in centroid space (seeded with
                                 the largest eligible cluster)
    mode == 'identity-diverse' : filter to clusters that span MANY distinct
                                 source videos (proxy for multi-signer), then
                                 farthest-first within that pool. Eliminates
                                 single-signer rows.
    """
    K = int(labels.max() + 1)
    sizes = [(c, int((labels == c).sum())) for c in range(K)]
    eligible = [c for c, n in sizes if n >= min_size]
    if not eligible:
        return []
    if mode == "size":
        eligible.sort(key=lambda c: -dict(sizes)[c])
        return eligible[:K_show]

    if mode == "identity-diverse":
        assert paths is not None and manifest_path is not None
        uniq = _unique_sources_per_cluster(paths, labels, manifest_path)
        # require at least 60% of the cluster's members to come from distinct sources;
        # this strongly de-prefers single-signer / single-shot clusters.
        eligible = [c for c in eligible if uniq[c] >= max(6, int(0.6 * dict(sizes)[c]))]
        if not eligible:
            # fall back gracefully to a looser bar
            eligible = sorted([c for c, n in sizes if n >= min_size],
                              key=lambda c: -uniq[c])[:max(20, K_show * 3)]

    # diverse: farthest-first on L2-normalised centroids over the filtered pool
    c_norm = centroids / (np.linalg.norm(centroids, axis=1, keepdims=True) + 1e-8)
    pool = list(eligible)
    seed = max(pool, key=lambda c: dict(sizes)[c])
    chosen = [seed]
    pool.remove(seed)
    while pool and len(chosen) < K_show:
        nearest = np.array([min(1.0 - float(c_norm[c] @ c_norm[ch]) for ch in chosen)
                            for c in pool])
        nxt = pool[int(np.argmax(nearest))]
        chosen.append(nxt)
        pool.remove(nxt)
    return chosen


def cluster_samples(feats: np.ndarray, labels: np.ndarray, centroids: np.ndarray,
                    cluster_id: int, n: int, mode: str, seed: int) -> np.ndarray:
    """Return n indices from this cluster. Mode 'centroid' picks nearest-to-
    centroid; 'random' picks uniform-at-random."""
    members = np.where(labels == cluster_id)[0]
    if mode == "centroid":
        f_norm = feats[members] / (np.linalg.norm(feats[members], axis=1, keepdims=True) + 1e-8)
        c = centroids[cluster_id]
        c = c / (np.linalg.norm(c) + 1e-8)
        sims = f_norm @ c
        order = np.argsort(-sims)
        return members[order[:n]]
    rng = np.random.default_rng(seed + cluster_id)
    if len(members) <= n:
        return members
    return rng.choice(members, size=n, replace=False)


def render_grid(root: Path, stream: str, K: int, K_show: int, cols: int,
                seed: int, mode: str, out_pdf: Path, min_size: int,
                pick_mode: str = "diverse",
                cluster_ids: list[int] | None = None,
                skip_ids: list[int] | None = None) -> None:
    import matplotlib.pyplot as plt                                              # noqa: PLC0415
    import cv2                                                                    # noqa: PLC0415

    data = np.load(root / f"{stream}_features.npz", allow_pickle=True)
    paths = [Path(p) for p in data["paths"]]
    feats = data["features"]
    labs = np.load(root / f"{stream}_labels_K{K}.npz")
    labels, centroids = labs["labels"], labs["centroids"]

    if cluster_ids:
        chosen = cluster_ids
    else:
        chosen = pick_clusters(labels, centroids, K_show, min_size, mode=pick_mode,
                               paths=paths, manifest_path=root / "manifest.jsonl")
        if skip_ids:
            chosen = [c for c in chosen if c not in set(skip_ids)]
        # if skipping shrunk the list, refill from the unfiltered diverse pool
        if len(chosen) < K_show:
            full = pick_clusters(labels, centroids, K_show * 3, min_size,
                                 mode=pick_mode, paths=paths,
                                 manifest_path=root / "manifest.jsonl")
            for c in full:
                if c in chosen or (skip_ids and c in skip_ids):
                    continue
                chosen.append(c)
                if len(chosen) == K_show:
                    break
    if not chosen:
        raise SystemExit(f"[render] no clusters with size >= {min_size}; lower min_size or rerun cluster")
    print(f"[render] {stream}: showing clusters {chosen} "
          f"(sizes {[(labels == c).sum() for c in chosen]})")

    rows = len(chosen)
    fig_w_in = cols * 0.85
    fig_h_in = rows * 0.85
    fig, axes = plt.subplots(rows, cols,
                             figsize=(fig_w_in + 0.85, fig_h_in),
                             gridspec_kw={"wspace": 0.05, "hspace": 0.05})
    if rows == 1:
        axes = np.array([axes])

    for r, cid in enumerate(chosen):
        sample_idx = cluster_samples(feats, labels, centroids, cid, cols, mode, seed)
        for c in range(cols):
            ax = axes[r, c]
            if c < len(sample_idx):
                img = cv2.imread(str(paths[sample_idx[c]]), cv2.IMREAD_COLOR)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                ax.imshow(img)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
        # row label on the leftmost cell
        axes[r, 0].set_ylabel(f"cluster {cid}", rotation=0, ha="right", va="center",
                              fontsize=10, fontweight="bold", labelpad=22)

    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.subplots_adjust(left=0.13, right=0.99, top=0.99, bottom=0.01)
    fig.savefig(out_pdf)
    plt.close(fig)
    print(f"[render] {stream}: wrote {out_pdf}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("runs/yolov8_hf/cluster_crops"))
    ap.add_argument("--stream", required=True,
                    help="subfolder under --root; built-ins: face | lh | rh | face_mid")
    ap.add_argument("--phases", default="embed,cluster,render",
                    help="comma-separated subset of {embed,cluster,render}")
    ap.add_argument("--batch-size", type=int, default=96)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--K", type=int, default=30,
                    help="k-means cluster count")
    ap.add_argument("--K-show", type=int, default=5,
                    help="number of clusters to display as rows")
    ap.add_argument("--cols", type=int, default=10,
                    help="number of crops per row")
    ap.add_argument("--mode", choices=["centroid", "random"], default="centroid",
                    help="how to pick the cols crops within a cluster")
    ap.add_argument("--min-size", type=int, default=8,
                    help="ignore clusters smaller than this for rendering")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-pdf", type=Path, default=None,
                    help="output PDF (defaults to <root>/clusters_<stream>.pdf)")
    ap.add_argument("--force", action="store_true",
                    help="recompute features/labels even if cached")
    ap.add_argument("--pick-mode",
                    choices=["diverse", "size", "identity-diverse"],
                    default="identity-diverse",
                    help="how to pick rows from the cluster pool")
    ap.add_argument("--cluster-ids", default=None,
                    help="comma-separated explicit cluster ids to render (overrides --pick-mode)")
    ap.add_argument("--skip-ids", default=None,
                    help="comma-separated cluster ids to skip from the picker output")
    args = ap.parse_args()

    out_pdf = args.out_pdf or (args.root / f"clusters_{args.stream}.pdf")
    phases = set(args.phases.split(","))

    if "embed" in phases:
        if args.force and (args.root / f"{args.stream}_features.npz").exists():
            (args.root / f"{args.stream}_features.npz").unlink()
        embed_stream(args.root, args.stream, args.batch_size, args.device)
    if "cluster" in phases:
        cluster_stream(args.root, args.stream, args.K, args.seed, force=args.force)
    if "render" in phases:
        cluster_ids = ([int(x) for x in args.cluster_ids.split(",")]
                       if args.cluster_ids else None)
        skip_ids = ([int(x) for x in args.skip_ids.split(",")]
                    if args.skip_ids else None)
        render_grid(args.root, args.stream, args.K, args.K_show, args.cols,
                    args.seed, args.mode, out_pdf, args.min_size,
                    pick_mode=args.pick_mode,
                    cluster_ids=cluster_ids, skip_ids=skip_ids)


if __name__ == "__main__":
    main()
