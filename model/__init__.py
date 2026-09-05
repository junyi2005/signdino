"""Image-stream temporal-DINO model.

Three modules:
- FrozenDinov3FrameEmbedder: per-frame 768-d CLS from a frozen DINOv3 ViT-B/16.
- TemporalTransformer: small Transformer over per-frame embeddings, with
  sinusoidal time PE and a learnable CLS token.
- SignTemporalDinoModel: student + EMA teacher + DINO/iBOT/Gram heads, exposes
  step() for the train loop.
"""

from .backbone import FrozenDinov3FrameEmbedder
from .temporal_encoder import TemporalTransformer
from .meta_arch import SignTemporalDinoModel
from .live_upstream import LiveUpstream, LiveUpstreamConfig

__all__ = [
    "FrozenDinov3FrameEmbedder",
    "TemporalTransformer",
    "SignTemporalDinoModel",
    "LiveUpstream",
    "LiveUpstreamConfig",
]
