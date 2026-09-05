"""Light image-level augmentation for per-frame crops, applied only when the
embedding cache is disabled (i.e. we're forwarding through DINOv3 live).

The mirror / 3D / identity-swap augmentations live upstream
(phase6_augment); here we only do color/blur jitter so the SSL student gets
a per-iteration nuisance signal on top of the upstream variants.
"""
from __future__ import annotations

import torch
from PIL import Image
from torchvision import transforms as T


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_image_aug(input_size: int = 224, jitter_strength: float = 0.4) -> T.Compose:
    """Return a torchvision transform: PIL.Image -> normalised (3, H, W) tensor."""
    return T.Compose([
        T.Resize((input_size, input_size), interpolation=T.InterpolationMode.BICUBIC),
        T.ColorJitter(
            brightness=jitter_strength,
            contrast=jitter_strength,
            saturation=jitter_strength,
            hue=0.1,
        ),
        T.RandomApply([T.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))], p=0.3),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def build_image_eval(input_size: int = 224) -> T.Compose:
    return T.Compose([
        T.Resize((input_size, input_size), interpolation=T.InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
