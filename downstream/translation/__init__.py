"""Sign-to-English translation downstream.

Mirrors SHuBERT § 4.2: ByT5-Base decoder, two-phase training (Phase 1 on
YouTube-ASL, Phase 2 on a benchmark such as How2Sign / OpenASL), beam
search inference with width 5. The novelty vs SHuBERT is upstream
(SignDINO-v2 features + 3-stream fusion); the downstream recipe is held
constant for a fair head-to-head.
"""

from .model import SignDinoTranslator, TranslatorConfig, LiveSignDinoTranslator
from .dataset import (
    TranslationDataset, translation_collate,
    LiveTranslationDataset, LiveTranslationDatasetConfig, live_translation_collate,
)
from .trainer import TranslationTrainer

__all__ = [
    "SignDinoTranslator", "TranslatorConfig", "LiveSignDinoTranslator",
    "TranslationDataset", "translation_collate",
    "LiveTranslationDataset", "LiveTranslationDatasetConfig", "live_translation_collate",
    "TranslationTrainer",
]
