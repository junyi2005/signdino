"""Isolated Sign Language Recognition (ISLR).

Mirrors SHuBERT § 4.3: 125 epochs, batch 128, Adam lr 1e-4, weight-decay
1e-4, early stop on Recall@1, rank-1 LoRA on every linear of the
upstream when fine-tuning. Reported on ASL Citizen, Sem-Lex, WLASL2000.
"""

from .lora import LoRALinear, apply_lora_to
from .model import SignDinoISLR, ISLRConfig, LiveSignDinoISLR
from .dataset import (
    ISLRDataset, islr_collate,
    LiveISLRDataset, LiveISLRDatasetConfig, live_islr_collate,
)
from .trainer import ISLRTrainer

__all__ = [
    "LoRALinear", "apply_lora_to",
    "SignDinoISLR", "ISLRConfig", "LiveSignDinoISLR",
    "ISLRDataset", "islr_collate",
    "LiveISLRDataset", "LiveISLRDatasetConfig", "live_islr_collate",
    "ISLRTrainer",
]
