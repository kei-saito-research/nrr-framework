"""
NRR Framework: Non-Resolution Reasoning Reference Implementation

Reference implementation for:
    Saito, K. (2025). "NRR-Core: Non-Resolution Reasoning as a 
    Computational Framework for Contextual Identity and Ambiguity 
    Preservation". arXiv:2512.13478
"""

from .data_generator import EntropyDataGenerator
from .tokenizer import SimpleTokenizer
from .baseline import BaselineModel
from .nrr_lite import NRRLiteModel
from .training import (
    train_model,
    compute_entropy,
    evaluate_turn1_entropy,
    evaluate_with_context,
)

__all__ = [
    "EntropyDataGenerator",
    "SimpleTokenizer",
    "BaselineModel",
    "NRRLiteModel",
    "train_model",
    "compute_entropy",
    "evaluate_turn1_entropy",
    "evaluate_with_context",
]
