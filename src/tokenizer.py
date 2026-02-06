"""
Simple Word-Level Tokenizer

Provides a minimal tokenizer for the CSWS experiment.  Words are
lowercased, split on whitespace, and mapped to integer indices.

Reference: Saito, K. (2025). "NRR-Core: Non-Resolution Reasoning as a Computational Framework for Contextual
    Identity and Ambiguity Preservation". arXiv:2512.13478, Appendix D.
"""

from __future__ import annotations

from typing import List

import numpy as np


class SimpleTokenizer:
    """Word-level tokenizer with fixed vocabulary.

    Special tokens:
        ``<PAD>`` (index 0) — padding.
        ``<UNK>`` (index 1) — unknown words.

    Attributes:
        word2idx: Mapping from word to integer index.
        idx2word: Reverse mapping.
        bank_idx: Index of the ambiguous token ``"bank"``.
    """

    def __init__(self) -> None:
        self.word2idx = {"<PAD>": 0, "<UNK>": 1}
        self.idx2word = {0: "<PAD>", 1: "<UNK>"}
        self.bank_idx: int | None = None

    def fit(self, texts: List[str]) -> None:
        """Build vocabulary from a list of sentences.

        Args:
            texts: Raw sentences (will be lowercased and split).
        """
        words = set()
        for text in texts:
            words.update(text.lower().split())

        for idx, word in enumerate(sorted(words), start=2):
            self.word2idx[word] = idx
            self.idx2word[idx] = word

        self.bank_idx = self.word2idx.get("bank", 1)

    def encode(self, text: str, max_len: int = 20) -> np.ndarray:
        """Encode a sentence to a fixed-length integer array.

        Args:
            text: Raw sentence.
            max_len: Pad or truncate to this length.

        Returns:
            1-D int64 array of token indices.
        """
        words = text.lower().split()
        indices = [self.word2idx.get(w, 1) for w in words]

        if len(indices) < max_len:
            indices += [0] * (max_len - len(indices))
        else:
            indices = indices[:max_len]

        return np.array(indices, dtype=np.int64)

    def get_neutral_ids(self, max_len: int = 20) -> np.ndarray:
        """Return an all-PAD sequence (neutral Turn 2).

        Returns:
            1-D int64 array of zeros.
        """
        return np.zeros(max_len, dtype=np.int64)

    @property
    def vocab_size(self) -> int:
        """Number of tokens in the vocabulary."""
        return len(self.word2idx)
