"""
Baseline Transformer (Single-Embedding Architecture)

A minimal two-turn classifier using one embedding vector per token.
This is the control model against which NRR-lite is compared.

Architecture:
    Turn 1 -> mean-pool embeddings -> turn1_vec
    Turn 2 -> mean-pool embeddings -> turn2_vec
    [turn1_vec ; turn2_vec] -> ReLU hidden -> softmax -> P(class)

Because each token has exactly one embedding, "bank" is forced into
a single representation even at Turn 1, causing *early semantic
collapse* -- the core phenomenon that NRR addresses.

Reference: Saito, K. (2025). "NRR-Core: Non-Resolution Reasoning as a 
    Computational Framework for Contextual Identity and Ambiguity 
    Preservation". arXiv:2512.13478, Section 6.
"""

from __future__ import annotations

from typing import Dict

import numpy as np


class BaselineModel:
    """Standard single-embedding classifier.

    Args:
        vocab_size: Number of tokens.
        embed_dim: Embedding dimension.
        hidden_dim: Hidden layer dimension.
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 32,
        hidden_dim: int = 64,
    ) -> None:
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim

        # Parameters
        self.embedding = np.random.randn(vocab_size, embed_dim) * 0.1
        self.W1 = np.random.randn(embed_dim * 2, hidden_dim) * 0.1
        self.b1 = np.zeros(hidden_dim)
        self.W2 = np.random.randn(hidden_dim, 2) * 0.1
        self.b2 = np.zeros(2)

    def forward(self, turn1_ids: np.ndarray, turn2_ids: np.ndarray) -> Dict:
        """Forward pass.

        Args:
            turn1_ids: (batch, seq_len) token indices for Turn 1.
            turn2_ids: (batch, seq_len) token indices for Turn 2.

        Returns:
            Dict with ``logits``, ``probs``, ``turn1_vec``, ``turn2_vec``.
        """
        turn1_emb = self.embedding[turn1_ids]
        turn2_emb = self.embedding[turn2_ids]

        turn1_vec = turn1_emb.mean(axis=1)
        turn2_vec = turn2_emb.mean(axis=1)

        combined = np.concatenate([turn1_vec, turn2_vec], axis=1)
        hidden = np.maximum(0, combined @ self.W1 + self.b1)  # ReLU
        logits = hidden @ self.W2 + self.b2

        # Stable softmax
        exp_logits = np.exp(logits - logits.max(axis=1, keepdims=True))
        probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)

        return {
            "logits": logits,
            "probs": probs,
            "turn1_vec": turn1_vec,
            "turn2_vec": turn2_vec,
        }

    def predict(self, turn1_ids: np.ndarray, turn2_ids: np.ndarray) -> np.ndarray:
        """Return predicted class indices."""
        return self.forward(turn1_ids, turn2_ids)["probs"].argmax(axis=1)
