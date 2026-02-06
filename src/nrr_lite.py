"""
NRR-lite: Multi-Vector Embedding with Context Gate

The minimal implementation of Non-Resolution Reasoning.  For the
ambiguous token "bank", NRR-lite maintains k=2 separate embedding
vectors (one per sense).  A context-dependent gate mixes these vectors
based on Turn 2 input.

Key properties:
    * When Turn 2 is neutral (all PAD), the gate stays balanced at
      [0.5, 0.5] -- preserving ambiguity (A ≠ A principle).
    * When Turn 2 contains a financial cue, the gate shifts toward
      the financial embedding.
    * When Turn 2 contains a river cue, the gate shifts toward the
      river embedding.

Architecture:
    Turn 2 -> mean-pool -> context_vec -> gate_W -> softmax -> gates
    "bank" embedding = gates[0] * bank_financial + gates[1] * bank_river
    [turn1_vec ; turn2_vec] -> ReLU hidden -> softmax -> P(class)

Reference: Saito, K. (2025). "NRR-Core: Non-Resolution Reasoning as a 
    Computational Framework for Contextual Identity and Ambiguity 
    Preservation". arXiv:2512.13478, Sections 5, 6.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np


class NRRLiteModel:
    """NRR-lite: Multi-Vector Embedding (k=2) + Context Gate.

    Args:
        vocab_size: Number of tokens.
        embed_dim: Embedding dimension.
        hidden_dim: Hidden layer dimension.
        bank_idx: Token index for the ambiguous word "bank".
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 32,
        hidden_dim: int = 64,
        bank_idx: Optional[int] = None,
    ) -> None:
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.bank_idx = bank_idx

        # Shared embedding table
        self.embedding = np.random.randn(vocab_size, embed_dim) * 0.1
        # PAD embedding is zero
        self.embedding[0] = np.zeros(embed_dim)

        # Multi-Vector Embedding: two senses for "bank"
        self.bank_financial = np.random.randn(embed_dim) * 0.1
        self.bank_river = np.random.randn(embed_dim) * 0.1

        # Context gate (bias = 0 -> softmax([0,0]) = [0.5, 0.5])
        self.gate_W = np.random.randn(embed_dim, 2) * 0.01
        self.gate_b = np.zeros(2)

        # Classifier
        self.W1 = np.random.randn(embed_dim * 2, hidden_dim) * 0.1
        self.b1 = np.zeros(hidden_dim)
        self.W2 = np.random.randn(hidden_dim, 2) * 0.1
        self.b2 = np.zeros(2)

    def compute_gate(self, context_vec: np.ndarray) -> np.ndarray:
        """Compute gate values from Turn 2 context.

        Args:
            context_vec: (batch, embed_dim) mean-pooled Turn 2 embedding.

        Returns:
            (batch, 2) gate weights (softmax-normalised).
        """
        logits = context_vec @ self.gate_W + self.gate_b
        exp_logits = np.exp(logits - logits.max(axis=1, keepdims=True))
        gates = exp_logits / exp_logits.sum(axis=1, keepdims=True)
        return gates

    def forward(self, turn1_ids: np.ndarray, turn2_ids: np.ndarray) -> Dict:
        """Forward pass with context-gated multi-vector embedding.

        Args:
            turn1_ids: (batch, seq_len) token indices for Turn 1.
            turn2_ids: (batch, seq_len) token indices for Turn 2.

        Returns:
            Dict with logits, probs, gates, turn1_vec, turn2_vec.
        """
        batch_size = turn1_ids.shape[0]

        # Turn 2 encoding
        turn2_emb = self.embedding[turn2_ids]
        turn2_vec = turn2_emb.mean(axis=1)

        # Context gate
        gates = self.compute_gate(turn2_vec)

        # Turn 1 encoding with multi-vector embedding for "bank"
        turn1_emb = self.embedding[turn1_ids].copy()

        for b in range(batch_size):
            bank_mask = turn1_ids[b] == self.bank_idx
            if bank_mask.any():
                mixed_bank = (
                    gates[b, 0] * self.bank_financial
                    + gates[b, 1] * self.bank_river
                )
                turn1_emb[b, bank_mask] = mixed_bank

        turn1_vec = turn1_emb.mean(axis=1)

        # Classifier
        combined = np.concatenate([turn1_vec, turn2_vec], axis=1)
        hidden = np.maximum(0, combined @ self.W1 + self.b1)
        logits = hidden @ self.W2 + self.b2

        exp_logits = np.exp(logits - logits.max(axis=1, keepdims=True))
        probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)

        return {
            "logits": logits,
            "probs": probs,
            "gates": gates,
            "turn1_vec": turn1_vec,
            "turn2_vec": turn2_vec,
        }

    def predict(self, turn1_ids: np.ndarray, turn2_ids: np.ndarray) -> np.ndarray:
        """Return predicted class indices."""
        return self.forward(turn1_ids, turn2_ids)["probs"].argmax(axis=1)
