"""
Training and Evaluation Utilities

Provides:
    * ``train_model`` -- SGD training loop with simplified backprop.
    * ``compute_entropy`` -- Shannon entropy of a probability vector.
    * ``evaluate_turn1_entropy`` -- Core metric: entropy at Turn 1
      when Turn 2 is neutral (all PAD).
    * ``evaluate_with_context`` -- Sanity check: accuracy when Turn 2
      context is provided.

Reference: Saito (2025). "NRR-Core: Non-Resolution Reasoning as a
    Computational Framework for Contextual Identity and Ambiguity
    Preservation". arXiv:2512.13478
"""

from __future__ import annotations

from typing import Dict, List, Union

import numpy as np

from .data_generator import EntropyDataGenerator
from .tokenizer import SimpleTokenizer


# ------------------------------------------------------------------
# Entropy
# ------------------------------------------------------------------

def compute_entropy(probs: np.ndarray) -> float:
    """Shannon entropy H(p) = -sum_i p_i log2(p_i).

    Args:
        probs: 1-D probability vector.

    Returns:
        Entropy in bits.
    """
    return float(-np.sum(probs * np.log2(probs + 1e-10)))


# ------------------------------------------------------------------
# Training
# ------------------------------------------------------------------

def train_model(
    model: object,
    train_data: List[Dict],
    tokenizer: SimpleTokenizer,
    epochs: int = 100,
    batch_size: int = 32,
    lr: float = 0.1,
    is_nrr: bool = False,
    verbose: bool = True,
) -> None:
    """Train a model with simplified SGD backpropagation.

    Supports both :class:`BaselineModel` and :class:`NRRLiteModel`.
    When ``is_nrr`` is True, additional gradients for the gate network
    and multi-vector embeddings are computed.

    Args:
        model: Model instance (BaselineModel or NRRLiteModel).
        train_data: List of training dicts (turn1, turn2, label).
        tokenizer: Fitted SimpleTokenizer.
        epochs: Number of training epochs.
        batch_size: Mini-batch size.
        lr: Learning rate.
        is_nrr: Enable NRR-specific gradient updates.
        verbose: Print progress every 20 epochs.
    """
    for epoch in range(epochs):
        np.random.shuffle(train_data)
        epoch_loss = 0.0
        epoch_correct = 0
        n_batches = 0

        for i in range(0, len(train_data), batch_size):
            batch = train_data[i : i + batch_size]

            turn1_ids = np.stack([tokenizer.encode(d["turn1"]) for d in batch])
            turn2_ids = np.stack([tokenizer.encode(d["turn2"]) for d in batch])
            labels = np.array([d["label"] for d in batch])
            batch_len = len(labels)

            outputs = model.forward(turn1_ids, turn2_ids)
            probs = outputs["probs"]

            # Cross-entropy loss
            batch_loss = -np.log(probs[range(batch_len), labels] + 1e-10).mean()
            epoch_loss += batch_loss

            preds = probs.argmax(axis=1)
            epoch_correct += (preds == labels).sum()

            # ---- Simplified backpropagation ----
            grad_probs = probs.copy()
            grad_probs[range(batch_len), labels] -= 1
            grad_probs /= batch_len

            turn1_vec = outputs["turn1_vec"]
            turn2_vec = outputs["turn2_vec"]
            combined = np.concatenate([turn1_vec, turn2_vec], axis=1)
            hidden = np.maximum(0, combined @ model.W1 + model.b1)

            # Output layer
            grad_W2 = hidden.T @ grad_probs
            grad_b2 = grad_probs.sum(axis=0)

            # Hidden layer
            grad_hidden = grad_probs @ model.W2.T
            grad_hidden[hidden <= 0] = 0

            grad_W1 = combined.T @ grad_hidden
            grad_b1 = grad_hidden.sum(axis=0)

            # Update classifier
            model.W2 -= lr * grad_W2
            model.b2 -= lr * grad_b2
            model.W1 -= lr * grad_W1
            model.b1 -= lr * grad_b1

            # Embedding gradient (Turn 2 tokens)
            grad_combined = grad_hidden @ model.W1.T
            grad_turn2_vec = grad_combined[:, model.embed_dim :]

            for b in range(batch_len):
                for t in range(turn2_ids.shape[1]):
                    if turn2_ids[b, t] != 0:
                        model.embedding[turn2_ids[b, t]] -= (
                            lr * 0.01 * grad_turn2_vec[b] / 20
                        )

            # NRR-specific updates
            if is_nrr and hasattr(model, "gate_W"):
                model.gate_W -= lr * 0.1 * (turn2_vec.T @ grad_probs[:, :2])
                model.gate_b -= lr * 0.1 * grad_probs[:, :2].sum(axis=0)

                gates = outputs["gates"]
                for b in range(batch_len):
                    if tokenizer.bank_idx in turn1_ids[b]:
                        model.bank_financial -= (
                            lr * 0.05 * gates[b, 0] * grad_turn2_vec[b]
                        )
                        model.bank_river -= (
                            lr * 0.05 * gates[b, 1] * grad_turn2_vec[b]
                        )

            n_batches += 1

        if verbose and (epoch + 1) % 20 == 0:
            acc = epoch_correct / len(train_data)
            print(
                f"  Epoch {epoch + 1}/{epochs}"
                f" - Loss: {epoch_loss / n_batches:.4f}"
                f", Acc: {acc:.4f}"
            )


# ------------------------------------------------------------------
# Evaluation: Turn 1 Entropy
# ------------------------------------------------------------------

def evaluate_turn1_entropy(
    model: object,
    test_data: List[Dict],
    tokenizer: SimpleTokenizer,
) -> Dict:
    """Measure output entropy at Turn 1 with neutral Turn 2.

    This is the **core metric** of the experiment.  A model that
    preserves ambiguity should yield high entropy; a model that
    collapses early yields low entropy.

    Args:
        model: Trained model.
        test_data: Entropy test samples (dicts with key ``turn1``).
        tokenizer: Fitted tokenizer.

    Returns:
        Dict with ``mean_entropy``, ``std_entropy``, ``entropies``,
        ``probs_list``, and (for NRR-lite) ``gate_entropies``.
    """
    entropies: List[float] = []
    probs_list: List[np.ndarray] = []
    gate_entropies: List[float] = []

    neutral_turn2 = tokenizer.get_neutral_ids().reshape(1, -1)

    for sample in test_data:
        turn1_ids = tokenizer.encode(sample["turn1"]).reshape(1, -1)
        outputs = model.forward(turn1_ids, neutral_turn2)
        probs = outputs["probs"][0]

        h = compute_entropy(probs)
        entropies.append(h)
        probs_list.append(probs)

        if "gates" in outputs:
            gate_h = compute_entropy(outputs["gates"][0])
            gate_entropies.append(gate_h)

    return {
        "mean_entropy": float(np.mean(entropies)),
        "std_entropy": float(np.std(entropies)),
        "max_entropy": float(np.max(entropies)),
        "min_entropy": float(np.min(entropies)),
        "entropies": entropies,
        "probs_list": probs_list,
        "gate_entropies": gate_entropies if gate_entropies else None,
    }


# ------------------------------------------------------------------
# Evaluation: With Context (Sanity Check)
# ------------------------------------------------------------------

def evaluate_with_context(
    model: object,
    test_data: List[Dict],
    tokenizer: SimpleTokenizer,
    generator: EntropyDataGenerator,
) -> Dict:
    """Verify that both models classify correctly when Turn 2 is given.

    Args:
        model: Trained model.
        test_data: Test samples (dicts with key ``turn1``).
        tokenizer: Fitted tokenizer.
        generator: Data generator (to produce Turn 2 sentences).

    Returns:
        Dict with ``financial_acc``, ``river_acc``, ``overall_acc``.
    """
    financial_acc: List[bool] = []
    river_acc: List[bool] = []

    for sample in test_data:
        turn1_ids = tokenizer.encode(sample["turn1"]).reshape(1, -1)

        # Financial context
        turn2_fin = generator.generate_turn2_financial()
        turn2_fin_ids = tokenizer.encode(turn2_fin).reshape(1, -1)
        pred_fin = model.predict(turn1_ids, turn2_fin_ids)[0]
        financial_acc.append(pred_fin == 0)

        # River context
        turn2_riv = generator.generate_turn2_river()
        turn2_riv_ids = tokenizer.encode(turn2_riv).reshape(1, -1)
        pred_riv = model.predict(turn1_ids, turn2_riv_ids)[0]
        river_acc.append(pred_riv == 1)

    return {
        "financial_acc": float(np.mean(financial_acc)),
        "river_acc": float(np.mean(river_acc)),
        "overall_acc": float(
            (np.mean(financial_acc) + np.mean(river_acc)) / 2
        ),
    }
