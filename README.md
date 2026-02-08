# NRR-Core

Reference implementation for:

> Saito, K. (2025). **NRR-Core: Non-Resolution Reasoning as a Computational Framework for Contextual Identity and Ambiguity Preservation**.
> *arXiv:2512.13478*

## Overview

This repository provides the **Turn 1 Entropy** experiment that validates the core principle of Non-Resolution Reasoning (NRR): ambiguity should be **preserved** until sufficient context arrives, rather than collapsed prematurely.

### The A≠A Principle

Standard NLP architectures assign a single embedding to each token. For polysemous words like "bank" (financial institution vs. river bank), this forces the model to commit to one interpretation immediately — even before disambiguating context is available. We call this **early semantic collapse**.

NRR-lite addresses this by maintaining **k = 2 separate embeddings** for ambiguous tokens and a **context-dependent gate** that selects the appropriate embedding only when context arrives.

## Experiment: Turn 1 Entropy Measurement

Rather than measuring classification accuracy (which can be seed-dependent), we measure the **entropy of the output distribution at Turn 1** — before any disambiguating context is provided.

**Task structure:**
- **Turn 1** (ambiguous): "The bank is {adj}." (e.g., solid, stable, old)
- **Turn 2** (context): Financial cue or river cue

**Core metric:** Entropy H of P(class) at Turn 1 with neutral Turn 2.
- High H → ambiguity preserved ✓
- Low H → early collapse ✗

### Results (Table 1)

| Model    | Turn 1 Entropy H     | Gate Entropy | Context Accuracy |
|----------|----------------------|--------------|------------------|
| Baseline | 0.15 ± 0.13          | —            | 100%             |
| NRR-lite | **0.91 ± 0.04**      | 1.00         | 100%             |

*All entropy values in **bits** (base-2 logarithm). Max entropy = log₂(2) = 1.00 bits.*  
*5 random seeds. t = 11.92, p < 0.001.*

**Key findings:**
1. NRR-lite maintains near-maximum entropy (H ≈ 0.91) at Turn 1 — ambiguity preserved
2. Baseline collapses to H ≈ 0.15 — premature commitment
3. Both models achieve 100% accuracy when Turn 2 context is provided
4. Results are stable across all 5 seeds (NRR-lite wins 5/5)

## Repository Structure

```
nrr-core/
├── README.md
├── LICENSE                         # CC BY 4.0
├── requirements.txt                # numpy, matplotlib
├── data/
│   └── csws_dataset.json           # Dataset configuration
├── src/
│   ├── __init__.py
│   ├── data_generator.py           # CSWS data generation
│   ├── tokenizer.py                # Word-level tokenizer
│   ├── baseline.py                 # Single-embedding baseline
│   ├── nrr_lite.py                 # Multi-vector embedding + context gate
│   └── training.py                 # Training loop and evaluation
├── experiments/
│   └── run_turn1_entropy.py        # Main experiment (reproduces Table 1)
└── results/
    └── turn1_entropy_output.json   # Verification output
```

## Quick Start

```bash
# Install dependencies (NumPy only required; matplotlib for plots)
pip install numpy matplotlib

# Single seed
python experiments/run_turn1_entropy.py

# Multi-seed verification (reproduces Table 1)
python experiments/run_turn1_entropy.py --multi

# Custom seed
python experiments/run_turn1_entropy.py --seed 123
```

### Expected Output (multi-seed)

```
MULTI-SEED SUMMARY [bits]
======================================================================

Seed       Baseline H      NRR-lite H      Diff       Winner
------------------------------------------------------------
42         0.163           0.865           0.702      NRR-lite
123        0.162           0.947           0.785      NRR-lite
456        0.033           0.942           0.909      NRR-lite
789        0.013           0.866           0.853      NRR-lite
1000       0.371           0.916           0.545      NRR-lite

VERIFICATION AGAINST TABLE 1
======================================================================
  [PASS] baseline_h_mean: expected 0.15, got 0.148 bits
  [PASS] nrr_h_mean: expected 0.91, got 0.907 bits
  [PASS] nrr_wins_all: expected 5, got 5
  [PASS] t_significant: expected > 2.0, got 11.921

  ALL CHECKS PASSED
```

**Note on entropy units**: All entropy values use **base-2 logarithm (bits)**. The maximum entropy for binary classification is H_max = log₂(2) = 1.00 bits.

## Architecture Details

### Baseline (Single Embedding)

```
Turn 1 tokens → Embedding[vocab, d] → mean pool → turn1_vec
Turn 2 tokens → Embedding[vocab, d] → mean pool → turn2_vec
[turn1_vec; turn2_vec] → Linear(2d, h) → ReLU → Linear(h, 2) → softmax
```

Each token has exactly one embedding. "bank" is forced into a single point in embedding space.

### NRR-lite (Multi-Vector Embedding + Context Gate)

```
Turn 2 tokens → Embedding[vocab, d] → mean pool → context_vec
context_vec → gate_W → softmax → gates = [g₀, g₁]

For "bank" token:
  bank_embedding = g₀ · bank_financial + g₁ · bank_river

Turn 1 tokens → Embedding (with gated "bank") → mean pool → turn1_vec
[turn1_vec; turn2_vec] → Linear(2d, h) → ReLU → Linear(h, 2) → softmax
```

When Turn 2 is neutral (all PAD → zero vector), gate bias is zero, so softmax([0, 0]) = [0.5, 0.5] — perfect ambiguity preservation.

## Dependencies

- **NumPy** ≥ 1.20 (required)
- **Matplotlib** ≥ 3.5 (optional, for plots)

No deep learning framework required. All operations are pure NumPy.

## Related Repositories

- [NRR-Phi](https://github.com/kei-saito-research/nrr-phi) - Text-to-state mapping *(arXiv:2601.19933)*
- [NRR-IME](https://github.com/kei-saito-research/nrr-ime) - Structure-aware optimization
- [NRR-Universal](https://github.com/kei-saito-research/nrr-universal) - Universal generality validation

## Citation

```bibtex
@article{saito2025nrr,
  title={NRR-Core: Non-Resolution Reasoning as a Computational Framework for Contextual Identity and Ambiguity Preservation},
  author={Saito, Kei},
  journal={arXiv preprint arXiv:2512.13478},
  year={2025}
}
```

## Technical Notes

### Entropy Calculation

The `compute_entropy` function in `training.py` uses base-2 logarithm to compute Shannon entropy in bits:

```python
def compute_entropy(probs: np.ndarray) -> float:
    """Shannon entropy H(p) = -sum_i p_i log₂(p_i)."""
    return float(-np.sum(probs * np.log2(probs + 1e-10)))
```

This ensures consistency with the paper's mathematical formulation (Section 6, Eq. 3) and with the figure values.

### Reproducibility

The experiment uses fixed random seeds (42, 123, 456, 789, 1000) for reproducibility. However, due to the simplified backpropagation and small model size, individual seed results may vary slightly from run to run. The aggregate statistics (mean ± std) should match the paper within tolerance (±0.03 bits).

## Commercial Use

If you plan to use this in a commercial or production setting,
a short message would be appreciated.

## License

CC BY 4.0 License. See [LICENSE](LICENSE).

---

## Contact

Kei Saito  
Independent Researcher, Japan  
ORCID: [0009-0006-4715-9176](https://orcid.org/0009-0006-4715-9176)  
Email: kei.saito.research@gmail.com
