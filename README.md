# NRR Framework

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

### Results (Paper 1, Table 1)

| Model    | Turn 1 Entropy H     | Gate Entropy | Context Accuracy |
|----------|----------------------|--------------|------------------|
| Baseline | 0.15 ± 0.08 bits     | —            | 100%             |
| NRR-lite | **0.91 ± 0.03 bits** | 1.0 bits     | 100%             |

*5 random seeds. Max entropy = log₂(2) = 1.0 bits. t = 13.3, p < 0.001.*

**Key findings:**
1. NRR-lite maintains near-maximum entropy (H ≈ 0.91 bits) at Turn 1 — ambiguity preserved
2. Baseline collapses to H ≈ 0.15 bits — premature commitment
3. Both models achieve 100% accuracy when Turn 2 context is provided
4. Results are stable across all 5 seeds (NRR-lite wins 5/5)

## Repository Structure
```
nrr-framework/
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

# Multi-seed verification (reproduces Paper 1 Table 1)
python experiments/run_turn1_entropy.py --multi

# Custom seed
python experiments/run_turn1_entropy.py --seed 123
```

### Expected Output (multi-seed)
```
MULTI-SEED SUMMARY
======================================================================

Seed       Baseline H      NRR-lite H      Diff       Winner
------------------------------------------------------------
42         0.1602          0.8640          0.7038     NRR-lite
123        0.1614          0.9463          0.7849     NRR-lite
456        0.0329          0.9413          0.9084     NRR-lite
789        0.0126          0.8657          0.8531     NRR-lite
1000       0.3711          0.9153          0.5442     NRR-lite

VERIFICATION AGAINST PAPER 1 TABLE 1
======================================================================
  [PASS] baseline_h_mean: expected 0.15, got 0.1476
  [PASS] nrr_h_mean: expected 0.91, got 0.9065
  [PASS] nrr_wins_all: expected 5, got 5
  [PASS] t_significant: expected > 2.0, got 13.33

  ALL CHECKS PASSED
```

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

## Related Papers

1. **NRR-Core** (this repository): Saito, K. (2025). NRR-Core: Non-Resolution Reasoning as a Computational Framework for Contextual Identity and Ambiguity Preservation. *arXiv:2512.13478*
2. **NRR-Phi**: Saito, K. (2026). NRR-Phi: Text-to-State Mapping for Ambiguity Preservation in LLM Inference. *arXiv:2601.19933* → [nrr-phi-mapping](https://github.com/kei-saito-research/nrr-phi-mapping)

## Citation
```bibtex
@article{saito2025nrrcore,
  title={NRR-Core: Non-Resolution Reasoning as a Computational Framework for Contextual Identity and Ambiguity Preservation},
  author={Saito, Kei},
  journal={arXiv preprint arXiv:2512.13478},
  year={2025}
}
```

## License

CC BY 4.0. See [LICENSE](LICENSE).
