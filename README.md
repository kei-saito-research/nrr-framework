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
*5 random seeds. t = 13.33, p < 0.001.*

**Key findings:**
1. NRR-lite maintains near-maximum entropy (H ≈ 0.91) at Turn 1 — ambiguity preserved
2. Baseline collapses to H ≈ 0.15 — premature commitment
3. Both models achieve 100% accuracy when Turn 2 context is provided
4. Results are stable across all 5 seeds (NRR-lite wins 5/5)

## Repository Structure
