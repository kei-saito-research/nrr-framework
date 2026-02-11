# NRR Program Map

This page is the single hub for the NRR paper series.
For the latest status, links, and read order, use this page only.

## Read Order
1. **NRR-Core** (foundation)
2. **NRR-Phi** (text-to-state mapping + operator principles)
3. **NRR-IME** (implementation on stateless APIs)
4. **NRR-Transfer** (cross-domain transfer)
5. **NRR-Principal** (in preparation)
6. **NRR-Hout** (output-side diagnostic; planned update after Principal)

## Paper / Code Links

### 1) NRR-Core
- Paper: [arXiv:2512.13478](https://arxiv.org/abs/2512.13478)
- Code: [github.com/kei-saito-research/nrr-core](https://github.com/kei-saito-research/nrr-core)

### 2) NRR-Phi
- Paper: [arXiv:2601.19933](https://arxiv.org/abs/2601.19933)
- Code: [github.com/kei-saito-research/nrr-phi](https://github.com/kei-saito-research/nrr-phi)

### 3) NRR-IME
- Paper: arXiv submission pending
- Code: [github.com/kei-saito-research/nrr-ime](https://github.com/kei-saito-research/nrr-ime)

### 4) NRR-Transfer
- Paper: arXiv submission pending
- Code: [github.com/kei-saito-research/nrr-transfer](https://github.com/kei-saito-research/nrr-transfer)

### 5) NRR-Principal
- Paper: in preparation
- Code: TBD

### 6) NRR-Hout
- Paper: update planned after Principal
- Code: TBD

## One-Line Scope per Paper
- **Core**: Introduces NRR and non-resolution as a computational principle.
- **Phi**: Defines text-to-state mapping and non-collapsing operator conditions.
- **IME**: Finds the stable implementation pattern (Phase 1.5) for stateless APIs.
- **Transfer**: Tests whether the same Phase 1.5 interface transfers across domains.
- **Principal**: Consolidates and formalizes shared design principles.
- **Hout**: Measures ambiguity preserved on the output side (`H_out`) as a diagnostic layer.

## Repro Entry Points
- Core: `experiments/run_turn1_entropy.py`
- Phi: see repository `README` and `experiments/`
- IME / Universal: see repository `README` and `experiments/`

## Contact
Questions, implementation discussions, and collaboration ideas are welcome via GitHub Issues / Discussions on the repositories above.
