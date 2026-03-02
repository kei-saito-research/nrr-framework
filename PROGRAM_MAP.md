# NRR Program Map

This page is the single hub for the NRR paper series.
For the latest status, links, and read order, use this page only.

## Series Numbering Policy
- Active set: 1, 2, 4, 5, 6, 7
- `paper3` is permanently skipped and never reused.
- Mapping:
  - `paper1`: Core
  - `paper2`: Phi
  - `paper4`: IME
  - `paper5`: Transfer
  - `paper6`: Coupled
  - `paper7`: Principles

## Read Order
1. **paper1 / NRR-Core** (foundation)
2. **paper2 / NRR-Phi** (text-to-state mapping + operator principles)
3. **paper3** (intentionally skipped)
4. **paper4 / NRR-IME** (implementation on stateless APIs; under moderation)
5. **paper5 / NRR-Transfer** (cross-domain transfer; pre-submission line)
6. **paper6 / NRR-Coupled** (dependency-consistency for coupled updates; pre-submission line)
7. **paper7 / NRR-Principles** (policy-level framing; pre-submission line)

`NRR-Hout` is deferred to a later phase (outside the current active set).

## Paper / Code Links

### paper1) NRR-Core
- Paper: [arXiv:2512.13478](https://arxiv.org/abs/2512.13478)
- Code: [github.com/kei-saito-research/nrr-core](https://github.com/kei-saito-research/nrr-core)

### paper2) NRR-Phi
- Paper: [arXiv:2601.19933](https://arxiv.org/abs/2601.19933)
- Code: [github.com/kei-saito-research/nrr-phi](https://github.com/kei-saito-research/nrr-phi)

### paper3) (permanently skipped)
- No paper is assigned to `paper3`.

### paper4) NRR-IME
- Paper: repository public; manuscript moderation result pending
- Code: [github.com/kei-saito-research/nrr-ime](https://github.com/kei-saito-research/nrr-ime)

### paper5) NRR-Transfer
- Paper: arXiv submission pending
- Code: [github.com/kei-saito-research/nrr-transfer](https://github.com/kei-saito-research/nrr-transfer)

### paper6) NRR-Coupled
- Paper: arXiv submission pending
- Code: [github.com/kei-saito-research/nrr-coupled](https://github.com/kei-saito-research/nrr-coupled)

### paper7) NRR-Principles
- Paper: in preparation
- Code: [github.com/kei-saito-research/nrr-principles](https://github.com/kei-saito-research/nrr-principles)

### NRR-Hout (deferred)
- Paper: planned for a later phase
- Code: TBD

## One-Line Scope per Paper
- **Core**: Introduces NRR and non-resolution as a computational principle.
- **Phi**: Defines text-to-state mapping and non-collapsing operator conditions.
- **IME**: Finds the stable implementation pattern (Phase 1.5) for stateless APIs.
- **Transfer**: Tests whether the same Phase 1.5 interface transfers across domains.
- **Coupled**: Tests dependency-consistency behavior under coupled client-side updates.
- **Principles**: Consolidates and formalizes shared design principles.
- **HOUT**: Measures ambiguity preserved on the output side (`H_out`) as a diagnostic layer.

## Repro Entry Points
- Core: `experiments/run_turn1_entropy.py`
- Phi: see repository `README` and `experiments/`
- IME / Transfer / Coupled / Principles: see each repository `README` and reproducibility guide.

## Contact
Questions, implementation discussions, and collaboration ideas are welcome via GitHub Issues / Discussions on the repositories above.
