# Reproducibility (NRR-Core)

## Scope

This repository snapshot bundles the current manuscript source package together with
the primary offline verification entrypoint used for the public Core line.

## Stable review-package commands

- Build the current manuscript to temp output:
  - `bash scripts/build_current_manuscript.sh`
  - output: `/tmp/nrr-core_current_build/paper1_nrr-core_v38.pdf`
- Verify the current review-package checksum manifest:
  - `bash scripts/verify_current_package.sh`
- Reproduce the primary result to temp output:
  - `bash scripts/run_primary_check.sh`
  - output: `/tmp/nrr_core_turn1_entropy_output.json`

## Current review package

- Main TeX: `manuscript/current/paper1_nrr-core_v38.tex`
- Current manuscript figure: `manuscript/current/figure_nrr_experiment.png`
- Checksum manifest: `manuscript/current/checksums_sha256.txt`

## Checksum policy

- `manuscript/current/checksums_sha256.txt` covers the tracked files that define the
  current review package for the latest manuscript line in `manuscript/current/`.
- Coverage includes the current main `.tex` file and each figure asset consumed by
  that current manuscript from `manuscript/current/`.
- Coverage excludes `checksums_sha256.txt` itself, older manuscript versions kept
  outside the current package, and repo-specific artifacts outside
  `manuscript/current/` unless a separate manifest is provided.

## Environment

- Python: 3.13.7 (`python3`)
- Main libraries: NumPy >= 1.20.0, Matplotlib >= 3.5.0
- OS: Darwin 25.2.0 arm64

## Fixed protocol settings

- Model comparison: Baseline (single embedding) vs NRR-lite (multi-vector + context gate)
- Model settings: `embed_dim=32`, `hidden_dim=64`
- Seed set: `[42, 123, 456, 789, 1000]`
- Temperature: N/A (no LLM sampling in this repo)
- Trials: 5 seeds (multi-seed aggregate)

## Artifact map

| Artifact | Command | Output |
|---|---|---|
| Paper Table 1 entropy verification | `bash scripts/run_primary_check.sh` | `/tmp/nrr_core_turn1_entropy_output.json` |
| Current manuscript build | `bash scripts/build_current_manuscript.sh` | `/tmp/nrr-core_current_build/paper1_nrr-core_v38.pdf` |
| Current package checksum verification | `bash scripts/verify_current_package.sh` | stdout verification for `manuscript/current/checksums_sha256.txt` |
| Current manuscript source snapshot | N/A (tracked artifact) | `manuscript/current/paper1_nrr-core_v38.tex` |
| Current manuscript figure snapshot | N/A (tracked artifact) | `manuscript/current/figure_nrr_experiment.png` |
| Version map | N/A (tracked artifact) | `VERSION_MAP.md` |

## Known limitations

- Exact floating-point values can vary slightly by Python/NumPy build.
- No container lockfile is provided; environment is documented but not fully pinned.
