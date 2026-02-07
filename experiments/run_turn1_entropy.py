#!/usr/bin/env python3
"""
Turn 1 Entropy Verification Experiment
=======================================

Reproduces Table 1 from:
    Saito, K. (2025). "NRR-Core: Non-Resolution Reasoning as a 
    Computational Framework for Contextual Identity and Ambiguity 
    Preservation". arXiv:2512.13478

Core idea:
    Measure the output entropy at Turn 1 (before disambiguating context).
    A model that preserves ambiguity should maintain HIGH entropy;
    a model that collapses early should show LOW entropy.

Expected results (Paper 1, Table 1, in BITS):

    +----------+---------------------+--------------+------------------+
    | Model    | Turn 1 Entropy H    | Gate Entropy | Context Accuracy |
    +----------+---------------------+--------------+------------------+
    | Baseline | 0.15 +/- 0.13       | ---          | 100%             |
    | NRR-lite | 0.91 +/- 0.04       | 1.00         | 100%             |
    +----------+---------------------+--------------+------------------+

    All entropy values use base-2 logarithm (bits).
    H_max = log₂(2) = 1.00 bits for binary classification.

Usage:
    python run_turn1_entropy.py              # single seed (42)
    python run_turn1_entropy.py --multi      # 5 seeds
    python run_turn1_entropy.py --seed 123   # custom seed
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

# Allow imports from parent directory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data_generator import EntropyDataGenerator
from src.tokenizer import SimpleTokenizer
from src.baseline import BaselineModel
from src.nrr_lite import NRRLiteModel
from src.training import (
    train_model,
    evaluate_turn1_entropy,
    evaluate_with_context,
)


# ======================================================================
# Paper 1 Table 1 reference values (in BITS)
# ======================================================================

PAPER1_TABLE1 = {
    "baseline_h_mean": 0.15,
    "baseline_h_std": 0.13,
    "nrr_h_mean": 0.91,
    "nrr_h_std": 0.04,
    "gate_entropy": 1.00,
    "context_accuracy": 1.0,
    "n_seeds": 5,
    "seeds": [42, 123, 456, 789, 1000],
    "unit": "bits (log₂)",
}


# ======================================================================
# Single-seed experiment
# ======================================================================

def run_single_seed(seed: int = 42, verbose: bool = True) -> dict:
    """Run the Turn 1 Entropy experiment for one seed.

    Returns:
        Dict with all metrics for this seed.
    """
    np.random.seed(seed)

    if verbose:
        print("=" * 70)
        print("Turn 1 Entropy Experiment")
        print("=" * 70)
        print(f"\nSeed: {seed}")
        print(f"Max entropy for binary: H_max = log₂(2) = {np.log2(2):.4f} bits")

    # Data
    if verbose:
        print("\n[1] Generating data...")
    generator = EntropyDataGenerator()
    train_data = generator.generate_training_data(n_samples=1000)
    test_data = generator.generate_entropy_test_data(n_samples=100)
    if verbose:
        print(f"  Training samples: {len(train_data)}")
        print(f"  Test samples: {len(test_data)}")

    # Tokenizer
    if verbose:
        print("\n[2] Building vocabulary...")
    tokenizer = SimpleTokenizer()
    all_texts = (
        [d["turn1"] for d in train_data]
        + [d["turn2"] for d in train_data]
        + [d["turn1"] for d in test_data]
    )
    tokenizer.fit(all_texts)
    if verbose:
        print(f"  Vocabulary size: {tokenizer.vocab_size}")
        print(f"  'bank' token index: {tokenizer.bank_idx}")

    # Train Baseline
    if verbose:
        print("\n[3] Training Baseline...")
    baseline = BaselineModel(
        vocab_size=tokenizer.vocab_size,
        embed_dim=32,
        hidden_dim=64,
    )
    train_model(baseline, train_data, tokenizer, epochs=100, is_nrr=False,
                verbose=verbose)

    # Train NRR-lite
    if verbose:
        print("\n[4] Training NRR-lite...")
    nrr_lite = NRRLiteModel(
        vocab_size=tokenizer.vocab_size,
        embed_dim=32,
        hidden_dim=64,
        bank_idx=tokenizer.bank_idx,
    )
    train_model(nrr_lite, train_data, tokenizer, epochs=100, is_nrr=True,
                verbose=verbose)

    # Evaluate Turn 1 Entropy
    if verbose:
        print("\n[5] Evaluating Turn 1 Entropy...")
    baseline_entropy = evaluate_turn1_entropy(baseline, test_data, tokenizer)
    nrr_entropy = evaluate_turn1_entropy(nrr_lite, test_data, tokenizer)

    # Evaluate with context
    if verbose:
        print("\n[6] Evaluating with context (sanity check)...")
    baseline_context = evaluate_with_context(
        baseline, test_data, tokenizer, generator
    )
    nrr_context = evaluate_with_context(
        nrr_lite, test_data, tokenizer, generator
    )

    # Display results
    if verbose:
        print("\n" + "=" * 70)
        print("RESULTS")
        print("=" * 70)

        print("\n  Turn 1 Entropy (Turn 2 = neutral) [bits]")
        print("  " + "-" * 50)
        print(f"  Baseline:  H = {baseline_entropy['mean_entropy']:.3f}"
              f" +/- {baseline_entropy['std_entropy']:.3f} bits")
        print(f"  NRR-lite:  H = {nrr_entropy['mean_entropy']:.3f}"
              f" +/- {nrr_entropy['std_entropy']:.3f} bits")
        print(f"  Max H:         {np.log2(2):.3f} bits (perfect ambiguity)")

        print("\n  Accuracy WITH Turn 2 Context")
        print("  " + "-" * 50)
        print(f"  Baseline:  Financial {baseline_context['financial_acc']:.1%}"
              f", River {baseline_context['river_acc']:.1%}")
        print(f"  NRR-lite:  Financial {nrr_context['financial_acc']:.1%}"
              f", River {nrr_context['river_acc']:.1%}")

        if nrr_entropy.get("gate_entropies"):
            gate_h = float(np.mean(nrr_entropy["gate_entropies"]))
            print(f"\n  NRR-lite Gate Entropy at Turn 1")
            print("  " + "-" * 50)
            print(f"  Gate H = {gate_h:.3f} bits (ideal: {np.log2(2):.3f} bits)")

    return {
        "seed": seed,
        "baseline_h_mean": baseline_entropy["mean_entropy"],
        "baseline_h_std": baseline_entropy["std_entropy"],
        "nrr_h_mean": nrr_entropy["mean_entropy"],
        "nrr_h_std": nrr_entropy["std_entropy"],
        "gate_entropy": (
            float(np.mean(nrr_entropy["gate_entropies"]))
            if nrr_entropy.get("gate_entropies")
            else None
        ),
        "baseline_context_acc": baseline_context["overall_acc"],
        "nrr_context_acc": nrr_context["overall_acc"],
    }


# ======================================================================
# Multi-seed experiment
# ======================================================================

def run_multi_seed(seeds=None, verbose: bool = True) -> dict:
    """Run the experiment across multiple seeds and aggregate.

    Returns:
        Dict with per-seed results and aggregate statistics.
    """
    if seeds is None:
        seeds = PAPER1_TABLE1["seeds"]

    if verbose:
        print("\n" + "#" * 70)
        print("# MULTI-SEED TURN 1 ENTROPY EXPERIMENT")
        print("#" * 70)

    all_results = []
    for seed in seeds:
        if verbose:
            print(f"\n{'=' * 60}")
            print(f"Running with seed {seed}")
            print(f"{'=' * 60}")
        result = run_single_seed(seed=seed, verbose=verbose)
        all_results.append(result)

    # Aggregate
    baseline_hs = [r["baseline_h_mean"] for r in all_results]
    nrr_hs = [r["nrr_h_mean"] for r in all_results]
    h_diffs = [n - b for n, b in zip(nrr_hs, baseline_hs)]

    mean_diff = float(np.mean(h_diffs))
    se_diff = float(np.std(h_diffs, ddof=1) / np.sqrt(len(h_diffs)))
    t_stat = mean_diff / se_diff if se_diff > 0 else 0.0
    nrr_wins = sum(1 for d in h_diffs if d > 0)

    summary = {
        "seeds": seeds,
        "per_seed": all_results,
        "baseline_h_mean": float(np.mean(baseline_hs)),
        "baseline_h_std": float(np.std(baseline_hs)),
        "nrr_h_mean": float(np.mean(nrr_hs)),
        "nrr_h_std": float(np.std(nrr_hs)),
        "mean_diff": mean_diff,
        "se_diff": se_diff,
        "t_statistic": t_stat,
        "nrr_wins": nrr_wins,
        "total_seeds": len(seeds),
        "unit": "bits (log₂)",
    }

    if verbose:
        print("\n" + "=" * 70)
        print("MULTI-SEED SUMMARY [bits]")
        print("=" * 70)
        print(f"\n{'Seed':<10} {'Baseline H':<15} {'NRR-lite H':<15}"
              f" {'Diff':<10} {'Winner'}")
        print("-" * 60)
        for r in all_results:
            diff = r["nrr_h_mean"] - r["baseline_h_mean"]
            winner = "NRR-lite" if diff > 0 else "Baseline"
            print(f"{r['seed']:<10} {r['baseline_h_mean']:<15.3f}"
                  f" {r['nrr_h_mean']:<15.3f} {diff:<10.3f} {winner}")
        print("-" * 60)
        print(f"{'Mean':<10} {summary['baseline_h_mean']:<15.3f}"
              f" {summary['nrr_h_mean']:<15.3f}")
        print(f"{'Std':<10} {summary['baseline_h_std']:<15.3f}"
              f" {summary['nrr_h_std']:<15.3f}")
        print(f"\nNRR-lite wins: {nrr_wins}/{len(seeds)}")
        print(f"Mean difference: {mean_diff:.3f} +/- {se_diff:.3f} bits")
        print(f"t-statistic: {t_stat:.2f}")
        if t_stat > 2:
            print("-> Statistically significant (t > 2)")

    return summary


# ======================================================================
# Verification against Paper 1
# ======================================================================

def verify_against_paper(summary: dict) -> dict:
    """Compare multi-seed results against Paper 1 Table 1.

    Returns:
        Dict with pass/fail for each metric.
    """
    ref = PAPER1_TABLE1
    tol = 0.03  # tolerance in bits

    checks = {
        "baseline_h_mean": {
            "expected": ref["baseline_h_mean"],
            "actual": summary["baseline_h_mean"],
            "pass": abs(summary["baseline_h_mean"] - ref["baseline_h_mean"]) < tol,
        },
        "nrr_h_mean": {
            "expected": ref["nrr_h_mean"],
            "actual": summary["nrr_h_mean"],
            "pass": abs(summary["nrr_h_mean"] - ref["nrr_h_mean"]) < tol,
        },
        "nrr_wins_all": {
            "expected": ref["n_seeds"],
            "actual": summary["nrr_wins"],
            "pass": summary["nrr_wins"] == summary["total_seeds"],
        },
        "t_significant": {
            "expected": "> 2.0",
            "actual": summary["t_statistic"],
            "pass": summary["t_statistic"] > 2.0,
        },
    }

    all_pass = all(c["pass"] for c in checks.values())
    checks["all_pass"] = all_pass

    print("\n" + "=" * 70)
    print("VERIFICATION AGAINST PAPER 1 TABLE 1")
    print("=" * 70)

    for name, check in checks.items():
        if name == "all_pass":
            continue
        status = "PASS" if check["pass"] else "FAIL"
        print(f"  [{status}] {name}: "
              f"expected {check['expected']}, got {check['actual']:.3f}"
              if isinstance(check["actual"], float)
              else f"  [{status}] {name}: "
              f"expected {check['expected']}, got {check['actual']}")

    print(f"\n  {'ALL CHECKS PASSED' if all_pass else 'SOME CHECKS FAILED'}")

    return checks


# ======================================================================
# Main
# ======================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Turn 1 Entropy Verification (Paper 1, Table 1)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for single-seed run (default: 42)",
    )
    parser.add_argument(
        "--multi", action="store_true",
        help="Run multi-seed experiment (5 seeds)",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Path to save JSON results",
    )
    args = parser.parse_args()

    if args.multi:
        summary = run_multi_seed(verbose=True)
        checks = verify_against_paper(summary)

        output = {
            "experiment": "Turn 1 Entropy Verification",
            "paper": "arXiv:2512.13478",
            "table": "Table 1",
            "unit": "bits (log₂)",
            "summary": summary,
            "verification": {
                k: {kk: vv for kk, vv in v.items()}
                for k, v in checks.items()
                if k != "all_pass"
            },
            "all_pass": checks["all_pass"],
        }
    else:
        result = run_single_seed(seed=args.seed, verbose=True)
        output = {
            "experiment": "Turn 1 Entropy Verification (single seed)",
            "paper": "arXiv:2512.13478",
            "table": "Table 1",
            "unit": "bits (log₂)",
            "result": result,
        }

    # Save results
    out_path = args.output
    if out_path is None:
        out_dir = os.path.join(os.path.dirname(__file__), "..", "results")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, "turn1_entropy_output.json")

    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to: {out_path}")


if __name__ == "__main__":
    main()
