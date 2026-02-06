"""
CSWS Data Generator for Turn 1 Entropy Experiment

Generates two-turn disambiguation data for the Context-Switch Word Sense
(CSWS) task.  The ambiguous token "bank" appears in Turn 1 with a neutral
adjective; Turn 2 supplies a financial or river-related cue.

Reference: Saito, K. (2025). "NRR-Core: Non-Resolution Reasoning as a 
    Computational Framework for Contextual Identity and Ambiguity 
    Preservation". arXiv:2512.13478, Section 6.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np


class EntropyDataGenerator:
    """Generate training and evaluation data for Turn 1 Entropy measurement.

    Turn 1 is always ambiguous ("The bank is {adj}.").  Turn 2 provides
    either a financial cue (label 0) or a river cue (label 1).

    Attributes:
        ambiguous_word: The polysemous target token.
        neutral_adjs: Adjectives that do not disambiguate.
        financial_cues: Nouns implying the financial sense.
        river_cues: Nouns implying the river sense.
    """

    def __init__(self) -> None:
        self.ambiguous_word = "bank"

        self.neutral_adjs = ["solid", "stable", "old", "new", "quiet", "large"]

        self.financial_cues = [
            "investor", "account", "loan", "deposit", "money", "interest",
        ]
        self.river_cues = [
            "ducks", "river", "water", "fish", "boat", "shore",
        ]

        self.financial_templates = [
            "The {cue} is important.",
            "The {cue} arrived today.",
            "Check the {cue}.",
        ]
        self.river_templates = [
            "The {cue} are nearby.",
            "The {cue} is beautiful.",
            "Look at the {cue}.",
        ]

    # ------------------------------------------------------------------
    # Single-turn generators
    # ------------------------------------------------------------------

    def generate_turn1(self) -> str:
        """Return an ambiguous Turn 1 sentence."""
        adj = np.random.choice(self.neutral_adjs)
        return f"The {self.ambiguous_word} is {adj}."

    def generate_turn2_financial(self) -> str:
        """Return a Turn 2 sentence with a financial cue."""
        cue = np.random.choice(self.financial_cues)
        template = np.random.choice(self.financial_templates)
        return template.format(cue=cue)

    def generate_turn2_river(self) -> str:
        """Return a Turn 2 sentence with a river cue."""
        cue = np.random.choice(self.river_cues)
        template = np.random.choice(self.river_templates)
        return template.format(cue=cue)

    # ------------------------------------------------------------------
    # Dataset generators
    # ------------------------------------------------------------------

    def generate_training_data(self, n_samples: int = 1000) -> List[Dict]:
        """Generate balanced training pairs (50 % financial, 50 % river).

        Returns:
            List of dicts with keys ``turn1``, ``turn2``, ``label``.
        """
        data: List[Dict] = []
        for _ in range(n_samples):
            turn1 = self.generate_turn1()
            if np.random.rand() > 0.5:
                turn2 = self.generate_turn2_financial()
                label = 0  # FINANCIAL
            else:
                turn2 = self.generate_turn2_river()
                label = 1  # RIVER
            data.append({"turn1": turn1, "turn2": turn2, "label": label})
        return data

    def generate_entropy_test_data(self, n_samples: int = 100) -> List[Dict]:
        """Generate Turn 1-only samples for entropy measurement.

        Returns:
            List of dicts with key ``turn1``.
        """
        return [{"turn1": self.generate_turn1()} for _ in range(n_samples)]
