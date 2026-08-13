"""Shared configuration for conceptual belief space analyses."""

from __future__ import annotations

from pathlib import Path

ACTIONS = ["UP", "DOWN", "LEFT", "RIGHT"]
DEFAULT_OUTPUT_DIR = Path("outputs/conceptual_belief_space")
DEFAULT_FIGURE_DIR = Path("figures/conceptual_belief_space")
DEFAULT_SEED = 42
EPS = 1e-12
