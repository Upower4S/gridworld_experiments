#!/usr/bin/env python3
"""Create directories for conceptual belief space analyses."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.conceptual_belief_space.config import DEFAULT_FIGURE_DIR, DEFAULT_OUTPUT_DIR
from src.conceptual_belief_space.io import ensure_dir, save_markdown


OUTPUT_SUBDIRS = [
    "",
    "splits",
    "extreme_examples",
    "behavior_manifold",
    "activation_manifold",
    "geometry_alignment",
    "probes",
    "trajectories",
    "strict_pre_action",
    "controls",
    "task_concepts",
    "models",
]

FIGURE_SUBDIRS = [
    "",
    "behavior",
    "activation",
    "geometry_alignment",
    "probes",
    "trajectories",
    "strict_pre_action",
    "controls",
]

README_TEXT = """# Conceptual Belief Space Pipeline

This directory is for the paper-style conceptual belief space analysis of the gridworld commitment experiment.

Blind PCA is diagnostic only. The main method uses extreme-example-anchored manifolds, action centroid distance matrices, and sparse probes from activation states to behavior action-belief vectors.

Behavior action-belief vectors are constructed over `UP`, `DOWN`, `LEFT`, and `RIGHT`. Activation analyses should use the existing extracted activation features; this pipeline does not extract model activations.

All fitting must be train-only under grouped splits, preferably by `trajectory_id` and otherwise by `decision_id`. Extreme examples, centroids, manifold anchors, preprocessing parameters, and probes must be selected or fit on training rows only before evaluating held-out decisions.
"""


def main() -> None:
    created: list[Path] = []
    for subdir in OUTPUT_SUBDIRS:
        path = REPO_ROOT / DEFAULT_OUTPUT_DIR / subdir
        ensure_dir(path)
        created.append(path)
    for subdir in FIGURE_SUBDIRS:
        path = REPO_ROOT / DEFAULT_FIGURE_DIR / subdir
        ensure_dir(path)
        created.append(path)

    readme_path = REPO_ROOT / DEFAULT_OUTPUT_DIR / "README.md"
    save_markdown(readme_path, README_TEXT)

    print(f"Ensured {len(created)} directories")
    print(f"Wrote {readme_path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
