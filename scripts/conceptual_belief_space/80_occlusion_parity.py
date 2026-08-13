#!/usr/bin/env python3
"""Occlusion parity: layer ablations read from BOTH heads of the trained fold-0 probe (D2d).

The existing occlusion diagnostic (77) reports only the commitment head. Because the same
trained transformer also carries a q head, ablating a layer and reading both heads on the
same rows yields apples-to-apples occlusion curves for commitment (Spearman drop) and for
action identity (top-action balanced-accuracy drop). Run on the fold-0 held-out rows, in
two scopes: all test prefixes and the strict pre-action-mention subset.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

Q_COLS = [f"q_{action}" for action in ("UP", "DOWN", "LEFT", "RIGHT")]
STRICT_COL = "strict_before_any_action_mention"

INK = "#333333"
GRID = "#e6e6e6"
SCOPE_COLORS = {"all": "#2a78d6", "strict": "#eb6834"}


def parse_args() -> argparse.Namespace:
    base = REPO_ROOT / "outputs/conceptual_belief_space/cot_window_layer_transition_probe"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path,
                        default=base / "fold0_20260721/models/bidirectional_probe_split0.pt")
    parser.add_argument("--method", type=str, default="bidirectional_transformer")
    parser.add_argument("--activation_index", type=Path, default=base / "activations/activation_index.parquet")
    parser.add_argument("--beliefs", type=Path, default=base / "activations/extracted_beliefs.parquet")
    parser.add_argument("--splits", type=Path,
                        default=REPO_ROOT / "outputs/conceptual_belief_space/splits/group_splits_mlx_full.json")
    parser.add_argument("--split_id", type=int, default=0)
    parser.add_argument("--max_rows", type=int, default=4096)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out_dir", type=Path,
                        default=REPO_ROOT / "outputs/conceptual_belief_space/layerwise_dissociation")
    parser.add_argument("--figure_dir", type=Path, default=REPO_ROOT / "figures/conceptual_belief_space")
    return parser.parse_args()


def load_probe77() -> Any:
    path = REPO_ROOT / "scripts/conceptual_belief_space/77_train_forced_suffix_commitment_probe.py"
    spec = importlib.util.spec_from_file_location("probe77", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def main() -> int:
    args = parse_args()
    probe77 = load_probe77()
    import torch
    from sklearn.metrics import balanced_accuracy_score

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    table = probe77.prepare_table(args.activation_index, args.beliefs, None)
    splits = probe77.load_outer_splits(args.splits, table)
    split = next(item for item in splits if int(item["split_id"]) == args.split_id)
    test_idx = np.asarray(split["test_idx"], dtype=np.int64)
    strict_mask = table[STRICT_COL].to_numpy(dtype=bool)
    action_codes = table[Q_COLS].to_numpy(dtype=np.float64).argmax(axis=1)

    model, _payload, config = probe77.load_checkpoint_model(args.checkpoint, args.method, device)
    print(f"[occlusion-parity] restored {args.method} on {device}; n_layers={config.n_layers}")

    scope_indices = {
        "all": test_idx,
        "strict": test_idx[strict_mask[test_idx]],
    }
    rows: list[dict[str, Any]] = []
    for scope, indices in scope_indices.items():
        rng = np.random.default_rng(args.seed + args.split_id)
        selected = indices
        if len(selected) > args.max_rows:
            selected = np.sort(rng.choice(selected, size=args.max_rows, replace=False))
        dataset = probe77.ActivationDataset(table, selected)
        base = probe77.predict_torch_model(model, dataset, batch_size=args.batch_size, device=device)
        truth_commitment = table.iloc[base["row_idx"]]["entropy_commitment"].to_numpy(dtype=np.float64)
        truth_action = action_codes[base["row_idx"]]
        base_spearman = probe77.safe_spearman(truth_commitment, base["commitment"])
        base_balanced = float(balanced_accuracy_score(truth_action, base["q"].argmax(axis=1)))
        print(f"[occlusion-parity] scope={scope} n={len(selected)} "
              f"base: commit_spearman={base_spearman:.4f} action_balanced={base_balanced:.4f}", flush=True)
        for layer in range(config.n_layers):
            occluded = probe77.predict_torch_model(
                model, dataset, batch_size=args.batch_size, device=device, ablate_layer=layer)
            occluded_spearman = probe77.safe_spearman(truth_commitment, occluded["commitment"])
            occluded_balanced = float(balanced_accuracy_score(truth_action, occluded["q"].argmax(axis=1)))
            rows.append({
                "scope": scope, "layer": layer, "n_rows": len(selected),
                "base_commit_spearman": base_spearman,
                "occluded_commit_spearman": occluded_spearman,
                "commit_spearman_drop": base_spearman - occluded_spearman,
                "base_action_balanced": base_balanced,
                "occluded_action_balanced": occluded_balanced,
                "action_balanced_drop": base_balanced - occluded_balanced,
            })
            print(f"[occlusion-parity] scope={scope} layer {layer:2d}: "
                  f"commit_drop={base_spearman - occluded_spearman:+.4f} "
                  f"action_drop={base_balanced - occluded_balanced:+.4f}", flush=True)

    frame = pd.DataFrame(rows)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out_dir / "occlusion_parity.csv", index=False)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    args.figure_dir.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    specs = ((axes[0], "commit_spearman_drop", "commitment Spearman drop"),
             (axes[1], "action_balanced_drop", "top-action balanced-accuracy drop"))
    for axis, column, label in specs:
        for scope, group in frame.groupby("scope"):
            axis.plot(group.layer, group[column], color=SCOPE_COLORS[scope], linewidth=2,
                      linestyle="-" if scope == "all" else "--")
        axis.set_title(label, color=INK, fontsize=11)
        axis.set_xlabel("occluded layer", color=INK)
        axis.set_ylabel(label, color=INK)
        axis.grid(True, color=GRID, linewidth=0.8)
        axis.tick_params(colors=INK)
        for spine in ("top", "right"):
            axis.spines[spine].set_visible(False)
    axes[0].legend(handles=[plt.Line2D([], [], color=SCOPE_COLORS["all"], linewidth=2, label="all test prefixes"),
                            plt.Line2D([], [], color=SCOPE_COLORS["strict"], linewidth=2, linestyle="--",
                                       label="strict pre-mention")],
                   loc="upper right", frameon=False, fontsize=9, labelcolor=INK)
    figure.suptitle("Layer occlusion read from both heads of the same fold-0 probe", color=INK)
    figure.tight_layout()
    figure_path = args.figure_dir / "occlusion_parity.png"
    figure.savefig(figure_path, dpi=200)
    plt.close(figure)

    top = frame.sort_values("commit_spearman_drop", ascending=False).groupby("scope").head(3)
    lines = [
        "# Occlusion Parity (fold-0 transformer, both heads)",
        "",
        f"Checkpoint: `{args.checkpoint.name}`; scopes: all test prefixes and strict pre-mention;",
        f"{args.max_rows} rows max per scope; layer content zeroed via `layer_keep_mask`.",
        "",
        frame.round(4).to_markdown(index=False),
        "",
        "Top-3 commitment-drop layers per scope:",
        "",
        top[["scope", "layer", "commit_spearman_drop", "action_balanced_drop"]].round(4).to_markdown(index=False),
    ]
    (args.out_dir / "occlusion_parity.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {args.out_dir / 'occlusion_parity.csv'} and {figure_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
