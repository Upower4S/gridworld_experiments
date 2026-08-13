#!/usr/bin/env python3
"""Conjecture C10: is there a static prior direction in the pre-CoT state?

Proposition 2.8 found a large prior asymmetry: before any reasoning, mean commitment is
0.664 for RIGHT steps but 0.144 for UP/DOWN steps, and RIGHT steps settle earliest. C10
says this prior is already linearly present in the prompt-final residual state, and that
its RIGHT component predicts which steps settle early.

The prompt-only cut (prefix_token_index = 0) has exactly one valid window token per step
(verified: 1276/1276 rows), so its activation is literally the last state before any CoT
token exists. There is no positional confound to control here -- every row sits at the
same position -- so the baseline is simply the class prior.
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

ACTIONS = ("UP", "DOWN", "LEFT", "RIGHT")
Q_COLS = [f"q_{action}" for action in ACTIONS]

INK = "#333333"
GRID = "#e6e6e6"
ACTION_COLORS = {"UP": "#2a78d6", "DOWN": "#eb6834", "LEFT": "#1baf7a", "RIGHT": "#4a3aa7"}


def parse_args() -> argparse.Namespace:
    base = REPO_ROOT / "outputs/conceptual_belief_space/cot_window_layer_transition_probe"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--activation_index", type=Path, default=base / "activations/activation_index.parquet")
    parser.add_argument("--beliefs", type=Path, default=base / "activations/extracted_beliefs.parquet")
    parser.add_argument("--pooled_cache", type=Path,
                        default=REPO_ROOT / "outputs/conceptual_belief_space/layerwise_dissociation/pooled_cot_window_f16.npy")
    parser.add_argument("--boundary_steps", type=Path,
                        default=REPO_ROOT / "outputs/commitment_boundary_steps.parquet")
    parser.add_argument("--out_dir", type=Path, default=REPO_ROOT / "outputs/conjectures/c10_prompt_prior")
    parser.add_argument("--figure_dir", type=Path, default=REPO_ROOT / "figures/conjectures")
    parser.add_argument("--layers", type=str, default="0,3,6,10,15,18,23,mean")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
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
    from scipy.stats import spearmanr
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.metrics import balanced_accuracy_score, r2_score
    from sklearn.model_selection import GroupKFold
    from sklearn.preprocessing import StandardScaler

    probe77 = load_probe77()
    table = probe77.prepare_table(args.activation_index, args.beliefs, None)
    cached_ids = np.load(args.pooled_cache.with_suffix(".prefix_ids.npy"))
    if not (cached_ids == table["prefix_id"].to_numpy()).all():
        raise SystemExit("pooled cache does not correspond to this table ordering")
    pooled = np.load(args.pooled_cache, mmap_mode="r")

    prompt_mask = table["prefix_token_index"].to_numpy() == 0
    rows = np.nonzero(prompt_mask)[0]
    meta = table.iloc[rows].reset_index(drop=True)
    q_matrix = meta[Q_COLS].to_numpy(dtype=np.float64)
    labels = q_matrix.argmax(axis=1)
    groups = meta["trajectory_id"].to_numpy()
    print(f"prompt-only rows: {len(rows)}; trajectories: {len(set(groups))}; "
          f"argmax distribution: {np.bincount(labels, minlength=4) / len(labels)}")

    boundary = pd.read_parquet(args.boundary_steps, columns=[
        "trajectory_id", "step_id", "stabilization_progress", "agent_action",
        "num_cot_tokens", "prompt_only_commitment"])
    meta = meta.drop(columns=[c for c in ("agent_action", "num_cot_tokens") if c in meta.columns])
    meta = meta.merge(boundary, on=["trajectory_id", "step_id"], how="left", validate="one_to_one")

    layer_tokens = [token.strip() for token in args.layers.split(",") if token.strip()]
    splitter = GroupKFold(n_splits=args.folds)
    result_rows: list[dict[str, Any]] = []
    best_predictions: dict[str, np.ndarray] = {}

    for token in layer_tokens:
        block = np.asarray(pooled[rows]) if token == "mean" else None
        features = (block.mean(axis=1, dtype=np.float32) if token == "mean"
                    else np.asarray(pooled[rows, int(token)], dtype=np.float32))
        name = "all_layer_mean" if token == "mean" else f"layer_{int(token)}"

        oof_q = np.zeros_like(q_matrix)
        oof_label = np.zeros(len(features), dtype=int)
        for train_idx, test_idx in splitter.split(features, labels, groups):
            scaler = StandardScaler().fit(features[train_idx])
            x_train, x_test = scaler.transform(features[train_idx]), scaler.transform(features[test_idx])
            # Inner grouped selection of the ridge penalty (n=1276, d=512).
            inner = GroupKFold(n_splits=3)
            best_alpha, best_score = None, -np.inf
            for alpha in (1.0, 10.0, 100.0, 1000.0):
                scores = []
                for inner_train, inner_val in inner.split(x_train, labels[train_idx], groups[train_idx]):
                    model = Ridge(alpha=alpha).fit(x_train[inner_train], q_matrix[train_idx][inner_train])
                    scores.append(r2_score(q_matrix[train_idx][inner_val],
                                           model.predict(x_train[inner_val])))
                if np.mean(scores) > best_score:
                    best_alpha, best_score = alpha, float(np.mean(scores))
            oof_q[test_idx] = Ridge(alpha=best_alpha).fit(x_train, q_matrix[train_idx]).predict(x_test)
            oof_label[test_idx] = LogisticRegression(
                max_iter=2000, class_weight="balanced", C=0.1,
                random_state=args.seed).fit(x_train, labels[train_idx]).predict(x_test)

        entry = {
            "representation": name,
            "r2_overall": float(r2_score(q_matrix, oof_q)),
            "balanced_accuracy": float(balanced_accuracy_score(labels, oof_label)),
        }
        for index, action in enumerate(ACTIONS):
            entry[f"r2_{action}"] = float(r2_score(q_matrix[:, index], oof_q[:, index]))
        result_rows.append(entry)
        best_predictions[name] = oof_q
        print(f"{name:16s} R2={entry['r2_overall']:.4f} balanced_acc={entry['balanced_accuracy']:.4f} "
              f"R2_RIGHT={entry['r2_RIGHT']:.4f}")

    results = pd.DataFrame(result_rows)
    best_name = results.loc[results["r2_overall"].idxmax(), "representation"]
    predicted = best_predictions[best_name]

    stabilization = meta["stabilization_progress"].to_numpy(dtype=float)
    correlations = {
        "predicted_q_RIGHT_vs_stabilization": spearmanr(predicted[:, 3], stabilization).statistic,
        "true_q_RIGHT_vs_stabilization": spearmanr(q_matrix[:, 3], stabilization).statistic,
        "true_prompt_commitment_vs_stabilization": spearmanr(
            meta["prompt_only_commitment"].to_numpy(dtype=float), stabilization).statistic,
        "predicted_prompt_margin_vs_stabilization": spearmanr(
            np.sort(predicted, axis=1)[:, -1] - np.sort(predicted, axis=1)[:, -2], stabilization).statistic,
    }
    per_action = meta.assign(predicted_right=predicted[:, 3]).groupby("agent_action").agg(
        mean_predicted_q_RIGHT=("predicted_right", "mean"),
        mean_true_q_RIGHT=(Q_COLS[3], "mean"),
        median_stabilization=("stabilization_progress", "median"),
        n=("step_id", "size")).round(4)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.out_dir / "decoding.csv", index=False)
    per_action.to_csv(args.out_dir / "per_action.csv")
    pd.DataFrame([correlations]).to_csv(args.out_dir / "correlations.csv", index=False)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    args.figure_dir.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, 2, figsize=(11.5, 4.3))
    physical = results[results.representation != "all_layer_mean"].copy()
    physical["layer"] = physical.representation.str.replace("layer_", "").astype(int)
    physical = physical.sort_values("layer")
    axes[0].plot(physical["layer"], physical["r2_overall"], color="#2a78d6", linewidth=2,
                 marker="o", markersize=5)
    axes[0].axhline(0.0, color=INK, linewidth=1.2, linestyle=":")
    axes[0].annotate("predicting the mean", (physical["layer"].iloc[0], 0.0),
                     textcoords="offset points", xytext=(0, 5), color=INK, fontsize=8)
    axes[0].set_xlabel("layer (block output)", color=INK)
    axes[0].set_ylabel("held-out $R^2$ for the prompt-only belief", color=INK)
    axes[0].set_title("Is the pre-CoT belief linearly present?", color=INK, fontsize=11)

    for action in ACTIONS:
        subset = meta["agent_action"] == action
        axes[1].scatter(predicted[subset.to_numpy(), 3], stabilization[subset.to_numpy()],
                        s=16, alpha=0.55, color=ACTION_COLORS[action], edgecolors="white",
                        linewidths=0.4, label=action)
    axes[1].set_xlabel("out-of-fold predicted prompt-only $q_{\\mathrm{RIGHT}}$", color=INK)
    axes[1].set_ylabel("argmax stabilization progress", color=INK)
    axes[1].set_title(f"Prior RIGHT-ness vs settling time "
                      f"($\\rho$={correlations['predicted_q_RIGHT_vs_stabilization']:.3f})",
                      color=INK, fontsize=11)
    axes[1].legend(frameon=False, fontsize=8, labelcolor=INK, loc="upper right")
    for axis in axes:
        axis.grid(True, color=GRID, linewidth=0.8)
        axis.tick_params(colors=INK)
        for spine in ("top", "right"):
            axis.spines[spine].set_visible(False)
    figure.suptitle("C10: a static prior direction in the pre-CoT state", color=INK)
    figure.tight_layout()
    figure_path = args.figure_dir / "c10_prompt_prior.png"
    figure.savefig(figure_path, dpi=200)
    plt.close(figure)

    best_row = results.set_index("representation").loc[best_name]
    part_a = bool(best_row["r2_overall"] > 0.1 and best_row["balanced_accuracy"] > 0.35)
    part_b = bool(correlations["predicted_q_RIGHT_vs_stabilization"] < -0.1)
    lines = [
        "# Conjecture C10: a static default-action direction in the prompt state",
        "",
        f"- {len(meta)} steps (one prompt-only prefix each); single valid window token per row;",
        f"  trajectory-grouped {args.folds}-fold CV with inner alpha selection. Chance balanced",
        f"  accuracy 0.25.",
        "",
        "## Decoding the pre-CoT belief from the pre-CoT state",
        "",
        results.round(4).to_markdown(index=False),
        "",
        "## Does prior RIGHT-ness predict early settling?",
        "",
        *[f"- {key}: {value:+.4f}" for key, value in correlations.items()],
        "",
        per_action.to_markdown(),
        "",
        "## Verdict",
        "",
        f"- (a) prompt-only belief linearly decodable (best {best_name}: "
        f"R2 {best_row['r2_overall']:.4f}, balanced acc {best_row['balanced_accuracy']:.4f}): {part_a}",
        f"- (b) prior RIGHT-component predicts earlier stabilization "
        f"(rho {correlations['predicted_q_RIGHT_vs_stabilization']:+.4f} < -0.1): {part_b}",
        "",
        f"Figure: `{figure_path}`",
    ]
    (args.out_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print()
    print(results.round(4).to_string(index=False))
    print()
    for key, value in correlations.items():
        print(f"{key}: {value:+.4f}")
    print(f"\nVerdict: (a)={part_a}, (b)={part_b}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
