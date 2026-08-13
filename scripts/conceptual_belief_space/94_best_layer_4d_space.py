#!/usr/bin/env python3
"""A 4-d belief space assembled from the best layer for each ingredient, viewed in 2-d.

Every previous rendering handicapped itself somewhere: layer-averaged activations, or a
single layer forced to supply both content and strength, or only two action axes. This
builds the most favourable version available:

  coordinates 1-3  three LDA discriminant axes fitted on the layer where ACTION decodes
                   best (layer 17, balanced accuracy 0.521)
  coordinate 4     the ridge commitment direction fitted on the layer where COMMITMENT
                   decodes best (layer 15, Spearman 0.670)

The four coordinates come from different feature spaces and different scales, so they are
standardised on training statistics before PCA; otherwise the projection would simply follow
whichever coordinate happened to have the largest raw variance. PCA(2) is fitted on training
trajectories and applied to held-out ones, and the explained-variance split is reported --
if the four standardised coordinates are close to uncorrelated, their eigenvalues are close
to equal and the 2-d view is close to an arbitrary rotation, which the reader needs to know.
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
Q_COLS = [f"q_{a}" for a in ACTIONS]
STRICT_COL = "strict_before_any_action_mention"
ACTION_COLORS = {"UP": "#2a78d6", "DOWN": "#eb6834", "LEFT": "#1baf7a", "RIGHT": "#4a3aa7"}
INK, GRID, MUTED = "#222222", "#d8d8d8", "#5f6368"

HELMERT = np.array([[1, -1, 0, 0], [1, 1, -2, 0], [1, 1, 1, -3]], dtype=np.float64)
HELMERT = HELMERT / np.linalg.norm(HELMERT, axis=1, keepdims=True)


def parse_args() -> argparse.Namespace:
    base = REPO_ROOT / "outputs/conceptual_belief_space/cot_window_layer_transition_probe"
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--activation_index", type=Path, default=base / "activations/activation_index.parquet")
    p.add_argument("--beliefs", type=Path, default=base / "activations/extracted_beliefs.parquet")
    p.add_argument("--pooled_cache", type=Path, default=REPO_ROOT
                   / "outputs/conceptual_belief_space/layerwise_dissociation/pooled_cot_window_f16.npy")
    p.add_argument("--splits", type=Path, default=REPO_ROOT
                   / "outputs/conceptual_belief_space/splits/group_splits_mlx_full.json")
    p.add_argument("--out_dir", type=Path, default=REPO_ROOT
                   / "outputs/conceptual_belief_space/best_layer_4d_space")
    p.add_argument("--figure_dir", type=Path, default=REPO_ROOT / "figures/conceptual_belief_space")
    p.add_argument("--action_layer", type=int, default=17)
    p.add_argument("--commitment_layer", type=int, default=15)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def load_probe77() -> Any:
    path = REPO_ROOT / "scripts/conceptual_belief_space/77_train_forced_suffix_commitment_probe.py"
    spec = importlib.util.spec_from_file_location("probe77", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def arm_statistics(z: np.ndarray, labels: np.ndarray, commitment: np.ndarray) -> dict[str, float]:
    from scipy.stats import spearmanr

    z = z - z.mean(axis=0)
    radius = np.linalg.norm(z, axis=1)
    keys = [a for a in ACTIONS if (labels == a).any()]
    matrix = np.stack([z[labels == a].mean(axis=0) for a in keys])
    unit_c = matrix / np.maximum(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-12)
    cos = (z / np.maximum(radius[:, None], 1e-12)) @ unit_c.T
    own = np.array([keys.index(a) for a in labels])
    return {
        "own_centroid_cosine": float(np.mean(cos[np.arange(len(z)), own])),
        "angular_purity": float(np.mean(cos.argmax(axis=1) == own)),
        "radius_vs_commitment_rho": float(spearmanr(radius, commitment).statistic),
        "centroid_radius_ratio": float(np.mean(np.linalg.norm(matrix, axis=1)) / np.mean(radius)),
    }


def main() -> int:
    args = parse_args()
    from sklearn.decomposition import PCA
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler

    probe77 = load_probe77()
    table = probe77.prepare_table(args.activation_index, args.beliefs, None)
    splits = probe77.load_outer_splits(args.splits, table)
    split = next(s for s in splits if int(s["split_id"]) == 0)
    cached = np.load(args.pooled_cache.with_suffix(".prefix_ids.npy"))
    if not (cached == table["prefix_id"].to_numpy()).all():
        raise SystemExit("pooled cache does not match this table ordering")
    pooled = np.load(args.pooled_cache, mmap_mode="r")

    keep = table[STRICT_COL].to_numpy(dtype=bool)
    train_idx = split["train_idx"][keep[split["train_idx"]]]
    test_idx = split["test_idx"][keep[split["test_idx"]]]
    layer_features = lambda idx, layer: np.asarray(pooled[idx, layer], dtype=np.float32)

    labels_of = lambda idx: np.asarray(ACTIONS)[
        table.iloc[idx][Q_COLS].to_numpy(dtype=np.float64).argmax(axis=1)]
    commitment_of = lambda idx: table.iloc[idx]["entropy_commitment"].to_numpy(dtype=np.float64)
    labels_train, labels_test = labels_of(train_idx), labels_of(test_idx)
    commitment_train, commitment_test = commitment_of(train_idx), commitment_of(test_idx)
    print(f"[4d] action axes from layer {args.action_layer}, commitment axis from layer "
          f"{args.commitment_layer}; {len(train_idx)} train / {len(test_idx)} held-out")

    # Three action coordinates from the best action layer.
    action_scaler = StandardScaler().fit(layer_features(train_idx, args.action_layer))
    lda = LinearDiscriminantAnalysis(n_components=3).fit(
        action_scaler.transform(layer_features(train_idx, args.action_layer)), labels_train)
    action_train = lda.transform(action_scaler.transform(layer_features(train_idx, args.action_layer)))
    action_test = lda.transform(action_scaler.transform(layer_features(test_idx, args.action_layer)))

    # One commitment coordinate from the best commitment layer.
    commit_scaler = StandardScaler().fit(layer_features(train_idx, args.commitment_layer))
    ridge = Ridge(alpha=1.0).fit(
        commit_scaler.transform(layer_features(train_idx, args.commitment_layer)), commitment_train)
    direction = ridge.coef_ / np.linalg.norm(ridge.coef_)
    commit_train = commit_scaler.transform(layer_features(train_idx, args.commitment_layer)) @ direction
    commit_test = commit_scaler.transform(layer_features(test_idx, args.commitment_layer)) @ direction

    four_train = np.column_stack([action_train, commit_train])
    four_test = np.column_stack([action_test, commit_test])
    # Standardise so PCA is not simply chasing the coordinate with the largest raw scale.
    four_scaler = StandardScaler().fit(four_train)
    four_train_std, four_test_std = four_scaler.transform(four_train), four_scaler.transform(four_test)

    correlation = pd.DataFrame(np.corrcoef(four_train_std.T).round(3),
                               index=["action1", "action2", "action3", "commitment"],
                               columns=["action1", "action2", "action3", "commitment"])
    print("\ncorrelations among the four standardised coordinates (train):")
    print(correlation.to_string())

    pca = PCA(n_components=2, random_state=args.seed).fit(four_train_std)
    z2d = pca.transform(four_test_std)
    ratios = pca.explained_variance_ratio_
    all_ratios = PCA(n_components=4, random_state=args.seed).fit(
        four_train_std).explained_variance_ratio_
    print(f"\nexplained variance: PC1 {ratios[0]:.3f}, PC2 {ratios[1]:.3f} "
          f"(total {ratios.sum():.3f}); all four eigen-ratios "
          f"{np.round(all_ratios, 3).tolist()}")
    loadings = pd.DataFrame(pca.components_.T.round(3), index=correlation.index,
                            columns=["PC1", "PC2"])
    print("\nPCA loadings:")
    print(loadings.to_string())

    log_q = np.log(np.clip(table.iloc[test_idx][Q_COLS].to_numpy(dtype=np.float64), 1e-12, None))
    z_behaviour = (log_q - log_q.mean(axis=1, keepdims=True)) @ HELMERT.T
    behaviour_pca = PCA(n_components=2, random_state=args.seed).fit(z_behaviour)
    z_behaviour_2d = behaviour_pca.transform(z_behaviour)

    frame = pd.DataFrame([
        {"space": "behavioural belief, 3-d", **arm_statistics(z_behaviour, labels_test, commitment_test)},
        {"space": "behavioural belief, PCA 2-d", **arm_statistics(z_behaviour_2d, labels_test, commitment_test)},
        {"space": f"activation 4-d (act L{args.action_layer} + commit L{args.commitment_layer})",
         **arm_statistics(four_test_std, labels_test, commitment_test)},
        {"space": "activation 4-d, PCA 2-d", **arm_statistics(z2d, labels_test, commitment_test)},
    ])
    args.out_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out_dir / "arm_statistics.csv", index=False)
    correlation.to_csv(args.out_dir / "coordinate_correlations.csv")
    loadings.to_csv(args.out_dir / "pca_loadings.csv")
    print()
    print(frame.round(4).to_string(index=False))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.stats import spearmanr

    figure, axes = plt.subplots(1, 3, figsize=(18.0, 6.2))
    for axis, (title, zz) in zip(axes[:2], [
            ("Behavioural belief space\n(reference: arms known to be here)", z_behaviour_2d),
            (f"Activation 4-d space collapsed to 2-d\n(action axes L{args.action_layer}, "
             f"commitment axis L{args.commitment_layer})", z2d)]):
        flat = zz - zz.mean(axis=0)
        span = float(np.percentile(np.abs(flat), 99.2))
        for action in ACTIONS:
            m = labels_test == action
            axis.scatter(flat[m, 0], flat[m, 1], s=7, alpha=0.30, color=ACTION_COLORS[action],
                         linewidths=0, rasterized=True)
        for action in ACTIONS:
            m = labels_test == action
            c = flat[m].mean(axis=0)
            reach = span / max(np.linalg.norm(c), 1e-9)
            axis.plot([0, c[0] * reach], [0, c[1] * reach], color=ACTION_COLORS[action],
                      linewidth=2.0, alpha=0.75, zorder=4)
            axis.scatter([c[0]], [c[1]], s=190, color=ACTION_COLORS[action],
                         edgecolors="white", linewidths=2.2, zorder=6)
            axis.annotate(action, (c[0] * reach * 0.92, c[1] * reach * 0.92),
                          color=ACTION_COLORS[action], fontsize=13, fontweight="bold",
                          ha="center", va="center", zorder=7,
                          bbox=dict(boxstyle="round,pad=0.22", facecolor="white",
                                    edgecolor="none", alpha=0.78))
        rho = float(spearmanr(np.linalg.norm(flat, axis=1), commitment_test).statistic)
        axis.set_xlim(-span, span); axis.set_ylim(-span, span); axis.set_aspect("equal")
        axis.set_title(f"{title}\ndistance from centre vs commitment:  Spearman {rho:+.3f}",
                       color=INK, fontsize=12)
        axis.set_xticks([]); axis.set_yticks([])
        for sp in axis.spines.values():
            sp.set_color(GRID)

    axis = axes[2]
    positions = np.arange(4)
    axis.bar(positions - 0.2, np.abs(pca.components_[0]), 0.4, color="#2a78d6", label="PC1")
    axis.bar(positions + 0.2, np.abs(pca.components_[1]), 0.4, color="#eb6834", label="PC2")
    axis.set_xticks(positions, ["action 1", "action 2", "action 3", "commitment"],
                    rotation=15, ha="right", fontsize=10)
    axis.set_ylabel("|loading|", color=INK, fontsize=11)
    axis.set_title(f"What the 2-d view keeps\nPC1 {ratios[0]:.0%}, PC2 {ratios[1]:.0%} "
                   f"of variance", color=INK, fontsize=12)
    axis.legend(frameon=False, fontsize=10, labelcolor=INK)
    axis.grid(True, axis="y", color=GRID, linewidth=0.7); axis.set_axisbelow(True)
    axis.tick_params(colors=INK, labelsize=10)
    for sp in ("top", "right"):
        axis.spines[sp].set_visible(False)

    figure.suptitle("Four-dimensional belief space from the best layer for each ingredient, "
                    "collapsed to two dimensions", color=INK, fontsize=13.5)
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    fig_path = args.figure_dir / "best_layer_4d_2d_view.png"
    figure.savefig(fig_path, dpi=200)
    plt.close(figure)

    # Standalone large version of just the two scatters, for looking at.
    figure2, axes2 = plt.subplots(1, 2, figsize=(14.6, 7.4))
    for axis, (title, zz) in zip(axes2, [
            ("Behavioural belief space  (reference)", z_behaviour_2d),
            (f"Activation 4-d space  (action axes L{args.action_layer}, "
             f"commitment axis L{args.commitment_layer})", z2d)]):
        flat = zz - zz.mean(axis=0)
        span = float(np.percentile(np.abs(flat), 99.2))
        for action in ACTIONS:
            m = labels_test == action
            axis.scatter(flat[m, 0], flat[m, 1], s=8, alpha=0.32, color=ACTION_COLORS[action],
                         linewidths=0, rasterized=True)
        for action in ACTIONS:
            m = labels_test == action
            c = flat[m].mean(axis=0)
            reach = span / max(np.linalg.norm(c), 1e-9)
            axis.plot([0, c[0] * reach], [0, c[1] * reach], color=ACTION_COLORS[action],
                      linewidth=2.2, alpha=0.8, zorder=4)
            axis.scatter([c[0]], [c[1]], s=210, color=ACTION_COLORS[action],
                         edgecolors="white", linewidths=2.4, zorder=6)
            axis.annotate(action, (c[0] * reach * 0.9, c[1] * reach * 0.9),
                          color=ACTION_COLORS[action], fontsize=14, fontweight="bold",
                          ha="center", va="center", zorder=7,
                          bbox=dict(boxstyle="round,pad=0.24", facecolor="white",
                                    edgecolor="none", alpha=0.8))
        rho = float(spearmanr(np.linalg.norm(flat, axis=1), commitment_test).statistic)
        axis.set_xlim(-span, span); axis.set_ylim(-span, span); axis.set_aspect("equal")
        axis.set_title(f"{title}\ndistance from centre vs commitment:  Spearman {rho:+.3f}",
                       color=INK, fontsize=12.5)
        axis.set_xticks([]); axis.set_yticks([])
        for sp in axis.spines.values():
            sp.set_color(GRID)
    figure2.suptitle("Best layer for each ingredient: three action axes plus commitment, "
                     "collapsed to 2-d", color=INK, fontsize=13.5)
    figure2.tight_layout(rect=(0, 0, 1, 0.94))
    fig2_path = args.figure_dir / "best_layer_4d_2d_large.png"
    figure2.savefig(fig2_path, dpi=200)
    plt.close(figure2)

    lines = [
        "# 4-d belief space from the best layer per ingredient, viewed in 2-d",
        "",
        f"- Action axes: three LDA discriminants on layer {args.action_layer}. "
        f"Commitment axis: ridge direction on layer {args.commitment_layer}.",
        f"- {len(train_idx)} training-trajectory prefixes fitted, {len(test_idx)} held out.",
        f"- Explained variance of the 2-d view: PC1 {ratios[0]:.3f}, PC2 {ratios[1]:.3f}, "
        f"total {ratios.sum():.3f}. All four eigen-ratios: {np.round(all_ratios, 3).tolist()}.",
        "",
        "## Correlations among the four coordinates (train)",
        "",
        correlation.to_markdown(),
        "",
        "## PCA loadings",
        "",
        loadings.to_markdown(),
        "",
        "## Arm statistics",
        "",
        frame.round(4).to_markdown(index=False),
        "",
        f"Figure: `{fig_path}`",
    ]
    (args.out_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nWrote {args.out_dir}/report.md and {fig_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
