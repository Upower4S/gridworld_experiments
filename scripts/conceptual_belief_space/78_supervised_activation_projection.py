#!/usr/bin/env python3
"""Supervised vs unsupervised low-dimensional projection of pre-verbal activations.

Asks whether the four action-specific arms of the behavioural manifold have an
activation-side counterpart. Unsupervised PCA maximises total variance, which in
the residual stream is dominated by token content and position, so a low-variance
action-identity direction never reaches the top components. This stage compares
PCA against two supervised projections of the same activations:

  * LDA on the four forced-answer top actions (3 discriminant directions);
  * a linear regression onto the CLR simplex coordinates (3 degrees of freedom,
    matching the behavioural manifold's own dimensionality).

Restricted to strict pre-action-mention prefixes of a held-out outer fold, so the
comparison is the activation-side analogue of the behavioural strict-subset PCA.
All reported accuracies fit the projection inside trajectory-grouped CV folds;
the figure is fitted on the full held-out subset and is descriptive only.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.conceptual_belief_space.forced_suffix_probe import (  # noqa: E402
    ACTIONS,
    ProjectedShardStore,
)

# Validated categorical palette (all-pairs, light mode): worst CVD dE 9.2,
# worst normal-vision dE 16.3. Aqua sits below 3:1 on the surface, so the
# relief rule applies and the figure carries direct centroid labels.
ACTION_COLORS = {"UP": "#2a78d6", "DOWN": "#eb6834", "LEFT": "#1baf7a", "RIGHT": "#4a3aa7"}
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_MUTED = "#52514e"
Q_COLS = [f"q_{action}" for action in ACTIONS]
CLR_COLS = [f"clr_{action}" for action in ACTIONS]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    base = Path("outputs/conceptual_belief_space/cot_window_layer_transition_probe")
    parser.add_argument("--activation_index", type=Path, default=base / "activations/activation_index.parquet")
    parser.add_argument("--beliefs", type=Path, default=base / "activations/extracted_beliefs.parquet")
    parser.add_argument("--splits", type=Path, default=Path("outputs/conceptual_belief_space/splits/group_splits_mlx_full.json"))
    parser.add_argument("--output_dir", type=Path, default=base / "supervised_projection")
    parser.add_argument("--figure_dir", type=Path, default=Path("figures/conceptual_belief_space"))
    parser.add_argument("--split_id", type=int, default=0)
    parser.add_argument("--layers", default="mean,3,10,18")
    parser.add_argument("--max_rows", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_pooled_activations(table: pd.DataFrame, indices: np.ndarray) -> np.ndarray:
    """Mean-pool each row's CoT window over its valid tokens -> [rows, layers, dim]."""
    store = ProjectedShardStore(max_open=16)
    columns = ["state_path", "rms_path", "shard_row", "shard_n_rows", "n_layers", "n_input_tokens", "projected_dim"]
    selected = table.iloc[indices]
    records = selected[columns].to_dict(orient="records")
    masks = np.asarray(selected["_input_token_mask"].tolist(), dtype=np.float32)
    n_layers = int(selected["n_layers"].iloc[0])
    dim = int(selected["projected_dim"].iloc[0])
    out = np.zeros((len(records), n_layers, dim), dtype=np.float32)
    for position, record in enumerate(records):
        projected, _rms = store.load_row(record)
        state = np.asarray(projected, dtype=np.float32)
        weights = masks[position]
        total = float(weights.sum())
        if total <= 0:
            raise ValueError("row has no valid CoT token")
        out[position] = np.tensordot(weights, state, axes=([0], [1])) / total
        if position % 1000 == 0:
            print(f"  loaded {position}/{len(records)}", flush=True)
    return out


def grouped_balanced_accuracy(features: np.ndarray, labels: np.ndarray, groups: np.ndarray, *, mode: str, n_components: int, seed: int) -> float:
    """Fit the projection inside each training fold, then score a linear readout."""
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.decomposition import PCA
    from sklearn.metrics import balanced_accuracy_score
    from sklearn.model_selection import GroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    scores = []
    for train_idx, test_idx in GroupKFold(n_splits=5).split(features, labels, groups):
        x_train, x_test = features[train_idx], features[test_idx]
        y_train, y_test = labels[train_idx], labels[test_idx]
        if mode == "pca":
            projector = make_pipeline(StandardScaler(), PCA(n_components=n_components, random_state=seed))
            projector.fit(x_train)
        elif mode == "lda":
            projector = make_pipeline(StandardScaler(), LinearDiscriminantAnalysis(n_components=n_components))
            projector.fit(x_train, y_train)
        elif mode == "clr":
            # Regress activations onto the behavioural simplex coordinates, then
            # use the fitted 3-d prediction as the projection.
            projector = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
            projector.fit(x_train, CLR_TARGET[train_idx])
        else:
            raise ValueError(mode)
        z_train = projector.predict(x_train) if mode == "clr" else projector.transform(x_train)
        z_test = projector.predict(x_test) if mode == "clr" else projector.transform(x_test)
        readout = LogisticRegression(max_iter=2000, class_weight="balanced").fit(z_train, y_train)
        scores.append(balanced_accuracy_score(y_test, readout.predict(z_test)))
    return float(np.mean(scores))


def main() -> int:
    global CLR_TARGET
    args = parse_args()
    sys.path.insert(0, str(REPO_ROOT / "scripts" / "conceptual_belief_space"))
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "probe77", REPO_ROOT / "scripts/conceptual_belief_space/77_train_forced_suffix_commitment_probe.py"
    )
    probe77 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(probe77)

    table = probe77.prepare_table(args.activation_index, args.beliefs, None)
    splits = probe77.load_outer_splits(args.splits, table)
    split = next(s for s in splits if int(s["split_id"]) == int(args.split_id))
    test_idx = split["test_idx"]

    strict = table.iloc[test_idx]["strict_before_any_action_mention"].to_numpy(dtype=bool)
    indices = test_idx[strict]
    if args.max_rows and len(indices) > int(args.max_rows):
        # Sample across the fold rather than truncating, which would keep only the
        # first trajectories and leave too few groups for grouped CV.
        rng = np.random.default_rng(int(args.seed))
        indices = np.sort(rng.choice(indices, size=int(args.max_rows), replace=False))
    meta = table.iloc[indices]
    print(f"held-out fold {args.split_id}: {len(test_idx)} rows; strict pre-action: {len(indices)}")
    print(f"trajectories: {meta.trajectory_id.nunique()}")

    q = meta[Q_COLS].to_numpy(dtype=np.float64)
    labels = np.asarray(ACTIONS)[q.argmax(axis=1)]
    groups = meta["trajectory_id"].to_numpy()
    clr = meta[CLR_COLS].to_numpy(dtype=np.float64) if all(c in meta.columns for c in CLR_COLS) else None
    if clr is None:
        logq = np.log(np.clip(q, 1e-12, None))
        clr = logq - logq.mean(axis=1, keepdims=True)
    CLR_TARGET = clr

    print("loading activations...", flush=True)
    pooled = load_pooled_activations(table, indices)
    print(f"pooled activations: {pooled.shape}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.figure_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    representations = {}
    for token in [t.strip() for t in args.layers.split(",") if t.strip()]:
        if token == "mean":
            features, name = pooled.mean(axis=1), "all_layer_mean"
        else:
            features, name = pooled[:, int(token)], f"layer_{int(token)}"
        representations[name] = features
        for mode, n_components in [("pca", 3), ("lda", 3), ("clr", 3)]:
            score = grouped_balanced_accuracy(features, labels, groups, mode=mode, n_components=n_components, seed=args.seed)
            rows.append({"representation": name, "projection": mode, "balanced_accuracy": score})
            print(f"  {name:16s} {mode:4s} -> grouped balanced accuracy {score:.4f}", flush=True)

    metrics = pd.DataFrame(rows)
    metrics.to_csv(args.output_dir / "projection_balanced_accuracy.csv", index=False)
    chance = 1.0 / len(ACTIONS)
    majority = float(pd.Series(labels).value_counts(normalize=True).max())
    summary = {
        "split_id": int(args.split_id),
        "n_rows": int(len(indices)),
        "n_trajectories": int(meta.trajectory_id.nunique()),
        "chance_balanced_accuracy": chance,
        "majority_class_rate": majority,
        "label_distribution": pd.Series(labels).value_counts(normalize=True).round(4).to_dict(),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))

    make_figure(representations["all_layer_mean"], labels, clr, metrics, args)
    return 0


def make_figure(features: np.ndarray, labels: np.ndarray, clr: np.ndarray, metrics: pd.DataFrame, args) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.linear_model import Ridge
    from sklearn.decomposition import PCA
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    panels = []
    pca = make_pipeline(StandardScaler(), PCA(n_components=2, random_state=args.seed)).fit(features)
    panels.append(("Unsupervised PCA", pca.transform(features), "pca"))
    lda = make_pipeline(StandardScaler(), LinearDiscriminantAnalysis(n_components=2)).fit(features, labels)
    panels.append(("Supervised LDA (4 actions)", lda.transform(features), "lda"))
    ridge = make_pipeline(StandardScaler(), Ridge(alpha=1.0)).fit(features, clr)
    panels.append(("Regression onto CLR simplex", ridge.predict(features)[:, :2], "clr"))

    fig, axes = plt.subplots(1, 3, figsize=(15, 5.2), facecolor=SURFACE)
    for ax, (title, coords, mode) in zip(axes, panels, strict=True):
        ax.set_facecolor(SURFACE)
        for action in ACTIONS:
            mask = labels == action
            if not mask.any():
                continue
            ax.scatter(
                coords[mask, 0], coords[mask, 1], s=5, alpha=0.45, linewidths=0,
                color=ACTION_COLORS[action], label=action,
            )
        # Relief rule: direct centroid labels, since one hue is below 3:1 contrast.
        for action in ACTIONS:
            mask = labels == action
            if mask.sum() < 20:
                continue
            cx, cy = coords[mask, 0].mean(), coords[mask, 1].mean()
            ax.annotate(
                action, (cx, cy), color=INK, fontsize=10, fontweight="bold", ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.25", facecolor=SURFACE, edgecolor=ACTION_COLORS[action], linewidth=1.4, alpha=0.92),
            )
        score = metrics[(metrics.representation == "all_layer_mean") & (metrics.projection == mode)]
        subtitle = f"grouped balanced acc {float(score.balanced_accuracy.iloc[0]):.3f}" if len(score) else ""
        ax.set_title(f"{title}\n{subtitle}", color=INK, fontsize=11)
        ax.set_xlabel("component 1", color=INK_MUTED, fontsize=9)
        ax.set_ylabel("component 2", color=INK_MUTED, fontsize=9)
        ax.tick_params(colors=INK_MUTED, labelsize=8)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        for spine in ("left", "bottom"):
            ax.spines[spine].set_color("#d8d7d2")
        ax.grid(True, color="#ececE8", linewidth=0.6)
        ax.set_axisbelow(True)
    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, legend_labels, loc="lower center", ncol=4, frameon=False,
        fontsize=10, labelcolor=INK, markerscale=3.0, bbox_to_anchor=(0.5, -0.02),
    )
    fig.suptitle(
        "Pre-verbal action structure in activations: unsupervised vs supervised projection\n"
        f"held-out fold {args.split_id}, strict before-any-action-mention prefixes, chance = 0.25",
        color=INK, fontsize=12,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.94))
    path = args.figure_dir / "supervised_activation_projection.png"
    fig.savefig(path, dpi=170, facecolor=SURFACE, bbox_inches="tight")
    print(f"wrote {path}")


if __name__ == "__main__":
    raise SystemExit(main())
