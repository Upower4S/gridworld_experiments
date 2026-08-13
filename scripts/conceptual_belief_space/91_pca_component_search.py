#!/usr/bin/env python3
"""Does ANY pair of principal components show the behavioural four-arm structure?

The earlier negative result ("no four-arm manifold in activation space") rested on the
leading components only: PCA(3), visualised as PC1 vs PC2, whose top components were shown
to track token position rather than action. That does not rule out a later pair. Here we
search exhaustively.

Design, with the selection problem handled explicitly:
  1. PCA to `--n_components` on pre-verbal activations (unsupervised, so the projection
     itself cannot leak labels).
  2. Cheap screen over all component PAIRS: a multiclass trace-ratio separability statistic
     (between-class scatter / within-class scatter) in each 2-d plane.
  3. A permutation null for the SCREEN MAXIMUM -- shuffle action labels many times and
     record the best pair each time. Searching hundreds of pairs guarantees some winner;
     the question is whether the real winner beats the winner you get from noise.
  4. Trajectory-grouped cross-validated decoding and silhouette on the top pairs only,
     against the PC1-PC2 baseline and a supervised LDA reference.
"""

from __future__ import annotations

import argparse
import importlib.util
import itertools
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
INK, GRID, MUTED = "#222222", "#e0e0e0", "#5f6368"


def parse_args() -> argparse.Namespace:
    base = REPO_ROOT / "outputs/conceptual_belief_space/cot_window_layer_transition_probe"
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--activation_index", type=Path, default=base / "activations/activation_index.parquet")
    p.add_argument("--beliefs", type=Path, default=base / "activations/extracted_beliefs.parquet")
    p.add_argument("--pooled_cache", type=Path, default=REPO_ROOT
                   / "outputs/conceptual_belief_space/layerwise_dissociation/pooled_cot_window_f16.npy")
    p.add_argument("--out_dir", type=Path, default=REPO_ROOT / "outputs/conceptual_belief_space/pca_component_search")
    p.add_argument("--figure_dir", type=Path, default=REPO_ROOT / "figures/conceptual_belief_space")
    p.add_argument("--representation", type=str, default="mean", help="'mean' or a layer index.")
    p.add_argument("--n_components", type=int, default=30)
    p.add_argument("--permutations", type=int, default=200)
    p.add_argument("--top_k", type=int, default=6)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def load_probe77() -> Any:
    path = REPO_ROOT / "scripts/conceptual_belief_space/77_train_forced_suffix_commitment_probe.py"
    spec = importlib.util.spec_from_file_location("probe77", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def trace_ratio(z: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Between/within class scatter for every 2-d plane, vectorised over pairs.

    Returns a [k,k] matrix; entry (i,j) is the separability of the plane spanned by
    components i and j. Because scatter is additive across orthogonal PCA axes, the
    2-d statistic is the sum of the two 1-d statistics, so we compute k of them, not k^2.
    """
    grand = z.mean(axis=0)
    between = np.zeros(z.shape[1])
    within = np.zeros(z.shape[1])
    for cls in np.unique(labels):
        m = labels == cls
        n = int(m.sum())
        if n < 2:
            continue
        centroid = z[m].mean(axis=0)
        between += n * (centroid - grand) ** 2
        within += ((z[m] - centroid) ** 2).sum(axis=0)
    per_axis = between / np.maximum(within, 1e-12)
    return per_axis


def arm_geometry(z2: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    """Four-arm diagnostics: are class centroids angularly spread, and do points
    radiate along their class direction rather than forming one blob?"""
    centred = z2 - z2.mean(axis=0)
    radius = np.linalg.norm(centred, axis=1)
    angles = {}
    for cls in np.unique(labels):
        m = labels == cls
        c = centred[m].mean(axis=0)
        angles[cls] = np.arctan2(c[1], c[0])
    ordered = np.sort(np.array(list(angles.values())))
    gaps = np.diff(np.concatenate([ordered, [ordered[0] + 2 * np.pi]]))
    # 1.0 = perfectly even 90-degree spacing of four centroids.
    evenness = float(1.0 - np.abs(gaps - np.pi / 2).sum() / (2 * np.pi))
    centroid_norm = float(np.mean([np.linalg.norm(centred[labels == c].mean(axis=0))
                                   for c in np.unique(labels)]))
    return {"centroid_angular_evenness": evenness,
            "centroid_radius_over_spread": centroid_norm / float(radius.mean() + 1e-12)}


def grouped_scores(z2: np.ndarray, labels: np.ndarray, groups: np.ndarray, seed: int) -> dict[str, float]:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import balanced_accuracy_score, silhouette_score
    from sklearn.model_selection import GroupKFold

    accs = []
    for tr, te in GroupKFold(n_splits=5).split(z2, labels, groups):
        model = LogisticRegression(max_iter=2000, class_weight="balanced",
                                   random_state=seed).fit(z2[tr], labels[tr])
        accs.append(balanced_accuracy_score(labels[te], model.predict(z2[te])))
    rng = np.random.default_rng(seed)
    sub = rng.choice(len(z2), size=min(4000, len(z2)), replace=False)
    return {"balanced_accuracy": float(np.mean(accs)),
            "silhouette": float(silhouette_score(z2[sub], labels[sub]))}


def main() -> int:
    args = parse_args()
    from sklearn.decomposition import PCA
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.preprocessing import StandardScaler

    probe77 = load_probe77()
    table = probe77.prepare_table(args.activation_index, args.beliefs, None)
    cached = np.load(args.pooled_cache.with_suffix(".prefix_ids.npy"))
    if not (cached == table["prefix_id"].to_numpy()).all():
        raise SystemExit("pooled cache does not match this table ordering")
    pooled = np.load(args.pooled_cache, mmap_mode="r")

    rows = np.nonzero(table[STRICT_COL].to_numpy(dtype=bool))[0]
    meta = table.iloc[rows]
    features = (np.asarray(pooled[rows]).mean(axis=1, dtype=np.float32)
                if args.representation == "mean"
                else np.asarray(pooled[rows, int(args.representation)], dtype=np.float32))
    q = meta[Q_COLS].to_numpy(dtype=np.float64)
    labels = np.asarray(ACTIONS)[q.argmax(axis=1)]
    groups = meta["trajectory_id"].to_numpy()
    position = np.log1p(meta["prefix_token_index"].to_numpy(dtype=np.float64))
    print(f"[pca] {len(rows)} pre-verbal rows, {len(set(groups))} trajectories, "
          f"representation={args.representation}")

    z = PCA(n_components=args.n_components, random_state=args.seed).fit_transform(
        StandardScaler().fit_transform(features))

    per_axis = trace_ratio(z, labels)
    pairs = list(itertools.combinations(range(args.n_components), 2))
    pair_stat = np.array([per_axis[i] + per_axis[j] for i, j in pairs])
    order = np.argsort(pair_stat)[::-1]

    # Permutation null for the SEARCH MAXIMUM, respecting trajectory blocks.
    rng = np.random.default_rng(args.seed)
    unique_groups = np.unique(groups)
    null_max = []
    for _ in range(args.permutations):
        permuted = labels.copy()
        for g in unique_groups:            # shuffle within trajectory: keeps class balance
            m = groups == g
            permuted[m] = rng.permutation(permuted[m])
        axis_null = trace_ratio(z, permuted)
        null_max.append(float(max(axis_null[i] + axis_null[j] for i, j in pairs)))
    null_max = np.array(null_max)
    observed_max = float(pair_stat[order[0]])
    p_value = float((null_max >= observed_max).mean())

    # Per-component diagnostics: action separability vs position tracking.
    comp_rows = []
    for k in range(args.n_components):
        comp_rows.append({"component": k, "action_trace_ratio": float(per_axis[k]),
                          "abs_corr_with_log_position": float(abs(np.corrcoef(z[:, k], position)[0, 1]))})
    components = pd.DataFrame(comp_rows)

    # Confirm the top pairs properly, plus references.
    results = []
    candidates = [pairs[i] for i in order[: args.top_k]] + [(0, 1)]
    for (i, j) in dict.fromkeys(candidates):
        z2 = z[:, [i, j]]
        entry = {"pair": f"PC{i+1}-PC{j+1}", "screen_stat": float(per_axis[i] + per_axis[j])}
        entry.update(grouped_scores(z2, labels, groups, args.seed))
        entry.update(arm_geometry(z2, labels))
        results.append(entry)
        print(f"[pca] {entry['pair']:>10}: bal_acc={entry['balanced_accuracy']:.4f} "
              f"silhouette={entry['silhouette']:+.4f}", flush=True)
    frame = pd.DataFrame(results).sort_values("balanced_accuracy", ascending=False)

    lda2 = LinearDiscriminantAnalysis(n_components=2).fit_transform(
        StandardScaler().fit_transform(features), labels)
    lda_scores = grouped_scores(lda2, labels, groups, args.seed)
    lda_scores.update(arm_geometry(lda2, labels))
    print(f"[pca] supervised LDA(2) reference: bal_acc={lda_scores['balanced_accuracy']:.4f} "
          f"silhouette={lda_scores['silhouette']:+.4f}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out_dir / "top_pairs.csv", index=False)
    components.to_csv(args.out_dir / "components.csv", index=False)
    pd.DataFrame({"null_max_trace_ratio": null_max}).to_csv(args.out_dir / "permutation_null.csv", index=False)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    best = frame.iloc[0]["pair"]
    bi, bj = (int(t[2:]) - 1 for t in best.split("-"))
    figure, axes = plt.subplots(1, 3, figsize=(13.5, 4.3))
    for axis, (zz, title) in zip(axes, [(z[:, [0, 1]], "PC1-PC2 (what was shown before)"),
                                        (z[:, [bi, bj]], f"{best} (best of {len(pairs)} pairs)"),
                                        (lda2, "supervised LDA (reference)")]):
        for action in ACTIONS:
            m = labels == action
            axis.scatter(zz[m, 0], zz[m, 1], s=3, alpha=0.25, color=ACTION_COLORS[action],
                         linewidths=0, rasterized=True)
            c = zz[m].mean(axis=0)
            axis.scatter([c[0]], [c[1]], s=90, color=ACTION_COLORS[action],
                         edgecolors="white", linewidths=1.6, zorder=5)
            axis.annotate(action, (c[0], c[1]), xytext=(6, 4), textcoords="offset points",
                          color=INK, fontsize=9, fontweight="bold")
        axis.set_title(title, color=INK, fontsize=11)
        axis.set_xticks([]); axis.set_yticks([])
        for sp in axis.spines.values():
            sp.set_color(GRID)
    figure.suptitle("Is there any principal-component plane with a four-arm structure?", color=INK)
    figure.tight_layout()
    fig_path = args.figure_dir / "pca_component_search.png"
    figure.savefig(fig_path, dpi=190)
    plt.close(figure)

    lines = [
        "# Exhaustive PCA component-pair search for four-arm structure",
        "",
        f"- {len(rows)} pre-verbal prefixes, {len(set(groups))} trajectories, "
        f"representation `{args.representation}`, {args.n_components} components, "
        f"{len(pairs)} pairs searched.",
        "",
        "## Does the best pair beat what label-shuffling produces?",
        "",
        f"- best observed pair statistic: {observed_max:.4f} (PC{pairs[order[0]][0]+1}-PC{pairs[order[0]][1]+1})",
        f"- permutation null for the search maximum ({args.permutations} shuffles): "
        f"mean {null_max.mean():.4f}, 95th pct {np.quantile(null_max, 0.95):.4f}, max {null_max.max():.4f}",
        f"- **p = {p_value:.3f}**",
        "",
        "## Top pairs, confirmed with trajectory-grouped CV",
        "",
        frame.round(4).to_markdown(index=False),
        "",
        f"Supervised LDA(2) reference: balanced accuracy {lda_scores['balanced_accuracy']:.4f}, "
        f"silhouette {lda_scores['silhouette']:+.4f}, "
        f"angular evenness {lda_scores['centroid_angular_evenness']:.4f}.",
        "",
        "## Per-component: action separability vs position tracking",
        "",
        components.sort_values("action_trace_ratio", ascending=False).head(10).round(4).to_markdown(index=False),
        "",
        f"Figure: `{fig_path}`",
    ]
    (args.out_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nbest={best} p={p_value:.3f}  (null mean {null_max.mean():.4f}, obs {observed_max:.4f})")
    print(f"Wrote {args.out_dir}/report.md and {fig_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
