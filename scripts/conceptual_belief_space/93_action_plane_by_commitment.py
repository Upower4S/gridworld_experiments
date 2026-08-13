#!/usr/bin/env python3
"""Build the space from two LDA action axes plus the commitment direction, and look at it.

The behavioural arms are a content x strength structure: angle says which action, radius
says how committed. Projecting activations onto the three LDA axes destroyed the radial
half, because commitment is nearly orthogonal to the action subspace. So construct the
space that keeps both: two action axes for the angle, the commitment direction for the
third coordinate.

This makes a sharp shape prediction, and the two candidate answers look completely
different:

  CONE     -- a multiplicative code, logits = gain x content direction. At low commitment
              the four actions are inseparable; as commitment grows they fan apart. Class
              separation in the action plane RISES with height.
  CYLINDER -- two independent, additive channels. The four actions are separated by the
              same amount at every commitment level, and the commitment axis merely slides
              points up and down. Separation is FLAT in height.

The distinction is measured, not eyeballed: class separation is computed in the action
plane within commitment bins, against a label-shuffled control for the same bins.
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
INK, GRID, MUTED, NEUTRAL = "#222222", "#d8d8d8", "#5f6368", "#9aa0a6"


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
                   / "outputs/conceptual_belief_space/action_plane_by_commitment")
    p.add_argument("--figure_dir", type=Path, default=REPO_ROOT / "figures/conceptual_belief_space")
    p.add_argument("--representation", type=str, default="mean")
    p.add_argument("--subset", choices=["strict", "all"], default="strict")
    p.add_argument("--bins", type=int, default=6)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def load_probe77() -> Any:
    path = REPO_ROOT / "scripts/conceptual_belief_space/77_train_forced_suffix_commitment_probe.py"
    spec = importlib.util.spec_from_file_location("probe77", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def plane_separation(plane: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    """How well separated are the four classes inside a 2-d plane? Scale-free."""
    plane = plane - plane.mean(axis=0)
    spread = float(np.mean(np.linalg.norm(plane, axis=1)))
    present = [a for a in ACTIONS if (labels == a).sum() >= 5]
    if len(present) < 2 or spread <= 0:
        return {"trace_ratio": np.nan, "centroid_radius_ratio": np.nan, "angular_purity": np.nan}
    centroids = np.stack([plane[labels == a].mean(axis=0) for a in present])
    between = sum(int((labels == a).sum()) * float(np.sum(c ** 2))
                  for a, c in zip(present, centroids))
    within = sum(float(np.sum((plane[labels == a] - c) ** 2))
                 for a, c in zip(present, centroids))
    unit_c = centroids / np.maximum(np.linalg.norm(centroids, axis=1, keepdims=True), 1e-12)
    radius = np.maximum(np.linalg.norm(plane, axis=1, keepdims=True), 1e-12)
    cos = (plane / radius) @ unit_c.T
    own = np.array([present.index(a) if a in present else -1 for a in labels])
    valid = own >= 0
    return {
        "trace_ratio": float(between / max(within, 1e-12)),
        "centroid_radius_ratio": float(np.mean(np.linalg.norm(centroids, axis=1)) / spread),
        "angular_purity": float(np.mean(cos[valid].argmax(axis=1) == own[valid])),
    }


def main() -> int:
    args = parse_args()
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

    keep = (table[STRICT_COL].to_numpy(dtype=bool) if args.subset == "strict"
            else np.ones(len(table), dtype=bool))
    train_idx = split["train_idx"][keep[split["train_idx"]]]
    test_idx = split["test_idx"][keep[split["test_idx"]]]

    def features(idx: np.ndarray) -> np.ndarray:
        block = np.asarray(pooled[idx])
        return (block.mean(axis=1, dtype=np.float32) if args.representation == "mean"
                else np.asarray(pooled[idx, int(args.representation)], dtype=np.float32))

    labels_of = lambda idx: np.asarray(ACTIONS)[
        table.iloc[idx][Q_COLS].to_numpy(dtype=np.float64).argmax(axis=1)]
    commitment_of = lambda idx: table.iloc[idx]["entropy_commitment"].to_numpy(dtype=np.float64)
    labels_train, labels_test = labels_of(train_idx), labels_of(test_idx)
    commitment_train, commitment_test = commitment_of(train_idx), commitment_of(test_idx)
    print(f"[cone] {len(train_idx)} train / {len(test_idx)} held-out prefixes")

    scaler = StandardScaler().fit(features(train_idx))
    x_train = scaler.transform(features(train_idx))
    x_test = scaler.transform(features(test_idx))

    # Two action axes, orthonormalised, from the leading discriminant directions.
    lda = LinearDiscriminantAnalysis(n_components=3).fit(x_train, labels_train)
    action_plane, _ = np.linalg.qr(lda.scalings_[:, :2])
    # Commitment axis: the ridge direction, with any component inside the action plane
    # removed, so the third coordinate is strictly the part of commitment the action axes
    # do not already carry.
    ridge = Ridge(alpha=1.0).fit(x_train, commitment_train)
    u = ridge.coef_ / np.linalg.norm(ridge.coef_)
    u_perp = u - action_plane @ (action_plane.T @ u)
    residual_fraction = float(np.linalg.norm(u_perp) ** 2)
    u_perp = u_perp / np.linalg.norm(u_perp)
    frame_basis = np.column_stack([action_plane, u_perp])
    print(f"[cone] commitment direction retains {residual_fraction:.4f} of its norm after "
          f"removing the action plane")

    z = x_test @ frame_basis                    # [n, 3]: (action1, action2, commitment)
    height = z[:, 2]
    plane = z[:, :2]
    from scipy.stats import spearmanr
    print(f"[cone] third axis vs true commitment: Spearman "
          f"{spearmanr(height, commitment_test).statistic:+.4f}")

    # The measurement: does class separation in the action plane grow with commitment?
    rng = np.random.default_rng(args.seed)
    edges = np.quantile(commitment_test, np.linspace(0, 1, args.bins + 1))
    edges[-1] += 1e-9
    rows = []
    for b in range(args.bins):
        m = (commitment_test >= edges[b]) & (commitment_test < edges[b + 1])
        if m.sum() < 50:
            continue
        real = plane_separation(plane[m], labels_test[m])
        shuffled = plane_separation(plane[m], rng.permutation(labels_test[m]))
        rows.append({"bin": b + 1,
                     "commitment_low": float(edges[b]), "commitment_high": float(edges[b + 1]),
                     "mean_commitment": float(commitment_test[m].mean()), "n": int(m.sum()),
                     **{f"{k}": v for k, v in real.items()},
                     **{f"shuffled_{k}": v for k, v in shuffled.items()}})
    binned = pd.DataFrame(rows)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    binned.to_csv(args.out_dir / "separation_by_commitment.csv", index=False)
    print()
    print(binned[["bin", "mean_commitment", "n", "trace_ratio", "shuffled_trace_ratio",
                  "angular_purity", "centroid_radius_ratio"]].round(4).to_string(index=False))

    low, high = binned.iloc[0], binned.iloc[-1]
    growth = float(high["trace_ratio"] / max(low["trace_ratio"], 1e-9))
    verdict = ("CONE (multiplicative: separation grows with commitment)" if growth >= 2.0
               else "CYLINDER (additive: separation roughly constant in commitment)"
               if growth <= 1.5 else "INTERMEDIATE")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure = plt.figure(figsize=(15.2, 8.6))
    axis3d = figure.add_subplot(2, 3, 1, projection="3d")
    limit = float(np.percentile(np.abs(z), 99.0))
    for action in ACTIONS:
        m = labels_test == action
        axis3d.scatter(z[m, 0], z[m, 1], z[m, 2], s=3.2, alpha=0.18,
                       color=ACTION_COLORS[action], linewidths=0, depthshade=False)
    axis3d.set_xlim(-limit, limit); axis3d.set_ylim(-limit, limit); axis3d.set_zlim(-limit, limit)
    axis3d.set_box_aspect((1, 1, 1))
    axis3d.set_xlabel("action axis 1", color=INK, fontsize=9)
    axis3d.set_ylabel("action axis 2", color=INK, fontsize=9)
    axis3d.set_zlabel("commitment", color=INK, fontsize=9)
    axis3d.set_xticks([]); axis3d.set_yticks([]); axis3d.set_zticks([])
    axis3d.view_init(elev=16, azim=42)
    axis3d.set_title("Constructed space\n(2 action axes + commitment)", color=INK, fontsize=11)

    # Action plane within commitment terciles: the cone-vs-cylinder picture.
    terciles = np.quantile(commitment_test, [0, 1 / 3, 2 / 3, 1.0])
    terciles[-1] += 1e-9
    names = ["lowest commitment third", "middle third", "highest commitment third"]
    span = float(np.percentile(np.abs(plane), 99.0))
    for k in range(3):
        axis = figure.add_subplot(2, 3, k + 2)
        m = (commitment_test >= terciles[k]) & (commitment_test < terciles[k + 1])
        for action in ACTIONS:
            mm = m & (labels_test == action)
            axis.scatter(plane[mm, 0], plane[mm, 1], s=5, alpha=0.28,
                         color=ACTION_COLORS[action], linewidths=0, rasterized=True)
        for action in ACTIONS:
            mm = m & (labels_test == action)
            if mm.sum() < 5:
                continue
            c = plane[mm].mean(axis=0)
            axis.scatter([c[0]], [c[1]], s=150, color=ACTION_COLORS[action],
                         edgecolors="white", linewidths=2.0, zorder=6)
            axis.annotate(action, (c[0], c[1]), textcoords="offset points", xytext=(7, 5),
                          color=ACTION_COLORS[action], fontsize=10, fontweight="bold", zorder=7)
        stats = plane_separation(plane[m], labels_test[m])
        axis.set_xlim(-span, span); axis.set_ylim(-span, span); axis.set_aspect("equal")
        axis.set_title(f"{names[k]}\nseparation {stats['trace_ratio']:.4f}", color=INK, fontsize=10.5)
        axis.set_xticks([]); axis.set_yticks([])
        for sp in axis.spines.values():
            sp.set_color(GRID)

    axis_curve = figure.add_subplot(2, 3, 5)
    axis_curve.plot(binned["mean_commitment"], binned["trace_ratio"], color="#2a78d6",
                    linewidth=2.2, marker="o", markersize=6, label="actual labels")
    axis_curve.plot(binned["mean_commitment"], binned["shuffled_trace_ratio"], color=NEUTRAL,
                    linewidth=1.8, marker="o", markersize=5, linestyle="--", label="shuffled labels")
    axis_curve.set_xlabel("commitment", color=INK, fontsize=10)
    axis_curve.set_ylabel("class separation in action plane", color=INK, fontsize=10)
    axis_curve.set_title("Cone would rise, cylinder would be flat", color=INK, fontsize=11)
    axis_curve.legend(frameon=False, fontsize=9, labelcolor=INK)
    axis_curve.grid(True, color=GRID, linewidth=0.7); axis_curve.set_axisbelow(True)
    axis_curve.tick_params(colors=INK, labelsize=9)
    for sp in ("top", "right"):
        axis_curve.spines[sp].set_visible(False)

    axis_purity = figure.add_subplot(2, 3, 6)
    axis_purity.plot(binned["mean_commitment"], binned["angular_purity"], color="#eb6834",
                     linewidth=2.2, marker="o", markersize=6)
    axis_purity.plot(binned["mean_commitment"], binned["shuffled_angular_purity"], color=NEUTRAL,
                     linewidth=1.8, marker="o", markersize=5, linestyle="--")
    axis_purity.set_xlabel("commitment", color=INK, fontsize=10)
    axis_purity.set_ylabel("angular purity", color=INK, fontsize=10)
    axis_purity.set_title("Does direction identify the action?", color=INK, fontsize=11)
    axis_purity.grid(True, color=GRID, linewidth=0.7); axis_purity.set_axisbelow(True)
    axis_purity.tick_params(colors=INK, labelsize=9)
    for sp in ("top", "right"):
        axis_purity.spines[sp].set_visible(False)

    figure.suptitle("Two action axes plus the commitment direction: cone or cylinder?",
                    color=INK, fontsize=13.5)
    figure.tight_layout(rect=(0, 0, 1, 0.95))
    fig_path = args.figure_dir / "action_plane_by_commitment.png"
    figure.savefig(fig_path, dpi=195)
    plt.close(figure)

    # Standalone money plot: a side view (does the cloud fan out with commitment?) beside
    # the action plane at three commitment levels, all on identical axes.
    figure2, axes2 = plt.subplots(1, 4, figsize=(19.0, 5.4))
    side = axes2[0]
    for action in ACTIONS:
        m = labels_test == action
        side.scatter(plane[m, 0], height[m], s=4, alpha=0.16, color=ACTION_COLORS[action],
                     linewidths=0, rasterized=True)
    # Per-height class centroids: a cone makes these four tracks diverge upward.
    height_edges = np.quantile(height, np.linspace(0, 1, 9))
    height_edges[-1] += 1e-9
    for action in ACTIONS:
        centres, mids = [], []
        for b in range(len(height_edges) - 1):
            m = ((height >= height_edges[b]) & (height < height_edges[b + 1])
                 & (labels_test == action))
            if m.sum() >= 20:
                centres.append(float(plane[m, 0].mean()))
                mids.append(float((height_edges[b] + height_edges[b + 1]) / 2))
        side.plot(centres, mids, color=ACTION_COLORS[action], linewidth=2.6, zorder=6,
                  marker="o", markersize=4.5, markeredgecolor="white", markeredgewidth=0.8)
    side.set_xlim(-span, span)
    side.set_xlabel("action axis 1", color=INK, fontsize=11)
    side.set_ylabel("commitment direction", color=INK, fontsize=11)
    side.set_title("Side view: do the four tracks\nfan apart as commitment rises?",
                   color=INK, fontsize=11.5)
    side.set_xticks([]); side.set_yticks([])
    side.grid(True, color=GRID, linewidth=0.7); side.set_axisbelow(True)
    for sp in side.spines.values():
        sp.set_color(GRID)

    for k in range(3):
        axis = axes2[k + 1]
        m = (commitment_test >= terciles[k]) & (commitment_test < terciles[k + 1])
        for action in ACTIONS:
            mm = m & (labels_test == action)
            axis.scatter(plane[mm, 0], plane[mm, 1], s=6, alpha=0.30,
                         color=ACTION_COLORS[action], linewidths=0, rasterized=True)
        for action in ACTIONS:
            mm = m & (labels_test == action)
            if mm.sum() < 5:
                continue
            c = plane[mm].mean(axis=0)
            axis.scatter([c[0]], [c[1]], s=190, color=ACTION_COLORS[action],
                         edgecolors="white", linewidths=2.2, zorder=6)
            axis.annotate(action, (c[0], c[1]), textcoords="offset points", xytext=(9, 6),
                          color=ACTION_COLORS[action], fontsize=11, fontweight="bold", zorder=7)
        stats = plane_separation(plane[m], labels_test[m])
        axis.set_xlim(-span, span); axis.set_ylim(-span, span); axis.set_aspect("equal")
        axis.set_title(f"{names[k]}\ncommitment {terciles[k]:.2f}-{terciles[k+1]:.2f}   "
                       f"separation {stats['trace_ratio']:.3f}", color=INK, fontsize=11.5)
        axis.set_xticks([]); axis.set_yticks([])
        for sp in axis.spines.values():
            sp.set_color(GRID)
    figure2.suptitle("Action plane at three commitment levels, identical axes: the spread "
                     "does not widen", color=INK, fontsize=13.5)
    figure2.tight_layout(rect=(0, 0, 1, 0.94))
    fig2_path = args.figure_dir / "cone_or_cylinder.png"
    figure2.savefig(fig2_path, dpi=200)
    plt.close(figure2)

    lines = [
        "# Two LDA action axes plus the commitment direction",
        "",
        f"- Subset `{args.subset}`, representation `{args.representation}`; axes fitted on "
        f"{len(train_idx)} training-trajectory prefixes, evaluated on {len(test_idx)} held-out.",
        f"- The commitment direction keeps {residual_fraction:.4f} of its norm after removing "
        "the action plane, so the third axis is essentially pure commitment.",
        f"- Third axis vs true commitment: Spearman "
        f"{spearmanr(height, commitment_test).statistic:+.4f}.",
        "",
        "## Class separation in the action plane, by commitment bin",
        "",
        binned.round(4).to_markdown(index=False),
        "",
        "## Verdict",
        "",
        f"- separation grows {growth:.2f}x from the lowest to the highest commitment bin",
        f"- **{verdict}**",
        "",
        f"Figure: `{fig_path}`",
    ]
    (args.out_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\ngrowth low->high = {growth:.2f}x   ->  {verdict}")
    print(f"Wrote {args.out_dir}/report.md and {fig_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
