#!/usr/bin/env python3
"""Project activations onto the three LDA axes and look for the four-arm shape in 3-d.

Earlier work only ever displayed 2-d views of the LDA projection, so the natural question
-- does the full 3-d projection show four arms? -- was never actually answered.

The behavioural four-arm shape has a specific geometry: RADIUS encodes commitment (how far
the belief sits from uniform) and ANGLE encodes which action. An arm is therefore a
commitment x content structure, and you only see it if both coordinates are present.

That yields a prediction. The commitment direction was measured to be nearly orthogonal to
the action subspace, so projecting onto the three LDA axes should discard most of the
radial coordinate and leave four displaced blobs rather than four arms.

To avoid eyeballing, the same four "armness" statistics are computed on:
  (a) the behavioural CLR belief space, mapped to an orthonormal 3-d basis of the
      sum-to-zero hyperplane -- the space where the arms are known to exist; and
  (b) the 3-d LDA projection of activations,
on the SAME held-out prefixes, so (a) calibrates what an arm actually scores.

Projections are fitted on training trajectories and evaluated on held-out ones, so nothing
here is LDA overfitting a separation into existence.
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

# Orthonormal basis of the sum-to-zero hyperplane in R^4 (Helmert contrasts). CLR vectors
# sum to zero by construction, so this maps them to 3 coordinates without distortion.
HELMERT = np.array([
    [1, -1, 0, 0],
    [1, 1, -2, 0],
    [1, 1, 1, -3],
], dtype=np.float64)
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
    p.add_argument("--out_dir", type=Path, default=REPO_ROOT / "outputs/conceptual_belief_space/lda_3d_arms")
    p.add_argument("--figure_dir", type=Path, default=REPO_ROOT / "figures/conceptual_belief_space")
    p.add_argument("--representation", type=str, default="mean")
    p.add_argument("--subset", choices=["strict", "all"], default="strict")
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
    """Four diagnostics of a four-arm (radial) structure, all scale-free."""
    from scipy.stats import spearmanr

    z = z - z.mean(axis=0)
    radius = np.linalg.norm(z, axis=1)
    centroids = {a: z[labels == a].mean(axis=0) for a in ACTIONS if (labels == a).any()}
    keys = list(centroids)
    matrix = np.stack([centroids[a] for a in keys])
    unit_centroids = matrix / np.maximum(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-12)

    unit_points = z / np.maximum(radius[:, None], 1e-12)
    cosines = unit_points @ unit_centroids.T                    # [n, n_classes]
    own = np.array([keys.index(a) for a in labels])
    own_cos = cosines[np.arange(len(z)), own]
    nearest = cosines.argmax(axis=1)

    return {
        # Arms: points lie along their own class ray, so this is high. Blobs: ~0.
        "own_centroid_cosine": float(np.mean(own_cos)),
        # Arms: the direction you point in identifies your action.
        "angular_purity": float(np.mean(nearest == own)),
        # Arms: distance from the origin IS commitment. This is the defining property.
        "radius_vs_commitment_rho": float(spearmanr(radius, commitment).statistic),
        # Arms: class centroids sit far out. Blobs: they huddle near the origin.
        "centroid_radius_ratio": float(np.mean(np.linalg.norm(matrix, axis=1)) / np.mean(radius)),
    }


def main() -> int:
    args = parse_args()
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
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
    print(f"[arms] subset={args.subset}: {len(train_idx)} train / {len(test_idx)} held-out prefixes")

    def features(idx: np.ndarray) -> np.ndarray:
        block = np.asarray(pooled[idx])
        return (block.mean(axis=1, dtype=np.float32) if args.representation == "mean"
                else np.asarray(pooled[idx, int(args.representation)], dtype=np.float32))

    q_test = table.iloc[test_idx][Q_COLS].to_numpy(dtype=np.float64)
    labels_test = np.asarray(ACTIONS)[q_test.argmax(axis=1)]
    commitment_test = table.iloc[test_idx]["entropy_commitment"].to_numpy(dtype=np.float64)
    q_train = table.iloc[train_idx][Q_COLS].to_numpy(dtype=np.float64)
    labels_train = np.asarray(ACTIONS)[q_train.argmax(axis=1)]

    # (a) Behavioural belief space: CLR -> orthonormal 3-d coordinates.
    log_q = np.log(np.clip(q_test, 1e-12, None))
    clr = log_q - log_q.mean(axis=1, keepdims=True)
    z_behaviour = clr @ HELMERT.T

    # (b) Activation space: the three LDA axes, fitted on TRAIN trajectories only.
    scaler = StandardScaler().fit(features(train_idx))
    lda = LinearDiscriminantAnalysis(n_components=3).fit(scaler.transform(features(train_idx)),
                                                         labels_train)
    z_activation = lda.transform(scaler.transform(features(test_idx)))

    # Euclidean (orthonormalised) variant, for robustness: LDA.transform whitens, which is
    # the more favourable rendering; V'x preserves ordinary geometry.
    basis, _ = np.linalg.qr(lda.scalings_[:, :3])
    z_activation_euclid = scaler.transform(features(test_idx)) @ basis

    # Collapse each 3-d cloud to its best-variance 2-d view. This is a rotation plus a
    # projection of the SAME cloud, so it cannot manufacture radial structure that is not
    # already there -- it just picks a canonical viewing plane instead of LD1 vs LD2.
    from sklearn.decomposition import PCA

    log_q_train = np.log(np.clip(q_train, 1e-12, None))
    z_behaviour_train = (log_q_train - log_q_train.mean(axis=1, keepdims=True)) @ HELMERT.T
    pca_behaviour = PCA(n_components=2, random_state=args.seed).fit(z_behaviour_train)
    pca_activation = PCA(n_components=2, random_state=args.seed).fit(
        lda.transform(scaler.transform(features(train_idx))))
    z_behaviour_2d = pca_behaviour.transform(z_behaviour)
    z_activation_2d = pca_activation.transform(z_activation)
    print(f"[arms] PCA(3->2) explained variance: behaviour "
          f"{pca_behaviour.explained_variance_ratio_.sum():.3f}, activation "
          f"{pca_activation.explained_variance_ratio_.sum():.3f}")

    rows = [
        {"space": "behavioural belief (CLR), 3-d", **arm_statistics(z_behaviour, labels_test, commitment_test)},
        {"space": "behavioural belief, PCA to 2-d", **arm_statistics(z_behaviour_2d, labels_test, commitment_test)},
        {"space": "activation LDA (whitened), 3-d", **arm_statistics(z_activation, labels_test, commitment_test)},
        {"space": "activation LDA, PCA to 2-d", **arm_statistics(z_activation_2d, labels_test, commitment_test)},
        {"space": "activation LDA (euclidean), 3-d", **arm_statistics(z_activation_euclid, labels_test, commitment_test)},
    ]
    # An in-sample activation row shows whether overfitting would have manufactured arms.
    z_in_sample = lda.transform(scaler.transform(features(train_idx)))
    rows.append({"space": "activation LDA (in-sample)",
                 **arm_statistics(z_in_sample, labels_train,
                                  table.iloc[train_idx]["entropy_commitment"].to_numpy(dtype=np.float64))})
    frame = pd.DataFrame(rows)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out_dir / "arm_statistics.csv", index=False)
    print()
    print(frame.round(4).to_string(index=False))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure = plt.figure(figsize=(15.6, 8.8))
    panels = [("Behavioural belief space (CLR)", z_behaviour, z_behaviour_2d, 0),
              ("Activation space (three LDA axes)", z_activation, z_activation_2d, 1)]
    for title, z, z2d, row in panels:
        column = row + 1
        z = z - z.mean(axis=0)
        # Frame on the bulk of the cloud; a few arm tips reach far beyond it.
        limit = float(np.percentile(np.abs(z), 99.0))
        axis = figure.add_subplot(2, 3, row * 3 + 1, projection="3d")
        axis.set_xlim(-limit, limit); axis.set_ylim(-limit, limit); axis.set_zlim(-limit, limit)
        axis.set_box_aspect((1, 1, 1))
        for action in ACTIONS:
            m = labels_test == action
            axis.scatter(z[m, 0], z[m, 1], z[m, 2], s=3.5, alpha=0.20,
                         color=ACTION_COLORS[action], linewidths=0, depthshade=False)
            c = z[m].mean(axis=0)
            axis.scatter([c[0]], [c[1]], [c[2]], s=110, color=ACTION_COLORS[action],
                         edgecolors="white", linewidths=1.8, depthshade=False)
            # Extend the centroid ray to the frame edge so the four directions are visible.
            scale = limit / max(np.linalg.norm(c), 1e-9)
            axis.plot([0, c[0] * scale], [0, c[1] * scale], [0, c[2] * scale],
                      color=ACTION_COLORS[action], linewidth=1.6, alpha=0.55)
            axis.text(c[0] * scale, c[1] * scale, c[2] * scale, f" {action}",
                      color=ACTION_COLORS[action], fontsize=10, fontweight="bold")
        axis.set_title(title, color=INK, fontsize=11.5, pad=0)
        axis.set_xticks([]); axis.set_yticks([]); axis.set_zticks([])
        axis.view_init(elev=22, azim=45)
        axis.grid(False)
        for pane in (axis.xaxis, axis.yaxis, axis.zaxis):
            pane.pane.set_edgecolor(GRID); pane.pane.set_alpha(0.06)

        # Middle column: the same cloud collapsed to its best-variance plane.
        axis_2d = figure.add_subplot(2, 3, row * 3 + 2)
        flat = z2d - z2d.mean(axis=0)
        span = float(np.percentile(np.abs(flat), 99.0))
        for action in ACTIONS:
            m = labels_test == action
            axis_2d.scatter(flat[m, 0], flat[m, 1], s=3.5, alpha=0.20,
                            color=ACTION_COLORS[action], linewidths=0, rasterized=True)
        for action in ACTIONS:
            m = labels_test == action
            c = flat[m].mean(axis=0)
            reach = span / max(np.linalg.norm(c), 1e-9)
            axis_2d.plot([0, c[0] * reach], [0, c[1] * reach], color=ACTION_COLORS[action],
                         linewidth=1.5, alpha=0.6, zorder=4)
            axis_2d.scatter([c[0]], [c[1]], s=110, color=ACTION_COLORS[action],
                            edgecolors="white", linewidths=1.8, zorder=6)
            axis_2d.annotate(action, (c[0] * reach, c[1] * reach), textcoords="offset points",
                             xytext=(4, 3), color=ACTION_COLORS[action], fontsize=9.5,
                             fontweight="bold")
        axis_2d.set_xlim(-span, span); axis_2d.set_ylim(-span, span)
        axis_2d.set_aspect("equal")
        axis_2d.set_title("collapsed to 2-d by PCA", color=INK, fontsize=11)
        axis_2d.set_xticks([]); axis_2d.set_yticks([])
        for sp in axis_2d.spines.values():
            sp.set_color(GRID)

        axis2 = figure.add_subplot(2, 3, row * 3 + 3)
        radius = np.linalg.norm(z, axis=1)
        for action in ACTIONS:
            m = labels_test == action
            axis2.scatter(commitment_test[m], radius[m], s=3, alpha=0.16,
                          color=ACTION_COLORS[action], linewidths=0)
        from scipy.stats import spearmanr
        rho = spearmanr(radius, commitment_test).statistic
        axis2.set_xlabel("commitment", color=INK, fontsize=10)
        axis2.set_ylabel("distance from origin", color=INK, fontsize=10)
        axis2.set_title(f"radius vs commitment   (Spearman {rho:+.3f})", color=INK, fontsize=11)
        axis2.tick_params(colors=INK, labelsize=9)
        axis2.grid(True, color=GRID, linewidth=0.7)
        axis2.set_axisbelow(True)
        for sp in ("top", "right"):
            axis2.spines[sp].set_visible(False)

    figure.suptitle("Do the three LDA axes reproduce the behavioural four-arm shape?",
                    color=INK, fontsize=13)
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    fig_path = args.figure_dir / "lda_3d_arm_test.png"
    figure.savefig(fig_path, dpi=190)
    plt.close(figure)

    # Standalone, large version of just the two 2-d views, for looking at properly.
    from scipy.stats import spearmanr as _spearman

    figure2, axes2 = plt.subplots(1, 2, figsize=(14.0, 7.2))
    which = ("mean over all 24 layers" if args.representation == "mean"
             else f"layer {int(args.representation)}")
    for axis, (title, z2d) in zip(axes2, [("Behavioural belief space (forced-answer q)", z_behaviour_2d),
                                          (f"Activation space, three LDA axes ({which})", z_activation_2d)]):
        flat = z2d - z2d.mean(axis=0)
        span = float(np.percentile(np.abs(flat), 99.2))
        for action in ACTIONS:
            m = labels_test == action
            axis.scatter(flat[m, 0], flat[m, 1], s=7, alpha=0.30, color=ACTION_COLORS[action],
                         linewidths=0, rasterized=True, label=action)
        for action in ACTIONS:
            m = labels_test == action
            c = flat[m].mean(axis=0)
            reach = span / max(np.linalg.norm(c), 1e-9)
            axis.plot([0, c[0] * reach], [0, c[1] * reach], color=ACTION_COLORS[action],
                      linewidth=2.0, alpha=0.75, zorder=4)
            axis.scatter([c[0]], [c[1]], s=190, color=ACTION_COLORS[action],
                         edgecolors="white", linewidths=2.2, zorder=6)
            axis.annotate(action, (c[0] * reach * 0.93, c[1] * reach * 0.93),
                          color=ACTION_COLORS[action], fontsize=13, fontweight="bold",
                          ha="center", va="center", zorder=7,
                          bbox=dict(boxstyle="round,pad=0.22", facecolor="white",
                                    edgecolor="none", alpha=0.78))
        radius = np.linalg.norm(flat, axis=1)
        rho = float(_spearman(radius, commitment_test).statistic)
        axis.set_xlim(-span, span); axis.set_ylim(-span, span)
        axis.set_aspect("equal")
        axis.set_title(f"{title}\ndistance from centre vs commitment:  Spearman {rho:+.3f}",
                       color=INK, fontsize=12.5)
        axis.set_xticks([]); axis.set_yticks([])
        for sp in axis.spines.values():
            sp.set_color(GRID)
    figure2.suptitle(f"Both spaces collapsed to two dimensions by PCA, same held-out prefixes "
                     f"({which} activations)", color=INK, fontsize=13.5)
    figure2.tight_layout(rect=(0, 0, 1, 0.95))
    fig2_path = args.figure_dir / "lda_2d_views.png"
    figure2.savefig(fig2_path, dpi=200)
    plt.close(figure2)

    behaviour = frame.iloc[0]
    activation = frame.iloc[1]
    lines = [
        "# Are there four arms in the 3-d LDA projection of activations?",
        "",
        f"- Subset `{args.subset}`, representation `{args.representation}`; projections fitted on "
        f"{len(train_idx)} training-trajectory prefixes and evaluated on {len(test_idx)} held-out ones.",
        "- The behavioural CLR row is the calibration: it is the space where the arms are known",
        "  to exist, measured on the same prefixes with the same statistics.",
        "",
        frame.round(4).to_markdown(index=False),
        "",
        "## Reading",
        "",
        f"- Radius encodes commitment in the behavioural space (Spearman "
        f"{behaviour['radius_vs_commitment_rho']:+.3f}) but not in the LDA projection "
        f"({activation['radius_vs_commitment_rho']:+.3f}). That is the arm-defining property, and",
        "  it is the one that fails.",
        f"- Points lie along their own class ray in the behavioural space "
        f"(cosine {behaviour['own_centroid_cosine']:+.3f}) far more than in the LDA projection "
        f"({activation['own_centroid_cosine']:+.3f}).",
        f"- Class centroids sit at {behaviour['centroid_radius_ratio']:.3f} of the cloud radius "
        f"behaviourally versus {activation['centroid_radius_ratio']:.3f} in activations: arms reach",
        "  outward, blobs huddle at the origin.",
        "",
        f"Figure: `{fig_path}`",
    ]
    (args.out_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nWrote {args.out_dir}/report.md and {fig_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
