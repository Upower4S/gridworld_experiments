#!/usr/bin/env python3
"""Matched per-layer decoding curves for commitment vs action identity (D2a-c).

The fold-0 dissociation evidence was methodologically asymmetric: action identity was
measured with single-layer LDA decodability while commitment used transformer layer
occlusion. This stage measures BOTH targets with the same instrument at every layer:
for each outer fold and each physical layer L (plus the all-layer mean), a ridge
regression predicts `entropy_commitment` (Spearman/R2) and a balanced multinomial
logistic regression predicts the forced top action (balanced accuracy), from the same
mask-pooled CoT-window activations. Positional bars, an early-vs-late block contrast,
and within-layer geometry (principal angle of the commitment direction against the LDA
action subspace, plus projection-removal tests) complete the claim.

Anti-leakage contract: activation-only inputs, trajectory-grouped outer folds with zero
overlap, forced-suffix tokens excluded from inputs, <1M readout parameters per probe.
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

from src.conceptual_belief_space.forced_suffix_probe import (  # noqa: E402
    ACTIONS,
    ProjectedShardStore,
)

Q_COLS = [f"q_{action}" for action in ACTIONS]
STRICT_COL = "strict_before_any_action_mention"
EARLY_BLOCK = tuple(range(0, 8))
LATE_BLOCK = tuple(range(16, 24))
REMOVAL_LAYERS = ("3", "10", "18", "mean")

INK = "#333333"
GRID = "#e6e6e6"
COLOR_ACTION = "#2a78d6"
COLOR_COMMIT = "#eb6834"


def parse_args() -> argparse.Namespace:
    base = REPO_ROOT / "outputs/conceptual_belief_space/cot_window_layer_transition_probe"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--activation_index", type=Path, default=base / "activations/activation_index.parquet")
    parser.add_argument("--beliefs", type=Path, default=base / "activations/extracted_beliefs.parquet")
    parser.add_argument("--splits", type=Path,
                        default=REPO_ROOT / "outputs/conceptual_belief_space/splits/group_splits_mlx_full.json")
    parser.add_argument("--out_dir", type=Path,
                        default=REPO_ROOT / "outputs/conceptual_belief_space/layerwise_dissociation")
    parser.add_argument("--figure_dir", type=Path, default=REPO_ROOT / "figures/conceptual_belief_space")
    parser.add_argument("--subsets", type=str, default="strict,all")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_folds", type=int, default=None)
    return parser.parse_args()


def load_probe77() -> Any:
    path = REPO_ROOT / "scripts/conceptual_belief_space/77_train_forced_suffix_commitment_probe.py"
    spec = importlib.util.spec_from_file_location("probe77", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def pooled_activation_matrix(table: pd.DataFrame, cache_path: Path) -> np.ndarray:
    """Mask-weighted mean over the CoT-window token axis for every row -> [N, layers, dim] f16."""
    n_layers = int(table["n_layers"].iloc[0])
    dim = int(table["projected_dim"].iloc[0])
    ids_path = cache_path.with_suffix(".prefix_ids.npy")
    if cache_path.exists() and ids_path.exists():
        cached_ids = np.load(ids_path)
        if len(cached_ids) == len(table) and (cached_ids == table["prefix_id"].to_numpy()).all():
            print(f"[pooled] reusing cache {cache_path}")
            return np.load(cache_path, mmap_mode="r")
    store = ProjectedShardStore(max_open=16)
    out = np.empty((len(table), n_layers, dim), dtype=np.float16)
    work = table.copy()
    work["_out_pos"] = np.arange(len(work))
    done = 0
    for state_path, group in work.groupby("state_path", sort=False):
        records = group.to_dict("records")
        states, _rms = store.load_rows(records)  # [B, layers, tokens, dim] float32
        masks = np.stack([np.asarray(mask, dtype=np.float32) for mask in group["_input_token_mask"]])
        weights = masks / masks.sum(axis=1, keepdims=True)
        pooled = np.einsum("ba,blad->bld", weights, states)
        out[group["_out_pos"].to_numpy()] = pooled.astype(np.float16)
        done += len(group)
        if done % 20000 < len(group):
            print(f"[pooled] {done}/{len(work)} rows", flush=True)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache_path, out)
    np.save(ids_path, table["prefix_id"].to_numpy())
    print(f"[pooled] wrote cache {cache_path} shape={out.shape}")
    return out


def layer_features(pooled: np.ndarray, indices: np.ndarray, layer_token: str) -> np.ndarray:
    block = np.asarray(pooled[indices])
    if layer_token == "mean":
        return block.mean(axis=1, dtype=np.float32)
    return block[:, int(layer_token)].astype(np.float32)


def fit_action_probe(x_train: np.ndarray, y_train: np.ndarray, seed: int) -> Any:
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    return make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, class_weight="balanced", C=1.0, random_state=seed),
    ).fit(x_train, y_train)


def fit_commitment_probe(x_train: np.ndarray, y_train: np.ndarray) -> Any:
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    return make_pipeline(StandardScaler(), Ridge(alpha=1.0)).fit(x_train, y_train)


def positional_bars(table: pd.DataFrame, train_idx: np.ndarray, test_idx: np.ndarray,
                    action_codes: np.ndarray, commitment: np.ndarray, seed: int) -> dict[str, float]:
    from scipy.stats import spearmanr
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.metrics import balanced_accuracy_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import SplineTransformer, StandardScaler

    position = np.log1p(table["prefix_token_index"].to_numpy(dtype=np.float64))[:, None]
    action_model = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed)
    action_model.fit(position[train_idx], action_codes[train_idx])
    action_bar = float(balanced_accuracy_score(action_codes[test_idx], action_model.predict(position[test_idx])))
    commit_model = make_pipeline(StandardScaler(), SplineTransformer(n_knots=5, degree=3),
                                 StandardScaler(), Ridge(alpha=1.0))
    commit_model.fit(position[train_idx], commitment[train_idx])
    predictions = commit_model.predict(position[test_idx])
    commit_bar = float(spearmanr(commitment[test_idx], predictions).statistic)
    return {"action_balanced_accuracy": action_bar, "commitment_spearman": commit_bar}


def orthonormal_action_basis(x_std: np.ndarray, y: np.ndarray) -> np.ndarray:
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

    lda = LinearDiscriminantAnalysis(n_components=3).fit(x_std, y)
    basis, _ = np.linalg.qr(lda.scalings_[:, :3])
    return basis


def main() -> int:
    args = parse_args()
    probe77 = load_probe77()
    from scipy.stats import spearmanr
    from sklearn.metrics import balanced_accuracy_score, r2_score
    from sklearn.preprocessing import StandardScaler

    table = probe77.prepare_table(args.activation_index, args.beliefs, None)
    splits = probe77.load_outer_splits(args.splits, table)
    if args.max_folds is not None:
        splits = splits[: args.max_folds]
    n_layers = int(table["n_layers"].iloc[0])
    layer_tokens = [str(layer) for layer in range(n_layers)] + ["mean"]

    q_matrix = table[Q_COLS].to_numpy(dtype=np.float64)
    action_codes = q_matrix.argmax(axis=1)
    commitment = table["entropy_commitment"].to_numpy(dtype=np.float64)
    strict_mask = table[STRICT_COL].to_numpy(dtype=bool)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    pooled = pooled_activation_matrix(table, args.out_dir / "pooled_cot_window_f16.npy")

    subset_masks = {"strict": strict_mask, "all": np.ones(len(table), dtype=bool)}
    subsets = [token.strip() for token in args.subsets.split(",") if token.strip()]

    curve_rows: list[dict[str, Any]] = []
    bar_rows: list[dict[str, Any]] = []
    geometry_rows: list[dict[str, Any]] = []
    removal_rows: list[dict[str, Any]] = []

    for split in splits:
        fold = int(split["split_id"])
        for subset in subsets:
            mask = subset_masks[subset]
            train_idx = split["train_idx"][mask[split["train_idx"]]]
            test_idx = split["test_idx"][mask[split["test_idx"]]]
            bar = positional_bars(table, train_idx, test_idx, action_codes, commitment, args.seed)
            bar_rows.append({"fold": fold, "subset": subset, "n_train": len(train_idx),
                             "n_test": len(test_idx), **bar})
            print(f"[fold {fold} | {subset}] n_train={len(train_idx)} n_test={len(test_idx)} "
                  f"bars: action={bar['action_balanced_accuracy']:.4f} "
                  f"commit={bar['commitment_spearman']:.4f}", flush=True)

            for layer_token in layer_tokens:
                x_train = layer_features(pooled, train_idx, layer_token)
                x_test = layer_features(pooled, test_idx, layer_token)

                commit_probe = fit_commitment_probe(x_train, commitment[train_idx])
                commit_pred = commit_probe.predict(x_test)
                commit_spearman = float(spearmanr(commitment[test_idx], commit_pred).statistic)
                commit_r2 = float(r2_score(commitment[test_idx], commit_pred))

                action_probe = fit_action_probe(x_train, action_codes[train_idx], args.seed)
                action_pred = action_probe.predict(x_test)
                action_balanced = float(balanced_accuracy_score(action_codes[test_idx], action_pred))

                curve_rows.append({
                    "fold": fold, "subset": subset, "layer": layer_token,
                    "commitment_spearman": commit_spearman, "commitment_r2": commit_r2,
                    "action_balanced_accuracy": action_balanced,
                    "n_train": len(train_idx), "n_test": len(test_idx),
                })
                print(f"[fold {fold} | {subset}] layer {layer_token:>4}: "
                      f"action={action_balanced:.4f} commit={commit_spearman:.4f}", flush=True)

                if subset != "strict":
                    continue
                scaler = StandardScaler().fit(x_train)
                x_train_std = scaler.transform(x_train)
                x_test_std = scaler.transform(x_test)
                ridge_direction = commit_probe.named_steps["ridge"].coef_.astype(np.float64)
                ridge_direction = ridge_direction / np.linalg.norm(ridge_direction)
                basis = orthonormal_action_basis(x_train_std, action_codes[train_idx])
                cos_sq = float(np.sum((basis.T @ ridge_direction) ** 2))
                geometry_rows.append({
                    "fold": fold, "layer": layer_token, "commitment_in_action_subspace_cos_sq": cos_sq,
                    "chance_cos_sq": basis.shape[1] / x_train_std.shape[1],
                })

                if layer_token in REMOVAL_LAYERS:
                    x_train_no_action = x_train_std - (x_train_std @ basis) @ basis.T
                    x_test_no_action = x_test_std - (x_test_std @ basis) @ basis.T
                    commit_removed = fit_commitment_probe(x_train_no_action, commitment[train_idx])
                    removed_pred = commit_removed.predict(x_test_no_action)
                    commit_spearman_removed = float(spearmanr(commitment[test_idx], removed_pred).statistic)

                    projection = np.outer(ridge_direction, ridge_direction)
                    x_train_no_commit = x_train_std - x_train_std @ projection
                    x_test_no_commit = x_test_std - x_test_std @ projection
                    action_removed = fit_action_probe(x_train_no_commit, action_codes[train_idx], args.seed)
                    action_balanced_removed = float(balanced_accuracy_score(
                        action_codes[test_idx], action_removed.predict(x_test_no_commit)))
                    removal_rows.append({
                        "fold": fold, "layer": layer_token,
                        "commitment_spearman_full": commit_spearman,
                        "commitment_spearman_wo_action_subspace": commit_spearman_removed,
                        "action_balanced_full": action_balanced,
                        "action_balanced_wo_commitment_direction": action_balanced_removed,
                    })

    curves = pd.DataFrame(curve_rows)
    bars = pd.DataFrame(bar_rows)
    geometry = pd.DataFrame(geometry_rows)
    removal = pd.DataFrame(removal_rows)
    curves.to_csv(args.out_dir / "layerwise_curves.csv", index=False)
    bars.to_csv(args.out_dir / "positional_bars.csv", index=False)
    geometry.to_csv(args.out_dir / "geometry.csv", index=False)
    removal.to_csv(args.out_dir / "removal_tests.csv", index=False)

    physical = curves[curves.layer != "mean"].copy()
    physical["layer_index"] = physical.layer.astype(int)
    contrast_rows = []
    for (fold, subset), group in physical.groupby(["fold", "subset"]):
        early = group[group.layer_index.isin(EARLY_BLOCK)]
        late = group[group.layer_index.isin(LATE_BLOCK)]
        action_delta = float(late.action_balanced_accuracy.mean() - early.action_balanced_accuracy.mean())
        commit_delta = float(late.commitment_spearman.mean() - early.commitment_spearman.mean())
        contrast_rows.append({
            "fold": fold, "subset": subset,
            "action_late_minus_early": action_delta,
            "commitment_late_minus_early": commit_delta,
            "interaction_action_minus_commitment": action_delta - commit_delta,
            "action_peak_layer": int(group.loc[group.action_balanced_accuracy.idxmax(), "layer_index"]),
            "commitment_peak_layer": int(group.loc[group.commitment_spearman.idxmax(), "layer_index"]),
        })
    contrast = pd.DataFrame(contrast_rows)
    contrast.to_csv(args.out_dir / "block_contrast.csv", index=False)

    draw_figures(curves, bars, geometry, args.figure_dir)
    write_report(args, curves, bars, geometry, removal, contrast)
    print(f"Wrote outputs to {args.out_dir}")
    return 0


def draw_figures(curves: pd.DataFrame, bars: pd.DataFrame, geometry: pd.DataFrame,
                 figure_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure_dir.mkdir(parents=True, exist_ok=True)
    physical = curves[curves.layer != "mean"].copy()
    physical["layer_index"] = physical.layer.astype(int)

    for subset in physical.subset.unique():
        sub = physical[physical.subset == subset]
        figure, axes = plt.subplots(1, 2, figsize=(11, 4.2))
        specs = ((axes[0], "action_balanced_accuracy", "top-action balanced accuracy",
                  COLOR_ACTION, "action_balanced_accuracy"),
                 (axes[1], "commitment_spearman", "commitment Spearman",
                  COLOR_COMMIT, "commitment_spearman"))
        for axis, column, label, color, bar_column in specs:
            stats = sub.groupby("layer_index")[column].agg(["mean", "min", "max"])
            axis.fill_between(stats.index, stats["min"], stats["max"], color=color, alpha=0.18, linewidth=0)
            axis.plot(stats.index, stats["mean"], color=color, linewidth=2)
            bar_value = bars.loc[bars.subset == subset, bar_column].mean()
            axis.axhline(bar_value, color=INK, linewidth=1.2, linestyle=":")
            axis.annotate("position-only bar", (0.3, bar_value), textcoords="offset points",
                          xytext=(0, 4), color=INK, fontsize=8)
            best = stats["mean"].idxmax()
            axis.annotate(f"peak L{best}", (best, stats.loc[best, 'mean']), textcoords="offset points",
                          xytext=(0, 6), ha="center", color=INK, fontsize=9)
            axis.set_title(label, color=INK, fontsize=11)
            axis.set_xlabel("layer (block output)", color=INK)
            axis.set_ylabel(label, color=INK)
            axis.grid(True, color=GRID, linewidth=0.8)
            axis.tick_params(colors=INK)
            for spine in ("top", "right"):
                axis.spines[spine].set_visible(False)
        figure.suptitle(f"Per-layer decoding, same instrument, {subset} subset "
                        f"(mean and fold range over {physical.fold.nunique()} folds)", color=INK)
        figure.tight_layout()
        figure.savefig(figure_dir / f"layerwise_dissociation_{subset}.png", dpi=200)
        plt.close(figure)

    if not geometry.empty:
        physical_geometry = geometry[geometry.layer != "mean"].copy()
        physical_geometry["layer_index"] = physical_geometry.layer.astype(int)
        figure, axis = plt.subplots(figsize=(7, 4))
        stats = physical_geometry.groupby("layer_index")["commitment_in_action_subspace_cos_sq"] \
            .agg(["mean", "min", "max"])
        axis.fill_between(stats.index, stats["min"], stats["max"], color=COLOR_COMMIT, alpha=0.18, linewidth=0)
        axis.plot(stats.index, stats["mean"], color=COLOR_COMMIT, linewidth=2)
        chance = physical_geometry["chance_cos_sq"].iloc[0]
        axis.axhline(chance, color=INK, linewidth=1.2, linestyle=":")
        axis.annotate("random-direction chance", (0.3, chance), textcoords="offset points",
                      xytext=(0, 4), color=INK, fontsize=8)
        axis.set_title("Fraction of the commitment direction inside the LDA action subspace",
                       color=INK, fontsize=11)
        axis.set_xlabel("layer (block output)", color=INK)
        axis.set_ylabel("cos^2 into 3-d action subspace", color=INK)
        axis.grid(True, color=GRID, linewidth=0.8)
        axis.tick_params(colors=INK)
        for spine in ("top", "right"):
            axis.spines[spine].set_visible(False)
        figure.tight_layout()
        figure.savefig(figure_dir / "layerwise_commitment_action_geometry.png", dpi=200)
        plt.close(figure)


def write_report(args: argparse.Namespace, curves: pd.DataFrame, bars: pd.DataFrame,
                 geometry: pd.DataFrame, removal: pd.DataFrame, contrast: pd.DataFrame) -> None:
    lines = [
        "# Layerwise Dissociation: commitment vs action identity, matched instruments",
        "",
        "Per-layer linear decoding of `entropy_commitment` (ridge, Spearman) and forced top action",
        "(balanced logistic, balanced accuracy) from mask-pooled CoT-window activations, all outer",
        "folds, trajectory-grouped, faithful labels. Positional bars use only `prefix_token_index`.",
        "",
        "## Block contrast (late L16-23 minus early L0-7, per fold)",
        "",
        contrast.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Peak layers per fold",
        "",
        contrast[["fold", "subset", "action_peak_layer", "commitment_peak_layer"]]
        .to_markdown(index=False),
        "",
        "## Positional bars",
        "",
        bars.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Removal tests (strict subset)",
        "",
        removal.to_markdown(index=False, floatfmt=".4f") if not removal.empty else "(none)",
        "",
        "## Mean curve values",
        "",
        curves.groupby(["subset", "layer"])[["action_balanced_accuracy", "commitment_spearman",
                                             "commitment_r2"]].mean().round(4).to_markdown(),
        "",
    ]
    (args.out_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
