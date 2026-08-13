#!/usr/bin/env python3
"""Embed activation features into a 2D activation manifold."""

from __future__ import annotations

import argparse
import json
import math
import os
from itertools import combinations
from pathlib import Path
from typing import Any

if "XDG_CACHE_HOME" not in os.environ:
    _xdg_cache = Path(os.environ.get("TMPDIR", "/tmp")) / "spar_cache"
    _xdg_cache.mkdir(parents=True, exist_ok=True)
    os.environ["XDG_CACHE_HOME"] = str(_xdg_cache)
if "MPLCONFIGDIR" not in os.environ:
    _mpl_cache = Path(os.environ.get("TMPDIR", "/tmp")) / "spar_matplotlib"
    _mpl_cache.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(_mpl_cache)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import IncrementalPCA, PCA
from sklearn.preprocessing import StandardScaler

try:
    import joblib
except ModuleNotFoundError as exc:  # pragma: no cover
    raise SystemExit("Missing dependency: joblib. Install joblib to save PCA models.") from exc


ACTIONS = ["UP", "DOWN", "LEFT", "RIGHT"]
ACTION_COLORS = {
    "UP": "#4C78A8",
    "DOWN": "#F58518",
    "LEFT": "#54A24B",
    "RIGHT": "#B279A2",
}
DEFAULT_REPRESENTATION = "delta_pre"
DEFAULT_PROGRESS_COVARIATES = ["progress", "prefix_token_index", "num_cot_tokens"]


def parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    lowered = value.lower()
    if lowered in {"true", "1", "yes", "y"}:
        return True
    if lowered in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected boolean value, got {value!r}")


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def infer_repo_root(path: Path) -> Path:
    path = path.resolve()
    for parent in [path.parent, *path.parents]:
        if (parent / "outputs").exists() and (parent / "scripts").exists():
            return parent
    return Path.cwd().resolve()


def resolve_path(path: str | Path, repo_root: Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return repo_root / path


def feature_file(features_dir: Path, layer: int, representation: str) -> Path:
    return features_dir / f"layer_{layer:02d}_{representation}.npy"


def discover_layers(features_dir: Path, representation: str) -> list[int]:
    layers: set[int] = set()
    for path in features_dir.glob(f"layer_*_{representation}.npy"):
        parts = path.stem.split("_")
        if len(parts) >= 3 and parts[0] == "layer":
            try:
                layers.add(int(parts[1]))
            except ValueError:
                continue
    return sorted(layers)


def recursively_find_layer(obj: Any) -> int | None:
    if isinstance(obj, dict):
        if "layer" in obj and obj["layer"] is not None:
            try:
                return int(obj["layer"])
            except (TypeError, ValueError):
                pass
        preferred = [
            "best_layer_behavior_clr_prediction",
            "best_layer_commitment_prediction",
            "best_layer_strict_pre_action_decoding",
            "best_representation",
        ]
        for key in preferred:
            if key in obj:
                found = recursively_find_layer(obj[key])
                if found is not None:
                    return found
        for value in obj.values():
            found = recursively_find_layer(value)
            if found is not None:
                return found
    elif isinstance(obj, list):
        for value in obj:
            found = recursively_find_layer(value)
            if found is not None:
                return found
    return None


def select_layer(
    *,
    requested_layer: int | None,
    config_path: Path | None,
    available_layers: list[int],
) -> tuple[int, str]:
    if not available_layers:
        raise SystemExit("No activation feature layers are available for the selected representation.")
    if requested_layer is not None:
        if requested_layer not in available_layers:
            raise SystemExit(
                f"Requested layer {requested_layer} is unavailable. Available layers: {available_layers}"
            )
        return requested_layer, "explicit --layer"

    if config_path is not None and config_path.exists():
        try:
            config = json.loads(config_path.read_text())
            layer = recursively_find_layer(config)
            if layer in available_layers:
                return int(layer), f"best_activation_config: {config_path}"
            if layer is not None:
                print(
                    f"WARNING: config selected layer {layer}, but it is unavailable for this representation; using fallback."
                )
        except Exception as exc:
            print(f"WARNING: failed to read {config_path}: {exc}; using fallback layer.")

    index = int(round(0.75 * (len(available_layers) - 1)))
    return available_layers[index], "mid-late fallback from available feature layers"


def finite_row_mask(matrix: np.ndarray, *, batch_size: int) -> np.ndarray:
    mask = np.ones(matrix.shape[0], dtype=bool)
    for start in range(0, matrix.shape[0], batch_size):
        stop = min(start + batch_size, matrix.shape[0])
        mask[start:stop] = np.isfinite(matrix[start:stop]).all(axis=1)
    return mask


def alignment_mask(meta: pd.DataFrame) -> np.ndarray:
    if "alignment_status" not in meta.columns:
        return np.ones(len(meta), dtype=bool)
    status = meta["alignment_status"].fillna("").astype(str)
    return ~status.str.startswith("failed").to_numpy()


def l2_normalize_rows(matrix: np.ndarray, eps: float = 1e-12) -> tuple[np.ndarray, dict[str, Any]]:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    near_zero = norms[:, 0] < eps
    safe_norms = np.where(norms < eps, 1.0, norms)
    normalized = matrix / safe_norms
    return normalized.astype(np.float32, copy=False), {
        "eps": eps,
        "near_zero_rows": int(near_zero.sum()),
        "mean_norm_before": float(np.mean(norms)),
        "median_norm_before": float(np.median(norms)),
    }


def residualize_covariates(
    matrix: np.ndarray,
    meta: pd.DataFrame,
    covariate_columns: list[str],
) -> tuple[np.ndarray, dict[str, Any]]:
    present = [col for col in covariate_columns if col in meta.columns]
    if not present:
        return matrix, {"enabled": False, "reason": "no requested covariates present", "columns": []}

    cov = meta[present].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    finite_cols = np.isfinite(cov).all(axis=0)
    present = [col for col, ok in zip(present, finite_cols) if ok]
    cov = cov[:, finite_cols]
    if cov.shape[1] == 0:
        return matrix, {"enabled": False, "reason": "covariates are non-finite", "columns": []}

    means = cov.mean(axis=0)
    scales = cov.std(axis=0)
    scales = np.where(scales < 1e-12, 1.0, scales)
    cov_z = (cov - means) / scales
    beta, *_ = np.linalg.lstsq(cov_z, matrix, rcond=None)
    residual = matrix - cov_z @ beta
    return residual.astype(np.float32, copy=False), {
        "enabled": True,
        "columns": present,
        "means": means.tolist(),
        "scales": scales.tolist(),
        "beta_shape": list(beta.shape),
    }


def fit_pca_embedding(
    matrix: np.ndarray,
    *,
    random_state: int,
    use_incremental: bool,
    batch_size: int,
) -> tuple[Any, np.ndarray, str]:
    if use_incremental:
        pca = IncrementalPCA(n_components=2, batch_size=batch_size)
        embedding = pca.fit_transform(matrix)
        return pca, embedding, "IncrementalPCA"
    pca = PCA(n_components=2, svd_solver="randomized", random_state=random_state)
    embedding = pca.fit_transform(matrix)
    return pca, embedding, "PCA"


def fit_umap_if_requested(
    matrix: np.ndarray,
    *,
    use_umap: bool,
    n_neighbors: int,
    min_dist: float,
    random_state: int,
) -> tuple[Any | None, np.ndarray | None, str]:
    if not use_umap:
        return None, None, "not requested"
    try:
        import umap
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "UMAP requested with --use_umap, but umap-learn is not installed. "
            "Install umap-learn or omit --use_umap."
        ) from exc
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric="euclidean",
        random_state=random_state,
    )
    return reducer, reducer.fit_transform(matrix), "fit"


def action_cluster_diagnostic(df: pd.DataFrame, action_col: str, *, mask: np.ndarray | None = None) -> dict[str, Any]:
    if action_col not in df.columns:
        return {"available": False, "reason": f"missing {action_col}"}
    work = df.loc[mask].copy() if mask is not None else df.copy()
    work = work.dropna(subset=[action_col, "act_pca1", "act_pca2"])
    if work[action_col].nunique() < 2:
        return {"available": False, "reason": "fewer than two action classes"}
    centroids: dict[str, np.ndarray] = {}
    within: list[float] = []
    for action, group in work.groupby(action_col):
        xy = group[["act_pca1", "act_pca2"]].to_numpy(dtype=float)
        centroid = xy.mean(axis=0)
        centroids[str(action)] = centroid
        within.extend(np.linalg.norm(xy - centroid, axis=1).tolist())
    distances = [float(np.linalg.norm(centroids[a] - centroids[b])) for a, b in combinations(centroids, 2)]
    mean_between = float(np.mean(distances)) if distances else float("nan")
    mean_within = float(np.mean(within)) if within else float("nan")
    ratio = mean_between / mean_within if mean_within and np.isfinite(mean_within) else float("nan")
    if not np.isfinite(ratio):
        label = "unavailable"
    elif ratio >= 0.75:
        label = "moderate separation"
    elif ratio >= 0.4:
        label = "weak separation"
    else:
        label = "little visible separation"
    return {
        "available": True,
        "classes": sorted(centroids),
        "mean_between_centroid_distance": mean_between,
        "mean_within_class_radius": mean_within,
        "between_within_ratio": float(ratio),
        "interpretation": label,
    }


def commitment_periphery_diagnostic(df: pd.DataFrame) -> dict[str, Any]:
    if "commitment_to_final" not in df.columns:
        return {"available": False, "reason": "missing commitment_to_final"}
    radius = np.linalg.norm(df[["act_pca1", "act_pca2"]].to_numpy(dtype=float), axis=1)
    commitment = pd.to_numeric(df["commitment_to_final"], errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(radius) & np.isfinite(commitment)
    high = valid & (commitment >= 0.8)
    if high.sum() == 0 or valid.sum() == 0:
        return {"available": False, "reason": "no high-commitment points"}
    high_mean = float(radius[high].mean())
    all_mean = float(radius[valid].mean())
    ratio = high_mean / all_mean if all_mean else float("nan")
    if ratio >= 1.1:
        label = "high-commitment points are more peripheral on average"
    elif ratio <= 0.9:
        label = "high-commitment points are more central on average"
    else:
        label = "high-commitment points have similar radius to all points"
    return {
        "available": True,
        "high_commitment_points": int(high.sum()),
        "mean_radius_high_commitment": high_mean,
        "mean_radius_all": all_mean,
        "radius_ratio": float(ratio),
        "interpretation": label,
    }


def strict_mask(df: pd.DataFrame, column: str) -> np.ndarray:
    if column not in df.columns:
        return np.zeros(len(df), dtype=bool)
    return df[column].fillna(True).astype(bool).to_numpy()


def scatter_by_action(
    df: pd.DataFrame,
    *,
    action_col: str,
    path: Path,
    title: str,
    mask: np.ndarray | None = None,
    alpha: float = 0.65,
    size: float = 10.0,
) -> None:
    work = df.loc[mask].copy() if mask is not None else df.copy()
    fig, ax = plt.subplots(figsize=(7, 5))
    if work.empty or action_col not in work.columns:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
    else:
        seen = set()
        for action in ACTIONS + sorted(set(work[action_col].dropna().astype(str)) - set(ACTIONS)):
            group = work[work[action_col].astype(str).eq(action)]
            if group.empty:
                continue
            seen.add(action)
            ax.scatter(
                group["act_pca1"],
                group["act_pca2"],
                s=size,
                alpha=alpha,
                color=ACTION_COLORS.get(action, "#666666"),
                label=action,
                linewidths=0,
            )
        if seen:
            ax.legend(title=action_col, markerscale=1.5)
    ax.set_xlabel("act_pca1")
    ax.set_ylabel("act_pca2")
    ax.set_title(title)
    ax.grid(True, alpha=0.2)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def scatter_by_scalar(
    df: pd.DataFrame,
    *,
    value_col: str,
    path: Path,
    title: str,
    cmap: str = "viridis",
    alpha: float = 0.75,
    size: float = 10.0,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    if df.empty or value_col not in df.columns:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
    else:
        values = pd.to_numeric(df[value_col], errors="coerce")
        scatter = ax.scatter(
            df["act_pca1"],
            df["act_pca2"],
            c=values,
            s=size,
            alpha=alpha,
            cmap=cmap,
            linewidths=0,
        )
        fig.colorbar(scatter, ax=ax, label=value_col)
    ax.set_xlabel("act_pca1")
    ax.set_ylabel("act_pca2")
    ax.set_title(title)
    ax.grid(True, alpha=0.2)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def choose_example_decisions(df: pd.DataFrame, max_decisions: int) -> list[str]:
    if "decision_id" not in df.columns:
        return []
    counts = df.groupby("decision_id").size().sort_values(ascending=False)
    if counts.empty:
        return []
    selected: list[str] = []
    if "agent_action" in df.columns:
        for _, action_group in df.groupby("agent_action", dropna=False):
            action_counts = action_group.groupby("decision_id").size().sort_values(ascending=False)
            for decision_id in action_counts.index:
                if decision_id not in selected:
                    selected.append(str(decision_id))
                    break
                if len(selected) >= max_decisions:
                    return selected[:max_decisions]
    for decision_id in counts.index:
        if str(decision_id) not in selected:
            selected.append(str(decision_id))
        if len(selected) >= max_decisions:
            break
    return selected[:max_decisions]


def plot_example_trajectories(df: pd.DataFrame, *, path: Path, max_decisions: int) -> dict[str, Any]:
    decisions = choose_example_decisions(df, max_decisions)
    fig, ax = plt.subplots(figsize=(8, 6))
    if not decisions:
        ax.text(0.5, 0.5, "No decision trajectories available", ha="center", va="center", transform=ax.transAxes)
    else:
        colors = plt.cm.tab10(np.linspace(0, 1, max(len(decisions), 1)))
        used_labels: set[str] = set()
        post_marked = False
        for color, decision_id in zip(colors, decisions):
            group = df[df["decision_id"].astype(str).eq(decision_id)].copy()
            sort_cols = [col for col in ["prefix_token_index", "progress"] if col in group.columns]
            if sort_cols:
                group = group.sort_values(sort_cols)
            xy = group[["act_pca1", "act_pca2"]].to_numpy(dtype=float)
            if len(xy) == 0:
                continue
            action = str(group["agent_action"].iloc[0]) if "agent_action" in group.columns else decision_id
            ax.plot(xy[:, 0], xy[:, 1], color=color, alpha=0.75, linewidth=1.3, label=f"{action}: {decision_id}")
            start_label = "start/pre_reasoning" if "start/pre_reasoning" not in used_labels else None
            final_label = "final prefix" if "final prefix" not in used_labels else None
            ax.scatter(xy[0, 0], xy[0, 1], s=70, facecolor="white", edgecolor=color, linewidth=1.8, label=start_label, zorder=4)
            ax.scatter(xy[-1, 0], xy[-1, 1], s=70, marker="X", color=color, edgecolor="black", linewidth=0.5, label=final_label, zorder=4)
            used_labels.update(label for label in [start_label, final_label] if label)

            token_type_col = "activation_token_type" if "activation_token_type" in group.columns else "token_type" if "token_type" in group.columns else None
            if token_type_col and group[token_type_col].astype(str).eq("post_reasoning").any():
                post = group[group[token_type_col].astype(str).eq("post_reasoning")].iloc[-1]
                post_label = "post_reasoning" if "post_reasoning" not in used_labels else None
                ax.scatter(post["act_pca1"], post["act_pca2"], s=80, marker="D", color=color, edgecolor="black", label=post_label, zorder=5)
                post_marked = True
                if post_label:
                    used_labels.add(post_label)
    ax.set_xlabel("act_pca1")
    ax.set_ylabel("act_pca2")
    ax.set_title("Example activation trajectories")
    ax.grid(True, alpha=0.2)
    ax.legend(fontsize=7, loc="best")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return {"decision_ids": decisions, "post_reasoning_marked": bool(locals().get("post_marked", False))}


def fmt_float(value: Any, digits: int = 4) -> str:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return "NA"
    if not math.isfinite(value):
        return "NA"
    return f"{value:.{digits}f}"


def write_report(
    path: Path,
    *,
    out: pd.DataFrame,
    layer: int,
    representation: str,
    layer_source: str,
    feature_path: Path,
    model_path: Path,
    pca: Any,
    pca_method: str,
    preprocessing: dict[str, Any],
    umap_status: str,
    diagnostics: dict[str, Any],
    figures: list[Path],
) -> None:
    lines: list[str] = []
    lines.append("# Activation Manifold Report\n")
    lines.append("## Summary\n")
    lines.append(f"- Number of points: {len(out):,}")
    lines.append(f"- Number of decisions: {out['decision_id'].nunique() if 'decision_id' in out.columns else 'NA'}")
    lines.append(f"- Number of trajectories: {out['trajectory_id'].nunique() if 'trajectory_id' in out.columns else 'NA'}")
    lines.append(f"- Selected layer: {layer}")
    lines.append(f"- Selected representation: `{representation}`")
    lines.append(f"- Layer source: {layer_source}")
    lines.append(f"- Feature file: `{feature_path}`")
    lines.append(f"- PCA method: {pca_method}")
    lines.append(f"- PCA explained variance ratio: `{[float(x) for x in pca.explained_variance_ratio_]}`")
    lines.append(f"- UMAP status: {umap_status}")
    lines.append(f"- Model/preprocessing output: `{model_path}`")
    lines.append("")

    lines.append("## Preprocessing\n")
    for key, value in preprocessing.items():
        if key == "scaler":
            continue
        lines.append(f"- `{key}`: `{json.dumps(value, default=str)}`")
    lines.append("")

    cluster = diagnostics.get("agent_action_clusters", {})
    lines.append("## Qualitative Diagnostics\n")
    if cluster.get("available"):
        lines.append(
            "- Action clusters: "
            f"{cluster['interpretation']} "
            f"(between/within ratio {fmt_float(cluster['between_within_ratio'])})."
        )
    else:
        lines.append(f"- Action clusters: unavailable ({cluster.get('reason', 'unknown reason')}).")

    periphery = diagnostics.get("commitment_periphery", {})
    if periphery.get("available"):
        lines.append(
            "- High-commitment geometry: "
            f"{periphery['interpretation']} "
            f"(radius ratio {fmt_float(periphery['radius_ratio'])})."
        )
    else:
        lines.append(f"- High-commitment geometry: unavailable ({periphery.get('reason', 'unknown reason')}).")

    strict = diagnostics.get("strict_pre_action_clusters", {})
    if strict.get("available"):
        lines.append(
            "- Strict pre-action final-action separation: "
            f"{strict['interpretation']} "
            f"(between/within ratio {fmt_float(strict['between_within_ratio'])})."
        )
    else:
        lines.append(f"- Strict pre-action final-action separation: unavailable ({strict.get('reason', 'unknown reason')}).")
    lines.append("- Caveat: PCA is unsupervised; visible separation is descriptive and should not be treated as decoding evidence by itself.")
    lines.append("")

    if "agent_action" in out.columns:
        lines.append("## Agent Action Counts\n")
        counts = out["agent_action"].value_counts(dropna=False).sort_index()
        lines.append(counts.to_markdown())
        lines.append("")
    if "top_action" in out.columns:
        lines.append("## Top Action Counts\n")
        counts = out["top_action"].value_counts(dropna=False).sort_index()
        lines.append(counts.to_markdown())
        lines.append("")

    lines.append("## Figures\n")
    for fig_path in figures:
        lines.append(f"- `{fig_path}`")
    lines.append("")

    ensure_dir(path.parent)
    path.write_text("\n".join(lines))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", default="outputs/activation/activation_feature_metadata.parquet")
    parser.add_argument("--features_dir", default="outputs/activation/features")
    parser.add_argument("--config", default="outputs/activation/best_activation_config.json")
    parser.add_argument("--output", default="outputs/activation/activation_manifold.parquet")
    parser.add_argument("--models_dir", default="outputs/activation/models")
    parser.add_argument("--report", default="outputs/activation/reports/activation_manifold_report.md")
    parser.add_argument("--figures_dir", default="figures/activation/manifolds")
    parser.add_argument("--representation", default=DEFAULT_REPRESENTATION)
    parser.add_argument("--layer", type=int, default=None)
    parser.add_argument("--l2_normalize", type=parse_bool, default=False)
    parser.add_argument("--residualize_progress", type=parse_bool, default=False)
    parser.add_argument("--residualize_columns", default=",".join(DEFAULT_PROGRESS_COVARIATES))
    parser.add_argument("--use_umap", action="store_true")
    parser.add_argument("--umap_n_neighbors", type=int, default=15)
    parser.add_argument("--umap_min_dist", type=float, default=0.1)
    parser.add_argument("--use_incremental_pca", choices=["auto", "true", "false"], default="auto")
    parser.add_argument("--incremental_threshold_mb", type=float, default=1024.0)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--random_state", type=int, default=0)
    parser.add_argument("--num_example_decisions", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metadata_path = Path(args.metadata)
    repo_root = infer_repo_root(metadata_path)
    metadata_path = resolve_path(metadata_path, repo_root)
    features_dir = resolve_path(args.features_dir, repo_root)
    config_path = resolve_path(args.config, repo_root) if args.config else None
    output_path = resolve_path(args.output, repo_root)
    models_dir = resolve_path(args.models_dir, repo_root)
    report_path = resolve_path(args.report, repo_root)
    figures_dir = resolve_path(args.figures_dir, repo_root)

    meta = pd.read_parquet(metadata_path)
    available_layers = discover_layers(features_dir, args.representation)
    layer, layer_source = select_layer(
        requested_layer=args.layer,
        config_path=config_path,
        available_layers=available_layers,
    )
    selected_feature_path = feature_file(features_dir, layer, args.representation)
    if not selected_feature_path.exists():
        raise SystemExit(f"Feature array not found: {selected_feature_path}")

    X_mmap = np.load(selected_feature_path, mmap_mode="r")
    if X_mmap.shape[0] != len(meta):
        raise SystemExit(
            f"Feature row count mismatch: {X_mmap.shape[0]} rows in {selected_feature_path}, "
            f"but metadata has {len(meta)} rows."
        )
    if X_mmap.shape[0] < 2 or X_mmap.shape[1] < 2:
        raise SystemExit(f"Need at least a 2x2 feature matrix for PCA, got {X_mmap.shape}")

    align_mask = alignment_mask(meta)
    finite_mask = finite_row_mask(X_mmap, batch_size=args.batch_size)
    keep_mask = align_mask & finite_mask
    dropped_alignment = int((~align_mask).sum())
    dropped_nonfinite = int((align_mask & ~finite_mask).sum())
    if keep_mask.sum() < 2:
        raise SystemExit("Fewer than two valid aligned/finite rows remain for manifold embedding.")

    meta_kept = meta.loc[keep_mask].reset_index(drop=True)
    X = np.asarray(X_mmap[keep_mask], dtype=np.float32)

    scaler = StandardScaler()
    X_proc = scaler.fit_transform(X).astype(np.float32, copy=False)
    preprocessing: dict[str, Any] = {
        "metadata_path": str(metadata_path),
        "feature_path": str(selected_feature_path),
        "dropped_failed_alignment_rows": dropped_alignment,
        "dropped_nonfinite_feature_rows": dropped_nonfinite,
        "standardized": True,
        "standard_scaler_mean_shape": list(scaler.mean_.shape),
        "standard_scaler_scale_shape": list(scaler.scale_.shape),
        "l2_normalize": bool(args.l2_normalize),
        "residualize_progress": bool(args.residualize_progress),
    }

    l2_params: dict[str, Any] | None = None
    if args.l2_normalize:
        X_proc, l2_params = l2_normalize_rows(X_proc)
    preprocessing["l2_params"] = l2_params

    residualizer: dict[str, Any] | None = None
    if args.residualize_progress:
        residual_cols = [col.strip() for col in args.residualize_columns.split(",") if col.strip()]
        X_proc, residualizer = residualize_covariates(X_proc, meta_kept, residual_cols)
    preprocessing["residualizer"] = residualizer

    matrix_mb = float(X_proc.nbytes / (1024**2))
    if args.use_incremental_pca == "true":
        use_incremental = True
    elif args.use_incremental_pca == "false":
        use_incremental = False
    else:
        use_incremental = matrix_mb >= args.incremental_threshold_mb

    pca, pca_embedding, pca_method = fit_pca_embedding(
        X_proc,
        random_state=args.random_state,
        use_incremental=use_incremental,
        batch_size=args.batch_size,
    )
    umap_model, umap_embedding, umap_status = fit_umap_if_requested(
        X_proc,
        use_umap=args.use_umap,
        n_neighbors=args.umap_n_neighbors,
        min_dist=args.umap_min_dist,
        random_state=args.random_state,
    )

    out = meta_kept.copy()
    out["act_pca1"] = pca_embedding[:, 0]
    out["act_pca2"] = pca_embedding[:, 1]
    if umap_embedding is not None:
        out["act_umap1"] = umap_embedding[:, 0]
        out["act_umap2"] = umap_embedding[:, 1]
    out["layer"] = int(layer)
    out["representation"] = args.representation

    ensure_dir(output_path.parent)
    out.to_parquet(output_path, index=False)

    model_path = models_dir / f"activation_pca_layer_{layer}_{args.representation}.joblib"
    ensure_dir(model_path.parent)
    joblib.dump(
        {
            "pca": pca,
            "pca_method": pca_method,
            "scaler": scaler,
            "umap": umap_model,
            "layer": int(layer),
            "representation": args.representation,
            "preprocessing": preprocessing,
            "feature_shape_before_filter": list(X_mmap.shape),
            "feature_shape_after_filter": list(X_proc.shape),
            "pca_explained_variance_ratio": [float(x) for x in pca.explained_variance_ratio_],
        },
        model_path,
    )

    figures: list[Path] = []
    fig = figures_dir / "pca_by_agent_action.png"
    scatter_by_action(out, action_col="agent_action", path=fig, title="Activation PCA by agent action")
    figures.append(fig)
    fig = figures_dir / "pca_by_top_action.png"
    scatter_by_action(out, action_col="top_action", path=fig, title="Activation PCA by top action")
    figures.append(fig)
    fig = figures_dir / "pca_by_commitment_to_final.png"
    scatter_by_scalar(out, value_col="commitment_to_final", path=fig, title="Activation PCA by commitment to final")
    figures.append(fig)
    fig = figures_dir / "pca_by_entropy_commitment.png"
    scatter_by_scalar(out, value_col="entropy_commitment", path=fig, title="Activation PCA by entropy commitment")
    figures.append(fig)
    fig = figures_dir / "pca_by_progress.png"
    scatter_by_scalar(out, value_col="progress", path=fig, title="Activation PCA by progress")
    figures.append(fig)
    fig = figures_dir / "pca_strict_before_any_action_mention_by_agent_action.png"
    before_any_mask = strict_mask(out, "is_before_first_action_mention")
    scatter_by_action(
        out,
        action_col="agent_action",
        path=fig,
        title="Strict before any action mention by agent action",
        mask=before_any_mask,
    )
    figures.append(fig)
    fig = figures_dir / "pca_strict_before_final_action_mention_by_agent_action.png"
    before_final_mask = strict_mask(out, "is_before_first_final_action_mention")
    scatter_by_action(
        out,
        action_col="agent_action",
        path=fig,
        title="Strict before final-action mention by agent action",
        mask=before_final_mask,
    )
    figures.append(fig)
    fig = figures_dir / "example_activation_trajectories.png"
    trajectory_info = plot_example_trajectories(out, path=fig, max_decisions=args.num_example_decisions)
    figures.append(fig)

    diagnostics = {
        "agent_action_clusters": action_cluster_diagnostic(out, "agent_action"),
        "commitment_periphery": commitment_periphery_diagnostic(out),
        "strict_pre_action_clusters": action_cluster_diagnostic(out, "agent_action", mask=before_any_mask),
        "example_trajectories": trajectory_info,
    }
    write_report(
        report_path,
        out=out,
        layer=layer,
        representation=args.representation,
        layer_source=layer_source,
        feature_path=selected_feature_path,
        model_path=model_path,
        pca=pca,
        pca_method=pca_method,
        preprocessing=preprocessing,
        umap_status=umap_status,
        diagnostics=diagnostics,
        figures=figures,
    )

    print(f"Selected layer {layer} ({layer_source}), representation {args.representation}")
    print(f"PCA explained variance ratio: {[float(x) for x in pca.explained_variance_ratio_]}")
    print(f"Wrote {output_path}")
    print(f"Wrote {model_path}")
    print(f"Wrote {report_path}")
    print(f"Wrote {len(figures)} figures under {figures_dir}")


if __name__ == "__main__":
    main()
