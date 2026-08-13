#!/usr/bin/env python3
"""Step 7: sparse linear probes from activations to behavior belief."""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.conceptual_belief_space.activation_features import (  # noqa: E402
    activation_feature_path,
    fit_train_standardizer,
    load_activation_memmap,
    parse_layers_arg,
    parse_representations_arg,
    transform_indices_with_standardizer,
)
from src.conceptual_belief_space.beliefs import stable_softmax  # noqa: E402
from src.conceptual_belief_space.config import ACTIONS, DEFAULT_FIGURE_DIR, DEFAULT_OUTPUT_DIR, DEFAULT_SEED  # noqa: E402
from src.conceptual_belief_space.io import ensure_dir, read_json, read_parquet_checked, save_markdown, write_json  # noqa: E402
from src.conceptual_belief_space.metrics import (  # noqa: E402
    action_argmax,
    classification_metrics,
    commitment_from_q,
    normalize_q_predictions,
    q_from_clr,
    q_prediction_metrics,
    regression_vector_metrics,
    scalar_regression_metrics,
)
from src.conceptual_belief_space.plotting import line_metric_by_layer, savefig  # noqa: E402
from src.conceptual_belief_space.probes import (  # noqa: E402
    can_fit_classifier,
    fit_predict_torch_ridge,
    make_logistic_regression,
    make_multitask_elasticnet,
    make_ridge,
    resolve_torch_device_name,
    shuffled_copy,
)


Q_COLS = [f"q_{action}" for action in ACTIONS]
CLR_COLS = [f"clr_{action}" for action in ACTIONS]
SUBSETS = [
    "all_prefixes",
    "strict_before_any_action_mention",
    "strict_before_final_action_mention",
    "high_commitment",
    "low_commitment",
]
META_COLS = [
    "row_id",
    "decision_id",
    "trajectory_id",
    "step_id",
    "prefix_id",
    "prefix_token_index",
    "progress",
    "agent_action",
    "top_action",
    "commitment_to_final",
    "entropy_commitment",
]


def resolve_repo_path(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def rel(path: str | Path) -> str:
    path = Path(path)
    try:
        return str(path.resolve().relative_to(REPO_ROOT.resolve()))
    except ValueError:
        return str(path)


def load_splits(path: Path) -> list[dict[str, Any]]:
    with path.open("r") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"{path} must contain a list of split records")
    return data


class ParquetBlockWriter:
    def __init__(self, path: Path):
        import pyarrow as pa  # type: ignore
        import pyarrow.parquet as pq  # type: ignore

        self.path = path
        ensure_dir(path.parent)
        self.pa = pa
        self.pq = pq
        self.writer = None
        self.rows_written = 0

    def write(self, df: pd.DataFrame) -> None:
        table = self.pa.Table.from_pandas(df, preserve_index=False)
        if self.writer is None:
            self.writer = self.pq.ParquetWriter(self.path, table.schema)
        self.writer.write_table(table)
        self.rows_written += len(df)

    def close(self) -> None:
        if self.writer is not None:
            self.writer.close()


def validate_inputs(beliefs: pd.DataFrame, splits: list[dict[str, Any]]) -> None:
    required = ["row_id", "trajectory_id", "decision_id", "agent_action", "top_action", *Q_COLS, *CLR_COLS]
    missing = [col for col in required if col not in beliefs.columns]
    if missing:
        raise ValueError(f"belief table missing columns: {missing}")
    expected = np.arange(len(beliefs), dtype=np.int64)
    if not np.array_equal(beliefs["row_id"].to_numpy(dtype=np.int64), expected):
        raise ValueError("belief row_id must match row order")
    if not beliefs["row_id"].is_unique:
        raise ValueError("belief row_id must be unique")
    invalid = sorted(set(beliefs["agent_action"].dropna().astype(str)) - set(ACTIONS))
    if invalid:
        raise ValueError(f"invalid agent_action labels: {invalid}")

    for split in splits:
        train_groups = set(map(str, split.get("train_groups", [])))
        test_groups = set(map(str, split.get("test_groups", [])))
        if train_groups and test_groups and train_groups & test_groups:
            raise ValueError(f"group leakage in split {split.get('split_id')}")


def subset_mask(df: pd.DataFrame, subset: str) -> tuple[pd.Series | None, str | None]:
    if subset == "all_prefixes":
        return pd.Series(True, index=df.index), None
    if subset in {"strict_before_any_action_mention", "strict_before_final_action_mention"}:
        if subset not in df.columns:
            return None, f"missing {subset}"
        return df[subset].fillna(False).astype(bool), None
    if subset == "high_commitment":
        if "commitment_to_final" not in df.columns:
            return None, "missing commitment_to_final"
        return df["commitment_to_final"] >= 0.9, None
    if subset == "low_commitment":
        if "commitment_to_final" not in df.columns:
            return None, "missing commitment_to_final"
        return df["commitment_to_final"] <= 0.4, None
    return None, f"unknown subset {subset}"


def parse_subsets_arg(subsets_arg: str) -> list[str]:
    """Parse `all` or a comma-separated subset list."""
    if subsets_arg.strip().lower() == "all":
        return list(SUBSETS)
    requested = [part.strip() for part in subsets_arg.split(",") if part.strip()]
    if not requested:
        raise ValueError("at least one subset is required")
    missing = [subset for subset in requested if subset not in SUBSETS]
    if missing:
        raise ValueError(f"requested subsets are not supported: {missing}; available={SUBSETS}")
    return requested


def add_context(
    base: dict[str, Any],
    *,
    split_id: int,
    subset: str,
    layer: int,
    activation_representation: str,
    model: str,
    target: str,
    control_type: str,
    n_train: int,
    n_test: int,
) -> dict[str, Any]:
    out = {
        "split_id": int(split_id),
        "subset": subset,
        "layer": int(layer),
        "activation_representation": activation_representation,
        "model": model,
        "target": target,
        "control_type": control_type,
        "n_train": int(n_train),
        "n_test": int(n_test),
        "chance_level": 0.25,
    }
    out.update(base)
    return out


def resolve_ridge_backend(ridge_backend: str, torch_device: str) -> tuple[str, str]:
    """Resolve Ridge backend and torch device once for a run."""
    ridge_backend = str(ridge_backend).lower()
    if ridge_backend not in {"auto", "sklearn", "torch"}:
        raise ValueError("ridge backend must be one of: auto, sklearn, torch")
    if ridge_backend == "sklearn":
        return "sklearn", "cpu"
    try:
        resolved_device = resolve_torch_device_name(torch_device)
    except Exception:
        if ridge_backend == "auto":
            return "sklearn", "cpu"
        raise
    if ridge_backend == "auto" and resolved_device == "cpu":
        return "sklearn", "cpu"
    return "torch", resolved_device


def fit_predict_ridge(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    alpha: float,
    *,
    ridge_backend: str,
    torch_device: str,
    torch_batch_size: int,
) -> np.ndarray:
    if ridge_backend == "torch":
        return fit_predict_torch_ridge(
            x_train,
            y_train,
            x_test,
            alpha,
            device=torch_device,
            batch_size=torch_batch_size,
        )
    model = make_ridge(alpha)
    model.fit(x_train, y_train)
    return np.asarray(model.predict(x_test))


def fit_predict_elasticnet(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    *,
    alpha: float,
    l1_ratio: float,
    max_iter: int,
) -> np.ndarray:
    model = make_multitask_elasticnet(alpha, l1_ratio, max_iter)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(x_train, y_train)
    return np.asarray(model.predict(x_test))


def fit_predict_logreg(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    *,
    max_iter: int,
) -> np.ndarray | None:
    if not can_fit_classifier(y_train):
        return None
    model = make_logistic_regression(max_iter)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(x_train, y_train)
    return np.asarray(model.predict(x_test), dtype=object)


def prediction_block_template(
    beliefs: pd.DataFrame,
    test_indices: np.ndarray,
    *,
    split_id: int,
    subset: str,
    layer: int,
    activation_representation: str,
) -> pd.DataFrame:
    block = beliefs.iloc[test_indices][[col for col in META_COLS if col in beliefs.columns]].copy()
    block.insert(0, "activation_representation", activation_representation)
    block.insert(0, "layer", int(layer))
    block.insert(0, "subset", subset)
    block.insert(0, "split_id", int(split_id))
    block = block.rename(
        columns={
            "commitment_to_final": "true_commitment_to_final",
            "entropy_commitment": "true_entropy_commitment",
        }
    )
    return block.reset_index(drop=True)


def add_q_prediction_columns(block: pd.DataFrame, prefix: str, q_pred: np.ndarray, agent_actions: pd.Series) -> None:
    q_norm = normalize_q_predictions(q_pred)
    for i, action in enumerate(ACTIONS):
        block[f"{prefix}_q_{action}"] = q_norm[:, i]
    block[f"{prefix}_pred_top_action"] = action_argmax(q_norm)
    block[f"{prefix}_pred_commitment_to_final"] = commitment_from_q(q_norm, agent_actions.astype(str))


def ensure_prediction_schema(block: pd.DataFrame, *, run_elasticnet: bool) -> pd.DataFrame:
    """Keep streamed parquet blocks schema-compatible across skipped classifiers."""
    q_prefixes = ["ridge_clr", "ridge_q"]
    if run_elasticnet:
        q_prefixes.append("elasticnet_clr")
    for prefix in q_prefixes:
        for action in ACTIONS:
            col = f"{prefix}_q_{action}"
            if col not in block.columns:
                block[col] = np.nan
        for col in [f"{prefix}_pred_top_action", f"{prefix}_pred_commitment_to_final"]:
            if col not in block.columns:
                if col.endswith("_pred_top_action"):
                    block[col] = ""
                else:
                    block[col] = np.nan
    for action in ACTIONS:
        col = f"ridge_clr_pred_clr_{action}"
        if col not in block.columns:
            block[col] = np.nan
    for col in [
        "ridge_commitment_pred",
        "ridge_entropy_commitment_pred",
        "logreg_top_action_pred",
        "logreg_agent_action_pred",
    ]:
        if col not in block.columns:
            block[col] = "" if col.startswith("logreg") else np.nan
    return block


def run_one_fit(
    *,
    beliefs: pd.DataFrame,
    x_train: np.ndarray,
    x_test: np.ndarray,
    train_indices: np.ndarray,
    test_indices: np.ndarray,
    split_id: int,
    subset: str,
    layer: int,
    activation_representation: str,
    seed: int,
    ridge_alpha: float,
    elasticnet_alpha: float,
    elasticnet_l1_ratio: float,
    max_iter: int,
    run_elasticnet: bool,
    run_controls: bool,
    run_logistic: bool,
    ridge_backend: str,
    torch_device: str,
    torch_batch_size: int,
) -> tuple[list[dict[str, Any]], pd.DataFrame]:
    metrics: list[dict[str, Any]] = []
    n_train = len(train_indices)
    n_test = len(test_indices)
    y_clr_train = beliefs.iloc[train_indices][CLR_COLS].to_numpy(dtype=np.float32)
    y_clr_test = beliefs.iloc[test_indices][CLR_COLS].to_numpy(dtype=np.float32)
    y_q_train = beliefs.iloc[train_indices][Q_COLS].to_numpy(dtype=np.float32)
    y_q_test = beliefs.iloc[test_indices][Q_COLS].to_numpy(dtype=np.float32)
    y_commit_train = beliefs.iloc[train_indices]["commitment_to_final"].to_numpy(dtype=np.float32)
    y_commit_test = beliefs.iloc[test_indices]["commitment_to_final"].to_numpy(dtype=np.float32)
    y_entropy_train = beliefs.iloc[train_indices]["entropy_commitment"].to_numpy(dtype=np.float32)
    y_entropy_test = beliefs.iloc[test_indices]["entropy_commitment"].to_numpy(dtype=np.float32)
    y_top_train = beliefs.iloc[train_indices]["top_action"].astype(str).to_numpy()
    y_top_test = beliefs.iloc[test_indices]["top_action"].astype(str).to_numpy()
    y_agent_train = beliefs.iloc[train_indices]["agent_action"].astype(str).to_numpy()
    y_agent_test = beliefs.iloc[test_indices]["agent_action"].astype(str).to_numpy()
    agent_test_series = beliefs.iloc[test_indices]["agent_action"].astype(str)

    pred_block = prediction_block_template(
        beliefs,
        test_indices,
        split_id=split_id,
        subset=subset,
        layer=layer,
        activation_representation=activation_representation,
    )

    pred_clr = fit_predict_ridge(
        x_train,
        y_clr_train,
        x_test,
        ridge_alpha,
        ridge_backend=ridge_backend,
        torch_device=torch_device,
        torch_batch_size=torch_batch_size,
    )
    q_from_pred_clr = q_from_clr(pred_clr)
    for i, action in enumerate(ACTIONS):
        pred_block[f"ridge_clr_pred_clr_{action}"] = pred_clr[:, i]
    add_q_prediction_columns(pred_block, "ridge_clr", q_from_pred_clr, agent_test_series)
    m = regression_vector_metrics(y_clr_test, pred_clr, dim_names=ACTIONS)
    m.update(q_prediction_metrics(y_q_test, q_from_pred_clr, true_top_action=y_top_test, agent_action=y_agent_test))
    metrics.append(
        add_context(
            m,
            split_id=split_id,
            subset=subset,
            layer=layer,
            activation_representation=activation_representation,
            model="ridge",
            target="behavior_clr",
            control_type="true",
            n_train=n_train,
            n_test=n_test,
        )
    )

    pred_q_raw = fit_predict_ridge(
        x_train,
        y_q_train,
        x_test,
        ridge_alpha,
        ridge_backend=ridge_backend,
        torch_device=torch_device,
        torch_batch_size=torch_batch_size,
    )
    q_pred = normalize_q_predictions(pred_q_raw)
    add_q_prediction_columns(pred_block, "ridge_q", q_pred, agent_test_series)
    m = regression_vector_metrics(y_q_test, q_pred, dim_names=ACTIONS)
    m.update(q_prediction_metrics(y_q_test, q_pred, true_top_action=y_top_test, agent_action=y_agent_test))
    metrics.append(
        add_context(
            m,
            split_id=split_id,
            subset=subset,
            layer=layer,
            activation_representation=activation_representation,
            model="ridge",
            target="behavior_q",
            control_type="true",
            n_train=n_train,
            n_test=n_test,
        )
    )

    pred_commit = fit_predict_ridge(
        x_train,
        y_commit_train,
        x_test,
        ridge_alpha,
        ridge_backend=ridge_backend,
        torch_device=torch_device,
        torch_batch_size=torch_batch_size,
    ).reshape(-1)
    pred_block["ridge_commitment_pred"] = pred_commit
    metrics.append(
        add_context(
            scalar_regression_metrics(y_commit_test, pred_commit),
            split_id=split_id,
            subset=subset,
            layer=layer,
            activation_representation=activation_representation,
            model="ridge",
            target="commitment_to_final",
            control_type="true",
            n_train=n_train,
            n_test=n_test,
        )
    )

    pred_entropy = fit_predict_ridge(
        x_train,
        y_entropy_train,
        x_test,
        ridge_alpha,
        ridge_backend=ridge_backend,
        torch_device=torch_device,
        torch_batch_size=torch_batch_size,
    ).reshape(-1)
    pred_block["ridge_entropy_commitment_pred"] = pred_entropy
    metrics.append(
        add_context(
            scalar_regression_metrics(y_entropy_test, pred_entropy),
            split_id=split_id,
            subset=subset,
            layer=layer,
            activation_representation=activation_representation,
            model="ridge",
            target="entropy_commitment",
            control_type="true",
            n_train=n_train,
            n_test=n_test,
        )
    )

    if run_elasticnet:
        pred_sparse = fit_predict_elasticnet(
            x_train,
            y_clr_train,
            x_test,
            alpha=elasticnet_alpha,
            l1_ratio=elasticnet_l1_ratio,
            max_iter=max_iter,
        )
        q_sparse = q_from_clr(pred_sparse)
        add_q_prediction_columns(pred_block, "elasticnet_clr", q_sparse, agent_test_series)
        m = regression_vector_metrics(y_clr_test, pred_sparse, dim_names=ACTIONS)
        m.update(q_prediction_metrics(y_q_test, q_sparse, true_top_action=y_top_test, agent_action=y_agent_test))
        metrics.append(
            add_context(
                m,
                split_id=split_id,
                subset=subset,
                layer=layer,
                activation_representation=activation_representation,
                model="multitask_elasticnet",
                target="behavior_clr",
                control_type="true",
                n_train=n_train,
                n_test=n_test,
            )
        )

    if run_logistic:
        pred_top = fit_predict_logreg(x_train, y_top_train, x_test, max_iter=max_iter)
        if pred_top is not None:
            pred_block["logreg_top_action_pred"] = pred_top
            metrics.append(
                add_context(
                    classification_metrics(y_top_test, pred_top),
                    split_id=split_id,
                    subset=subset,
                    layer=layer,
                    activation_representation=activation_representation,
                    model="logistic_regression",
                    target="top_action",
                    control_type="true",
                    n_train=n_train,
                    n_test=n_test,
                )
            )

        pred_agent = fit_predict_logreg(x_train, y_agent_train, x_test, max_iter=max_iter)
        if pred_agent is not None:
            pred_block["logreg_agent_action_pred"] = pred_agent
            metrics.append(
                add_context(
                    classification_metrics(y_agent_test, pred_agent),
                    split_id=split_id,
                    subset=subset,
                    layer=layer,
                    activation_representation=activation_representation,
                    model="logistic_regression",
                    target="agent_action",
                    control_type="true",
                    n_train=n_train,
                    n_test=n_test,
                )
            )

    if run_controls:
        control_seed = seed + split_id * 100000 + layer * 1000 + len(subset)
        shuffled_clr = shuffled_copy(y_clr_train, control_seed)
        pred = fit_predict_ridge(
            x_train,
            shuffled_clr,
            x_test,
            ridge_alpha,
            ridge_backend=ridge_backend,
            torch_device=torch_device,
            torch_batch_size=torch_batch_size,
        )
        q_pred_control = q_from_clr(pred)
        m = regression_vector_metrics(y_clr_test, pred, dim_names=ACTIONS)
        m.update(q_prediction_metrics(y_q_test, q_pred_control, true_top_action=y_top_test, agent_action=y_agent_test))
        metrics.append(
            add_context(
                m,
                split_id=split_id,
                subset=subset,
                layer=layer,
                activation_representation=activation_representation,
                model="ridge",
                target="behavior_clr",
                control_type="shuffled_behavior_vector",
                n_train=n_train,
                n_test=n_test,
            )
        )

        shuffled_q = shuffled_copy(y_q_train, control_seed + 1)
        pred = normalize_q_predictions(
            fit_predict_ridge(
                x_train,
                shuffled_q,
                x_test,
                ridge_alpha,
                ridge_backend=ridge_backend,
                torch_device=torch_device,
                torch_batch_size=torch_batch_size,
            )
        )
        m = regression_vector_metrics(y_q_test, pred, dim_names=ACTIONS)
        m.update(q_prediction_metrics(y_q_test, pred, true_top_action=y_top_test, agent_action=y_agent_test))
        metrics.append(
            add_context(
                m,
                split_id=split_id,
                subset=subset,
                layer=layer,
                activation_representation=activation_representation,
                model="ridge",
                target="behavior_q",
                control_type="shuffled_behavior_vector",
                n_train=n_train,
                n_test=n_test,
            )
        )

        if run_logistic and can_fit_classifier(y_top_train):
            pred = fit_predict_logreg(
                x_train,
                shuffled_copy(y_top_train, control_seed + 2),
                x_test,
                max_iter=max_iter,
            )
            if pred is not None:
                metrics.append(
                    add_context(
                        classification_metrics(y_top_test, pred),
                        split_id=split_id,
                        subset=subset,
                        layer=layer,
                        activation_representation=activation_representation,
                        model="logistic_regression",
                        target="top_action",
                        control_type="shuffled_labels",
                        n_train=n_train,
                        n_test=n_test,
                    )
                )
        if run_logistic and can_fit_classifier(y_agent_train):
            pred = fit_predict_logreg(
                x_train,
                shuffled_copy(y_agent_train, control_seed + 3),
                x_test,
                max_iter=max_iter,
            )
            if pred is not None:
                metrics.append(
                    add_context(
                        classification_metrics(y_agent_test, pred),
                        split_id=split_id,
                        subset=subset,
                        layer=layer,
                        activation_representation=activation_representation,
                        model="logistic_regression",
                        target="agent_action",
                        control_type="shuffled_labels",
                        n_train=n_train,
                        n_test=n_test,
                    )
                )

    pred_block = ensure_prediction_schema(pred_block, run_elasticnet=run_elasticnet)
    return metrics, pred_block


def best_config(metrics: pd.DataFrame) -> dict[str, Any]:
    def best_row(filter_mask: pd.Series, metric_col: str) -> dict[str, Any] | None:
        part = metrics.loc[filter_mask].copy()
        if metric_col not in part.columns:
            return None
        part = part.loc[np.isfinite(pd.to_numeric(part[metric_col], errors="coerce"))]
        if part.empty:
            return None
        grouped = (
            part.groupby(["subset", "layer", "activation_representation", "model", "target"], dropna=False)[metric_col]
            .mean()
            .reset_index()
            .sort_values(metric_col, ascending=False)
        )
        if grouped.empty:
            return None
        row = grouped.iloc[0].to_dict()
        row[metric_col] = float(row[metric_col])
        return row

    true = metrics["control_type"] == "true"
    return {
        "chance_level": 0.25,
        "best_behavior_clr_probe": best_row(true & (metrics["target"] == "behavior_clr"), "mean_r2"),
        "best_behavior_q_probe": best_row(true & (metrics["target"] == "behavior_q"), "mean_r2"),
        "best_commitment_probe": best_row(true & (metrics["target"] == "commitment_to_final"), "r2"),
        "best_top_action_probe": best_row(true & (metrics["target"] == "top_action"), "balanced_accuracy"),
        "selection_note": "Best configs are selected by mean cross-split metric and may still have negative R2.",
    }


def make_figures(metrics: pd.DataFrame, predictions_path: Path, figure_dir: Path, best: dict[str, Any]) -> list[dict[str, str]]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ensure_dir(figure_dir)
    figures: list[dict[str, str]] = []
    true = metrics.loc[metrics["control_type"] == "true"].copy()

    r2_part = true.loc[(true["target"] == "behavior_clr") & (true["model"] == "ridge")]
    if not r2_part.empty:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        line_metric_by_layer(
            r2_part,
            metric_col="mean_r2",
            title="Behavior CLR probe mean R2 by layer",
            ylabel="mean R2",
            ax=ax,
        )
        out = figure_dir / "mean_r2_by_layer_and_representation.png"
        savefig(out, fig=fig)
        figures.append({"path": rel(out), "description": "Mean R2 by layer and representation"})

    top_part = true.loc[(true["target"] == "behavior_clr") & (true["model"] == "ridge")]
    if not top_part.empty and "top_action_accuracy_vs_true_top_action" in top_part.columns:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        line_metric_by_layer(
            top_part,
            metric_col="top_action_accuracy_vs_true_top_action",
            title="Predicted top-action accuracy by layer",
            ylabel="accuracy vs true top_action",
            ax=ax,
        )
        ax.axhline(0.25, color="#444444", ls="--", lw=0.8, label="chance 0.25")
        out = figure_dir / "predicted_top_action_accuracy_by_layer.png"
        savefig(out, fig=fig)
        figures.append({"path": rel(out), "description": "Predicted top-action accuracy by layer"})

    strict_part = r2_part.loc[r2_part["subset"].astype(str).str.contains("strict_before")]
    if not strict_part.empty:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        line_metric_by_layer(
            strict_part,
            metric_col="mean_r2",
            title="Strict pre-action behavior CLR probe mean R2 by layer",
            ylabel="mean R2",
            ax=ax,
        )
        out = figure_dir / "strict_pre_action_probe_metrics_by_layer.png"
        savefig(out, fig=fig)
        figures.append({"path": rel(out), "description": "Strict pre-action probe metrics by layer"})

    comparison = metrics.loc[
        (metrics["target"] == "behavior_clr")
        & (metrics["model"] == "ridge")
        & (metrics["control_type"].isin(["true", "shuffled_behavior_vector"]))
    ]
    if not comparison.empty:
        grouped = comparison.groupby("control_type")["mean_r2"].mean().sort_index()
        fig, ax = plt.subplots(figsize=(5.5, 4))
        ax.bar(grouped.index.astype(str), grouped.to_numpy(dtype=float), color=["#4C78A8", "#F58518"][: len(grouped)])
        ax.axhline(0, color="#888888", lw=0.8)
        ax.set_ylabel("mean R2")
        ax.set_title("True vs shuffled behavior-vector baseline")
        out = figure_dir / "true_vs_shuffled_metrics.png"
        savefig(out, fig=fig)
        figures.append({"path": rel(out), "description": "True vs shuffled metrics"})

    if predictions_path.exists():
        preds = pd.read_parquet(predictions_path)
        if {"true_commitment_to_final", "ridge_clr_pred_commitment_to_final"}.issubset(preds.columns):
            plot_df = preds.dropna(subset=["true_commitment_to_final", "ridge_clr_pred_commitment_to_final"])
            if len(plot_df) > 5000:
                plot_df = plot_df.sample(5000, random_state=DEFAULT_SEED)
            if not plot_df.empty:
                fig, ax = plt.subplots(figsize=(5.5, 5))
                ax.scatter(
                    plot_df["true_commitment_to_final"],
                    plot_df["ridge_clr_pred_commitment_to_final"],
                    s=8,
                    alpha=0.35,
                    edgecolors="none",
                )
                ax.set_xlabel("true commitment_to_final")
                ax.set_ylabel("predicted commitment_to_final")
                ax.set_title("Predicted vs true commitment_to_final")
                out = figure_dir / "predicted_vs_true_commitment_to_final.png"
                savefig(out, fig=fig)
                figures.append({"path": rel(out), "description": "Predicted vs true commitment_to_final"})

    return figures


def write_report(
    path: Path,
    *,
    io_map_path: Path,
    beliefs_path: Path,
    splits_path: Path,
    feature_dir: Path,
    metrics_path: Path,
    predictions_path: Path,
    best_config_path: Path,
    layers: list[int],
    representations: list[str],
    subsets: list[str],
    metrics: pd.DataFrame,
    skipped: list[dict[str, Any]],
    best: dict[str, Any],
    figures: list[dict[str, str]],
    run_elasticnet: bool,
    run_controls: bool,
    run_logistic: bool,
    l2_normalize: bool,
    ridge_backend: str,
    torch_device: str,
) -> None:
    true = metrics.loc[metrics["control_type"] == "true"] if not metrics.empty else pd.DataFrame()
    shuffled = metrics.loc[metrics["control_type"] != "true"] if not metrics.empty else pd.DataFrame()
    lines = [
        "# Sparse Activation Probes",
        "",
        "Step 7 trains grouped cross-validated linear probes from high-dimensional activation features to behavior belief targets.",
        "",
        "## Inputs and Outputs",
        "",
        f"- IO map: `{rel(io_map_path)}`",
        f"- Beliefs: `{rel(beliefs_path)}`",
        f"- Splits: `{rel(splits_path)}`",
        f"- Activation feature directory: `{rel(feature_dir)}`",
        f"- Metrics CSV: `{rel(metrics_path)}`",
        f"- Predictions parquet: `{rel(predictions_path)}`",
        f"- Best config JSON: `{rel(best_config_path)}`",
        "",
        "## Method",
        "",
        "- Probe inputs are high-dimensional activation feature arrays, not UMAP or PCA coordinates.",
        "- Scalers are fit on training rows only for each split/subset/layer/representation.",
        "- Splits are loaded from `group_splits.json`; prefixes from held-out trajectories stay held out.",
        "- Main target is full behavior belief (`behavior_clr` and `behavior_q`), not just final action labels.",
        f"- Layers: `{layers}`",
        f"- Activation representations: `{representations}`",
        f"- Subsets: `{subsets}`",
        f"- ElasticNet sparse probe enabled: `{run_elasticnet}`",
        f"- Logistic classification probes enabled: `{run_logistic}`",
        f"- Shuffled controls enabled: `{run_controls}`",
        f"- L2 normalize after scaling: `{l2_normalize}`",
        f"- Ridge backend: `{ridge_backend}`",
        f"- Torch device for Ridge backend: `{torch_device}`",
        "- Chance level for four-way action classification: `0.25`.",
        "- Negative R2 values are reported honestly.",
        "",
        "## Counts",
        "",
        f"- Metric rows: `{len(metrics)}`",
        f"- True metric rows: `{len(true)}`",
        f"- Control metric rows: `{len(shuffled)}`",
        f"- Skipped fit blocks: `{len(skipped)}`",
    ]

    if not true.empty:
        ridge_clr = true.loc[(true["target"] == "behavior_clr") & (true["model"] == "ridge")]
        if not ridge_clr.empty:
            lines.extend(
                [
                    "",
                    "## Behavior CLR Ridge Summary",
                    "",
                    f"- Mean of fold/subset mean R2 values: `{float(ridge_clr['mean_r2'].mean())}`",
                    f"- Best observed mean R2 row: `{float(ridge_clr['mean_r2'].max())}`",
                    f"- Worst observed mean R2 row: `{float(ridge_clr['mean_r2'].min())}`",
                ]
            )

    lines.extend(["", "## Best Config", "", "```json", json.dumps(best, indent=2, sort_keys=True), "```"])

    lines.extend(["", "## Figures", ""])
    if figures:
        for fig in figures:
            lines.append(f"- `{fig['path']}`: {fig['description']}")
    else:
        lines.append("- No figures generated.")

    lines.extend(["", "## Skipped Blocks", ""])
    if skipped:
        for item in skipped[:100]:
            lines.append(f"- {item}")
        if len(skipped) > 100:
            lines.append(f"- ... {len(skipped) - 100} additional skipped blocks omitted.")
    else:
        lines.append("- None.")

    lines.extend(
        [
            "",
            "## Interpretation Caveat",
            "",
            "Do not claim activation-behavior alignment from this step if probe performance is weak or similar to shuffled controls. Sparse probes are evidence only to the extent that held-out grouped performance exceeds controls.",
        ]
    )
    save_markdown(path, "\n".join(lines))


def run(
    *,
    io_map_path: Path,
    beliefs_path: Path,
    splits_path: Path,
    metrics_output: Path,
    predictions_output: Path,
    best_config_output: Path,
    report_output: Path,
    figure_dir: Path,
    layers_arg: str,
    representations_arg: str,
    subsets_arg: str,
    ridge_alpha: float,
    elasticnet_alpha: float,
    elasticnet_l1_ratio: float,
    max_iter: int,
    min_train_rows: int,
    min_test_rows: int,
    seed: int,
    l2_normalize: bool,
    run_elasticnet: bool,
    run_controls: bool,
    run_logistic: bool,
    ridge_backend: str,
    torch_device: str,
    torch_batch_size: int,
) -> pd.DataFrame:
    io_map = read_json(io_map_path)
    if not isinstance(io_map, dict):
        raise ValueError(f"{io_map_path} must contain a JSON object")
    feature_dir = resolve_repo_path(str(io_map["activation_feature_dir"]))
    layers = parse_layers_arg(layers_arg, io_map.get("available_layers", []))
    representations = parse_representations_arg(representations_arg, io_map.get("available_representations", []))
    subsets = parse_subsets_arg(subsets_arg)
    beliefs = read_parquet_checked(beliefs_path)
    splits = load_splits(splits_path)
    validate_inputs(beliefs, splits)
    resolved_ridge_backend, resolved_torch_device = resolve_ridge_backend(ridge_backend, torch_device)

    metrics_rows: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    writer = ParquetBlockWriter(predictions_output)
    try:
        for layer in layers:
            for representation in representations:
                feature_path = activation_feature_path(feature_dir, layer, representation)
                features = load_activation_memmap(feature_path, expected_rows=len(beliefs))
                for split in splits:
                    split_id = int(split["split_id"])
                    train_base = np.asarray(split["train_indices"], dtype=np.int64)
                    test_base = np.asarray(split["test_indices"], dtype=np.int64)
                    for subset in subsets:
                        mask, reason = subset_mask(beliefs, subset)
                        if reason is not None or mask is None:
                            skipped.append(
                                {"split_id": split_id, "subset": subset, "layer": layer, "representation": representation, "reason": reason}
                            )
                            continue
                        mask_arr = mask.to_numpy(dtype=bool)
                        train_idx = train_base[mask_arr[train_base]]
                        test_idx = test_base[mask_arr[test_base]]
                        if len(train_idx) < min_train_rows or len(test_idx) < min_test_rows:
                            skipped.append(
                                {
                                    "split_id": split_id,
                                    "subset": subset,
                                    "layer": layer,
                                    "representation": representation,
                                    "reason": f"too few rows train={len(train_idx)} test={len(test_idx)}",
                                }
                            )
                            continue
                        standardizer = fit_train_standardizer(features, train_idx)
                        x_train = transform_indices_with_standardizer(
                            features,
                            train_idx,
                            standardizer,
                            l2_normalize=l2_normalize,
                        )
                        x_test = transform_indices_with_standardizer(
                            features,
                            test_idx,
                            standardizer,
                            l2_normalize=l2_normalize,
                        )
                        block_metrics, pred_block = run_one_fit(
                            beliefs=beliefs,
                            x_train=x_train,
                            x_test=x_test,
                            train_indices=train_idx,
                            test_indices=test_idx,
                            split_id=split_id,
                            subset=subset,
                            layer=layer,
                            activation_representation=representation,
                            seed=seed,
                            ridge_alpha=ridge_alpha,
                            elasticnet_alpha=elasticnet_alpha,
                            elasticnet_l1_ratio=elasticnet_l1_ratio,
                            max_iter=max_iter,
                            run_elasticnet=run_elasticnet,
                            run_controls=run_controls,
                            run_logistic=run_logistic,
                            ridge_backend=resolved_ridge_backend,
                            torch_device=resolved_torch_device,
                            torch_batch_size=torch_batch_size,
                        )
                        metrics_rows.extend(block_metrics)
                        writer.write(pred_block)
    finally:
        writer.close()

    metrics = pd.DataFrame(metrics_rows)
    if metrics.empty:
        raise SystemExit("No probe fits were completed. Check subsets, splits, and feature files.")
    metrics["ridge_backend"] = resolved_ridge_backend
    metrics["torch_device"] = resolved_torch_device
    ensure_dir(metrics_output.parent)
    metrics.to_csv(metrics_output, index=False)
    best = best_config(metrics)
    write_json(best_config_output, best)
    figures = make_figures(metrics, predictions_output, figure_dir, best)
    write_report(
        report_output,
        io_map_path=io_map_path,
        beliefs_path=beliefs_path,
        splits_path=splits_path,
        feature_dir=feature_dir,
        metrics_path=metrics_output,
        predictions_path=predictions_output,
        best_config_path=best_config_output,
        layers=layers,
        representations=representations,
        subsets=subsets,
        metrics=metrics,
        skipped=skipped,
        best=best,
        figures=figures,
        run_elasticnet=run_elasticnet,
        run_controls=run_controls,
        run_logistic=run_logistic,
        l2_normalize=l2_normalize,
        ridge_backend=resolved_ridge_backend,
        torch_device=resolved_torch_device,
    )
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--io_map", default=DEFAULT_OUTPUT_DIR / "io_map.json")
    parser.add_argument("--beliefs", default=DEFAULT_OUTPUT_DIR / "action_belief_vectors.parquet")
    parser.add_argument("--splits", default=DEFAULT_OUTPUT_DIR / "splits/group_splits.json")
    parser.add_argument("--representations", default="raw,delta_pre,unit_delta_pre,velocity")
    parser.add_argument("--layers", default="all")
    parser.add_argument("--subsets", default="all", help=f"`all` or comma-separated subset list from {SUBSETS}")
    parser.add_argument("--metrics_output", default=DEFAULT_OUTPUT_DIR / "probes/probe_metrics.csv")
    parser.add_argument("--predictions_output", default=DEFAULT_OUTPUT_DIR / "probes/probe_predictions.parquet")
    parser.add_argument("--best_config_output", default=DEFAULT_OUTPUT_DIR / "probes/best_probe_config.json")
    parser.add_argument("--report", default=DEFAULT_OUTPUT_DIR / "probes/07_probe_report.md")
    parser.add_argument("--figure_dir", default=DEFAULT_FIGURE_DIR / "probes")
    parser.add_argument("--ridge_alpha", type=float, default=1.0)
    parser.add_argument("--elasticnet_alpha", type=float, default=0.001)
    parser.add_argument("--elasticnet_l1_ratio", type=float, default=0.5)
    parser.add_argument("--max_iter", type=int, default=1000)
    parser.add_argument("--min_train_rows", type=int, default=20)
    parser.add_argument("--min_test_rows", type=int, default=5)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--l2_normalize", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--run_elasticnet", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--run_controls", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--run_logistic", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--ridge_backend",
        choices=["auto", "sklearn", "torch"],
        default="auto",
        help="Use torch for Ridge on CUDA/MPS when available; sklearn otherwise.",
    )
    parser.add_argument(
        "--torch_device",
        choices=["auto", "cpu", "cuda", "mps"],
        default="auto",
        help="Torch device used when --ridge_backend resolves to torch.",
    )
    parser.add_argument(
        "--torch_batch_size",
        type=int,
        default=65536,
        help="Prediction batch size for torch Ridge.",
    )
    args = parser.parse_args()

    metrics = run(
        io_map_path=resolve_repo_path(args.io_map),
        beliefs_path=resolve_repo_path(args.beliefs),
        splits_path=resolve_repo_path(args.splits),
        metrics_output=resolve_repo_path(args.metrics_output),
        predictions_output=resolve_repo_path(args.predictions_output),
        best_config_output=resolve_repo_path(args.best_config_output),
        report_output=resolve_repo_path(args.report),
        figure_dir=resolve_repo_path(args.figure_dir),
        layers_arg=args.layers,
        representations_arg=args.representations,
        subsets_arg=args.subsets,
        ridge_alpha=args.ridge_alpha,
        elasticnet_alpha=args.elasticnet_alpha,
        elasticnet_l1_ratio=args.elasticnet_l1_ratio,
        max_iter=args.max_iter,
        min_train_rows=args.min_train_rows,
        min_test_rows=args.min_test_rows,
        seed=args.seed,
        l2_normalize=args.l2_normalize,
        run_elasticnet=args.run_elasticnet,
        run_controls=args.run_controls,
        run_logistic=args.run_logistic,
        ridge_backend=args.ridge_backend,
        torch_device=args.torch_device,
        torch_batch_size=args.torch_batch_size,
    )
    print(f"Wrote {rel(args.metrics_output)} with {len(metrics)} metric rows")
    print(f"Wrote {rel(args.predictions_output)}")
    print(f"Wrote {rel(args.best_config_output)}")
    print(f"Wrote {rel(args.report)}")


if __name__ == "__main__":
    main()
