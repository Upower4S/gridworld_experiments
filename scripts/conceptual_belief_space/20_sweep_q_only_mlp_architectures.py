#!/usr/bin/env python
"""Sweep q-only MLP architectures for action-belief prediction.

This script answers a narrower question than the supervised branch-manifold
experiment: how well can activation features predict the four-action belief
distribution q when the only training loss is soft-label cross entropy.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.conceptual_belief_space.config import ACTIONS, EPS
from src.conceptual_belief_space.io import ensure_dir, save_markdown, write_json
from src.conceptual_belief_space.mlp_branch_utils import (
    Q_COLS,
    SplitDef,
    action_indices_to_labels,
    action_labels_to_indices,
    class_weights,
    fit_standardizer,
    json_ready,
    load_or_create_splits,
    make_behavior_table,
    make_train_val_split,
    normalize_q,
    q_metrics,
    q_top_action_coverage,
    softmax_np,
    subset_mask,
    top_actions_from_q,
    transform_features,
)


@dataclass(frozen=True)
class ArchSpec:
    name: str
    hidden_dims: tuple[int, ...]
    bottleneck_dim: int | None = None

    @property
    def uses_bottleneck(self) -> bool:
        return self.bottleneck_dim is not None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--beliefs", default="outputs/conceptual_belief_space/action_belief_vectors.parquet")
    parser.add_argument("--splits", default="outputs/conceptual_belief_space/splits/group_splits.json")
    parser.add_argument("--feature_dir", default="outputs/activation/features")
    parser.add_argument("--output_dir", default="outputs/conceptual_belief_space/q_only_mlp_sweep")
    parser.add_argument("--figure_dir", default="figures/conceptual_belief_space/mlp_branch_probe/q_only_sweep")
    parser.add_argument("--layer", type=int, default=10)
    parser.add_argument("--representation", default="delta_pre")
    parser.add_argument("--subset", default="strict_before_final_action_mention")
    parser.add_argument("--group_col", default="trajectory_id")
    parser.add_argument("--architectures", nargs="+", default=[
        "linear",
        "mlp_128",
        "mlp_256",
        "mlp_512",
        "mlp_256_128",
        "mlp_512_256",
        "mlp_512_256_128",
        "bottleneck_2",
        "bottleneck_4",
        "bottleneck_8",
        "bottleneck_16",
        "bottleneck_32",
        "bottleneck_64",
    ])
    parser.add_argument("--bottleneck_hidden_dims", default="256,128")
    parser.add_argument("--loss_weighting", choices=["plain", "balanced", "both"], default="both")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--max_epochs", type=int, default=120)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--standardize", action="store_true", default=True)
    parser.add_argument("--no-standardize", action="store_false", dest="standardize")
    parser.add_argument("--l2_normalize", action="store_true")
    parser.add_argument("--collapse_penalty", type=float, default=0.25)
    parser.add_argument("--limit_splits", type=int, default=None)
    parser.add_argument("--save_best_predictions", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def parse_hidden_dims(text: str) -> tuple[int, ...]:
    if not text.strip():
        return ()
    return tuple(int(part) for part in text.replace("-", "_").split(",") if part.strip())


def parse_architecture(name: str, bottleneck_hidden_dims: tuple[int, ...]) -> ArchSpec:
    if name == "linear":
        return ArchSpec(name=name, hidden_dims=())
    if name.startswith("mlp_"):
        dims = tuple(int(part) for part in name[len("mlp_") :].split("_") if part)
        if not dims:
            raise ValueError(f"invalid MLP architecture: {name}")
        return ArchSpec(name=name, hidden_dims=dims)
    if name.startswith("bottleneck_"):
        dim = int(name[len("bottleneck_") :])
        return ArchSpec(name=name, hidden_dims=bottleneck_hidden_dims, bottleneck_dim=dim)
    raise ValueError(f"unknown architecture: {name}")


def choose_device(device_arg: str):
    import torch

    if device_arg != "auto":
        return torch.device(device_arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def target_entropy_nats(q: np.ndarray) -> np.ndarray:
    qn = normalize_q(q)
    return -np.sum(qn * np.log(np.clip(qn, EPS, None)), axis=1)


def make_targets(df: pd.DataFrame, indices: np.ndarray) -> dict[str, np.ndarray]:
    rows = df.iloc[np.asarray(indices, dtype=np.int64)]
    q = normalize_q(rows[Q_COLS].to_numpy(dtype=np.float32))
    return {
        "q": q,
        "top": action_labels_to_indices(rows["top_action"]),
        "agent": action_labels_to_indices(rows["agent_action"]),
    }


def build_model(input_dim: int, spec: ArchSpec, dropout: float):
    import torch.nn as nn

    class QOnlyMLP(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            layers: list[nn.Module] = [nn.LayerNorm(input_dim)]
            prev = input_dim
            for hidden_dim in spec.hidden_dims:
                layers.extend([nn.Linear(prev, hidden_dim), nn.GELU(), nn.Dropout(dropout)])
                prev = hidden_dim
            self.encoder = nn.Sequential(*layers)
            if spec.uses_bottleneck:
                self.bottleneck = nn.Linear(prev, int(spec.bottleneck_dim))
                self.q_head = nn.Linear(int(spec.bottleneck_dim), len(ACTIONS))
            else:
                self.bottleneck = None
                self.q_head = nn.Linear(prev, len(ACTIONS))

        def forward(self, x):
            hidden = self.encoder(x)
            if self.bottleneck is None:
                z = hidden
                source = hidden
            else:
                z = self.bottleneck(hidden)
                source = z
            return {"q_logits": self.q_head(source), "z": z}

    return QOnlyMLP()


def predict_q(model, x: np.ndarray, batch_size: int, device) -> tuple[np.ndarray, np.ndarray]:
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, TensorDataset

    loader = DataLoader(TensorDataset(torch.tensor(x, dtype=torch.float32)), batch_size=batch_size, shuffle=False)
    q_parts: list[np.ndarray] = []
    z_parts: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for (xb,) in loader:
            out = model(xb.to(device))
            q_parts.append(F.softmax(out["q_logits"], dim=1).detach().cpu().numpy())
            z = out["z"].detach().cpu().numpy()
            if z.ndim == 1:
                z = z[:, None]
            z_parts.append(z[:, : min(8, z.shape[1])])
    return normalize_q(np.concatenate(q_parts, axis=0)), np.concatenate(z_parts, axis=0).astype(np.float32)


def validation_score(q_true: np.ndarray, q_pred: np.ndarray, top: np.ndarray, collapse_penalty: float) -> tuple[float, dict[str, Any]]:
    top_labels = action_indices_to_labels(top)
    q_m = q_metrics(q_true, q_pred, top_action=top_labels, agent_action=top_labels)
    coverage = q_top_action_coverage(q_pred, true_top_action=top_labels, min_count=1)
    penalty = collapse_penalty * len(coverage["missing_actions"])
    score = q_m["q_top_action_balanced_accuracy"] - 0.1 * q_m["q_kl"] - penalty
    return float(score), {"validation_q_kl": q_m["q_kl"], "validation_q_top_action_balanced_accuracy": q_m["q_top_action_balanced_accuracy"], "validation_q_collapse_penalty": penalty, "validation_q_missing_actions_json": json.dumps(coverage["missing_actions"])}


def train_one(
    x_train: np.ndarray,
    y_train: dict[str, np.ndarray],
    x_val: np.ndarray,
    y_val: dict[str, np.ndarray],
    spec: ArchSpec,
    args: argparse.Namespace,
    device,
    *,
    seed: int,
    balanced_loss: bool,
) -> tuple[Any, dict[str, Any]]:
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, TensorDataset

    torch.manual_seed(seed)
    np.random.seed(seed)
    model = build_model(x_train.shape[1], spec, args.dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    top_weights = torch.tensor(class_weights(y_train["top"]), dtype=torch.float32, device=device)
    dataset = TensorDataset(torch.tensor(x_train, dtype=torch.float32), torch.tensor(y_train["q"], dtype=torch.float32), torch.tensor(y_train["top"], dtype=torch.long))
    generator = torch.Generator()
    generator.manual_seed(seed)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, generator=generator)

    best_state = None
    best_score = -float("inf")
    best_epoch = -1
    best_info: dict[str, Any] = {}
    bad_epochs = 0
    for epoch in range(int(args.max_epochs)):
        model.train()
        for xb, q, top in loader:
            xb = xb.to(device)
            q = q.to(device)
            top = top.to(device)
            out = model(xb)
            loss_per_row = -(q * F.log_softmax(out["q_logits"], dim=1)).sum(dim=1)
            if balanced_loss:
                weights = top_weights[top]
                loss = (loss_per_row * weights).sum() / torch.clamp(weights.sum(), min=1e-8)
            else:
                loss = loss_per_row.mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if args.grad_clip and args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
        q_val, _ = predict_q(model, x_val, args.batch_size, device)
        score, val_info = validation_score(y_val["q"], q_val, y_val["top"], args.collapse_penalty)
        if score > best_score:
            best_score = score
            best_epoch = epoch
            best_info = val_info
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, {"best_epoch": best_epoch, "validation_score": best_score, **best_info}


def evaluate_predictions(
    *,
    df: pd.DataFrame,
    indices: np.ndarray,
    q_pred: np.ndarray,
    split: SplitDef,
    spec: ArchSpec,
    loss_weighting: str,
    train_n: int,
    val_n: int,
    train_info: dict[str, Any],
) -> dict[str, Any]:
    rows = df.iloc[np.asarray(indices, dtype=np.int64)].copy()
    q_true = normalize_q(rows[Q_COLS].to_numpy(dtype=np.float32))
    q_m = q_metrics(q_true, q_pred, top_action=rows["top_action"], agent_action=rows["agent_action"])
    coverage = q_top_action_coverage(q_pred, true_top_action=rows["top_action"], min_count=1)
    entropy = target_entropy_nats(q_true)
    uniform_kl = math.log(len(ACTIONS)) - float(np.mean(entropy))
    pred_top = top_actions_from_q(q_pred)
    oracle_agent_acc = float(np.mean(rows["top_action"].astype(str).to_numpy() == rows["agent_action"].astype(str).to_numpy()))
    row: dict[str, Any] = {
        "run_id": f"qonly_{spec.name}_{loss_weighting}_split{split.split_id}_{int(time.time())}",
        "split_id": int(split.split_id),
        "layer": -1,
        "representation": "",
        "subset": "",
        "architecture": spec.name,
        "hidden_dims_json": json.dumps(list(spec.hidden_dims)),
        "uses_bottleneck": bool(spec.uses_bottleneck),
        "bottleneck_dim": int(spec.bottleneck_dim or 0),
        "loss_weighting": loss_weighting,
        "n_train": int(train_n),
        "n_val": int(val_n),
        "n_test": int(len(indices)),
        "n_train_trajectories": int(len(set(df.iloc[split.train_indices][split.group_col].astype(str)))),
        "n_test_trajectories": int(len(set(rows[split.group_col].astype(str)))),
        "best_epoch": int(train_info.get("best_epoch", -1)),
        "validation_score": float(train_info.get("validation_score", float("nan"))),
        "validation_q_kl": float(train_info.get("validation_q_kl", float("nan"))),
        "validation_q_top_action_balanced_accuracy": float(train_info.get("validation_q_top_action_balanced_accuracy", float("nan"))),
        "validation_q_missing_actions_json": train_info.get("validation_q_missing_actions_json", "[]"),
        "test_q_kl": q_m["q_kl"],
        "test_q_mse": q_m["q_mse"],
        "test_q_top_action_accuracy": q_m["q_top_action_accuracy"],
        "test_q_top_action_balanced_accuracy": q_m["q_top_action_balanced_accuracy"],
        "test_q_agent_action_accuracy": q_m["q_agent_action_accuracy"],
        "test_q_agent_action_balanced_accuracy": q_m["q_agent_action_balanced_accuracy"],
        "test_commitment_from_q_spearman": q_m["commitment_to_final_spearman_from_q"],
        "test_target_entropy_nats": float(np.mean(entropy)),
        "test_oracle_soft_ce_nats": float(np.mean(entropy)),
        "test_model_soft_ce_nats": float(np.mean(entropy) + q_m["q_kl"]),
        "test_uniform_soft_ce_nats": float(math.log(len(ACTIONS))),
        "test_uniform_q_kl": uniform_kl,
        "test_oracle_excess_ce_kl": 0.0,
        "test_oracle_q_top_action_accuracy": 1.0,
        "test_oracle_q_agent_action_accuracy": oracle_agent_acc,
        "test_q_pred_top_action_collapse": coverage["has_action_collapse"],
        "test_q_pred_top_num_actions": coverage["num_predicted_actions"],
        "test_q_pred_top_missing_actions_json": json.dumps(coverage["missing_actions"]),
    }
    for action in ACTIONS:
        idx = ACTIONS.index(action)
        row[f"test_true_top_count_{action}"] = coverage["true_counts"][action]
        row[f"test_q_pred_top_count_{action}"] = coverage["pred_counts"][action]
        row[f"test_mean_q_pred_{action}"] = float(np.mean(q_pred[:, idx]))
        row[f"test_max_q_pred_{action}"] = float(np.max(q_pred[:, idx]))
    return row


def mean_q_baseline(df: pd.DataFrame, train: np.ndarray, test: np.ndarray, split: SplitDef, loss_weighting: str) -> dict[str, Any]:
    q_train = normalize_q(df.iloc[train][Q_COLS].to_numpy(dtype=np.float32))
    q_prior = normalize_q(q_train.mean(axis=0, keepdims=True))
    q_pred = np.repeat(q_prior, len(test), axis=0)
    spec = ArchSpec(name="train_mean_q_baseline", hidden_dims=())
    return evaluate_predictions(df=df, indices=test, q_pred=q_pred, split=split, spec=spec, loss_weighting=loss_weighting, train_n=len(train), val_n=0, train_info={"best_epoch": -1})


def save_best_predictions(
    path: Path,
    df: pd.DataFrame,
    test_indices: np.ndarray,
    q_pred: np.ndarray,
    *,
    row: dict[str, Any],
) -> None:
    out = df.iloc[np.asarray(test_indices, dtype=np.int64)][["row_id", "prefix_id", "decision_id", "trajectory_id", "step_id", "prefix_token_index", "progress", "agent_action", "top_action", *Q_COLS]].copy()
    out.insert(0, "run_id", row["run_id"])
    out.insert(1, "architecture", row["architecture"])
    out.insert(2, "loss_weighting", row["loss_weighting"])
    qn = normalize_q(q_pred)
    for i, action in enumerate(ACTIONS):
        out[f"q_pred_{action}"] = qn[:, i]
    out["q_pred_top_action"] = top_actions_from_q(qn)
    ensure_dir(path.parent)
    out.to_parquet(path, index=False)


def plot_summary(metrics: pd.DataFrame, figure_dir: Path) -> None:
    import matplotlib.pyplot as plt

    ensure_dir(figure_dir)
    main = metrics[~metrics["architecture"].astype(str).str.contains("baseline", na=False)].copy()
    if main.empty:
        return
    summary = (
        main.groupby(["architecture", "loss_weighting"], as_index=False)
        .agg(
            mean_bal_acc=("test_q_top_action_balanced_accuracy", "mean"),
            mean_acc=("test_q_top_action_accuracy", "mean"),
            mean_kl=("test_q_kl", "mean"),
            mean_ce=("test_model_soft_ce_nats", "mean"),
            collapse_rows=("test_q_pred_top_action_collapse", "sum"),
        )
        .sort_values("mean_bal_acc", ascending=False)
    )
    labels = [f"{r.architecture}\n{r.loss_weighting}" for r in summary.itertuples()]
    x = np.arange(len(summary))
    fig, ax = plt.subplots(figsize=(max(9, len(summary) * 0.45), 5))
    ax.bar(x, summary["mean_bal_acc"].to_numpy(), color="C0")
    ax.axhline(0.25, color="black", linewidth=1, linestyle="--", label="chance")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=70, ha="right", fontsize=8)
    ax.set_ylabel("test balanced accuracy")
    ax.set_title("q-only MLP action-belief prediction by architecture")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figure_dir / "q_only_architecture_balanced_accuracy.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(max(9, len(summary) * 0.45), 5))
    ax.bar(x, summary["mean_kl"].to_numpy(), color="C1")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=70, ha="right", fontsize=8)
    ax.set_ylabel("test KL(q || p), lower is better")
    ax.set_title("q-only MLP excess soft-label CE by architecture")
    fig.tight_layout()
    fig.savefig(figure_dir / "q_only_architecture_kl.png", dpi=200)
    plt.close(fig)


def write_report(metrics: pd.DataFrame, output_dir: Path, figure_dir: Path, args: argparse.Namespace) -> None:
    ok = metrics.copy()
    main = ok[~ok["architecture"].astype(str).str.contains("baseline", na=False)].copy()
    summary = (
        main.groupby(["architecture", "loss_weighting"], as_index=False)
        .agg(
            splits=("split_id", "nunique"),
            mean_q_top_acc=("test_q_top_action_accuracy", "mean"),
            mean_q_top_bal_acc=("test_q_top_action_balanced_accuracy", "mean"),
            mean_q_agent_acc=("test_q_agent_action_accuracy", "mean"),
            mean_q_kl=("test_q_kl", "mean"),
            mean_model_ce=("test_model_soft_ce_nats", "mean"),
            mean_oracle_ce=("test_oracle_soft_ce_nats", "mean"),
            mean_uniform_ce=("test_uniform_soft_ce_nats", "mean"),
            collapse_rows=("test_q_pred_top_action_collapse", "sum"),
        )
        .sort_values(["mean_q_top_bal_acc", "mean_q_kl"], ascending=[False, True])
    )
    best = summary.iloc[0].to_dict() if not summary.empty else {}
    lines = [
        "# q-only MLP Architecture Sweep",
        "",
        "Goal: predict the four-action behavior belief vector `q` from activation features using only soft-label cross entropy.",
        "",
        "## Setup",
        "",
        f"- layer: `{args.layer}`",
        f"- representation: `{args.representation}`",
        f"- subset: `{args.subset}`",
        f"- loss weighting: `{args.loss_weighting}`",
        f"- max epochs: {args.max_epochs}",
        f"- patience: {args.patience}",
        f"- output dir: `{output_dir}`",
        "",
        "## Theory",
        "",
        "For target distribution q and model prediction p, soft-label cross entropy is `CE(q,p) = -sum_a q_a log p_a`.",
        "The theoretical minimum is achieved at `p=q`, giving `CE_min = H(q) = -sum_a q_a log q_a`.",
        "The reported `test_q_kl` is exactly the excess above this optimum: `KL(q||p) = CE(q,p) - H(q)`.",
        "Therefore the best possible `test_q_kl` is 0, and the best possible q-top-action accuracy is 1.0 except for exact ties.",
        "",
        "## Best Mean Result",
        "",
    ]
    if best:
        lines.extend(
            [
                f"- architecture: `{best['architecture']}`",
                f"- loss weighting: `{best['loss_weighting']}`",
                f"- mean q top-action accuracy: {best['mean_q_top_acc']:.4f}",
                f"- mean q top-action balanced accuracy: {best['mean_q_top_bal_acc']:.4f}",
                f"- mean q agent-action accuracy: {best['mean_q_agent_acc']:.4f}",
                f"- mean q KL / excess CE: {best['mean_q_kl']:.4f}",
                f"- mean model CE: {best['mean_model_ce']:.4f}",
                f"- mean theoretical CE lower bound H(q): {best['mean_oracle_ce']:.4f}",
                f"- mean uniform CE: {best['mean_uniform_ce']:.4f}",
                f"- collapse rows: {int(best['collapse_rows'])}",
            ]
        )
    lines.extend(["", "## Architecture Summary", "", summary.to_markdown(index=False), ""])
    lines.extend(
        [
            "## Figures",
            "",
            f"- `{figure_dir / 'q_only_architecture_balanced_accuracy.png'}`",
            f"- `{figure_dir / 'q_only_architecture_kl.png'}`",
        ]
    )
    save_markdown(output_dir / "q_only_mlp_architecture_report.md", "\n".join(lines))
    if best:
        write_json(output_dir / "best_q_only_mlp_config.json", json_ready(best))


def main() -> None:
    args = parse_args()
    output_dir = ensure_dir(args.output_dir)
    figure_dir = ensure_dir(args.figure_dir)
    metrics_path = output_dir / "q_only_mlp_architecture_metrics.csv"
    if metrics_path.exists() and not args.force:
        raise SystemExit(f"{metrics_path} exists; pass --force to overwrite")

    import torch

    device = choose_device(args.device)
    df = make_behavior_table(args.beliefs)
    sub_idx = np.flatnonzero(subset_mask(df, args.subset))
    if len(sub_idx) == 0:
        raise ValueError(f"subset has zero rows: {args.subset}")
    splits = load_or_create_splits(args.splits, df, args.group_col, seed=args.random_seed)
    if args.limit_splits is not None:
        splits = splits[: int(args.limit_splits)]
    if not splits:
        raise ValueError("no splits available")
    feature_path = Path(args.feature_dir) / f"layer_{args.layer:02d}_{args.representation}.npy"
    if not feature_path.exists():
        raise FileNotFoundError(feature_path)
    features = np.load(feature_path, mmap_mode="r")
    if features.shape[0] != len(df):
        raise ValueError(f"feature rows {features.shape[0]} != behavior rows {len(df)}")

    bottleneck_hidden_dims = parse_hidden_dims(args.bottleneck_hidden_dims)
    arch_specs = [parse_architecture(name, bottleneck_hidden_dims) for name in args.architectures]
    loss_weightings = ["plain", "balanced"] if args.loss_weighting == "both" else [args.loss_weighting]
    print(json.dumps({"device": str(device), "rows": len(df), "subset_rows": int(len(sub_idx)), "feature_path": str(feature_path), "architectures": [s.name for s in arch_specs], "loss_weightings": loss_weightings}, indent=2))

    metrics: list[dict[str, Any]] = []
    best_for_prediction: tuple[float, dict[str, Any], np.ndarray, np.ndarray] | None = None

    for split in splits:
        train_pool = np.intersect1d(split.train_indices, sub_idx, assume_unique=False)
        test = np.intersect1d(split.test_indices, sub_idx, assume_unique=False)
        train, val = make_train_val_split(df, train_pool, split.group_col, seed=args.random_seed + split.split_id)
        if len(train) == 0 or len(val) == 0 or len(test) == 0:
            raise ValueError(f"empty split {split.split_id}: train={len(train)} val={len(val)} test={len(test)}")
        scaler = fit_standardizer(features, train, standardize=args.standardize)
        x_train = transform_features(features, train, scaler, l2_normalize=args.l2_normalize)
        x_val = transform_features(features, val, scaler, l2_normalize=args.l2_normalize)
        x_test = transform_features(features, test, scaler, l2_normalize=args.l2_normalize)
        y_train = make_targets(df, train)
        y_val = make_targets(df, val)
        print(f"split={split.split_id} train={len(train)} val={len(val)} test={len(test)}")
        metrics.append(mean_q_baseline(df, train, test, split, "plain"))

        for spec in arch_specs:
            for weighting in loss_weightings:
                seed = args.random_seed + split.split_id * 1000 + abs(hash((spec.name, weighting))) % 997
                print(f"Training split={split.split_id} arch={spec.name} weighting={weighting}")
                model, train_info = train_one(
                    x_train,
                    y_train,
                    x_val,
                    y_val,
                    spec,
                    args,
                    device,
                    seed=seed,
                    balanced_loss=weighting == "balanced",
                )
                q_test, _ = predict_q(model, x_test, args.batch_size, device)
                row = evaluate_predictions(
                    df=df,
                    indices=test,
                    q_pred=q_test,
                    split=split,
                    spec=spec,
                    loss_weighting=weighting,
                    train_n=len(train),
                    val_n=len(val),
                    train_info=train_info,
                )
                row["layer"] = int(args.layer)
                row["representation"] = args.representation
                row["subset"] = args.subset
                metrics.append(row)
                score = float(row["test_q_top_action_balanced_accuracy"])
                if best_for_prediction is None or score > best_for_prediction[0]:
                    best_for_prediction = (score, row, test.copy(), q_test.copy())
                del model
                gc.collect()
                if str(device) == "mps" and hasattr(torch, "mps"):
                    torch.mps.empty_cache()

    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv(metrics_path, index=False)
    plot_summary(metrics_df, Path(figure_dir))
    write_report(metrics_df, Path(output_dir), Path(figure_dir), args)
    if args.save_best_predictions and best_for_prediction is not None:
        _, row, test_indices, q_pred = best_for_prediction
        save_best_predictions(Path(output_dir) / "best_q_only_mlp_predictions.parquet", df, test_indices, q_pred, row=row)
    print(
        json.dumps(
            {
                "metrics": str(metrics_path),
                "report": str(Path(output_dir) / "q_only_mlp_architecture_report.md"),
                "figure_dir": str(figure_dir),
                "rows": int(len(metrics_df)),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
