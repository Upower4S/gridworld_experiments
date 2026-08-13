#!/usr/bin/env python
"""Sweep MLP classifiers for hard top-action prediction.

Target:
    top_action = argmax(q_UP, q_DOWN, q_LEFT, q_RIGHT)

This is intentionally different from q-only soft-label prediction. The model is
trained as a four-class classifier and evaluated on held-out trajectory groups.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
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
from src.conceptual_belief_space.metrics import balanced_accuracy_np, macro_f1_np
from src.conceptual_belief_space.mlp_branch_utils import (
    SplitDef,
    action_indices_to_labels,
    action_labels_to_indices,
    class_weights,
    fit_standardizer,
    json_ready,
    load_or_create_splits,
    make_behavior_table,
    make_train_val_split,
    subset_mask,
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
    parser.add_argument("--output_dir", default="outputs/conceptual_belief_space/top_action_mlp_sweep")
    parser.add_argument("--figure_dir", default="figures/conceptual_belief_space/mlp_branch_probe/top_action_sweep")
    parser.add_argument("--layer", type=int, default=10)
    parser.add_argument("--representation", default="delta_pre")
    parser.add_argument("--subset", default="strict_before_final_action_mention")
    parser.add_argument("--group_col", default="trajectory_id")
    parser.add_argument(
        "--architectures",
        nargs="+",
        default=[
            "linear",
            "mlp_128",
            "mlp_256",
            "mlp_512",
            "mlp_1024",
            "mlp_256_128",
            "mlp_512_256",
            "mlp_1024_512",
            "mlp_512_256_128",
            "mlp_1024_512_256",
            "bottleneck_2",
            "bottleneck_4",
            "bottleneck_8",
            "bottleneck_16",
            "bottleneck_32",
            "bottleneck_64",
            "bottleneck_128",
            "bottleneck_256",
        ],
    )
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
    parser.add_argument("--collapse_penalty", type=float, default=0.05)
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
        return ArchSpec(name=name, hidden_dims=bottleneck_hidden_dims, bottleneck_dim=int(name[len("bottleneck_") :]))
    raise ValueError(f"unknown architecture: {name}")


def stable_seed(*parts: object, base: int = 42) -> int:
    text = "::".join(str(part) for part in parts)
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return int((base + int(digest[:8], 16)) % (2**31 - 1))


def choose_device(device_arg: str):
    import torch

    if device_arg != "auto":
        return torch.device(device_arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_model(input_dim: int, spec: ArchSpec, dropout: float):
    import torch.nn as nn

    class TopActionMLP(nn.Module):
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
                self.classifier = nn.Linear(int(spec.bottleneck_dim), len(ACTIONS))
            else:
                self.bottleneck = None
                self.classifier = nn.Linear(prev, len(ACTIONS))

        def forward(self, x):
            hidden = self.encoder(x)
            z = self.bottleneck(hidden) if self.bottleneck is not None else hidden
            return {"logits": self.classifier(z), "z": z}

    return TopActionMLP()


def prediction_coverage(y_pred: np.ndarray, y_true: np.ndarray) -> dict[str, Any]:
    pred_counts = {action: int(np.sum(y_pred == i)) for i, action in enumerate(ACTIONS)}
    true_counts = {action: int(np.sum(y_true == i)) for i, action in enumerate(ACTIONS)}
    present = [action for i, action in enumerate(ACTIONS) if true_counts[action] > 0]
    missing = [action for action in present if pred_counts[action] == 0]
    return {
        "pred_counts": pred_counts,
        "true_counts": true_counts,
        "missing_actions": missing,
        "num_predicted_actions": int(sum(v > 0 for v in pred_counts.values())),
        "has_action_collapse": bool(missing),
    }


def confusion_matrix_json(y_true: np.ndarray, y_pred: np.ndarray) -> str:
    matrix: dict[str, dict[str, int]] = {}
    for i, actual in enumerate(ACTIONS):
        matrix[actual] = {}
        for j, predicted in enumerate(ACTIONS):
            matrix[actual][predicted] = int(np.sum((y_true == i) & (y_pred == j)))
    return json.dumps(matrix, sort_keys=True)


def metrics_from_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    true_labels = action_indices_to_labels(y_true)
    pred_labels = action_indices_to_labels(y_pred)
    coverage = prediction_coverage(y_pred, y_true)
    row: dict[str, Any] = {
        "accuracy": float(np.mean(y_true == y_pred)) if len(y_true) else float("nan"),
        "balanced_accuracy": balanced_accuracy_np(true_labels, pred_labels),
        "macro_f1": macro_f1_np(true_labels, pred_labels),
        "collapse": coverage["has_action_collapse"],
        "num_predicted_actions": coverage["num_predicted_actions"],
        "missing_actions_json": json.dumps(coverage["missing_actions"]),
        "confusion_matrix_json": confusion_matrix_json(y_true, y_pred),
    }
    for action in ACTIONS:
        row[f"true_count_{action}"] = coverage["true_counts"][action]
        row[f"pred_count_{action}"] = coverage["pred_counts"][action]
    return row


def predict_labels(model, x: np.ndarray, batch_size: int, device) -> tuple[np.ndarray, np.ndarray]:
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    loader = DataLoader(TensorDataset(torch.tensor(x, dtype=torch.float32)), batch_size=batch_size, shuffle=False)
    preds: list[np.ndarray] = []
    z_parts: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for (xb,) in loader:
            out = model(xb.to(device))
            preds.append(out["logits"].argmax(dim=1).detach().cpu().numpy())
            z = out["z"].detach().cpu().numpy()
            if z.ndim == 1:
                z = z[:, None]
            z_parts.append(z[:, : min(8, z.shape[1])])
    return np.concatenate(preds, axis=0).astype(np.int64), np.concatenate(z_parts, axis=0).astype(np.float32)


def train_one(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
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
    weights = torch.tensor(class_weights(y_train), dtype=torch.float32, device=device) if balanced_loss else None
    dataset = TensorDataset(torch.tensor(x_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
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
        for xb, yb in loader:
            out = model(xb.to(device))
            loss = F.cross_entropy(out["logits"], yb.to(device), weight=weights)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if args.grad_clip and args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
        val_pred, _ = predict_labels(model, x_val, args.batch_size, device)
        val_m = metrics_from_predictions(y_val, val_pred)
        score = float(val_m["balanced_accuracy"]) - float(args.collapse_penalty) * len(json.loads(val_m["missing_actions_json"]))
        if score > best_score:
            best_score = score
            best_epoch = epoch
            best_info = {
                "validation_accuracy": val_m["accuracy"],
                "validation_balanced_accuracy": val_m["balanced_accuracy"],
                "validation_macro_f1": val_m["macro_f1"],
                "validation_missing_actions_json": val_m["missing_actions_json"],
                "validation_num_predicted_actions": val_m["num_predicted_actions"],
            }
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, {"best_epoch": best_epoch, "validation_score": best_score, **best_info}


def evaluate_row(
    *,
    df: pd.DataFrame,
    test_indices: np.ndarray,
    y_pred: np.ndarray,
    split: SplitDef,
    spec: ArchSpec,
    loss_weighting: str,
    train_n: int,
    val_n: int,
    train_info: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    rows = df.iloc[np.asarray(test_indices, dtype=np.int64)]
    y_true = action_labels_to_indices(rows["top_action"])
    y_agent = action_labels_to_indices(rows["agent_action"])
    top_m = metrics_from_predictions(y_true, y_pred)
    agent_m = metrics_from_predictions(y_agent, y_pred)
    row: dict[str, Any] = {
        "run_id": f"topaction_{spec.name}_{loss_weighting}_split{split.split_id}_{int(time.time())}",
        "split_id": int(split.split_id),
        "layer": int(args.layer),
        "representation": args.representation,
        "subset": args.subset,
        "architecture": spec.name,
        "hidden_dims_json": json.dumps(list(spec.hidden_dims)),
        "uses_bottleneck": bool(spec.uses_bottleneck),
        "bottleneck_dim": int(spec.bottleneck_dim or 0),
        "loss_weighting": loss_weighting,
        "n_train": int(train_n),
        "n_val": int(val_n),
        "n_test": int(len(test_indices)),
        "n_train_trajectories": int(len(set(df.iloc[split.train_indices][split.group_col].astype(str)))),
        "n_test_trajectories": int(len(set(rows[split.group_col].astype(str)))),
        "best_epoch": int(train_info.get("best_epoch", -1)),
        "validation_score": float(train_info.get("validation_score", float("nan"))),
        "validation_accuracy": float(train_info.get("validation_accuracy", float("nan"))),
        "validation_balanced_accuracy": float(train_info.get("validation_balanced_accuracy", float("nan"))),
        "validation_macro_f1": float(train_info.get("validation_macro_f1", float("nan"))),
        "test_top_action_accuracy": top_m["accuracy"],
        "test_top_action_balanced_accuracy": top_m["balanced_accuracy"],
        "test_top_action_macro_f1": top_m["macro_f1"],
        "test_agent_action_accuracy": agent_m["accuracy"],
        "test_agent_action_balanced_accuracy": agent_m["balanced_accuracy"],
        "test_action_collapse": top_m["collapse"],
        "test_num_predicted_actions": top_m["num_predicted_actions"],
        "test_missing_actions_json": top_m["missing_actions_json"],
        "top_action_confusion_matrix_json": top_m["confusion_matrix_json"],
        "agent_action_confusion_matrix_json": agent_m["confusion_matrix_json"],
    }
    for action in ACTIONS:
        row[f"test_true_top_count_{action}"] = top_m[f"true_count_{action}"]
        row[f"test_pred_top_count_{action}"] = top_m[f"pred_count_{action}"]
        row[f"test_true_agent_count_{action}"] = agent_m[f"true_count_{action}"]
    return row


def majority_baseline(df: pd.DataFrame, train: np.ndarray, test: np.ndarray, split: SplitDef, args: argparse.Namespace) -> dict[str, Any]:
    y_train = action_labels_to_indices(df.iloc[train]["top_action"])
    counts = np.bincount(y_train, minlength=len(ACTIONS))
    majority = int(np.argmax(counts))
    y_pred = np.full(len(test), majority, dtype=np.int64)
    spec = ArchSpec(name="train_majority_baseline", hidden_dims=())
    return evaluate_row(df=df, test_indices=test, y_pred=y_pred, split=split, spec=spec, loss_weighting="plain", train_n=len(train), val_n=0, train_info={"best_epoch": -1}, args=args)


def save_best_predictions(path: Path, df: pd.DataFrame, test_indices: np.ndarray, y_pred: np.ndarray, row: dict[str, Any]) -> None:
    cols = ["row_id", "prefix_id", "decision_id", "trajectory_id", "step_id", "prefix_token_index", "progress", "agent_action", "top_action"]
    out = df.iloc[np.asarray(test_indices, dtype=np.int64)][[col for col in cols if col in df.columns]].copy()
    out.insert(0, "run_id", row["run_id"])
    out.insert(1, "architecture", row["architecture"])
    out.insert(2, "loss_weighting", row["loss_weighting"])
    out["pred_top_action"] = action_indices_to_labels(y_pred)
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
            mean_bal=("test_top_action_balanced_accuracy", "mean"),
            mean_acc=("test_top_action_accuracy", "mean"),
            mean_f1=("test_top_action_macro_f1", "mean"),
            collapse_rows=("test_action_collapse", "sum"),
        )
        .sort_values("mean_bal", ascending=False)
    )
    labels = [f"{r.architecture}\n{r.loss_weighting}" for r in summary.itertuples()]
    x = np.arange(len(summary))
    fig, ax = plt.subplots(figsize=(max(10, len(summary) * 0.38), 5))
    ax.bar(x, summary["mean_bal"].to_numpy(), color="C0", label="balanced accuracy")
    ax.bar(x, summary["mean_acc"].to_numpy(), color="C1", alpha=0.45, label="accuracy")
    ax.axhline(0.25, color="black", linewidth=1, linestyle="--", label="chance")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=70, ha="right", fontsize=8)
    ax.set_ylabel("test score")
    ax.set_title("Hard top-action classifier by architecture")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figure_dir / "top_action_architecture_accuracy.png", dpi=200)
    plt.close(fig)


def write_report(metrics: pd.DataFrame, output_dir: Path, figure_dir: Path, args: argparse.Namespace) -> None:
    main = metrics[~metrics["architecture"].astype(str).str.contains("baseline", na=False)].copy()
    summary = (
        main.groupby(["architecture", "loss_weighting"], as_index=False)
        .agg(
            splits=("split_id", "nunique"),
            mean_acc=("test_top_action_accuracy", "mean"),
            std_acc=("test_top_action_accuracy", "std"),
            mean_bal=("test_top_action_balanced_accuracy", "mean"),
            std_bal=("test_top_action_balanced_accuracy", "std"),
            mean_f1=("test_top_action_macro_f1", "mean"),
            mean_agent_acc=("test_agent_action_accuracy", "mean"),
            collapse_rows=("test_action_collapse", "sum"),
            mean_pred_UP=("test_pred_top_count_UP", "mean"),
            mean_pred_DOWN=("test_pred_top_count_DOWN", "mean"),
            mean_pred_LEFT=("test_pred_top_count_LEFT", "mean"),
            mean_pred_RIGHT=("test_pred_top_count_RIGHT", "mean"),
        )
        .sort_values(["mean_bal", "mean_acc"], ascending=[False, False])
    )
    summary.to_csv(output_dir / "top_action_mlp_architecture_summary.csv", index=False)
    best = summary.iloc[0].to_dict() if not summary.empty else {}
    baselines = metrics[metrics["architecture"].astype(str).str.contains("baseline", na=False)].copy()
    baseline_summary = (
        baselines.groupby(["architecture", "loss_weighting"], as_index=False)
        .agg(mean_acc=("test_top_action_accuracy", "mean"), mean_bal=("test_top_action_balanced_accuracy", "mean"), mean_f1=("test_top_action_macro_f1", "mean"))
        if not baselines.empty
        else pd.DataFrame()
    )
    lines = [
        "# Hard top-action MLP Classifier Sweep",
        "",
        "Target is `top_action = argmax(q_UP, q_DOWN, q_LEFT, q_RIGHT)`. This experiment does not train on the full q distribution.",
        "",
        "## Setup",
        "",
        f"- layer: `{args.layer}`",
        f"- representation: `{args.representation}`",
        f"- subset: `{args.subset}`",
        f"- loss weighting: `{args.loss_weighting}`",
        f"- grouped split column: `{args.group_col}`",
        "",
        "## Best Mean Result",
        "",
    ]
    if best:
        for key in ["architecture", "loss_weighting", "splits", "mean_acc", "mean_bal", "mean_f1", "mean_agent_acc", "collapse_rows"]:
            value = best[key]
            lines.append(f"- {key}: {value:.6f}" if isinstance(value, float) else f"- {key}: `{value}`")
    lines.extend(["", "## Top Architectures", "", summary.head(25).to_markdown(index=False), ""])
    if not baseline_summary.empty:
        lines.extend(["", "## Baseline", "", baseline_summary.to_markdown(index=False), ""])
    lines.extend(["", "## Figure", "", f"- `{figure_dir / 'top_action_architecture_accuracy.png'}`"])
    save_markdown(output_dir / "top_action_mlp_architecture_report.md", "\n".join(lines))
    if best:
        write_json(output_dir / "best_top_action_mlp_config.json", json_ready(best))


def main() -> None:
    args = parse_args()
    output_dir = ensure_dir(args.output_dir)
    figure_dir = ensure_dir(args.figure_dir)
    metrics_path = output_dir / "top_action_mlp_architecture_metrics.csv"
    if metrics_path.exists() and not args.force:
        raise SystemExit(f"{metrics_path} exists; pass --force to overwrite")

    import torch

    device = choose_device(args.device)
    df = make_behavior_table(args.beliefs)
    sub_idx = np.flatnonzero(subset_mask(df, args.subset))
    splits = load_or_create_splits(args.splits, df, args.group_col, seed=args.random_seed)
    if args.limit_splits is not None:
        splits = splits[: int(args.limit_splits)]
    feature_path = Path(args.feature_dir) / f"layer_{args.layer:02d}_{args.representation}.npy"
    if not feature_path.exists():
        raise FileNotFoundError(feature_path)
    features = np.load(feature_path, mmap_mode="r")
    if features.shape[0] != len(df):
        raise ValueError(f"feature rows {features.shape[0]} != behavior rows {len(df)}")

    bottleneck_hidden_dims = parse_hidden_dims(args.bottleneck_hidden_dims)
    arch_specs = [parse_architecture(name, bottleneck_hidden_dims) for name in args.architectures]
    loss_weightings = ["plain", "balanced"] if args.loss_weighting == "both" else [args.loss_weighting]
    print(
        json.dumps(
            {
                "device": str(device),
                "rows": len(df),
                "subset_rows": int(len(sub_idx)),
                "feature_path": str(feature_path),
                "architectures": [spec.name for spec in arch_specs],
                "loss_weightings": loss_weightings,
            },
            indent=2,
        )
    )

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
        y_train = action_labels_to_indices(df.iloc[train]["top_action"])
        y_val = action_labels_to_indices(df.iloc[val]["top_action"])
        print(f"split={split.split_id} train={len(train)} val={len(val)} test={len(test)}")
        metrics.append(majority_baseline(df, train, test, split, args))

        for spec in arch_specs:
            for weighting in loss_weightings:
                seed = stable_seed(spec.name, weighting, split.split_id, args.layer, args.representation, args.subset, base=args.random_seed)
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
                y_pred, _ = predict_labels(model, x_test, args.batch_size, device)
                row = evaluate_row(df=df, test_indices=test, y_pred=y_pred, split=split, spec=spec, loss_weighting=weighting, train_n=len(train), val_n=len(val), train_info=train_info, args=args)
                metrics.append(row)
                score = float(row["test_top_action_balanced_accuracy"])
                if best_for_prediction is None or score > best_for_prediction[0]:
                    best_for_prediction = (score, row, test.copy(), y_pred.copy())
                del model
                gc.collect()
                if str(device) == "mps" and hasattr(torch, "mps"):
                    torch.mps.empty_cache()

    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv(metrics_path, index=False)
    plot_summary(metrics_df, Path(figure_dir))
    write_report(metrics_df, Path(output_dir), Path(figure_dir), args)
    if args.save_best_predictions and best_for_prediction is not None:
        _, row, test_indices, y_pred = best_for_prediction
        save_best_predictions(output_dir / "best_top_action_mlp_predictions.parquet", df, test_indices, y_pred, row)
    print(
        json.dumps(
            {
                "metrics": str(metrics_path),
                "summary": str(output_dir / "top_action_mlp_architecture_summary.csv"),
                "report": str(output_dir / "top_action_mlp_architecture_report.md"),
                "figure_dir": str(figure_dir),
                "rows": int(len(metrics_df)),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
