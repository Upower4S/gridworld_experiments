#!/usr/bin/env python
"""Train sequence Transformers for hard top-action prediction.

This treats each decision_id as a prefix sequence and predicts
top_action = argmax(q) at every prefix. By default the Transformer is causal, so
the prediction at prefix k can attend only to prefixes <= k.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
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

from src.conceptual_belief_space.config import ACTIONS
from src.conceptual_belief_space.io import ensure_dir, save_markdown, write_json
from src.conceptual_belief_space.metrics import balanced_accuracy_np, macro_f1_np
from src.conceptual_belief_space.mlp_branch_utils import (
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
class TransformerSpec:
    name: str
    d_model: int
    n_layers: int
    n_heads: int
    dim_feedforward: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--beliefs", default="outputs/conceptual_belief_space/action_belief_vectors.parquet")
    parser.add_argument("--splits", default="outputs/conceptual_belief_space/splits/group_splits.json")
    parser.add_argument("--feature_dir", default="outputs/activation/features")
    parser.add_argument("--output_dir", default="outputs/conceptual_belief_space/top_action_sequence_transformer")
    parser.add_argument("--figure_dir", default="figures/conceptual_belief_space/mlp_branch_probe/top_action_sequence_transformer")
    parser.add_argument("--layer", type=int, default=10)
    parser.add_argument("--representation", default="delta_pre")
    parser.add_argument("--subset", default="strict_before_final_action_mention")
    parser.add_argument("--group_col", default="trajectory_id")
    parser.add_argument(
        "--architectures",
        nargs="+",
        default=[
            "tfm_d128_l2_h4",
            "tfm_d256_l2_h4",
            "tfm_d256_l4_h8",
        ],
    )
    parser.add_argument("--attention_modes", nargs="+", choices=["causal", "bidirectional"], default=["causal"])
    parser.add_argument("--loss_weighting", choices=["plain", "balanced", "both"], default="both")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--standardize", action="store_true", default=True)
    parser.add_argument("--no-standardize", action="store_false", dest="standardize")
    parser.add_argument("--l2_normalize", action="store_true")
    parser.add_argument("--collapse_penalty", type=float, default=0.05)
    parser.add_argument("--limit_splits", type=int, default=None)
    parser.add_argument("--save_best_predictions", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def parse_architecture(text: str) -> TransformerSpec:
    # Expected: tfm_d256_l2_h4 or tfm_d256_l2_h4_ff1024
    parts = text.split("_")
    if len(parts) < 4 or parts[0] != "tfm":
        raise ValueError(f"invalid transformer architecture: {text}")
    vals: dict[str, int] = {}
    for part in parts[1:]:
        if part.startswith("d"):
            vals["d_model"] = int(part[1:])
        elif part.startswith("l"):
            vals["n_layers"] = int(part[1:])
        elif part.startswith("h"):
            vals["n_heads"] = int(part[1:])
        elif part.startswith("ff"):
            vals["dim_feedforward"] = int(part[2:])
    d_model = vals["d_model"]
    n_layers = vals["n_layers"]
    n_heads = vals["n_heads"]
    ff = vals.get("dim_feedforward", 4 * d_model)
    if d_model % n_heads != 0:
        raise ValueError(f"d_model must be divisible by n_heads: {text}")
    return TransformerSpec(text, d_model, n_layers, n_heads, ff)


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


def make_sequences(df: pd.DataFrame, indices: np.ndarray) -> list[np.ndarray]:
    idx = np.asarray(indices, dtype=np.int64)
    if len(idx) == 0:
        return []
    temp = df.iloc[idx][["decision_id", "prefix_token_index"]].copy()
    temp["_row_pos"] = idx
    temp = temp.sort_values(["decision_id", "prefix_token_index", "_row_pos"])
    return [group["_row_pos"].to_numpy(dtype=np.int64) for _, group in temp.groupby("decision_id", sort=False)]


def labels_for_rows(df: pd.DataFrame, indices: np.ndarray) -> np.ndarray:
    return action_labels_to_indices(df.iloc[np.asarray(indices, dtype=np.int64)]["top_action"])


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


class SequenceDataset:
    def __init__(self, x: np.ndarray, y: np.ndarray, row_indices: np.ndarray, sequences: list[np.ndarray], row_to_local: dict[int, int]) -> None:
        self.x = x
        self.y = y
        self.row_indices = row_indices
        self.sequences = []
        for seq in sequences:
            local = [row_to_local[int(row)] for row in seq if int(row) in row_to_local]
            if local:
                self.sequences.append(np.asarray(local, dtype=np.int64))

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, i: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        local = self.sequences[i]
        return self.x[local], self.y[local], self.row_indices[local]


def collate_sequences(batch: list[tuple[np.ndarray, np.ndarray, np.ndarray]]):
    import torch

    max_len = max(x.shape[0] for x, _, _ in batch)
    feat_dim = batch[0][0].shape[1]
    x = np.zeros((len(batch), max_len, feat_dim), dtype=np.float32)
    y = np.full((len(batch), max_len), -100, dtype=np.int64)
    mask = np.ones((len(batch), max_len), dtype=bool)
    rows = np.full((len(batch), max_len), -1, dtype=np.int64)
    for i, (seq_x, seq_y, seq_rows) in enumerate(batch):
        n = seq_x.shape[0]
        x[i, :n] = seq_x
        y[i, :n] = seq_y
        mask[i, :n] = False
        rows[i, :n] = seq_rows
    return torch.tensor(x), torch.tensor(y), torch.tensor(mask), torch.tensor(rows)


def make_local_dataset(features: np.ndarray, df: pd.DataFrame, rows: np.ndarray, sequences: list[np.ndarray], scaler: dict[str, np.ndarray], l2_normalize: bool) -> SequenceDataset:
    rows = np.asarray(rows, dtype=np.int64)
    x = transform_features(features, rows, scaler, l2_normalize=l2_normalize)
    y = labels_for_rows(df, rows)
    row_to_local = {int(row): i for i, row in enumerate(rows)}
    return SequenceDataset(x, y, rows, sequences, row_to_local)


def build_model(input_dim: int, spec: TransformerSpec, max_seq_len: int, dropout: float):
    import torch
    import torch.nn as nn

    class PrefixTransformer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.input_proj = nn.Sequential(
                nn.LayerNorm(input_dim),
                nn.Linear(input_dim, spec.d_model),
            )
            self.pos_embedding = nn.Parameter(torch.zeros(1, max_seq_len, spec.d_model))
            nn.init.normal_(self.pos_embedding, mean=0.0, std=0.02)
            layer = nn.TransformerEncoderLayer(
                d_model=spec.d_model,
                nhead=spec.n_heads,
                dim_feedforward=spec.dim_feedforward,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.encoder = nn.TransformerEncoder(layer, num_layers=spec.n_layers)
            self.norm = nn.LayerNorm(spec.d_model)
            self.classifier = nn.Linear(spec.d_model, len(ACTIONS))

        def forward(self, x, padding_mask, causal: bool):
            length = x.shape[1]
            h = self.input_proj(x) + self.pos_embedding[:, :length]
            attn_mask = None
            if causal:
                # Keep the attention mask boolean so it matches the padding mask.
                # Mixed float/bool masks can produce backend-specific behavior on MPS.
                attn_mask = torch.triu(
                    torch.ones((length, length), dtype=torch.bool, device=x.device),
                    diagonal=1,
                )
            h = self.encoder(h, mask=attn_mask, src_key_padding_mask=padding_mask)
            return self.classifier(self.norm(h))

    return PrefixTransformer()


def predict(model, dataset: SequenceDataset, batch_size: int, device, causal: bool) -> tuple[np.ndarray, np.ndarray]:
    import torch
    from torch.utils.data import DataLoader

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_sequences)
    pred_by_row: dict[int, int] = {}
    true_by_row: dict[int, int] = {}
    model.eval()
    with torch.no_grad():
        for xb, yb, pad_mask, rows in loader:
            xb = xb.to(device)
            pad_mask = pad_mask.to(device)
            logits = model(xb, pad_mask, causal=causal)
            pred = logits.argmax(dim=-1).detach().cpu().numpy()
            y_np = yb.numpy()
            rows_np = rows.numpy()
            valid = rows_np >= 0
            for row, yy, pp in zip(rows_np[valid].reshape(-1), y_np[valid].reshape(-1), pred[valid].reshape(-1)):
                pred_by_row[int(row)] = int(pp)
                true_by_row[int(row)] = int(yy)
    ordered_rows = np.asarray(sorted(pred_by_row), dtype=np.int64)
    y_true = np.asarray([true_by_row[int(row)] for row in ordered_rows], dtype=np.int64)
    y_pred = np.asarray([pred_by_row[int(row)] for row in ordered_rows], dtype=np.int64)
    return ordered_rows, y_true, y_pred


def train_one(
    train_ds: SequenceDataset,
    val_ds: SequenceDataset,
    input_dim: int,
    max_seq_len: int,
    spec: TransformerSpec,
    args: argparse.Namespace,
    device,
    *,
    seed: int,
    balanced_loss: bool,
    causal: bool,
) -> tuple[Any, dict[str, Any]]:
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader

    torch.manual_seed(seed)
    np.random.seed(seed)
    model = build_model(input_dim, spec, max_seq_len, args.dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    weights = torch.tensor(class_weights(train_ds.y), dtype=torch.float32, device=device) if balanced_loss else None
    generator = torch.Generator()
    generator.manual_seed(seed)
    loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, generator=generator, collate_fn=collate_sequences)

    best_state = None
    best_score = -float("inf")
    best_epoch = -1
    best_info: dict[str, Any] = {}
    bad_epochs = 0
    for epoch in range(int(args.max_epochs)):
        model.train()
        for xb, yb, pad_mask, _ in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            pad_mask = pad_mask.to(device)
            logits = model(xb, pad_mask, causal=causal)
            loss = F.cross_entropy(logits.reshape(-1, len(ACTIONS)), yb.reshape(-1), ignore_index=-100, weight=weights)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if args.grad_clip and args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
        _, y_val, p_val = predict(model, val_ds, args.batch_size, device, causal=causal)
        val_m = metrics_from_predictions(y_val, p_val)
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
    row_ids: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    split: Any,
    spec: TransformerSpec,
    attention_mode: str,
    loss_weighting: str,
    train_n: int,
    val_n: int,
    train_seq_n: int,
    val_seq_n: int,
    test_seq_n: int,
    train_info: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    top_m = metrics_from_predictions(y_true, y_pred)
    y_agent = action_labels_to_indices(df.loc[row_ids, "agent_action"])
    agent_m = metrics_from_predictions(y_agent, y_pred)
    row: dict[str, Any] = {
        "run_id": f"seqtfm_{spec.name}_{attention_mode}_{loss_weighting}_split{split.split_id}_{int(time.time())}",
        "split_id": int(split.split_id),
        "layer": int(args.layer),
        "representation": args.representation,
        "subset": args.subset,
        "architecture": spec.name,
        "d_model": int(spec.d_model),
        "n_layers": int(spec.n_layers),
        "n_heads": int(spec.n_heads),
        "dim_feedforward": int(spec.dim_feedforward),
        "attention_mode": attention_mode,
        "loss_weighting": loss_weighting,
        "n_train": int(train_n),
        "n_val": int(val_n),
        "n_test": int(len(row_ids)),
        "n_train_sequences": int(train_seq_n),
        "n_val_sequences": int(val_seq_n),
        "n_test_sequences": int(test_seq_n),
        "n_train_trajectories": int(len(set(df.iloc[split.train_indices][split.group_col].astype(str)))),
        "n_test_trajectories": int(len(set(df.loc[row_ids, split.group_col].astype(str)))),
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


def majority_baseline(df: pd.DataFrame, train_rows: np.ndarray, test_rows: np.ndarray, split: Any, args: argparse.Namespace) -> dict[str, Any]:
    y_train = action_labels_to_indices(df.iloc[train_rows]["top_action"])
    y_true = action_labels_to_indices(df.iloc[test_rows]["top_action"])
    counts = np.bincount(y_train, minlength=len(ACTIONS))
    pred = np.full(len(test_rows), int(np.argmax(counts)), dtype=np.int64)
    spec = TransformerSpec("train_majority_baseline", 0, 0, 0, 0)
    return evaluate_row(
        df=df,
        row_ids=test_rows,
        y_true=y_true,
        y_pred=pred,
        split=split,
        spec=spec,
        attention_mode="none",
        loss_weighting="plain",
        train_n=len(train_rows),
        val_n=0,
        train_seq_n=0,
        val_seq_n=0,
        test_seq_n=0,
        train_info={"best_epoch": -1},
        args=args,
    )


def save_best_predictions(path: Path, df: pd.DataFrame, row_ids: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray, row: dict[str, Any]) -> None:
    cols = ["row_id", "prefix_id", "decision_id", "trajectory_id", "step_id", "prefix_token_index", "progress", "agent_action", "top_action"]
    out = df.loc[np.asarray(row_ids, dtype=np.int64), [col for col in cols if col in df.columns]].copy()
    out.insert(0, "run_id", row["run_id"])
    out.insert(1, "architecture", row["architecture"])
    out.insert(2, "attention_mode", row["attention_mode"])
    out.insert(3, "loss_weighting", row["loss_weighting"])
    out["pred_top_action"] = action_indices_to_labels(y_pred)
    out["target_top_action"] = action_indices_to_labels(y_true)
    ensure_dir(path.parent)
    out.to_parquet(path, index=False)


def plot_summary(metrics: pd.DataFrame, figure_dir: Path) -> None:
    import matplotlib.pyplot as plt

    ensure_dir(figure_dir)
    main = metrics[~metrics["architecture"].astype(str).str.contains("baseline", na=False)].copy()
    if main.empty:
        return
    summary = (
        main.groupby(["architecture", "attention_mode", "loss_weighting"], as_index=False)
        .agg(mean_bal=("test_top_action_balanced_accuracy", "mean"), mean_acc=("test_top_action_accuracy", "mean"), mean_f1=("test_top_action_macro_f1", "mean"))
        .sort_values("mean_bal", ascending=False)
    )
    labels = [f"{r.architecture}\n{r.attention_mode}/{r.loss_weighting}" for r in summary.itertuples()]
    x = np.arange(len(summary))
    fig, ax = plt.subplots(figsize=(max(10, len(summary) * 0.48), 5))
    ax.bar(x, summary["mean_bal"].to_numpy(), color="C0", label="balanced accuracy")
    ax.bar(x, summary["mean_acc"].to_numpy(), color="C1", alpha=0.45, label="accuracy")
    ax.axhline(0.25, color="black", linestyle="--", linewidth=1, label="chance")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=70, ha="right", fontsize=8)
    ax.set_ylabel("test score")
    ax.set_title("Sequence Transformer top-action prediction")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figure_dir / "sequence_transformer_top_action_accuracy.png", dpi=200)
    plt.close(fig)


def write_report(metrics: pd.DataFrame, output_dir: Path, figure_dir: Path, args: argparse.Namespace) -> None:
    main = metrics[~metrics["architecture"].astype(str).str.contains("baseline", na=False)].copy()
    summary = (
        main.groupby(["architecture", "attention_mode", "loss_weighting"], as_index=False)
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
    summary.to_csv(output_dir / "sequence_transformer_summary.csv", index=False)
    baselines = metrics[metrics["architecture"].astype(str).str.contains("baseline", na=False)].copy()
    baseline_summary = (
        baselines.groupby(["architecture", "loss_weighting"], as_index=False).agg(mean_acc=("test_top_action_accuracy", "mean"), mean_bal=("test_top_action_balanced_accuracy", "mean"), mean_f1=("test_top_action_macro_f1", "mean"))
        if not baselines.empty
        else pd.DataFrame()
    )
    best = summary.iloc[0].to_dict() if not summary.empty else {}
    lines = [
        "# Sequence Transformer Top-Action Classifier",
        "",
        "Target is hard `top_action = argmax(q)`. This sequence model sees prefix activations ordered within each decision.",
        "For `attention_mode=causal`, the prediction at prefix k attends only to prefixes <= k. `bidirectional` is a diagnostic that can see later prefixes and should not be used as a strict online readout.",
        "",
        "## Setup",
        "",
        f"- layer: `{args.layer}`",
        f"- representation: `{args.representation}`",
        f"- subset: `{args.subset}`",
        f"- attention modes: `{', '.join(args.attention_modes)}`",
        f"- loss weighting: `{args.loss_weighting}`",
        "",
        "## Best Mean Result",
        "",
    ]
    if best:
        for key in ["architecture", "attention_mode", "loss_weighting", "splits", "mean_acc", "mean_bal", "mean_f1", "mean_agent_acc", "collapse_rows"]:
            value = best[key]
            lines.append(f"- {key}: {value:.6f}" if isinstance(value, float) else f"- {key}: `{value}`")
    lines.extend(["", "## Summary", "", summary.to_markdown(index=False)])
    if not baseline_summary.empty:
        lines.extend(["", "## Baseline", "", baseline_summary.to_markdown(index=False)])
    lines.extend(["", "## Figure", "", f"- `{figure_dir / 'sequence_transformer_top_action_accuracy.png'}`"])
    save_markdown(output_dir / "sequence_transformer_report.md", "\n".join(lines))
    if best:
        write_json(output_dir / "best_sequence_transformer_config.json", json_ready(best))


def main() -> None:
    args = parse_args()
    output_dir = ensure_dir(args.output_dir)
    figure_dir = ensure_dir(args.figure_dir)
    metrics_path = output_dir / "sequence_transformer_metrics.csv"
    if metrics_path.exists() and not args.force:
        raise SystemExit(f"{metrics_path} exists; pass --force to overwrite")

    import torch

    device = choose_device(args.device)
    df = make_behavior_table(args.beliefs).reset_index(drop=True)
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
    specs = [parse_architecture(text) for text in args.architectures]
    loss_weightings = ["plain", "balanced"] if args.loss_weighting == "both" else [args.loss_weighting]
    print(
        json.dumps(
            {
                "device": str(device),
                "rows": len(df),
                "subset_rows": int(len(sub_idx)),
                "feature_path": str(feature_path),
                "architectures": [spec.name for spec in specs],
                "attention_modes": args.attention_modes,
                "loss_weightings": loss_weightings,
            },
            indent=2,
        )
    )

    metrics: list[dict[str, Any]] = []
    best_for_prediction: tuple[float, dict[str, Any], np.ndarray, np.ndarray, np.ndarray] | None = None
    for split in splits:
        train_pool = np.intersect1d(split.train_indices, sub_idx, assume_unique=False)
        test_rows = np.intersect1d(split.test_indices, sub_idx, assume_unique=False)
        train_rows, val_rows = make_train_val_split(df, train_pool, split.group_col, seed=args.random_seed + split.split_id)
        if len(train_rows) == 0 or len(val_rows) == 0 or len(test_rows) == 0:
            raise ValueError(f"empty split {split.split_id}: train={len(train_rows)} val={len(val_rows)} test={len(test_rows)}")
        scaler = fit_standardizer(features, train_rows, standardize=args.standardize)
        train_seqs = make_sequences(df, train_rows)
        val_seqs = make_sequences(df, val_rows)
        test_seqs = make_sequences(df, test_rows)
        train_ds = make_local_dataset(features, df, train_rows, train_seqs, scaler, args.l2_normalize)
        val_ds = make_local_dataset(features, df, val_rows, val_seqs, scaler, args.l2_normalize)
        test_ds = make_local_dataset(features, df, test_rows, test_seqs, scaler, args.l2_normalize)
        max_seq_len = max(max(len(seq) for seq in train_ds.sequences), max(len(seq) for seq in val_ds.sequences), max(len(seq) for seq in test_ds.sequences))
        print(f"split={split.split_id} train_rows={len(train_rows)} val_rows={len(val_rows)} test_rows={len(test_rows)} train_seq={len(train_ds)} val_seq={len(val_ds)} test_seq={len(test_ds)} max_seq_len={max_seq_len}")
        metrics.append(majority_baseline(df, train_rows, test_rows, split, args))

        for spec in specs:
            for attention_mode in args.attention_modes:
                causal = attention_mode == "causal"
                for weighting in loss_weightings:
                    seed = stable_seed(spec.name, attention_mode, weighting, split.split_id, args.layer, args.representation, args.subset, base=args.random_seed)
                    print(f"Training split={split.split_id} arch={spec.name} attention={attention_mode} weighting={weighting}")
                    model, train_info = train_one(
                        train_ds,
                        val_ds,
                        features.shape[1],
                        max_seq_len,
                        spec,
                        args,
                        device,
                        seed=seed,
                        balanced_loss=weighting == "balanced",
                        causal=causal,
                    )
                    row_ids, y_true, y_pred = predict(model, test_ds, args.batch_size, device, causal=causal)
                    row = evaluate_row(
                        df=df,
                        row_ids=row_ids,
                        y_true=y_true,
                        y_pred=y_pred,
                        split=split,
                        spec=spec,
                        attention_mode=attention_mode,
                        loss_weighting=weighting,
                        train_n=len(train_rows),
                        val_n=len(val_rows),
                        train_seq_n=len(train_ds),
                        val_seq_n=len(val_ds),
                        test_seq_n=len(test_ds),
                        train_info=train_info,
                        args=args,
                    )
                    metrics.append(row)
                    score = float(row["test_top_action_balanced_accuracy"])
                    if best_for_prediction is None or score > best_for_prediction[0]:
                        best_for_prediction = (score, row, row_ids.copy(), y_true.copy(), y_pred.copy())
                    del model
                    gc.collect()
                    if str(device) == "mps" and hasattr(torch, "mps"):
                        torch.mps.empty_cache()

    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv(metrics_path, index=False)
    plot_summary(metrics_df, Path(figure_dir))
    write_report(metrics_df, Path(output_dir), Path(figure_dir), args)
    if args.save_best_predictions and best_for_prediction is not None:
        _, row, row_ids, y_true, y_pred = best_for_prediction
        save_best_predictions(output_dir / "best_sequence_transformer_predictions.parquet", df, row_ids, y_true, y_pred, row)
    print(
        json.dumps(
            {
                "metrics": str(metrics_path),
                "summary": str(output_dir / "sequence_transformer_summary.csv"),
                "report": str(output_dir / "sequence_transformer_report.md"),
                "figure_dir": str(figure_dir),
                "rows": int(len(metrics_df)),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
