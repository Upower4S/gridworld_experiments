#!/usr/bin/env python
"""Sweep layer/feature-set inputs for causal Transformer top-action prediction.

Feature-set syntax:
  single:10
  mean:0,2,4,6
  mean:all
  concat:8,10,12,14
  mix:all

`mix` stacks layers and learns one scalar softmax weight per layer before the
sequence Transformer. It lets us use all layers without concatenating them into
a huge row matrix.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import re
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
    action_indices_to_labels,
    action_labels_to_indices,
    class_weights,
    json_ready,
    load_or_create_splits,
    make_behavior_table,
    make_train_val_split,
    subset_mask,
)


@dataclass(frozen=True)
class TransformerSpec:
    name: str
    d_model: int
    n_layers: int
    n_heads: int
    dim_feedforward: int


@dataclass(frozen=True)
class FeatureSet:
    name: str
    mode: str
    layers: tuple[int, ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--beliefs", default="outputs/conceptual_belief_space/action_belief_vectors.parquet")
    parser.add_argument("--splits", default="outputs/conceptual_belief_space/splits/group_splits.json")
    parser.add_argument("--feature_dir", default="outputs/activation/features")
    parser.add_argument("--output_dir", default="outputs/conceptual_belief_space/top_action_sequence_transformer_feature_sets")
    parser.add_argument("--figure_dir", default="figures/conceptual_belief_space/mlp_branch_probe/top_action_sequence_transformer_feature_sets")
    parser.add_argument("--representation", default="delta_pre")
    parser.add_argument("--subset", default="strict_before_final_action_mention")
    parser.add_argument("--group_col", default="trajectory_id")
    parser.add_argument("--feature_sets", nargs="+", default=["single:10", "mean:all", "mix:all"])
    parser.add_argument("--architectures", nargs="+", default=["tfm_d128_l2_h4"])
    parser.add_argument("--attention_modes", nargs="+", choices=["causal", "bidirectional"], default=["causal"])
    parser.add_argument("--loss_weighting", choices=["plain", "balanced", "both"], default="balanced")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--standardize", action="store_true", default=True)
    parser.add_argument("--no-standardize", action="store_false", dest="standardize")
    parser.add_argument("--l2_normalize", action="store_true")
    parser.add_argument("--collapse_penalty", type=float, default=0.05)
    parser.add_argument("--limit_splits", type=int, default=None)
    parser.add_argument("--save_best_predictions", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def available_layers(feature_dir: Path, representation: str) -> list[int]:
    out = []
    pattern = re.compile(rf"layer_(\d+)_{re.escape(representation)}\.npy$")
    for path in sorted(feature_dir.glob(f"layer_*_{representation}.npy")):
        match = pattern.match(path.name)
        if match:
            out.append(int(match.group(1)))
    return sorted(set(out))


def parse_feature_set(text: str, all_layers: list[int]) -> FeatureSet:
    if ":" not in text:
        raise ValueError(f"invalid feature set {text!r}; expected mode:layers")
    mode, layer_text = text.split(":", 1)
    mode = mode.strip().lower()
    if mode not in {"single", "mean", "concat", "mix"}:
        raise ValueError(f"invalid feature mode: {mode}")
    if layer_text == "all":
        layers = tuple(all_layers)
    else:
        layers = tuple(int(part) for part in layer_text.split(",") if part.strip())
    if not layers:
        raise ValueError(f"empty feature set: {text}")
    if mode == "single" and len(layers) != 1:
        raise ValueError("single feature set must contain exactly one layer")
    missing = sorted(set(layers) - set(all_layers))
    if missing:
        raise ValueError(f"feature set {text!r} references unavailable layers: {missing}")
    name = f"{mode}_{'_'.join(f'{layer:02d}' for layer in layers)}"
    if len(layers) == len(all_layers) and tuple(layers) == tuple(all_layers):
        name = f"{mode}_all"
    return FeatureSet(name=name, mode=mode, layers=layers)


def parse_architecture(text: str) -> TransformerSpec:
    parts = text.split("_")
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
    return TransformerSpec(text, d_model, n_layers, n_heads, vals.get("dim_feedforward", 4 * d_model))


def stable_seed(*parts: object, base: int = 42) -> int:
    digest = hashlib.sha256("::".join(str(p) for p in parts).encode()).hexdigest()
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
    temp = df.iloc[idx][["decision_id", "prefix_token_index"]].copy()
    temp["_row_pos"] = idx
    temp = temp.sort_values(["decision_id", "prefix_token_index", "_row_pos"])
    return [group["_row_pos"].to_numpy(dtype=np.int64) for _, group in temp.groupby("decision_id", sort=False)]


def compute_scalers(layer_arrays: dict[int, np.ndarray], train_rows: np.ndarray, standardize: bool) -> dict[int, dict[str, np.ndarray]]:
    scalers = {}
    for layer, arr in layer_arrays.items():
        if not standardize:
            scalers[layer] = {
                "mean": np.zeros(arr.shape[1], dtype=np.float32),
                "scale": np.ones(arr.shape[1], dtype=np.float32),
            }
            continue
        x = np.asarray(arr[np.asarray(train_rows, dtype=np.int64)], dtype=np.float32)
        mean = x.mean(axis=0, dtype=np.float64).astype(np.float32)
        scale = x.std(axis=0, dtype=np.float64).astype(np.float32)
        scale = np.where(scale < 1e-8, 1.0, scale).astype(np.float32)
        scalers[layer] = {"mean": mean, "scale": scale}
    return scalers


def transform_layer(arr: np.ndarray, rows: np.ndarray, scaler: dict[str, np.ndarray], l2_normalize: bool) -> np.ndarray:
    x = np.asarray(arr[np.asarray(rows, dtype=np.int64)], dtype=np.float32)
    x = (x - scaler["mean"]) / scaler["scale"]
    if l2_normalize:
        norm = np.linalg.norm(x, axis=1, keepdims=True)
        x = x / np.where(norm < 1e-8, 1.0, norm)
    return x.astype(np.float32)


class DynamicFeatureSequenceDataset:
    def __init__(
        self,
        *,
        df: pd.DataFrame,
        sequences: list[np.ndarray],
        feature_set: FeatureSet,
        layer_arrays: dict[int, np.ndarray],
        scalers: dict[int, dict[str, np.ndarray]],
        l2_normalize: bool,
    ) -> None:
        self.df = df
        self.sequences = sequences
        self.feature_set = feature_set
        self.layer_arrays = layer_arrays
        self.scalers = scalers
        self.l2_normalize = l2_normalize
        self.y_by_row = action_labels_to_indices(df["top_action"])

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, i: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        rows = self.sequences[i]
        xs = [
            transform_layer(self.layer_arrays[layer], rows, self.scalers[layer], self.l2_normalize)
            for layer in self.feature_set.layers
        ]
        if self.feature_set.mode == "single":
            x = xs[0]
        elif self.feature_set.mode == "mean":
            x = np.mean(np.stack(xs, axis=0), axis=0).astype(np.float32)
        elif self.feature_set.mode == "concat":
            x = np.concatenate(xs, axis=1).astype(np.float32)
        elif self.feature_set.mode == "mix":
            x = np.stack(xs, axis=1).astype(np.float32)  # T, L, H
        else:
            raise ValueError(self.feature_set.mode)
        return x, self.y_by_row[rows].astype(np.int64), rows.astype(np.int64)


def collate_sequences(batch):
    import torch

    max_len = max(item[0].shape[0] for item in batch)
    first = batch[0][0]
    if first.ndim == 2:
        x_shape = (len(batch), max_len, first.shape[1])
    else:
        x_shape = (len(batch), max_len, first.shape[1], first.shape[2])
    x = np.zeros(x_shape, dtype=np.float32)
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


def build_model(input_dim: int, feature_set: FeatureSet, spec: TransformerSpec, max_seq_len: int, dropout: float):
    import torch
    import torch.nn as nn

    class FeatureSetTransformer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.is_mix = feature_set.mode == "mix"
            if self.is_mix:
                self.layer_logits = nn.Parameter(torch.zeros(len(feature_set.layers)))
            self.input_proj = nn.Sequential(nn.LayerNorm(input_dim), nn.Linear(input_dim, spec.d_model))
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

        def forward(self, x, padding_mask, *, causal: bool = True):
            if self.is_mix:
                weights = torch.softmax(self.layer_logits, dim=0).to(x.dtype)
                x = torch.sum(x * weights.view(1, 1, -1, 1), dim=2)
            length = x.shape[1]
            h = self.input_proj(x) + self.pos_embedding[:, :length]
            causal_mask = torch.triu(torch.ones((length, length), dtype=torch.bool, device=x.device), diagonal=1) if causal else None
            h = self.encoder(h, mask=causal_mask, src_key_padding_mask=padding_mask)
            return self.classifier(self.norm(h))

    return FeatureSetTransformer()


def prediction_coverage(y_pred: np.ndarray, y_true: np.ndarray) -> dict[str, Any]:
    pred_counts = {action: int(np.sum(y_pred == i)) for i, action in enumerate(ACTIONS)}
    true_counts = {action: int(np.sum(y_true == i)) for i, action in enumerate(ACTIONS)}
    present = [action for i, action in enumerate(ACTIONS) if true_counts[action] > 0]
    missing = [action for action in present if pred_counts[action] == 0]
    return {"pred_counts": pred_counts, "true_counts": true_counts, "missing_actions": missing, "num_predicted_actions": int(sum(v > 0 for v in pred_counts.values())), "has_action_collapse": bool(missing)}


def confusion_matrix_json(y_true: np.ndarray, y_pred: np.ndarray) -> str:
    matrix = {}
    for i, actual in enumerate(ACTIONS):
        matrix[actual] = {}
        for j, predicted in enumerate(ACTIONS):
            matrix[actual][predicted] = int(np.sum((y_true == i) & (y_pred == j)))
    return json.dumps(matrix, sort_keys=True)


def metrics_from_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    true_labels = action_indices_to_labels(y_true)
    pred_labels = action_indices_to_labels(y_pred)
    coverage = prediction_coverage(y_pred, y_true)
    row = {
        "accuracy": float(np.mean(y_true == y_pred)),
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


def predict(model, dataset, batch_size: int, device, *, causal: bool = True) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    import torch
    from torch.utils.data import DataLoader

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_sequences)
    pred_by_row: dict[int, int] = {}
    true_by_row: dict[int, int] = {}
    model.eval()
    with torch.no_grad():
        for xb, yb, pad_mask, rows in loader:
            logits = model(xb.to(device), pad_mask.to(device), causal=causal)
            pred = logits.argmax(dim=-1).detach().cpu().numpy()
            y_np = yb.numpy()
            rows_np = rows.numpy()
            valid = rows_np >= 0
            for row, yy, pp in zip(rows_np[valid].reshape(-1), y_np[valid].reshape(-1), pred[valid].reshape(-1)):
                pred_by_row[int(row)] = int(pp)
                true_by_row[int(row)] = int(yy)
    ordered = np.asarray(sorted(pred_by_row), dtype=np.int64)
    return ordered, np.asarray([true_by_row[int(r)] for r in ordered], dtype=np.int64), np.asarray([pred_by_row[int(r)] for r in ordered], dtype=np.int64)


def train_one(train_ds, val_ds, input_dim: int, feature_set: FeatureSet, spec: TransformerSpec, max_seq_len: int, args, device, *, seed: int, balanced_loss: bool, causal: bool):
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader

    torch.manual_seed(seed)
    np.random.seed(seed)
    model = build_model(input_dim, feature_set, spec, max_seq_len, args.dropout).to(device)
    y_train = np.concatenate([train_ds.y_by_row[seq] for seq in train_ds.sequences])
    weights = torch.tensor(class_weights(y_train), dtype=torch.float32, device=device) if balanced_loss else None
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    generator = torch.Generator()
    generator.manual_seed(seed)
    loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, generator=generator, collate_fn=collate_sequences)
    best_state = None
    best_score = -float("inf")
    best_epoch = -1
    best_info = {}
    bad_epochs = 0
    for epoch in range(int(args.max_epochs)):
        model.train()
        for xb, yb, pad_mask, _ in loader:
            logits = model(xb.to(device), pad_mask.to(device), causal=causal)
            loss = F.cross_entropy(logits.reshape(-1, len(ACTIONS)), yb.to(device).reshape(-1), ignore_index=-100, weight=weights)
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
            best_info = {"validation_accuracy": val_m["accuracy"], "validation_balanced_accuracy": val_m["balanced_accuracy"], "validation_macro_f1": val_m["macro_f1"], "validation_missing_actions_json": val_m["missing_actions_json"]}
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, {"best_epoch": best_epoch, "validation_score": best_score, **best_info}


def evaluate_row(df, row_ids, y_true, y_pred, split, feature_set, spec, attention_mode, loss_weighting, train_n, val_n, train_seq_n, val_seq_n, test_seq_n, train_info, args):
    top_m = metrics_from_predictions(y_true, y_pred)
    y_agent = action_labels_to_indices(df.loc[row_ids, "agent_action"])
    agent_m = metrics_from_predictions(y_agent, y_pred)
    row = {
        "run_id": f"fsseq_{feature_set.name}_{spec.name}_{loss_weighting}_split{split.split_id}_{int(time.time())}",
        "split_id": int(split.split_id),
        "representation": args.representation,
        "subset": args.subset,
        "feature_set": feature_set.name,
        "feature_mode": feature_set.mode,
        "feature_layers_json": json.dumps(list(feature_set.layers)),
        "n_feature_layers": len(feature_set.layers),
        "architecture": spec.name,
        "d_model": spec.d_model,
        "n_layers": spec.n_layers,
        "n_heads": spec.n_heads,
        "attention_mode": attention_mode,
        "loss_weighting": loss_weighting,
        "n_train": int(train_n),
        "n_val": int(val_n),
        "n_test": int(len(row_ids)),
        "n_train_sequences": int(train_seq_n),
        "n_val_sequences": int(val_seq_n),
        "n_test_sequences": int(test_seq_n),
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
    }
    for action in ACTIONS:
        row[f"test_true_top_count_{action}"] = top_m[f"true_count_{action}"]
        row[f"test_pred_top_count_{action}"] = top_m[f"pred_count_{action}"]
        row[f"test_true_agent_count_{action}"] = agent_m[f"true_count_{action}"]
    return row


def plot_summary(metrics: pd.DataFrame, figure_dir: Path) -> None:
    import matplotlib.pyplot as plt

    ensure_dir(figure_dir)
    summary = (
        metrics.groupby(["feature_set", "architecture", "attention_mode", "loss_weighting"], as_index=False)
        .agg(mean_bal=("test_top_action_balanced_accuracy", "mean"), mean_acc=("test_top_action_accuracy", "mean"))
        .sort_values("mean_bal", ascending=False)
    )
    labels = [f"{r.feature_set}\n{r.architecture}/{r.attention_mode}" for r in summary.itertuples()]
    x = np.arange(len(summary))
    fig, ax = plt.subplots(figsize=(max(10, len(summary) * 0.45), 5))
    ax.bar(x, summary["mean_bal"].to_numpy(), label="balanced accuracy", color="C0")
    ax.bar(x, summary["mean_acc"].to_numpy(), label="accuracy", color="C1", alpha=0.45)
    ax.axhline(0.25, color="black", linestyle="--", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=70, ha="right", fontsize=8)
    ax.set_ylabel("test score")
    ax.set_title("Causal Transformer top-action prediction by feature set")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figure_dir / "feature_set_top_action_accuracy.png", dpi=200)
    plt.close(fig)


def write_report(metrics: pd.DataFrame, output_dir: Path, figure_dir: Path, args) -> None:
    summary = (
        metrics.groupby(["feature_set", "feature_mode", "architecture", "attention_mode", "loss_weighting"], as_index=False)
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
    summary.to_csv(output_dir / "feature_set_sequence_transformer_summary.csv", index=False)
    best = summary.iloc[0].to_dict() if not summary.empty else {}
    lines = [
        "# Sequence Transformer Feature-Set Sweep",
        "",
        f"- representation: `{args.representation}`",
        f"- subset: `{args.subset}`",
        f"- feature sets: `{', '.join(args.feature_sets)}`",
        f"- architectures: `{', '.join(args.architectures)}`",
        f"- attention modes: `{', '.join(args.attention_modes)}`",
        "",
        "## Best Mean Result",
        "",
    ]
    if best:
        for key in ["feature_set", "feature_mode", "architecture", "attention_mode", "loss_weighting", "splits", "mean_acc", "mean_bal", "mean_f1", "mean_agent_acc", "collapse_rows"]:
            val = best[key]
            lines.append(f"- {key}: {val:.6f}" if isinstance(val, float) else f"- {key}: `{val}`")
    lines.extend(["", "## Summary", "", summary.to_markdown(index=False), "", "## Figure", "", f"- `{figure_dir / 'feature_set_top_action_accuracy.png'}`"])
    save_markdown(output_dir / "feature_set_sequence_transformer_report.md", "\n".join(lines))
    if best:
        write_json(output_dir / "best_feature_set_sequence_transformer_config.json", json_ready(best))


def main() -> None:
    args = parse_args()
    output_dir = ensure_dir(args.output_dir)
    figure_dir = ensure_dir(args.figure_dir)
    metrics_path = output_dir / "feature_set_sequence_transformer_metrics.csv"
    if metrics_path.exists() and not args.force:
        raise SystemExit(f"{metrics_path} exists; pass --force to overwrite")
    import torch

    device = choose_device(args.device)
    df = make_behavior_table(args.beliefs).reset_index(drop=True)
    sub_idx = np.flatnonzero(subset_mask(df, args.subset))
    feature_dir = Path(args.feature_dir)
    all_layers = available_layers(feature_dir, args.representation)
    if not all_layers:
        raise ValueError(f"no layers found for representation {args.representation}")
    feature_sets = [parse_feature_set(text, all_layers) for text in args.feature_sets]
    needed_layers = sorted(set(layer for fs in feature_sets for layer in fs.layers))
    layer_arrays = {layer: np.load(feature_dir / f"layer_{layer:02d}_{args.representation}.npy", mmap_mode="r") for layer in needed_layers}
    n_rows = len(df)
    hidden_dim = next(iter(layer_arrays.values())).shape[1]
    for layer, arr in layer_arrays.items():
        if arr.shape[0] != n_rows:
            raise ValueError(f"layer {layer} rows {arr.shape[0]} != behavior rows {n_rows}")
    specs = [parse_architecture(text) for text in args.architectures]
    loss_weightings = ["plain", "balanced"] if args.loss_weighting == "both" else [args.loss_weighting]
    splits = load_or_create_splits(args.splits, df, args.group_col, seed=args.random_seed)
    if args.limit_splits is not None:
        splits = splits[: int(args.limit_splits)]
    print(json.dumps({"device": str(device), "available_layers": all_layers, "feature_sets": [fs.name for fs in feature_sets], "architectures": [s.name for s in specs], "attention_modes": args.attention_modes, "loss_weightings": loss_weightings, "subset_rows": int(len(sub_idx))}, indent=2))

    metrics: list[dict[str, Any]] = []
    for split in splits:
        train_pool = np.intersect1d(split.train_indices, sub_idx, assume_unique=False)
        test_rows = np.intersect1d(split.test_indices, sub_idx, assume_unique=False)
        train_rows, val_rows = make_train_val_split(df, train_pool, split.group_col, seed=args.random_seed + split.split_id)
        train_seqs = make_sequences(df, train_rows)
        val_seqs = make_sequences(df, val_rows)
        test_seqs = make_sequences(df, test_rows)
        max_seq_len = max(max(len(seq) for seq in train_seqs), max(len(seq) for seq in val_seqs), max(len(seq) for seq in test_seqs))
        print(f"split={split.split_id} train_rows={len(train_rows)} val_rows={len(val_rows)} test_rows={len(test_rows)} max_seq_len={max_seq_len}")
        for feature_set in feature_sets:
            print(f"Preparing feature_set={feature_set.name}")
            selected_arrays = {layer: layer_arrays[layer] for layer in feature_set.layers}
            scalers = compute_scalers(selected_arrays, train_rows, args.standardize)
            train_ds = DynamicFeatureSequenceDataset(df=df, sequences=train_seqs, feature_set=feature_set, layer_arrays=selected_arrays, scalers=scalers, l2_normalize=args.l2_normalize)
            val_ds = DynamicFeatureSequenceDataset(df=df, sequences=val_seqs, feature_set=feature_set, layer_arrays=selected_arrays, scalers=scalers, l2_normalize=args.l2_normalize)
            test_ds = DynamicFeatureSequenceDataset(df=df, sequences=test_seqs, feature_set=feature_set, layer_arrays=selected_arrays, scalers=scalers, l2_normalize=args.l2_normalize)
            if feature_set.mode == "concat":
                input_dim = hidden_dim * len(feature_set.layers)
            else:
                input_dim = hidden_dim
            for spec in specs:
                for attention_mode in args.attention_modes:
                    causal = attention_mode == "causal"
                    for weighting in loss_weightings:
                        seed = stable_seed(feature_set.name, spec.name, attention_mode, weighting, split.split_id, base=args.random_seed)
                        print(f"Training split={split.split_id} feature_set={feature_set.name} arch={spec.name} attention={attention_mode} weighting={weighting}")
                        model, info = train_one(train_ds, val_ds, input_dim, feature_set, spec, max_seq_len, args, device, seed=seed, balanced_loss=weighting == "balanced", causal=causal)
                        row_ids, y_true, y_pred = predict(model, test_ds, args.batch_size, device, causal=causal)
                        metrics.append(evaluate_row(df, row_ids, y_true, y_pred, split, feature_set, spec, attention_mode, weighting, len(train_rows), len(val_rows), len(train_ds), len(val_ds), len(test_ds), info, args))
                        del model
                        gc.collect()
                        if str(device) == "mps" and hasattr(torch, "mps"):
                            torch.mps.empty_cache()
            del train_ds, val_ds, test_ds, scalers
            gc.collect()
    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv(metrics_path, index=False)
    plot_summary(metrics_df, Path(figure_dir))
    write_report(metrics_df, Path(output_dir), Path(figure_dir), args)
    print(json.dumps({"metrics": str(metrics_path), "summary": str(output_dir / "feature_set_sequence_transformer_summary.csv"), "report": str(output_dir / "feature_set_sequence_transformer_report.md"), "figure_dir": str(figure_dir), "rows": int(len(metrics_df))}, indent=2))


if __name__ == "__main__":
    main()
