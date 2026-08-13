#!/usr/bin/env python
"""Sweep multitask sequence Transformers for top-action prediction.

Feature-set syntax:
  single:10
  mean:0,2,4,6
  mean:all
  concat:8,10,12,14
  mix:all
  attn:all

This extends the feature-set Transformer with optional auxiliary losses:
agent-action classification and soft q-distribution matching. The reported
metric is still held-out top_action prediction. Auxiliary targets are not used
as input features.
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
    Q_COLS,
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
    model_type: str = "transformer"
    rnn_bidirectional: bool = False


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
    parser.add_argument("--top_loss_weight", type=float, default=1.0)
    parser.add_argument("--agent_loss_weight", type=float, default=0.0)
    parser.add_argument("--q_loss_weight", type=float, default=0.0)
    parser.add_argument("--q_loss_kind", choices=["kl", "q_mse", "clr_mse"], default="kl")
    parser.add_argument("--add_metadata_features", action="store_true")
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
    if mode not in {"single", "mean", "concat", "mix", "attn"}:
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
    if text.startswith(("gru_", "bigru_")):
        vals: dict[str, int] = {}
        for part in text.split("_")[1:]:
            if part.startswith("h"):
                vals["hidden"] = int(part[1:])
            elif part.startswith("l"):
                vals["n_layers"] = int(part[1:])
        hidden = vals.get("hidden", 256)
        n_layers = vals.get("n_layers", 2)
        return TransformerSpec(
            text,
            hidden,
            n_layers,
            1,
            4 * hidden,
            model_type="gru",
            rnn_bidirectional=text.startswith("bigru_"),
        )
    if text.startswith("tcn_"):
        vals: dict[str, int] = {}
        for part in text.split("_")[1:]:
            if part.startswith("c"):
                vals["hidden"] = int(part[1:])
            elif part.startswith("l"):
                vals["n_layers"] = int(part[1:])
            elif part.startswith("k"):
                vals["kernel"] = int(part[1:])
        hidden = vals.get("hidden", 256)
        n_layers = vals.get("n_layers", 4)
        kernel = vals.get("kernel", 3)
        return TransformerSpec(
            text,
            hidden,
            n_layers,
            kernel,
            4 * hidden,
            model_type="tcn",
            rnn_bidirectional=False,
        )
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


META_COLS = ["progress", "prefix_token_index", "num_cot_tokens", "prefix_fraction", "log_prefix_fraction"]


def raw_metadata_features(df: pd.DataFrame, rows: np.ndarray) -> np.ndarray:
    part = df.iloc[np.asarray(rows, dtype=np.int64)]
    progress = pd.to_numeric(part.get("progress", 0.0), errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
    prefix = pd.to_numeric(part.get("prefix_token_index", 0.0), errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
    n_tokens = pd.to_numeric(part.get("num_cot_tokens", np.nan), errors="coerce").fillna(np.nan).to_numpy(dtype=np.float32)
    n_tokens = np.where(np.isfinite(n_tokens) & (n_tokens > 0), n_tokens, np.nanmax(np.where(prefix > 0, prefix, 1.0)))
    n_tokens = np.where(n_tokens > 0, n_tokens, 1.0).astype(np.float32)
    prefix_fraction = np.clip(prefix / n_tokens, 0.0, 2.0).astype(np.float32)
    log_prefix_fraction = (
        np.log1p(np.maximum(prefix, 0.0)) / np.maximum(np.log1p(np.maximum(n_tokens, 1.0)), 1e-6)
    ).astype(np.float32)
    return np.stack([progress, prefix, n_tokens, prefix_fraction, log_prefix_fraction], axis=1).astype(np.float32)


def compute_meta_scaler(df: pd.DataFrame, train_rows: np.ndarray, enabled: bool) -> dict[str, np.ndarray]:
    if not enabled:
        return {
            "mean": np.zeros(0, dtype=np.float32),
            "scale": np.ones(0, dtype=np.float32),
        }
    x = raw_metadata_features(df, train_rows)
    mean = x.mean(axis=0, dtype=np.float64).astype(np.float32)
    scale = x.std(axis=0, dtype=np.float64).astype(np.float32)
    scale = np.where(scale < 1e-8, 1.0, scale).astype(np.float32)
    return {"mean": mean, "scale": scale}


def transform_metadata(df: pd.DataFrame, rows: np.ndarray, scaler: dict[str, np.ndarray], enabled: bool) -> np.ndarray:
    if not enabled:
        return np.zeros((len(rows), 0), dtype=np.float32)
    x = raw_metadata_features(df, rows)
    return ((x - scaler["mean"]) / scaler["scale"]).astype(np.float32)


class DynamicFeatureSequenceDataset:
    def __init__(
        self,
        *,
        df: pd.DataFrame,
        sequences: list[np.ndarray],
        feature_set: FeatureSet,
        layer_arrays: dict[int, np.ndarray],
        scalers: dict[int, dict[str, np.ndarray]],
        meta_scaler: dict[str, np.ndarray],
        add_metadata_features: bool,
        l2_normalize: bool,
    ) -> None:
        self.df = df
        self.sequences = sequences
        self.feature_set = feature_set
        self.layer_arrays = layer_arrays
        self.scalers = scalers
        self.meta_scaler = meta_scaler
        self.add_metadata_features = add_metadata_features
        self.l2_normalize = l2_normalize
        self.y_top_by_row = action_labels_to_indices(df["top_action"])
        self.y_agent_by_row = action_labels_to_indices(df["agent_action"])
        self.q_by_row = df[Q_COLS].to_numpy(dtype=np.float32)

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
        elif self.feature_set.mode in {"mix", "attn"}:
            x = np.stack(xs, axis=1).astype(np.float32)  # T, L, H
        else:
            raise ValueError(self.feature_set.mode)
        meta = transform_metadata(self.df, rows, self.meta_scaler, self.add_metadata_features)
        return (
            x,
            meta,
            self.y_top_by_row[rows].astype(np.int64),
            self.y_agent_by_row[rows].astype(np.int64),
            self.q_by_row[rows].astype(np.float32),
            rows.astype(np.int64),
        )


def collate_sequences(batch):
    import torch

    max_len = max(item[0].shape[0] for item in batch)
    first = batch[0][0]
    if first.ndim == 2:
        x_shape = (len(batch), max_len, first.shape[1])
    else:
        x_shape = (len(batch), max_len, first.shape[1], first.shape[2])
    x = np.zeros(x_shape, dtype=np.float32)
    meta_dim = batch[0][1].shape[1]
    meta = np.zeros((len(batch), max_len, meta_dim), dtype=np.float32)
    y_top = np.full((len(batch), max_len), -100, dtype=np.int64)
    y_agent = np.full((len(batch), max_len), -100, dtype=np.int64)
    q = np.zeros((len(batch), max_len, len(ACTIONS)), dtype=np.float32)
    mask = np.ones((len(batch), max_len), dtype=bool)
    rows = np.full((len(batch), max_len), -1, dtype=np.int64)
    for i, (seq_x, seq_meta, seq_y_top, seq_y_agent, seq_q, seq_rows) in enumerate(batch):
        n = seq_x.shape[0]
        x[i, :n] = seq_x
        meta[i, :n] = seq_meta
        y_top[i, :n] = seq_y_top
        y_agent[i, :n] = seq_y_agent
        q[i, :n] = seq_q
        mask[i, :n] = False
        rows[i, :n] = seq_rows
    return torch.tensor(x), torch.tensor(meta), torch.tensor(y_top), torch.tensor(y_agent), torch.tensor(q), torch.tensor(mask), torch.tensor(rows)


def build_model(input_dim: int, meta_dim: int, feature_set: FeatureSet, spec: TransformerSpec, max_seq_len: int, dropout: float):
    import torch
    import torch.nn as nn

    class FeatureInputMixin:
        def mix_features(self, x, meta):
            if self.is_mix:
                weights = torch.softmax(self.layer_logits, dim=0).to(x.dtype)
                x = torch.sum(x * weights.view(1, 1, -1, 1), dim=2)
            elif self.is_attn:
                scores = torch.einsum("btlh,h->btl", x, self.layer_query.to(x.dtype)) / (float(x.shape[-1]) ** 0.5)
                scores = scores + self.layer_bias.view(1, 1, -1).to(x.dtype)
                weights = torch.softmax(scores, dim=2)
                x = torch.sum(x * weights.unsqueeze(-1), dim=2)
            if meta.shape[-1] > 0:
                x = torch.cat([x, meta.to(dtype=x.dtype)], dim=-1)
            return x

    class FeatureSetTransformer(FeatureInputMixin, nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.is_mix = feature_set.mode == "mix"
            self.is_attn = feature_set.mode == "attn"
            if self.is_mix:
                self.layer_logits = nn.Parameter(torch.zeros(len(feature_set.layers)))
            if self.is_attn:
                self.layer_query = nn.Parameter(torch.empty(input_dim))
                nn.init.normal_(self.layer_query, mean=0.0, std=0.02)
                self.layer_bias = nn.Parameter(torch.zeros(len(feature_set.layers)))
            self.input_proj = nn.Sequential(nn.LayerNorm(input_dim + meta_dim), nn.Linear(input_dim + meta_dim, spec.d_model))
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
            self.top_classifier = nn.Linear(spec.d_model, len(ACTIONS))
            self.agent_classifier = nn.Linear(spec.d_model, len(ACTIONS))

        def forward(self, x, meta, padding_mask, *, causal: bool = True):
            x = self.mix_features(x, meta)
            length = x.shape[1]
            h = self.input_proj(x) + self.pos_embedding[:, :length]
            causal_mask = torch.triu(torch.ones((length, length), dtype=torch.bool, device=x.device), diagonal=1) if causal else None
            h = self.encoder(h, mask=causal_mask, src_key_padding_mask=padding_mask)
            h = self.norm(h)
            return {"top": self.top_classifier(h), "agent": self.agent_classifier(h)}

    class FeatureSetGRU(FeatureInputMixin, nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.is_mix = feature_set.mode == "mix"
            self.is_attn = feature_set.mode == "attn"
            if self.is_mix:
                self.layer_logits = nn.Parameter(torch.zeros(len(feature_set.layers)))
            if self.is_attn:
                self.layer_query = nn.Parameter(torch.empty(input_dim))
                nn.init.normal_(self.layer_query, mean=0.0, std=0.02)
                self.layer_bias = nn.Parameter(torch.zeros(len(feature_set.layers)))
            self.input_proj = nn.Sequential(nn.LayerNorm(input_dim + meta_dim), nn.Linear(input_dim + meta_dim, spec.d_model), nn.GELU())
            self.gru = nn.GRU(
                input_size=spec.d_model,
                hidden_size=spec.d_model,
                num_layers=spec.n_layers,
                dropout=dropout if spec.n_layers > 1 else 0.0,
                batch_first=True,
                bidirectional=spec.rnn_bidirectional,
            )
            out_dim = spec.d_model * (2 if spec.rnn_bidirectional else 1)
            self.norm = nn.LayerNorm(out_dim)
            self.top_classifier = nn.Linear(out_dim, len(ACTIONS))
            self.agent_classifier = nn.Linear(out_dim, len(ACTIONS))

        def forward(self, x, meta, padding_mask, *, causal: bool = True):
            x = self.mix_features(x, meta)
            h = self.input_proj(x)
            lengths = (~padding_mask).sum(dim=1).detach().cpu()
            packed = nn.utils.rnn.pack_padded_sequence(h, lengths, batch_first=True, enforce_sorted=False)
            packed_out, _ = self.gru(packed)
            h, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True, total_length=x.shape[1])
            h = self.norm(h)
            return {"top": self.top_classifier(h), "agent": self.agent_classifier(h)}

    class TCNBlock(nn.Module):
        def __init__(self, channels: int, kernel_size: int, dilation: int, dropout: float) -> None:
            super().__init__()
            self.kernel_size = kernel_size
            self.dilation = dilation
            self.conv1 = nn.Conv1d(channels, channels, kernel_size, dilation=dilation)
            self.conv2 = nn.Conv1d(channels, channels, kernel_size, dilation=dilation)
            self.dropout = nn.Dropout(dropout)
            self.norm1 = nn.LayerNorm(channels)
            self.norm2 = nn.LayerNorm(channels)

        def _conv_same(self, conv, x, *, causal: bool):
            import torch.nn.functional as F

            pad_total = (self.kernel_size - 1) * self.dilation
            if causal:
                x = F.pad(x, (pad_total, 0))
            else:
                left = pad_total // 2
                right = pad_total - left
                x = F.pad(x, (left, right))
            return conv(x)

        def forward(self, x, *, causal: bool):
            residual = x
            y = self._conv_same(self.conv1, x.transpose(1, 2), causal=causal).transpose(1, 2)
            y = self.dropout(torch.nn.functional.gelu(self.norm1(y)))
            y = self._conv_same(self.conv2, y.transpose(1, 2), causal=causal).transpose(1, 2)
            y = self.dropout(torch.nn.functional.gelu(self.norm2(y)))
            return residual + y

    class FeatureSetTCN(FeatureInputMixin, nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.is_mix = feature_set.mode == "mix"
            self.is_attn = feature_set.mode == "attn"
            if self.is_mix:
                self.layer_logits = nn.Parameter(torch.zeros(len(feature_set.layers)))
            if self.is_attn:
                self.layer_query = nn.Parameter(torch.empty(input_dim))
                nn.init.normal_(self.layer_query, mean=0.0, std=0.02)
                self.layer_bias = nn.Parameter(torch.zeros(len(feature_set.layers)))
            self.input_proj = nn.Sequential(nn.LayerNorm(input_dim + meta_dim), nn.Linear(input_dim + meta_dim, spec.d_model), nn.GELU())
            kernel_size = int(spec.n_heads)
            self.blocks = nn.ModuleList(
                [TCNBlock(spec.d_model, kernel_size, dilation=2**i, dropout=dropout) for i in range(spec.n_layers)]
            )
            self.norm = nn.LayerNorm(spec.d_model)
            self.top_classifier = nn.Linear(spec.d_model, len(ACTIONS))
            self.agent_classifier = nn.Linear(spec.d_model, len(ACTIONS))

        def forward(self, x, meta, padding_mask, *, causal: bool = True):
            h = self.input_proj(self.mix_features(x, meta))
            for block in self.blocks:
                h = block(h, causal=causal)
            h = self.norm(h)
            return {"top": self.top_classifier(h), "agent": self.agent_classifier(h)}

    if spec.model_type == "gru":
        return FeatureSetGRU()
    if spec.model_type == "tcn":
        return FeatureSetTCN()
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
    ordered, y_true, y_pred, _probs = predict_with_probs(model, dataset, batch_size, device, causal=causal)
    return ordered, y_true, y_pred


def predict_with_probs(model, dataset, batch_size: int, device, *, causal: bool = True) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    import torch
    from torch.utils.data import DataLoader

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_sequences)
    pred_by_row: dict[int, int] = {}
    true_by_row: dict[int, int] = {}
    prob_by_row: dict[int, np.ndarray] = {}
    model.eval()
    with torch.no_grad():
        for xb, meta, yb, _agent, _q, pad_mask, rows in loader:
            outputs = model(xb.to(device), meta.to(device), pad_mask.to(device), causal=causal)
            logits = outputs["top"]
            probs = torch.softmax(logits, dim=-1).detach().cpu().numpy()
            pred = logits.argmax(dim=-1).detach().cpu().numpy()
            y_np = yb.numpy()
            rows_np = rows.numpy()
            valid = rows_np >= 0
            for row, yy, pp, prob in zip(rows_np[valid].reshape(-1), y_np[valid].reshape(-1), pred[valid].reshape(-1), probs[valid].reshape(-1, len(ACTIONS))):
                pred_by_row[int(row)] = int(pp)
                true_by_row[int(row)] = int(yy)
                prob_by_row[int(row)] = np.asarray(prob, dtype=np.float32)
    ordered = np.asarray(sorted(pred_by_row), dtype=np.int64)
    return (
        ordered,
        np.asarray([true_by_row[int(r)] for r in ordered], dtype=np.int64),
        np.asarray([pred_by_row[int(r)] for r in ordered], dtype=np.int64),
        np.stack([prob_by_row[int(r)] for r in ordered], axis=0).astype(np.float32),
    )


def train_one(train_ds, val_ds, input_dim: int, feature_set: FeatureSet, spec: TransformerSpec, max_seq_len: int, args, device, *, seed: int, balanced_loss: bool, causal: bool):
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader

    torch.manual_seed(seed)
    np.random.seed(seed)
    meta_dim = int(train_ds.meta_scaler["mean"].shape[0]) if train_ds.add_metadata_features else 0
    model = build_model(input_dim, meta_dim, feature_set, spec, max_seq_len, args.dropout).to(device)
    y_top_train = np.concatenate([train_ds.y_top_by_row[seq] for seq in train_ds.sequences])
    y_agent_train = np.concatenate([train_ds.y_agent_by_row[seq] for seq in train_ds.sequences])
    top_weights = torch.tensor(class_weights(y_top_train), dtype=torch.float32, device=device) if balanced_loss else None
    agent_weights = torch.tensor(class_weights(y_agent_train), dtype=torch.float32, device=device) if balanced_loss else None
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
        for xb, meta, y_top, y_agent, q_target, pad_mask, _ in loader:
            y_top_d = y_top.to(device)
            y_agent_d = y_agent.to(device)
            q_target_d = q_target.to(device)
            outputs = model(xb.to(device), meta.to(device), pad_mask.to(device), causal=causal)
            top_logits = outputs["top"]
            agent_logits = outputs["agent"]
            loss = torch.zeros((), dtype=top_logits.dtype, device=device)
            if args.top_loss_weight > 0:
                top_ce = F.cross_entropy(
                    top_logits.reshape(-1, len(ACTIONS)),
                    y_top_d.reshape(-1),
                    ignore_index=-100,
                    weight=top_weights,
                )
                loss = loss + float(args.top_loss_weight) * top_ce
            if args.agent_loss_weight > 0:
                agent_ce = F.cross_entropy(
                    agent_logits.reshape(-1, len(ACTIONS)),
                    y_agent_d.reshape(-1),
                    ignore_index=-100,
                    weight=agent_weights,
                )
                loss = loss + float(args.agent_loss_weight) * agent_ce
            if args.q_loss_weight > 0:
                flat_valid = y_top_d.reshape(-1) != -100
                flat_logits = top_logits.reshape(-1, len(ACTIONS))[flat_valid]
                q_flat = q_target_d.reshape(-1, len(ACTIONS))[flat_valid]
                if args.q_loss_kind == "kl":
                    q_loss = F.kl_div(F.log_softmax(flat_logits, dim=-1), q_flat, reduction="batchmean")
                elif args.q_loss_kind == "q_mse":
                    q_loss = F.mse_loss(F.softmax(flat_logits, dim=-1), q_flat)
                elif args.q_loss_kind == "clr_mse":
                    log_q = torch.log(torch.clamp(q_flat, min=1e-8))
                    target_clr = log_q - log_q.mean(dim=-1, keepdim=True)
                    pred_clr = flat_logits - flat_logits.mean(dim=-1, keepdim=True)
                    q_loss = F.mse_loss(pred_clr, target_clr)
                else:
                    raise ValueError(args.q_loss_kind)
                loss = loss + float(args.q_loss_weight) * q_loss
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
        "top_loss_weight": float(args.top_loss_weight),
        "agent_loss_weight": float(args.agent_loss_weight),
        "q_loss_weight": float(args.q_loss_weight),
        "q_loss_kind": args.q_loss_kind,
        "add_metadata_features": bool(args.add_metadata_features),
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


def make_prediction_frame(
    row_ids: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    *,
    split_id: int,
    split_role: str,
    feature_set: FeatureSet,
    spec: TransformerSpec,
    attention_mode: str,
    loss_weighting: str,
    args,
) -> pd.DataFrame:
    pred_df = pd.DataFrame(
        {
            "split_id": int(split_id),
            "split_role": split_role,
            "row_id": row_ids.astype(np.int64),
            "true_top_action": action_indices_to_labels(y_true),
            "pred_top_action": action_indices_to_labels(y_pred),
            "feature_set": feature_set.name,
            "architecture": spec.name,
            "attention_mode": attention_mode,
            "loss_weighting": loss_weighting,
            "top_loss_weight": float(args.top_loss_weight),
            "agent_loss_weight": float(args.agent_loss_weight),
            "q_loss_weight": float(args.q_loss_weight),
            "q_loss_kind": args.q_loss_kind,
            "add_metadata_features": bool(args.add_metadata_features),
        }
    )
    for i, action in enumerate(ACTIONS):
        pred_df[f"prob_{action}"] = y_prob[:, i]
    return pred_df


def plot_summary(metrics: pd.DataFrame, figure_dir: Path) -> None:
    import matplotlib.pyplot as plt

    ensure_dir(figure_dir)
    summary = (
        metrics.groupby(
            [
                "feature_set",
                "architecture",
                "attention_mode",
                "loss_weighting",
                "top_loss_weight",
                "agent_loss_weight",
                "q_loss_weight",
                "q_loss_kind",
                "add_metadata_features",
            ],
            as_index=False,
        )
        .agg(mean_bal=("test_top_action_balanced_accuracy", "mean"), mean_acc=("test_top_action_accuracy", "mean"))
        .sort_values("mean_bal", ascending=False)
    )
    labels = [
        f"{r.feature_set}\n{r.architecture}/{r.attention_mode}\na{r.agent_loss_weight:g}/q{r.q_loss_weight:g}/m{int(r.add_metadata_features)}"
        for r in summary.itertuples()
    ]
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
        metrics.groupby(
            [
                "feature_set",
                "feature_mode",
                "architecture",
                "attention_mode",
                "loss_weighting",
                "top_loss_weight",
                "agent_loss_weight",
                "q_loss_weight",
                "q_loss_kind",
                "add_metadata_features",
            ],
            as_index=False,
        )
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
        f"- top loss weight: `{args.top_loss_weight}`",
        f"- agent loss weight: `{args.agent_loss_weight}`",
        f"- q loss weight: `{args.q_loss_weight}`",
        f"- q loss kind: `{args.q_loss_kind}`",
        f"- metadata features: `{args.add_metadata_features}`",
        "",
        "## Best Mean Result",
        "",
    ]
    if best:
        for key in [
            "feature_set",
            "feature_mode",
            "architecture",
            "attention_mode",
            "loss_weighting",
            "top_loss_weight",
            "agent_loss_weight",
            "q_loss_weight",
            "q_loss_kind",
            "add_metadata_features",
            "splits",
            "mean_acc",
            "mean_bal",
            "mean_f1",
            "mean_agent_acc",
            "collapse_rows",
        ]:
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
    print(
        json.dumps(
            {
                "device": str(device),
                "available_layers": all_layers,
                "feature_sets": [fs.name for fs in feature_sets],
                "architectures": [s.name for s in specs],
                "attention_modes": args.attention_modes,
                "loss_weightings": loss_weightings,
                "top_loss_weight": args.top_loss_weight,
                "agent_loss_weight": args.agent_loss_weight,
                "q_loss_weight": args.q_loss_weight,
                "q_loss_kind": args.q_loss_kind,
                "add_metadata_features": args.add_metadata_features,
                "metadata_columns": META_COLS if args.add_metadata_features else [],
                "subset_rows": int(len(sub_idx)),
            },
            indent=2,
        )
    )

    metrics: list[dict[str, Any]] = []
    prediction_frames: list[pd.DataFrame] = []
    validation_prediction_frames: list[pd.DataFrame] = []
    for split in splits:
        train_pool = np.intersect1d(split.train_indices, sub_idx, assume_unique=False)
        test_rows = np.intersect1d(split.test_indices, sub_idx, assume_unique=False)
        if len(train_pool) == 0 or len(test_rows) == 0:
            print(f"Skipping split={split.split_id} because train_rows={len(train_pool)} test_rows={len(test_rows)} after subset filtering")
            continue
        train_rows, val_rows = make_train_val_split(df, train_pool, split.group_col, seed=args.random_seed + split.split_id)
        if len(train_rows) == 0 or len(val_rows) == 0:
            print(f"Skipping split={split.split_id} because train/val split is empty after subset filtering")
            continue
        train_seqs = make_sequences(df, train_rows)
        val_seqs = make_sequences(df, val_rows)
        test_seqs = make_sequences(df, test_rows)
        max_seq_len = max(max(len(seq) for seq in train_seqs), max(len(seq) for seq in val_seqs), max(len(seq) for seq in test_seqs))
        print(f"split={split.split_id} train_rows={len(train_rows)} val_rows={len(val_rows)} test_rows={len(test_rows)} max_seq_len={max_seq_len}")
        for feature_set in feature_sets:
            print(f"Preparing feature_set={feature_set.name}")
            selected_arrays = {layer: layer_arrays[layer] for layer in feature_set.layers}
            scalers = compute_scalers(selected_arrays, train_rows, args.standardize)
            meta_scaler = compute_meta_scaler(df, train_rows, args.add_metadata_features)
            train_ds = DynamicFeatureSequenceDataset(
                df=df,
                sequences=train_seqs,
                feature_set=feature_set,
                layer_arrays=selected_arrays,
                scalers=scalers,
                meta_scaler=meta_scaler,
                add_metadata_features=args.add_metadata_features,
                l2_normalize=args.l2_normalize,
            )
            val_ds = DynamicFeatureSequenceDataset(
                df=df,
                sequences=val_seqs,
                feature_set=feature_set,
                layer_arrays=selected_arrays,
                scalers=scalers,
                meta_scaler=meta_scaler,
                add_metadata_features=args.add_metadata_features,
                l2_normalize=args.l2_normalize,
            )
            test_ds = DynamicFeatureSequenceDataset(
                df=df,
                sequences=test_seqs,
                feature_set=feature_set,
                layer_arrays=selected_arrays,
                scalers=scalers,
                meta_scaler=meta_scaler,
                add_metadata_features=args.add_metadata_features,
                l2_normalize=args.l2_normalize,
            )
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
                        row_ids, y_true, y_pred, y_prob = predict_with_probs(model, test_ds, args.batch_size, device, causal=causal)
                        eval_row = evaluate_row(
                            df,
                            row_ids,
                            y_true,
                            y_pred,
                            split,
                            feature_set,
                            spec,
                            attention_mode,
                            weighting,
                            len(train_rows),
                            len(val_rows),
                            len(train_ds),
                            len(val_ds),
                            len(test_ds),
                            info,
                            args,
                        )
                        metrics.append(eval_row)
                        pd.DataFrame(metrics).to_csv(metrics_path, index=False)
                        print(
                            "RESULT "
                            f"split={split.split_id} "
                            f"feature_set={feature_set.name} "
                            f"arch={spec.name} "
                            f"attention={attention_mode} "
                            f"bal={eval_row['test_top_action_balanced_accuracy']:.6f} "
                            f"acc={eval_row['test_top_action_accuracy']:.6f}",
                            flush=True,
                        )
                        if args.save_best_predictions:
                            prediction_frames.append(
                                make_prediction_frame(
                                    row_ids,
                                    y_true,
                                    y_pred,
                                    y_prob,
                                    split_id=split.split_id,
                                    split_role="test",
                                    feature_set=feature_set,
                                    spec=spec,
                                    attention_mode=attention_mode,
                                    loss_weighting=weighting,
                                    args=args,
                                )
                            )
                            val_row_ids, y_val_true, y_val_pred, y_val_prob = predict_with_probs(model, val_ds, args.batch_size, device, causal=causal)
                            validation_prediction_frames.append(
                                make_prediction_frame(
                                    val_row_ids,
                                    y_val_true,
                                    y_val_pred,
                                    y_val_prob,
                                    split_id=split.split_id,
                                    split_role="validation",
                                    feature_set=feature_set,
                                    spec=spec,
                                    attention_mode=attention_mode,
                                    loss_weighting=weighting,
                                    args=args,
                                )
                            )
                        del model
                        gc.collect()
                        if str(device) == "mps" and hasattr(torch, "mps"):
                            torch.mps.empty_cache()
            del train_ds, val_ds, test_ds, scalers, meta_scaler
            gc.collect()
    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv(metrics_path, index=False)
    if args.save_best_predictions and prediction_frames:
        pd.concat(prediction_frames, ignore_index=True).to_parquet(output_dir / "test_predictions.parquet", index=False)
    if args.save_best_predictions and validation_prediction_frames:
        pd.concat(validation_prediction_frames, ignore_index=True).to_parquet(output_dir / "validation_predictions.parquet", index=False)
    plot_summary(metrics_df, Path(figure_dir))
    write_report(metrics_df, Path(output_dir), Path(figure_dir), args)
    print(json.dumps({"metrics": str(metrics_path), "summary": str(output_dir / "feature_set_sequence_transformer_summary.csv"), "report": str(output_dir / "feature_set_sequence_transformer_report.md"), "figure_dir": str(figure_dir), "rows": int(len(metrics_df))}, indent=2))


if __name__ == "__main__":
    main()
