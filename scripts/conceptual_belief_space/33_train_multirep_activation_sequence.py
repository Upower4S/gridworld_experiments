#!/usr/bin/env python
"""Train sequence models on fused activation representations for top-action prediction.

This is a focused follow-up to ``24_train_multitask_sequence_transformer.py``.
It keeps the same grouped validation discipline and target losses, but allows
multiple activation representations to be concatenated after per-layer
train-only standardization and mean pooling across a selected layer window.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
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

from src.conceptual_belief_space.config import ACTIONS
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
class ModelSpec:
    name: str
    d_model: int
    n_layers: int
    n_heads: int
    dim_feedforward: int
    model_type: str = "gru"
    bidirectional: bool = True


@dataclass(frozen=True)
class FusionSpec:
    name: str
    representations: tuple[str, ...]
    layers: tuple[int, ...]
    rep_fusion: str


META_COLS = ["progress", "prefix_token_index", "num_cot_tokens", "prefix_fraction", "log_prefix_fraction"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--beliefs", default="outputs/conceptual_belief_space/action_belief_vectors.parquet")
    parser.add_argument("--splits", default="outputs/conceptual_belief_space/splits/group_splits.json")
    parser.add_argument("--feature_dir", default="outputs/activation/features")
    parser.add_argument("--output_dir", default="outputs/conceptual_belief_space/top_action_multirep_sequence")
    parser.add_argument("--figure_dir", default="figures/conceptual_belief_space/mlp_branch_probe/top_action_multirep_sequence")
    parser.add_argument("--subset", default="strict_before_final_action_mention")
    parser.add_argument("--group_col", default="trajectory_id")
    parser.add_argument("--layers", default="10,12,14,16,18")
    parser.add_argument("--fusion_sets", nargs="+", default=["raw,delta_pre", "raw,velocity", "raw,delta_pre,velocity"])
    parser.add_argument("--rep_fusion", choices=["concat", "mean"], default="concat")
    parser.add_argument("--architectures", nargs="+", default=["bigru_h512_l2"])
    parser.add_argument("--attention_modes", nargs="+", choices=["causal", "bidirectional"], default=["bidirectional"])
    parser.add_argument("--loss_weighting", choices=["plain", "balanced", "both"], default="balanced")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--batch_size", type=int, default=6)
    parser.add_argument("--max_epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--standardize", action="store_true", default=True)
    parser.add_argument("--no-standardize", action="store_false", dest="standardize")
    parser.add_argument("--l2_normalize", action="store_true")
    parser.add_argument("--collapse_penalty", type=float, default=0.05)
    parser.add_argument("--top_loss_weight", type=float, default=0.0)
    parser.add_argument("--agent_loss_weight", type=float, default=0.3)
    parser.add_argument("--q_loss_weight", type=float, default=1.0)
    parser.add_argument("--q_loss_kind", choices=["kl", "q_mse", "clr_mse"], default="kl")
    parser.add_argument("--q_sample_weighting", choices=["none", "top_balanced"], default="none")
    parser.add_argument("--add_metadata_features", action="store_true")
    parser.add_argument("--add_grid_features", action="store_true")
    parser.add_argument("--canonical", default="outputs/canonical_steps.parquet")
    parser.add_argument("--limit_splits", type=int, default=None)
    parser.add_argument("--save_best_predictions", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


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


def parse_layers(text: str, feature_dir: Path, reps: set[str]) -> tuple[int, ...]:
    if text.strip().lower() == "all":
        by_rep = []
        for rep in reps:
            layers = []
            pattern = re.compile(rf"layer_(\d+)_{re.escape(rep)}\.npy$")
            for path in sorted(feature_dir.glob(f"layer_*_{rep}.npy")):
                match = pattern.match(path.name)
                if match:
                    layers.append(int(match.group(1)))
            by_rep.append(set(layers))
        common = set.intersection(*by_rep) if by_rep else set()
        return tuple(sorted(common))
    return tuple(int(part) for part in text.split(",") if part.strip())


def parse_model_spec(text: str) -> ModelSpec:
    if text.startswith(("lstm_", "bilstm_")):
        vals: dict[str, int] = {}
        for part in text.split("_")[1:]:
            if part.startswith("h"):
                vals["hidden"] = int(part[1:])
            elif part.startswith("l"):
                vals["n_layers"] = int(part[1:])
        hidden = vals.get("hidden", 256)
        n_layers = vals.get("n_layers", 2)
        return ModelSpec(text, hidden, n_layers, 1, 4 * hidden, model_type="lstm", bidirectional=text.startswith("bilstm_"))
    if text.startswith(("gru_", "bigru_")):
        vals: dict[str, int] = {}
        for part in text.split("_")[1:]:
            if part.startswith("h"):
                vals["hidden"] = int(part[1:])
            elif part.startswith("l"):
                vals["n_layers"] = int(part[1:])
        hidden = vals.get("hidden", 256)
        n_layers = vals.get("n_layers", 2)
        return ModelSpec(text, hidden, n_layers, 1, 4 * hidden, model_type="gru", bidirectional=text.startswith("bigru_"))
    if text.startswith("tfm_"):
        vals: dict[str, int] = {}
        for part in text.split("_")[1:]:
            if part.startswith("d"):
                vals["d_model"] = int(part[1:])
            elif part.startswith("l"):
                vals["n_layers"] = int(part[1:])
            elif part.startswith("h"):
                vals["n_heads"] = int(part[1:])
            elif part.startswith("ff"):
                vals["dim_feedforward"] = int(part[2:])
        d_model = vals.get("d_model", 256)
        return ModelSpec(text, d_model, vals.get("n_layers", 2), vals.get("n_heads", 4), vals.get("dim_feedforward", 4 * d_model), model_type="transformer", bidirectional=False)
    raise ValueError(f"unsupported architecture: {text}")


def make_sequences(df: pd.DataFrame, indices: np.ndarray) -> list[np.ndarray]:
    idx = np.asarray(indices, dtype=np.int64)
    temp = df.iloc[idx][["decision_id", "prefix_token_index"]].copy()
    temp["_row_pos"] = idx
    temp = temp.sort_values(["decision_id", "prefix_token_index", "_row_pos"])
    return [group["_row_pos"].to_numpy(dtype=np.int64) for _, group in temp.groupby("decision_id", sort=False)]


def raw_metadata_features(df: pd.DataFrame, rows: np.ndarray, extra_cols: list[str] | None = None) -> np.ndarray:
    part = df.iloc[np.asarray(rows, dtype=np.int64)]
    progress = pd.to_numeric(part.get("progress", 0.0), errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
    prefix = pd.to_numeric(part.get("prefix_token_index", 0.0), errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
    n_tokens = pd.to_numeric(part.get("num_cot_tokens", np.nan), errors="coerce").fillna(np.nan).to_numpy(dtype=np.float32)
    fallback = float(np.nanmax(np.where(prefix > 0, prefix, 1.0))) if len(prefix) else 1.0
    n_tokens = np.where(np.isfinite(n_tokens) & (n_tokens > 0), n_tokens, fallback)
    n_tokens = np.where(n_tokens > 0, n_tokens, 1.0).astype(np.float32)
    prefix_fraction = np.clip(prefix / n_tokens, 0.0, 2.0).astype(np.float32)
    log_prefix_fraction = (np.log1p(np.maximum(prefix, 0.0)) / np.maximum(np.log1p(np.maximum(n_tokens, 1.0)), 1e-6)).astype(np.float32)
    base = np.stack([progress, prefix, n_tokens, prefix_fraction, log_prefix_fraction], axis=1).astype(np.float32)
    if extra_cols:
        extra = part[extra_cols].to_numpy(dtype=np.float32, copy=False)
        extra = np.nan_to_num(extra, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
        return np.concatenate([base, extra], axis=1).astype(np.float32)
    return base


def compute_meta_scaler(df: pd.DataFrame, train_rows: np.ndarray, enabled: bool, extra_cols: list[str] | None = None) -> dict[str, np.ndarray]:
    if not enabled:
        return {"mean": np.zeros(0, dtype=np.float32), "scale": np.ones(0, dtype=np.float32)}
    x = raw_metadata_features(df, train_rows, extra_cols=extra_cols)
    mean = x.mean(axis=0, dtype=np.float64).astype(np.float32)
    scale = x.std(axis=0, dtype=np.float64).astype(np.float32)
    return {"mean": mean, "scale": np.where(scale < 1e-8, 1.0, scale).astype(np.float32)}


def transform_metadata(df: pd.DataFrame, rows: np.ndarray, scaler: dict[str, np.ndarray], enabled: bool, extra_cols: list[str] | None = None) -> np.ndarray:
    if not enabled:
        return np.zeros((len(rows), 0), dtype=np.float32)
    x = raw_metadata_features(df, rows, extra_cols=extra_cols)
    return ((x - scaler["mean"]) / scaler["scale"]).astype(np.float32)


def attach_grid_features(df: pd.DataFrame, canonical_path: str | Path) -> tuple[pd.DataFrame, list[str]]:
    helper_path = ROOT / "scripts" / "conceptual_belief_space" / "31_gridworld_state_top_action_baseline.py"
    spec = importlib.util.spec_from_file_location("gridworld_state_top_action_baseline", helper_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load grid feature helper: {helper_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("gridworld_state_top_action_baseline", module)
    spec.loader.exec_module(module)
    missing = [col for col in ["trajectory_id", "step_id"] if col not in df.columns]
    if missing:
        raise ValueError(f"grid features require behavior columns: {missing}")
    canonical = pd.read_parquet(canonical_path, columns=["trajectory_id", "step_id", "grid_state_text", "carrying_key"])
    canonical["step_id"] = canonical["step_id"].astype(df["step_id"].dtype)
    joined = df[["trajectory_id", "step_id"]].merge(canonical, on=["trajectory_id", "step_id"], how="left", validate="many_to_one")
    if joined["grid_state_text"].isna().any():
        raise ValueError("missing grid_state_text after joining canonical steps")
    grid_df = pd.DataFrame([module.state_features(row) for _, row in joined.iterrows()]).astype(np.float32)
    grid_df = grid_df.add_prefix("gridfeat_")
    out = pd.concat([df.reset_index(drop=True), grid_df.reset_index(drop=True)], axis=1)
    return out, list(grid_df.columns)


def compute_scalers(feature_arrays: dict[tuple[str, int], np.ndarray], train_rows: np.ndarray, standardize: bool) -> dict[tuple[str, int], dict[str, np.ndarray]]:
    scalers: dict[tuple[str, int], dict[str, np.ndarray]] = {}
    train_rows = np.asarray(train_rows, dtype=np.int64)
    for key, arr in feature_arrays.items():
        if not standardize:
            scalers[key] = {"mean": np.zeros(arr.shape[1], dtype=np.float32), "scale": np.ones(arr.shape[1], dtype=np.float32)}
            continue
        x = np.asarray(arr[train_rows], dtype=np.float32)
        mean = x.mean(axis=0, dtype=np.float64).astype(np.float32)
        scale = x.std(axis=0, dtype=np.float64).astype(np.float32)
        scalers[key] = {"mean": mean, "scale": np.where(scale < 1e-8, 1.0, scale).astype(np.float32)}
    return scalers


def transform_array(arr: np.ndarray, rows: np.ndarray, scaler: dict[str, np.ndarray], l2_normalize: bool) -> np.ndarray:
    x = np.asarray(arr[np.asarray(rows, dtype=np.int64)], dtype=np.float32)
    x = (x - scaler["mean"]) / scaler["scale"]
    if l2_normalize:
        norm = np.linalg.norm(x, axis=1, keepdims=True)
        x = x / np.where(norm < 1e-8, 1.0, norm)
    return x.astype(np.float32)


class MultiRepSequenceDataset:
    def __init__(
        self,
        *,
        df: pd.DataFrame,
        sequences: list[np.ndarray],
        fusion: FusionSpec,
        feature_arrays: dict[tuple[str, int], np.ndarray],
        scalers: dict[tuple[str, int], dict[str, np.ndarray]],
        meta_scaler: dict[str, np.ndarray],
        add_metadata_features: bool,
        extra_meta_cols: list[str] | None,
        l2_normalize: bool,
    ) -> None:
        self.df = df
        self.sequences = sequences
        self.fusion = fusion
        self.feature_arrays = feature_arrays
        self.scalers = scalers
        self.meta_scaler = meta_scaler
        self.add_metadata_features = add_metadata_features
        self.extra_meta_cols = extra_meta_cols or []
        self.l2_normalize = l2_normalize
        self.y_top_by_row = action_labels_to_indices(df["top_action"])
        self.y_agent_by_row = action_labels_to_indices(df["agent_action"])
        self.q_by_row = df[Q_COLS].to_numpy(dtype=np.float32)
        self.meta_by_row = transform_metadata(
            df,
            np.arange(len(df), dtype=np.int64),
            meta_scaler,
            add_metadata_features,
            extra_cols=self.extra_meta_cols,
        )

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, i: int):
        rows = self.sequences[i]
        rep_blocks = []
        for rep in self.fusion.representations:
            layer_blocks = [
                transform_array(self.feature_arrays[(rep, layer)], rows, self.scalers[(rep, layer)], self.l2_normalize)
                for layer in self.fusion.layers
            ]
            rep_blocks.append(np.mean(np.stack(layer_blocks, axis=0), axis=0).astype(np.float32))
        if self.fusion.rep_fusion == "concat":
            x = np.concatenate(rep_blocks, axis=1).astype(np.float32)
        elif self.fusion.rep_fusion == "mean":
            x = np.mean(np.stack(rep_blocks, axis=0), axis=0).astype(np.float32)
        else:
            raise ValueError(self.fusion.rep_fusion)
        meta = self.meta_by_row[rows]
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
    feat_dim = batch[0][0].shape[1]
    meta_dim = batch[0][1].shape[1]
    x = np.zeros((len(batch), max_len, feat_dim), dtype=np.float32)
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


def build_model(input_dim: int, meta_dim: int, spec: ModelSpec, max_seq_len: int, dropout: float):
    import torch
    import torch.nn as nn

    class SequenceGRU(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.input_proj = nn.Sequential(nn.LayerNorm(input_dim + meta_dim), nn.Linear(input_dim + meta_dim, spec.d_model), nn.GELU())
            self.gru = nn.GRU(
                input_size=spec.d_model,
                hidden_size=spec.d_model,
                num_layers=spec.n_layers,
                dropout=dropout if spec.n_layers > 1 else 0.0,
                batch_first=True,
                bidirectional=spec.bidirectional,
            )
            out_dim = spec.d_model * (2 if spec.bidirectional else 1)
            self.norm = nn.LayerNorm(out_dim)
            self.top_classifier = nn.Linear(out_dim, len(ACTIONS))
            self.agent_classifier = nn.Linear(out_dim, len(ACTIONS))

        def forward(self, x, meta, padding_mask, *, causal: bool = True):
            if meta.shape[-1] > 0:
                x = torch.cat([x, meta.to(dtype=x.dtype)], dim=-1)
            h = self.input_proj(x)
            lengths = (~padding_mask).sum(dim=1).detach().cpu()
            packed = nn.utils.rnn.pack_padded_sequence(h, lengths, batch_first=True, enforce_sorted=False)
            packed_out, _ = self.gru(packed)
            h, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True, total_length=x.shape[1])
            h = self.norm(h)
            return {"top": self.top_classifier(h), "agent": self.agent_classifier(h)}

    class SequenceLSTM(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.input_proj = nn.Sequential(nn.LayerNorm(input_dim + meta_dim), nn.Linear(input_dim + meta_dim, spec.d_model), nn.GELU())
            self.lstm = nn.LSTM(
                input_size=spec.d_model,
                hidden_size=spec.d_model,
                num_layers=spec.n_layers,
                dropout=dropout if spec.n_layers > 1 else 0.0,
                batch_first=True,
                bidirectional=spec.bidirectional,
            )
            out_dim = spec.d_model * (2 if spec.bidirectional else 1)
            self.norm = nn.LayerNorm(out_dim)
            self.top_classifier = nn.Linear(out_dim, len(ACTIONS))
            self.agent_classifier = nn.Linear(out_dim, len(ACTIONS))

        def forward(self, x, meta, padding_mask, *, causal: bool = True):
            if meta.shape[-1] > 0:
                x = torch.cat([x, meta.to(dtype=x.dtype)], dim=-1)
            h = self.input_proj(x)
            lengths = (~padding_mask).sum(dim=1).detach().cpu()
            packed = nn.utils.rnn.pack_padded_sequence(h, lengths, batch_first=True, enforce_sorted=False)
            packed_out, _ = self.lstm(packed)
            h, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True, total_length=x.shape[1])
            h = self.norm(h)
            return {"top": self.top_classifier(h), "agent": self.agent_classifier(h)}

    class SequenceTransformer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
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
            if meta.shape[-1] > 0:
                x = torch.cat([x, meta.to(dtype=x.dtype)], dim=-1)
            length = x.shape[1]
            h = self.input_proj(x) + self.pos_embedding[:, :length]
            causal_mask = torch.triu(torch.ones((length, length), dtype=torch.bool, device=x.device), diagonal=1) if causal else None
            h = self.encoder(h, mask=causal_mask, src_key_padding_mask=padding_mask)
            h = self.norm(h)
            return {"top": self.top_classifier(h), "agent": self.agent_classifier(h)}

    if spec.model_type == "transformer":
        return SequenceTransformer()
    if spec.model_type == "lstm":
        return SequenceLSTM()
    return SequenceGRU()


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


def train_one(train_ds, val_ds, input_dim: int, spec: ModelSpec, max_seq_len: int, args, device, *, seed: int, balanced_loss: bool, causal: bool):
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader

    torch.manual_seed(seed)
    np.random.seed(seed)
    meta_dim = int(train_ds.meta_scaler["mean"].shape[0]) if train_ds.add_metadata_features else 0
    model = build_model(input_dim, meta_dim, spec, max_seq_len, args.dropout).to(device)
    y_top_train = np.concatenate([train_ds.y_top_by_row[seq] for seq in train_ds.sequences])
    y_agent_train = np.concatenate([train_ds.y_agent_by_row[seq] for seq in train_ds.sequences])
    top_weights = torch.tensor(class_weights(y_top_train), dtype=torch.float32, device=device) if balanced_loss else None
    agent_weights = torch.tensor(class_weights(y_agent_train), dtype=torch.float32, device=device) if balanced_loss else None
    q_sample_weights = (
        torch.tensor(class_weights(y_top_train), dtype=torch.float32, device=device)
        if args.q_sample_weighting == "top_balanced"
        else None
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
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
        for xb, meta, y_top, y_agent, q_target, pad_mask, _ in loader:
            y_top_d = y_top.to(device)
            y_agent_d = y_agent.to(device)
            q_target_d = q_target.to(device)
            outputs = model(xb.to(device), meta.to(device), pad_mask.to(device), causal=causal)
            top_logits = outputs["top"]
            agent_logits = outputs["agent"]
            loss = torch.zeros((), dtype=top_logits.dtype, device=device)
            if args.top_loss_weight > 0:
                loss = loss + float(args.top_loss_weight) * F.cross_entropy(
                    top_logits.reshape(-1, len(ACTIONS)),
                    y_top_d.reshape(-1),
                    ignore_index=-100,
                    weight=top_weights,
                )
            if args.agent_loss_weight > 0:
                loss = loss + float(args.agent_loss_weight) * F.cross_entropy(
                    agent_logits.reshape(-1, len(ACTIONS)),
                    y_agent_d.reshape(-1),
                    ignore_index=-100,
                    weight=agent_weights,
                )
            if args.q_loss_weight > 0:
                valid = y_top_d.reshape(-1) != -100
                y_valid = y_top_d.reshape(-1)[valid]
                logits_flat = top_logits.reshape(-1, len(ACTIONS))[valid]
                q_flat = q_target_d.reshape(-1, len(ACTIONS))[valid]
                if args.q_loss_kind == "kl":
                    q_loss_by_row = F.kl_div(F.log_softmax(logits_flat, dim=-1), q_flat, reduction="none").sum(dim=-1)
                elif args.q_loss_kind == "q_mse":
                    q_loss_by_row = (F.softmax(logits_flat, dim=-1) - q_flat).pow(2).mean(dim=-1)
                elif args.q_loss_kind == "clr_mse":
                    log_q = torch.log(torch.clamp(q_flat, min=1e-8))
                    target_clr = log_q - log_q.mean(dim=-1, keepdim=True)
                    pred_clr = logits_flat - logits_flat.mean(dim=-1, keepdim=True)
                    q_loss_by_row = (pred_clr - target_clr).pow(2).mean(dim=-1)
                else:
                    raise ValueError(args.q_loss_kind)
                if q_sample_weights is not None:
                    row_weights = q_sample_weights[y_valid.clamp_min(0)]
                    q_loss = (q_loss_by_row * row_weights).sum() / torch.clamp(row_weights.sum(), min=1e-8)
                else:
                    q_loss = q_loss_by_row.mean()
                loss = loss + float(args.q_loss_weight) * q_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if args.grad_clip and args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
        _, y_val, p_val, _ = predict_with_probs(model, val_ds, args.batch_size, device, causal=causal)
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


def evaluate_row(df, row_ids, y_true, y_pred, split, fusion, spec, attention_mode, loss_weighting, train_n, val_n, train_seq_n, val_seq_n, test_seq_n, train_info, args):
    top_m = metrics_from_predictions(y_true, y_pred)
    y_agent = action_labels_to_indices(df.loc[row_ids, "agent_action"])
    agent_m = metrics_from_predictions(y_agent, y_pred)
    row = {
        "run_id": f"multirep_{fusion.name}_{spec.name}_{loss_weighting}_split{split.split_id}_{int(time.time())}",
        "split_id": int(split.split_id),
        "representations": ",".join(fusion.representations),
        "subset": args.subset,
        "fusion_set": fusion.name,
        "rep_fusion": fusion.rep_fusion,
        "feature_layers_json": json.dumps(list(fusion.layers)),
        "n_feature_layers": len(fusion.layers),
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
        "q_sample_weighting": args.q_sample_weighting,
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


def make_prediction_frame(row_ids, y_true, y_pred, y_prob, *, split_id: int, split_role: str, fusion: FusionSpec, spec: ModelSpec, attention_mode: str, loss_weighting: str, args) -> pd.DataFrame:
    pred_df = pd.DataFrame(
        {
            "split_id": int(split_id),
            "split_role": split_role,
            "row_id": row_ids.astype(np.int64),
            "true_top_action": action_indices_to_labels(y_true),
            "pred_top_action": action_indices_to_labels(y_pred),
            "fusion_set": fusion.name,
            "representations": ",".join(fusion.representations),
            "rep_fusion": fusion.rep_fusion,
            "architecture": spec.name,
            "attention_mode": attention_mode,
            "loss_weighting": loss_weighting,
            "top_loss_weight": float(args.top_loss_weight),
            "agent_loss_weight": float(args.agent_loss_weight),
            "q_loss_weight": float(args.q_loss_weight),
            "q_loss_kind": args.q_loss_kind,
            "q_sample_weighting": args.q_sample_weighting,
            "add_metadata_features": bool(args.add_metadata_features),
        }
    )
    for i, action in enumerate(ACTIONS):
        pred_df[f"prob_{action}"] = y_prob[:, i]
    return pred_df


def plot_summary(metrics: pd.DataFrame, figure_dir: Path) -> None:
    import matplotlib.pyplot as plt

    ensure_dir(figure_dir)
    if metrics.empty:
        return
    summary = (
        metrics.groupby(["fusion_set", "architecture", "attention_mode", "loss_weighting", "top_loss_weight", "agent_loss_weight", "q_loss_weight", "q_loss_kind", "q_sample_weighting", "add_metadata_features"], as_index=False)
        .agg(mean_bal=("test_top_action_balanced_accuracy", "mean"), mean_acc=("test_top_action_accuracy", "mean"))
        .sort_values("mean_bal", ascending=False)
    )
    labels = [f"{r.fusion_set}\n{r.architecture}/{r.attention_mode}\na{r.agent_loss_weight:g}/q{r.q_loss_weight:g}/{r.q_sample_weighting}\nm{int(r.add_metadata_features)}" for r in summary.itertuples()]
    x = np.arange(len(summary))
    fig, ax = plt.subplots(figsize=(max(10, len(summary) * 0.5), 5))
    ax.bar(x, summary["mean_bal"].to_numpy(), label="balanced accuracy", color="C0")
    ax.bar(x, summary["mean_acc"].to_numpy(), label="accuracy", color="C1", alpha=0.45)
    ax.axhline(0.25, color="black", linestyle="--", linewidth=1)
    ax.axhline(0.8, color="C3", linestyle="--", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=70, ha="right", fontsize=8)
    ax.set_ylabel("held-out test score")
    ax.set_title("Multi-representation activation sequence models")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figure_dir / "multirep_top_action_accuracy.png", dpi=200)
    plt.close(fig)


def write_report(metrics: pd.DataFrame, output_dir: Path, figure_dir: Path, args) -> None:
    if metrics.empty:
        save_markdown(output_dir / "multirep_sequence_report.md", "# Multi-Representation Sequence Sweep\n\nNo metrics were produced.\n")
        return
    summary = (
        metrics.groupby(["fusion_set", "representations", "rep_fusion", "architecture", "attention_mode", "loss_weighting", "top_loss_weight", "agent_loss_weight", "q_loss_weight", "q_loss_kind", "q_sample_weighting", "add_metadata_features"], as_index=False)
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
    summary.to_csv(output_dir / "multirep_sequence_summary.csv", index=False)
    best = summary.iloc[0].to_dict() if not summary.empty else {}
    lines = [
        "# Multi-Representation Activation Sequence Sweep",
        "",
        f"- subset: `{args.subset}`",
        f"- layers: `{args.layers}`",
        f"- fusion sets: `{', '.join(args.fusion_sets)}`",
        f"- rep fusion: `{args.rep_fusion}`",
        f"- architectures: `{', '.join(args.architectures)}`",
        f"- attention modes: `{', '.join(args.attention_modes)}`",
        f"- top loss weight: `{args.top_loss_weight}`",
        f"- agent loss weight: `{args.agent_loss_weight}`",
        f"- q loss weight: `{args.q_loss_weight}`",
        f"- q loss kind: `{args.q_loss_kind}`",
        f"- q sample weighting: `{args.q_sample_weighting}`",
        f"- metadata features: `{args.add_metadata_features}`",
        "",
        "## Best Mean Result",
        "",
    ]
    if best:
        for key in ["fusion_set", "representations", "architecture", "attention_mode", "loss_weighting", "splits", "mean_acc", "mean_bal", "mean_f1", "mean_agent_acc", "collapse_rows"]:
            val = best[key]
            lines.append(f"- {key}: {val:.6f}" if isinstance(val, float) else f"- {key}: `{val}`")
    lines.extend(["", "## Summary", "", summary.to_markdown(index=False), "", "## Figure", "", f"- `{figure_dir / 'multirep_top_action_accuracy.png'}`"])
    save_markdown(output_dir / "multirep_sequence_report.md", "\n".join(lines))
    if best:
        write_json(output_dir / "best_multirep_sequence_config.json", json_ready(best))


def main() -> None:
    args = parse_args()
    output_dir = ensure_dir(args.output_dir)
    figure_dir = ensure_dir(args.figure_dir)
    metrics_path = output_dir / "multirep_sequence_metrics.csv"
    if metrics_path.exists() and not args.force:
        raise SystemExit(f"{metrics_path} exists; pass --force to overwrite")
    import torch

    device = choose_device(args.device)
    df = make_behavior_table(args.beliefs).reset_index(drop=True)
    extra_meta_cols: list[str] = []
    if args.add_grid_features:
        df, extra_meta_cols = attach_grid_features(df, args.canonical)
        args.add_metadata_features = True
    sub_idx = np.flatnonzero(subset_mask(df, args.subset))
    feature_dir = Path(args.feature_dir)
    fusion_reps = [tuple(part.strip() for part in text.split(",") if part.strip()) for text in args.fusion_sets]
    all_reps = set(rep for reps in fusion_reps for rep in reps)
    layers = parse_layers(args.layers, feature_dir, all_reps)
    if not layers:
        raise ValueError("no layers selected")
    fusions = []
    for reps in fusion_reps:
        missing = [f"layer_{layer:02d}_{rep}.npy" for rep in reps for layer in layers if not (feature_dir / f"layer_{layer:02d}_{rep}.npy").exists()]
        if missing:
            raise FileNotFoundError(f"missing feature files for fusion {reps}: {missing[:8]}")
        name = f"{args.rep_fusion}_{'_'.join(reps)}_layers_{'_'.join(f'{layer:02d}' for layer in layers)}"
        fusions.append(FusionSpec(name=name, representations=reps, layers=layers, rep_fusion=args.rep_fusion))

    needed = sorted({(rep, layer) for fusion in fusions for rep in fusion.representations for layer in fusion.layers})
    feature_arrays = {(rep, layer): np.load(feature_dir / f"layer_{layer:02d}_{rep}.npy", mmap_mode="r") for rep, layer in needed}
    n_rows = len(df)
    hidden_dim = next(iter(feature_arrays.values())).shape[1]
    for key, arr in feature_arrays.items():
        if arr.shape[0] != n_rows:
            raise ValueError(f"feature {key} rows {arr.shape[0]} != behavior rows {n_rows}")
        if arr.shape[1] != hidden_dim:
            raise ValueError(f"feature {key} hidden dim {arr.shape[1]} != {hidden_dim}")
    specs = [parse_model_spec(text) for text in args.architectures]
    loss_weightings = ["plain", "balanced"] if args.loss_weighting == "both" else [args.loss_weighting]
    splits = load_or_create_splits(args.splits, df, args.group_col, seed=args.random_seed)
    if args.limit_splits is not None:
        splits = splits[: int(args.limit_splits)]
    print(
        json.dumps(
            {
                "device": str(device),
                "layers": list(layers),
                "fusion_sets": [fusion.name for fusion in fusions],
                "architectures": [spec.name for spec in specs],
                "attention_modes": args.attention_modes,
                "loss_weightings": loss_weightings,
                "top_loss_weight": args.top_loss_weight,
                "agent_loss_weight": args.agent_loss_weight,
                "q_loss_weight": args.q_loss_weight,
                "q_loss_kind": args.q_loss_kind,
                "q_sample_weighting": args.q_sample_weighting,
                "add_metadata_features": args.add_metadata_features,
                "add_grid_features": args.add_grid_features,
                "n_grid_features": len(extra_meta_cols),
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
            print(f"Skipping split={split.split_id} because train_rows={len(train_pool)} test_rows={len(test_rows)}")
            continue
        train_rows, val_rows = make_train_val_split(df, train_pool, split.group_col, seed=args.random_seed + split.split_id)
        if len(train_rows) == 0 or len(val_rows) == 0:
            print(f"Skipping split={split.split_id} because train/val is empty")
            continue
        train_seqs = make_sequences(df, train_rows)
        val_seqs = make_sequences(df, val_rows)
        test_seqs = make_sequences(df, test_rows)
        max_seq_len = max(max(len(seq) for seq in train_seqs), max(len(seq) for seq in val_seqs), max(len(seq) for seq in test_seqs))
        print(f"split={split.split_id} train_rows={len(train_rows)} val_rows={len(val_rows)} test_rows={len(test_rows)} max_seq_len={max_seq_len}")
        for fusion in fusions:
            selected_arrays = {(rep, layer): feature_arrays[(rep, layer)] for rep in fusion.representations for layer in fusion.layers}
            print(f"Preparing fusion={fusion.name}")
            scalers = compute_scalers(selected_arrays, train_rows, args.standardize)
            meta_scaler = compute_meta_scaler(df, train_rows, args.add_metadata_features, extra_cols=extra_meta_cols)
            train_ds = MultiRepSequenceDataset(df=df, sequences=train_seqs, fusion=fusion, feature_arrays=selected_arrays, scalers=scalers, meta_scaler=meta_scaler, add_metadata_features=args.add_metadata_features, extra_meta_cols=extra_meta_cols, l2_normalize=args.l2_normalize)
            val_ds = MultiRepSequenceDataset(df=df, sequences=val_seqs, fusion=fusion, feature_arrays=selected_arrays, scalers=scalers, meta_scaler=meta_scaler, add_metadata_features=args.add_metadata_features, extra_meta_cols=extra_meta_cols, l2_normalize=args.l2_normalize)
            test_ds = MultiRepSequenceDataset(df=df, sequences=test_seqs, fusion=fusion, feature_arrays=selected_arrays, scalers=scalers, meta_scaler=meta_scaler, add_metadata_features=args.add_metadata_features, extra_meta_cols=extra_meta_cols, l2_normalize=args.l2_normalize)
            input_dim = hidden_dim * len(fusion.representations) if fusion.rep_fusion == "concat" else hidden_dim
            for spec in specs:
                for attention_mode in args.attention_modes:
                    causal = attention_mode == "causal"
                    for weighting in loss_weightings:
                        seed = stable_seed(fusion.name, spec.name, attention_mode, weighting, split.split_id, base=args.random_seed)
                        print(f"Training split={split.split_id} fusion={fusion.name} arch={spec.name} attention={attention_mode} weighting={weighting}")
                        model, info = train_one(train_ds, val_ds, input_dim, spec, max_seq_len, args, device, seed=seed, balanced_loss=weighting == "balanced", causal=causal)
                        row_ids, y_true, y_pred, y_prob = predict_with_probs(model, test_ds, args.batch_size, device, causal=causal)
                        eval_row = evaluate_row(df, row_ids, y_true, y_pred, split, fusion, spec, attention_mode, weighting, len(train_rows), len(val_rows), len(train_ds), len(val_ds), len(test_ds), info, args)
                        metrics.append(eval_row)
                        pd.DataFrame(metrics).to_csv(metrics_path, index=False)
                        print(
                            "RESULT "
                            f"split={split.split_id} "
                            f"fusion={fusion.name} "
                            f"arch={spec.name} "
                            f"attention={attention_mode} "
                            f"bal={eval_row['test_top_action_balanced_accuracy']:.6f} "
                            f"acc={eval_row['test_top_action_accuracy']:.6f}",
                            flush=True,
                        )
                        if args.save_best_predictions:
                            prediction_frames.append(make_prediction_frame(row_ids, y_true, y_pred, y_prob, split_id=split.split_id, split_role="test", fusion=fusion, spec=spec, attention_mode=attention_mode, loss_weighting=weighting, args=args))
                            val_row_ids, y_val_true, y_val_pred, y_val_prob = predict_with_probs(model, val_ds, args.batch_size, device, causal=causal)
                            validation_prediction_frames.append(make_prediction_frame(val_row_ids, y_val_true, y_val_pred, y_val_prob, split_id=split.split_id, split_role="validation", fusion=fusion, spec=spec, attention_mode=attention_mode, loss_weighting=weighting, args=args))
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
    plot_summary(metrics_df, figure_dir)
    write_report(metrics_df, output_dir, figure_dir, args)
    print(json.dumps({"metrics": str(metrics_path), "summary": str(output_dir / "multirep_sequence_summary.csv"), "report": str(output_dir / "multirep_sequence_report.md"), "figure_dir": str(figure_dir), "rows": int(len(metrics_df))}, indent=2))


if __name__ == "__main__":
    main()
