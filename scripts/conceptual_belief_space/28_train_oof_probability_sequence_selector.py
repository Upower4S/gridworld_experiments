#!/usr/bin/env python
"""Train grouped OOF probability-sequence selectors for top_action.

The inputs are held-out base-model probability predictions. The selector never
uses behavior q columns, agent_action, or top_action as input features.
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
from typing import Any, Iterable

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.conceptual_belief_space.config import ACTIONS
from src.conceptual_belief_space.io import ensure_dir, save_markdown, write_json


PROB_COLS = [f"prob_{action}" for action in ACTIONS]


@dataclass(frozen=True)
class ModelSpec:
    name: str
    directory: str
    feature_set: str | None = None


@dataclass(frozen=True)
class ArchSpec:
    name: str
    model_type: str
    hidden: int
    layers: int
    heads: int = 4
    bidirectional: bool = False


MODEL_SPECS = [
    ModelSpec("top_meta", "top_action_valpred_top_meta_full"),
    ModelSpec("q05_meta", "top_action_valpred_q05_meta_full"),
    ModelSpec("agent03_q03_meta", "top_action_valpred_agent03_q03_meta_full"),
    ModelSpec("mean_all_tfm", "top_action_valpred_mean_all_tfm_full"),
    ModelSpec("mean_all_bigru_h256", "top_action_valpred_mean_all_bigru_full"),
    ModelSpec("mean_late_bigru", "top_action_valpred_mean_late_bigru_full"),
    ModelSpec("mix_all_bigru", "top_action_valpred_mix_all_bigru_full"),
    ModelSpec("mean_all_bigru_h512_raw", "top_action_valpred_mean_all_bigru_h512_full"),
    ModelSpec("mean_all_bigru_h512_delta", "top_action_valpred_mean_all_bigru_h512_delta_pre_full"),
    ModelSpec("mean_all_bigru_h512_qonly", "top_action_valpred_mean_all_bigru_h512_qonly_agent03_full"),
    ModelSpec(
        "layer_10_12_14_16_18_h512_qonly",
        "top_action_valpred_layer_windows_h512_qonly_full",
        "mean_10_12_14_16_18",
    ),
    ModelSpec(
        "layer_08_10_12_14_16_h512_qonly",
        "top_action_valpred_layer_windows_h512_qonly_full",
        "mean_08_10_12_14_16",
    ),
    ModelSpec("layer_10_12_14_16_18_20_h512_qonly", "top_action_valpred_layer_10_20_h512_qonly_full"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base_dir", default="outputs/conceptual_belief_space")
    parser.add_argument("--beliefs", default="outputs/conceptual_belief_space/action_belief_vectors.parquet")
    parser.add_argument("--output_dir", default="outputs/conceptual_belief_space/top_action_oof_probability_sequence_selector")
    parser.add_argument(
        "--architectures",
        nargs="+",
        default=["bigru_h128_l2", "bigru_h256_l2", "tfm_d128_l2_h4"],
    )
    parser.add_argument("--group_col", default="trajectory_id")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--lr", type=float, default=8e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--loss_weighting", choices=["balanced", "plain"], default="balanced")
    parser.add_argument("--label_smoothing", type=float, default=0.02)
    parser.add_argument("--include_context_features", action="store_true", default=True)
    parser.add_argument("--no_context_features", action="store_false", dest="include_context_features")
    parser.add_argument("--activation_feature_dir", default="outputs/activation/features")
    parser.add_argument("--activation_representation", default="raw")
    parser.add_argument(
        "--activation_layers",
        default="",
        help="Optional comma-separated activation layers to average and append as input features, e.g. 10,12,14,16,18.",
    )
    parser.add_argument("--transition_weights", default="0,0.05,0.1,0.15,0.25,0.4,0.6")
    parser.add_argument("--limit_splits", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def parse_architecture(text: str) -> ArchSpec:
    if text.startswith(("gru_", "bigru_")):
        hidden = 128
        layers = 2
        for part in text.split("_")[1:]:
            if part.startswith("h"):
                hidden = int(part[1:])
            elif part.startswith("l"):
                layers = int(part[1:])
        return ArchSpec(text, "gru", hidden, layers, bidirectional=text.startswith("bigru_"))
    if text.startswith("tfm_"):
        hidden = 128
        layers = 2
        heads = 4
        for part in text.split("_")[1:]:
            if part.startswith("d"):
                hidden = int(part[1:])
            elif part.startswith("l"):
                layers = int(part[1:])
            elif part.startswith("h"):
                heads = int(part[1:])
        return ArchSpec(text, "transformer", hidden, layers, heads=heads)
    if text.startswith("tcn_"):
        hidden = 128
        layers = 4
        for part in text.split("_")[1:]:
            if part.startswith("c"):
                hidden = int(part[1:])
            elif part.startswith("l"):
                layers = int(part[1:])
        return ArchSpec(text, "tcn", hidden, layers)
    raise ValueError(f"unknown architecture: {text}")


def stable_seed(*parts: object, base: int) -> int:
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


def load_prediction(base_dir: Path, spec: ModelSpec) -> pd.DataFrame:
    path = base_dir / spec.directory / "test_predictions.parquet"
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_parquet(path)
    if spec.feature_set is not None:
        df = df[df["feature_set"] == spec.feature_set].copy()
    needed = ["split_id", "row_id", "true_top_action", *PROB_COLS]
    missing = [col for col in needed if col not in df.columns]
    if missing:
        raise ValueError(f"{path} missing columns: {missing}")
    df = df[needed].copy()
    if df.empty:
        raise ValueError(f"empty predictions for {spec.name}")
    if df["row_id"].duplicated().any():
        dup = df.loc[df["row_id"].duplicated(), "row_id"].head().tolist()
        raise ValueError(f"duplicate row_id in {spec.name}: {dup}")
    df["row_id"] = df["row_id"].astype(np.int64)
    return df.sort_values("row_id").reset_index(drop=True)


def labels_to_idx(labels: Iterable[str]) -> np.ndarray:
    mapping = {action: i for i, action in enumerate(ACTIONS)}
    return np.asarray([mapping[str(label)] for label in labels], dtype=np.int64)


def idx_to_labels(idx: np.ndarray) -> np.ndarray:
    return np.asarray([ACTIONS[int(i)] for i in idx], dtype=object)


def make_base_table(args: argparse.Namespace) -> tuple[pd.DataFrame, np.ndarray, list[str]]:
    base_dir = Path(args.base_dir)
    metadata_cols = [
        "row_id",
        "decision_id",
        "trajectory_id",
        "step_id",
        "prefix_token_index",
        "progress",
        "top_action",
        "num_cot_tokens",
        "first_action_mention_token_index",
        "first_final_action_mention_token_index",
        "strict_before_any_action_mention",
        "strict_before_final_action_mention",
    ]
    beliefs_all_cols = pd.read_parquet(args.beliefs).columns.tolist()
    metadata_cols = [col for col in metadata_cols if col in beliefs_all_cols]
    meta = pd.read_parquet(args.beliefs, columns=metadata_cols).copy()
    meta["row_id"] = meta["row_id"].astype(np.int64)
    pred_frames = {spec.name: load_prediction(base_dir, spec) for spec in MODEL_SPECS}
    common = set(meta["row_id"].astype(int))
    for df in pred_frames.values():
        common &= set(df["row_id"].astype(int))
    if not common:
        raise ValueError("no common row_id across base predictions")
    row_ids = np.asarray(sorted(common), dtype=np.int64)
    table = pd.DataFrame({"row_id": row_ids}).merge(meta, on="row_id", how="left", validate="one_to_one")
    if table[["decision_id", "trajectory_id", "prefix_token_index", "top_action"]].isna().any().any():
        raise ValueError("missing required metadata after merge")
    feature_blocks = []
    feature_names = []
    true_label = None
    for name, df in pred_frames.items():
        part = df[df["row_id"].isin(row_ids)].sort_values("row_id")
        if not np.array_equal(part["row_id"].to_numpy(dtype=np.int64), row_ids):
            raise ValueError(f"row order mismatch for {name}")
        labels = part["true_top_action"].to_numpy()
        if true_label is None:
            true_label = labels
        elif not np.array_equal(true_label, labels):
            raise ValueError(f"label mismatch for {name}")
        probs = part[PROB_COLS].to_numpy(dtype=np.float32)
        probs = np.clip(probs, 1e-8, None)
        probs = probs / probs.sum(axis=1, keepdims=True)
        feature_blocks.append(probs)
        feature_names.extend([f"{name}_p_{a}" for a in ACTIONS])
        logp = np.log(probs)
        feature_blocks.append(logp)
        feature_names.extend([f"{name}_logp_{a}" for a in ACTIONS])
        pred = probs.argmax(axis=1)
        votes = np.eye(len(ACTIONS), dtype=np.float32)[pred]
        feature_blocks.append(votes)
        feature_names.extend([f"{name}_vote_{a}" for a in ACTIONS])
        entropy = -(probs * logp).sum(axis=1, keepdims=True) / math.log(len(ACTIONS))
        top2 = np.sort(probs, axis=1)[:, -2:]
        margin = (top2[:, 1] - top2[:, 0]).reshape(-1, 1)
        feature_blocks.append(np.concatenate([entropy, margin], axis=1).astype(np.float32))
        feature_names.extend([f"{name}_entropy", f"{name}_margin"])
    table["true_top_action"] = true_label
    prob_stack = np.stack([block for block in feature_blocks[0::4]], axis=0)
    for reducer, values in [
        ("mean", prob_stack.mean(axis=0)),
        ("std", prob_stack.std(axis=0)),
        ("max", prob_stack.max(axis=0)),
        ("min", prob_stack.min(axis=0)),
    ]:
        feature_blocks.append(values.astype(np.float32))
        feature_names.extend([f"base_{reducer}_p_{a}" for a in ACTIONS])
    meta_features = make_metadata_features(table)
    feature_blocks.append(meta_features)
    feature_names.extend([f"meta_{i}" for i in range(meta_features.shape[1])])
    if args.activation_layers.strip():
        layers = [int(part) for part in args.activation_layers.split(",") if part.strip()]
        act = None
        for layer in layers:
            path = Path(args.activation_feature_dir) / f"layer_{layer:02d}_{args.activation_representation}.npy"
            if not path.exists():
                raise FileNotFoundError(path)
            arr = np.load(path, mmap_mode="r")
            block = np.asarray(arr[row_ids], dtype=np.float32)
            act = block if act is None else act + block
        assert act is not None
        act = (act / float(len(layers))).astype(np.float32)
        feature_blocks.append(act)
        feature_names.extend([f"activation_mean_{args.activation_representation}_{'_'.join(map(str, layers))}_{i}" for i in range(act.shape[1])])
    x = np.concatenate(feature_blocks, axis=1).astype(np.float32)
    return table.reset_index(drop=True), x, feature_names


def make_metadata_features(table: pd.DataFrame) -> np.ndarray:
    prefix = pd.to_numeric(table.get("prefix_token_index", 0), errors="coerce").fillna(0).to_numpy(dtype=np.float32)
    progress = pd.to_numeric(table.get("progress", 0), errors="coerce").fillna(0).to_numpy(dtype=np.float32)
    if "num_cot_tokens" in table.columns:
        n_tokens = pd.to_numeric(table["num_cot_tokens"], errors="coerce").to_numpy(dtype=np.float32)
    else:
        n_tokens = np.full(len(table), np.nan, dtype=np.float32)
    fallback_n = max(float(np.nanmax(np.maximum(prefix, 1.0))), 1.0)
    n_tokens = np.where(np.isfinite(n_tokens) & (n_tokens > 0), n_tokens, fallback_n).astype(np.float32)
    frac = np.clip(prefix / np.maximum(n_tokens, 1.0), 0.0, 2.0)
    log_frac = np.log1p(np.maximum(prefix, 0.0)) / np.maximum(np.log1p(n_tokens), 1e-6)
    feats = [progress, prefix, n_tokens, frac, log_frac]
    for col in ["first_action_mention_token_index", "first_final_action_mention_token_index"]:
        if col in table.columns:
            v = pd.to_numeric(table[col], errors="coerce").to_numpy(dtype=np.float32)
            v = np.where(np.isfinite(v) & (v > 0), v, n_tokens)
            feats.append(np.clip(prefix / np.maximum(v, 1.0), 0.0, 2.0))
    for col in ["strict_before_any_action_mention", "strict_before_final_action_mention"]:
        if col in table.columns:
            feats.append(table[col].fillna(False).astype(bool).to_numpy(dtype=np.float32))
    return np.stack(feats, axis=1).astype(np.float32)


def grouped_splits(table: pd.DataFrame, group_col: str, n_splits: int, seed: int):
    from sklearn.model_selection import GroupKFold

    groups = table[group_col].astype(str).to_numpy()
    unique_groups = np.unique(groups)
    k = min(n_splits, len(unique_groups))
    if k < 2:
        raise ValueError(f"need at least 2 groups for {group_col}")
    # Shuffle group labels deterministically before GroupKFold assignment.
    rng = np.random.default_rng(seed)
    shuffled = unique_groups.copy()
    rng.shuffle(shuffled)
    rank = {g: i for i, g in enumerate(shuffled)}
    sorted_idx = np.argsort([rank[g] for g in groups])
    inv = np.empty_like(sorted_idx)
    inv[sorted_idx] = np.arange(len(sorted_idx))
    dummy = np.zeros(len(table), dtype=np.int8)
    for split_id, (train_s, test_s) in enumerate(GroupKFold(n_splits=k).split(dummy[sorted_idx], groups=groups[sorted_idx])):
        yield split_id, sorted_idx[train_s], sorted_idx[test_s]


def train_val_split(table: pd.DataFrame, train_rows: np.ndarray, group_col: str, seed: int) -> tuple[np.ndarray, np.ndarray]:
    groups = table.iloc[train_rows][group_col].astype(str).to_numpy()
    unique = np.unique(groups)
    rng = np.random.default_rng(seed)
    rng.shuffle(unique)
    n_val = max(1, int(round(0.18 * len(unique))))
    val_groups = set(unique[:n_val])
    mask = np.asarray([g in val_groups for g in groups], dtype=bool)
    return train_rows[~mask], train_rows[mask]


def compute_scaler(x: np.ndarray, rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    sub = x[rows].astype(np.float32)
    mean = sub.mean(axis=0, dtype=np.float64).astype(np.float32)
    scale = sub.std(axis=0, dtype=np.float64).astype(np.float32)
    scale = np.where(scale < 1e-8, 1.0, scale).astype(np.float32)
    return mean, scale


def make_sequences(table: pd.DataFrame, rows: np.ndarray) -> list[np.ndarray]:
    temp = table.iloc[np.asarray(rows, dtype=np.int64)][["decision_id", "prefix_token_index"]].copy()
    temp["_pos"] = np.asarray(rows, dtype=np.int64)
    temp = temp.sort_values(["decision_id", "prefix_token_index", "_pos"])
    return [g["_pos"].to_numpy(dtype=np.int64) for _, g in temp.groupby("decision_id", sort=False)]


class SequenceDataset:
    def __init__(self, x: np.ndarray, y: np.ndarray, sequences: list[np.ndarray], mean: np.ndarray, scale: np.ndarray):
        self.x = x
        self.y = y
        self.sequences = sequences
        self.mean = mean.astype(np.float32)
        self.scale = scale.astype(np.float32)

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, i: int):
        rows = self.sequences[i]
        x = ((self.x[rows] - self.mean) / self.scale).astype(np.float32)
        return x, self.y[rows].astype(np.int64), rows.astype(np.int64)


def collate(batch):
    import torch

    max_len = max(item[0].shape[0] for item in batch)
    dim = batch[0][0].shape[1]
    x = np.zeros((len(batch), max_len, dim), dtype=np.float32)
    y = np.full((len(batch), max_len), -100, dtype=np.int64)
    rows = np.full((len(batch), max_len), -1, dtype=np.int64)
    pad = np.ones((len(batch), max_len), dtype=bool)
    for i, (xb, yb, rb) in enumerate(batch):
        n = len(yb)
        x[i, :n] = xb
        y[i, :n] = yb
        rows[i, :n] = rb
        pad[i, :n] = False
    return torch.tensor(x), torch.tensor(y), torch.tensor(rows), torch.tensor(pad)


def build_model(input_dim: int, spec: ArchSpec, max_len: int, dropout: float):
    import torch
    import torch.nn as nn

    class GRUModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.input = nn.Sequential(nn.LayerNorm(input_dim), nn.Linear(input_dim, spec.hidden), nn.GELU())
            self.gru = nn.GRU(
                input_size=spec.hidden,
                hidden_size=spec.hidden,
                num_layers=spec.layers,
                dropout=dropout if spec.layers > 1 else 0.0,
                batch_first=True,
                bidirectional=spec.bidirectional,
            )
            out_dim = spec.hidden * (2 if spec.bidirectional else 1)
            self.norm = nn.LayerNorm(out_dim)
            self.head = nn.Linear(out_dim, len(ACTIONS))

        def forward(self, x, pad_mask):
            h = self.input(x)
            lengths = (~pad_mask).sum(dim=1).detach().cpu()
            packed = nn.utils.rnn.pack_padded_sequence(h, lengths, batch_first=True, enforce_sorted=False)
            out, _ = self.gru(packed)
            h, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True, total_length=x.shape[1])
            return self.head(self.norm(h))

    class TransformerModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.input = nn.Sequential(nn.LayerNorm(input_dim), nn.Linear(input_dim, spec.hidden))
            self.pos = nn.Parameter(torch.zeros(1, max_len, spec.hidden))
            nn.init.normal_(self.pos, mean=0.0, std=0.02)
            layer = nn.TransformerEncoderLayer(
                d_model=spec.hidden,
                nhead=spec.heads,
                dim_feedforward=4 * spec.hidden,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.encoder = nn.TransformerEncoder(layer, num_layers=spec.layers)
            self.norm = nn.LayerNorm(spec.hidden)
            self.head = nn.Linear(spec.hidden, len(ACTIONS))

        def forward(self, x, pad_mask):
            h = self.input(x) + self.pos[:, : x.shape[1]]
            h = self.encoder(h, src_key_padding_mask=pad_mask)
            return self.head(self.norm(h))

    class TCNBlock(nn.Module):
        def __init__(self, channels: int, dilation: int):
            super().__init__()
            self.dilation = dilation
            self.conv1 = nn.Conv1d(channels, channels, kernel_size=3, dilation=dilation)
            self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, dilation=dilation)
            self.norm1 = nn.LayerNorm(channels)
            self.norm2 = nn.LayerNorm(channels)
            self.drop = nn.Dropout(dropout)

        def same(self, conv, x):
            import torch.nn.functional as F

            pad = 2 * self.dilation
            x = F.pad(x, (pad // 2, pad - pad // 2))
            return conv(x)

        def forward(self, x):
            y = self.same(self.conv1, x.transpose(1, 2)).transpose(1, 2)
            y = self.drop(torch.nn.functional.gelu(self.norm1(y)))
            y = self.same(self.conv2, y.transpose(1, 2)).transpose(1, 2)
            y = self.drop(torch.nn.functional.gelu(self.norm2(y)))
            return x + y

    class TCNModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.input = nn.Sequential(nn.LayerNorm(input_dim), nn.Linear(input_dim, spec.hidden), nn.GELU())
            self.blocks = nn.ModuleList([TCNBlock(spec.hidden, 2**i) for i in range(spec.layers)])
            self.norm = nn.LayerNorm(spec.hidden)
            self.head = nn.Linear(spec.hidden, len(ACTIONS))

        def forward(self, x, pad_mask):
            h = self.input(x)
            for block in self.blocks:
                h = block(h)
            return self.head(self.norm(h))

    if spec.model_type == "gru":
        return GRUModel()
    if spec.model_type == "transformer":
        return TransformerModel()
    if spec.model_type == "tcn":
        return TCNModel()
    raise ValueError(spec.model_type)


def class_weights(y: np.ndarray) -> np.ndarray:
    counts = np.bincount(y, minlength=len(ACTIONS)).astype(np.float64)
    counts = np.maximum(counts, 1.0)
    weights = counts.sum() / (len(ACTIONS) * counts)
    return (weights / weights.mean()).astype(np.float32)


def metrics_np(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

    mat = np.zeros((len(ACTIONS), len(ACTIONS)), dtype=int)
    for t, p in zip(y_true, y_pred):
        mat[int(t), int(p)] += 1
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "confusion_matrix_json": json.dumps(mat.tolist()),
    }


def predict(model, ds: SequenceDataset, batch_size: int, device) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    import torch
    from torch.utils.data import DataLoader

    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collate)
    pred: dict[int, int] = {}
    true: dict[int, int] = {}
    prob: dict[int, np.ndarray] = {}
    model.eval()
    with torch.no_grad():
        for xb, yb, rows, pad in loader:
            logits = model(xb.to(device), pad.to(device))
            probs = torch.softmax(logits, dim=-1).detach().cpu().numpy()
            pp = probs.argmax(axis=-1)
            y_np = yb.numpy()
            r_np = rows.numpy()
            valid = r_np >= 0
            for row, yy, pred_i, prob_i in zip(r_np[valid].reshape(-1), y_np[valid].reshape(-1), pp[valid].reshape(-1), probs[valid].reshape(-1, len(ACTIONS))):
                pred[int(row)] = int(pred_i)
                true[int(row)] = int(yy)
                prob[int(row)] = np.asarray(prob_i, dtype=np.float32)
    ordered = np.asarray(sorted(pred), dtype=np.int64)
    return (
        ordered,
        np.asarray([true[int(r)] for r in ordered], dtype=np.int64),
        np.asarray([pred[int(r)] for r in ordered], dtype=np.int64),
        np.stack([prob[int(r)] for r in ordered], axis=0).astype(np.float32),
    )


def train_one(
    train_ds: SequenceDataset,
    val_ds: SequenceDataset,
    input_dim: int,
    spec: ArchSpec,
    max_len: int,
    args: argparse.Namespace,
    device,
    seed: int,
):
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader

    torch.manual_seed(seed)
    np.random.seed(seed)
    model = build_model(input_dim, spec, max_len, args.dropout).to(device)
    y_train = np.concatenate([train_ds.y[seq] for seq in train_ds.sequences])
    weights = torch.tensor(class_weights(y_train), dtype=torch.float32, device=device) if args.loss_weighting == "balanced" else None
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    generator = torch.Generator()
    generator.manual_seed(seed)
    loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, generator=generator, collate_fn=collate)
    best_state = None
    best_info: dict[str, Any] = {}
    best_score = -np.inf
    bad = 0
    best_epoch = -1
    for epoch in range(args.max_epochs):
        model.train()
        for xb, yb, _rows, pad in loader:
            logits = model(xb.to(device), pad.to(device))
            loss = F.cross_entropy(
                logits.reshape(-1, len(ACTIONS)),
                yb.to(device).reshape(-1),
                ignore_index=-100,
                weight=weights,
                label_smoothing=float(args.label_smoothing),
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
        _, y_val, p_val, _ = predict(model, val_ds, args.batch_size, device)
        m = metrics_np(y_val, p_val)
        score = m["balanced_accuracy"]
        if score > best_score:
            best_score = score
            best_epoch = epoch
            best_info = {f"validation_{k}": v for k, v in m.items() if k != "confusion_matrix_json"}
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= args.patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    best_info["best_epoch"] = int(best_epoch)
    best_info["validation_score"] = float(best_score)
    return model, best_info


def viterbi(rows: pd.DataFrame, probs: np.ndarray, transition_weight: float) -> np.ndarray:
    probs = np.clip(probs, 1e-12, None)
    probs = probs / probs.sum(axis=1, keepdims=True)
    logp = np.log(probs)
    ordered = rows[["decision_id", "prefix_token_index"]].copy()
    ordered["_pos"] = np.arange(len(rows), dtype=np.int64)
    ordered = ordered.sort_values(["decision_id", "prefix_token_index", "_pos"])
    pred = np.zeros(len(rows), dtype=np.int64)
    stay = transition_weight * np.eye(len(ACTIONS), dtype=np.float64)
    for _, group in ordered.groupby("decision_id", sort=False):
        pos = group["_pos"].to_numpy(dtype=np.int64)
        emit = logp[pos]
        dp = emit[0].copy()
        back = np.zeros_like(emit, dtype=np.int64)
        for t in range(1, len(pos)):
            score = dp[:, None] + stay
            back[t] = score.argmax(axis=0)
            dp = emit[t] + score.max(axis=0)
        seq = np.zeros(len(pos), dtype=np.int64)
        seq[-1] = int(dp.argmax())
        for t in range(len(pos) - 1, 0, -1):
            seq[t - 1] = back[t, seq[t]]
        pred[pos] = seq
    return pred


def choose_tw(rows: pd.DataFrame, y_true: np.ndarray, probs: np.ndarray, grid: list[float]) -> tuple[float, float]:
    best_tw = 0.0
    best_bal = -np.inf
    for tw in grid:
        pred = probs.argmax(axis=1) if tw == 0 else viterbi(rows, probs, tw)
        bal = metrics_np(y_true, pred)["balanced_accuracy"]
        if bal > best_bal:
            best_bal = float(bal)
            best_tw = float(tw)
    return best_tw, best_bal


def main() -> None:
    args = parse_args()
    out_dir = ensure_dir(args.output_dir)
    metrics_path = out_dir / "oof_probability_sequence_selector_metrics.csv"
    if metrics_path.exists() and not args.force:
        raise SystemExit(f"{metrics_path} exists; pass --force")

    import torch

    device = choose_device(args.device)
    table, x, feature_names = make_base_table(args)
    y = labels_to_idx(table["top_action"])
    specs = [parse_architecture(text) for text in args.architectures]
    transition_grid = [float(x) for x in args.transition_weights.split(",") if x.strip()]
    splits = list(grouped_splits(table, args.group_col, 5, args.random_seed))
    if args.limit_splits is not None:
        splits = splits[: args.limit_splits]
    print(
        json.dumps(
            {
                "device": str(device),
                "rows": int(len(table)),
                "decisions": int(table["decision_id"].nunique()),
                "groups": int(table[args.group_col].nunique()),
                "input_dim": int(x.shape[1]),
                "base_models": [spec.name for spec in MODEL_SPECS],
                "architectures": [spec.name for spec in specs],
            },
            indent=2,
        ),
        flush=True,
    )

    metric_rows: list[dict[str, Any]] = []
    pred_frames: list[pd.DataFrame] = []
    max_len_global = int(table.groupby("decision_id").size().max())
    for split_id, train_pool, test_rows in splits:
        train_rows, val_rows = train_val_split(table, train_pool, args.group_col, seed=args.random_seed + split_id)
        mean, scale = compute_scaler(x, train_rows)
        train_ds = SequenceDataset(x, y, make_sequences(table, train_rows), mean, scale)
        val_ds = SequenceDataset(x, y, make_sequences(table, val_rows), mean, scale)
        test_ds = SequenceDataset(x, y, make_sequences(table, test_rows), mean, scale)
        val_ordered_rows = None
        print(f"split={split_id} train={len(train_rows)} val={len(val_rows)} test={len(test_rows)}", flush=True)
        for spec in specs:
            seed = stable_seed(spec.name, split_id, base=args.random_seed)
            print(f"Training split={split_id} arch={spec.name}", flush=True)
            model, info = train_one(train_ds, val_ds, x.shape[1], spec, max_len_global, args, device, seed)
            val_row_ids, y_val, p_val, prob_val = predict(model, val_ds, args.batch_size, device)
            test_row_ids, y_test, p_test, prob_test = predict(model, test_ds, args.batch_size, device)
            val_rows_df = table.iloc[val_row_ids].reset_index(drop=True)
            test_rows_df = table.iloc[test_row_ids].reset_index(drop=True)
            tw, val_tw_bal = choose_tw(val_rows_df, y_val, prob_val, transition_grid)
            p_test_v = viterbi(test_rows_df, prob_test, tw) if tw > 0 else p_test
            for method, pred, tw_used, val_bal in [
                ("raw", p_test, 0.0, info.get("validation_balanced_accuracy", np.nan)),
                ("viterbi_val_tw", p_test_v, tw, val_tw_bal),
            ]:
                m = metrics_np(y_test, pred)
                row = {
                    "split_id": int(split_id),
                    "architecture": spec.name,
                    "method": method,
                    "n_train": int(len(train_rows)),
                    "n_val": int(len(val_rows)),
                    "n_test": int(len(test_rows)),
                    "best_epoch": int(info.get("best_epoch", -1)),
                    "validation_balanced_accuracy": float(val_bal),
                    "transition_weight": float(tw_used),
                    **m,
                }
                metric_rows.append(row)
                pred_df = pd.DataFrame(
                    {
                        "split_id": int(split_id),
                        "architecture": spec.name,
                        "method": method,
                        "row_id": test_row_ids.astype(np.int64),
                        "true_top_action": idx_to_labels(y_test),
                        "pred_top_action": idx_to_labels(pred),
                    }
                )
                for i, action in enumerate(ACTIONS):
                    pred_df[f"prob_{action}"] = prob_test[:, i]
                pred_frames.append(pred_df)
                print(
                    f"RESULT split={split_id} arch={spec.name} method={method} "
                    f"bal={m['balanced_accuracy']:.6f} acc={m['accuracy']:.6f} tw={tw_used}",
                    flush=True,
                )
            del model
            gc.collect()
            if str(device) == "mps" and hasattr(torch, "mps"):
                torch.mps.empty_cache()
        pd.DataFrame(metric_rows).to_csv(metrics_path, index=False)

    metrics_df = pd.DataFrame(metric_rows)
    summary = (
        metrics_df.groupby(["architecture", "method"], as_index=False)
        .agg(
            folds=("split_id", "nunique"),
            mean_balanced_accuracy=("balanced_accuracy", "mean"),
            std_balanced_accuracy=("balanced_accuracy", "std"),
            mean_accuracy=("accuracy", "mean"),
            mean_macro_f1=("macro_f1", "mean"),
        )
        .sort_values("mean_balanced_accuracy", ascending=False)
    )
    metrics_df.to_csv(metrics_path, index=False)
    summary.to_csv(out_dir / "oof_probability_sequence_selector_summary.csv", index=False)
    pd.concat(pred_frames, ignore_index=True).to_parquet(out_dir / "oof_probability_sequence_selector_predictions.parquet", index=False)
    write_json(
        out_dir / "oof_probability_sequence_selector_config.json",
        {
            "base_models": [spec.__dict__ for spec in MODEL_SPECS],
            "feature_names": feature_names,
            "architectures": args.architectures,
            "group_col": args.group_col,
            "note": "Input features are base OOF probabilities/log-probabilities/votes/metadata only; behavior q and labels are not input features.",
        },
    )
    lines = [
        "# OOF Probability Sequence Selector",
        "",
        "Inputs are held-out base-model predictions only. Meta evaluation uses grouped folds by trajectory_id.",
        "",
        "## Summary",
        "",
        summary.to_markdown(index=False),
        "",
        "## Per-Fold Metrics",
        "",
        metrics_df.to_markdown(index=False),
    ]
    save_markdown(out_dir / "oof_probability_sequence_selector_report.md", "\n".join(lines))
    print(
        json.dumps(
            {
                "summary": str(out_dir / "oof_probability_sequence_selector_summary.csv"),
                "metrics": str(metrics_path),
                "predictions": str(out_dir / "oof_probability_sequence_selector_predictions.parquet"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
