#!/usr/bin/env python3
"""Train a causal last-token-query attention probe for forced q-top action.

The frozen language-model activations are the only model inputs.  For every
prefix, the final generated token supplies the attention query.  Five small
linear experts provide action evidence from causal activation summaries:

* the final token;
* the last 100 generated tokens;
* the cumulative prefix mean and its decision-start residual;
* a multi-scale segment view (prefix, last 20, and last 100, plus residuals);
* a pairwise q-order expert over the last 20 tokens.

The attention gate is query-conditioned and action-specific.  It attends over
the expert evidence, so the complete model remains a compact readout on top of
a frozen backbone.  Evaluation is trajectory-grouped.  Prefix metadata is
used only for indexing, splitting, labels, and reporting, never as model input.
"""

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import math
import random
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.conceptual_belief_space.config import ACTIONS, DEFAULT_SEED

FIXED_SCRIPT = Path(__file__).with_name("47_nested_fixed_fusion_validation.py")
spec = importlib.util.spec_from_file_location("last_token_attention_fixed_impl", FIXED_SCRIPT)
fixed = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = fixed
assert spec and spec.loader
spec.loader.exec_module(fixed)

seg = fixed.seg_impl
Q_COLS = [f"q_{action}" for action in ACTIONS]
SUBSET_COLS = ["strict_before_any_action_mention", "strict_before_final_action_mention"]
PAIR_INDICES = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]

SOURCE_TERMS: OrderedDict[str, list[str]] = OrderedDict(
    [
        ("last1_ovr", ["raw1"]),
        ("mean100_ovr", ["raw100"]),
        ("prefixmean_ovr", ["rawpfx", "residpfx"]),
        (
            "segment_ovr",
            ["rawpfx", "raw20", "raw100", "residpfx", "resid20", "resid100"],
        ),
        ("identity_pairwise", ["raw20"]),
    ]
)
SOURCE_NAMES = list(SOURCE_TERMS)
OVR_SOURCE_NAMES = [name for name in SOURCE_NAMES if name != "identity_pairwise"]
UNIQUE_TERMS = list(dict.fromkeys(term for terms in SOURCE_TERMS.values() for term in terms))

FORBIDDEN_MODEL_INPUTS = [
    "grid/task features",
    "progress",
    "prefix_token_index",
    "num_cot_tokens",
    "trajectory_id",
    "decision_id",
    "step_id",
    "agent_action",
    "forced answer text/tokens",
    "future activations/tokens",
]


def parse_layers(text: str) -> list[int]:
    return [int(value.strip()) for value in text.split(",") if value.strip()]


def parse_split_ids(text: str, all_ids: list[int]) -> list[int]:
    if text == "all":
        return list(all_ids)
    return [int(value.strip()) for value in text.split(",") if value.strip()]


def action_indices(values: pd.Series) -> np.ndarray:
    mapping = {action: idx for idx, action in enumerate(ACTIONS)}
    out = values.astype(str).map(mapping)
    if out.isna().any():
        raise ValueError(f"Unknown action labels: {sorted(values[out.isna()].astype(str).unique())}")
    return out.to_numpy(dtype=np.int64)


def capped_prefix_rows(
    df: pd.DataFrame,
    indices: np.ndarray,
    max_per_decision: int,
    max_rows: int,
    seed: int,
) -> np.ndarray:
    """Keep evenly spaced causal prefixes per decision for efficient training."""
    indices = np.asarray(indices, dtype=np.int64)
    if max_per_decision > 0:
        selected: list[np.ndarray] = []
        local = df.iloc[indices]
        for _decision_id, positions in local.groupby("decision_id", sort=False).indices.items():
            rows = indices[np.asarray(positions, dtype=np.int64)]
            if len(rows) > max_per_decision:
                take = np.linspace(0, len(rows) - 1, num=max_per_decision, dtype=np.int64)
                rows = rows[take]
            selected.append(rows)
        indices = np.concatenate(selected) if selected else np.asarray([], dtype=np.int64)
    if max_rows > 0 and len(indices) > max_rows:
        rng = np.random.default_rng(seed)
        indices = np.sort(rng.choice(indices, size=max_rows, replace=False))
    return indices.astype(np.int64, copy=False)


def assert_group_disjoint(df: pd.DataFrame, left: np.ndarray, right: np.ndarray, label: str) -> None:
    left_groups = set(df.iloc[left]["trajectory_id"].astype(str))
    right_groups = set(df.iloc[right]["trajectory_id"].astype(str))
    overlap = left_groups & right_groups
    if overlap:
        raise RuntimeError(f"Trajectory leakage in {label}: {sorted(overlap)[:3]}")


def compute_term_scalers(
    arrays: dict[tuple[int, int], np.ndarray],
    layers: list[int],
    train_idx: np.ndarray,
    feature_rows: np.ndarray,
    pre_rows: np.ndarray,
    batch_size: int,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    scalers: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for term in UNIQUE_TERMS:
        print(f"    scaler term={term}", flush=True)
        scalers[term] = seg.compute_scaler(
            arrays,
            [term],
            layers,
            train_idx,
            feature_rows,
            pre_rows,
            batch_size=max(batch_size, 256),
        )
    return scalers


def build_term_tensors(
    arrays: dict[tuple[int, int], np.ndarray],
    layers: list[int],
    indices: np.ndarray,
    feature_rows: np.ndarray,
    pre_rows: np.ndarray,
    scalers: dict[str, tuple[np.ndarray, np.ndarray]],
    device,
    *,
    terms: list[str] | None = None,
) -> dict[str, Any]:
    import torch

    out: dict[str, Any] = {}
    for term in terms or UNIQUE_TERMS:
        mean, scale = scalers[term]
        values = seg.build_batch(arrays, [term], layers, indices, feature_rows, pre_rows)
        values = (values - mean) / scale
        out[term] = torch.as_tensor(values, dtype=torch.float32, device=device)
    return out


def rowz_torch(scores):
    centered = scores - scores.mean(dim=-1, keepdim=True)
    scale = centered.std(dim=-1, keepdim=True, unbiased=False).clamp_min(1e-6)
    return centered / scale


def q_pair_targets(q_tensor):
    import torch

    pair_i = torch.tensor([i for i, _ in PAIR_INDICES], dtype=torch.long, device=q_tensor.device)
    pair_j = torch.tensor([j for _, j in PAIR_INDICES], dtype=torch.long, device=q_tensor.device)
    diff = q_tensor[:, pair_i] - q_tensor[:, pair_j]
    target = (diff > 0).float()
    mask = (diff.abs() > 1e-6).float()
    return target, mask


def masked_pair_bce(pair_logits, q_tensor):
    import torch

    target, mask = q_pair_targets(q_tensor)
    raw = torch.nn.functional.binary_cross_entropy_with_logits(pair_logits, target, reduction="none")
    return (raw * mask).sum() / mask.sum().clamp_min(1.0)


def q_rank_loss(action_logits, q_tensor):
    import torch

    pair_i = torch.tensor([i for i, _ in PAIR_INDICES], dtype=torch.long, device=action_logits.device)
    pair_j = torch.tensor([j for _, j in PAIR_INDICES], dtype=torch.long, device=action_logits.device)
    target, mask = q_pair_targets(q_tensor)
    pair_logits = action_logits[:, pair_i] - action_logits[:, pair_j]
    raw = torch.nn.functional.binary_cross_entropy_with_logits(pair_logits, target, reduction="none")
    return (raw * mask).sum() / mask.sum().clamp_min(1.0)


def soft_q_cross_entropy(action_logits, q_tensor):
    import torch

    target = q_tensor / q_tensor.sum(dim=1, keepdim=True).clamp_min(1e-8)
    return -(target * torch.nn.functional.log_softmax(action_logits, dim=1)).sum(dim=1).mean()


def make_probe_class():
    import torch

    class LastTokenQueryAttentionProbe(torch.nn.Module):
        def __init__(self, term_dim: int, attention_dim: int, dropout: float):
            super().__init__()
            self.term_dim = int(term_dim)
            self.attention_dim = int(attention_dim)
            self.dropout = torch.nn.Dropout(float(dropout))
            self.source_heads = torch.nn.ModuleDict()
            for source_name, terms in SOURCE_TERMS.items():
                out_dim = len(PAIR_INDICES) if source_name == "identity_pairwise" else len(ACTIONS)
                self.source_heads[source_name] = torch.nn.Linear(term_dim * len(terms), out_dim)

            # One query per candidate action, all read from the actual final token.
            self.query_projection = torch.nn.Linear(term_dim, len(ACTIONS) * attention_dim)
            # Keys depend on each expert's current four-action evidence.
            self.key_projection = torch.nn.Linear(
                len(ACTIONS), len(ACTIONS) * attention_dim, bias=False
            )
            self.source_key_embedding = torch.nn.Parameter(
                torch.zeros(len(SOURCE_NAMES), len(ACTIONS), attention_dim)
            )
            initial_gate_bias = torch.tensor([-2.0, 0.0, 0.0, 0.0, -1.0], dtype=torch.float32)
            self.global_attention_bias = torch.nn.Parameter(
                initial_gate_bias[:, None].repeat(1, len(ACTIONS))
            )
            self.output_scale = torch.nn.Parameter(torch.ones(len(ACTIONS)))
            self.action_bias = torch.nn.Parameter(torch.zeros(len(ACTIONS)))

            pair_to_action = torch.zeros(len(PAIR_INDICES), len(ACTIONS))
            for pair_id, (i, j) in enumerate(PAIR_INDICES):
                pair_to_action[pair_id, i] = 1.0
                pair_to_action[pair_id, j] = -1.0
            self.register_buffer("pair_to_action", pair_to_action)

            # Start close to uniform attention; let held-in training data earn
            # query-dependent deviations instead of beginning with a sharp gate.
            torch.nn.init.normal_(self.query_projection.weight, mean=0.0, std=0.005)
            torch.nn.init.zeros_(self.query_projection.bias)
            torch.nn.init.normal_(self.key_projection.weight, mean=0.0, std=0.005)

        def encode_sources(self, term_features: dict[str, Any]):
            raw_scores: dict[str, Any] = {}
            action_scores = []
            pair_logits = None
            for source_name, terms in SOURCE_TERMS.items():
                features = torch.cat([term_features[term] for term in terms], dim=1)
                scores = self.source_heads[source_name](self.dropout(features))
                raw_scores[source_name] = scores
                if source_name == "identity_pairwise":
                    pair_logits = scores
                    scores = scores @ self.pair_to_action
                action_scores.append(scores)
            # Match the established fixed-fusion baseline: normalize each
            # expert row, then turn it into comparable action evidence.
            evidence = torch.softmax(rowz_torch(torch.stack(action_scores, dim=1)), dim=-1)
            return evidence, raw_scores, pair_logits

        def attend(self, evidence, query_features, mode: str = "dynamic"):
            batch = evidence.shape[0]
            if mode in {"fixed_ovr3", "fixed_core4"}:
                selected = (
                    ["mean100_ovr", "prefixmean_ovr", "segment_ovr"]
                    if mode == "fixed_ovr3"
                    else ["mean100_ovr", "prefixmean_ovr", "segment_ovr", "identity_pairwise"]
                )
                weights = torch.zeros_like(evidence)
                for source_name in selected:
                    weights[:, SOURCE_NAMES.index(source_name), :] = 1.0 / len(selected)
            elif mode == "uniform":
                weights = torch.full_like(evidence, 1.0 / len(SOURCE_NAMES))
            elif mode == "global":
                weights = torch.softmax(self.global_attention_bias, dim=0).unsqueeze(0).expand(batch, -1, -1)
            elif mode == "dynamic":
                query = self.query_projection(self.dropout(query_features)).reshape(
                    batch, len(ACTIONS), self.attention_dim
                )
                keys = self.key_projection(evidence.reshape(-1, len(ACTIONS))).reshape(
                    batch, len(SOURCE_NAMES), len(ACTIONS), self.attention_dim
                )
                keys = keys + self.source_key_embedding.unsqueeze(0)
                attention_logits = (
                    query.unsqueeze(1) * keys
                ).sum(dim=-1) / math.sqrt(float(self.attention_dim))
                attention_logits = attention_logits + self.global_attention_bias.unsqueeze(0)
                weights = torch.softmax(attention_logits, dim=1)
            else:
                raise ValueError(f"Unknown attention mode: {mode}")
            fused = (weights * evidence).sum(dim=1)
            # Fixed controls are the literal pre-specified probability means;
            # all learned gate variants retain the train-side calibration.
            logits = fused if mode.startswith("fixed_") else fused * self.output_scale + self.action_bias
            return logits, weights

        def forward(self, term_features: dict[str, Any], mode: str = "dynamic", query_features=None):
            evidence, raw_scores, pair_logits = self.encode_sources(term_features)
            query = term_features["raw1"] if query_features is None else query_features
            logits, weights = self.attend(evidence, query, mode=mode)
            return logits, weights, evidence, raw_scores, pair_logits

    return LastTokenQueryAttentionProbe


def parameter_counts(model) -> dict[str, int]:
    attention_prefixes = (
        "query_projection",
        "key_projection",
        "source_key_embedding",
        "global_attention_bias",
        "output_scale",
        "action_bias",
    )
    total = sum(parameter.numel() for parameter in model.parameters())
    attention = sum(
        parameter.numel()
        for name, parameter in model.named_parameters()
        if name.startswith(attention_prefixes)
    )
    return {
        "total_trainable_parameters": int(total),
        "attention_trainable_parameters": int(attention),
        "expert_trainable_parameters": int(total - attention),
    }


def compute_loss(
    logits,
    evidence,
    raw_scores,
    pair_logits,
    y_tensor,
    q_tensor,
    class_weights,
    pos_weight,
    *,
    gate_loss: str,
    expert_weight: float,
    pair_weight: float,
    q_rank_weight: float,
    q_soft_weight: float,
):
    import torch

    target = torch.zeros((len(y_tensor), len(ACTIONS)), dtype=torch.float32, device=logits.device)
    target.scatter_(1, y_tensor[:, None], 1.0)
    if gate_loss == "ovr_bce":
        fused_loss = torch.nn.functional.binary_cross_entropy_with_logits(
            logits, target, pos_weight=pos_weight
        )
    elif gate_loss == "balanced_ce":
        fused_loss = torch.nn.functional.cross_entropy(logits, y_tensor, weight=class_weights)
    else:
        raise ValueError(f"Unknown gate loss: {gate_loss}")
    expert_losses = [
        torch.nn.functional.binary_cross_entropy_with_logits(
            raw_scores[name], target, pos_weight=pos_weight
        )
        for name in OVR_SOURCE_NAMES
    ]
    # These heads have disjoint parameters.  Summing gives every head the same
    # gradient scale it receives when trained as an independent linear probe.
    expert_loss = torch.stack(expert_losses).sum()
    pair_loss = masked_pair_bce(pair_logits, q_tensor)
    rank_loss = q_rank_loss(logits, q_tensor)
    soft_loss = soft_q_cross_entropy(logits, q_tensor)
    loss = (
        fused_loss
        + float(expert_weight) * expert_loss
        + float(pair_weight) * pair_loss
        + float(q_rank_weight) * rank_loss
        + float(q_soft_weight) * soft_loss
    )
    return loss, {
        "loss": float(loss.detach().cpu()),
        "fused_loss": float(fused_loss.detach().cpu()),
        "expert_loss": float(expert_loss.detach().cpu()),
        "pair_loss": float(pair_loss.detach().cpu()),
        "rank_loss": float(rank_loss.detach().cpu()),
        "soft_q_loss": float(soft_loss.detach().cpu()),
    }


def compute_source_pretrain_loss(raw_scores, pair_logits, y_tensor, q_tensor, pos_weight, pair_weight: float):
    """Train the memory experts with their established independent objectives."""
    import torch

    target = torch.zeros((len(y_tensor), len(ACTIONS)), dtype=torch.float32, device=y_tensor.device)
    target.scatter_(1, y_tensor[:, None], 1.0)
    expert_losses = [
        torch.nn.functional.binary_cross_entropy_with_logits(
            raw_scores[name], target, pos_weight=pos_weight
        )
        for name in OVR_SOURCE_NAMES
    ]
    expert_loss = torch.stack(expert_losses).sum()
    pair_loss = masked_pair_bce(pair_logits, q_tensor)
    loss = expert_loss + float(pair_weight) * pair_loss
    return loss, float(expert_loss.detach().cpu()), float(pair_loss.detach().cpu())


def predict_probe(
    model,
    arrays,
    layers,
    indices,
    feature_rows,
    pre_rows,
    scalers,
    batch_size,
    device,
    *,
    modes: tuple[str, ...] = ("dynamic",),
    shuffled_query_indices: np.ndarray | None = None,
) -> dict[str, Any]:
    import torch

    score_parts: dict[str, list[np.ndarray]] = {mode: [] for mode in modes}
    source_parts: list[np.ndarray] = []
    attention_parts: list[np.ndarray] = []
    shuffled_parts: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(indices), batch_size):
            batch_idx = indices[start : start + batch_size]
            term_features = build_term_tensors(
                arrays,
                layers,
                batch_idx,
                feature_rows,
                pre_rows,
                scalers,
                device,
            )
            evidence, _raw_scores, _pair_logits = model.encode_sources(term_features)
            for mode in modes:
                scores, weights = model.attend(evidence, term_features["raw1"], mode=mode)
                score_parts[mode].append(scores.cpu().numpy().astype(np.float32))
                if mode == "dynamic":
                    attention_parts.append(weights.cpu().numpy().astype(np.float32))
            source_parts.append(evidence.cpu().numpy().astype(np.float32))
            if shuffled_query_indices is not None:
                query_idx = shuffled_query_indices[start : start + len(batch_idx)]
                shuffled_query = build_term_tensors(
                    arrays,
                    layers,
                    query_idx,
                    feature_rows,
                    pre_rows,
                    scalers,
                    device,
                    terms=["raw1"],
                )["raw1"]
                shuffled_scores, _ = model.attend(evidence, shuffled_query, mode="dynamic")
                shuffled_parts.append(shuffled_scores.cpu().numpy().astype(np.float32))
    empty_scores = np.zeros((0, len(ACTIONS)), dtype=np.float32)
    out: dict[str, Any] = {
        mode: np.concatenate(parts, axis=0) if parts else empty_scores.copy()
        for mode, parts in score_parts.items()
    }
    out["source_evidence"] = (
        np.concatenate(source_parts, axis=0)
        if source_parts
        else np.zeros((0, len(SOURCE_NAMES), len(ACTIONS)), dtype=np.float32)
    )
    out["attention"] = (
        np.concatenate(attention_parts, axis=0)
        if attention_parts
        else np.zeros((0, len(SOURCE_NAMES), len(ACTIONS)), dtype=np.float32)
    )
    if shuffled_query_indices is not None:
        out["shuffled_query"] = np.concatenate(shuffled_parts, axis=0) if shuffled_parts else empty_scores
    return out


def train_fold(
    arrays,
    layers,
    df,
    y,
    q_all,
    train_idx,
    val_idx,
    test_idx,
    feature_rows,
    pre_rows,
    args,
    split_id: int,
):
    import torch

    random.seed(args.seed + split_id)
    np.random.seed(args.seed + split_id)
    torch.manual_seed(args.seed + split_id)
    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu"))
    effective_train = capped_prefix_rows(
        df,
        train_idx,
        args.max_train_prefixes_per_decision,
        args.max_train_rows,
        args.seed + split_id,
    )
    print(
        f"  effective train prefixes={len(effective_train)} from {len(train_idx)}; "
        f"val={len(val_idx)} test={len(test_idx)} device={device}",
        flush=True,
    )
    scalers = compute_term_scalers(
        arrays,
        layers,
        effective_train,
        feature_rows,
        pre_rows,
        args.batch_size,
    )
    term_dim = len(next(iter(scalers.values()))[0])
    Probe = make_probe_class()
    model = Probe(term_dim, args.attention_dim, args.dropout).to(device)
    counts = np.bincount(y[effective_train], minlength=len(ACTIONS)).astype(np.float32)
    class_weight_np = counts.sum() / np.clip(counts, 1.0, None)
    class_weight_np /= class_weight_np.mean()
    class_weights = torch.tensor(class_weight_np, dtype=torch.float32, device=device)
    pos_weight = torch.tensor(
        (len(effective_train) - counts) / np.clip(counts, 1.0, None),
        dtype=torch.float32,
        device=device,
    )
    rng = np.random.default_rng(args.seed + split_id)
    validation_rows = val_idx if len(val_idx) else effective_train
    history: list[dict[str, Any]] = []

    # Stage 1: independently establish useful causal memory experts.  This
    # prevents the flexible gate loss from sacrificing individual decoders.
    source_optimizer = torch.optim.AdamW(
        model.source_heads.parameters(),
        lr=args.source_pretrain_lr,
        weight_decay=args.weight_decay,
    )
    source_best_states: dict[str, dict[str, Any]] = {}
    source_best_values = {name: -1.0 for name in SOURCE_NAMES}
    source_best_epochs = {name: -1 for name in SOURCE_NAMES}
    source_stale = {name: 0 for name in SOURCE_NAMES}
    for source_epoch in range(args.source_pretrain_epochs):
        order = rng.permutation(effective_train)
        model.train()
        total_source_loss = 0.0
        total_expert_loss = 0.0
        total_pair_loss = 0.0
        seen = 0
        for start in range(0, len(order), args.batch_size):
            idx = order[start : start + args.batch_size]
            term_features = build_term_tensors(
                arrays,
                layers,
                idx,
                feature_rows,
                pre_rows,
                scalers,
                device,
            )
            _evidence, raw_scores, pair_logits = model.encode_sources(term_features)
            y_tensor = torch.tensor(y[idx], dtype=torch.long, device=device)
            q_tensor = torch.tensor(q_all[idx], dtype=torch.float32, device=device)
            source_loss, expert_loss, pair_loss = compute_source_pretrain_loss(
                raw_scores,
                pair_logits,
                y_tensor,
                q_tensor,
                pos_weight,
                args.source_pretrain_pair_weight,
            )
            source_optimizer.zero_grad(set_to_none=True)
            source_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.source_heads.parameters(), args.grad_clip)
            source_optimizer.step()
            total_source_loss += float(source_loss.detach().cpu()) * len(idx)
            total_expert_loss += expert_loss * len(idx)
            total_pair_loss += pair_loss * len(idx)
            seen += len(idx)

        val_out = predict_probe(
            model,
            arrays,
            layers,
            validation_rows,
            feature_rows,
            pre_rows,
            scalers,
            args.eval_batch_size,
            device,
            modes=("fixed_ovr3",),
        )
        val_pred = val_out["fixed_ovr3"].argmax(axis=1)
        val_bal = fixed.balanced_accuracy(y[validation_rows], val_pred)
        val_acc = float((val_pred == y[validation_rows]).mean())
        source_balanced_values = {
            name: fixed.balanced_accuracy(
                y[validation_rows], val_out["source_evidence"][:, source_id, :].argmax(axis=1)
            )
            for source_id, name in enumerate(SOURCE_NAMES)
        }
        source_balanced = {
            f"validation_{name}_balanced_accuracy": value
            for name, value in source_balanced_values.items()
        }
        history.append(
            {
                "split_id": split_id,
                "stage": "source_pretrain",
                "epoch": source_epoch,
                "train_rows": int(len(effective_train)),
                "validation_rows": int(len(validation_rows)),
                "validation_balanced_accuracy": float(val_bal),
                "validation_accuracy": val_acc,
                "loss": total_source_loss / max(seen, 1),
                "expert_loss": total_expert_loss / max(seen, 1),
                "pair_loss": total_pair_loss / max(seen, 1),
                **source_balanced,
            }
        )
        print(
            f"    source_epoch={source_epoch + 1}/{args.source_pretrain_epochs} "
            f"loss={total_source_loss / max(seen, 1):.4f} "
            f"ovr3_val_bal={val_bal:.4f} val_acc={val_acc:.4f}",
            flush=True,
        )
        for source_name, source_value in source_balanced_values.items():
            if source_value > source_best_values[source_name] + args.min_delta:
                source_best_values[source_name] = float(source_value)
                source_best_epochs[source_name] = int(source_epoch)
                source_best_states[source_name] = copy.deepcopy(
                    {
                        key: value.detach().cpu()
                        for key, value in model.source_heads[source_name].state_dict().items()
                    }
                )
                source_stale[source_name] = 0
            else:
                source_stale[source_name] += 1
        if all(value >= args.source_pretrain_patience for value in source_stale.values()):
            break

    missing_sources = [name for name in SOURCE_NAMES if name not in source_best_states]
    if missing_sources:
        raise RuntimeError(f"Source pretraining did not checkpoint: {missing_sources}")
    for source_name, state in source_best_states.items():
        model.source_heads[source_name].load_state_dict(state)
    model.to(device)
    combined_val = predict_probe(
        model,
        arrays,
        layers,
        validation_rows,
        feature_rows,
        pre_rows,
        scalers,
        args.eval_batch_size,
        device,
        modes=("fixed_ovr3",),
    )["fixed_ovr3"]
    pretrain_best_val = fixed.balanced_accuracy(y[validation_rows], combined_val.argmax(axis=1))
    pretrain_best_epoch = max(source_best_epochs.values())
    pretrain_best_state = copy.deepcopy(
        {key: value.detach().cpu() for key, value in model.state_dict().items()}
    )
    for parameter in model.source_heads.parameters():
        parameter.requires_grad_(False)

    # Stage 2: train only the compact last-token-query attention gate.
    attention_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        attention_parameters,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    initial_gate_out = predict_probe(
        model,
        arrays,
        layers,
        validation_rows,
        feature_rows,
        pre_rows,
        scalers,
        args.eval_batch_size,
        device,
        modes=("dynamic",),
    )["dynamic"]
    initial_gate_pred = initial_gate_out.argmax(axis=1)
    initial_gate_val = fixed.balanced_accuracy(y[validation_rows], initial_gate_pred)
    initial_gate_acc = float((initial_gate_pred == y[validation_rows]).mean())
    history.append(
        {
            "split_id": split_id,
            "stage": "attention_gate_initial",
            "epoch": -1,
            "train_rows": int(len(effective_train)),
            "validation_rows": int(len(validation_rows)),
            "validation_balanced_accuracy": float(initial_gate_val),
            "validation_accuracy": initial_gate_acc,
        }
    )
    print(
        f"    gate_initial val_bal={initial_gate_val:.4f} val_acc={initial_gate_acc:.4f}",
        flush=True,
    )
    best_state = copy.deepcopy(
        {key: value.detach().cpu() for key, value in model.state_dict().items()}
    )
    best_val = float(initial_gate_val)
    best_epoch = -1
    stale = 0

    for epoch in range(args.epochs):
        order = rng.permutation(effective_train)
        model.train()
        totals: dict[str, float] = {}
        seen = 0
        for start in range(0, len(order), args.batch_size):
            idx = order[start : start + args.batch_size]
            term_features = build_term_tensors(
                arrays,
                layers,
                idx,
                feature_rows,
                pre_rows,
                scalers,
                device,
            )
            logits, _weights, evidence, raw_scores, pair_logits = model(term_features, mode="dynamic")
            y_tensor = torch.tensor(y[idx], dtype=torch.long, device=device)
            q_tensor = torch.tensor(q_all[idx], dtype=torch.float32, device=device)
            loss, parts = compute_loss(
                logits,
                evidence,
                raw_scores,
                pair_logits,
                y_tensor,
                q_tensor,
                class_weights,
                pos_weight,
                gate_loss=args.gate_loss,
                expert_weight=args.expert_loss_weight,
                pair_weight=args.pair_loss_weight,
                q_rank_weight=args.q_rank_loss_weight,
                q_soft_weight=args.q_soft_loss_weight,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            for key, value in parts.items():
                totals[key] = totals.get(key, 0.0) + value * len(idx)
            seen += len(idx)

        validation_rows = val_idx if len(val_idx) else effective_train
        val_out = predict_probe(
            model,
            arrays,
            layers,
            validation_rows,
            feature_rows,
            pre_rows,
            scalers,
            args.eval_batch_size,
            device,
            modes=("dynamic",),
        )
        val_pred = val_out["dynamic"].argmax(axis=1)
        val_bal = fixed.balanced_accuracy(y[validation_rows], val_pred)
        val_acc = float((val_pred == y[validation_rows]).mean())
        row = {
            "split_id": split_id,
            "stage": "attention_gate",
            "epoch": epoch,
            "train_rows": int(len(effective_train)),
            "validation_rows": int(len(validation_rows)),
            "validation_balanced_accuracy": float(val_bal),
            "validation_accuracy": val_acc,
            **{key: value / max(seen, 1) for key, value in totals.items()},
        }
        history.append(row)
        print(
            f"    gate_epoch={epoch + 1}/{args.epochs} loss={row['loss']:.4f} "
            f"val_bal={val_bal:.4f} val_acc={val_acc:.4f}",
            flush=True,
        )
        if val_bal > best_val + args.min_delta:
            best_val = float(val_bal)
            best_epoch = int(epoch)
            best_state = copy.deepcopy({key: value.detach().cpu() for key, value in model.state_dict().items()})
            stale = 0
        else:
            stale += 1
            if stale >= args.patience:
                break

    if best_state is None:
        raise RuntimeError("Training did not produce a checkpoint")
    model.load_state_dict(best_state)
    model.to(device)
    test_rng = np.random.default_rng(args.seed + 100_000 + split_id)
    shuffled_query_indices = np.asarray(test_idx, dtype=np.int64).copy()
    test_rng.shuffle(shuffled_query_indices)
    outputs = predict_probe(
        model,
        arrays,
        layers,
        test_idx,
        feature_rows,
        pre_rows,
        scalers,
        args.eval_batch_size,
        device,
        modes=("dynamic", "global", "uniform", "fixed_ovr3", "fixed_core4"),
        shuffled_query_indices=shuffled_query_indices,
    )
    info = {
        "best_epoch": best_epoch,
        "validation_balanced_accuracy": best_val,
        "source_pretrain_best_epoch": pretrain_best_epoch,
        "source_pretrain_validation_balanced_accuracy": pretrain_best_val,
        "source_best_epochs_json": json.dumps(source_best_epochs, sort_keys=True),
        "source_best_validation_balanced_accuracy_json": json.dumps(source_best_values, sort_keys=True),
        "device": str(device),
        "term_dim": int(term_dim),
        "effective_train_rows": int(len(effective_train)),
        **parameter_counts(model),
    }
    checkpoint = {
        "state_dict": best_state,
        "scalers": scalers,
        "layers": layers,
        "actions": list(ACTIONS),
        "source_terms": dict(SOURCE_TERMS),
        "attention_dim": int(args.attention_dim),
        "dropout": float(args.dropout),
        "split_id": int(split_id),
        "train_trajectory_ids": sorted(df.iloc[train_idx]["trajectory_id"].astype(str).unique().tolist()),
        "validation_trajectory_ids": sorted(df.iloc[val_idx]["trajectory_id"].astype(str).unique().tolist()),
        "test_trajectory_ids": sorted(df.iloc[test_idx]["trajectory_id"].astype(str).unique().tolist()),
        "info": info,
    }
    return outputs, info, history, checkpoint


def add_all_metrics(
    metric_rows: list[dict[str, Any]],
    df: pd.DataFrame,
    test_idx: np.ndarray,
    outputs: dict[str, Any],
    split_id: int,
    info: dict[str, Any],
) -> None:
    variants: OrderedDict[str, np.ndarray] = OrderedDict(
        [
            ("last_token_query_attention", outputs["dynamic"]),
            ("control_global_attention", outputs["global"]),
            ("control_uniform_attention", outputs["uniform"]),
            ("control_fixed_ovr3_fusion", outputs["fixed_ovr3"]),
            ("control_fixed_core4_fusion", outputs["fixed_core4"]),
            ("control_shuffled_last_token_query", outputs["shuffled_query"]),
        ]
    )
    for source_id, source_name in enumerate(SOURCE_NAMES):
        variants[f"source_{source_name}"] = outputs["source_evidence"][:, source_id, :]
    for model_name, scores in variants.items():
        start = len(metric_rows)
        fixed.add_subset_metrics(
            metric_rows,
            df,
            test_idx,
            scores,
            split_id=split_id,
            source_name=model_name,
        )
        for row in metric_rows[start:]:
            row.update(info)
            row["evaluation_phase"] = "development" if split_id == 0 else "confirmatory"


def make_prediction_frame(
    df: pd.DataFrame,
    test_idx: np.ndarray,
    outputs: dict[str, Any],
    split_id: int,
) -> pd.DataFrame:
    sub = df.iloc[test_idx].copy()
    dynamic = outputs["dynamic"]
    for action_id, action in enumerate(ACTIONS):
        sub[f"attention_probe_score_{action}"] = dynamic[:, action_id]
        sub[f"global_control_score_{action}"] = outputs["global"][:, action_id]
        sub[f"uniform_control_score_{action}"] = outputs["uniform"][:, action_id]
        sub[f"fixed_ovr3_control_score_{action}"] = outputs["fixed_ovr3"][:, action_id]
        sub[f"fixed_core4_control_score_{action}"] = outputs["fixed_core4"][:, action_id]
        sub[f"shuffled_query_control_score_{action}"] = outputs["shuffled_query"][:, action_id]
        for source_id, source_name in enumerate(SOURCE_NAMES):
            sub[f"source_{source_name}_score_{action}"] = outputs["source_evidence"][:, source_id, action_id]
            sub[f"attention_{source_name}_{action}"] = outputs["attention"][:, source_id, action_id]
    sub["predicted_top_action"] = np.asarray(ACTIONS, dtype=object)[dynamic.argmax(axis=1)]
    sub["split_id"] = int(split_id)
    sub["split_role"] = "test"
    sub["in_sample"] = False
    sub["model_name"] = "last_token_query_attention"
    metadata = [
        "row_id",
        "prefix_id",
        "decision_id",
        "trajectory_id",
        "step_id",
        "prefix_token_index",
        "progress",
        "agent_action",
        "top_action",
        "predicted_top_action",
        "split_id",
        "split_role",
        "in_sample",
        "model_name",
        *Q_COLS,
        *SUBSET_COLS,
    ]
    score_cols = [column for column in sub.columns if column.startswith(("attention_probe_score_", "global_control_score_", "uniform_control_score_", "fixed_ovr3_control_score_", "fixed_core4_control_score_", "shuffled_query_control_score_", "source_", "attention_"))]
    keep = [column for column in [*metadata, *score_cols] if column in sub.columns]
    return sub[keep]


def attention_statistics(outputs: dict[str, Any], split_id: int) -> list[dict[str, Any]]:
    weights = outputs["attention"]
    rows: list[dict[str, Any]] = []
    for source_id, source_name in enumerate(SOURCE_NAMES):
        for action_id, action in enumerate(ACTIONS):
            values = weights[:, source_id, action_id]
            rows.append(
                {
                    "split_id": int(split_id),
                    "source_name": source_name,
                    "action": action,
                    "mean_attention": float(values.mean()),
                    "std_attention": float(values.std()),
                    "p10_attention": float(np.quantile(values, 0.1)),
                    "median_attention": float(np.quantile(values, 0.5)),
                    "p90_attention": float(np.quantile(values, 0.9)),
                }
            )
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--beliefs", default="outputs/conceptual_belief_space/action_belief_vectors_mlx_full.parquet")
    parser.add_argument("--prefixes", default="outputs/prefixes.parquet")
    parser.add_argument("--output_dir", default="outputs/conceptual_belief_space/last_token_attention_probe")
    parser.add_argument("--layers", default="10,12,14,16,18,20,22")
    parser.add_argument("--n_splits", type=int, default=5)
    parser.add_argument("--split_ids", default="0")
    parser.add_argument("--limit_decisions", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--min_delta", type=float, default=1e-4)
    parser.add_argument("--val_fraction", type=float, default=0.15)
    parser.add_argument("--max_train_prefixes_per_decision", type=int, default=48)
    parser.add_argument("--max_train_rows", type=int, default=0)
    parser.add_argument("--attention_dim", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--source_pretrain_epochs", type=int, default=12)
    parser.add_argument("--source_pretrain_patience", type=int, default=4)
    parser.add_argument("--source_pretrain_lr", type=float, default=5e-4)
    parser.add_argument("--source_pretrain_pair_weight", type=float, default=1.0)
    parser.add_argument("--gate_loss", choices=["ovr_bce", "balanced_ce"], default="ovr_bce")
    parser.add_argument("--expert_loss_weight", type=float, default=0.0)
    parser.add_argument("--pair_loss_weight", type=float, default=0.0)
    parser.add_argument("--q_rank_loss_weight", type=float, default=0.1)
    parser.add_argument("--q_soft_loss_weight", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-3)
    parser.add_argument("--grad_clip", type=float, default=5.0)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--eval_batch_size", type=int, default=256)
    parser.add_argument("--save_predictions", action="store_true")
    parser.add_argument("--save_checkpoints", action="store_true")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser.parse_args()


def main() -> int:
    import torch

    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    layers = parse_layers(args.layers)
    df = fixed.rank_impl.prepare_belief_rows(
        Path(args.beliefs),
        limit_decisions=args.limit_decisions,
        seed=args.seed,
        prefix_path=Path(args.prefixes),
    ).reset_index(drop=True)
    y = action_indices(df["top_action"])
    q_all = df[Q_COLS].to_numpy(dtype=np.float32)
    q_top = q_all.argmax(axis=1)
    if not np.array_equal(y, q_top):
        mismatch = int((y != q_top).sum())
        raise RuntimeError(f"top_action is not q argmax for {mismatch} rows")
    feature_rows = seg.ms_impl.feature_rows_by_position(df)
    pre_rows = seg.ms_impl.pre_feature_rows_by_position(df)
    splits = fixed.rank_impl.make_group_splits(df, args.n_splits, args.seed)
    all_split_ids = [int(split["split_id"]) for split in splits]
    wanted = set(parse_split_ids(args.split_ids, all_split_ids))
    splits = [split for split in splits if int(split["split_id"]) in wanted]
    if not splits:
        raise ValueError(f"No requested splits found; available={all_split_ids}")

    arrays = seg.load_arrays(UNIQUE_TERMS, layers)
    hidden_dim = int(next(iter(arrays.values())).shape[1])
    manifest = {
        "probe_definition": "compact learned readout on frozen backbone activations",
        "activation_only": True,
        "frozen_backbone": True,
        "backbone_name": "openai/gpt-oss-20b",
        "target": "forced-answer q-top action",
        "target_column": "top_action",
        "query": "actual last generated prefix-token activation (raw1)",
        "memory": "causal last-100, cumulative-prefix, multi-scale segment, and pairwise last-20 activation summaries",
        "source_terms": dict(SOURCE_TERMS),
        "unique_activation_terms": UNIQUE_TERMS,
        "layers": layers,
        "hidden_dim": hidden_dim,
        "attention_dim": int(args.attention_dim),
        "trajectory_grouped_outer_splits": int(args.n_splits),
        "validation_fraction_within_outer_train": float(args.val_fraction),
        "development_split": 0,
        "confirmatory_splits": [split_id for split_id in all_split_ids if split_id != 0],
        "rows": int(len(df)),
        "decisions": int(df["decision_id"].nunique()),
        "trajectories": int(df["trajectory_id"].nunique()),
        "excluded_model_inputs": FORBIDDEN_MODEL_INPUTS,
        "metadata_policy": "metadata is used only for cache indexing, grouped splits, training-row subsampling, labels, and reports",
        "causality": "every query and memory summary ends at the current prefix; no future prefix is read",
        "training_stages": ["independent source pretraining", "frozen-source last-token attention gate"],
        "controls": ["global attention", "uniform attention", "fixed OVR3 fusion", "fixed core4 fusion", "shuffled last-token query", *SOURCE_NAMES],
        "arguments": vars(args),
    }
    (output_dir / "last_token_attention_probe_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    metric_rows: list[dict[str, Any]] = []
    history_rows: list[dict[str, Any]] = []
    info_rows: list[dict[str, Any]] = []
    attention_rows: list[dict[str, Any]] = []
    prediction_frames: list[pd.DataFrame] = []

    for split in splits:
        split_id = int(split["split_id"])
        outer_train = np.asarray(split["train_idx"], dtype=np.int64)
        test_idx = np.asarray(split["test_idx"], dtype=np.int64)
        assert_group_disjoint(df, outer_train, test_idx, f"outer split {split_id}")
        if args.val_fraction > 0:
            train_idx, val_idx = seg.ms_impl.train_val_split_by_group(
                df,
                outer_train,
                args.seed + split_id,
                args.val_fraction,
            )
            assert_group_disjoint(df, train_idx, val_idx, f"validation split {split_id}")
            assert_group_disjoint(df, val_idx, test_idx, f"validation/test split {split_id}")
        else:
            train_idx = outer_train
            val_idx = np.asarray([], dtype=np.int64)
        print(
            f"Outer split {split_id}: train={len(train_idx)} val={len(val_idx)} test={len(test_idx)}",
            flush=True,
        )
        outputs, info, history, checkpoint = train_fold(
            arrays,
            layers,
            df,
            y,
            q_all,
            train_idx,
            val_idx,
            test_idx,
            feature_rows,
            pre_rows,
            args,
            split_id,
        )
        info_rows.append({"split_id": split_id, **info})
        history_rows.extend(history)
        add_all_metrics(metric_rows, df, test_idx, outputs, split_id, info)
        attention_rows.extend(attention_statistics(outputs, split_id))
        if args.save_predictions:
            prediction_frames.append(make_prediction_frame(df, test_idx, outputs, split_id))
        if args.save_checkpoints:
            torch.save(checkpoint, output_dir / f"last_token_attention_probe_split_{split_id}.pt")

        pd.DataFrame(metric_rows).to_csv(output_dir / "last_token_attention_probe_metrics.partial.csv", index=False)
        pd.DataFrame(history_rows).to_csv(output_dir / "last_token_attention_probe_history.partial.csv", index=False)
        pd.DataFrame(info_rows).to_csv(output_dir / "last_token_attention_probe_fold_info.partial.csv", index=False)
        pd.DataFrame(attention_rows).to_csv(output_dir / "last_token_attention_probe_attention.partial.csv", index=False)
        current = pd.DataFrame(metric_rows)
        print(
            current[
                current["split_id"].eq(split_id) & current["eval_subset"].eq("all_prefixes")
            ][["source_name", "balanced_accuracy", "accuracy", "macro_f1"]].to_string(index=False),
            flush=True,
        )

    metrics = pd.DataFrame(metric_rows)
    history_df = pd.DataFrame(history_rows)
    info_df = pd.DataFrame(info_rows)
    attention_df = pd.DataFrame(attention_rows)
    metrics.to_csv(output_dir / "last_token_attention_probe_metrics.csv", index=False)
    history_df.to_csv(output_dir / "last_token_attention_probe_history.csv", index=False)
    info_df.to_csv(output_dir / "last_token_attention_probe_fold_info.csv", index=False)
    attention_df.to_csv(output_dir / "last_token_attention_probe_attention.csv", index=False)
    if args.save_predictions and prediction_frames:
        pd.concat(prediction_frames, ignore_index=True).to_parquet(
            output_dir / "last_token_attention_probe_predictions.parquet", index=False
        )

    all_prefixes = metrics[metrics["eval_subset"].eq("all_prefixes")]
    summary = (
        all_prefixes.groupby("source_name", as_index=False)
        .agg(
            splits=("split_id", "nunique"),
            mean_accuracy=("accuracy", "mean"),
            mean_balanced_accuracy=("balanced_accuracy", "mean"),
            min_balanced_accuracy=("balanced_accuracy", "min"),
            mean_macro_f1=("macro_f1", "mean"),
            mean_pairwise_accuracy=("pairwise_accuracy", "mean"),
        )
        .sort_values("mean_balanced_accuracy", ascending=False)
    )
    confirmatory = all_prefixes[all_prefixes["split_id"].ne(0)]
    confirmatory_summary = (
        confirmatory.groupby("source_name", as_index=False)
        .agg(
            splits=("split_id", "nunique"),
            mean_accuracy=("accuracy", "mean"),
            mean_balanced_accuracy=("balanced_accuracy", "mean"),
            min_balanced_accuracy=("balanced_accuracy", "min"),
            mean_macro_f1=("macro_f1", "mean"),
        )
        .sort_values("mean_balanced_accuracy", ascending=False)
        if not confirmatory.empty
        else pd.DataFrame()
    )
    summary.to_csv(output_dir / "last_token_attention_probe_summary.csv", index=False)
    confirmatory_summary.to_csv(output_dir / "last_token_attention_probe_confirmatory_summary.csv", index=False)

    report = [
        "# Last-Token-Query Attention Probe",
        "",
        "## Probe contract",
        "",
        "- Frozen `openai/gpt-oss-20b` backbone; only the downstream readout is trained.",
        "- The actual last generated prefix token supplies the query.",
        "- Attention memory contains only causal activation summaries ending at that prefix.",
        "- No grid/task state, progress, IDs, future tokens, or forced-answer text are model inputs.",
        "- Train, validation, and test trajectories are disjoint.",
        "- Split 0 is treated as development; splits 1-4 are confirmatory when present.",
        "",
        "## All evaluated folds",
        "",
        summary.to_markdown(index=False) if not summary.empty else "No completed folds.",
        "",
        "## Confirmatory folds (excluding split 0)",
        "",
        confirmatory_summary.to_markdown(index=False) if not confirmatory_summary.empty else "No confirmatory folds completed.",
        "",
        "## Fold model sizes",
        "",
        info_df.to_markdown(index=False) if not info_df.empty else "No fold information.",
    ]
    (output_dir / "last_token_attention_probe_report.md").write_text("\n".join(report) + "\n")
    print(summary.to_string(index=False) if not summary.empty else "No summary rows", flush=True)
    print(f"Wrote {output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
