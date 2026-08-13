#!/usr/bin/env python3
"""Strict nested training for the last-token-query attention probe.

For each held-out trajectory fold, linear activation experts generate source
evidence for the outer training rows through inner GroupKFold OOF models.  A
small attention gate then uses the actual last-token activation as its query
over those source scores.  Final experts are retrained on all outer-training
trajectories before the frozen gate is applied to the held-out trajectories.

This protocol prevents the attention gate from learning on in-sample expert
scores and prevents every held-out trajectory from influencing source models,
gate selection, or gate fitting.
"""

from __future__ import annotations

import argparse
import copy
import gc
import importlib.util
import json
import math
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

BASE_SCRIPT = Path(__file__).with_name("67_train_last_token_attention_probe.py")
spec = importlib.util.spec_from_file_location("nested_last_token_attention_base", BASE_SCRIPT)
base = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = base
assert spec and spec.loader
spec.loader.exec_module(base)

ACTIONS = base.ACTIONS
Q_COLS = base.Q_COLS
SOURCE_NAMES = base.SOURCE_NAMES
UNIQUE_TERMS = base.UNIQUE_TERMS
seg = base.seg
fixed = base.fixed


def make_gate_class():
    import torch

    class ScoreAttentionGate(torch.nn.Module):
        """Action-specific attention from a last-token query to source evidence."""

        def __init__(self, query_dim: int, attention_dim: int, dropout: float):
            super().__init__()
            self.attention_dim = int(attention_dim)
            self.dropout = torch.nn.Dropout(float(dropout))
            self.query_projection = torch.nn.Linear(
                query_dim, len(ACTIONS) * attention_dim
            )
            self.key_projection = torch.nn.Linear(
                len(ACTIONS), len(ACTIONS) * attention_dim, bias=False
            )
            self.source_key_embedding = torch.nn.Parameter(
                torch.zeros(len(SOURCE_NAMES), len(ACTIONS), attention_dim)
            )
            initial_bias = torch.tensor([-2.0, 0.0, 0.0, 0.0, -1.0], dtype=torch.float32)
            self.global_attention_bias = torch.nn.Parameter(
                initial_bias[:, None].repeat(1, len(ACTIONS))
            )
            self.output_scale = torch.nn.Parameter(torch.ones(len(ACTIONS)))
            self.action_bias = torch.nn.Parameter(torch.zeros(len(ACTIONS)))
            # Standard attention-block query residual.  It gives the actual
            # final token a direct corrective route while attention remains the
            # primary memory aggregation mechanism.
            self.query_action_head = torch.nn.Linear(query_dim, len(ACTIONS))
            self.query_residual_scale = torch.nn.Parameter(torch.tensor(0.05))
            torch.nn.init.normal_(self.query_projection.weight, mean=0.0, std=0.005)
            torch.nn.init.zeros_(self.query_projection.bias)
            torch.nn.init.normal_(self.key_projection.weight, mean=0.0, std=0.005)
            torch.nn.init.normal_(self.source_key_embedding, mean=0.0, std=0.005)
            torch.nn.init.normal_(self.query_action_head.weight, mean=0.0, std=0.001)
            torch.nn.init.zeros_(self.query_action_head.bias)

        def forward(self, source_evidence, query_features, mode: str = "dynamic"):
            batch = source_evidence.shape[0]
            if mode in {"fixed_ovr3", "fixed_core4"}:
                selected = (
                    ["mean100_ovr", "prefixmean_ovr", "segment_ovr"]
                    if mode == "fixed_ovr3"
                    else ["mean100_ovr", "prefixmean_ovr", "segment_ovr", "identity_pairwise"]
                )
                weights = torch.zeros_like(source_evidence)
                for source_name in selected:
                    weights[:, SOURCE_NAMES.index(source_name), :] = 1.0 / len(selected)
            elif mode == "uniform":
                weights = torch.full_like(source_evidence, 1.0 / len(SOURCE_NAMES))
            elif mode == "global":
                weights = torch.softmax(self.global_attention_bias, dim=0).unsqueeze(0).expand(
                    batch, -1, -1
                )
            elif mode == "dynamic":
                query = self.query_projection(self.dropout(query_features)).reshape(
                    batch, len(ACTIONS), self.attention_dim
                )
                keys = self.key_projection(
                    source_evidence.reshape(-1, len(ACTIONS))
                ).reshape(
                    batch,
                    len(SOURCE_NAMES),
                    len(ACTIONS),
                    self.attention_dim,
                )
                keys = keys + self.source_key_embedding.unsqueeze(0)
                attention_logits = (
                    query.unsqueeze(1) * keys
                ).sum(dim=-1) / math.sqrt(float(self.attention_dim))
                attention_logits = attention_logits + self.global_attention_bias.unsqueeze(0)
                weights = torch.softmax(attention_logits, dim=1)
            else:
                raise ValueError(f"Unknown gate mode: {mode}")
            fused = (weights * source_evidence).sum(dim=1)
            logits = fused if mode.startswith("fixed_") else fused * self.output_scale + self.action_bias
            if mode == "dynamic":
                logits = logits + self.query_residual_scale * self.query_action_head(
                    self.dropout(query_features)
                )
            return logits, weights

    return ScoreAttentionGate


def source_parameter_count(model) -> int:
    return int(sum(parameter.numel() for parameter in model.source_heads.parameters()))


def train_source_bundle(
    arrays,
    layers,
    df,
    y,
    q_all,
    train_idx,
    score_idx,
    feature_rows,
    pre_rows,
    args,
    *,
    seed: int,
    label: str,
    keep_checkpoint: bool,
):
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else (
            "mps"
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
            else "cpu"
        )
    )
    effective_train = base.capped_prefix_rows(
        df,
        np.asarray(train_idx, dtype=np.int64),
        args.max_source_train_prefixes_per_decision,
        args.max_source_train_rows,
        seed,
    )
    scalers = base.compute_term_scalers(
        arrays,
        layers,
        effective_train,
        feature_rows,
        pre_rows,
        args.batch_size,
    )
    term_dim = int(len(next(iter(scalers.values()))[0]))
    Probe = base.make_probe_class()
    model = Probe(term_dim, args.attention_dim, args.dropout).to(device)
    counts = np.bincount(y[effective_train], minlength=len(ACTIONS)).astype(np.float32)
    pos_weight = torch.tensor(
        (len(effective_train) - counts) / np.clip(counts, 1.0, None),
        dtype=torch.float32,
        device=device,
    )
    optimizer = torch.optim.AdamW(
        model.source_heads.parameters(),
        lr=args.source_lr,
        weight_decay=args.weight_decay,
    )
    rng = np.random.default_rng(seed)
    epoch_rows: list[dict[str, Any]] = []
    for epoch in range(args.source_epochs):
        order = rng.permutation(effective_train)
        model.train()
        total_loss = 0.0
        total_expert = 0.0
        total_pair = 0.0
        seen = 0
        for start in range(0, len(order), args.batch_size):
            idx = order[start : start + args.batch_size]
            term_features = base.build_term_tensors(
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
            loss, expert_loss, pair_loss = base.compute_source_pretrain_loss(
                raw_scores,
                pair_logits,
                y_tensor,
                q_tensor,
                pos_weight,
                args.source_pair_loss_weight,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.source_heads.parameters(), args.grad_clip)
            optimizer.step()
            total_loss += float(loss.detach().cpu()) * len(idx)
            total_expert += expert_loss * len(idx)
            total_pair += pair_loss * len(idx)
            seen += len(idx)
        epoch_row = {
            "bundle_label": label,
            "epoch": int(epoch),
            "train_rows": int(len(effective_train)),
            "score_rows": int(len(score_idx)),
            "loss": total_loss / max(seen, 1),
            "expert_loss": total_expert / max(seen, 1),
            "pair_loss": total_pair / max(seen, 1),
        }
        epoch_rows.append(epoch_row)
        print(
            f"      {label} source_epoch={epoch + 1}/{args.source_epochs} "
            f"loss={epoch_row['loss']:.4f}",
            flush=True,
        )

    source_evidence = base.predict_probe(
        model,
        arrays,
        layers,
        np.asarray(score_idx, dtype=np.int64),
        feature_rows,
        pre_rows,
        scalers,
        args.eval_batch_size,
        device,
        modes=("fixed_ovr3",),
    )["source_evidence"]
    checkpoint = None
    if keep_checkpoint:
        checkpoint = {
            "source_heads_state_dict": copy.deepcopy(
                {
                    key: value.detach().cpu()
                    for key, value in model.source_heads.state_dict().items()
                }
            ),
            "scalers": scalers,
            "term_dim": term_dim,
            "layers": list(layers),
            "source_terms": dict(base.SOURCE_TERMS),
            "source_epochs": int(args.source_epochs),
            "source_parameters": source_parameter_count(model),
        }
    info = {
        "bundle_label": label,
        "train_rows": int(len(effective_train)),
        "score_rows": int(len(score_idx)),
        "source_epochs": int(args.source_epochs),
        "final_source_loss": float(epoch_rows[-1]["loss"]),
        "source_parameters": source_parameter_count(model),
        "device": str(device),
    }
    del model
    gc.collect()
    return source_evidence, info, epoch_rows, checkpoint


def inner_group_splits(df: pd.DataFrame, indices: np.ndarray, n_splits: int):
    from sklearn.model_selection import GroupKFold

    indices = np.asarray(indices, dtype=np.int64)
    groups = df.iloc[indices]["trajectory_id"].astype(str).to_numpy()
    unique_groups = np.unique(groups)
    if len(unique_groups) < n_splits:
        raise ValueError(
            f"Need at least {n_splits} trajectory groups for nested OOF; got {len(unique_groups)}"
        )
    splitter = GroupKFold(n_splits=n_splits)
    return [
        (indices[train_local], indices[score_local])
        for train_local, score_local in splitter.split(indices, groups=groups)
    ]


def build_inner_oof_evidence(
    arrays,
    layers,
    df,
    y,
    q_all,
    outer_train,
    feature_rows,
    pre_rows,
    args,
    split_id: int,
):
    evidence = np.full(
        (len(df), len(SOURCE_NAMES), len(ACTIONS)),
        np.nan,
        dtype=np.float32,
    )
    audit_rows: list[dict[str, Any]] = []
    info_rows: list[dict[str, Any]] = []
    history_rows: list[dict[str, Any]] = []
    coverage = np.zeros(len(df), dtype=np.int16)
    for inner_id, (inner_train, inner_score) in enumerate(
        inner_group_splits(df, outer_train, args.n_inner_splits)
    ):
        base.assert_group_disjoint(
            df, inner_train, inner_score, f"outer {split_id} inner {inner_id}"
        )
        print(
            f"    inner={inner_id} train={len(inner_train)} score={len(inner_score)}",
            flush=True,
        )
        scores, info, history, _ = train_source_bundle(
            arrays,
            layers,
            df,
            y,
            q_all,
            inner_train,
            inner_score,
            feature_rows,
            pre_rows,
            args,
            seed=args.seed + split_id * 1000 + inner_id,
            label=f"outer{split_id}_inner{inner_id}",
            keep_checkpoint=False,
        )
        evidence[inner_score] = scores
        coverage[inner_score] += 1
        info_rows.append({"outer_split_id": split_id, "role": "inner_oof", "inner_id": inner_id, **info})
        history_rows.extend(
            {"outer_split_id": split_id, "role": "inner_oof", "inner_id": inner_id, **row}
            for row in history
        )
        train_groups = set(df.iloc[inner_train]["trajectory_id"].astype(str))
        score_groups = set(df.iloc[inner_score]["trajectory_id"].astype(str))
        audit_rows.append(
            {
                "outer_split_id": split_id,
                "inner_id": inner_id,
                "train_rows": int(len(inner_train)),
                "score_rows": int(len(inner_score)),
                "train_trajectories": int(len(train_groups)),
                "score_trajectories": int(len(score_groups)),
                "trajectory_overlap": int(len(train_groups & score_groups)),
            }
        )
    if not np.all(coverage[outer_train] == 1):
        values, counts = np.unique(coverage[outer_train], return_counts=True)
        raise RuntimeError(f"Inner OOF coverage must equal one: {dict(zip(values.tolist(), counts.tolist()))}")
    if np.isnan(evidence[outer_train]).any():
        raise RuntimeError("Missing inner OOF source evidence")
    return evidence, audit_rows, info_rows, history_rows


def build_query_tensor(
    arrays,
    layers,
    indices,
    feature_rows,
    pre_rows,
    query_scaler,
    device,
):
    import torch

    mean, scale = query_scaler
    values = seg.build_batch(
        arrays, ["raw1"], layers, indices, feature_rows, pre_rows
    )
    values = (values - mean) / scale
    return torch.as_tensor(values, dtype=torch.float32, device=device)


def gate_loss(
    logits,
    y_tensor,
    q_tensor,
    pos_weight,
    class_weights,
    loss_kind: str,
    rank_weight: float,
    soft_weight: float,
):
    import torch

    target = torch.zeros((len(y_tensor), len(ACTIONS)), dtype=torch.float32, device=logits.device)
    target.scatter_(1, y_tensor[:, None], 1.0)
    if loss_kind == "ovr_bce":
        primary = torch.nn.functional.binary_cross_entropy_with_logits(
            logits, target, pos_weight=pos_weight
        )
    elif loss_kind == "balanced_ce":
        primary = torch.nn.functional.cross_entropy(logits, y_tensor, weight=class_weights)
    else:
        raise ValueError(f"Unknown gate loss: {loss_kind}")
    rank = base.q_rank_loss(logits, q_tensor)
    soft = base.soft_q_cross_entropy(logits, q_tensor)
    loss = primary + float(rank_weight) * rank + float(soft_weight) * soft
    return loss, float(primary.detach().cpu()), float(rank.detach().cpu()), float(soft.detach().cpu())


def predict_gate(
    gate,
    source_evidence: np.ndarray,
    row_indices: np.ndarray,
    arrays,
    layers,
    feature_rows,
    pre_rows,
    query_scaler,
    batch_size: int,
    device,
    *,
    query_mode: str,
    shuffled_query_indices: np.ndarray | None = None,
):
    import torch

    modes = ("dynamic", "global", "uniform", "fixed_ovr3", "fixed_core4")
    parts: dict[str, list[np.ndarray]] = {mode: [] for mode in modes}
    attention_parts: list[np.ndarray] = []
    shuffled_parts: list[np.ndarray] = []
    row_to_local = (
        {int(row): position for position, row in enumerate(row_indices)}
        if shuffled_query_indices is not None and query_mode == "last1_scores"
        else None
    )
    gate.eval()
    with torch.no_grad():
        for start in range(0, len(row_indices), batch_size):
            rows = row_indices[start : start + batch_size]
            local_scores = source_evidence[start : start + len(rows)]
            evidence_tensor = torch.as_tensor(local_scores, dtype=torch.float32, device=device)
            if query_mode == "last1_scores":
                query_tensor = evidence_tensor[:, SOURCE_NAMES.index("last1_ovr"), :]
            elif query_mode == "raw1":
                query_tensor = build_query_tensor(
                    arrays,
                    layers,
                    rows,
                    feature_rows,
                    pre_rows,
                    query_scaler,
                    device,
                )
            else:
                raise ValueError(f"Unknown query mode: {query_mode}")
            for mode in modes:
                logits, weights = gate(evidence_tensor, query_tensor, mode=mode)
                parts[mode].append(logits.cpu().numpy().astype(np.float32))
                if mode == "dynamic":
                    attention_parts.append(weights.cpu().numpy().astype(np.float32))
            if shuffled_query_indices is not None:
                shuffled_rows = shuffled_query_indices[start : start + len(rows)]
                if query_mode == "last1_scores":
                    assert row_to_local is not None
                    shuffled_local = np.asarray(
                        [row_to_local[int(row)] for row in shuffled_rows], dtype=np.int64
                    )
                    shuffled_query = torch.as_tensor(
                        source_evidence[shuffled_local, SOURCE_NAMES.index("last1_ovr"), :],
                        dtype=torch.float32,
                        device=device,
                    )
                else:
                    shuffled_query = build_query_tensor(
                        arrays,
                        layers,
                        shuffled_rows,
                        feature_rows,
                        pre_rows,
                        query_scaler,
                        device,
                    )
                logits, _ = gate(evidence_tensor, shuffled_query, mode="dynamic")
                shuffled_parts.append(logits.cpu().numpy().astype(np.float32))
    out = {mode: np.concatenate(value, axis=0) for mode, value in parts.items()}
    out["attention"] = np.concatenate(attention_parts, axis=0)
    out["source_evidence"] = np.asarray(source_evidence, dtype=np.float32)
    if shuffled_query_indices is not None:
        out["shuffled_query"] = np.concatenate(shuffled_parts, axis=0)
    return out


def train_attention_gate(
    oof_evidence,
    outer_train,
    arrays,
    layers,
    df,
    y,
    q_all,
    feature_rows,
    pre_rows,
    args,
    split_id: int,
):
    import torch

    meta_train, meta_val = seg.ms_impl.train_val_split_by_group(
        df,
        outer_train,
        args.seed + 50_000 + split_id,
        args.meta_val_fraction,
    )
    base.assert_group_disjoint(df, meta_train, meta_val, f"outer {split_id} gate selection")
    if args.query_mode == "last1_scores":
        query_scaler = None
        query_dim = len(ACTIONS)
    else:
        query_scaler = seg.compute_scaler(
            arrays,
            ["raw1"],
            layers,
            outer_train,
            feature_rows,
            pre_rows,
            batch_size=max(args.batch_size, 256),
        )
        query_dim = int(len(query_scaler[0]))
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else (
            "mps"
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
            else "cpu"
        )
    )
    Gate = make_gate_class()
    gate = Gate(query_dim, args.attention_dim, args.dropout).to(device)
    counts = np.bincount(y[meta_train], minlength=len(ACTIONS)).astype(np.float32)
    pos_weight = torch.tensor(
        (len(meta_train) - counts) / np.clip(counts, 1.0, None),
        dtype=torch.float32,
        device=device,
    )
    class_weight_np = counts.sum() / np.clip(counts, 1.0, None)
    class_weight_np /= class_weight_np.mean()
    class_weights = torch.tensor(class_weight_np, dtype=torch.float32, device=device)
    optimizer = torch.optim.AdamW(
        gate.parameters(), lr=args.gate_lr, weight_decay=args.weight_decay
    )
    rng = np.random.default_rng(args.seed + 60_000 + split_id)

    def validation_scores():
        local = oof_evidence[meta_val]
        return predict_gate(
            gate,
            local,
            meta_val,
            arrays,
            layers,
            feature_rows,
            pre_rows,
            query_scaler,
            args.eval_batch_size,
            device,
            query_mode=args.query_mode,
        )["dynamic"]

    initial = validation_scores()
    initial_pred = initial.argmax(axis=1)
    best_val = fixed.balanced_accuracy(y[meta_val], initial_pred)
    best_epoch = -1
    best_state = copy.deepcopy(
        {key: value.detach().cpu() for key, value in gate.state_dict().items()}
    )
    history = [
        {
            "outer_split_id": split_id,
            "stage": "gate_initial",
            "epoch": -1,
            "meta_train_rows": int(len(meta_train)),
            "meta_val_rows": int(len(meta_val)),
            "validation_balanced_accuracy": float(best_val),
            "validation_accuracy": float((initial_pred == y[meta_val]).mean()),
        }
    ]
    print(
        f"    gate initial val_bal={best_val:.4f} val_acc={history[-1]['validation_accuracy']:.4f}",
        flush=True,
    )
    stale = 0
    for epoch in range(args.gate_epochs):
        order = rng.permutation(meta_train)
        gate.train()
        total_loss = total_primary = total_rank = total_soft = 0.0
        seen = 0
        for start in range(0, len(order), args.gate_batch_size):
            idx = order[start : start + args.gate_batch_size]
            evidence_tensor = torch.as_tensor(
                oof_evidence[idx], dtype=torch.float32, device=device
            )
            if args.query_mode == "last1_scores":
                query_tensor = evidence_tensor[:, SOURCE_NAMES.index("last1_ovr"), :]
            else:
                query_tensor = build_query_tensor(
                    arrays,
                    layers,
                    idx,
                    feature_rows,
                    pre_rows,
                    query_scaler,
                    device,
                )
            logits, _weights = gate(evidence_tensor, query_tensor, mode="dynamic")
            y_tensor = torch.tensor(y[idx], dtype=torch.long, device=device)
            q_tensor = torch.tensor(q_all[idx], dtype=torch.float32, device=device)
            loss, primary, rank, soft = gate_loss(
                logits,
                y_tensor,
                q_tensor,
                pos_weight,
                class_weights,
                args.gate_loss,
                args.q_rank_loss_weight,
                args.q_soft_loss_weight,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(gate.parameters(), args.grad_clip)
            optimizer.step()
            total_loss += float(loss.detach().cpu()) * len(idx)
            total_primary += primary * len(idx)
            total_rank += rank * len(idx)
            total_soft += soft * len(idx)
            seen += len(idx)
        val_scores = validation_scores()
        val_pred = val_scores.argmax(axis=1)
        val_bal = fixed.balanced_accuracy(y[meta_val], val_pred)
        row = {
            "outer_split_id": split_id,
            "stage": "attention_gate",
            "epoch": epoch,
            "meta_train_rows": int(len(meta_train)),
            "meta_val_rows": int(len(meta_val)),
            "validation_balanced_accuracy": float(val_bal),
            "validation_accuracy": float((val_pred == y[meta_val]).mean()),
            "loss": total_loss / max(seen, 1),
            "primary_loss": total_primary / max(seen, 1),
            "rank_loss": total_rank / max(seen, 1),
            "soft_q_loss": total_soft / max(seen, 1),
        }
        history.append(row)
        print(
            f"    gate_epoch={epoch + 1}/{args.gate_epochs} loss={row['loss']:.4f} "
            f"val_bal={val_bal:.4f} val_acc={row['validation_accuracy']:.4f}",
            flush=True,
        )
        if val_bal > best_val + args.min_delta:
            best_val = float(val_bal)
            best_epoch = int(epoch)
            best_state = copy.deepcopy(
                {key: value.detach().cpu() for key, value in gate.state_dict().items()}
            )
            stale = 0
        else:
            stale += 1
            if stale >= args.gate_patience:
                break
    gate.load_state_dict(best_state)
    gate.to(device)
    info = {
        "best_gate_epoch": int(best_epoch),
        "gate_validation_balanced_accuracy": float(best_val),
        "gate_parameters": int(sum(parameter.numel() for parameter in gate.parameters())),
        "query_dim": query_dim,
        "query_mode": args.query_mode,
        "meta_train_rows": int(len(meta_train)),
        "meta_val_rows": int(len(meta_val)),
        "device": str(device),
    }
    return gate, query_scaler, device, history, info, best_state


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--beliefs", default="outputs/conceptual_belief_space/action_belief_vectors_mlx_full.parquet")
    parser.add_argument("--prefixes", default="outputs/prefixes.parquet")
    parser.add_argument("--output_dir", default="outputs/conceptual_belief_space/nested_last_token_attention_probe")
    parser.add_argument("--layers", default="10,12,14,16,18,20,22")
    parser.add_argument("--n_splits", type=int, default=5)
    parser.add_argument("--split_ids", default="0")
    parser.add_argument("--limit_decisions", type=int, default=None)
    parser.add_argument("--n_inner_splits", type=int, default=3)
    parser.add_argument("--meta_val_fraction", type=float, default=0.15)
    parser.add_argument("--source_epochs", type=int, default=8)
    parser.add_argument("--source_lr", type=float, default=5e-4)
    parser.add_argument("--source_pair_loss_weight", type=float, default=1.0)
    parser.add_argument("--max_source_train_prefixes_per_decision", type=int, default=0)
    parser.add_argument("--max_source_train_rows", type=int, default=0)
    parser.add_argument("--gate_epochs", type=int, default=20)
    parser.add_argument("--gate_patience", type=int, default=5)
    parser.add_argument("--gate_lr", type=float, default=1e-4)
    parser.add_argument("--gate_loss", choices=["balanced_ce", "ovr_bce"], default="balanced_ce")
    parser.add_argument("--query_mode", choices=["last1_scores", "raw1"], default="last1_scores")
    parser.add_argument("--attention_dim", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--q_rank_loss_weight", type=float, default=0.0)
    parser.add_argument("--q_soft_loss_weight", type=float, default=0.0)
    parser.add_argument("--weight_decay", type=float, default=1e-3)
    parser.add_argument("--grad_clip", type=float, default=5.0)
    parser.add_argument("--min_delta", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--gate_batch_size", type=int, default=256)
    parser.add_argument("--eval_batch_size", type=int, default=256)
    parser.add_argument("--save_predictions", action="store_true")
    parser.add_argument("--save_checkpoints", action="store_true")
    parser.add_argument("--seed", type=int, default=base.DEFAULT_SEED)
    return parser.parse_args()


def main() -> int:
    import torch

    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    layers = base.parse_layers(args.layers)
    df = fixed.rank_impl.prepare_belief_rows(
        Path(args.beliefs),
        limit_decisions=args.limit_decisions,
        seed=args.seed,
        prefix_path=Path(args.prefixes),
    ).reset_index(drop=True)
    y = base.action_indices(df["top_action"])
    q_all = df[Q_COLS].to_numpy(dtype=np.float32)
    if not np.array_equal(y, q_all.argmax(axis=1)):
        raise RuntimeError("top_action must equal forced-answer q argmax")
    feature_rows = seg.ms_impl.feature_rows_by_position(df)
    pre_rows = seg.ms_impl.pre_feature_rows_by_position(df)
    splits = fixed.rank_impl.make_group_splits(df, args.n_splits, args.seed)
    all_ids = [int(split["split_id"]) for split in splits]
    wanted = set(base.parse_split_ids(args.split_ids, all_ids))
    splits = [split for split in splits if int(split["split_id"]) in wanted]
    if not splits:
        raise ValueError(f"No selected folds; available={all_ids}")
    arrays = seg.load_arrays(UNIQUE_TERMS, layers)

    manifest = {
        "probe_definition": "nested OOF causal last-token-query attention readout",
        "activation_only": True,
        "frozen_backbone": True,
        "backbone_name": "openai/gpt-oss-20b",
        "target": "forced-answer q-top action",
        "query": (
            "four-action evidence decoded linearly from the actual last generated prefix token"
            if args.query_mode == "last1_scores"
            else "actual last generated prefix-token activation (raw1)"
        ),
        "memory_sources": dict(base.SOURCE_TERMS),
        "attention": "action-specific scaled dot-product attention over row-normalized source evidence with a standard last-token query residual",
        "outer_split_policy": "trajectory-grouped held-out evaluation",
        "inner_split_policy": "trajectory-grouped OOF source evidence for every gate-training row",
        "n_inner_splits": int(args.n_inner_splits),
        "layers": layers,
        "rows": int(len(df)),
        "decisions": int(df["decision_id"].nunique()),
        "trajectories": int(df["trajectory_id"].nunique()),
        "development_split": 0,
        "confirmatory_splits": [split_id for split_id in all_ids if split_id != 0],
        "excluded_model_inputs": base.FORBIDDEN_MODEL_INPUTS,
        "causality": "all activation views end at the current prefix; no future or forced answer tokens",
        "controls": [
            "global attention",
            "uniform attention",
            "fixed OVR3 fusion",
            "fixed core4 fusion",
            "shuffled last-token query",
            *SOURCE_NAMES,
        ],
        "arguments": vars(args),
    }
    (output_dir / "nested_last_token_attention_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )

    metric_rows: list[dict[str, Any]] = []
    prediction_frames: list[pd.DataFrame] = []
    attention_rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    source_info_rows: list[dict[str, Any]] = []
    source_history_rows: list[dict[str, Any]] = []
    gate_history_rows: list[dict[str, Any]] = []
    fold_info_rows: list[dict[str, Any]] = []

    for split in splits:
        split_id = int(split["split_id"])
        outer_train = np.asarray(split["train_idx"], dtype=np.int64)
        test_idx = np.asarray(split["test_idx"], dtype=np.int64)
        base.assert_group_disjoint(df, outer_train, test_idx, f"outer split {split_id}")
        print(
            f"Outer split {split_id}: train={len(outer_train)} test={len(test_idx)}",
            flush=True,
        )
        print("  Building strict inner-OOF source evidence", flush=True)
        oof_evidence, fold_audit, inner_info, inner_history = build_inner_oof_evidence(
            arrays,
            layers,
            df,
            y,
            q_all,
            outer_train,
            feature_rows,
            pre_rows,
            args,
            split_id,
        )
        audit_rows.extend(fold_audit)
        source_info_rows.extend(inner_info)
        source_history_rows.extend(inner_history)
        np.savez_compressed(
            output_dir / f"nested_last_token_attention_oof_evidence_split_{split_id}.npz",
            row_indices=outer_train,
            source_evidence=oof_evidence[outer_train],
        )
        print("  Training last-token attention gate on OOF evidence", flush=True)
        gate, query_scaler, gate_device, gate_history, gate_info, gate_state = train_attention_gate(
            oof_evidence,
            outer_train,
            arrays,
            layers,
            df,
            y,
            q_all,
            feature_rows,
            pre_rows,
            args,
            split_id,
        )
        gate_history_rows.extend(gate_history)
        print("  Training final experts on all outer-training trajectories", flush=True)
        test_evidence, final_info, final_history, source_checkpoint = train_source_bundle(
            arrays,
            layers,
            df,
            y,
            q_all,
            outer_train,
            test_idx,
            feature_rows,
            pre_rows,
            args,
            seed=args.seed + 100_000 + split_id,
            label=f"outer{split_id}_final",
            keep_checkpoint=args.save_checkpoints,
        )
        source_info_rows.append(
            {"outer_split_id": split_id, "role": "final_test", "inner_id": -1, **final_info}
        )
        source_history_rows.extend(
            {"outer_split_id": split_id, "role": "final_test", "inner_id": -1, **row}
            for row in final_history
        )
        np.savez_compressed(
            output_dir / f"nested_last_token_attention_test_evidence_split_{split_id}.npz",
            row_indices=test_idx,
            source_evidence=test_evidence,
        )
        shuffle_rng = np.random.default_rng(args.seed + 200_000 + split_id)
        shuffled_query = test_idx.copy()
        shuffle_rng.shuffle(shuffled_query)
        outputs = predict_gate(
            gate,
            test_evidence,
            test_idx,
            arrays,
            layers,
            feature_rows,
            pre_rows,
            query_scaler,
            args.eval_batch_size,
            gate_device,
            query_mode=args.query_mode,
            shuffled_query_indices=shuffled_query,
        )
        info = {
            **gate_info,
            "source_parameters": int(final_info["source_parameters"]),
            "total_probe_parameters": int(
                gate_info["gate_parameters"] + final_info["source_parameters"]
            ),
            "n_inner_splits": int(args.n_inner_splits),
            "source_epochs": int(args.source_epochs),
        }
        fold_info_rows.append({"split_id": split_id, **info})
        base.add_all_metrics(metric_rows, df, test_idx, outputs, split_id, info)
        attention_rows.extend(base.attention_statistics(outputs, split_id))
        if args.save_predictions:
            prediction_frames.append(base.make_prediction_frame(df, test_idx, outputs, split_id))
        if args.save_checkpoints:
            torch.save(
                {
                    "gate_state_dict": gate_state,
                    "query_scaler": query_scaler,
                    "source_checkpoint": source_checkpoint,
                    "actions": list(ACTIONS),
                    "source_names": SOURCE_NAMES,
                    "layers": layers,
                    "split_id": split_id,
                    "info": info,
                },
                output_dir / f"nested_last_token_attention_split_{split_id}.pt",
            )

        pd.DataFrame(metric_rows).to_csv(output_dir / "nested_last_token_attention_metrics.partial.csv", index=False)
        pd.DataFrame(audit_rows).to_csv(output_dir / "nested_last_token_attention_inner_audit.partial.csv", index=False)
        pd.DataFrame(source_info_rows).to_csv(output_dir / "nested_last_token_attention_source_info.partial.csv", index=False)
        pd.DataFrame(source_history_rows).to_csv(output_dir / "nested_last_token_attention_source_history.partial.csv", index=False)
        pd.DataFrame(gate_history_rows).to_csv(output_dir / "nested_last_token_attention_gate_history.partial.csv", index=False)
        pd.DataFrame(fold_info_rows).to_csv(output_dir / "nested_last_token_attention_fold_info.partial.csv", index=False)
        current = pd.DataFrame(metric_rows)
        print(
            current[
                current["split_id"].eq(split_id)
                & current["eval_subset"].eq("all_prefixes")
            ][["source_name", "balanced_accuracy", "accuracy", "macro_f1"]].to_string(index=False),
            flush=True,
        )

    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(output_dir / "nested_last_token_attention_metrics.csv", index=False)
    pd.DataFrame(audit_rows).to_csv(output_dir / "nested_last_token_attention_inner_audit.csv", index=False)
    pd.DataFrame(source_info_rows).to_csv(output_dir / "nested_last_token_attention_source_info.csv", index=False)
    pd.DataFrame(source_history_rows).to_csv(output_dir / "nested_last_token_attention_source_history.csv", index=False)
    pd.DataFrame(gate_history_rows).to_csv(output_dir / "nested_last_token_attention_gate_history.csv", index=False)
    fold_info_df = pd.DataFrame(fold_info_rows)
    fold_info_df.to_csv(output_dir / "nested_last_token_attention_fold_info.csv", index=False)
    attention_df = pd.DataFrame(attention_rows)
    attention_df.to_csv(output_dir / "nested_last_token_attention_attention.csv", index=False)
    if args.save_predictions and prediction_frames:
        pd.concat(prediction_frames, ignore_index=True).to_parquet(
            output_dir / "nested_last_token_attention_predictions.parquet", index=False
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
    summary.to_csv(output_dir / "nested_last_token_attention_summary.csv", index=False)
    confirmatory_summary.to_csv(output_dir / "nested_last_token_attention_confirmatory_summary.csv", index=False)
    report = [
        "# Strict Nested Last-Token Attention Probe",
        "",
        "## Probe contract",
        "",
        "- Frozen 20B backbone; the complete trained readout is about one million parameters.",
        "- The current prefix's actual last generated token is the attention query.",
        "- Memory experts read only causal activation summaries ending at that prefix.",
        "- Inner grouped OOF experts score every row used to train the gate.",
        "- Held-out outer trajectories influence neither expert training nor gate training/selection.",
        "- Grid/task features, metadata, future tokens, and forced-answer text are excluded from model inputs.",
        "",
        "## All evaluated folds",
        "",
        summary.to_markdown(index=False) if not summary.empty else "No completed folds.",
        "",
        "## Confirmatory folds (excluding development split 0)",
        "",
        confirmatory_summary.to_markdown(index=False) if not confirmatory_summary.empty else "No confirmatory folds completed.",
        "",
        "## Fold information",
        "",
        fold_info_df.to_markdown(index=False) if not fold_info_df.empty else "No fold information.",
    ]
    (output_dir / "nested_last_token_attention_report.md").write_text(
        "\n".join(report) + "\n"
    )
    print(summary.to_string(index=False), flush=True)
    print(f"Wrote {output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
