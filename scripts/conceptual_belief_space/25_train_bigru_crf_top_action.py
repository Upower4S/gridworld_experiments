#!/usr/bin/env python
"""Train a BiGRU/Transformer emission model with a linear-chain CRF head.

This is a sequence-level top_action diagnostic. It reuses the feature loading
and sequence batching utilities from 24_train_multitask_sequence_transformer.py
but replaces independent token classification with a CRF sequence likelihood.
"""

from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import sys
import time
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

BASE_SCRIPT = Path(__file__).with_name("24_train_multitask_sequence_transformer.py")
spec = importlib.util.spec_from_file_location("seq24", BASE_SCRIPT)
seq24 = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules["seq24"] = seq24
spec.loader.exec_module(seq24)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--beliefs", default="outputs/conceptual_belief_space/action_belief_vectors.parquet")
    parser.add_argument("--splits", default="outputs/conceptual_belief_space/splits/group_splits.json")
    parser.add_argument("--feature_dir", default="outputs/activation/features")
    parser.add_argument("--output_dir", default="outputs/conceptual_belief_space/top_action_bigru_crf")
    parser.add_argument("--figure_dir", default="figures/conceptual_belief_space/mlp_branch_probe/top_action_bigru_crf")
    parser.add_argument("--representation", default="raw")
    parser.add_argument("--subset", default="all_prefixes")
    parser.add_argument("--group_col", default="trajectory_id")
    parser.add_argument("--feature_set", default="mean:all")
    parser.add_argument("--architecture", default="bigru_h512_l2")
    parser.add_argument("--attention_mode", choices=["causal", "bidirectional"], default="bidirectional")
    parser.add_argument("--loss_weighting", choices=["plain", "balanced"], default="balanced")
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
    parser.add_argument("--crf_loss_weight", type=float, default=1.0)
    parser.add_argument("--agent_loss_weight", type=float, default=0.3)
    parser.add_argument("--q_loss_weight", type=float, default=0.0)
    parser.add_argument("--add_metadata_features", action="store_true")
    parser.add_argument("--limit_splits", type=int, default=None)
    parser.add_argument("--save_best_predictions", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


class LinearChainCRF:
    def __init__(self, num_tags: int, device):
        import torch

        self.num_tags = num_tags
        self.start = torch.nn.Parameter(torch.zeros(num_tags, device=device))
        self.end = torch.nn.Parameter(torch.zeros(num_tags, device=device))
        self.trans = torch.nn.Parameter(torch.zeros(num_tags, num_tags, device=device))

    def parameters(self):
        return [self.start, self.end, self.trans]

    def state_dict(self) -> dict[str, Any]:
        return {"start": self.start.detach().cpu(), "end": self.end.detach().cpu(), "trans": self.trans.detach().cpu()}

    def load_state_dict(self, state: dict[str, Any], device) -> None:
        self.start.data.copy_(state["start"].to(device))
        self.end.data.copy_(state["end"].to(device))
        self.trans.data.copy_(state["trans"].to(device))

    def neg_log_likelihood(self, emissions, tags, mask):
        import torch

        tags = tags.clamp_min(0)
        mask = mask.bool()
        log_den = self._log_partition(emissions, mask)
        log_num = self._sequence_score(emissions, tags, mask)
        return (log_den - log_num).mean()

    def _sequence_score(self, emissions, tags, mask):
        import torch

        batch, seq_len, _ = emissions.shape
        score = self.start[tags[:, 0]] + emissions[torch.arange(batch, device=emissions.device), 0, tags[:, 0]]
        for t in range(1, seq_len):
            emit = emissions[torch.arange(batch, device=emissions.device), t, tags[:, t]]
            trans = self.trans[tags[:, t - 1], tags[:, t]]
            score = score + (emit + trans) * mask[:, t]
        lengths = mask.long().sum(dim=1).clamp_min(1) - 1
        last_tags = tags[torch.arange(batch, device=emissions.device), lengths]
        return score + self.end[last_tags]

    def _log_partition(self, emissions, mask):
        import torch

        alpha = self.start + emissions[:, 0]
        for t in range(1, emissions.shape[1]):
            scores = alpha.unsqueeze(2) + self.trans.unsqueeze(0) + emissions[:, t].unsqueeze(1)
            next_alpha = torch.logsumexp(scores, dim=1)
            alpha = torch.where(mask[:, t].unsqueeze(1), next_alpha, alpha)
        return torch.logsumexp(alpha + self.end, dim=1)

    def decode(self, emissions, mask) -> np.ndarray:
        import torch

        mask = mask.bool()
        batch, seq_len, num_tags = emissions.shape
        score = self.start + emissions[:, 0]
        history = []
        for t in range(1, seq_len):
            next_score = score.unsqueeze(2) + self.trans.unsqueeze(0) + emissions[:, t].unsqueeze(1)
            best_score, best_tag = next_score.max(dim=1)
            score = torch.where(mask[:, t].unsqueeze(1), best_score, score)
            history.append(best_tag)
        score = score + self.end
        best_last = score.argmax(dim=1)
        paths = torch.zeros((batch, seq_len), dtype=torch.long, device=emissions.device)
        lengths = mask.long().sum(dim=1).clamp_min(1)
        for b in range(batch):
            tag = best_last[b]
            last_t = int(lengths[b].item()) - 1
            paths[b, last_t] = tag
            for hist_t in range(last_t - 1, -1, -1):
                tag = history[hist_t][b, tag]
                paths[b, hist_t] = tag
        return paths.detach().cpu().numpy()


def metrics_from_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    true_labels = action_indices_to_labels(y_true)
    pred_labels = action_indices_to_labels(y_pred)
    row = {
        "accuracy": float(np.mean(y_true == y_pred)),
        "balanced_accuracy": balanced_accuracy_np(true_labels, pred_labels),
        "macro_f1": macro_f1_np(true_labels, pred_labels),
    }
    missing = []
    for i, action in enumerate(ACTIONS):
        true_count = int(np.sum(y_true == i))
        pred_count = int(np.sum(y_pred == i))
        row[f"true_count_{action}"] = true_count
        row[f"pred_count_{action}"] = pred_count
        if true_count > 0 and pred_count == 0:
            missing.append(action)
    row["missing_actions_json"] = json.dumps(missing)
    row["collapse"] = bool(missing)
    row["num_predicted_actions"] = int(sum(row[f"pred_count_{a}"] > 0 for a in ACTIONS))
    matrix = {}
    for i, actual in enumerate(ACTIONS):
        matrix[actual] = {}
        for j, predicted in enumerate(ACTIONS):
            matrix[actual][predicted] = int(np.sum((y_true == i) & (y_pred == j)))
    row["confusion_matrix_json"] = json.dumps(matrix, sort_keys=True)
    return row


def predict_with_probs(model, crf: LinearChainCRF, dataset, batch_size: int, device, *, causal: bool):
    import torch
    from torch.utils.data import DataLoader

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=seq24.collate_sequences)
    pred_by_row: dict[int, int] = {}
    true_by_row: dict[int, int] = {}
    prob_by_row: dict[int, np.ndarray] = {}
    model.eval()
    with torch.no_grad():
        for xb, meta, yb, _agent, _q, pad_mask, rows in loader:
            outputs = model(xb.to(device), meta.to(device), pad_mask.to(device), causal=causal)
            emissions = outputs["top"]
            valid_mask = ~pad_mask.to(device)
            decoded = crf.decode(emissions, valid_mask)
            probs = torch.softmax(emissions, dim=-1).detach().cpu().numpy()
            y_np = yb.numpy()
            rows_np = rows.numpy()
            valid = rows_np >= 0
            for row, yy, pp, prob in zip(
                rows_np[valid].reshape(-1),
                y_np[valid].reshape(-1),
                decoded[valid].reshape(-1),
                probs[valid].reshape(-1, len(ACTIONS)),
            ):
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


def train_one(train_ds, val_ds, input_dim: int, feature_set, spec, max_seq_len: int, args, device, *, seed: int, causal: bool):
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader

    torch.manual_seed(seed)
    np.random.seed(seed)
    meta_dim = int(train_ds.meta_scaler["mean"].shape[0]) if train_ds.add_metadata_features else 0
    model = seq24.build_model(input_dim, meta_dim, feature_set, spec, max_seq_len, args.dropout).to(device)
    crf = LinearChainCRF(len(ACTIONS), device)
    y_agent_train = np.concatenate([train_ds.y_agent_by_row[seq] for seq in train_ds.sequences])
    agent_weights = torch.tensor(class_weights(y_agent_train), dtype=torch.float32, device=device) if args.loss_weighting == "balanced" else None
    optimizer = torch.optim.AdamW(list(model.parameters()) + crf.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    generator = torch.Generator()
    generator.manual_seed(seed)
    loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, generator=generator, collate_fn=seq24.collate_sequences)
    best_state = None
    best_crf_state = None
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
            pad_mask_d = pad_mask.to(device)
            outputs = model(xb.to(device), meta.to(device), pad_mask_d, causal=causal)
            emissions = outputs["top"]
            valid_mask = ~pad_mask_d
            loss = float(args.crf_loss_weight) * crf.neg_log_likelihood(emissions, y_top_d, valid_mask)
            if args.agent_loss_weight > 0:
                agent_ce = F.cross_entropy(
                    outputs["agent"].reshape(-1, len(ACTIONS)),
                    y_agent_d.reshape(-1),
                    ignore_index=-100,
                    weight=agent_weights,
                )
                loss = loss + float(args.agent_loss_weight) * agent_ce
            if args.q_loss_weight > 0:
                flat_valid = y_top_d.reshape(-1) != -100
                log_q_pred = F.log_softmax(emissions.reshape(-1, len(ACTIONS))[flat_valid], dim=-1)
                q_flat = q_target_d.reshape(-1, len(ACTIONS))[flat_valid]
                q_kl = F.kl_div(log_q_pred, q_flat, reduction="batchmean")
                loss = loss + float(args.q_loss_weight) * q_kl
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if args.grad_clip and args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(list(model.parameters()) + crf.parameters(), args.grad_clip)
            optimizer.step()
        _, y_val, p_val, _ = predict_with_probs(model, crf, val_ds, args.batch_size, device, causal=causal)
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
            }
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_crf_state = crf.state_dict()
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    if best_crf_state is not None:
        crf.load_state_dict(best_crf_state, device)
    return model, crf, {"best_epoch": best_epoch, "validation_score": best_score, **best_info}


def make_prediction_frame(row_ids, y_true, y_pred, y_prob, *, split_id: int, split_role: str, feature_set, spec, args) -> pd.DataFrame:
    pred_df = pd.DataFrame(
        {
            "split_id": int(split_id),
            "split_role": split_role,
            "row_id": row_ids.astype(np.int64),
            "true_top_action": action_indices_to_labels(y_true),
            "pred_top_action": action_indices_to_labels(y_pred),
            "feature_set": feature_set.name,
            "architecture": spec.name,
            "attention_mode": args.attention_mode,
            "loss_weighting": args.loss_weighting,
            "crf_loss_weight": float(args.crf_loss_weight),
            "agent_loss_weight": float(args.agent_loss_weight),
            "q_loss_weight": float(args.q_loss_weight),
            "add_metadata_features": bool(args.add_metadata_features),
        }
    )
    for i, action in enumerate(ACTIONS):
        pred_df[f"prob_{action}"] = y_prob[:, i]
    return pred_df


def main() -> None:
    args = parse_args()
    output_dir = ensure_dir(args.output_dir)
    figure_dir = ensure_dir(args.figure_dir)
    metrics_path = output_dir / "bigru_crf_metrics.csv"
    if metrics_path.exists() and not args.force:
        raise SystemExit(f"{metrics_path} exists; pass --force to overwrite")
    import torch

    device = seq24.choose_device(args.device)
    df = make_behavior_table(args.beliefs).reset_index(drop=True)
    sub_idx = np.flatnonzero(subset_mask(df, args.subset))
    feature_dir = Path(args.feature_dir)
    all_layers = seq24.available_layers(feature_dir, args.representation)
    feature_set = seq24.parse_feature_set(args.feature_set, all_layers)
    spec_obj = seq24.parse_architecture(args.architecture)
    needed_layers = sorted(feature_set.layers)
    layer_arrays = {layer: np.load(feature_dir / f"layer_{layer:02d}_{args.representation}.npy", mmap_mode="r") for layer in needed_layers}
    hidden_dim = next(iter(layer_arrays.values())).shape[1]
    splits = load_or_create_splits(args.splits, df, args.group_col, seed=args.random_seed)
    if args.limit_splits is not None:
        splits = splits[: int(args.limit_splits)]
    print(json.dumps({"device": str(device), "feature_set": feature_set.name, "architecture": spec_obj.name, "subset_rows": int(len(sub_idx))}, indent=2))

    metrics: list[dict[str, Any]] = []
    pred_frames: list[pd.DataFrame] = []
    val_pred_frames: list[pd.DataFrame] = []
    for split in splits:
        train_pool = np.intersect1d(split.train_indices, sub_idx, assume_unique=False)
        test_rows = np.intersect1d(split.test_indices, sub_idx, assume_unique=False)
        train_rows, val_rows = make_train_val_split(df, train_pool, split.group_col, seed=args.random_seed + split.split_id)
        train_seqs = seq24.make_sequences(df, train_rows)
        val_seqs = seq24.make_sequences(df, val_rows)
        test_seqs = seq24.make_sequences(df, test_rows)
        max_seq_len = max(max(len(seq) for seq in train_seqs), max(len(seq) for seq in val_seqs), max(len(seq) for seq in test_seqs))
        print(f"split={split.split_id} train_rows={len(train_rows)} val_rows={len(val_rows)} test_rows={len(test_rows)} max_seq_len={max_seq_len}")
        selected_arrays = {layer: layer_arrays[layer] for layer in feature_set.layers}
        scalers = seq24.compute_scalers(selected_arrays, train_rows, args.standardize)
        meta_scaler = seq24.compute_meta_scaler(df, train_rows, args.add_metadata_features)
        train_ds = seq24.DynamicFeatureSequenceDataset(
            df=df,
            sequences=train_seqs,
            feature_set=feature_set,
            layer_arrays=selected_arrays,
            scalers=scalers,
            meta_scaler=meta_scaler,
            add_metadata_features=args.add_metadata_features,
            l2_normalize=args.l2_normalize,
        )
        val_ds = seq24.DynamicFeatureSequenceDataset(
            df=df,
            sequences=val_seqs,
            feature_set=feature_set,
            layer_arrays=selected_arrays,
            scalers=scalers,
            meta_scaler=meta_scaler,
            add_metadata_features=args.add_metadata_features,
            l2_normalize=args.l2_normalize,
        )
        test_ds = seq24.DynamicFeatureSequenceDataset(
            df=df,
            sequences=test_seqs,
            feature_set=feature_set,
            layer_arrays=selected_arrays,
            scalers=scalers,
            meta_scaler=meta_scaler,
            add_metadata_features=args.add_metadata_features,
            l2_normalize=args.l2_normalize,
        )
        input_dim = hidden_dim * len(feature_set.layers) if feature_set.mode == "concat" else hidden_dim
        seed = seq24.stable_seed(feature_set.name, spec_obj.name, args.attention_mode, args.loss_weighting, split.split_id, base=args.random_seed)
        causal = args.attention_mode == "causal"
        model, crf, info = train_one(train_ds, val_ds, input_dim, feature_set, spec_obj, max_seq_len, args, device, seed=seed, causal=causal)
        row_ids, y_true, y_pred, y_prob = predict_with_probs(model, crf, test_ds, args.batch_size, device, causal=causal)
        top_m = metrics_from_predictions(y_true, y_pred)
        metrics.append(
            {
                "split_id": int(split.split_id),
                "representation": args.representation,
                "subset": args.subset,
                "feature_set": feature_set.name,
                "feature_mode": feature_set.mode,
                "feature_layers_json": json.dumps(list(feature_set.layers)),
                "architecture": spec_obj.name,
                "attention_mode": args.attention_mode,
                "loss_weighting": args.loss_weighting,
                "crf_loss_weight": float(args.crf_loss_weight),
                "agent_loss_weight": float(args.agent_loss_weight),
                "q_loss_weight": float(args.q_loss_weight),
                "add_metadata_features": bool(args.add_metadata_features),
                "n_train": int(len(train_rows)),
                "n_val": int(len(val_rows)),
                "n_test": int(len(test_rows)),
                "best_epoch": int(info.get("best_epoch", -1)),
                "validation_score": float(info.get("validation_score", float("nan"))),
                "validation_accuracy": float(info.get("validation_accuracy", float("nan"))),
                "validation_balanced_accuracy": float(info.get("validation_balanced_accuracy", float("nan"))),
                "validation_macro_f1": float(info.get("validation_macro_f1", float("nan"))),
                "test_top_action_accuracy": top_m["accuracy"],
                "test_top_action_balanced_accuracy": top_m["balanced_accuracy"],
                "test_top_action_macro_f1": top_m["macro_f1"],
                "test_action_collapse": top_m["collapse"],
                "test_num_predicted_actions": top_m["num_predicted_actions"],
                "test_missing_actions_json": top_m["missing_actions_json"],
                "top_action_confusion_matrix_json": top_m["confusion_matrix_json"],
            }
        )
        if args.save_best_predictions:
            pred_frames.append(make_prediction_frame(row_ids, y_true, y_pred, y_prob, split_id=split.split_id, split_role="test", feature_set=feature_set, spec=spec_obj, args=args))
            val_row_ids, y_val, p_val, val_prob = predict_with_probs(model, crf, val_ds, args.batch_size, device, causal=causal)
            val_pred_frames.append(make_prediction_frame(val_row_ids, y_val, p_val, val_prob, split_id=split.split_id, split_role="validation", feature_set=feature_set, spec=spec_obj, args=args))
        del model, crf, train_ds, val_ds, test_ds, scalers, meta_scaler
        gc.collect()
        if str(device) == "mps" and hasattr(torch, "mps"):
            torch.mps.empty_cache()
    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv(metrics_path, index=False)
    summary = (
        metrics_df.groupby(["feature_set", "architecture", "attention_mode", "crf_loss_weight", "agent_loss_weight", "q_loss_weight"], as_index=False)
        .agg(
            splits=("split_id", "nunique"),
            mean_acc=("test_top_action_accuracy", "mean"),
            std_acc=("test_top_action_accuracy", "std"),
            mean_bal=("test_top_action_balanced_accuracy", "mean"),
            std_bal=("test_top_action_balanced_accuracy", "std"),
            mean_f1=("test_top_action_macro_f1", "mean"),
            collapse_rows=("test_action_collapse", "sum"),
        )
        .sort_values("mean_bal", ascending=False)
    )
    summary.to_csv(output_dir / "bigru_crf_summary.csv", index=False)
    if args.save_best_predictions and pred_frames:
        pd.concat(pred_frames, ignore_index=True).to_parquet(output_dir / "test_predictions.parquet", index=False)
    if args.save_best_predictions and val_pred_frames:
        pd.concat(val_pred_frames, ignore_index=True).to_parquet(output_dir / "validation_predictions.parquet", index=False)
    lines = [
        "# BiGRU CRF Top-Action Sequence Model",
        "",
        f"- representation: `{args.representation}`",
        f"- feature set: `{feature_set.name}`",
        f"- architecture: `{spec_obj.name}`",
        f"- attention mode: `{args.attention_mode}`",
        f"- CRF loss weight: `{args.crf_loss_weight}`",
        f"- agent loss weight: `{args.agent_loss_weight}`",
        f"- q loss weight: `{args.q_loss_weight}`",
        "",
        "## Summary",
        "",
        summary.to_markdown(index=False),
    ]
    save_markdown(output_dir / "bigru_crf_report.md", "\n".join(lines))
    if not summary.empty:
        write_json(output_dir / "best_bigru_crf_config.json", json_ready(summary.iloc[0].to_dict()))
    print(json.dumps({"metrics": str(metrics_path), "summary": str(output_dir / "bigru_crf_summary.csv"), "rows": int(len(metrics_df))}, indent=2))


if __name__ == "__main__":
    main()
