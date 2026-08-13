#!/usr/bin/env python3
"""Linear probes on cached activation segment-combination features.

This script tests richer activation-only summaries than simple mean windows:
prefix cumulative mean, recent windows, residuals from pre_reasoning, and
recent-minus-prefix contrasts.  The probe remains a single affine readout; all
feature terms are linear combinations of activation vectors.  Metadata is used
only for row alignment, residual indexing, grouped splits, and reporting.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.conceptual_belief_space.config import ACTIONS, DEFAULT_SEED

RANK_SCRIPT = Path(__file__).with_name("39_train_activation_rank_probes.py")
spec = importlib.util.spec_from_file_location("activation_rank_probe_impl", RANK_SCRIPT)
rank_impl = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = rank_impl
assert spec and spec.loader
spec.loader.exec_module(rank_impl)

MULTI_SCRIPT = Path(__file__).with_name("43_train_cached_multiscale_linear.py")
mspec = importlib.util.spec_from_file_location("cached_multiscale_impl", MULTI_SCRIPT)
ms_impl = importlib.util.module_from_spec(mspec)
sys.modules[mspec.name] = ms_impl
assert mspec and mspec.loader
mspec.loader.exec_module(ms_impl)

Q_COLS = [f"q_{a}" for a in ACTIONS]
PAIR_INDICES = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
SUBSET_COLS = ["strict_before_any_action_mention", "strict_before_final_action_mention"]
CACHE_DIRS = {
    1: Path("outputs/conceptual_belief_space/activation_multilayer_multipool_linear_screen/feature_cache"),
    20: Path("outputs/conceptual_belief_space/activation_multilayer_identity_linear_screen/feature_cache"),
    50: Path("outputs/conceptual_belief_space/activation_window_mean50_linear_screen/feature_cache"),
    100: Path("outputs/conceptual_belief_space/activation_window_mean100_linear_screen/feature_cache"),
    -1: Path("outputs/conceptual_belief_space/activation_prefix_cumulative_mean_linear_screen/feature_cache"),
}

MODE_TERMS = {
    "raw_pfx_20_100": ["rawpfx", "raw20", "raw100"],
    "rawresid_pfx_20_100": ["rawpfx", "raw20", "raw100", "residpfx", "resid20", "resid100"],
    "rawresid_pfx_20_50_100": ["rawpfx", "raw20", "raw50", "raw100", "residpfx", "resid20", "resid50", "resid100"],
    "contrast_pfx_20_100": ["rawpfx", "diff100_pfx", "diff20_100", "residpfx"],
    "contrast_pfx_20_50_100": ["rawpfx", "diff100_pfx", "diff50_100", "diff20_50", "residpfx"],
    "all_segment_contrasts": ["rawpfx", "raw100", "diff20_100", "diff50_100", "residpfx", "resid100", "resid20"],
}


def parse_layers(text: str) -> list[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def parse_modes(text: str) -> list[str]:
    return [x.strip() for x in text.split(",") if x.strip()]


def parse_losses(text: str) -> list[str]:
    vals = [x.strip() for x in text.split(",") if x.strip()]
    allowed = {"ovr_bce", "ce", "balanced_ce", "focal_ce", "balanced_focal", "ovr_rank", "ce_rank", "balanced_ce_rank"}
    bad = [x for x in vals if x not in allowed]
    if bad:
        raise ValueError(f"Unsupported losses: {bad}; allowed={sorted(allowed)}")
    return vals


def window_from_token(token: str) -> int:
    if token.endswith("pfx"):
        return -1
    for suffix in ["100", "50", "20", "1"]:
        if token.endswith(suffix):
            return int(suffix)
    raise ValueError(f"Cannot parse window from term {token}")


def cache_path(window: int, layer: int) -> Path:
    base = CACHE_DIRS[int(window)]
    if int(window) == -1:
        names = [f"features_layer_{layer:02d}_raw_prefixmean.npy", f"features_layer_{layer}_raw_prefixmean.npy"]
    elif int(window) == 1:
        names = [f"features_layer_{layer:02d}_raw_last1.npy", f"features_layer_{layer}_raw_last1.npy"]
    else:
        names = [f"features_layer_{layer:02d}_raw_mean{window}.npy", f"features_layer_{layer}_raw_mean{window}.npy"]
    for name in names:
        path = base / name
        if path.exists():
            return path
    raise FileNotFoundError(f"Missing cache for window={window} layer={layer} under {base}")


def windows_needed(terms: list[str]) -> set[int]:
    out: set[int] = set()
    for term in terms:
        if term.startswith("diff"):
            left, right = term.replace("diff", "").split("_")
            out.add(window_from_token("raw" + left))
            out.add(window_from_token("raw" + right))
        else:
            out.add(window_from_token(term))
    return out


def load_arrays(terms: list[str], layers: list[int]) -> dict[tuple[int, int], np.ndarray]:
    out = {}
    for window in windows_needed(terms):
        for layer in layers:
            out[(int(window), int(layer))] = np.load(cache_path(window, layer), mmap_mode="r")
    return out


def get_values(arrays, window: int, layer: int, row_idx: np.ndarray) -> np.ndarray:
    return np.asarray(arrays[(int(window), int(layer))][row_idx], dtype=np.float32)


def term_values(term: str, arrays, layer: int, row_idx: np.ndarray, pre_idx: np.ndarray) -> np.ndarray:
    if term.startswith("raw"):
        return get_values(arrays, window_from_token(term), layer, row_idx)
    if term.startswith("resid"):
        window = window_from_token(term)
        arr = arrays[(int(window), int(layer))]
        return np.asarray(arr[row_idx], dtype=np.float32) - np.asarray(arr[pre_idx], dtype=np.float32)
    if term.startswith("diff"):
        left, right = term.replace("diff", "").split("_")
        return get_values(arrays, window_from_token("raw" + left), layer, row_idx) - get_values(arrays, window_from_token("raw" + right), layer, row_idx)
    raise ValueError(f"Unknown term: {term}")


def feature_dim(arrays, terms: list[str], layers: list[int]) -> int:
    hidden = next(iter(arrays.values())).shape[1]
    return int(hidden * len(terms) * len(layers))


def build_batch(arrays, terms: list[str], layers: list[int], indices: np.ndarray, feature_rows: np.ndarray, pre_rows: np.ndarray) -> np.ndarray:
    row_idx = feature_rows[indices]
    pre_idx = pre_rows[indices]
    parts = []
    for term in terms:
        for layer in layers:
            parts.append(term_values(term, arrays, layer, row_idx, pre_idx))
    return np.concatenate(parts, axis=1).astype(np.float32, copy=False)


def compute_scaler(arrays, terms, layers, indices, feature_rows, pre_rows, batch_size: int):
    dim = feature_dim(arrays, terms, layers)
    total = np.zeros(dim, dtype=np.float64)
    total_sq = np.zeros(dim, dtype=np.float64)
    count = 0
    for start in range(0, len(indices), batch_size):
        idx = indices[start:start + batch_size]
        xb = build_batch(arrays, terms, layers, idx, feature_rows, pre_rows)
        total += xb.sum(axis=0, dtype=np.float64)
        total_sq += np.square(xb, dtype=np.float32).sum(axis=0, dtype=np.float64)
        count += len(idx)
    mean = (total / max(count, 1)).astype(np.float32)
    var = total_sq / max(count, 1) - mean.astype(np.float64) ** 2
    var[var < 1e-12] = 1.0
    scale = np.sqrt(var).astype(np.float32)
    scale[scale < 1e-6] = 1.0
    return mean, scale


def action_indices(values: pd.Series) -> np.ndarray:
    mapping = {a: i for i, a in enumerate(ACTIONS)}
    return values.astype(str).map(mapping).to_numpy(dtype=np.int64)


def predict(model, arrays, terms, layers, indices, feature_rows, pre_rows, mean, scale, batch_size: int):
    import torch
    device = next(model.parameters()).device
    outs = []
    with torch.no_grad():
        for start in range(0, len(indices), batch_size):
            idx = indices[start:start + batch_size]
            xb = (build_batch(arrays, terms, layers, idx, feature_rows, pre_rows) - mean) / scale
            logits = model(torch.tensor(xb, dtype=torch.float32, device=device)).cpu().numpy()
            outs.append(logits.astype(np.float32))
    return np.concatenate(outs, axis=0) if outs else np.zeros((0, len(ACTIONS)), dtype=np.float32)


def balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    vals = []
    for c in range(len(ACTIONS)):
        mask = y_true == c
        if mask.any():
            vals.append(float((y_pred[mask] == c).mean()))
    return float(np.mean(vals)) if vals else float("nan")


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    vals = []
    for c in range(len(ACTIONS)):
        tp = float(((y_true == c) & (y_pred == c)).sum())
        fp = float(((y_true != c) & (y_pred == c)).sum())
        fn = float(((y_true == c) & (y_pred != c)).sum())
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        vals.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return float(np.mean(vals))


def pairwise_accuracy(scores: np.ndarray, q: np.ndarray, tie_eps: float = 1e-6) -> float:
    correct = total = 0
    for i, j in PAIR_INDICES:
        valid = np.abs(q[:, i] - q[:, j]) > tie_eps
        if not valid.any():
            continue
        correct += int(((scores[:, i] > scores[:, j]) == (q[:, i] > q[:, j]))[valid].sum())
        total += int(valid.sum())
    return float(correct / total) if total else float("nan")


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> list[list[int]]:
    out = np.zeros((len(ACTIONS), len(ACTIONS)), dtype=int)
    for t, p in zip(y_true, y_pred):
        out[int(t), int(p)] += 1
    return out.tolist()


def balanced_epoch_indices(train_idx: np.ndarray, y: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Sample a same-sized epoch with equal action classes, using replacement as needed."""
    train_idx = np.asarray(train_idx, dtype=np.int64)
    per_class = max(1, int(np.ceil(len(train_idx) / len(ACTIONS))))
    chunks = []
    for c in range(len(ACTIONS)):
        cls = train_idx[y[train_idx] == c]
        if len(cls) == 0:
            continue
        chunks.append(rng.choice(cls, size=per_class, replace=len(cls) < per_class))
    if not chunks:
        return rng.permutation(train_idx)
    order = np.concatenate(chunks)[: len(train_idx)]
    rng.shuffle(order)
    return order.astype(np.int64, copy=False)


def focal_cross_entropy(logits, target, *, weight=None, gamma: float = 2.0):
    import torch
    log_probs = torch.nn.functional.log_softmax(logits, dim=1)
    probs = torch.exp(log_probs)
    idx = torch.arange(len(target), device=target.device)
    pt = probs[idx, target].clamp_min(1e-8)
    log_pt = log_probs[idx, target]
    loss = -((1.0 - pt) ** gamma) * log_pt
    if weight is not None:
        loss = loss * weight[target]
    return loss.mean()


def pairwise_q_rank_loss(logits, q_batch: np.ndarray, *, tie_eps: float = 1e-6):
    import torch
    pair_i = torch.tensor([i for i, _ in PAIR_INDICES], dtype=torch.long, device=logits.device)
    pair_j = torch.tensor([j for _, j in PAIR_INDICES], dtype=torch.long, device=logits.device)
    q_tensor = torch.tensor(q_batch, dtype=torch.float32, device=logits.device)
    diff = q_tensor[:, pair_i] - q_tensor[:, pair_j]
    target = (diff > 0).float()
    mask = (diff.abs() > tie_eps).float()
    pair_logits = logits[:, pair_i] - logits[:, pair_j]
    raw = torch.nn.functional.binary_cross_entropy_with_logits(pair_logits, target, reduction="none")
    return (raw * mask).sum() / mask.sum().clamp_min(1.0)


def train_model(arrays, terms, layers, y, q_all, train_idx, val_idx, test_idx, feature_rows, pre_rows, *, loss_kind, epochs, patience, lr, weight_decay, batch_size, seed, focal_gamma: float = 2.0, rank_loss_weight: float = 0.5):
    import torch
    rng = np.random.default_rng(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu"))
    mean, scale = compute_scaler(arrays, terms, layers, train_idx, feature_rows, pre_rows, batch_size=max(batch_size, 512))
    model = torch.nn.Linear(len(mean), len(ACTIONS)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    counts = np.bincount(y[train_idx], minlength=len(ACTIONS)).astype(np.float32)
    ce_weights = counts.sum() / np.clip(counts, 1.0, None)
    ce_weights = ce_weights / ce_weights.mean()
    ce_weight_tensor = torch.tensor(ce_weights, dtype=torch.float32, device=device)
    pos_weight = torch.tensor((len(train_idx) - counts) / np.clip(counts, 1.0, None), dtype=torch.float32, device=device)
    score_idx = val_idx if len(val_idx) else train_idx
    best_state = None
    best_val = -1.0
    best_epoch = -1
    stale = 0
    for epoch in range(int(epochs)):
        if loss_kind.startswith("balanced_"):
            order = balanced_epoch_indices(train_idx, y, rng)
        else:
            order = rng.permutation(train_idx)
        model.train()
        for start in range(0, len(order), batch_size):
            idx = order[start:start + batch_size]
            xb = (build_batch(arrays, terms, layers, idx, feature_rows, pre_rows) - mean) / scale
            logits = model(torch.tensor(xb, dtype=torch.float32, device=device))
            y_tensor = torch.tensor(y[idx], dtype=torch.long, device=device)
            rank_loss = None
            if loss_kind.endswith("_rank"):
                rank_loss = pairwise_q_rank_loss(logits, q_all[idx])
            if loss_kind == "ce" or loss_kind == "ce_rank":
                loss = torch.nn.functional.cross_entropy(logits, y_tensor, weight=ce_weight_tensor)
            elif loss_kind == "balanced_ce" or loss_kind == "balanced_ce_rank":
                loss = torch.nn.functional.cross_entropy(logits, y_tensor)
            elif loss_kind == "focal_ce":
                loss = focal_cross_entropy(logits, y_tensor, weight=ce_weight_tensor, gamma=focal_gamma)
            elif loss_kind == "balanced_focal":
                loss = focal_cross_entropy(logits, y_tensor, weight=None, gamma=focal_gamma)
            else:
                target = np.zeros((len(idx), len(ACTIONS)), dtype=np.float32)
                target[np.arange(len(idx)), y[idx]] = 1.0
                loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, torch.tensor(target, dtype=torch.float32, device=device), pos_weight=pos_weight)
            if rank_loss is not None:
                loss = loss + float(rank_loss_weight) * rank_loss
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        model.eval()
        val_logits = predict(model, arrays, terms, layers, score_idx, feature_rows, pre_rows, mean, scale, batch_size)
        val_bal = balanced_accuracy(y[score_idx], val_logits.argmax(axis=1))
        if val_bal > best_val + 1e-5:
            best_val = val_bal
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    logits = predict(model, arrays, terms, layers, test_idx, feature_rows, pre_rows, mean, scale, batch_size)
    return logits, {"best_epoch": int(best_epoch), "validation_balanced_accuracy": float(best_val), "device": str(device), "feature_dim": int(len(mean))}


def add_metrics(rows, df, indices, logits, *, mode, loss_kind, terms, layers, split_id, info):
    sub = df.iloc[indices]
    y_true = action_indices(sub["top_action"])
    y_pred = logits.argmax(axis=1)
    q = sub[Q_COLS].to_numpy(dtype=np.float64)
    subsets = {"all_prefixes": np.ones(len(indices), dtype=bool)}
    for col in SUBSET_COLS:
        if col in sub.columns:
            subsets[col] = sub[col].fillna(False).astype(bool).to_numpy()
    for subset, mask in subsets.items():
        if not mask.any():
            continue
        row = {
            "feature_family": "cached_segment_combo_activation_linear",
            "mode": mode,
            "loss_kind": loss_kind,
            "terms_json": json.dumps(terms),
            "layers_json": json.dumps(layers),
            "split_id": int(split_id),
            "eval_subset": subset,
            "n_rows": int(mask.sum()),
            "accuracy": float((y_pred[mask] == y_true[mask]).mean()),
            "balanced_accuracy": balanced_accuracy(y_true[mask], y_pred[mask]),
            "macro_f1": macro_f1(y_true[mask], y_pred[mask]),
            "pairwise_accuracy": pairwise_accuracy(logits[mask], q[mask]),
            "confusion_matrix_json": json.dumps(confusion_matrix(y_true[mask], y_pred[mask])),
            **info,
        }
        for i, action in enumerate(ACTIONS):
            am = y_true[mask] == i
            row[f"recall_{action}"] = float((y_pred[mask][am] == i).mean()) if am.any() else float("nan")
        rows.append(row)


def make_predictions(df, indices, logits, *, split_id, mode, loss_kind, terms, layers):
    sub = df.iloc[indices].copy()
    for i, action in enumerate(ACTIONS):
        sub[f"score_{action}"] = logits[:, i]
    sub["predicted_top_action"] = np.asarray(ACTIONS, dtype=object)[logits.argmax(axis=1)]
    sub["split_id"] = int(split_id)
    sub["mode"] = mode
    sub["loss_kind"] = loss_kind
    sub["terms_json"] = json.dumps(terms)
    sub["layers_json"] = json.dumps(layers)
    keep = [c for c in ["row_id", "prefix_id", "decision_id", "trajectory_id", "step_id", "prefix_token_index", "progress", "agent_action", "top_action", "predicted_top_action", "split_id", "mode", "loss_kind", "terms_json", "layers_json", *Q_COLS, *SUBSET_COLS, *[f"score_{a}" for a in ACTIONS]] if c in sub.columns]
    return sub[keep]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--beliefs", default="outputs/conceptual_belief_space/action_belief_vectors_mlx_full.parquet")
    p.add_argument("--prefixes", default="outputs/prefixes.parquet")
    p.add_argument("--output_dir", default="outputs/conceptual_belief_space/activation_segment_combo_linear")
    p.add_argument("--layers", default="10,12,14,16,18,20,22")
    p.add_argument("--modes", default="rawresid_pfx_20_100,contrast_pfx_20_100")
    p.add_argument("--losses", default="ovr_bce")
    p.add_argument("--focal_gamma", type=float, default=2.0)
    p.add_argument("--rank_loss_weight", type=float, default=0.5)
    p.add_argument("--n_splits", type=int, default=5)
    p.add_argument("--split_ids", default="0")
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--weight_decay", type=float, default=1e-3)
    p.add_argument("--batch_size", type=int, default=384)
    p.add_argument("--val_fraction", type=float, default=0.0)
    p.add_argument("--save_predictions", action="store_true")
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    df = rank_impl.prepare_belief_rows(Path(args.beliefs), limit_decisions=None, seed=args.seed, prefix_path=Path(args.prefixes))
    layers = parse_layers(args.layers)
    y = action_indices(df["top_action"])
    q_all = df[Q_COLS].to_numpy(dtype=np.float32)
    feature_rows = ms_impl.feature_rows_by_position(df)
    pre_rows = ms_impl.pre_feature_rows_by_position(df)
    splits = rank_impl.make_group_splits(df, args.n_splits, args.seed)
    if args.split_ids != "all":
        want = {int(x) for x in args.split_ids.split(",") if x.strip()}
        splits = [s for s in splits if int(s["split_id"]) in want]
    manifest = {
        "activation_only": True,
        "excluded_model_inputs": ["grid/task features", "progress", "prefix_token_index", "num_cot_tokens", "trajectory_id", "decision_id", "agent_action"],
        "feature_terms": MODE_TERMS,
        "supported_losses": ["ovr_bce", "ce", "balanced_ce", "focal_ce", "balanced_focal", "ovr_rank", "ce_rank", "balanced_ce_rank"],
        "layers": layers,
        "rows": int(len(df)),
        "decisions": int(df["decision_id"].nunique()),
        "trajectories": int(df["trajectory_id"].nunique()),
    }
    (output_dir / "segment_combo_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    rows = []
    preds = []
    for mode in parse_modes(args.modes):
        if mode not in MODE_TERMS:
            raise ValueError(f"Unsupported mode {mode}; known modes: {sorted(MODE_TERMS)}")
        terms = MODE_TERMS[mode]
        arrays = load_arrays(terms, layers)
        print(f"Mode {mode} terms={terms} feature_dim={feature_dim(arrays, terms, layers)}", flush=True)
        for loss_kind in parse_losses(args.losses):
            for split in splits:
                split_id = int(split["split_id"])
                train_idx = split["train_idx"]
                test_idx = split["test_idx"]
                if set(df.iloc[train_idx]["trajectory_id"].astype(str)) & set(df.iloc[test_idx]["trajectory_id"].astype(str)):
                    raise RuntimeError(f"Trajectory leakage in split {split_id}")
                subtrain_idx, val_idx = ms_impl.train_val_split_by_group(df, train_idx, args.seed + split_id, args.val_fraction)
                print(f"Training mode={mode} loss={loss_kind} split={split_id} train={len(subtrain_idx)} val={len(val_idx)} test={len(test_idx)}", flush=True)
                logits, info = train_model(arrays, terms, layers, y, q_all, subtrain_idx, val_idx, test_idx, feature_rows, pre_rows, loss_kind=loss_kind, epochs=args.epochs, patience=args.patience, lr=args.lr, weight_decay=args.weight_decay, batch_size=args.batch_size, seed=args.seed + split_id, focal_gamma=args.focal_gamma, rank_loss_weight=args.rank_loss_weight)
                add_metrics(rows, df, test_idx, logits, mode=mode, loss_kind=loss_kind, terms=terms, layers=layers, split_id=split_id, info=info)
                if args.save_predictions:
                    preds.append(make_predictions(df, test_idx, logits, split_id=split_id, mode=mode, loss_kind=loss_kind, terms=terms, layers=layers))
                pd.DataFrame(rows).to_csv(output_dir / "segment_combo_metrics.partial.csv", index=False)
                print(pd.DataFrame(rows).tail(3).to_string(index=False), flush=True)
    metrics = pd.DataFrame(rows)
    metrics.to_csv(output_dir / "segment_combo_metrics.csv", index=False)
    if args.save_predictions and preds:
        pd.concat(preds, ignore_index=True).to_parquet(output_dir / "segment_combo_predictions.parquet", index=False)
    all_rows = metrics[metrics["eval_subset"].eq("all_prefixes")]
    summary = all_rows.groupby(["mode", "loss_kind"], as_index=False).agg(
        splits=("split_id", "nunique"),
        mean_accuracy=("accuracy", "mean"),
        mean_balanced_accuracy=("balanced_accuracy", "mean"),
        mean_macro_f1=("macro_f1", "mean"),
        mean_pairwise_accuracy=("pairwise_accuracy", "mean"),
    ).sort_values("mean_balanced_accuracy", ascending=False)
    summary.to_csv(output_dir / "segment_combo_summary.csv", index=False)
    (output_dir / "segment_combo_report.md").write_text("# Cached Segment-Combination Linear Probe\n\n" + (summary.to_markdown(index=False) if not summary.empty else "No rows") + "\n")
    print(f"Wrote {output_dir / 'segment_combo_metrics.csv'}")
    print(f"Wrote {output_dir / 'segment_combo_summary.csv'}")
    print(summary.to_string(index=False) if not summary.empty else "No summary")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
