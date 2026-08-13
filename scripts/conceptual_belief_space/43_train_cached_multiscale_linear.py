#!/usr/bin/env python3
"""Full grouped linear probes on cached multi-window activation features.

This continues the activation-only goal after the split-0 multiscale screen.
The probe input is a concatenation of cached activation windows, e.g.
residualized mean20 and mean100 features across selected layers.  No grid/task
features or prefix metadata are included in the feature matrix.
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
if spec is None or spec.loader is None:
    raise RuntimeError(f"Could not import helpers from {RANK_SCRIPT}")
rank_impl = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = rank_impl
spec.loader.exec_module(rank_impl)

Q_COLS = [f"q_{a}" for a in ACTIONS]
PAIR_INDICES = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
SUBSET_COLS = ["strict_before_any_action_mention", "strict_before_final_action_mention"]

CACHE_DIRS = {
    20: Path("outputs/conceptual_belief_space/activation_multilayer_identity_linear_screen/feature_cache"),
    50: Path("outputs/conceptual_belief_space/activation_window_mean50_linear_screen/feature_cache"),
    100: Path("outputs/conceptual_belief_space/activation_window_mean100_linear_screen/feature_cache"),
}


def parse_layers(text: str) -> list[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def parse_modes(text: str) -> list[str]:
    return [x.strip() for x in text.split(",") if x.strip()]


def parse_losses(text: str) -> list[str]:
    vals = [x.strip() for x in text.split(",") if x.strip()]
    bad = [x for x in vals if x not in {"ovr_bce", "ce"}]
    if bad:
        raise ValueError(f"Unsupported losses: {bad}")
    return vals


def parse_mode(mode: str) -> list[tuple[str, int]]:
    # Examples: raw_20_100, resid_20_100, raw20_resid100, resid_20_50_100
    if mode.startswith("raw_"):
        return [("raw", int(x)) for x in mode.split("_")[1:]]
    if mode.startswith("resid_"):
        return [("resid", int(x)) for x in mode.split("_")[1:]]
    if mode == "raw20_resid100":
        return [("raw", 20), ("resid", 100)]
    raise ValueError(f"Unsupported mode: {mode}")


def feature_path(window: int, layer: int) -> Path:
    base = CACHE_DIRS[int(window)]
    names = [
        f"features_layer_{layer:02d}_raw_mean{window}.npy",
        f"features_layer_{layer}_raw_mean{window}.npy",
    ]
    for name in names:
        path = base / name
        if path.exists():
            return path
    raise FileNotFoundError(f"No cached feature for window={window} layer={layer} under {base}")


def load_arrays(terms: list[tuple[str, int]], layers: list[int]) -> dict[tuple[int, int], np.ndarray]:
    out = {}
    for _kind, window in terms:
        for layer in layers:
            key = (int(window), int(layer))
            if key not in out:
                out[key] = np.load(feature_path(window, layer), mmap_mode="r")
    return out


def action_indices(values: pd.Series) -> np.ndarray:
    mapping = {a: i for i, a in enumerate(ACTIONS)}
    return values.astype(str).map(mapping).to_numpy(dtype=np.int64)


def feature_rows_by_position(df: pd.DataFrame) -> np.ndarray:
    if "row_id" not in df.columns:
        return np.arange(len(df), dtype=np.int64)
    return df["row_id"].to_numpy(dtype=np.int64)


def pre_feature_rows_by_position(df: pd.DataFrame) -> np.ndarray:
    feature_rows = feature_rows_by_position(df)
    out = np.empty(len(df), dtype=np.int64)
    row_positions = np.arange(len(df), dtype=np.int64)
    tmp = df[["decision_id", "prefix_token_index"]].copy()
    tmp["_pos"] = row_positions
    for _decision_id, group in tmp.groupby("decision_id", sort=False):
        pre = group[group["prefix_token_index"].astype(int).eq(0)]
        pre_pos = int(pre.iloc[0]["_pos"]) if not pre.empty else int(group.iloc[0]["_pos"])
        out[group["_pos"].to_numpy(dtype=np.int64)] = feature_rows[pre_pos]
    return out


def feature_dim(arrays: dict[tuple[int, int], np.ndarray], terms: list[tuple[str, int]], layers: list[int]) -> int:
    hidden = next(iter(arrays.values())).shape[1]
    return int(hidden * len(terms) * len(layers))


def build_batch(
    arrays: dict[tuple[int, int], np.ndarray],
    terms: list[tuple[str, int]],
    layers: list[int],
    indices: np.ndarray,
    pre_rows: np.ndarray,
    feature_rows: np.ndarray,
) -> np.ndarray:
    parts = []
    row_idx = feature_rows[indices]
    pre_idx = pre_rows[indices]
    for kind, window in terms:
        for layer in layers:
            arr = arrays[(int(window), int(layer))]
            values = np.asarray(arr[row_idx], dtype=np.float32)
            if kind == "resid":
                values = values - np.asarray(arr[pre_idx], dtype=np.float32)
            parts.append(values)
    return np.concatenate(parts, axis=1).astype(np.float32, copy=False)


def compute_scaler(arrays, terms, layers, indices, pre_rows, feature_rows, batch_size: int) -> tuple[np.ndarray, np.ndarray]:
    dim = feature_dim(arrays, terms, layers)
    total = np.zeros(dim, dtype=np.float64)
    total_sq = np.zeros(dim, dtype=np.float64)
    count = 0
    for start in range(0, len(indices), batch_size):
        idx = indices[start:start + batch_size]
        xb = build_batch(arrays, terms, layers, idx, pre_rows, feature_rows)
        total += xb.sum(axis=0, dtype=np.float64)
        total_sq += np.square(xb, dtype=np.float32).sum(axis=0, dtype=np.float64)
        count += len(idx)
    mean = (total / max(count, 1)).astype(np.float32)
    var = total_sq / max(count, 1) - mean.astype(np.float64) ** 2
    var[var < 1e-12] = 1.0
    scale = np.sqrt(var).astype(np.float32)
    scale[scale < 1e-6] = 1.0
    return mean, scale


def train_val_split_by_group(df: pd.DataFrame, train_idx: np.ndarray, seed: int, val_fraction: float) -> tuple[np.ndarray, np.ndarray]:
    groups = sorted(df.iloc[train_idx]["trajectory_id"].astype(str).unique())
    rng = random.Random(seed)
    rng.shuffle(groups)
    n_val = max(1, int(round(len(groups) * val_fraction))) if len(groups) > 1 and val_fraction > 0 else 0
    val_groups = set(groups[:n_val])
    local = df.iloc[train_idx]["trajectory_id"].astype(str).isin(val_groups).to_numpy()
    val_idx = train_idx[local]
    subtrain_idx = train_idx[~local]
    if len(val_idx) == 0 or len(subtrain_idx) == 0:
        return train_idx, train_idx[:0]
    return subtrain_idx, val_idx


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
    correct = 0
    total = 0
    for i, j in PAIR_INDICES:
        valid = np.abs(q[:, i] - q[:, j]) > tie_eps
        if not valid.any():
            continue
        correct += int(((scores[:, i] > scores[:, j]) == (q[:, i] > q[:, j]))[valid].sum())
        total += int(valid.sum())
    return float(correct / total) if total else float("nan")


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> list[list[int]]:
    mat = np.zeros((len(ACTIONS), len(ACTIONS)), dtype=int)
    for t, p in zip(y_true, y_pred):
        mat[int(t), int(p)] += 1
    return mat.tolist()


def predict(model, arrays, terms, layers, indices, pre_rows, feature_rows, mean, scale, batch_size: int):
    import torch
    device = next(model.parameters()).device
    outs = []
    with torch.no_grad():
        for start in range(0, len(indices), batch_size):
            idx = indices[start:start + batch_size]
            xb = (build_batch(arrays, terms, layers, idx, pre_rows, feature_rows) - mean) / scale
            logits = model(torch.tensor(xb, dtype=torch.float32, device=device)).cpu().numpy()
            outs.append(logits.astype(np.float32))
    return np.concatenate(outs, axis=0) if outs else np.zeros((0, len(ACTIONS)), dtype=np.float32)


def fit_model(
    arrays,
    terms,
    layers,
    y,
    train_idx,
    val_idx,
    test_idx,
    pre_rows,
    feature_rows,
    *,
    loss_kind: str,
    epochs: int,
    patience: int,
    lr: float,
    weight_decay: float,
    batch_size: int,
    seed: int,
):
    import torch
    rng = np.random.default_rng(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu"))
    mean, scale = compute_scaler(arrays, terms, layers, train_idx, pre_rows, feature_rows, batch_size=max(batch_size, 512))
    model = torch.nn.Linear(len(mean), len(ACTIONS)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    counts = np.bincount(y[train_idx], minlength=len(ACTIONS)).astype(np.float32)
    ce_weights = counts.sum() / np.clip(counts, 1.0, None)
    ce_weights = ce_weights / ce_weights.mean()
    ce_weight_tensor = torch.tensor(ce_weights, dtype=torch.float32, device=device)
    pos = counts
    neg = len(train_idx) - counts
    pos_weight = torch.tensor(neg / np.clip(pos, 1.0, None), dtype=torch.float32, device=device)
    best_state = None
    best_val = -1.0
    best_epoch = -1
    stale = 0
    score_idx = val_idx if len(val_idx) else train_idx
    for epoch in range(int(epochs)):
        model.train()
        order = rng.permutation(train_idx)
        losses = []
        for start in range(0, len(order), batch_size):
            idx = order[start:start + batch_size]
            xb = (build_batch(arrays, terms, layers, idx, pre_rows, feature_rows) - mean) / scale
            logits = model(torch.tensor(xb, dtype=torch.float32, device=device))
            if loss_kind == "ce":
                yb = torch.tensor(y[idx], dtype=torch.long, device=device)
                loss = torch.nn.functional.cross_entropy(logits, yb, weight=ce_weight_tensor)
            else:
                target = np.zeros((len(idx), len(ACTIONS)), dtype=np.float32)
                target[np.arange(len(idx)), y[idx]] = 1.0
                loss = torch.nn.functional.binary_cross_entropy_with_logits(
                    logits,
                    torch.tensor(target, dtype=torch.float32, device=device),
                    pos_weight=pos_weight,
                )
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            losses.append(float(loss.detach().cpu()))
        model.eval()
        val_logits = predict(model, arrays, terms, layers, score_idx, pre_rows, feature_rows, mean, scale, batch_size)
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
    test_logits = predict(model, arrays, terms, layers, test_idx, pre_rows, feature_rows, mean, scale, batch_size)
    return test_logits, {"best_epoch": int(best_epoch), "validation_balanced_accuracy": float(best_val), "device": str(device), "feature_dim": int(len(mean))}


def add_metric_rows(rows, df, indices, logits, *, split_id, mode, loss_kind, terms, layers, info):
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
            "feature_family": "cached_multiscale_activation_linear",
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
            "feature_dim": int(info.get("feature_dim", logits.shape[1])),
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


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--beliefs", default="outputs/conceptual_belief_space/action_belief_vectors_mlx_full.parquet")
    p.add_argument("--prefixes", default="outputs/prefixes.parquet")
    p.add_argument("--output_dir", default="outputs/conceptual_belief_space/activation_cached_multiscale_linear_full")
    p.add_argument("--layers", default="10,12,14,16,18,20,22")
    p.add_argument("--modes", default="resid_20_100")
    p.add_argument("--losses", default="ovr_bce")
    p.add_argument("--n_splits", type=int, default=5)
    p.add_argument("--split_ids", default="all")
    p.add_argument("--limit_decisions", type=int, default=None)
    p.add_argument("--epochs", type=int, default=14)
    p.add_argument("--patience", type=int, default=4)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--weight_decay", type=float, default=1e-3)
    p.add_argument("--batch_size", type=int, default=768)
    p.add_argument("--val_fraction", type=float, default=0.2)
    p.add_argument("--save_predictions", action="store_true")
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    df = rank_impl.prepare_belief_rows(Path(args.beliefs), limit_decisions=args.limit_decisions, seed=args.seed, prefix_path=Path(args.prefixes))
    layers = parse_layers(args.layers)
    splits = rank_impl.make_group_splits(df, args.n_splits, args.seed)
    if args.split_ids != "all":
        want = {int(x) for x in args.split_ids.split(",") if x.strip()}
        splits = [s for s in splits if int(s["split_id"]) in want]
    y = action_indices(df["top_action"])
    feature_rows = feature_rows_by_position(df)
    pre_rows = pre_feature_rows_by_position(df)
    all_metrics = []
    all_preds = []
    manifest = {
        "activation_only": True,
        "excluded_model_inputs": ["grid/task features", "progress", "prefix_token_index", "num_cot_tokens", "trajectory_id", "decision_id", "agent_action"],
        "decision_id_use": "find prefix-0 pre_reasoning rows for residualization and grouped splitting only",
        "target": "top_action = argmax(q_UP,q_DOWN,q_LEFT,q_RIGHT)",
        "rows": int(len(df)),
        "decisions": int(df["decision_id"].nunique()),
        "trajectories": int(df["trajectory_id"].nunique()),
        "layers": layers,
        "modes": parse_modes(args.modes),
        "losses": parse_losses(args.losses),
    }
    (output_dir / "cached_multiscale_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    for mode in parse_modes(args.modes):
        terms = parse_mode(mode)
        arrays = load_arrays(terms, layers)
        dim = feature_dim(arrays, terms, layers)
        print(f"Mode {mode} terms={terms} feature_dim={dim}")
        for loss_kind in parse_losses(args.losses):
            for split in splits:
                split_id = int(split["split_id"])
                train_idx = split["train_idx"]
                test_idx = split["test_idx"]
                if set(df.iloc[train_idx]["trajectory_id"].astype(str)) & set(df.iloc[test_idx]["trajectory_id"].astype(str)):
                    raise RuntimeError(f"Trajectory leakage in split {split_id}")
                subtrain_idx, val_idx = train_val_split_by_group(df, train_idx, args.seed + split_id, args.val_fraction)
                print(f"Training mode={mode} loss={loss_kind} split={split_id} train={len(subtrain_idx)} val={len(val_idx)} test={len(test_idx)}", flush=True)
                logits, info = fit_model(
                    arrays,
                    terms,
                    layers,
                    y,
                    subtrain_idx,
                    val_idx,
                    test_idx,
                    pre_rows,
                    feature_rows,
                    loss_kind=loss_kind,
                    epochs=args.epochs,
                    patience=args.patience,
                    lr=args.lr,
                    weight_decay=args.weight_decay,
                    batch_size=args.batch_size,
                    seed=args.seed + split_id,
                )
                add_metric_rows(all_metrics, df, test_idx, logits, split_id=split_id, mode=mode, loss_kind=loss_kind, terms=terms, layers=layers, info=info)
                if args.save_predictions:
                    all_preds.append(make_predictions(df, test_idx, logits, split_id=split_id, mode=mode, loss_kind=loss_kind, terms=terms, layers=layers))
                pd.DataFrame(all_metrics).to_csv(output_dir / "cached_multiscale_metrics.partial.csv", index=False)
                print(pd.DataFrame(all_metrics).tail(3).to_string(index=False), flush=True)
    metrics = pd.DataFrame(all_metrics)
    metrics.to_csv(output_dir / "cached_multiscale_metrics.csv", index=False)
    if args.save_predictions:
        pd.concat(all_preds, ignore_index=True).to_parquet(output_dir / "cached_multiscale_predictions.parquet", index=False)
    all_rows = metrics[metrics["eval_subset"].eq("all_prefixes")]
    summary = all_rows.groupby(["mode", "loss_kind"], as_index=False).agg(
        splits=("split_id", "nunique"),
        mean_accuracy=("accuracy", "mean"),
        mean_balanced_accuracy=("balanced_accuracy", "mean"),
        mean_macro_f1=("macro_f1", "mean"),
        mean_pairwise_accuracy=("pairwise_accuracy", "mean"),
    ).sort_values("mean_balanced_accuracy", ascending=False)
    summary.to_csv(output_dir / "cached_multiscale_summary.csv", index=False)
    lines = [
        "# Cached Multiscale Activation-Only Linear Probe",
        "",
        "- Feature input: cached activation windows only.",
        "- No grid/task features or prefix metadata are model inputs.",
        "- Residual features subtract each decision's prefix-0 pre_reasoning feature.",
        "- Evaluation uses grouped trajectory splits.",
        "",
        "## Summary",
        "",
        summary.to_markdown(index=False),
    ]
    (output_dir / "cached_multiscale_report.md").write_text("\n".join(lines) + "\n")
    print(f"Wrote {output_dir / 'cached_multiscale_metrics.csv'}")
    print(f"Wrote {output_dir / 'cached_multiscale_summary.csv'}")
    print(summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
