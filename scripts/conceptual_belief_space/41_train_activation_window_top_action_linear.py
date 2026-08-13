#!/usr/bin/env python3
"""Linear top-action probes on smart activation-window combinations.

This script is intentionally activation-only. It builds fixed combinations of
actual CoT-token activations from the prefix window, then trains a linear probe
to predict the forced-answer q-top action. No grid/task features and no prefix
metadata are used as probe inputs.
"""

from __future__ import annotations

import argparse
import hashlib
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

from src.conceptual_belief_space.config import ACTIONS, DEFAULT_SEED

RANK_SCRIPT = Path(__file__).with_name("39_train_activation_rank_probes.py")
spec = importlib.util.spec_from_file_location("activation_rank_probe_impl", RANK_SCRIPT)
if spec is None or spec.loader is None:
    raise RuntimeError(f"Could not import helpers from {RANK_SCRIPT}")
rank_impl = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = rank_impl
spec.loader.exec_module(rank_impl)

Q_COLS = [f"q_{a}" for a in ACTIONS]
POOLINGS = ["last1", "mean20", "exp20_tau2", "exp20_tau5", "exp20_tau10", "multipool_last_mean_exp5", "concat20"]


def parse_layer_sets(text: str) -> list[tuple[int, ...]]:
    sets = []
    for chunk in text.split(";"):
        vals = tuple(int(x.strip()) for x in chunk.split(",") if x.strip())
        if vals:
            sets.append(vals)
    if not sets:
        raise ValueError("No layer sets parsed")
    return sets


def parse_csv(text: str) -> list[str]:
    return [x.strip() for x in text.split(",") if x.strip()]


def stable_id(*parts: object) -> str:
    raw = "::".join(str(p) for p in parts)
    return hashlib.sha1(raw.encode()).hexdigest()[:12]


def action_labels_to_indices(values: pd.Series | np.ndarray) -> np.ndarray:
    mapping = {a: i for i, a in enumerate(ACTIONS)}
    return np.asarray([mapping[str(v)] for v in values], dtype=np.int64)


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


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> list[list[int]]:
    out = np.zeros((len(ACTIONS), len(ACTIONS)), dtype=int)
    for t, p in zip(y_true, y_pred):
        out[int(t), int(p)] += 1
    return out.tolist()


def train_val_split_by_group(df: pd.DataFrame, train_idx: np.ndarray, seed: int, val_fraction: float) -> tuple[np.ndarray, np.ndarray]:
    groups = sorted(df.iloc[train_idx]["trajectory_id"].astype(str).unique())
    rng = random.Random(seed)
    rng.shuffle(groups)
    n_val = max(1, int(round(len(groups) * val_fraction))) if len(groups) > 1 else 0
    val_groups = set(groups[:n_val])
    local = df.iloc[train_idx]["trajectory_id"].astype(str).isin(val_groups).to_numpy()
    val_idx = train_idx[local]
    subtrain_idx = train_idx[~local]
    if len(val_idx) == 0 or len(subtrain_idx) == 0:
        return train_idx, train_idx[:0]
    return subtrain_idx, val_idx


def exp_weights(n: int, tau: float) -> np.ndarray:
    ages = np.arange(n - 1, -1, -1, dtype=np.float32)
    w = np.exp(-ages / float(tau))
    return w / max(float(w.sum()), 1e-12)


def represent_tokens(cot: np.ndarray, h_pre: np.ndarray, representation: str) -> np.ndarray:
    if representation == "raw":
        return cot.astype(np.float32, copy=False)
    delta = cot.astype(np.float32, copy=False) - h_pre.astype(np.float32, copy=False)[None, :]
    if representation == "delta_pre":
        return delta
    if representation == "unit_delta_pre":
        norm = np.linalg.norm(delta, axis=1, keepdims=True)
        return delta / np.clip(norm, 1e-12, None)
    raise ValueError(f"Unsupported representation: {representation}")


def pool_rows(tokens: np.ndarray, end_indices: np.ndarray, pooling: str, h_pre_rep: np.ndarray, window_size: int) -> np.ndarray:
    hidden = tokens.shape[1] if tokens.ndim == 2 and tokens.shape[0] else h_pre_rep.shape[0]
    pieces: list[np.ndarray] = []
    for k_raw in end_indices:
        k = int(k_raw)
        if k <= 0 or tokens.shape[0] == 0:
            window = h_pre_rep.reshape(1, hidden)
        else:
            end = min(k, tokens.shape[0])
            start = max(0, end - window_size)
            window = tokens[start:end]
            if len(window) == 0:
                window = h_pre_rep.reshape(1, hidden)
        if pooling == "last1":
            feat = window[-1]
        elif pooling == "mean20":
            feat = window.mean(axis=0)
        elif pooling.startswith("exp20_tau"):
            tau = float(pooling.replace("exp20_tau", ""))
            w = exp_weights(len(window), tau)
            feat = (window * w[:, None]).sum(axis=0)
        elif pooling == "multipool_last_mean_exp5":
            w = exp_weights(len(window), 5.0)
            feat = np.concatenate([window[-1], window.mean(axis=0), (window * w[:, None]).sum(axis=0)], axis=0)
        elif pooling == "concat20":
            padded = np.zeros((window_size, hidden), dtype=np.float32)
            clipped = window[-window_size:]
            padded[-len(clipped):] = clipped
            feat = padded.reshape(-1)
        else:
            raise ValueError(f"Unsupported pooling: {pooling}")
        pieces.append(feat.astype(np.float32, copy=False))
    return np.asarray(pieces, dtype=np.float32)


def build_feature_cache(
    df: pd.DataFrame,
    pointers: dict[str, Any],
    *,
    layers: tuple[int, ...],
    representation: str,
    pooling: str,
    window_size: int,
    output_path: Path,
    dtype: np.dtype,
    force: bool,
) -> Path:
    if output_path.exists() and not force:
        arr = np.load(output_path, mmap_mode="r")
        if arr.shape[0] == len(df):
            return output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    hidden_dim = next(iter(pointers.values())).hidden_dim
    if pooling == "multipool_last_mean_exp5":
        mult = 3
    elif pooling == "concat20":
        mult = int(window_size)
    else:
        mult = 1
    shape = (len(df), hidden_dim * mult)
    out = np.lib.format.open_memmap(output_path, mode="w+", dtype=dtype, shape=shape)
    loader = rank_impl.BF16MatrixCache(max_open=96)
    df_local = df.copy()
    df_local["_row"] = np.arange(len(df_local), dtype=np.int64)
    for decision_id, group in df_local.groupby("decision_id", sort=False):
        pointer = pointers[str(decision_id)]
        end_indices = group["activation_window_end_token_index"].astype(int).to_numpy()
        pooled_by_layer = []
        for layer in layers:
            cot_path = rank_impl.path_for_layer(pointer.cot_file, layer)
            pre_path = rank_impl.path_for_layer(pointer.pre_file, layer)
            cot_mmap = loader.get(cot_path, pointer.hidden_dim)
            cot = rank_impl.bf16_uint16_to_float32(np.asarray(cot_mmap)).astype(np.float32, copy=False)
            h_pre = loader.load_row(pre_path, pointer.pre_row_index, pointer.hidden_dim)
            tokens = represent_tokens(cot, h_pre, representation)
            if representation == "raw":
                h_pre_rep = h_pre.astype(np.float32, copy=False)
            else:
                h_pre_rep = np.zeros(pointer.hidden_dim, dtype=np.float32)
            pooled_by_layer.append(pool_rows(tokens, end_indices, pooling, h_pre_rep, window_size))
        # Fixed smart layer combination: average selected layer features before the linear probe.
        feature = np.mean(np.stack(pooled_by_layer, axis=0), axis=0).astype(np.float32, copy=False)
        out[group["_row"].to_numpy(dtype=np.int64)] = feature.astype(dtype, copy=False)
    out.flush()
    return output_path


def compute_scaler(x: np.ndarray, rows: np.ndarray, batch_size: int) -> tuple[np.ndarray, np.ndarray]:
    rows = np.asarray(rows, dtype=np.int64)
    total = np.zeros(x.shape[1], dtype=np.float64)
    total_sq = np.zeros(x.shape[1], dtype=np.float64)
    count = 0
    for start in range(0, len(rows), batch_size):
        idx = rows[start:start+batch_size]
        xb = np.asarray(x[idx], dtype=np.float32)
        total += xb.sum(axis=0, dtype=np.float64)
        total_sq += np.square(xb, dtype=np.float32).sum(axis=0, dtype=np.float64)
        count += len(idx)
    mean = (total / max(count, 1)).astype(np.float32)
    var = total_sq / max(count, 1) - mean.astype(np.float64) ** 2
    var[var < 1e-12] = 1.0
    scale = np.sqrt(var).astype(np.float32)
    scale[scale < 1e-6] = 1.0
    return mean, scale


def fit_linear_classifier(
    x: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    *,
    epochs: int,
    patience: int,
    lr: float,
    weight_decay: float,
    batch_size: int,
    seed: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    import torch

    rng = np.random.default_rng(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu"))
    mean, scale = compute_scaler(x, train_idx, batch_size=max(batch_size * 4, batch_size))
    model = torch.nn.Linear(x.shape[1], len(ACTIONS)).to(device)
    counts = np.bincount(y[train_idx], minlength=len(ACTIONS)).astype(np.float32)
    weights = counts.sum() / np.clip(counts, 1.0, None)
    weights = weights / weights.mean()
    criterion = torch.nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32, device=device))
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    best_state = None
    best_val = -1.0
    best_epoch = -1
    no_improve = 0

    def predict(indices: np.ndarray) -> np.ndarray:
        batches = []
        with torch.no_grad():
            for start in range(0, len(indices), batch_size):
                idx = indices[start:start+batch_size]
                xb = (np.asarray(x[idx], dtype=np.float32) - mean) / scale
                logits = model(torch.tensor(xb, dtype=torch.float32, device=device)).cpu().numpy()
                batches.append(logits.astype(np.float32))
        return np.concatenate(batches, axis=0) if batches else np.zeros((0, len(ACTIONS)), dtype=np.float32)

    for epoch in range(int(epochs)):
        order = rng.permutation(train_idx)
        model.train()
        for start in range(0, len(order), batch_size):
            idx = order[start:start+batch_size]
            xb = (np.asarray(x[idx], dtype=np.float32) - mean) / scale
            yb = y[idx]
            logits = model(torch.tensor(xb, dtype=torch.float32, device=device))
            loss = criterion(logits, torch.tensor(yb, dtype=torch.long, device=device))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        model.eval()
        score_idx = val_idx if len(val_idx) else train_idx
        val_logits = predict(score_idx)
        val_pred = val_logits.argmax(axis=1)
        val_bal = balanced_accuracy(y[score_idx], val_pred)
        if val_bal > best_val + 1e-5:
            best_val = val_bal
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    test_logits = predict(test_idx)
    info = {"best_epoch": int(best_epoch), "validation_balanced_accuracy": float(best_val), "device": str(device)}
    return test_logits, info


def make_predictions(df: pd.DataFrame, indices: np.ndarray, logits: np.ndarray, split_id: int, config: dict[str, Any]) -> pd.DataFrame:
    sub = df.iloc[indices].copy()
    pred = logits.argmax(axis=1)
    for i, action in enumerate(ACTIONS):
        sub[f"linear_logit_{action}"] = logits[:, i]
    sub["predicted_top_action"] = np.asarray(ACTIONS, dtype=object)[pred]
    sub["split_id"] = split_id
    for k, v in config.items():
        sub[k] = json.dumps(v) if isinstance(v, (list, tuple)) else v
    keep = [c for c in ["row_id","prefix_id","decision_id","trajectory_id","step_id","prefix_token_index","progress","top_action","agent_action","predicted_top_action","split_id",*Q_COLS,*[f"linear_logit_{a}" for a in ACTIONS],*config.keys()] if c in sub.columns]
    return sub[keep]


def add_metrics(rows: list[dict[str, Any]], df: pd.DataFrame, indices: np.ndarray, logits: np.ndarray, split_id: int, config: dict[str, Any], info: dict[str, Any]) -> None:
    y_true = action_labels_to_indices(df.iloc[indices]["top_action"])
    y_pred = logits.argmax(axis=1)
    subsets = {"all_prefixes": np.ones(len(indices), dtype=bool)}
    for col in ["strict_before_any_action_mention", "strict_before_final_action_mention"]:
        if col in df.columns:
            subsets[col] = df.iloc[indices][col].fillna(False).astype(bool).to_numpy()
    q = df.iloc[indices][Q_COLS].to_numpy(dtype=np.float64)
    for subset, mask in subsets.items():
        if not mask.any():
            continue
        metric = {
            "split_id": split_id,
            "eval_subset": subset,
            "n_rows": int(mask.sum()),
            "accuracy": float((y_pred[mask] == y_true[mask]).mean()),
            "balanced_accuracy": balanced_accuracy(y_true[mask], y_pred[mask]),
            "macro_f1": macro_f1(y_true[mask], y_pred[mask]),
            "confusion_matrix_json": json.dumps(confusion_matrix(y_true[mask], y_pred[mask])),
            **config,
            **info,
        }
        metric.update(rank_impl.metrics_for_rows(logits[mask], q[mask], tie_eps=1e-6))
        rows.append(metric)


def plot_outputs(metrics: pd.DataFrame, figure_dir: Path) -> None:
    import matplotlib.pyplot as plt

    figure_dir.mkdir(parents=True, exist_ok=True)
    all_rows = metrics[metrics["eval_subset"].eq("all_prefixes")].copy()
    if all_rows.empty:
        return
    summary = all_rows.groupby(["layer_set", "representation", "pooling"])["balanced_accuracy"].mean().reset_index()
    summary = summary.sort_values("balanced_accuracy", ascending=False).head(25)
    fig, ax = plt.subplots(figsize=(10, 5))
    labels = summary.apply(lambda r: f"{r['representation']} {r['pooling']} L{r['layer_set']}", axis=1)
    ax.bar(np.arange(len(summary)), summary["balanced_accuracy"])
    ax.axhline(0.7, color="red", lw=1, ls="--", label="0.7 target")
    ax.set_xticks(np.arange(len(summary)), labels, rotation=70, ha="right", fontsize=7)
    ax.set_ylabel("mean test balanced accuracy")
    ax.set_title("Activation-only linear top-action decoding")
    ax.legend(frameon=False)
    fig.savefig(figure_dir / "linear_top_action_balanced_accuracy.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--beliefs", default="outputs/conceptual_belief_space/action_belief_vectors_mlx_full.parquet")
    p.add_argument("--prefixes", default="outputs/prefixes.parquet")
    p.add_argument("--activation_index", default="outputs/activation/activation_token_index.parquet")
    p.add_argument("--output_dir", default="outputs/conceptual_belief_space/activation_window_top_action_linear")
    p.add_argument("--figure_dir", default="figures/conceptual_belief_space/activation_window_top_action_linear")
    p.add_argument("--layer_sets", default="10,12,14,16,18")
    p.add_argument("--representations", default="raw")
    p.add_argument("--poolings", default="mean20,exp20_tau5,last1")
    p.add_argument("--window_size", type=int, default=20)
    p.add_argument("--n_splits", type=int, default=5)
    p.add_argument("--limit_decisions", type=int, default=None)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--patience", type=int, default=7)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--weight_decay", type=float, default=1e-3)
    p.add_argument("--batch_size", type=int, default=1024)
    p.add_argument("--val_fraction", type=float, default=0.2)
    p.add_argument("--dtype", choices=["float32", "float16"], default="float32")
    p.add_argument("--force_features", action="store_true")
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    out_dir = Path(args.output_dir)
    fig_dir = Path(args.figure_dir)
    cache_dir = out_dir / "feature_cache"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    beliefs = rank_impl.prepare_belief_rows(Path(args.beliefs), limit_decisions=args.limit_decisions, seed=args.seed, prefix_path=Path(args.prefixes))
    pointers = rank_impl.build_activation_pointers(Path(args.activation_index), set(beliefs["decision_id"].astype(str)), layers=sorted({l for s in parse_layer_sets(args.layer_sets) for l in s}))
    splits = rank_impl.make_group_splits(beliefs, args.n_splits, args.seed)
    y = action_labels_to_indices(beliefs["top_action"])
    rows = []
    preds = []
    manifest = {
        "activation_only": True,
        "excluded_inputs": ["grid/task features", "progress", "prefix_token_index", "num_cot_tokens", "trajectory_id", "decision_id", "agent_action"],
        "target": "top_action = argmax(q_UP,q_DOWN,q_LEFT,q_RIGHT)",
        "split_policy": "GroupKFold by trajectory_id, with train-only trajectory-group validation split for early stopping",
        "window_policy": "prefix 0 uses pre_reasoning; later prefixes use analysis-prefix token count from outputs/prefixes.parquet",
        "n_rows": int(len(beliefs)),
        "n_decisions": int(beliefs["decision_id"].nunique()),
        "n_trajectories": int(beliefs["trajectory_id"].nunique()),
    }
    dtype = np.float32 if args.dtype == "float32" else np.float16
    for layers in parse_layer_sets(args.layer_sets):
        layer_text = "_".join(str(l) for l in layers)
        for representation in parse_csv(args.representations):
            for pooling in parse_csv(args.poolings):
                if pooling not in POOLINGS:
                    raise ValueError(f"Unsupported pooling: {pooling}")
                config_id = stable_id(layer_text, representation, pooling, args.window_size, len(beliefs))
                feature_path = cache_dir / f"features_{config_id}_layers_{layer_text}_{representation}_{pooling}.npy"
                print(f"Building/loading features layers={layers} rep={representation} pooling={pooling}")
                build_feature_cache(
                    beliefs,
                    pointers,
                    layers=layers,
                    representation=representation,
                    pooling=pooling,
                    window_size=args.window_size,
                    output_path=feature_path,
                    dtype=np.dtype(dtype),
                    force=args.force_features,
                )
                x = np.load(feature_path, mmap_mode="r")
                config = {
                    "feature_family": "activation_window_layer_mean",
                    "layer_set": layer_text,
                    "representation": representation,
                    "pooling": pooling,
                    "window_size": int(args.window_size),
                    "feature_dim": int(x.shape[1]),
                    "feature_file": str(feature_path),
                    "linear_model": "single affine layer + softmax",
                }
                for split in splits:
                    split_id = int(split["split_id"])
                    train_idx = split["train_idx"]
                    test_idx = split["test_idx"]
                    train_groups = set(beliefs.iloc[train_idx]["trajectory_id"].astype(str))
                    test_groups = set(beliefs.iloc[test_idx]["trajectory_id"].astype(str))
                    leakage = sorted(train_groups & test_groups)
                    if leakage:
                        raise RuntimeError(f"Trajectory leakage in split {split_id}: {leakage[:5]}")
                    subtrain_idx, val_idx = train_val_split_by_group(beliefs, train_idx, args.seed + split_id, args.val_fraction)
                    logits, info = fit_linear_classifier(
                        x,
                        y,
                        subtrain_idx,
                        val_idx,
                        test_idx,
                        epochs=args.epochs,
                        patience=args.patience,
                        lr=args.lr,
                        weight_decay=args.weight_decay,
                        batch_size=args.batch_size,
                        seed=args.seed + split_id,
                    )
                    add_metrics(rows, beliefs, test_idx, logits, split_id, config, info)
                    preds.append(make_predictions(beliefs, test_idx, logits, split_id, config))
                if rows:
                    pd.DataFrame(rows).to_csv(out_dir / "linear_top_action_metrics.partial.csv", index=False)
                    print(f"Finished config layers={layer_text} rep={representation} pooling={pooling}", flush=True)
    metrics = pd.DataFrame(rows)
    metrics.to_csv(out_dir / "linear_top_action_metrics.csv", index=False)
    pred_df = pd.concat(preds, ignore_index=True) if preds else pd.DataFrame()
    pred_df.to_parquet(out_dir / "linear_top_action_predictions.parquet", index=False)
    rank_impl.write_json(out_dir / "linear_top_action_manifest.json", manifest)
    plot_outputs(metrics, fig_dir)
    all_rows = metrics[metrics["eval_subset"].eq("all_prefixes")]
    best = all_rows.sort_values("balanced_accuracy", ascending=False).head(10)
    lines = [
        "# Activation Window Linear Top-Action Probe Report",
        "",
        "- Probe input: activation-only fixed pooled vectors.",
        "- Excluded: grid/task features and prefix metadata.",
        "- Target: forced q-top action, `argmax(q_UP, q_DOWN, q_LEFT, q_RIGHT)`.",
        "- Evaluation: trajectory-grouped test splits; early stopping uses a disjoint train-only trajectory validation split.",
        "",
        "## Best All-Prefix Rows",
        "",
        best[["split_id", "layer_set", "representation", "pooling", "accuracy", "balanced_accuracy", "macro_f1", "validation_balanced_accuracy", "feature_dim"]].to_markdown(index=False) if not best.empty else "No metrics.",
        "",
        "## Notes",
        "",
        "The red 0.7 line in the summary figure is the requested target. Passing it on held-out trajectory groups is required before claiming success.",
    ]
    (out_dir / "linear_top_action_report.md").write_text("\n".join(lines) + "\n")
    print(f"Wrote {out_dir / 'linear_top_action_metrics.csv'}")
    print(f"Wrote {out_dir / 'linear_top_action_predictions.parquet'}")
    print(f"Wrote {out_dir / 'linear_top_action_report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
