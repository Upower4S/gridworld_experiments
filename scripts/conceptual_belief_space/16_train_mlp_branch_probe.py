#!/usr/bin/env python
"""Train supervised MLP bottleneck branch probes for activation features.

The 2D coordinates produced here are a supervised nonlinear readout. They are
not an unsupervised raw activation manifold.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.conceptual_belief_space.config import ACTIONS
from src.conceptual_belief_space.io import ensure_dir, save_markdown, write_json
from src.conceptual_belief_space.mlp_branch_utils import (
    ACTION_TO_INDEX,
    ACTION_VERTICES,
    Q_COLS,
    SplitDef,
    action_indices_to_labels,
    action_labels_to_indices,
    branch_targets_from_q,
    class_weights,
    fit_standardizer,
    json_ready,
    load_or_create_splits,
    make_behavior_table,
    make_train_val_split,
    metric_row_from_predictions,
    nearest_vertex,
    normalize_q,
    now_run_id,
    parse_bool,
    prediction_frame,
    progress_features,
    q_metrics,
    q_top_action_coverage,
    safe_spearman,
    save_best_configs,
    scalar_metrics,
    softmax_np,
    subset_mask,
    top_actions_from_q,
    transform_features,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--beliefs", default="outputs/conceptual_belief_space/action_belief_vectors.parquet")
    parser.add_argument("--splits", default="outputs/conceptual_belief_space/splits/group_splits.json")
    parser.add_argument("--activation_metadata", default="outputs/activation/activation_feature_metadata.parquet")
    parser.add_argument("--feature_dir", default="outputs/activation/features")
    parser.add_argument("--output_dir", default="outputs/conceptual_belief_space/mlp_branch_probe")
    parser.add_argument("--layers", nargs="+", default=["8", "10", "12", "14", "16"])
    parser.add_argument("--representations", nargs="+", default=["delta_pre", "raw", "unit_delta_pre"])
    parser.add_argument("--subsets", nargs="+", default=["all_prefixes", "strict_before_any_action_mention", "strict_before_final_action_mention"])
    parser.add_argument("--group_col", default="trajectory_id")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--hidden_dim2", type=int, default=128)
    parser.add_argument("--bottleneck_dim", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--max_epochs", type=int, default=150)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--lambda_q", type=float, default=1.0)
    parser.add_argument("--lambda_top_action", type=float, default=0.5)
    parser.add_argument("--lambda_agent_action", type=float, default=0.2)
    parser.add_argument("--lambda_commitment", type=float, default=0.5)
    parser.add_argument("--lambda_anchor", type=float, default=0.1)
    parser.add_argument("--lambda_smooth", type=float, default=0.0)
    parser.add_argument("--lambda_radius", type=float, default=0.0)
    parser.add_argument("--balanced_q_loss", "--balanced-q-loss", action="store_true", default=True)
    parser.add_argument("--no-balanced_q_loss", "--no-balanced-q-loss", action="store_false", dest="balanced_q_loss")
    parser.add_argument("--q_collapse_penalty", "--q-collapse-penalty", type=float, default=0.25)
    parser.add_argument("--min_predicted_action_count", "--min-predicted-action-count", type=int, default=1)
    parser.add_argument("--standardize", action="store_true", default=True)
    parser.add_argument("--no-standardize", action="store_false", dest="standardize")
    parser.add_argument("--l2_normalize", default="false")
    parser.add_argument("--save_predictions", choices=["best", "all", "none"], default="best")
    parser.add_argument("--save_all_embeddings", action="store_true")
    parser.add_argument("--run_linear_baseline", action="store_true")
    parser.add_argument("--run_shuffled_control", action="store_true")
    parser.add_argument("--run_progress_only_control", action="store_true")
    parser.add_argument("--run_label_permutation_control", action="store_true")
    parser.add_argument("--skip_bottleneck_constraint", default="false")
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def configure_pilot(args: argparse.Namespace) -> argparse.Namespace:
    if not args.pilot:
        return args
    args.layers = ["10", "12", "14", "16"]
    args.representations = ["delta_pre", "raw", "unit_delta_pre"]
    args.subsets = ["all_prefixes"]
    args.max_epochs = min(args.max_epochs, 60)
    args.patience = min(args.patience, 8)
    args.lambda_q = 1.0
    args.lambda_top_action = 0.5
    args.lambda_commitment = 0.5
    args.lambda_anchor = 0.05
    args.lambda_smooth = 0.0
    args.lambda_radius = 0.0
    return args


def choose_device(device_arg: str):
    import torch

    if device_arg != "auto":
        return torch.device(device_arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class MLPBranchProbeTorch:
    pass


def build_torch_model(input_dim: int, args: argparse.Namespace):
    import torch
    import torch.nn as nn

    class MLPBranchProbe(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.encoder = nn.Sequential(
                nn.LayerNorm(input_dim),
                nn.Linear(input_dim, args.hidden_dim),
                nn.GELU(),
                nn.Dropout(args.dropout),
                nn.Linear(args.hidden_dim, args.hidden_dim2),
                nn.GELU(),
                nn.Dropout(args.dropout),
            )
            self.bottleneck = nn.Linear(args.hidden_dim2, args.bottleneck_dim)
            head_input = args.hidden_dim2 if parse_bool(args.skip_bottleneck_constraint) else args.bottleneck_dim
            self.q_head = nn.Linear(head_input, len(ACTIONS))
            self.top_action_head = nn.Linear(head_input, len(ACTIONS))
            self.agent_action_head = nn.Linear(head_input, len(ACTIONS))
            self.commitment_head = nn.Linear(head_input, 1)

        def forward(self, x):
            hidden = self.encoder(x)
            z = self.bottleneck(hidden)
            source = hidden if parse_bool(args.skip_bottleneck_constraint) else z
            return {
                "z_raw": z,
                "z_plot": torch.tanh(z[:, :2]),
                "q_logits": self.q_head(source),
                "top_logits": self.top_action_head(source),
                "agent_logits": self.agent_action_head(source),
                "commitment": torch.sigmoid(self.commitment_head(source)).squeeze(-1),
            }

    return MLPBranchProbe()


def torch_dataset_arrays(df: pd.DataFrame, indices: np.ndarray) -> dict[str, np.ndarray]:
    q = normalize_q(df.iloc[indices][Q_COLS].to_numpy(dtype=np.float32))
    return {
        "q": q,
        "top": action_labels_to_indices(df.iloc[indices]["top_action"]),
        "agent": action_labels_to_indices(df.iloc[indices]["agent_action"]),
        "commitment": df.iloc[indices]["entropy_commitment"].to_numpy(dtype=np.float32),
        "branch_target": branch_targets_from_q(q).astype(np.float32),
    }


def train_mlp(
    x_train: np.ndarray,
    y_train: dict[str, np.ndarray],
    x_val: np.ndarray,
    y_val: dict[str, np.ndarray],
    args: argparse.Namespace,
    device,
    *,
    seed: int,
    vertex_permutation: np.ndarray | None = None,
) -> tuple[Any, dict[str, Any]]:
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, TensorDataset

    torch.manual_seed(seed)
    np.random.seed(seed)
    model = build_torch_model(x_train.shape[1], args).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    top_weights = torch.tensor(class_weights(y_train["top"]), dtype=torch.float32, device=device)
    agent_weights = torch.tensor(class_weights(y_train["agent"]), dtype=torch.float32, device=device)
    branch_target = y_train["branch_target"].copy()
    if vertex_permutation is not None:
        q = y_train["q"]
        vertices = np.stack([ACTION_VERTICES[ACTIONS[int(i)]] for i in vertex_permutation], axis=0)
        branch_target = q @ vertices
    dataset = TensorDataset(
        torch.tensor(x_train, dtype=torch.float32),
        torch.tensor(y_train["q"], dtype=torch.float32),
        torch.tensor(y_train["top"], dtype=torch.long),
        torch.tensor(y_train["agent"], dtype=torch.long),
        torch.tensor(y_train["commitment"], dtype=torch.float32),
        torch.tensor(branch_target, dtype=torch.float32),
    )
    generator = torch.Generator()
    generator.manual_seed(seed)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, generator=generator)
    best_state = None
    best_score = -float("inf")
    best_epoch = -1
    best_val_info: dict[str, Any] = {}
    bad_epochs = 0

    for epoch in range(int(args.max_epochs)):
        model.train()
        for batch in loader:
            xb, q, top, agent, commitment, branch = [item.to(device) for item in batch]
            out = model(xb)
            log_q = F.log_softmax(out["q_logits"], dim=1)
            loss_q_per_row = -(q * log_q).sum(dim=1)
            if bool(args.balanced_q_loss):
                q_sample_weights = top_weights[top]
                loss_q = (loss_q_per_row * q_sample_weights).sum() / torch.clamp(q_sample_weights.sum(), min=1e-8)
            else:
                loss_q = loss_q_per_row.mean()
            loss_top = F.cross_entropy(out["top_logits"], top, weight=top_weights)
            loss_agent = F.cross_entropy(out["agent_logits"], agent, weight=agent_weights)
            loss_commit = F.mse_loss(out["commitment"], commitment)
            loss_anchor = F.mse_loss(out["z_plot"], branch)
            radius = torch.linalg.norm(out["z_plot"], dim=1)
            loss_radius = F.mse_loss(radius, commitment)
            loss = (
                args.lambda_q * loss_q
                + args.lambda_top_action * loss_top
                + args.lambda_agent_action * loss_agent
                + args.lambda_commitment * loss_commit
                + args.lambda_anchor * loss_anchor
                + args.lambda_radius * loss_radius
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if args.grad_clip and args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
        val_pred = predict_mlp(model, x_val, args.batch_size, device) if len(x_val) else {}
        if len(x_val):
            q_m = q_metrics(y_val["q"], val_pred["q_pred"], top_action=action_indices_to_labels(y_val["top"]), agent_action=action_indices_to_labels(y_val["agent"]))
            coverage = q_top_action_coverage(
                val_pred["q_pred"],
                true_top_action=action_indices_to_labels(y_val["top"]),
                min_count=max(1, int(args.min_predicted_action_count)),
            )
            commit_s = safe_spearman(y_val["commitment"], val_pred["commitment_pred"])
            anchor_mse = float(np.mean((val_pred["z_plot"] - y_val["branch_target"]) ** 2))
            collapse_penalty = float(args.q_collapse_penalty) * float(len(coverage["missing_actions"]))
            score = q_m["q_top_action_balanced_accuracy"] + 0.5 * (commit_s if np.isfinite(commit_s) else 0.0) - 0.1 * q_m["q_kl"] - collapse_penalty
            val_info = {
                "validation_q_kl": q_m.get("q_kl", float("nan")),
                "validation_q_top_action_balanced_accuracy": q_m.get("q_top_action_balanced_accuracy", float("nan")),
                "validation_commitment_spearman": commit_s,
                "validation_anchor_mse": anchor_mse,
                "validation_q_pred_top_num_actions": coverage["num_predicted_actions"],
                "validation_q_pred_top_min_count_on_present_true_actions": coverage["min_predicted_count_on_present_true_actions"],
                "validation_q_pred_top_action_collapse": coverage["has_action_collapse"],
                "validation_q_pred_top_missing_actions_json": json.dumps(coverage["missing_actions"]),
                "validation_q_collapse_penalty": collapse_penalty,
            }
            for action in ACTIONS:
                val_info[f"validation_true_top_count_{action}"] = coverage["true_counts"][action]
                val_info[f"validation_q_pred_top_count_{action}"] = coverage["pred_counts"][action]
        else:
            score = -float(epoch)
            val_info = {
                "validation_q_kl": float("nan"),
                "validation_q_top_action_balanced_accuracy": float("nan"),
                "validation_commitment_spearman": float("nan"),
                "validation_anchor_mse": float("nan"),
                "validation_q_pred_top_num_actions": 0,
                "validation_q_pred_top_min_count_on_present_true_actions": 0,
                "validation_q_pred_top_action_collapse": True,
                "validation_q_pred_top_missing_actions_json": json.dumps(ACTIONS),
                "validation_q_collapse_penalty": 0.0,
            }
        if score > best_score:
            best_score = float(score)
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_val_info = val_info.copy()
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, {
        "best_epoch": best_epoch,
        "validation_composite": best_score,
        **best_val_info,
    }


def predict_mlp(model, x: np.ndarray, batch_size: int, device) -> dict[str, np.ndarray]:
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, TensorDataset

    if len(x) == 0:
        return {
            "q_pred": np.zeros((0, len(ACTIONS)), dtype=np.float32),
            "commitment_pred": np.zeros(0, dtype=np.float32),
            "z_raw": np.zeros((0, 2), dtype=np.float32),
            "z_plot": np.zeros((0, 2), dtype=np.float32),
            "top_head_pred": np.asarray([], dtype=object),
            "agent_head_pred": np.asarray([], dtype=object),
        }
    loader = DataLoader(TensorDataset(torch.tensor(x, dtype=torch.float32)), batch_size=batch_size, shuffle=False)
    qs, commits, z_raws, z_plots, top_preds, agent_preds = [], [], [], [], [], []
    model.eval()
    with torch.no_grad():
        for (xb,) in loader:
            out = model(xb.to(device))
            qs.append(F.softmax(out["q_logits"], dim=1).detach().cpu().numpy())
            commits.append(out["commitment"].detach().cpu().numpy())
            z_raws.append(out["z_raw"][:, :2].detach().cpu().numpy())
            z_plots.append(out["z_plot"].detach().cpu().numpy())
            top_preds.append(out["top_logits"].argmax(dim=1).detach().cpu().numpy())
            agent_preds.append(out["agent_logits"].argmax(dim=1).detach().cpu().numpy())
    return {
        "q_pred": normalize_q(np.concatenate(qs, axis=0)),
        "commitment_pred": np.concatenate(commits, axis=0).astype(np.float32),
        "z_raw": np.concatenate(z_raws, axis=0).astype(np.float32),
        "z_plot": np.concatenate(z_plots, axis=0).astype(np.float32),
        "top_head_pred": action_indices_to_labels(np.concatenate(top_preds, axis=0)),
        "agent_head_pred": action_indices_to_labels(np.concatenate(agent_preds, axis=0)),
    }


def make_prediction_frames(
    df: pd.DataFrame,
    split_indices: dict[str, np.ndarray],
    pred_by_role: dict[str, dict[str, np.ndarray]],
    *,
    run_id: str,
    split_id: int,
    layer: int,
    representation: str,
    subset: str,
    model_type: str,
) -> pd.DataFrame:
    frames = []
    for role, indices in split_indices.items():
        pred = pred_by_role[role]
        frame = prediction_frame(
            df,
            indices,
            run_id=run_id,
            split_id=split_id,
            layer=layer,
            representation=representation,
            subset=subset,
            model_type=model_type,
            split_role=role,
            q_pred=pred["q_pred"],
            commitment_pred=pred["commitment_pred"],
            z_raw=pred["z_raw"],
            z_plot=pred["z_plot"],
        )
        if "top_head_pred" in pred:
            frame["top_head_pred"] = pred["top_head_pred"]
        if "agent_head_pred" in pred:
            frame["agent_head_pred"] = pred["agent_head_pred"]
        frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def train_linear_baseline(
    x_train: np.ndarray,
    y_train: dict[str, np.ndarray],
    x_val: np.ndarray,
    y_val: dict[str, np.ndarray],
    x_test: np.ndarray,
    y_test: dict[str, np.ndarray],
) -> dict[str, dict[str, np.ndarray]]:
    from sklearn.linear_model import LogisticRegression, Ridge

    q_model = Ridge(alpha=1.0).fit(x_train, y_train["q"])
    c_model = Ridge(alpha=1.0).fit(x_train, y_train["commitment"])
    top_model = LogisticRegression(max_iter=1000, class_weight="balanced").fit(x_train, y_train["top"])
    agent_model = LogisticRegression(max_iter=1000, class_weight="balanced").fit(x_train, y_train["agent"])

    def pred(x: np.ndarray) -> dict[str, np.ndarray]:
        q_raw = q_model.predict(x)
        q_pred = softmax_np(q_raw, axis=1)
        commitment_pred = np.clip(c_model.predict(x), 0.0, 1.0).astype(np.float32)
        z_plot = branch_targets_from_q(q_pred)
        return {
            "q_pred": q_pred,
            "commitment_pred": commitment_pred,
            "z_raw": z_plot.copy(),
            "z_plot": z_plot,
            "top_head_pred": action_indices_to_labels(top_model.predict(x)),
            "agent_head_pred": action_indices_to_labels(agent_model.predict(x)),
        }

    return {"train": pred(x_train), "val": pred(x_val), "test": pred(x_test)}


def run_one_model(
    *,
    df: pd.DataFrame,
    features: np.ndarray | None,
    subset_indices: np.ndarray,
    split: SplitDef,
    layer: int,
    representation: str,
    subset: str,
    args: argparse.Namespace,
    device,
    model_type: str,
    seed: int,
    use_progress_features: bool = False,
    shuffle_targets: bool = False,
    vertex_permutation: np.ndarray | None = None,
) -> tuple[dict[str, Any], pd.DataFrame, dict[str, Any] | None]:
    train_pool = np.intersect1d(split.train_indices, subset_indices, assume_unique=False)
    test = np.intersect1d(split.test_indices, subset_indices, assume_unique=False)
    train, val = make_train_val_split(df, train_pool, split.group_col, seed=seed)
    if len(train) == 0 or len(test) == 0:
        raise ValueError(f"empty train/test after subset filtering: train={len(train)} test={len(test)}")
    if use_progress_features:
        x_train, scaler = progress_features(df, train)
        x_val, _ = progress_features(df, val, scaler)
        x_test, _ = progress_features(df, test, scaler)
    else:
        if features is None:
            raise ValueError("features are required")
        scaler = fit_standardizer(features, train, standardize=args.standardize)
        x_train = transform_features(features, train, scaler, l2_normalize=parse_bool(args.l2_normalize))
        x_val = transform_features(features, val, scaler, l2_normalize=parse_bool(args.l2_normalize))
        x_test = transform_features(features, test, scaler, l2_normalize=parse_bool(args.l2_normalize))
    y_train = torch_dataset_arrays(df, train)
    y_val = torch_dataset_arrays(df, val)
    y_test = torch_dataset_arrays(df, test)
    if shuffle_targets:
        rng = np.random.default_rng(seed)
        perm = rng.permutation(len(train))
        for key in ["q", "top", "agent", "commitment", "branch_target"]:
            y_train[key] = y_train[key][perm]
    run_id = now_run_id(
        model_type,
        split=split.split_id,
        layer=layer,
        rep=representation,
        subset=subset,
    )
    if model_type == "linear_baseline":
        pred_by_role = train_linear_baseline(x_train, y_train, x_val, y_val, x_test, y_test)
        train_info = {"best_epoch": -1, "validation_composite": float("nan")}
        model_payload = None
    else:
        model, train_info = train_mlp(
            x_train,
            y_train,
            x_val,
            y_val,
            args,
            device,
            seed=seed,
            vertex_permutation=vertex_permutation,
        )
        pred_by_role = {
            "train": predict_mlp(model, x_train, args.batch_size, device),
            "val": predict_mlp(model, x_val, args.batch_size, device),
            "test": predict_mlp(model, x_test, args.batch_size, device),
        }
        model_payload = {"model": model, "scaler": scaler, "train_info": train_info}
    pred_frame = make_prediction_frames(
        df,
        {"train": train, "val": val, "test": test},
        pred_by_role,
        run_id=run_id,
        split_id=split.split_id,
        layer=layer,
        representation=representation,
        subset=subset,
        model_type=model_type,
    )
    metric = metric_row_from_predictions(
        pred_frame,
        run_id=run_id,
        split_id=split.split_id,
        layer=layer,
        representation=representation,
        subset=subset,
        model_type=model_type,
        train_n=len(train),
        val_n=len(val),
        test_n=len(test),
        n_train_trajectories=len(set(df.iloc[train][split.group_col].astype(str))),
        n_val_trajectories=len(set(df.iloc[val][split.group_col].astype(str))),
        n_test_trajectories=len(set(df.iloc[test][split.group_col].astype(str))),
        args_dict=vars(args),
        best_epoch=train_info.get("best_epoch"),
        validation_composite=train_info.get("validation_composite"),
        warnings="MLP heads read hidden layer directly; not valid for branch visualization" if parse_bool(args.skip_bottleneck_constraint) else "",
    )
    for key, value in train_info.items():
        if str(key).startswith("validation_"):
            metric[key] = value
    if bool(metric.get("test_q_pred_top_action_collapse", False)):
        existing = str(metric.get("warnings", "") or "")
        collapse_warning = f"q_pred_top_action_collapse={metric.get('test_q_pred_top_missing_actions_json', '[]')}"
        metric["warnings"] = f"{existing}; {collapse_warning}" if existing else collapse_warning
    return metric, pred_frame, model_payload


def save_checkpoint(path: Path, payload: dict[str, Any], args: argparse.Namespace, split: SplitDef, metric: dict[str, Any]) -> None:
    import torch

    ensure_dir(path.parent)
    model = payload["model"]
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "scaler": payload["scaler"],
        "action_label_mapping": ACTION_TO_INDEX,
        "action_vertices": {k: v.tolist() for k, v in ACTION_VERTICES.items()},
        "training_args": vars(args),
        "split_info": {
            "split_id": split.split_id,
            "group_col": split.group_col,
            "train_groups": split.train_groups,
            "test_groups": split.test_groups,
        },
        "best_epoch": payload["train_info"].get("best_epoch"),
        "metrics": json_ready(metric),
    }
    torch.save(checkpoint, path)


def main() -> None:
    args = configure_pilot(parse_args())
    output_dir = Path(args.output_dir)
    models_dir = ensure_dir(output_dir / "models")
    predictions_dir = ensure_dir(output_dir / "predictions")
    ensure_dir(output_dir / "embeddings")
    ensure_dir(output_dir / "controls")
    reports_dir = ensure_dir(output_dir / "reports")
    ensure_dir("figures/conceptual_belief_space/mlp_branch_probe/branch_manifold")
    ensure_dir("figures/conceptual_belief_space/mlp_branch_probe/trajectories")
    ensure_dir("figures/conceptual_belief_space/mlp_branch_probe/metrics")
    ensure_dir("figures/conceptual_belief_space/mlp_branch_probe/controls")

    layers = [int(v) for v in args.layers]
    representations = [str(v) for v in args.representations]
    if args.dry_run:
        write_json(
            output_dir / "dry_run_manifest.json",
            {
                "layers": layers,
                "representations": representations,
                "subsets": args.subsets,
                "args": json_ready(vars(args)),
            },
        )
        print(json.dumps({"dry_run": True, "layers": layers, "representations": representations}, indent=2))
        return

    import torch

    device = choose_device(args.device)
    df = make_behavior_table(args.beliefs)
    if Path(args.activation_metadata).exists():
        meta = pd.read_parquet(args.activation_metadata)
        if len(meta) != len(df):
            raise ValueError(f"activation metadata rows {len(meta)} != behavior rows {len(df)}")
    splits = load_or_create_splits(args.splits, df, args.group_col, seed=args.random_seed)
    if not splits:
        raise ValueError("no splits available")

    metrics: list[dict[str, Any]] = []
    prediction_frames: list[pd.DataFrame] = []
    failures: list[str] = []
    jobs = 0

    for layer in layers:
        for representation in representations:
            feature_path = Path(args.feature_dir) / f"layer_{layer:02d}_{representation}.npy"
            if not feature_path.exists():
                failures.append(f"missing feature file: {feature_path}")
                continue
            features = np.load(feature_path, mmap_mode="r")
            if features.shape[0] != len(df):
                raise ValueError(f"{feature_path} rows {features.shape[0]} != behavior rows {len(df)}")
            print(f"Loaded {feature_path} shape={features.shape} dtype={features.dtype}")
            for subset in args.subsets:
                try:
                    sub_idx = np.flatnonzero(subset_mask(df, subset))
                except Exception as exc:
                    failures.append(f"skipped subset {subset}: {exc}")
                    continue
                if len(sub_idx) == 0:
                    failures.append(f"skipped subset {subset}: zero rows")
                    continue
                for split in splits:
                    base_seed = args.random_seed + split.split_id * 1000 + layer * 17 + hash((representation, subset)) % 997
                    model_specs = [("mlp_branch", {})]
                    if args.run_linear_baseline:
                        model_specs.append(("linear_baseline", {}))
                    if args.run_shuffled_control:
                        model_specs.append(("mlp_shuffled_target", {"shuffle_targets": True}))
                    if args.run_progress_only_control:
                        model_specs.append(("progress_only", {"use_progress_features": True}))
                    if args.run_label_permutation_control:
                        rng = np.random.default_rng(base_seed)
                        perm = rng.permutation(len(ACTIONS))
                        model_specs.append(("label_permutation", {"vertex_permutation": perm}))
                    for model_type, opts in model_specs:
                        jobs += 1
                        print(f"Training split={split.split_id} layer={layer} repr={representation} subset={subset} model={model_type}")
                        try:
                            metric, pred, payload = run_one_model(
                                df=df,
                                features=features,
                                subset_indices=sub_idx,
                                split=split,
                                layer=layer,
                                representation=representation,
                                subset=subset,
                                args=args,
                                device=device,
                                model_type=model_type,
                                seed=base_seed + len(model_type),
                                **opts,
                            )
                            metrics.append(metric)
                            if args.save_predictions == "all" or args.save_all_embeddings or model_type == "mlp_branch":
                                prediction_frames.append(pred)
                            if payload is not None and model_type == "mlp_branch":
                                ckpt_name = f"{metric['run_id']}.pt"
                                save_checkpoint(models_dir / ckpt_name, payload, args, split, metric)
                        except Exception as exc:
                            warning = f"{model_type} split={split.split_id} layer={layer} repr={representation} subset={subset}: {type(exc).__name__}: {exc}"
                            print(f"WARNING: {warning}")
                            failures.append(warning)
                            metrics.append(
                                {
                                    "run_id": now_run_id("failed", split=split.split_id, layer=layer, rep=representation, subset=subset),
                                    "split_id": split.split_id,
                                    "layer": layer,
                                    "representation": representation,
                                    "subset": subset,
                                    "model_type": model_type,
                                    "status": "failed",
                                    "warnings": warning,
                                }
                            )

    metrics_df = pd.DataFrame(metrics)
    ensure_dir(output_dir)
    metrics_path = output_dir / "mlp_branch_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)

    if prediction_frames and args.save_predictions != "none":
        pred_df = pd.concat(prediction_frames, ignore_index=True)
        if args.save_predictions == "best" and not metrics_df.empty:
            # Keep all MLP branch predictions plus baselines/controls needed for plots.
            keep_runs = set(metrics_df.loc[metrics_df["status"].astype(str).eq("ok"), "run_id"].astype(str).tolist())
            pred_df = pred_df[pred_df["run_id"].astype(str).isin(keep_runs)]
        pred_path = predictions_dir / "mlp_branch_predictions.parquet"
        pred_df.to_parquet(pred_path, index=False)
    else:
        pred_path = predictions_dir / "mlp_branch_predictions.parquet"
        pd.DataFrame().to_parquet(pred_path, index=False)

    configs = save_best_configs(metrics_df, output_dir / "best_mlp_branch_configs.json")
    collapse_rows = (
        int(metrics_df["test_q_pred_top_action_collapse"].fillna(False).astype(bool).sum())
        if "test_q_pred_top_action_collapse" in metrics_df.columns
        else 0
    )
    report = [
        "# MLP Branch Probe Training Report",
        "",
        "This output is a supervised MLP bottleneck embedding, not an unsupervised raw activation manifold.",
        "",
        f"- device: `{device}`",
        f"- balanced q loss: `{bool(args.balanced_q_loss)}`",
        f"- q collapse penalty: `{args.q_collapse_penalty}`",
        f"- min predicted action count: `{args.min_predicted_action_count}`",
        f"- jobs attempted: {jobs}",
        f"- metric rows: {len(metrics_df)}",
        f"- ok rows: {int((metrics_df.get('status', pd.Series(dtype=str)).astype(str) == 'ok').sum()) if 'status' in metrics_df else len(metrics_df)}",
        f"- q-head action-collapse metric rows: {collapse_rows}",
        f"- failures: {len(failures)}",
        f"- metrics: `{metrics_path}`",
        f"- predictions: `{pred_path}`",
        f"- best config keys: {', '.join(configs.keys()) if configs else 'none'}",
    ]
    if failures:
        report.extend(["", "## Warnings", ""])
        report.extend(f"- {item}" for item in failures[:50])
    save_markdown(reports_dir / "16_mlp_branch_training_report.md", "\n".join(report))
    print(
        json.dumps(
            {
                "metrics": str(metrics_path),
                "predictions": str(pred_path),
                "best_configs": str(output_dir / "best_mlp_branch_configs.json"),
                "rows": int(len(metrics_df)),
                "failures": len(failures),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
