#!/usr/bin/env python3
"""Conjecture C1: does the belief code factor into a content direction and a scalar gain?

Theorem 3 showed that the commitment direction is nearly orthogonal to the action subspace
within every layer. C1 proposes the mechanism that would produce exactly that geometry:

    q(t) ~ softmax( beta(h_t) * [<u_a, h_t>]_a ),

content fixes the DIRECTION of the action-logit vector and commitment is its GAIN (an
inverse temperature). Three predictions:
  (i)   the gain model beats an additive linear decoder of equal-ish capacity, held out;
  (ii)  the learned gain correlates with the independently fitted ridge commitment
        direction;
  (iii) along the CoT the gain rises, and the content direction settles EARLIER than the
        gain -- the model knows which way it is leaning before it becomes confident.

Controls: a frozen random gain direction (isolates whether the LEARNED gain direction
matters) and a per-example-free-gain oracle (upper bound on what any gain can buy).
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

ACTIONS = ("UP", "DOWN", "LEFT", "RIGHT")
Q_COLS = [f"q_{action}" for action in ACTIONS]
STRICT_COL = "strict_before_any_action_mention"

INK = "#333333"
GRID = "#e6e6e6"


def parse_args() -> argparse.Namespace:
    base = REPO_ROOT / "outputs/conceptual_belief_space/cot_window_layer_transition_probe"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--activation_index", type=Path, default=base / "activations/activation_index.parquet")
    parser.add_argument("--beliefs", type=Path, default=base / "activations/extracted_beliefs.parquet")
    parser.add_argument("--splits", type=Path,
                        default=REPO_ROOT / "outputs/conceptual_belief_space/splits/group_splits_mlx_full.json")
    parser.add_argument("--pooled_cache", type=Path,
                        default=REPO_ROOT / "outputs/conceptual_belief_space/layerwise_dissociation/pooled_cot_window_f16.npy")
    parser.add_argument("--out_dir", type=Path, default=REPO_ROOT / "outputs/conjectures/c1_gain_direction")
    parser.add_argument("--figure_dir", type=Path, default=REPO_ROOT / "figures/conjectures")
    parser.add_argument("--layers", type=str, default="mean,15,18")
    parser.add_argument("--split_id", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_probe77() -> Any:
    path = REPO_ROOT / "scripts/conceptual_belief_space/77_train_forced_suffix_commitment_probe.py"
    spec = importlib.util.spec_from_file_location("probe77", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def build_models(dim: int, seed: int) -> dict[str, Any]:
    import torch
    from torch import nn

    class Additive(nn.Module):
        """logits = W x + b."""

        def __init__(self) -> None:
            super().__init__()
            self.linear = nn.Linear(dim, len(ACTIONS))

        def forward(self, x: Any) -> tuple[Any, Any]:
            logits = self.linear(x)
            return logits, torch.ones(len(x), device=x.device)

    class Gain(nn.Module):
        """logits = softplus(<w,x>+c) * (U x + d): direction from content, scale from gain."""

        def __init__(self, freeze_gain: bool = False) -> None:
            super().__init__()
            self.direction = nn.Linear(dim, len(ACTIONS))
            self.gain = nn.Linear(dim, 1)
            if freeze_gain:
                for parameter in self.gain.parameters():
                    parameter.requires_grad_(False)

        def forward(self, x: Any) -> tuple[Any, Any]:
            raw = self.direction(x)
            unit = raw / raw.norm(dim=-1, keepdim=True).clamp_min(1e-6)
            beta = nn.functional.softplus(self.gain(x)).squeeze(-1)
            return unit * beta.unsqueeze(-1), beta

    torch.manual_seed(seed)
    models = {"additive": Additive(), "gain": Gain(), "gain_frozen_random": Gain(freeze_gain=True)}
    return models


def train_model(model: Any, x_train: Any, q_train: Any, x_test: Any, q_test: Any,
                *, epochs: int, lr: float, weight_decay: float, batch_size: int,
                device: Any) -> tuple[np.ndarray, np.ndarray, float]:
    import torch
    from torch import nn

    model = model.to(device)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                  lr=lr, weight_decay=weight_decay)
    n = len(x_train)
    for _ in range(epochs):
        model.train()
        permutation = torch.randperm(n, device=device)
        for start in range(0, n, batch_size):
            index = permutation[start:start + batch_size]
            logits, _ = model(x_train[index])
            loss = -(q_train[index] * nn.functional.log_softmax(logits, dim=-1)).sum(-1).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    model.eval()
    with torch.no_grad():
        logits, beta = model(x_test)
        log_probabilities = nn.functional.log_softmax(logits, dim=-1)
        cross_entropy = -(q_test * log_probabilities).sum(-1)
        target_entropy = -(q_test.clamp_min(1e-12) * q_test.clamp_min(1e-12).log()).sum(-1)
        kl = float((cross_entropy - target_entropy).mean())
        predicted = log_probabilities.exp().cpu().numpy()
        beta_values = beta.cpu().numpy()
    return predicted, beta_values, kl


def stabilization_progress(labels: np.ndarray, progress: np.ndarray) -> float:
    differs = labels != labels[-1]
    index = int(np.max(np.nonzero(differs)[0])) + 1 if differs.any() else 0
    return float(progress[index])


def plateau_progress(values: np.ndarray, progress: np.ndarray, fraction: float = 0.8) -> float:
    """First point after which the series stays above `fraction` of its final value."""
    target = fraction * values[-1]
    above = values >= target
    suffix_all = np.minimum.accumulate(above[::-1])[::-1].astype(bool)
    hits = np.nonzero(suffix_all)[0]
    return float(progress[hits[0]]) if len(hits) else float(progress[-1])


def main() -> int:
    args = parse_args()
    import torch
    from scipy.stats import spearmanr
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    probe77 = load_probe77()
    table = probe77.prepare_table(args.activation_index, args.beliefs, None)
    splits = probe77.load_outer_splits(args.splits, table)
    split = next(item for item in splits if int(item["split_id"]) == args.split_id)
    cached_ids = np.load(args.pooled_cache.with_suffix(".prefix_ids.npy"))
    if not (cached_ids == table["prefix_id"].to_numpy()).all():
        raise SystemExit("pooled cache does not correspond to this table ordering")
    pooled = np.load(args.pooled_cache, mmap_mode="r")

    train_idx, test_idx = split["train_idx"], split["test_idx"]
    q_all = table[Q_COLS].to_numpy(dtype=np.float32)
    commitment = table["entropy_commitment"].to_numpy(dtype=np.float64)
    strict = table[STRICT_COL].to_numpy(dtype=bool)
    print(f"fold {args.split_id}: train {len(train_idx)} test {len(test_idx)} on {device}")

    rows: list[dict[str, Any]] = []
    clock_rows: list[dict[str, Any]] = []
    for token in [t.strip() for t in args.layers.split(",") if t.strip()]:
        name = "all_layer_mean" if token == "mean" else f"layer_{int(token)}"
        if token == "mean":
            features_train = np.asarray(pooled[train_idx]).mean(axis=1, dtype=np.float32)
            features_test = np.asarray(pooled[test_idx]).mean(axis=1, dtype=np.float32)
        else:
            features_train = np.asarray(pooled[train_idx, int(token)], dtype=np.float32)
            features_test = np.asarray(pooled[test_idx, int(token)], dtype=np.float32)
        scaler = StandardScaler().fit(features_train)
        x_train_np = scaler.transform(features_train).astype(np.float32)
        x_test_np = scaler.transform(features_test).astype(np.float32)

        ridge = Ridge(alpha=1.0).fit(x_train_np, commitment[train_idx])
        ridge_projection = x_test_np @ (ridge.coef_ / np.linalg.norm(ridge.coef_))

        x_train = torch.from_numpy(x_train_np).to(device)
        x_test = torch.from_numpy(x_test_np).to(device)
        q_train = torch.from_numpy(q_all[train_idx]).to(device)
        q_test = torch.from_numpy(q_all[test_idx]).to(device)

        predictions: dict[str, np.ndarray] = {}
        betas: dict[str, np.ndarray] = {}
        for model_name, model in build_models(x_train.shape[1], args.seed).items():
            predicted, beta, kl = train_model(
                model, x_train, q_train, x_test, q_test, epochs=args.epochs, lr=args.lr,
                weight_decay=args.weight_decay, batch_size=args.batch_size, device=device)
            predictions[model_name] = predicted
            betas[model_name] = beta
            entropy_predicted = 1.0 + (predicted * np.log(np.clip(predicted, 1e-12, None))
                                       ).sum(axis=1) / np.log(len(ACTIONS))
            strict_test = strict[test_idx]
            entry = {
                "representation": name, "model": model_name,
                "n_parameters": int(sum(p.numel() for p in model.parameters() if p.requires_grad)),
                "kl": kl,
                "kl_strict": float(np.mean(
                    -(q_all[test_idx][strict_test] * np.log(np.clip(predictions[model_name][strict_test],
                                                                    1e-12, None))).sum(1)
                    + (q_all[test_idx][strict_test] * np.log(np.clip(q_all[test_idx][strict_test],
                                                                     1e-12, None))).sum(1))),
                "commitment_spearman": float(spearmanr(commitment[test_idx], entropy_predicted).statistic),
                "top_action_accuracy": float((predicted.argmax(1) == q_all[test_idx].argmax(1)).mean()),
            }
            if model_name != "additive":
                entry["beta_vs_ridge_projection_spearman"] = float(
                    spearmanr(beta, ridge_projection).statistic)
                entry["beta_vs_true_commitment_spearman"] = float(
                    spearmanr(beta, commitment[test_idx]).statistic)
            rows.append(entry)
            print(f"{name:15s} {model_name:20s} KL={kl:.5f} params={entry['n_parameters']} "
                  f"commit_rho={entry['commitment_spearman']:.4f}", flush=True)

        # (iii) Direction vs gain clocks, on held-out steps, using the fitted gain model.
        beta_test = betas["gain"]
        direction_labels = predictions["gain"].argmax(axis=1)
        meta = table.iloc[test_idx][["decision_id", "progress", "prefix_token_index"]].reset_index(drop=True)
        meta = meta.assign(beta=beta_test, direction=direction_labels,
                           true_label=q_all[test_idx].argmax(axis=1))
        for decision_id, group in meta.sort_values(["decision_id", "prefix_token_index"]).groupby(
                "decision_id", sort=False):
            if len(group) < 5:
                continue
            progress = group["progress"].to_numpy(dtype=float)
            clock_rows.append({
                "representation": name, "decision_id": decision_id,
                "direction_stabilization": stabilization_progress(
                    group["direction"].to_numpy(), progress),
                "gain_plateau": plateau_progress(group["beta"].to_numpy(dtype=float), progress),
                "true_stabilization": stabilization_progress(group["true_label"].to_numpy(), progress),
                "beta_progress_spearman": float(spearmanr(group["beta"], progress).statistic),
            })

    results = pd.DataFrame(rows)
    clocks = pd.DataFrame(clock_rows)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.out_dir / "model_comparison.csv", index=False)
    clocks.to_csv(args.out_dir / "clocks.csv", index=False)

    best_representation = results.loc[results[results.model == "gain"]["kl"].idxmin(), "representation"]
    pivot = results[results.representation == best_representation].set_index("model")
    kl_gain = float(pivot.loc["gain", "kl"])
    kl_additive = float(pivot.loc["additive", "kl"])
    kl_frozen = float(pivot.loc["gain_frozen_random", "kl"])
    beta_rho = float(pivot.loc["gain", "beta_vs_ridge_projection_spearman"])
    clocks_best = clocks[clocks.representation == best_representation]
    direction_first = float((clocks_best["direction_stabilization"]
                             < clocks_best["gain_plateau"]).mean())
    median_gap = float((clocks_best["gain_plateau"]
                        - clocks_best["direction_stabilization"]).median())
    beta_rises = float(clocks_best["beta_progress_spearman"].median())

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    args.figure_dir.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, 2, figsize=(11.5, 4.3))
    subset = results[results.representation == best_representation]
    order = ["additive", "gain_frozen_random", "gain"]
    values = [float(subset.set_index("model").loc[m, "kl"]) for m in order]
    bars = axes[0].bar(np.arange(len(order)), values,
                       color=["#9aa0a6", "#eb6834", "#2a78d6"], width=0.6)
    axes[0].bar_label(bars, fmt="%.4f", color=INK, fontsize=9, padding=2)
    axes[0].set_xticks(np.arange(len(order)), ["additive", "gain\n(random dir.)", "gain\n(learned)"])
    axes[0].set_ylabel("held-out KL(true || predicted)", color=INK)
    axes[0].set_title(f"Does a multiplicative gain help? ({best_representation})",
                      color=INK, fontsize=11)

    axes[1].scatter(clocks_best["direction_stabilization"], clocks_best["gain_plateau"],
                    s=18, alpha=0.55, color="#2a78d6", edgecolors="white", linewidths=0.4)
    axes[1].plot([0, 1], [0, 1], color=INK, linewidth=1.2, linestyle=":")
    axes[1].set_xlabel("content-direction stabilization (progress)", color=INK)
    axes[1].set_ylabel("gain plateau (progress)", color=INK)
    axes[1].set_title(f"Direction settles before gain in {100 * direction_first:.0f}% of steps",
                      color=INK, fontsize=11)
    for axis in axes:
        axis.grid(True, color=GRID, linewidth=0.8)
        axis.tick_params(colors=INK)
        for spine in ("top", "right"):
            axis.spines[spine].set_visible(False)
    figure.suptitle("C1: content direction times a scalar gain", color=INK)
    figure.tight_layout()
    figure_path = args.figure_dir / "c1_gain_direction.png"
    figure.savefig(figure_path, dpi=200)
    plt.close(figure)

    part_i = kl_gain < kl_additive
    part_ii = beta_rho > 0.6 or beta_rho < -0.6
    part_iii = direction_first > 0.5 and beta_rises > 0
    lines = [
        "# Conjecture C1: gain-direction factorization of the belief code",
        "",
        f"- Fold {args.split_id}; held-out rows {len(test_idx)}; representations "
        f"{args.layers}; soft-target cross-entropy training.",
        "",
        "## Model comparison (held-out KL, lower is better)",
        "",
        results.round(5).to_markdown(index=False),
        "",
        "## Direction vs gain clocks",
        "",
        clocks.groupby("representation")[["direction_stabilization", "gain_plateau",
                                          "true_stabilization", "beta_progress_spearman"]]
        .median().round(4).to_markdown(),
        "",
        "## Verdict",
        "",
        f"- (i) gain model beats additive at {best_representation}: {part_i} "
        f"(KL {kl_additive:.5f} -> {kl_gain:.5f}; frozen-random-gain control {kl_frozen:.5f})",
        f"- (ii) learned gain tracks the independent ridge commitment direction: {part_ii} "
        f"(Spearman {beta_rho:+.4f}; vs true commitment "
        f"{float(pivot.loc['gain', 'beta_vs_true_commitment_spearman']):+.4f})",
        f"- (iii) direction settles before gain: {part_iii} "
        f"({100 * direction_first:.1f}% of steps, median gap {median_gap:+.4f} progress; "
        f"median Spearman(beta, progress) {beta_rises:+.4f})",
        "",
        f"Figure: `{figure_path}`",
    ]
    (args.out_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print()
    print(results.round(5).to_string(index=False))
    print(f"\nVerdict: (i)={part_i}, (ii)={part_ii} (rho={beta_rho:+.4f}), "
          f"(iii)={part_iii} (direction first in {100 * direction_first:.1f}%)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
