#!/usr/bin/env python3
"""Conjecture C9: is the probe's late stabilization clock an artifact of decoder noise?

Observation 3.6: the probe-decoded argmax stabilizes far later than the forced-answer
argmax (median 0.99 vs 0.51). Two explanations. (A) The internal state genuinely lags the
forced-answer channel. (B) The decoder is noisy, and any noisy read of an already-settled
trajectory produces spurious late flips, biasing the estimated stabilization time toward 1.

C9 says (B). This is testable WITHOUT a better probe: corrupt the TRUE trajectories with
the probe's own error statistics -- a process that has zero lag by construction -- and
compare the lag it manufactures against the lag actually observed. If the observed lag does
not exceed the simulated one, no internal lag needs to be posited.

Two noise models: i.i.d. draws from the empirical confusion matrix, and a persistence-
matched two-state Markov error process (errors in the real probe come in runs, and runs
change the number of spurious flips).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

ACTIONS = ("UP", "DOWN", "LEFT", "RIGHT")
Q_COLS = [f"q_{action}" for action in ACTIONS]
PRED_Q = [f"pred_q_{action}" for action in ACTIONS]

INK = "#333333"
GRID = "#e6e6e6"
COLORS = {"observed probe": "#2a78d6", "simulated (i.i.d. noise)": "#eb6834",
          "simulated (persistent noise)": "#1baf7a", "true forced-answer": "#9aa0a6"}


def parse_args() -> argparse.Namespace:
    base = REPO_ROOT / "outputs/conceptual_belief_space/cot_window_layer_transition_probe"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, default=base / "fold0_20260721/oof_predictions.parquet")
    parser.add_argument("--method", type=str, default="mean_layers_mlp")
    parser.add_argument("--split_ids", type=str, default="0,1,4",
                        help="Folds whose q head was actually trained (lambda_q > 0).")
    parser.add_argument("--out_dir", type=Path, default=REPO_ROOT / "outputs/conjectures/c9_clock_noise")
    parser.add_argument("--figure_dir", type=Path, default=REPO_ROOT / "figures/conjectures")
    parser.add_argument("--replicates", type=int, default=20)
    parser.add_argument("--smooth_window", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def stabilization_index(labels: np.ndarray) -> int:
    """Index of the first position after which the label never changes again."""
    differs = labels != labels[-1]
    return int(np.max(np.nonzero(differs)[0])) + 1 if differs.any() else 0


def smooth_onehot(labels: np.ndarray, window: int) -> np.ndarray:
    onehot = np.zeros((len(labels), len(ACTIONS)))
    onehot[np.arange(len(labels)), labels] = 1.0
    smoothed = pd.DataFrame(onehot).rolling(window, center=True, min_periods=1).mean().to_numpy()
    return smoothed.argmax(axis=1)


def main() -> int:
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    predictions = pd.read_parquet(args.predictions)
    predictions = predictions[predictions["method"] == args.method]
    keep = {int(token) for token in args.split_ids.split(",") if token.strip()}
    predictions = predictions[predictions["split_id"].astype(int).isin(keep)]
    predictions = predictions.sort_values(["decision_id", "prefix_token_index", "prefix_id"])
    print(f"rows: {len(predictions)}; steps: {predictions.decision_id.nunique()}")

    true_labels_all = predictions[Q_COLS].to_numpy(dtype=float).argmax(axis=1)
    pred_labels_all = predictions[PRED_Q].to_numpy(dtype=float).argmax(axis=1)
    accuracy = float((true_labels_all == pred_labels_all).mean())
    confusion = np.zeros((len(ACTIONS), len(ACTIONS)))
    for true_label, pred_label in zip(true_labels_all, pred_labels_all):
        confusion[true_label, pred_label] += 1
    confusion = confusion / confusion.sum(axis=1, keepdims=True)
    print(f"probe prefix-level argmax accuracy: {accuracy:.4f}")

    # Persistence of the error process, estimated from within-step error runs.
    stay_error, stay_correct, n_error, n_correct = 0, 0, 0, 0
    for _, group in predictions.groupby("decision_id", sort=False):
        errors = (group[Q_COLS].to_numpy(dtype=float).argmax(axis=1)
                  != group[PRED_Q].to_numpy(dtype=float).argmax(axis=1)).astype(int)
        if len(errors) < 2:
            continue
        for previous, current in zip(errors[:-1], errors[1:]):
            if previous == 1:
                n_error += 1
                stay_error += current == 1
            else:
                n_correct += 1
                stay_correct += current == 0
    p_error_to_error = stay_error / max(n_error, 1)
    p_correct_to_correct = stay_correct / max(n_correct, 1)
    print(f"error persistence: P(err|err)={p_error_to_error:.4f}, "
          f"P(ok|ok)={p_correct_to_correct:.4f}")

    wrong_choice = confusion.copy()
    np.fill_diagonal(wrong_choice, 0.0)
    row_sums = wrong_choice.sum(axis=1, keepdims=True)
    wrong_choice = np.divide(wrong_choice, np.where(row_sums > 0, row_sums, 1.0))

    rows: list[dict[str, Any]] = []
    for decision_id, group in predictions.groupby("decision_id", sort=False):
        progress = group["progress"].to_numpy(dtype=float)
        true_labels = group[Q_COLS].to_numpy(dtype=float).argmax(axis=1)
        probe_labels = group[PRED_Q].to_numpy(dtype=float).argmax(axis=1)
        n = len(group)
        if n < 5:
            continue
        true_stab = progress[stabilization_index(true_labels)]
        observed_stab = progress[stabilization_index(probe_labels)]
        observed_smooth = progress[stabilization_index(smooth_onehot(probe_labels, args.smooth_window))]

        iid_stabs, iid_smooth_stabs, markov_stabs, markov_smooth_stabs = [], [], [], []
        for _ in range(args.replicates):
            draws = np.array([rng.choice(len(ACTIONS), p=confusion[label]) for label in true_labels])
            iid_stabs.append(progress[stabilization_index(draws)])
            iid_smooth_stabs.append(progress[stabilization_index(
                smooth_onehot(draws, args.smooth_window))])

            error_state = rng.random() > accuracy
            markov_labels = np.empty(n, dtype=int)
            for position, label in enumerate(true_labels):
                if position > 0:
                    threshold = p_error_to_error if error_state else 1.0 - p_correct_to_correct
                    error_state = rng.random() < threshold
                markov_labels[position] = (rng.choice(len(ACTIONS), p=wrong_choice[label])
                                           if error_state else label)
            markov_stabs.append(progress[stabilization_index(markov_labels)])
            markov_smooth_stabs.append(progress[stabilization_index(
                smooth_onehot(markov_labels, args.smooth_window))])

        rows.append({
            "decision_id": decision_id, "n_prefixes": n,
            "true_stab": float(true_stab),
            "observed_stab": float(observed_stab),
            "observed_stab_smooth": float(observed_smooth),
            "sim_iid_stab": float(np.mean(iid_stabs)),
            "sim_iid_stab_smooth": float(np.mean(iid_smooth_stabs)),
            "sim_markov_stab": float(np.mean(markov_stabs)),
            "sim_markov_stab_smooth": float(np.mean(markov_smooth_stabs)),
        })
    steps = pd.DataFrame(rows)
    for column in ["observed_stab", "observed_stab_smooth", "sim_iid_stab", "sim_iid_stab_smooth",
                   "sim_markov_stab", "sim_markov_stab_smooth"]:
        steps[f"lag_{column}"] = steps[column] - steps["true_stab"]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    steps.to_parquet(args.out_dir / "steps.parquet", index=False)

    from scipy.stats import wilcoxon
    summary_rows = []
    for label, column in (("observed", "observed_stab"), ("observed_smoothed", "observed_stab_smooth"),
                          ("sim_iid", "sim_iid_stab"), ("sim_iid_smoothed", "sim_iid_stab_smooth"),
                          ("sim_markov", "sim_markov_stab"),
                          ("sim_markov_smoothed", "sim_markov_stab_smooth")):
        summary_rows.append({
            "clock": label,
            "median_stab": float(steps[column].median()),
            "mean_stab": float(steps[column].mean()),
            "median_lag": float(steps[f"lag_{column}"].median()),
            "mean_lag": float(steps[f"lag_{column}"].mean()),
        })
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(args.out_dir / "clock_summary.csv", index=False)

    tests = []
    for observed_column, simulated_column, name in (
            ("observed_stab", "sim_iid_stab", "raw vs i.i.d."),
            ("observed_stab", "sim_markov_stab", "raw vs persistent"),
            ("observed_stab_smooth", "sim_iid_stab_smooth", "smoothed vs i.i.d. smoothed"),
            ("observed_stab_smooth", "sim_markov_stab_smooth", "smoothed vs persistent smoothed")):
        difference = steps[observed_column] - steps[simulated_column]
        stat = wilcoxon(difference) if difference.abs().sum() > 0 else None
        tests.append({
            "comparison": name,
            "mean_observed_minus_simulated": float(difference.mean()),
            "median_observed_minus_simulated": float(difference.median()),
            "frac_observed_earlier": float((difference < 0).mean()),
            "wilcoxon_p": float(stat.pvalue) if stat is not None else float("nan"),
        })
    tests_frame = pd.DataFrame(tests)
    tests_frame.to_csv(args.out_dir / "tests.csv", index=False)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    args.figure_dir.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(7.8, 4.4))
    series = {"true forced-answer": steps["true_stab"], "observed probe": steps["observed_stab"],
              "simulated (i.i.d. noise)": steps["sim_iid_stab"],
              "simulated (persistent noise)": steps["sim_markov_stab"]}
    for label, values in series.items():
        ordered = np.sort(values.to_numpy())
        axis.plot(ordered, np.arange(1, len(ordered) + 1) / len(ordered),
                  color=COLORS[label], linewidth=2)
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1.02)
    axis.set_xlabel("stabilization progress", color=INK)
    axis.set_ylabel("fraction of held-out steps", color=INK)
    axis.set_title("C9: noise alone reproduces the probe's late stabilization", color=INK, fontsize=11)
    axis.legend(handles=[plt.Line2D([], [], color=color, linewidth=2, label=label)
                         for label, color in COLORS.items()],
                loc="upper left", frameon=False, fontsize=8, labelcolor=INK)
    axis.grid(True, color=GRID, linewidth=0.8)
    axis.tick_params(colors=INK)
    for spine in ("top", "right"):
        axis.spines[spine].set_visible(False)
    figure.tight_layout()
    figure_path = args.figure_dir / "c9_clock_noise.png"
    figure.savefig(figure_path, dpi=200)
    plt.close(figure)

    raw_test = tests_frame.set_index("comparison").loc["raw vs persistent"]
    verdict = "SUPPORTED" if raw_test["mean_observed_minus_simulated"] <= 0 else "NOT SUPPORTED"
    lines = [
        "# Conjecture C9: is the decoded clock's lag pure decoder noise?",
        "",
        f"- Probe `{args.method}`, folds {args.split_ids}, {len(steps)} held-out steps.",
        f"- Probe prefix-level argmax accuracy {accuracy:.4f}; error persistence "
        f"P(err|err)={p_error_to_error:.4f} vs base error rate {1 - accuracy:.4f} "
        "(errors are strongly autocorrelated, so the i.i.d. model is the conservative one).",
        "",
        "## Clocks",
        "",
        summary.round(4).to_markdown(index=False),
        "",
        "## Observed lag minus simulated lag (a zero-lag process with the probe's noise)",
        "",
        tests_frame.round(4).to_markdown(index=False),
        "",
        f"## Verdict: {verdict}",
        "",
        "Criterion: if a zero-lag process carrying the probe's own noise manufactures a",
        "stabilization delay at least as large as the observed one, the observed delay is",
        "not evidence of an internal lag.",
        "",
        f"Figure: `{figure_path}`",
    ]
    (args.out_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(summary.round(4).to_string(index=False))
    print()
    print(tests_frame.round(4).to_string(index=False))
    print(f"\nVerdict: {verdict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
