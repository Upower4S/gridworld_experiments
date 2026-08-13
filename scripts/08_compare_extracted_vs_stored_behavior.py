#!/usr/bin/env python3
"""Compare behavioral commitment phenomenology: faithful extracted labels vs quarantined stored labels.

Stage D0 of the rebase (see RESEARCH_DIRECTIONS.md): every behavior-side headline was
originally computed on stage-03 stored labels that failed the 2026-07-21 integrity audit.
This script puts the recomputed (extracted-label) numbers side by side with the stored-label
numbers on identical prefix coverage, triangulates both against the independent Together
32-trajectory scores, and re-draws the headline commitment curves.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any


ACTIONS = ("UP", "DOWN", "LEFT", "RIGHT")
Q_COLUMNS = tuple(f"q_{action}" for action in ACTIONS)

# Validated categorical palette (see figures/conceptual_belief_space/supervised_activation_projection.png run).
SERIES_COLORS = {"extracted": "#2a78d6", "stored": "#eb6834"}
ACTION_COLORS = {"UP": "#2a78d6", "DOWN": "#eb6834", "LEFT": "#1baf7a", "RIGHT": "#4a3aa7"}
INK = "#333333"
GRID = "#e6e6e6"


def require_dependencies() -> tuple[Any, Any]:
    try:
        import numpy as np
        import pandas as pd
        import pyarrow  # noqa: F401
    except ModuleNotFoundError as exc:
        missing = exc.name or "a required module"
        raise SystemExit(f"Missing dependency: {missing}.") from exc
    return pd, np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extracted_prefix_metrics", type=Path,
                        default=Path("outputs/prefix_commitment_metrics_extracted.parquet"))
    parser.add_argument("--stored_prefix_metrics", type=Path,
                        default=Path("outputs/prefix_commitment_metrics_mlx_full.parquet"))
    parser.add_argument("--extracted_step_summary", type=Path,
                        default=Path("outputs/step_commitment_summary_extracted.parquet"))
    parser.add_argument("--stored_step_summary", type=Path,
                        default=Path("outputs/step_commitment_summary_mlx_full.parquet"))
    parser.add_argument("--together_scores", type=Path,
                        default=Path("outputs/prefix_action_scores.parquet"),
                        help="Independent Together-scored 32-trajectory table for triangulation.")
    parser.add_argument("--report", type=Path, default=Path("outputs/behavior_rebase_comparison.md"))
    parser.add_argument("--metrics_csv", type=Path, default=Path("outputs/behavior_rebase_comparison.csv"))
    parser.add_argument("--figure_dir", type=Path, default=Path("figures/behavior_rebase"))
    parser.add_argument("--num_progress_bins", type=int, default=10)
    parser.add_argument("--flag_threshold", type=float, default=0.02,
                        help="Absolute delta above which a scalar metric is flagged as moved.")
    return parser.parse_args()


def binned_mean(frame: Any, value_column: str, num_bins: int, pd: Any) -> Any:
    bins = pd.cut(frame["progress"].astype(float), bins=num_bins, labels=False, include_lowest=True)
    grouped = frame.groupby(bins)[value_column].mean()
    centers = [(index + 0.5) / num_bins for index in grouped.index]
    return centers, grouped.to_numpy()


def scalar_rows(extracted: Any, stored: Any, extracted_steps: Any, stored_steps: Any,
                pd: Any, np: Any) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    def add(metric: str, ext_value: float | None, sto_value: float | None) -> None:
        rows.append({"metric": metric, "extracted": ext_value, "stored": sto_value})

    prompt_ext = extracted[extracted.prefix_token_index == 0]
    prompt_sto = stored[stored.prefix_token_index == 0]
    strict_ext = extracted[extracted["is_before_first_action_mention"].fillna(True).astype(bool)]
    strict_sto = stored[stored["is_before_first_action_mention"].fillna(True).astype(bool)]

    add("mean commitment_to_final (all prefixes)", extracted.commitment_to_final.mean(), stored.commitment_to_final.mean())
    add("mean entropy_commitment (all prefixes)", extracted.entropy_commitment.mean(), stored.entropy_commitment.mean())
    add("frac top_action == agent_action (all prefixes)", extracted.is_top_action_final.mean(), stored.is_top_action_final.mean())
    add("mean commitment at prompt-only prefixes", prompt_ext.commitment_to_final.mean(), prompt_sto.commitment_to_final.mean())
    add("mean final-prefix commitment (per step)", extracted_steps.final_prefix_commitment_to_final.mean(),
        stored_steps.final_prefix_commitment_to_final.mean())
    add("frac final-prefix top==final (per step)", extracted_steps.final_prefix_is_top_action_final.mean(),
        stored_steps.final_prefix_is_top_action_final.mean())
    add("mean commitment (strict pre-mention prefixes)", strict_ext.commitment_to_final.mean(), strict_sto.commitment_to_final.mean())
    for name, ext_frame, sto_frame in (("strict pre-mention", strict_ext, strict_sto),):
        for edge, low, high in (("first progress decile", 0.0, 0.1), ("last progress decile", 0.9, 1.0)):
            ext_bin = ext_frame[(ext_frame.progress >= low) & (ext_frame.progress <= high)]
            sto_bin = sto_frame[(sto_frame.progress >= low) & (sto_frame.progress <= high)]
            add(f"mean commitment ({name}, {edge})", ext_bin.commitment_to_final.mean(), sto_bin.commitment_to_final.mean())
    add("median first_cross_0_5_progress", extracted_steps.first_cross_0_5_progress.median(),
        stored_steps.first_cross_0_5_progress.median())
    add("median first_cross_0_7_progress", extracted_steps.first_cross_0_7_progress.median(),
        stored_steps.first_cross_0_7_progress.median())
    add("median lock_in_progress", extracted_steps.lock_in_progress.median(), stored_steps.lock_in_progress.median())
    add("frac steps with lock-in", extracted_steps.lock_in_progress.notna().mean(), stored_steps.lock_in_progress.notna().mean())
    add("mean switch_count per step", extracted_steps.switch_count.mean(), stored_steps.switch_count.mean())
    add("mean total_variation per step", extracted_steps.total_variation.mean(), stored_steps.total_variation.mean())
    return rows


def per_action_table(extracted: Any, stored: Any, pd: Any) -> Any:
    records = []
    for regime, frame in (("extracted", extracted), ("stored", stored)):
        prompt_only = frame[frame.prefix_token_index == 0]
        step_tokens = frame.groupby(["trajectory_id", "step_id"], sort=False).agg(
            agent_action=("agent_action", "first"), num_analysis_tokens=("num_analysis_tokens", "first"))
        for action in ACTIONS:
            records.append({
                "regime": regime,
                "agent_action": action,
                "mean_commitment_all": float(frame.loc[frame.agent_action == action, "commitment_to_final"].mean()),
                "mean_commitment_prompt_only": float(
                    prompt_only.loc[prompt_only.agent_action == action, "commitment_to_final"].mean()),
                "mean_cot_tokens_per_step": float(
                    step_tokens.loc[step_tokens.agent_action == action, "num_analysis_tokens"].mean()),
            })
    return pd.DataFrame(records)


def triangulate_against_together(extracted: Any, stored: Any, together_path: Path,
                                 pd: Any, np: Any) -> dict[str, Any] | None:
    if not together_path.exists():
        return None
    together = pd.read_parquet(together_path,
                               columns=["prefix_id", *Q_COLUMNS, "top_action", "failed"])
    if "failed" in together.columns:
        together = together[~together["failed"].fillna(False).astype(bool)]
    together = together.dropna(subset=list(Q_COLUMNS))
    result: dict[str, Any] = {"n_rows": int(len(together))}
    for regime, frame in (("extracted", extracted), ("stored", stored)):
        merged = together.merge(frame[["prefix_id", *Q_COLUMNS, "top_action"]],
                                on="prefix_id", suffixes=("_together", ""), validate="one_to_one")
        q_together = merged[[f"{column}_together" for column in Q_COLUMNS]].to_numpy(dtype=float)
        q_together = q_together / q_together.sum(axis=1, keepdims=True)
        q_regime = merged[list(Q_COLUMNS)].to_numpy(dtype=float)
        result[f"{regime}_top_action_agreement"] = float((merged.top_action == merged.top_action_together).mean())
        result[f"{regime}_mean_max_abs_dq"] = float(np.abs(q_regime - q_together).max(axis=1).mean())
        result[f"{regime}_n_compared"] = int(len(merged))
    return result


def draw_figures(extracted: Any, stored: Any, figure_dir: Path, num_bins: int, pd: Any) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    def style_axis(axis: Any, title: str) -> None:
        axis.set_title(title, color=INK, fontsize=11)
        axis.set_xlabel("progress through CoT", color=INK)
        axis.set_ylabel("mean commitment to final action", color=INK)
        axis.set_ylim(0.0, 1.02)
        axis.grid(True, color=GRID, linewidth=0.8)
        axis.tick_params(colors=INK)
        for spine in ("top", "right"):
            axis.spines[spine].set_visible(False)

    strict_mask_ext = extracted["is_before_first_action_mention"].fillna(True).astype(bool)
    strict_mask_sto = stored["is_before_first_action_mention"].fillna(True).astype(bool)

    figure, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    panels = (("all prefixes", extracted, stored),
              ("strict pre-action-mention prefixes", extracted[strict_mask_ext], stored[strict_mask_sto]))
    for axis, (title, ext_frame, sto_frame) in zip(axes, panels):
        x_sto, y_sto = binned_mean(sto_frame, "commitment_to_final", num_bins, pd)
        x_ext, y_ext = binned_mean(ext_frame, "commitment_to_final", num_bins, pd)
        axis.plot(x_sto, y_sto, color=SERIES_COLORS["stored"], linewidth=2, linestyle="--")
        axis.plot(x_ext, y_ext, color=SERIES_COLORS["extracted"], linewidth=2)
        axis.annotate("extracted (faithful)", (x_ext[-1], y_ext[-1]), textcoords="offset points",
                      xytext=(-4, 8), ha="right", color=INK, fontsize=9)
        axis.annotate("stored (quarantined)", (x_sto[-1], y_sto[-1]), textcoords="offset points",
                      xytext=(-4, -14), ha="right", color=INK, fontsize=9)
        style_axis(axis, title)
    axes[0].legend(handles=[plt.Line2D([], [], color=SERIES_COLORS["extracted"], linewidth=2, label="extracted (faithful)"),
                            plt.Line2D([], [], color=SERIES_COLORS["stored"], linewidth=2, linestyle="--",
                                       label="stored (quarantined)")],
                   loc="lower right", frameon=False, fontsize=9, labelcolor=INK)
    figure.suptitle("Commitment vs progress: label rebase comparison (95 trajectories)", color=INK)
    figure.tight_layout()
    path = figure_dir / "commitment_vs_progress_rebase.png"
    figure.savefig(path, dpi=200)
    plt.close(figure)
    written.append(path)

    figure, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    panels = (("all prefixes (extracted labels)", extracted),
              ("strict pre-action-mention (extracted labels)", extracted[strict_mask_ext]))
    for axis, (title, frame) in zip(axes, panels):
        for action in ACTIONS:
            subset = frame[frame.agent_action == action]
            x_values, y_values = binned_mean(subset, "commitment_to_final", num_bins, pd)
            axis.plot(x_values, y_values, color=ACTION_COLORS[action], linewidth=2)
            # Curves converge to 1.0 at the right edge, so label the well-separated left ends.
            axis.annotate(action, (x_values[0], y_values[0]), textcoords="offset points",
                          xytext=(-6, -3), ha="right", color=INK, fontsize=9)
        style_axis(axis, title)
        axis.set_xlim(-0.16, 1.05)
    axes[0].legend(handles=[plt.Line2D([], [], color=ACTION_COLORS[action], linewidth=2, label=action)
                            for action in ACTIONS],
                   loc="lower right", frameon=False, fontsize=9, labelcolor=INK)
    figure.suptitle("Per-action commitment vs progress on faithful labels", color=INK)
    figure.tight_layout()
    path = figure_dir / "per_action_commitment_extracted.png"
    figure.savefig(path, dpi=200)
    plt.close(figure)
    written.append(path)
    return written


def markdown_table(frame: Any, float_digits: int = 4) -> str:
    headers = list(frame.columns)
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    for _, row in frame.iterrows():
        cells = []
        for column in headers:
            value = row[column]
            if isinstance(value, float):
                cells.append("NA" if value != value else f"{value:.{float_digits}f}")
            else:
                cells.append(str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    pd, np = require_dependencies()
    from scipy.stats import spearmanr

    extracted = pd.read_parquet(args.extracted_prefix_metrics)
    stored = pd.read_parquet(args.stored_prefix_metrics)
    extracted_steps = pd.read_parquet(args.extracted_step_summary)
    stored_steps = pd.read_parquet(args.stored_step_summary)

    common = sorted(set(extracted.prefix_id) & set(stored.prefix_id))
    extracted = extracted[extracted.prefix_id.isin(common)].sort_values("prefix_id").reset_index(drop=True)
    stored = stored[stored.prefix_id.isin(common)].sort_values("prefix_id").reset_index(drop=True)
    stored_steps = stored_steps.merge(extracted_steps[["trajectory_id", "step_id"]],
                                      on=["trajectory_id", "step_id"], validate="one_to_one")

    q_ext = extracted[list(Q_COLUMNS)].to_numpy(dtype=float)
    q_sto = stored[list(Q_COLUMNS)].to_numpy(dtype=float)
    q_sto = q_sto / q_sto.sum(axis=1, keepdims=True)
    max_abs_dq = np.abs(q_ext - q_sto).max(axis=1)
    label_shift = {
        "n_common_prefixes": len(common),
        "top_action_agreement": float((extracted.top_action == stored.top_action).mean()),
        "mean_max_abs_dq": float(max_abs_dq.mean()),
        "median_max_abs_dq": float(np.median(max_abs_dq)),
        "frac_max_abs_dq_gt_1e-4": float((max_abs_dq > 1e-4).mean()),
        "spearman_commitment_to_final": float(spearmanr(extracted.commitment_to_final, stored.commitment_to_final).statistic),
        "spearman_entropy_commitment": float(spearmanr(extracted.entropy_commitment, stored.entropy_commitment).statistic),
    }

    scalars = pd.DataFrame(scalar_rows(extracted, stored, extracted_steps, stored_steps, pd, np))
    scalars["delta"] = scalars["extracted"] - scalars["stored"]
    scalars["moved"] = scalars["delta"].abs() > args.flag_threshold

    actions = per_action_table(extracted, stored, pd)
    triangulation = triangulate_against_together(extracted, stored, args.together_scores, pd, np)
    figure_paths = draw_figures(extracted, stored, args.figure_dir, args.num_progress_bins, pd)

    args.metrics_csv.parent.mkdir(parents=True, exist_ok=True)
    scalars.to_csv(args.metrics_csv, index=False)

    lines = [
        "# Behavior Rebase Comparison: extracted (faithful) vs stored (quarantined) labels",
        "",
        f"- Coverage: {label_shift['n_common_prefixes']} common prefixes, "
        f"{extracted.trajectory_id.nunique()} trajectories, {len(extracted_steps)} steps.",
        "- `extracted` = CoT-window extraction labels (canonical since 2026-07-21 audit); "
        "`stored` = stage-03 MLX-cached labels (quarantined).",
        "",
        "## How far the labels moved",
        "",
        *[f"- {key}: {value:.4f}" if isinstance(value, float) else f"- {key}: {value}"
          for key, value in label_shift.items()],
        "",
        "## Headline scalar metrics",
        "",
        markdown_table(scalars),
        "",
        "## Per-action summary",
        "",
        markdown_table(actions.pivot(index="agent_action", columns="regime",
                                     values=["mean_commitment_all", "mean_commitment_prompt_only",
                                             "mean_cot_tokens_per_step"]).round(4).reset_index()
                       .set_axis(["agent_action",
                                  "mean_commitment_all (ext)", "mean_commitment_all (sto)",
                                  "mean_commitment_prompt (ext)", "mean_commitment_prompt (sto)",
                                  "mean_cot_tokens (ext)", "mean_cot_tokens (sto)"], axis=1)),
        "",
    ]
    if triangulation is not None:
        lines.extend([
            "## Triangulation against independent Together scores (32 trajectories)",
            "",
            "Together scored each prefix independently (no KV-cache reuse), so agreement with it",
            "is an external check on which regime is faithful.",
            "",
            *[f"- {key}: {value:.4f}" if isinstance(value, float) else f"- {key}: {value}"
              for key, value in triangulation.items()],
            "",
        ])
    lines.extend([
        "## Figures",
        "",
        *[f"- `{path}`" for path in figure_paths],
        "",
        "## Reading",
        "",
        "Metrics flagged `moved=True` shifted by more than "
        f"{args.flag_threshold} between label regimes and must not be quoted from stored-label runs. "
        "Metrics that held are stable under the rebase and are safe to cite from either regime, "
        "though the extracted numbers are the citable ones going forward.",
    ])
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Wrote {args.report}")
    print(f"Wrote {args.metrics_csv}")
    for path in figure_paths:
        print(f"Wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
