#!/usr/bin/env python3
"""Plot behavior-side commitment formation from a behavior manifold parquet."""

from __future__ import annotations

import argparse
import math
import os
import tempfile
from pathlib import Path
from typing import Any


os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "xdg-cache"))
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
Path(os.environ["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)

import matplotlib  # noqa: E402

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


ACTIONS = ("UP", "DOWN", "LEFT", "RIGHT")
Q_COLUMNS = tuple(f"q_{action}" for action in ACTIONS)
STEP_KEYS = ("trajectory_id", "source_file", "step_id")
ACTION_COLORS = {
    "UP": "#1f77b4",
    "DOWN": "#d62728",
    "LEFT": "#2ca02c",
    "RIGHT": "#9467bd",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("outputs/behavior_manifold.parquet"),
        help="Behavior manifold parquet input.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("figures/behavior"),
        help="Directory for behavior plots and report.",
    )
    parser.add_argument("--num_progress_bins", type=int, default=12)
    parser.add_argument("--max_example_steps", type=int, default=6)
    parser.add_argument("--max_switching_steps", type=int, default=4)
    return parser.parse_args()


def validate_required_columns(df: pd.DataFrame) -> None:
    required = [
        "beh_pca1",
        "beh_pca2",
        "agent_action",
        "progress",
        "prefix_id",
        "prefix_token_index",
        *STEP_KEYS,
        *Q_COLUMNS,
    ]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise SystemExit(f"Input is missing expected columns: {missing}")


def sort_prefixes(df: pd.DataFrame) -> pd.DataFrame:
    return df.sort_values([*STEP_KEYS, "prefix_token_index", "prefix_id"]).reset_index(drop=True)


def enrich_columns(df: pd.DataFrame) -> pd.DataFrame:
    work = sort_prefixes(df.copy())
    q_values = work[list(Q_COLUMNS)].astype(float).to_numpy()
    top_indices = np.argmax(q_values, axis=1)
    sorted_probs = np.sort(q_values, axis=1)
    work["top_action"] = [ACTIONS[index] for index in top_indices]
    work["top_prob"] = sorted_probs[:, -1]
    work["second_prob"] = sorted_probs[:, -2]
    work["margin_commitment"] = work["top_prob"] - work["second_prob"]

    if "commitment_to_final" not in work.columns:
        action_to_index = {action: index for index, action in enumerate(ACTIONS)}
        work["commitment_to_final"] = [
            q_values[row_index, action_to_index.get(str(action), 0)]
            if str(action) in action_to_index
            else np.nan
            for row_index, action in enumerate(work["agent_action"])
        ]
    if "entropy_commitment" not in work.columns:
        clipped = np.clip(q_values, 1e-300, 1.0)
        entropy = -(q_values * np.log(clipped)).sum(axis=1)
        work["entropy_commitment"] = 1.0 - entropy / math.log(len(ACTIONS))

    work["is_top_action_final"] = work["top_action"] == work["agent_action"].astype(str)
    grouped = work.groupby(list(STEP_KEYS), sort=False, dropna=False)
    prev_top = grouped["top_action"].shift()
    work["top_action_changed"] = prev_top.notna() & (work["top_action"] != prev_top)
    return work


def save_figure(fig: Any, output_dir: Path, filename: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_dir / filename, dpi=180, bbox_inches="tight")
    plt.close(fig)


def categorical_scatter(
    df: pd.DataFrame,
    *,
    color_column: str,
    title: str,
    output_dir: Path,
    filename: str,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 5.5))
    for value in sorted(df[color_column].dropna().astype(str).unique()):
        subset = df[df[color_column].astype(str) == value]
        ax.scatter(
            subset["beh_pca1"],
            subset["beh_pca2"],
            s=18,
            alpha=0.72,
            label=value,
            color=ACTION_COLORS.get(value),
            linewidths=0,
        )
    ax.set_title(title)
    ax.set_xlabel("beh_pca1")
    ax.set_ylabel("beh_pca2")
    ax.legend(title=color_column, frameon=False, markerscale=1.4)
    ax.grid(True, alpha=0.2)
    save_figure(fig, output_dir, filename)


def continuous_scatter(
    df: pd.DataFrame,
    *,
    color_column: str,
    title: str,
    output_dir: Path,
    filename: str,
    cmap: str = "viridis",
) -> None:
    fig, ax = plt.subplots(figsize=(7, 5.5))
    scatter = ax.scatter(
        df["beh_pca1"],
        df["beh_pca2"],
        c=df[color_column].astype(float),
        s=18,
        alpha=0.76,
        cmap=cmap,
        linewidths=0,
    )
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label(color_column)
    ax.set_title(title)
    ax.set_xlabel("beh_pca1")
    ax.set_ylabel("beh_pca2")
    ax.grid(True, alpha=0.2)
    save_figure(fig, output_dir, filename)


def placeholder_figure(output_dir: Path, filename: str, title: str, message: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.text(0.5, 0.5, message, ha="center", va="center", wrap=True)
    ax.set_title(title)
    ax.set_axis_off()
    save_figure(fig, output_dir, filename)


def before_action_subset(df: pd.DataFrame) -> pd.DataFrame | None:
    if "is_before_first_action_mention" not in df.columns:
        return None
    return df[df["is_before_first_action_mention"].fillna(True).astype(bool)].copy()


def progress_bin_summary(df: pd.DataFrame, *, value_column: str, num_bins: int) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["bin_midpoint", "mean", "count"])
    work = df.copy()
    bins = np.linspace(0.0, 1.0, num_bins + 1)
    work["progress_bin"] = pd.cut(
        work["progress"].astype(float).clip(0.0, 1.0),
        bins=bins,
        include_lowest=True,
    )
    summary = (
        work.groupby("progress_bin", observed=False)
        .agg(mean=(value_column, "mean"), count=(value_column, "size"))
        .reset_index()
    )
    summary["bin_midpoint"] = summary["progress_bin"].apply(lambda interval: interval.mid)
    summary = summary[summary["count"] > 0]
    return summary


def plot_progress_curve(
    df: pd.DataFrame,
    *,
    value_column: str,
    ylabel: str,
    title: str,
    output_dir: Path,
    filename: str,
    num_bins: int,
) -> None:
    summary = progress_bin_summary(df, value_column=value_column, num_bins=num_bins)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    if summary.empty:
        ax.text(0.5, 0.5, "No rows available for this split", ha="center", va="center")
    else:
        ax.plot(summary["bin_midpoint"], summary["mean"], marker="o", linewidth=2)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
    ax.set_title(title)
    ax.set_xlabel("reasoning progress")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    save_figure(fig, output_dir, filename)


def step_groups(df: pd.DataFrame) -> list[tuple[tuple[Any, Any, Any], pd.DataFrame]]:
    groups = []
    for key, group in df.groupby(list(STEP_KEYS), sort=False, dropna=False):
        groups.append((key, group.sort_values(["prefix_token_index", "prefix_id"])))
    return groups


def final_prefixes(df: pd.DataFrame) -> pd.DataFrame:
    rows = [group.iloc[-1] for _, group in step_groups(df) if not group.empty]
    if not rows:
        return df.iloc[0:0].copy()
    return pd.DataFrame(rows).reset_index(drop=True)


def step_label(key: tuple[Any, Any, Any]) -> str:
    trajectory_id, _source_file, step_id = key
    return f"{trajectory_id} / step {step_id}"


def step_stats(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for key, group in step_groups(df):
        final = group.iloc[-1]
        lock_progress = compute_lock_in_progress(group)
        rows.append(
            {
                "trajectory_id": key[0],
                "source_file": key[1],
                "step_id": int(key[2]),
                "agent_action": final["agent_action"],
                "final_commitment_to_final": float(final["commitment_to_final"]),
                "final_top_action": final["top_action"],
                "final_is_top_action_final": bool(final["is_top_action_final"]),
                "switch_count": int(group["top_action_changed"].sum()),
                "lock_in_progress": lock_progress,
                "max_progress": float(group["progress"].max()),
                "num_prefixes": int(len(group)),
            }
        )
    return pd.DataFrame(rows)


def compute_lock_in_progress(group: pd.DataFrame) -> float | None:
    condition = (
        (group["top_action"] == group["agent_action"].astype(str))
        & (group["margin_commitment"] >= 0.2)
    ).tolist()
    progresses = group["progress"].astype(float).tolist()
    for index in range(max(0, len(condition) - 2)):
        if condition[index] and condition[index + 1] and condition[index + 2]:
            return float(progresses[index])
    return None


def plot_example_manifold_trajectories(
    df: pd.DataFrame,
    *,
    output_dir: Path,
    max_steps: int,
) -> None:
    stats = step_stats(df)
    if stats.empty:
        placeholder_figure(
            output_dir,
            "example_manifold_trajectories.png",
            "Example manifold trajectories",
            "No step groups available.",
        )
        return
    selected_keys = stats.sort_values(
        ["switch_count", "final_commitment_to_final"],
        ascending=[False, False],
    ).head(max_steps)[list(STEP_KEYS)]
    selected = {
        tuple(row)
        for row in selected_keys.to_numpy()
    }

    fig, ax = plt.subplots(figsize=(7.5, 6))
    ax.scatter(df["beh_pca1"], df["beh_pca2"], s=8, color="#d0d0d0", alpha=0.35, linewidths=0)
    cmap = plt.get_cmap("tab10")
    for idx, (key, group) in enumerate(step_groups(df)):
        if key not in selected:
            continue
        color = cmap(idx % 10)
        ax.plot(group["beh_pca1"], group["beh_pca2"], color=color, linewidth=1.8, alpha=0.9)
        ax.scatter(group.iloc[0]["beh_pca1"], group.iloc[0]["beh_pca2"], color=color, marker="o", s=46)
        ax.scatter(group.iloc[-1]["beh_pca1"], group.iloc[-1]["beh_pca2"], color=color, marker="X", s=70)
    ax.set_title("Example manifold trajectories")
    ax.set_xlabel("beh_pca1")
    ax.set_ylabel("beh_pca2")
    ax.grid(True, alpha=0.2)
    ax.text(
        0.01,
        0.01,
        "circle=start, X=end",
        transform=ax.transAxes,
        fontsize=9,
        va="bottom",
    )
    save_figure(fig, output_dir, "example_manifold_trajectories.png")


def plot_switching_examples(
    df: pd.DataFrame,
    *,
    output_dir: Path,
    max_steps: int,
) -> None:
    switching = step_stats(df)
    switching = switching[switching["switch_count"] > 0].sort_values("switch_count", ascending=False)
    if switching.empty:
        placeholder_figure(
            output_dir,
            "switching_examples.png",
            "Switching examples",
            "No steps with top_action switches were found.",
        )
        return

    selected = {
        tuple(row)
        for row in switching.head(max_steps)[list(STEP_KEYS)].to_numpy()
    }
    selected_groups = [(key, group) for key, group in step_groups(df) if key in selected]
    n = len(selected_groups)
    fig, axes = plt.subplots(n, 1, figsize=(8, max(3.0, 2.7 * n)), sharex=False)
    if n == 1:
        axes = [axes]
    for ax, (key, group) in zip(axes, selected_groups, strict=True):
        for action in ACTIONS:
            ax.plot(
                group["prefix_token_index"],
                group[f"q_{action}"],
                label=action,
                color=ACTION_COLORS[action],
                linewidth=1.7,
            )
        ax.set_title(step_label(key), fontsize=10)
        ax.set_ylabel("q(action)")
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.22)
    axes[-1].set_xlabel("prefix_token_index")
    axes[0].legend(frameon=False, ncol=4, loc="upper right")
    save_figure(fig, output_dir, "switching_examples.png")


def plot_final_prefix_sanity(df: pd.DataFrame, output_dir: Path) -> float:
    finals = final_prefixes(df)
    fraction = float(finals["is_top_action_final"].mean()) if not finals.empty else float("nan")
    fig, ax = plt.subplots(figsize=(7, 4.5))
    if finals.empty:
        ax.text(0.5, 0.5, "No final prefixes available", ha="center", va="center")
    else:
        ax.hist(finals["commitment_to_final"], bins=np.linspace(0, 1, 21), color="#4c78a8", alpha=0.85)
        ax.axvline(finals["commitment_to_final"].mean(), color="#222222", linestyle="--", linewidth=1.5)
        ax.text(
            0.02,
            0.95,
            f"fraction final top == agent: {fraction:.3f}",
            transform=ax.transAxes,
            va="top",
        )
    ax.set_title("Final-prefix commitment sanity")
    ax.set_xlabel("final-prefix commitment_to_final")
    ax.set_ylabel("step count")
    ax.grid(True, alpha=0.22)
    save_figure(fig, output_dir, "final_prefix_sanity.png")
    return fraction


def plot_all(df: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    output_dir = args.output_dir
    categorical_scatter(
        df,
        color_column="top_action",
        title="PCA by top action",
        output_dir=output_dir,
        filename="pca_by_top_action.png",
    )
    categorical_scatter(
        df,
        color_column="agent_action",
        title="PCA by agent action",
        output_dir=output_dir,
        filename="pca_by_agent_action.png",
    )
    continuous_scatter(
        df,
        color_column="commitment_to_final",
        title="PCA by commitment to final action",
        output_dir=output_dir,
        filename="pca_by_commitment_to_final.png",
    )
    continuous_scatter(
        df,
        color_column="entropy_commitment",
        title="PCA by entropy commitment",
        output_dir=output_dir,
        filename="pca_by_entropy_commitment.png",
        cmap="plasma",
    )
    continuous_scatter(
        df,
        color_column="progress",
        title="PCA by reasoning progress",
        output_dir=output_dir,
        filename="pca_by_progress.png",
        cmap="cividis",
    )

    before = before_action_subset(df)
    if before is None:
        placeholder_figure(
            output_dir,
            "pca_before_first_action_mention.png",
            "PCA before first action mention",
            "Column is_before_first_action_mention is unavailable.",
        )
    else:
        categorical_scatter(
            before,
            color_column="top_action",
            title="PCA before first explicit action mention",
            output_dir=output_dir,
            filename="pca_before_first_action_mention.png",
        )

    plot_progress_curve(
        df,
        value_column="commitment_to_final",
        ylabel="mean commitment_to_final",
        title="Commitment to final action over progress",
        output_dir=output_dir,
        filename="commitment_to_final_over_progress.png",
        num_bins=args.num_progress_bins,
    )
    if before is None:
        placeholder_figure(
            output_dir,
            "commitment_to_final_over_progress_before_action_mention.png",
            "Commitment before first action mention",
            "Column is_before_first_action_mention is unavailable.",
        )
    else:
        plot_progress_curve(
            before,
            value_column="commitment_to_final",
            ylabel="mean commitment_to_final",
            title="Commitment over progress before first action mention",
            output_dir=output_dir,
            filename="commitment_to_final_over_progress_before_action_mention.png",
            num_bins=args.num_progress_bins,
        )
    plot_progress_curve(
        df,
        value_column="entropy_commitment",
        ylabel="mean entropy_commitment",
        title="Entropy commitment over progress",
        output_dir=output_dir,
        filename="entropy_commitment_over_progress.png",
        num_bins=args.num_progress_bins,
    )
    plot_example_manifold_trajectories(df, output_dir=output_dir, max_steps=args.max_example_steps)
    plot_switching_examples(df, output_dir=output_dir, max_steps=args.max_switching_steps)
    plot_final_prefix_sanity(df, output_dir)
    return step_stats(df)


def slope_by_progress(df: pd.DataFrame, value_column: str) -> float | None:
    subset = df[["progress", value_column]].dropna()
    if len(subset) < 2 or subset["progress"].nunique() < 2:
        return None
    return float(np.polyfit(subset["progress"].astype(float), subset[value_column].astype(float), 1)[0])


def early_late_difference(df: pd.DataFrame, value_column: str) -> float | None:
    if df.empty:
        return None
    early = df[df["progress"] <= 0.25][value_column].mean()
    late = df[df["progress"] >= 0.75][value_column].mean()
    if pd.isna(early) or pd.isna(late):
        return None
    return float(late - early)


def fmt_number(value: Any, digits: int = 4) -> str:
    if value is None:
        return "NA"
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return str(value)
    if math.isnan(numeric):
        return "NA"
    return f"{numeric:.{digits}f}"


def describe_trend(df: pd.DataFrame, value_column: str) -> str:
    slope = slope_by_progress(df, value_column)
    diff = early_late_difference(df, value_column)
    if slope is None:
        return "insufficient progress variation to estimate a trend"
    direction = "increases" if slope > 0 else "decreases" if slope < 0 else "is flat"
    return f"{direction} with linear slope {fmt_number(slope)}; late-minus-early difference {fmt_number(diff)}"


def format_step_examples(rows: pd.DataFrame, value_column: str | None = None, limit: int = 5) -> list[str]:
    if rows.empty:
        return ["- none found"]
    lines = []
    for row in rows.head(limit).to_dict("records"):
        extra = ""
        if value_column is not None:
            extra = f", {value_column}={fmt_number(row[value_column])}"
        lines.append(
            f"- `{row['trajectory_id']}` step `{int(row['step_id'])}`: "
            f"agent={row['agent_action']}, final_top={row['final_top_action']}, "
            f"switch_count={int(row['switch_count'])}, "
            f"final_commitment={fmt_number(row['final_commitment_to_final'])}{extra}"
        )
    return lines


def write_report(df: pd.DataFrame, stats: pd.DataFrame, output_dir: Path, input_path: Path) -> None:
    finals = final_prefixes(df)
    final_fraction = float(finals["is_top_action_final"].mean()) if not finals.empty else float("nan")
    before = before_action_subset(df)

    early_lock = stats[stats["lock_in_progress"].notna()].sort_values("lock_in_progress")
    late_or_no_lock = stats[
        stats["lock_in_progress"].isna() | (stats["lock_in_progress"] >= 0.75)
    ].sort_values(["lock_in_progress", "final_commitment_to_final"], na_position="last")
    switching = stats[stats["switch_count"] > 0].sort_values("switch_count", ascending=False)

    all_commitment_slope = slope_by_progress(df, "commitment_to_final")
    before_commitment_slope = slope_by_progress(before, "commitment_to_final") if before is not None else None
    all_entropy_slope = slope_by_progress(df, "entropy_commitment")
    before_entropy_slope = slope_by_progress(before, "entropy_commitment") if before is not None else None

    lines = [
        "# Behavior Commitment Plot Report",
        "",
        f"- Input file: `{input_path}`",
        f"- Prefix points: {len(df)}",
        f"- Steps: {len(stats)}",
        f"- Trajectory files: {df['source_file'].nunique() if 'source_file' in df.columns else 'NA'}",
        f"- Final-prefix top_action equals agent_action fraction: {fmt_number(final_fraction)}",
        "",
        "## Interpretation",
        "",
        f"- Final-prefix top_action usually matches agent_action: {'yes' if final_fraction >= 0.5 else 'no'} "
        f"({fmt_number(final_fraction)}).",
        f"- commitment_to_final over reasoning progress: {describe_trend(df, 'commitment_to_final')}.",
        f"- entropy_commitment over reasoning progress: {describe_trend(df, 'entropy_commitment')}.",
        "",
        "## All Prefixes vs Before Action Mention",
        "",
        f"- All-prefix commitment slope: {fmt_number(all_commitment_slope)}",
        f"- Before-action-mention commitment slope: {fmt_number(before_commitment_slope)}",
        f"- All-prefix entropy-commitment slope: {fmt_number(all_entropy_slope)}",
        f"- Before-action-mention entropy-commitment slope: {fmt_number(before_entropy_slope)}",
        f"- Before-action-mention prefix points: {len(before) if before is not None else 'unavailable'}",
        "",
        "## Early Lock-In Steps",
        "",
        *format_step_examples(early_lock, value_column="lock_in_progress"),
        "",
        "## Late or No Lock-In Steps",
        "",
        *format_step_examples(late_or_no_lock, value_column="lock_in_progress"),
        "",
        "## Action Switching Steps",
        "",
        *format_step_examples(switching),
        "",
        "## Generated Figures",
        "",
        "- `pca_by_top_action.png`",
        "- `pca_by_agent_action.png`",
        "- `pca_by_commitment_to_final.png`",
        "- `pca_by_entropy_commitment.png`",
        "- `pca_by_progress.png`",
        "- `pca_before_first_action_mention.png`",
        "- `commitment_to_final_over_progress.png`",
        "- `commitment_to_final_over_progress_before_action_mention.png`",
        "- `entropy_commitment_over_progress.png`",
        "- `example_manifold_trajectories.png`",
        "- `switching_examples.png`",
        "- `final_prefix_sanity.png`",
    ]
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "plot_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    if args.num_progress_bins < 2:
        raise SystemExit("--num_progress_bins must be >= 2")
    if not args.input.exists():
        raise SystemExit(f"Input file not found: {args.input}")

    raw = pd.read_parquet(args.input)
    validate_required_columns(raw)
    df = enrich_columns(raw)
    stats = plot_all(df, args)
    write_report(df, stats, args.output_dir, args.input)
    print(f"Wrote behavior figures and report to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
