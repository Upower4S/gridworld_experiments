"""Matplotlib-only plotting helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from .config import ACTIONS
from .io import ensure_dir

DEFAULT_ACTION_COLORS = {
    "UP": "#4C78A8",
    "DOWN": "#F58518",
    "LEFT": "#54A24B",
    "RIGHT": "#B279A2",
}


def savefig(path: str | Path, dpi: int = 200, *, fig=None, close: bool = True) -> Path:
    """Save the current or provided matplotlib figure."""
    import matplotlib.pyplot as plt

    path = Path(path)
    ensure_dir(path.parent)
    if fig is None:
        fig = plt.gcf()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    if close:
        plt.close(fig)
    return path


def action_color_map(colors: Mapping[str, str] | None = None) -> dict[str, str]:
    """Return optional action colors with stable defaults."""
    out = dict(DEFAULT_ACTION_COLORS)
    if colors:
        out.update(colors)
    return {action: out.get(action, "#666666") for action in ACTIONS}


def scatter(
    x: Sequence[float],
    y: Sequence[float],
    *,
    color=None,
    labels: Sequence[str] | None = None,
    ax=None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    title: str | None = None,
    cmap: str = "viridis",
    s: float = 12.0,
    alpha: float = 0.8,
):
    """Basic scatter plot supporting numeric colors or action labels."""
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 5))
    x_arr = np.asarray(x)
    y_arr = np.asarray(y)

    if labels is not None:
        color_map = action_color_map()
        labels_arr = np.asarray(labels)
        for label in ACTIONS:
            mask = labels_arr == label
            if mask.any():
                ax.scatter(
                    x_arr[mask],
                    y_arr[mask],
                    s=s,
                    alpha=alpha,
                    color=color_map[label],
                    label=label,
                    edgecolors="none",
                )
        ax.legend(frameon=False, fontsize=8)
    else:
        points = ax.scatter(x_arr, y_arr, c=color, cmap=cmap, s=s, alpha=alpha, edgecolors="none")
        if color is not None and np.asarray(color).dtype.kind in {"f", "i", "u"}:
            plt.colorbar(points, ax=ax)

    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    return ax


def heatmap(
    matrix: Sequence[Sequence[float]],
    *,
    row_labels: Sequence[str] | None = None,
    col_labels: Sequence[str] | None = None,
    ax=None,
    title: str | None = None,
    cmap: str = "viridis",
    annotate: bool = True,
    fmt: str = ".2f",
):
    """Basic heatmap helper using matplotlib only."""
    import matplotlib.pyplot as plt

    data = np.asarray(matrix, dtype=float)
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(data, cmap=cmap, aspect="auto")
    plt.colorbar(im, ax=ax)

    if row_labels is not None:
        ax.set_yticks(np.arange(len(row_labels)))
        ax.set_yticklabels(row_labels)
    if col_labels is not None:
        ax.set_xticks(np.arange(len(col_labels)))
        ax.set_xticklabels(col_labels, rotation=45, ha="right")
    if title:
        ax.set_title(title)
    if annotate:
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                ax.text(j, i, format(data[i, j], fmt), ha="center", va="center", fontsize=8)
    return ax


def behavior_umap_scatter(
    df,
    *,
    color_col: str,
    ax=None,
    title: str | None = None,
    numeric_cmap: str = "viridis",
    s: float = 10.0,
    alpha: float = 0.75,
):
    """Plot behavior UMAP coordinates with an action or numeric color."""
    if color_col not in df.columns:
        raise ValueError(f"missing color column: {color_col}")
    if color_col in {"agent_action", "top_action", "action"}:
        labels = df[color_col].astype(str).to_numpy()
        return scatter(
            df["beh_umap1"],
            df["beh_umap2"],
            labels=labels,
            ax=ax,
            xlabel="behavior UMAP dimension 1",
            ylabel="behavior UMAP dimension 2",
            title=title or f"Behavior UMAP by {color_col}",
            s=s,
            alpha=alpha,
        )
    return scatter(
        df["beh_umap1"],
        df["beh_umap2"],
        color=df[color_col].to_numpy(),
        ax=ax,
        xlabel="behavior UMAP dimension 1",
        ylabel="behavior UMAP dimension 2",
        title=title or f"Behavior UMAP by {color_col}",
        cmap=numeric_cmap,
        s=s,
        alpha=alpha,
    )


def plot_behavior_trajectories(
    df,
    *,
    decision_ids: Sequence[str],
    ax=None,
    title: str | None = None,
):
    """Connect selected decision trajectories in behavior UMAP space."""
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(7, 5))
    color_cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", ["#4C78A8"])
    for i, decision_id in enumerate(decision_ids):
        part = df.loc[df["decision_id"].astype(str) == str(decision_id)].sort_values("prefix_token_index")
        if part.empty:
            continue
        color = color_cycle[i % len(color_cycle)]
        ax.plot(part["beh_umap1"], part["beh_umap2"], "-", lw=1.5, color=color, alpha=0.85)
        ax.scatter(part["beh_umap1"].iloc[0], part["beh_umap2"].iloc[0], marker="o", s=42, color=color)
        ax.scatter(part["beh_umap1"].iloc[-1], part["beh_umap2"].iloc[-1], marker="s", s=48, color=color)
    ax.set_xlabel("behavior UMAP dimension 1")
    ax.set_ylabel("behavior UMAP dimension 2")
    ax.set_title(title or "Selected behavior belief trajectories")
    return ax


def plot_activation_trajectories(
    df,
    *,
    decision_ids: Sequence[str],
    ax=None,
    title: str | None = None,
):
    """Connect selected decision trajectories in activation UMAP space."""
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(7, 5))
    color_cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", ["#4C78A8"])
    for i, decision_id in enumerate(decision_ids):
        part = df.loc[df["decision_id"].astype(str) == str(decision_id)].sort_values("prefix_token_index")
        if part.empty:
            continue
        color = color_cycle[i % len(color_cycle)]
        ax.plot(part["act_umap1"], part["act_umap2"], "-", lw=1.5, color=color, alpha=0.85)
        ax.scatter(part["act_umap1"].iloc[0], part["act_umap2"].iloc[0], marker="o", s=42, color=color)
        ax.scatter(part["act_umap1"].iloc[-1], part["act_umap2"].iloc[-1], marker="s", s=48, color=color)
    ax.set_xlabel("activation UMAP dimension 1")
    ax.set_ylabel("activation UMAP dimension 2")
    ax.set_title(title or "Selected activation trajectories")
    return ax


def activation_umap_scatter(
    df,
    *,
    color_col: str,
    ax=None,
    title: str | None = None,
    numeric_cmap: str = "viridis",
    s: float = 10.0,
    alpha: float = 0.75,
):
    """Plot activation UMAP coordinates with an action or numeric color."""
    if color_col not in df.columns:
        raise ValueError(f"missing color column: {color_col}")
    if color_col in {"agent_action", "top_action", "action"}:
        return scatter(
            df["act_umap1"],
            df["act_umap2"],
            labels=df[color_col].astype(str).to_numpy(),
            ax=ax,
            xlabel="activation UMAP dimension 1",
            ylabel="activation UMAP dimension 2",
            title=title or f"Activation UMAP by {color_col}",
            s=s,
            alpha=alpha,
        )
    return scatter(
        df["act_umap1"],
        df["act_umap2"],
        color=df[color_col].to_numpy(),
        ax=ax,
        xlabel="activation UMAP dimension 1",
        ylabel="activation UMAP dimension 2",
        title=title or f"Activation UMAP by {color_col}",
        cmap=numeric_cmap,
        s=s,
        alpha=alpha,
    )


def distance_heatmap(
    matrix,
    *,
    ax=None,
    title: str | None = None,
    cmap: str = "magma",
):
    """Action distance-matrix heatmap."""
    return heatmap(
        matrix,
        row_labels=ACTIONS,
        col_labels=ACTIONS,
        ax=ax,
        title=title,
        cmap=cmap,
        annotate=True,
        fmt=".2f",
    )


def line_metric_by_layer(
    df,
    *,
    metric_col: str,
    title: str,
    ylabel: str,
    ax=None,
):
    """Plot a metric by layer with one line per activation representation."""
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4.5))
    grouped = (
        df.groupby(["activation_representation", "layer"])[metric_col]
        .mean()
        .reset_index()
        .sort_values("layer")
    )
    for representation, part in grouped.groupby("activation_representation"):
        ax.plot(part["layer"], part[metric_col], marker="o", label=str(representation))
    ax.axhline(0, color="#888888", lw=0.8)
    ax.set_xlabel("activation layer")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(frameon=False)
    return ax


def simple_bar(
    labels,
    values,
    *,
    ax=None,
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    color: str = "#4C78A8",
    rotation: int = 30,
):
    """Small categorical bar helper."""
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar([str(label) for label in labels], values, color=color)
    if title:
        ax.set_title(title)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=rotation)
    return ax


def simple_lines(
    df,
    *,
    x_col: str,
    y_col: str,
    group_col: str | None = None,
    ax=None,
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    marker: str = "o",
):
    """Line plot helper for already-aggregated tables."""
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4.5))
    if group_col and group_col in df.columns:
        for group, part in df.groupby(group_col):
            part = part.sort_values(x_col)
            ax.plot(part[x_col].astype(str), part[y_col], marker=marker, label=str(group))
        ax.legend(frameon=False, fontsize=8)
    else:
        part = df.sort_values(x_col)
        ax.plot(part[x_col].astype(str), part[y_col], marker=marker)
    if title:
        ax.set_title(title)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=30)
    return ax


def metric_heatmap_from_table(
    df,
    *,
    row_col: str,
    col_col: str,
    value_col: str,
    ax=None,
    title: str | None = None,
    cmap: str = "viridis",
):
    """Pivot a table and render it as a heatmap."""
    table = df.pivot_table(index=row_col, columns=col_col, values=value_col, aggfunc="mean")
    return heatmap(
        table.to_numpy(dtype=float),
        row_labels=[str(v) for v in table.index],
        col_labels=[str(v) for v in table.columns],
        ax=ax,
        title=title,
        cmap=cmap,
        annotate=True,
        fmt=".2f",
    )
