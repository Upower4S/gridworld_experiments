"""Small statistical helpers for conceptual belief space analyses."""

from __future__ import annotations

from itertools import permutations
from math import factorial
from typing import Callable, Sequence

import numpy as np


def _as_1d(x: Sequence[float] | np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=float).reshape(-1)
    return arr[np.isfinite(arr)]


def pearson_corr(x: Sequence[float] | np.ndarray, y: Sequence[float] | np.ndarray) -> float:
    """Pearson correlation with NaN for degenerate inputs."""
    x_arr = np.asarray(x, dtype=float).reshape(-1)
    y_arr = np.asarray(y, dtype=float).reshape(-1)
    mask = np.isfinite(x_arr) & np.isfinite(y_arr)
    x_arr = x_arr[mask]
    y_arr = y_arr[mask]
    if len(x_arr) < 2 or np.std(x_arr) == 0 or np.std(y_arr) == 0:
        return float("nan")
    return float(np.corrcoef(x_arr, y_arr)[0, 1])


def rankdata_average(values: Sequence[float] | np.ndarray) -> np.ndarray:
    """Average ranks for ties, equivalent to scipy.stats.rankdata(method='average')."""
    x = np.asarray(values, dtype=float)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=float)
    sorted_x = x[order]
    i = 0
    while i < len(x):
        j = i + 1
        while j < len(x) and sorted_x[j] == sorted_x[i]:
            j += 1
        rank = 0.5 * (i + 1 + j)
        ranks[order[i:j]] = rank
        i = j
    return ranks


def spearman_corr(x: Sequence[float] | np.ndarray, y: Sequence[float] | np.ndarray) -> float:
    """Spearman rank correlation without requiring SciPy."""
    x_arr = np.asarray(x, dtype=float).reshape(-1)
    y_arr = np.asarray(y, dtype=float).reshape(-1)
    mask = np.isfinite(x_arr) & np.isfinite(y_arr)
    x_arr = x_arr[mask]
    y_arr = y_arr[mask]
    if len(x_arr) < 2:
        return float("nan")
    return pearson_corr(rankdata_average(x_arr), rankdata_average(y_arr))


def kendall_tau(x: Sequence[float] | np.ndarray, y: Sequence[float] | np.ndarray) -> float:
    """Kendall tau if SciPy is available; otherwise NaN."""
    try:
        from scipy.stats import kendalltau  # type: ignore
    except Exception:
        return float("nan")
    x_arr = np.asarray(x, dtype=float).reshape(-1)
    y_arr = np.asarray(y, dtype=float).reshape(-1)
    mask = np.isfinite(x_arr) & np.isfinite(y_arr)
    if mask.sum() < 2:
        return float("nan")
    result = kendalltau(x_arr[mask], y_arr[mask], nan_policy="omit")
    return float(result.statistic)


def correlation_summary(x: Sequence[float] | np.ndarray, y: Sequence[float] | np.ndarray) -> dict[str, float]:
    """Return Pearson, Spearman, and Kendall correlations."""
    return {
        "pearson": pearson_corr(x, y),
        "spearman": spearman_corr(x, y),
        "kendall_tau": kendall_tau(x, y),
    }


def permutation_test_spearman(
    behavior_matrix: np.ndarray,
    activation_matrix: np.ndarray,
    *,
    n_permutations: int = 10000,
    seed: int = 42,
    upper_triangle_fn: Callable[[np.ndarray], np.ndarray],
) -> dict[str, float | int | str]:
    """Permutation test by shuffling activation action labels.

    The p-value is one-sided for positive geometry alignment: shuffled Spearman
    >= observed Spearman.
    """
    n_actions = behavior_matrix.shape[0]
    observed = spearman_corr(upper_triangle_fn(behavior_matrix), upper_triangle_fn(activation_matrix))
    if not np.isfinite(observed):
        return {
            "permutation_spearman_observed": float("nan"),
            "permutation_p_value": float("nan"),
            "n_permutations_effective": 0,
            "permutation_method": "not_run_degenerate",
        }

    max_exact = factorial(n_actions)
    if n_permutations >= max_exact:
        perm_iter = list(permutations(range(n_actions)))
        method = "exact_all_action_permutations"
    else:
        rng = np.random.default_rng(seed)
        perm_iter = [tuple(rng.permutation(n_actions).tolist()) for _ in range(n_permutations)]
        method = "random_action_label_permutations"

    count_ge = 0
    for perm in perm_iter:
        perm_matrix = activation_matrix[np.ix_(perm, perm)]
        perm_corr = spearman_corr(upper_triangle_fn(behavior_matrix), upper_triangle_fn(perm_matrix))
        if np.isfinite(perm_corr) and perm_corr >= observed - 1e-12:
            count_ge += 1
    n_eff = len(perm_iter)
    p_value = (count_ge + 1.0) / (n_eff + 1.0)
    return {
        "permutation_spearman_observed": float(observed),
        "permutation_p_value": float(p_value),
        "n_permutations_effective": int(n_eff),
        "permutation_method": method,
    }


def bootstrap_corr_ci(
    x: Sequence[float] | np.ndarray,
    y: Sequence[float] | np.ndarray,
    *,
    statistic: Callable[[Sequence[float] | np.ndarray, Sequence[float] | np.ndarray], float] = spearman_corr,
    n_bootstrap: int = 2000,
    seed: int = 42,
    ci: float = 0.95,
) -> dict[str, float | int | str]:
    """Bootstrap a correlation CI over paired distances.

    This is a fallback for centroid-only inputs. It is not a decision-level or
    trajectory-level bootstrap.
    """
    x_arr = np.asarray(x, dtype=float).reshape(-1)
    y_arr = np.asarray(y, dtype=float).reshape(-1)
    mask = np.isfinite(x_arr) & np.isfinite(y_arr)
    x_arr = x_arr[mask]
    y_arr = y_arr[mask]
    if len(x_arr) < 2:
        return {
            "bootstrap_ci_low": float("nan"),
            "bootstrap_ci_high": float("nan"),
            "n_bootstrap_effective": 0,
            "bootstrap_method": "not_run_degenerate",
        }

    rng = np.random.default_rng(seed)
    stats: list[float] = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, len(x_arr), size=len(x_arr))
        value = statistic(x_arr[idx], y_arr[idx])
        if np.isfinite(value):
            stats.append(float(value))
    if not stats:
        return {
            "bootstrap_ci_low": float("nan"),
            "bootstrap_ci_high": float("nan"),
            "n_bootstrap_effective": 0,
            "bootstrap_method": "not_run_all_degenerate",
        }
    alpha = (1.0 - ci) / 2.0
    return {
        "bootstrap_ci_low": float(np.quantile(stats, alpha)),
        "bootstrap_ci_high": float(np.quantile(stats, 1.0 - alpha)),
        "n_bootstrap_effective": int(len(stats)),
        "bootstrap_method": "paired_pairwise_distance_bootstrap_centroid_only",
    }
