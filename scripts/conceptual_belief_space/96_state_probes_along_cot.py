#!/usr/bin/env python
"""Poster-style state and agent-action probes along the CoT (companion to 95).

Trains the probe family from the Okcu et al. belief-action-gap poster as full
curves along the chain of thought, on this repo's multi-map corpus and under
script 95's comparability protocol (fixed decision panel, window-mean input,
trajectory-grouped nested CV, OOF evaluation, cluster bootstrap CIs):

  agent_action   the action the agent takes (4-class; poster row "Agent actions")
  location       agent (row, col): two 9-class probes; exact-cell = both correct
  has_key        carrying_key (binary)
  door_open      door removed from the grid (binary)

All targets are step-level constants, so the panel guarantees identical label
distributions in every bin; curve changes along t are purely representational.
Layer chosen per (target, fold) by inner grouped CV on bins {1, 4, 8}
(training folds only); C fixed at 1.0; readouts are linear (<= 9*512+9
parameters each, far under the 1M contract cap).

Interpretation caveat (stated in advance): the input window is the last 12 CoT
tokens, and the CoT text itself mentions coordinates and key/door facts, so
these curves measure which world-state information is linearly present in the
*working window* along reasoning — re-encoded text is not excluded here. That
is the phenomenon of interest (cf. the poster's pre/post position collapse),
not a leak, but it must not be read as "stored belief outside the text".

Run from the repo root or a worktree:
  interp/.venv/bin/python scripts/conceptual_belief_space/96_state_probes_along_cot.py
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))  # noqa: E402

_SPEC = importlib.util.spec_from_file_location(
    "s95", Path(__file__).parent / "95_optimal_action_probe_along_cot.py")
s95 = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(s95)

from src.conceptual_belief_space.io import ensure_dir, save_markdown  # noqa: E402

SEED = 42
N_BINS = 10
N_FOLDS = 5
N_INNER = 4
FIXED_C = 1.0
SELECT_BINS = (1, 4, 8)

INK = "#333333"
GRID_C = "#e6e6e6"
BLUE = "#2a78d6"
ORANGE = "#eb6834"
GREEN = "#1baf7a"
PURPLE = "#4a3aa7"
GOLD = "#c9a227"
NEUTRAL = "#9aa0a6"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--activation_index", type=Path, default=s95.DATA_ROOT / (
        "outputs/conceptual_belief_space/cot_window_layer_transition_probe/"
        "activations/activation_index.parquet"))
    parser.add_argument("--trajectories", type=Path, default=s95.DATA_ROOT /
                        "trajectories_key_door_100/trajectories_key_door")
    parser.add_argument("--optimal_curves", type=Path, default=s95.DATA_ROOT / (
        "outputs/conceptual_belief_space/optimal_action_probe_along_cot/"
        "spec_window_mean/curves.csv"))
    parser.add_argument("--out_dir", type=Path, default=s95.DATA_ROOT /
                        "outputs/conceptual_belief_space/state_probes_along_cot")
    parser.add_argument("--figure_dir", type=Path,
                        default=s95.DATA_ROOT / "figures/conceptual_belief_space")
    parser.add_argument("--n_boot", type=int, default=2000)
    return parser.parse_args()


def step_state_labels(trajectory_dir: Path) -> pd.DataFrame:
    """Per (trajectory, step): agent action, position, has_key, door_open."""
    rows = []
    for path in sorted(trajectory_dir.glob("*.json")):
        data = json.loads(path.read_text())
        for step in data["steps"]:
            cells = s95.parse_grid(step["grid_state"])
            pos = next((r, c) for r, row in enumerate(cells)
                       for c, ch in enumerate(row) if ch == "A")
            rows.append({
                "trajectory_id": path.stem, "step_id": int(step["step_id"]),
                "agent_action": str(step["agent_action"]),
                "agent_row": int(pos[0]), "agent_col": int(pos[1]),
                "has_key": int(step["carrying_key"]),
                "door_open": int(not any("D" in "".join(r) for r in cells)),
            })
    return pd.DataFrame(rows)


def as_sets(values, vocab) -> list[tuple[str, ...]]:
    return [(vocab[v] if not isinstance(v, str) else v,) for v in values]


def main() -> int:
    args = parse_args()
    ensure_dir(args.out_dir)
    ensure_dir(args.figure_dir)

    print(f"data root: {s95.DATA_ROOT}", flush=True)
    labels = step_state_labels(args.trajectories)
    print(f"labels: {len(labels)} steps; has_key mean {labels.has_key.mean():.3f}; "
          f"door_open mean {labels.door_open.mean():.3f}; "
          f"{labels.agent_row.nunique()} rows x {labels.agent_col.nunique()} cols",
          flush=True)

    index = pd.read_parquet(args.activation_index)
    panel = s95.build_panel(index)
    panel = panel.join(labels.set_index(["trajectory_id", "step_id"]),
                       on=["trajectory_id", "step_id"], how="inner")
    panel = panel.reset_index(drop=True)
    per_bin = panel.groupby("bin").decision.nunique()
    assert per_bin.nunique() == 1
    n_steps = int(per_bin.iloc[0])
    print(f"panel: {n_steps} steps x {N_BINS} bins", flush=True)

    print("features (window_mean) ...", flush=True)
    features = panel_features = s95.extract_features(panel, "window_mean")

    bins = panel.bin.to_numpy()
    groups_step = panel.trajectory_id.to_numpy()
    decision_of = panel.decision.to_numpy()
    decisions = panel.drop_duplicates("decision")[["decision", "trajectory_id"]]
    dec_traj = decisions.trajectory_id.to_numpy()
    dec_name = decisions.decision.to_numpy()
    fold_pairs = list(s95.grouped_folds(dec_traj, N_FOLDS,
                                        np.random.default_rng(SEED)))
    fold_of_decision = {}
    for f, (_, test_idx) in enumerate(fold_pairs):
        for d in dec_name[test_idx]:
            fold_of_decision[d] = f
    fold_arr = np.array([fold_of_decision[d] for d in decision_of])

    row_vocab = [str(v) for v in range(9)]
    targets: dict[str, list[tuple[str, ...]]] = {
        "agent_action": [(a,) for a in panel.agent_action],
        "agent_row": [(str(v),) for v in panel.agent_row],
        "agent_col": [(str(v),) for v in panel.agent_col],
        "has_key": [(str(v),) for v in panel.has_key],
        "door_open": [(str(v),) for v in panel.door_open],
    }
    vocab_of = {
        "agent_action": list(s95.ACTIONS),
        "agent_row": row_vocab, "agent_col": row_vocab,
        "has_key": ["0", "1"], "door_open": ["0", "1"],
    }

    def fit_eval(rows_tr, rows_te, sets_all, layer, vocab):
        scaler, model = fit_probe_vocab(features[rows_tr, layer],
                                        [sets_all[i] for i in rows_tr], vocab)
        probs = probs_vocab(scaler, model, features[rows_te, layer], vocab)
        top = probs.argmax(axis=1)
        return np.array([vocab[t] in sets_all[i]
                         for t, i in zip(top, rows_te)], dtype=float), probs

    def fit_probe_vocab(x_train, sets_train, vocab):
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler().fit(x_train)
        transformed = scaler.transform(x_train)
        xs, ys, ws = [], [], []
        for i, actions in enumerate(sets_train):
            for a in actions:
                xs.append(transformed[i])
                ys.append(vocab.index(a))
                ws.append(1.0 / len(actions))
        model = LogisticRegression(C=FIXED_C, max_iter=2000, tol=1e-4)
        model.fit(np.asarray(xs), np.asarray(ys), sample_weight=np.asarray(ws))
        return scaler, model

    def probs_vocab(scaler, model, x, vocab):
        raw = model.predict_proba(scaler.transform(x))
        out = np.zeros((len(x), len(vocab)))
        for j, cls in enumerate(model.classes_):
            out[:, int(cls)] = raw[:, j]
        return out

    hits: dict[str, np.ndarray] = {}
    chosen: dict[str, list[int]] = {}
    for name, sets_all in targets.items():
        vocab = vocab_of[name]
        print(f"target {name} ...", flush=True)
        layers_for_fold = []
        for f in range(N_FOLDS):
            train_traj = dec_traj[fold_pairs[f][0]]
            inner_pairs = list(s95.grouped_folds(
                train_traj, N_INNER, np.random.default_rng(SEED + 100 + f)))
            scores = np.zeros(24)
            for layer in range(24):
                accs = []
                for itr, ite in inner_pairs:
                    itr_traj = list(set(train_traj[itr]))
                    ite_traj = list(set(train_traj[ite]))
                    for b in SELECT_BINS:
                        row_tr = np.flatnonzero((bins == b)
                                                & np.isin(groups_step, itr_traj))
                        row_te = np.flatnonzero((bins == b)
                                                & np.isin(groups_step, ite_traj))
                        h, _ = fit_eval(row_tr, row_te, sets_all, layer, vocab)
                        accs.append(h.mean())
                scores[layer] = float(np.mean(accs))
            layers_for_fold.append(int(scores.argmax()))
        chosen[name] = layers_for_fold
        print(f"  layers {layers_for_fold}", flush=True)

        target_hits = np.zeros(len(panel))
        for f in range(N_FOLDS):
            layer = layers_for_fold[f]
            for b in range(N_BINS):
                tr = np.flatnonzero((bins == b) & (fold_arr != f))
                te = np.flatnonzero((bins == b) & (fold_arr == f))
                h, _ = fit_eval(tr, te, sets_all, layer, vocab)
                target_hits[te] = h
        hits[name] = target_hits
        panel[f"hit_{name}"] = target_hits

    panel["hit_location"] = panel.hit_agent_row * panel.hit_agent_col
    hits["location"] = panel.hit_location.to_numpy()

    # Priors (train-fold majority, evaluated OOF) for context lines.
    priors: dict[str, np.ndarray] = {}
    for name in [*targets, "location"]:
        col = {"agent_action": "agent_action", "agent_row": "agent_row",
               "agent_col": "agent_col", "has_key": "has_key",
               "door_open": "door_open", "location": None}[name]
        prior_hits = np.zeros(len(panel))
        for f in range(N_FOLDS):
            train = panel[fold_arr != f].drop_duplicates("decision")
            te = np.flatnonzero(fold_arr == f)
            if name == "location":
                top = (train.groupby(["agent_row", "agent_col"]).size().idxmax())
                prior_hits[te] = ((panel.agent_row.to_numpy()[te] == top[0])
                                  & (panel.agent_col.to_numpy()[te] == top[1]))
            else:
                top = train[col].mode().iloc[0]
                prior_hits[te] = panel[col].to_numpy()[te] == top
        priors[name] = prior_hits

    # Cluster bootstrap CIs per bin.
    traj_of_panel = panel.trajectory_id.to_numpy()
    unique_traj = np.array(sorted(set(traj_of_panel)))
    boot_take = np.random.default_rng(SEED).integers(
        0, len(unique_traj), (args.n_boot, len(unique_traj)))
    traj_rows = {t: np.flatnonzero(traj_of_panel == t) for t in unique_traj}

    def cluster_ci(values, sel):
        pts = []
        for draw in boot_take:
            rows = np.concatenate([traj_rows[unique_traj[t]] for t in draw])
            pts.append(values[rows][sel[rows]].mean())
        return float(np.quantile(pts, 0.025)), float(np.quantile(pts, 0.975))

    curve_rows = []
    for b in range(N_BINS):
        sel = bins == b
        entry = {"bin": b, "center": (b + 0.5) / N_BINS, "n": int(sel.sum())}
        for name in ["agent_action", "location", "agent_row", "agent_col",
                     "has_key", "door_open"]:
            lo, hi = cluster_ci(hits[name], sel)
            entry[f"{name}_acc"] = float(hits[name][sel].mean())
            entry[f"{name}_lo"], entry[f"{name}_hi"] = lo, hi
            entry[f"{name}_prior"] = float(priors[name][sel].mean())
        curve_rows.append(entry)
    curves = pd.DataFrame(curve_rows)
    curves.to_csv(args.out_dir / "curves.csv", index=False)

    optimal = pd.read_csv(args.optimal_curves)
    plot(curves, optimal, chosen,
         args.figure_dir / "poster_probes_along_cot.png")

    lines = [
        "# Poster-style state and agent-action probes along the CoT (script 96)",
        "",
        f"- {n_steps} decisions x {N_BINS} bins; window_mean input; per-(target,fold) "
        f"layers chosen on inner grouped CV over bins {SELECT_BINS}: "
        + "; ".join(f"{k}: {v}" for k, v in chosen.items()),
        "- Caveat: the 12-token working window re-encodes text the model is "
        "currently writing; curves measure state information present in the "
        "window along reasoning, not stored belief outside the text.",
        "",
        curves.round(4).to_markdown(index=False),
    ]
    save_markdown(args.out_dir / "report.md", "\n".join(lines))
    print(f"done -> {args.out_dir}")
    return 0


def plot(curves: pd.DataFrame, optimal: pd.DataFrame,
         chosen: dict, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(12.4, 4.8), sharex=True)

    axis = axes[0]
    x = optimal.center
    axis.fill_between(x, optimal.q_lo, optimal.q_hi, color=ORANGE, alpha=0.15,
                      linewidth=0)
    axis.plot(x, optimal.q_acc, color=ORANGE, linewidth=2.2, marker="o",
              markersize=4.5, label="forced-answer belief q")
    axis.fill_between(x, optimal.probe_lo, optimal.probe_hi, color=BLUE,
                      alpha=0.15, linewidth=0)
    axis.plot(x, optimal.probe_acc, color=BLUE, linewidth=2.2, marker="o",
              markersize=4.5, label="probe: optimal action (top-1-in-set)")
    axis.fill_between(curves.center, curves.agent_action_lo, curves.agent_action_hi,
                      color=GREEN, alpha=0.13, linewidth=0)
    axis.plot(curves.center, curves.agent_action_acc, color=GREEN, linewidth=2,
              marker="s", markersize=3.8, label="probe: agent action")
    axis.axhline(optimal.chance.iloc[0], color=INK, linewidth=1.0, linestyle=":",
                 label="chance (optimal-set)")
    axis.plot(curves.center, curves.agent_action_prior, color=NEUTRAL,
              linewidth=1.2, linestyle="--", label="prior (agent action)")
    axis.set_title("Action probes", color=INK, fontsize=11)
    axis.set_ylabel("accuracy", color=INK)
    axis.legend(frameon=False, fontsize=8.2, labelcolor=INK, loc="upper left")

    axis = axes[1]
    for name, color, label in (
            ("location", PURPLE, "location (exact cell, row & col)"),
            ("has_key", GOLD, "has_key"),
            ("door_open", GREEN, "door_open")):
        axis.fill_between(curves.center, curves[f"{name}_lo"], curves[f"{name}_hi"],
                          color=color, alpha=0.13, linewidth=0)
        axis.plot(curves.center, curves[f"{name}_acc"], color=color, linewidth=2,
                  marker="o", markersize=3.8, label=f"probe: {label}")
        axis.plot(curves.center, curves[f"{name}_prior"], color=color,
                  linewidth=1.0, linestyle="--", alpha=0.55)
    axis.set_title("World-state probes (dashed: majority prior)", color=INK,
                   fontsize=11)
    axis.legend(frameon=False, fontsize=8.2, labelcolor=INK, loc="center left")

    for axis in axes:
        axis.set_xlabel("relative CoT position (bin center)", color=INK)
        axis.set_ylim(0.0, 1.02)
        axis.grid(True, color=GRID_C, linewidth=0.8)
        axis.set_axisbelow(True)
        for spine in ("top", "right"):
            axis.spines[spine].set_visible(False)
        axis.tick_params(colors=INK)
    figure.suptitle("Poster probe family along the chain of thought "
                    "(window-mean input, OOF, 95% cluster CIs)",
                    color=INK, fontsize=12)
    figure.tight_layout()
    figure.savefig(path, dpi=220)
    plt.close(figure)


if __name__ == "__main__":
    raise SystemExit(main())
