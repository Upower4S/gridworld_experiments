#!/usr/bin/env python
"""Linear probes for the ENVIRONMENT-OPTIMAL action along the CoT, versus q.

At the same cut positions where the forced-answer belief q is measured, train
linear probes on residual-stream activations to decode the environment-optimal
action, and compare the probe's accuracy curve with q's own accuracy curve.
If the probe is high early while q converges late, a correct action is linearly
present in the state before the model's belief settles.

Design:
  Labels     O_s = set of optimal first moves from BFS over (row, col, has_key)
             with the door passable only while carrying the key. O_s is a SET
             (ties are real) and constant along the CoT.
  Metric     top-1-in-set accuracy 1[argmax(prediction) in O_s], identical for
             probe and q. Ties handled in training with soft targets uniform
             over O_s (sample duplication, weight 1/|O_s|).
  Panel      10 relative-position bins; each step contributes its nearest
             prefix to every bin -> identical step composition and identical
             label distribution in all bins; changes along t are purely
             representational (no survivorship, no label drift).
  Protocol   the same trajectory-grouped 5-fold layout for every bin and
             layer; per-fold layer chosen on inner grouped CV averaged across
             bins (training folds only); C chosen per bin by inner CV;
             evaluation is out-of-fold (every step scored exactly once per
             bin); trajectory-cluster bootstrap CIs.

Anti-leakage contract: frozen backbone; inputs are activations only (last
valid CoT token at the cut, one layer, 512-d projected); no forced-suffix
tokens, no future tokens, no grid/task features as INPUTS (the environment-
derived optimal action is the TARGET, which the contract permits); linear
readout 512*4+4 = 2,052 parameters << 1M; trajectory-grouped splits with zero
trajectory overlap.

Caveat stated up front: agent_action is in O_s for 98.3% of steps, so decoding
the optimal action and decoding the upcoming action are confounded at ~98% on
this corpus; the informative objects are the probe-vs-q timing comparison and
the multi-optimal structure.

Run from the repo root (or a worktree; data paths fall back to the primary
checkout automatically):
  interp/.venv/bin/python scripts/conceptual_belief_space/95_optimal_action_probe_along_cot.py
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, deque
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))  # noqa: E402

from src.conceptual_belief_space.forced_suffix_probe import ProjectedShardStore  # noqa: E402
from src.conceptual_belief_space.io import ensure_dir, save_markdown  # noqa: E402


def data_root() -> Path:
    """Untracked data lives in the primary checkout; fall back when run from a
    worktree under <main>/.claude/worktrees/<name>."""
    if (REPO_ROOT / "outputs").exists():
        return REPO_ROOT
    if (REPO_ROOT.parents[0].name == "worktrees"
            and REPO_ROOT.parents[1].name == ".claude"):
        return REPO_ROOT.parents[2]
    return REPO_ROOT


DATA_ROOT = data_root()

SEED = 42
ACTIONS = ("UP", "DOWN", "LEFT", "RIGHT")
MOVES = {"UP": (-1, 0), "DOWN": (1, 0), "LEFT": (0, -1), "RIGHT": (0, 1)}
N_BINS = 10
N_FOLDS = 5
N_INNER = 4
C_GRID = (0.01, 0.1, 1.0, 10.0)
FIXED_C = 1.0
N_BOOT = 2000

INK = "#333333"
GRID_C = "#e6e6e6"
BLUE = "#2a78d6"
ORANGE = "#eb6834"
NEUTRAL = "#9aa0a6"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--activation_index", type=Path, default=DATA_ROOT / (
        "outputs/conceptual_belief_space/cot_window_layer_transition_probe/"
        "activations/activation_index.parquet"))
    parser.add_argument("--trajectories", type=Path,
                        default=DATA_ROOT / "trajectories_key_door_100/trajectories_key_door")
    parser.add_argument("--out_dir", type=Path, default=DATA_ROOT /
                        "outputs/conceptual_belief_space/optimal_action_probe_along_cot")
    parser.add_argument("--figure_dir", type=Path,
                        default=DATA_ROOT / "figures/conceptual_belief_space")
    parser.add_argument("--n_boot", type=int, default=N_BOOT)
    parser.add_argument("--input_spec", choices=("window_mean", "last_token"),
                        default="window_mean",
                        help="probe input: masked mean over the 12-token CoT window "
                             "(primary; the bin-9 diagnostic showed last-token "
                             "features are near-null while window-mean reaches "
                             "0.544) or the last valid CoT token")
    parser.add_argument("--abs_bin_width", type=int, default=60,
                        help="absolute-token appendix: bin width in tokens")
    parser.add_argument("--abs_min_tokens", type=int, default=480,
                        help="absolute-token appendix: complete-case threshold")
    args = parser.parse_args()
    args.out_dir = args.out_dir / f"spec_{args.input_spec}"
    return args


# ---------------------------------------------------------------------------
# Optimal-action sets (BFS over (row, col, has_key)); validated in the grid
# audit: zero state mismatches against observed (position, carrying_key),
# 98.28% agent-optimality.
# ---------------------------------------------------------------------------

def parse_grid(rows: list[str]) -> list[list[str]]:
    cells = []
    for line in rows[1:]:
        body = line[2:]
        cells.append([body[k] for k in range(0, len(body), 2)])
    return cells


def passable(ch: str, has_key: int) -> bool:
    if ch == "#":
        return False
    if ch == "D":
        return bool(has_key)
    return True


def joint_distances(cells: list[list[str]], goal: tuple[int, int],
                    key_cells: set[tuple[int, int]]) -> dict:
    height, width = len(cells), len(cells[0])
    states = [(r, c, k) for r in range(height) for c in range(width)
              for k in (0, 1) if passable(cells[r][c], k)]
    reverse: dict = {s: [] for s in states}
    for (r, c, k) in states:
        for dr, dc in MOVES.values():
            nr, nc = r + dr, c + dc
            if not (0 <= nr < height and 0 <= nc < width):
                continue
            if not passable(cells[nr][nc], k):
                continue
            nk = 1 if (k or (nr, nc) in key_cells) else 0
            if (nr, nc, nk) in reverse:
                reverse[(nr, nc, nk)].append((r, c, k))
    dist: dict = {}
    queue: deque = deque()
    for k in (0, 1):
        s = (goal[0], goal[1], k)
        if s in reverse:
            dist[s] = 0
            queue.append(s)
    while queue:
        s = queue.popleft()
        for p in reverse[s]:
            if p not in dist:
                dist[p] = dist[s] + 1
                queue.append(p)
    return dist


def optimal_sets(trajectory_dir: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(trajectory_dir.glob("*.json")):
        data = json.loads(path.read_text())
        steps = data["steps"]
        cells0 = parse_grid(steps[0]["grid_state"])
        goal = next((r, c) for r, row in enumerate(cells0)
                    for c, ch in enumerate(row) if ch == "G")
        key_cells = {(r, c) for r, row in enumerate(cells0)
                     for c, ch in enumerate(row) if ch == "K"}
        dist = joint_distances(cells0, goal, key_cells)
        for step in steps:
            cells = parse_grid(step["grid_state"])
            pos = next((r, c) for r, row in enumerate(cells)
                       for c, ch in enumerate(row) if ch == "A")
            has_key = int(step["carrying_key"])
            d_now = dist.get((pos[0], pos[1], has_key))
            optimal = []
            for action, (dr, dc) in MOVES.items():
                nr, nc = pos[0] + dr, pos[1] + dc
                if not (0 <= nr < len(cells0) and 0 <= nc < len(cells0[0])):
                    continue
                if not passable(cells0[nr][nc], has_key):
                    continue
                nk = 1 if (has_key or (nr, nc) in key_cells) else 0
                d_next = dist.get((nr, nc, nk))
                if d_now is not None and d_next is not None and d_next == d_now - 1:
                    optimal.append(action)
            rows.append({
                "trajectory_id": path.stem, "step_id": int(step["step_id"]),
                "agent_action": str(step["agent_action"]),
                "optimal_actions": tuple(sorted(optimal)),
                "n_optimal": len(optimal),
                "agent_in_optimal": str(step["agent_action"]) in optimal,
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Panel construction
# ---------------------------------------------------------------------------

def build_panel(index: pd.DataFrame) -> pd.DataFrame:
    """One prefix per (step, bin): nearest to the bin center in relative position."""
    index = index.copy()
    index["decision"] = index.trajectory_id + "::" + index.step_id.astype(str)
    max_idx = index.groupby("decision").prefix_token_index.transform("max")
    index["progress"] = np.where(max_idx > 0, index.prefix_token_index / max_idx, 0.0)
    centers = (np.arange(N_BINS) + 0.5) / N_BINS
    picked: list[tuple[int, int]] = []
    for _, group in index.groupby("decision", sort=False):
        prog = group.progress.to_numpy()
        order = group.index.to_numpy()
        for b, center in enumerate(centers):
            j = int(np.argmin(np.abs(prog - center)))
            picked.append((order[j], b))
    rows = index.loc[[p for p, _ in picked]].copy()
    rows["bin"] = [b for _, b in picked]
    return rows.reset_index(drop=True)


def window_feature(state: np.ndarray, mask: list[int], input_spec: str) -> np.ndarray:
    """[24, 12, 512] -> [24, 512] per the input spec (masked mean or last token)."""
    valid = [i for i, m in enumerate(mask) if m]
    if not valid:
        valid = [state.shape[1] - 1]
    if input_spec == "last_token":
        return state[:, valid[-1], :]
    return state[:, valid, :].mean(axis=1)


def extract_features(panel: pd.DataFrame, input_spec: str) -> np.ndarray:
    store = ProjectedShardStore(max_open=8)
    features = np.empty((len(panel), 24, 512), dtype=np.float32)
    for _, group in panel.groupby("state_path", sort=False):
        for i, row in group.iterrows():
            state, _ = store.load_row(row)
            mask = json.loads(row.input_token_mask_json)
            features[i] = window_feature(state, mask, input_spec)
    return features


# ---------------------------------------------------------------------------
# Probing
# ---------------------------------------------------------------------------

def soft_expand(x: np.ndarray, sets: list[tuple[str, ...]]):
    """Duplicate each sample once per optimal action, weight 1/|O|."""
    xs, ys, ws = [], [], []
    for i, actions in enumerate(sets):
        for a in actions:
            xs.append(x[i])
            ys.append(ACTIONS.index(a))
            ws.append(1.0 / len(actions))
    return np.asarray(xs), np.asarray(ys), np.asarray(ws)


def fit_probe(x_train: np.ndarray, sets_train: list, c_value: float):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler().fit(x_train)
    xs, ys, ws = soft_expand(scaler.transform(x_train), sets_train)
    model = LogisticRegression(C=c_value, max_iter=2000, tol=1e-4)
    model.fit(xs, ys, sample_weight=ws)
    return scaler, model


def probe_probs(scaler, model, x: np.ndarray) -> np.ndarray:
    """Return [n, 4] probabilities in ACTIONS order (absent classes -> 0)."""
    raw = model.predict_proba(scaler.transform(x))
    out = np.zeros((len(x), 4))
    for j, cls in enumerate(model.classes_):
        out[:, int(cls)] = raw[:, j]
    return out


def hit_in_set(probs: np.ndarray, sets: list[tuple[str, ...]]) -> np.ndarray:
    top = probs.argmax(axis=1)
    return np.array([ACTIONS[t] in s for t, s in zip(top, sets)], dtype=float)


def grouped_folds(groups: np.ndarray, n_folds: int, rng: np.random.Generator):
    """Deterministic trajectory-grouped folds: shuffle unique groups, chunk."""
    unique = np.array(sorted(set(groups)))
    rng.shuffle(unique)
    for chunk in np.array_split(unique, n_folds):
        test_mask = np.isin(groups, chunk)
        yield np.flatnonzero(~test_mask), np.flatnonzero(test_mask)


def main() -> int:
    args = parse_args()
    ensure_dir(args.out_dir)
    ensure_dir(args.figure_dir)

    print(f"data root: {DATA_ROOT}", flush=True)
    print("optimal-action sets (BFS) ...", flush=True)
    sets_df = optimal_sets(args.trajectories)
    agent_rate = float(sets_df.agent_in_optimal.mean())
    print(f"  {len(sets_df)} steps; agent in O_s: {agent_rate:.4f}; "
          f"|O| dist: {dict(sorted(Counter(sets_df.n_optimal).items()))}")
    assert abs(agent_rate - 0.9828) < 0.002, "BFS drifted from the audited 98.28%"
    sets_df.assign(optimal_actions=[",".join(t) for t in sets_df.optimal_actions]) \
        .to_parquet(args.out_dir / "optimal_sets.parquet", index=False)

    print("panel ...", flush=True)
    index = pd.read_parquet(args.activation_index)
    panel = build_panel(index)
    key = sets_df.set_index(["trajectory_id", "step_id"])
    panel = panel.join(key, on=["trajectory_id", "step_id"], how="inner")
    panel = panel[panel.n_optimal > 0].reset_index(drop=True)

    per_bin = panel.groupby("bin").decision.nunique()
    assert per_bin.nunique() == 1, f"bins differ in step count: {per_bin.to_dict()}"
    n_steps = int(per_bin.iloc[0])
    reuse = 1.0 - panel.groupby("bin").prefix_id.nunique().min() / n_steps
    print(f"  {n_steps} steps x {N_BINS} bins; worst within-bin prefix reuse {reuse:.3f}")

    print("features ...", flush=True)
    features = extract_features(panel, args.input_spec)

    q_cols = [f"extracted_q_{a}" for a in ACTIONS]
    q_probs = panel[q_cols].to_numpy(dtype=float)
    sets_all = panel.optimal_actions.tolist()
    q_hits = hit_in_set(q_probs, sets_all)
    panel["q_hit"] = q_hits

    groups_step = panel.trajectory_id.to_numpy()
    decision_of = panel.decision.to_numpy()
    bins = panel.bin.to_numpy()

    decisions = panel.drop_duplicates("decision")[["decision", "trajectory_id"]]
    dec_traj = decisions.trajectory_id.to_numpy()
    dec_name = decisions.decision.to_numpy()
    fold_pairs = list(grouped_folds(dec_traj, N_FOLDS, np.random.default_rng(SEED)))
    fold_of_decision = {}
    for f, (_, test_idx) in enumerate(fold_pairs):
        for d in dec_name[test_idx]:
            fold_of_decision[d] = f
    fold_arr = np.array([fold_of_decision[d] for d in decision_of])

    def rows_of(bin_id: int, fold: int, train: bool) -> np.ndarray:
        keep = (bins == bin_id) & ((fold_arr != fold) if train else (fold_arr == fold))
        return np.flatnonzero(keep)

    # ---- Layer x bin heatmap at fixed C (outer-fold OOF accuracy) ----------
    print("layer x bin heatmap ...", flush=True)
    heat = np.zeros((24, N_BINS))
    for layer in range(24):
        hits = np.zeros(len(panel))
        for b in range(N_BINS):
            for f in range(N_FOLDS):
                tr = rows_of(b, f, True)
                te = rows_of(b, f, False)
                scaler, model = fit_probe(features[tr, layer],
                                          [sets_all[i] for i in tr], FIXED_C)
                hits[te] = hit_in_set(probe_probs(scaler, model, features[te, layer]),
                                      [sets_all[i] for i in te])
        for b in range(N_BINS):
            heat[layer, b] = hits[bins == b].mean()
        print(f"  layer {layer:2d}: mean {heat[layer].mean():.3f}", flush=True)
    pd.DataFrame(heat, columns=[f"bin{b}" for b in range(N_BINS)]) \
        .assign(layer=range(24)).to_csv(args.out_dir / "layer_bin_heatmap.csv",
                                        index=False)

    # ---- Per-fold layer choice on TRAINING data only (inner grouped CV) ----
    print("per-fold layer selection ...", flush=True)
    chosen_layers = []
    for f in range(N_FOLDS):
        train_traj = dec_traj[fold_pairs[f][0]]
        inner_pairs = list(grouped_folds(train_traj, N_INNER,
                                         np.random.default_rng(SEED + 100 + f)))
        scores = np.zeros(24)
        for layer in range(24):
            accs = []
            for itr, ite in inner_pairs:
                itr_traj = list(set(train_traj[itr]))
                ite_traj = list(set(train_traj[ite]))
                for b in range(N_BINS):
                    row_tr = np.flatnonzero((bins == b)
                                            & np.isin(groups_step, itr_traj))
                    row_te = np.flatnonzero((bins == b)
                                            & np.isin(groups_step, ite_traj))
                    scaler, model = fit_probe(features[row_tr, layer],
                                              [sets_all[i] for i in row_tr], FIXED_C)
                    accs.append(hit_in_set(
                        probe_probs(scaler, model, features[row_te, layer]),
                        [sets_all[i] for i in row_te]).mean())
            scores[layer] = float(np.mean(accs))
        chosen_layers.append(int(scores.argmax()))
        print(f"  fold {f}: layer {chosen_layers[-1]} "
              f"(inner acc {scores.max():.3f})", flush=True)

    # ---- Primary OOF curve with per-bin inner C selection ------------------
    print("primary probe curve ...", flush=True)
    probe_hits = np.zeros(len(panel))
    probe_prob_store = np.zeros((len(panel), 4))
    chosen_c = np.zeros((N_FOLDS, N_BINS))
    for f in range(N_FOLDS):
        layer = chosen_layers[f]
        train_traj = dec_traj[fold_pairs[f][0]]
        inner_pairs = list(grouped_folds(train_traj, N_INNER,
                                         np.random.default_rng(SEED + 200 + f)))
        for b in range(N_BINS):
            best_c, best_acc = FIXED_C, -1.0
            for c_value in C_GRID:
                accs = []
                for itr, ite in inner_pairs:
                    itr_traj = list(set(train_traj[itr]))
                    ite_traj = list(set(train_traj[ite]))
                    row_tr = np.flatnonzero((bins == b)
                                            & np.isin(groups_step, itr_traj))
                    row_te = np.flatnonzero((bins == b)
                                            & np.isin(groups_step, ite_traj))
                    scaler, model = fit_probe(features[row_tr, layer],
                                              [sets_all[i] for i in row_tr], c_value)
                    accs.append(hit_in_set(
                        probe_probs(scaler, model, features[row_te, layer]),
                        [sets_all[i] for i in row_te]).mean())
                if np.mean(accs) > best_acc:
                    best_acc, best_c = float(np.mean(accs)), c_value
            chosen_c[f, b] = best_c
            tr = rows_of(b, f, True)
            te = rows_of(b, f, False)
            scaler, model = fit_probe(features[tr, layer],
                                      [sets_all[i] for i in tr], best_c)
            probs = probe_probs(scaler, model, features[te, layer])
            probe_prob_store[te] = probs
            probe_hits[te] = hit_in_set(probs, [sets_all[i] for i in te])
        print(f"  fold {f} done (layer {layer}, C {chosen_c[f]})", flush=True)
    panel["probe_hit"] = probe_hits

    # ---- Baselines ---------------------------------------------------------
    chance = float(sets_df.n_optimal[sets_df.n_optimal > 0].mean() / 4.0)
    prior_hits = np.zeros(len(panel))
    for f in range(N_FOLDS):
        counts: Counter = Counter()
        for s in panel[fold_arr != f].drop_duplicates("decision").optimal_actions:
            for a in s:
                counts[a] += 1.0 / len(s)
        top = max(counts, key=counts.get)
        te = np.flatnonzero(fold_arr == f)
        prior_hits[te] = [top in sets_all[i] for i in te]

    perm_rng = np.random.default_rng(SEED + 999)
    unique_dec = panel.drop_duplicates("decision")[["decision", "optimal_actions"]]
    shuffled = unique_dec.optimal_actions.to_numpy().copy()
    perm_rng.shuffle(shuffled)
    perm_map = dict(zip(unique_dec.decision, shuffled))
    perm_sets = [perm_map[d] for d in decision_of]
    perm_hits = np.zeros(len(panel))
    for f in range(N_FOLDS):
        layer = chosen_layers[f]
        for b in range(N_BINS):
            tr = rows_of(b, f, True)
            te = rows_of(b, f, False)
            scaler, model = fit_probe(features[tr, layer],
                                      [perm_sets[i] for i in tr], FIXED_C)
            perm_hits[te] = hit_in_set(probe_probs(scaler, model, features[te, layer]),
                                       [perm_sets[i] for i in te])
    panel["prior_hit"] = prior_hits
    panel["perm_hit"] = perm_hits

    # ---- Secondary metrics -------------------------------------------------
    strict_mask = panel.n_optimal.to_numpy() == 1
    membership = np.zeros((len(panel), 4))
    for i, s in enumerate(sets_all):
        for a in s:
            membership[i, ACTIONS.index(a)] = 1.0

    def macro_auc(probs: np.ndarray, sel: np.ndarray) -> float:
        from sklearn.metrics import roc_auc_score
        aucs = []
        for j in range(4):
            y = membership[sel, j]
            if 0 < y.mean() < 1:
                aucs.append(roc_auc_score(y, probs[sel, j]))
        return float(np.mean(aucs))

    # ---- Curves + trajectory-cluster bootstrap -----------------------------
    print("bootstrap ...", flush=True)
    traj_of_panel = panel.trajectory_id.to_numpy()
    unique_traj = np.array(sorted(set(traj_of_panel)))
    boot_take = np.random.default_rng(SEED).integers(
        0, len(unique_traj), (args.n_boot, len(unique_traj)))
    traj_rows = {t: np.flatnonzero(traj_of_panel == t) for t in unique_traj}

    def cluster_ci(values: np.ndarray, sel: np.ndarray):
        pts = []
        for draw in boot_take:
            rows = np.concatenate([traj_rows[unique_traj[t]] for t in draw])
            mask = sel[rows]
            pts.append(values[rows][mask].mean())
        return float(np.quantile(pts, 0.025)), float(np.quantile(pts, 0.975))

    curve_rows = []
    for b in range(N_BINS):
        sel = bins == b
        p_lo, p_hi = cluster_ci(probe_hits, sel)
        q_lo, q_hi = cluster_ci(q_hits, sel)
        d_lo, d_hi = cluster_ci(probe_hits - q_hits, sel)
        curve_rows.append({
            "bin": b, "center": (b + 0.5) / N_BINS,
            "probe_acc": float(probe_hits[sel].mean()),
            "probe_lo": p_lo, "probe_hi": p_hi,
            "q_acc": float(q_hits[sel].mean()), "q_lo": q_lo, "q_hi": q_hi,
            "diff": float((probe_hits - q_hits)[sel].mean()),
            "diff_lo": d_lo, "diff_hi": d_hi,
            "prior_acc": float(prior_hits[sel].mean()),
            "perm_acc": float(perm_hits[sel].mean()),
            "chance": chance,
            "probe_strict": float(probe_hits[sel & strict_mask].mean()),
            "q_strict": float(q_hits[sel & strict_mask].mean()),
            "probe_multi": float(probe_hits[sel & ~strict_mask].mean()),
            "q_multi": float(q_hits[sel & ~strict_mask].mean()),
            "probe_macro_auc": macro_auc(probe_prob_store, sel),
            "q_macro_auc": macro_auc(q_probs, sel),
            "n": int(sel.sum()),
        })
    curves = pd.DataFrame(curve_rows)
    curves.to_csv(args.out_dir / "curves.csv", index=False)
    panel.drop(columns=["input_token_ids_json", "input_token_mask_json"],
               errors="ignore").assign(
        optimal_actions=[",".join(s) for s in sets_all]) \
        .to_parquet(args.out_dir / "panel.parquet", index=False)

    # ---- Absolute-token appendix (complete-case) ---------------------------
    max_idx = panel.groupby("decision").prefix_token_index.transform("max")
    complete_decisions = set(panel.loc[max_idx >= args.abs_min_tokens, "decision"])
    sub = panel[panel.decision.isin(complete_decisions)]
    abs_rows = []
    for edge in range(args.abs_bin_width, args.abs_min_tokens + 1, args.abs_bin_width):
        nearest = sub.loc[(sub.prefix_token_index - edge).abs()
                          .groupby(sub.decision).idxmin()]
        abs_rows.append({
            "token_index": int(edge),
            "probe_acc": float(nearest.probe_hit.mean()),
            "q_acc": float(nearest.q_hit.mean()),
            "n": int(len(nearest)),
        })
    pd.DataFrame(abs_rows).to_csv(args.out_dir / "absolute_token_curves.csv",
                                  index=False)

    # ---- Figures -----------------------------------------------------------
    print("figures ...", flush=True)
    plot_curves(curves, chosen_layers,
                args.figure_dir / f"optimal_action_probe_vs_q_{args.input_spec}.png")
    plot_heatmap(heat, args.figure_dir
                 / f"optimal_action_probe_layer_heatmap_{args.input_spec}.png")

    n_params = 512 * 4 + 4
    lines = [
        "# Optimal-action linear probes along the CoT, versus q (script 95)",
        "",
        f"- {n_steps} decisions x {N_BINS} relative-position bins; identical step set "
        "and label distribution in every bin by construction; worst within-bin prefix "
        f"reuse {reuse:.3f}.",
        f"- Labels: BFS optimal-action sets; agent in O_s {agent_rate:.4f}; "
        f"|O| distribution {dict(sorted(Counter(sets_df.n_optimal).items()))}; "
        f"uniform chance {chance:.3f}.",
        f"- Probe: linear {n_params}-parameter readout, input spec {args.input_spec}, "
        f"one layer, standardized; soft targets uniform over O_s; per-fold layers "
        f"{chosen_layers} chosen on inner grouped CV (training folds only); per-bin C "
        "by inner CV; evaluation out-of-fold; trajectory-grouped folds throughout.",
        "- Contract: activations-only inputs at the cut; no forced-suffix or future "
        "tokens; frozen backbone; readout well under 1M parameters. The environment-"
        "derived optimal action is the TARGET, not an input.",
        "- Caveat: agent action is in O_s for 98.3% of steps, so optimal-action and "
        "upcoming-action decoding are confounded at ~98%; the informative result is "
        "the probe-vs-q timing gap.",
        "",
        "## Curves (top-1-in-set accuracy, 95% trajectory-cluster CIs)",
        "",
        curves.round(4).to_markdown(index=False),
        "",
        f"## Absolute-token appendix (complete-case steps >= {args.abs_min_tokens} "
        "tokens)",
        "",
        pd.DataFrame(abs_rows).round(4).to_markdown(index=False),
    ]
    save_markdown(args.out_dir / "report.md", "\n".join(lines))
    print(f"done -> {args.out_dir}")
    return 0


def plot_curves(curves: pd.DataFrame, chosen_layers: list[int], path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(7.6, 4.6))
    x = curves.center
    axis.fill_between(x, curves.probe_lo, curves.probe_hi, color=BLUE, alpha=0.15,
                      linewidth=0)
    axis.plot(x, curves.probe_acc, color=BLUE, linewidth=2, marker="o", markersize=4.5,
              label=f"linear probe (layers {sorted(set(chosen_layers))})")
    axis.fill_between(x, curves.q_lo, curves.q_hi, color=ORANGE, alpha=0.15,
                      linewidth=0)
    axis.plot(x, curves.q_acc, color=ORANGE, linewidth=2, marker="o", markersize=4.5,
              label="forced-answer belief q")
    axis.plot(x, curves.prior_acc, color=NEUTRAL, linewidth=1.4, linestyle="--",
              label="prior-only probe")
    axis.plot(x, curves.perm_acc, color=NEUTRAL, linewidth=1.4, linestyle=":",
              label="permutation floor")
    axis.axhline(curves.chance.iloc[0], color=INK, linewidth=1.0, linestyle=":",
                 label="uniform chance")
    axis.set_xlabel("relative CoT position (bin center)", color=INK)
    axis.set_ylabel("top-1-in-optimal-set accuracy", color=INK)
    axis.set_ylim(0.2, 1.0)
    axis.grid(True, color=GRID_C, linewidth=0.8)
    axis.set_axisbelow(True)
    for spine in ("top", "right"):
        axis.spines[spine].set_visible(False)
    axis.tick_params(colors=INK)
    axis.legend(frameon=False, fontsize=8.6, labelcolor=INK, loc="lower right")
    axis.set_title("Decoding a correct action along the chain of thought",
                   color=INK, fontsize=11)
    figure.tight_layout()
    figure.savefig(path, dpi=220)
    plt.close(figure)


def plot_heatmap(heat: np.ndarray, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(6.8, 4.6))
    image = axis.imshow(heat, aspect="auto", origin="lower", cmap="Blues",
                        extent=(0, 1, -0.5, 23.5))
    axis.set_xlabel("relative CoT position", color=INK)
    axis.set_ylabel("layer (block output)", color=INK)
    axis.tick_params(colors=INK)
    bar = figure.colorbar(image, ax=axis)
    bar.set_label("OOF top-1-in-set accuracy", color=INK)
    bar.ax.tick_params(colors=INK)
    axis.set_title("Where in depth the optimal action is decodable, along time",
                   color=INK, fontsize=11)
    figure.tight_layout()
    figure.savefig(path, dpi=220)
    plt.close(figure)


if __name__ == "__main__":
    raise SystemExit(main())
