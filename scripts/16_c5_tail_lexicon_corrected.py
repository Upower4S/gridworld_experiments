#!/usr/bin/env python
"""Corrected C5 tail-lexicon analysis (supersedes the lexical half of script 13).

The 2026-08-11 audit (RUN_HISTORY "C5 lexicon audit") found seven flaws in the
published Figure-9-right analysis. This script fixes each one:

  F1 weighting ambiguity  -> per-step AND character-pooled ratios, everywhere
  F2 class impurity       -> conclusion markers split out of "verification"
  F3 fragile lexicons     -> leave-one-literal-out ranges + top-literal share
  F4 position confound    -> event-aligned densities around T_stab with a
                             shuffled-stabilization placebo, plus a
                             position-matched decile control
  F5 ignored clustering   -> trajectory-cluster bootstrap CIs (95 trajectories)
  F6 action-word ambiguity-> up/down/left/right split by context
                             (answer-naming / movement / descriptive / bare)
  F7 no validation        -> Fightin' Words data-driven contrast vocabulary,
                             embedding soft categories (seed-phrase centroids),
                             and an author-labeled precision sample per class

CODEBOOK (frozen 2026-08-11 before any head/tail statistic was computed; the
action-word context rules received one 20-example spot check, documented in the
report, then were frozen).

Revised regex classes (case-insensitive, word-bounded):
  checking        verify*/check*/confirm*/double-check/re-check/ensure/valid*/
                  consistent/correct/makes sense  (= published verification
                  MINUS conclusion markers)
  concluding      therefore/thus/hence/so the answer/indeed
  self_correction wait/actually/however/hmm/oops/mistake/wrong/instead/
                  re-consider/scratch that  (= published correction MINUS but)
  contrast_but    "but" alone (transparency row; dominated published correction)
  sequencing      then/first/next (split out of published plan)
  planning        plan/path/route/need to/go to/move
  coordinates     literal (r, c) pairs
  grid_nouns      grid/row/col(umn)/cell/room/wall/position/coordinate
  objects         key/door/goal
  action_total    up/down/left/right, any context, plus four disjoint
                  sub-contexts by precedence: action_answer > action_movement >
                  action_descriptive > action_bare

Embedding classes score sentences by cosine to a centroid of frozen seed
phrases (semantic classes only; surface classes stay regex):
  checking / concluding / self_correction / planning / spatial_description /
  answer_naming (seeds in EMBED_SEEDS below).

A class is labeled ROBUST iff its cluster-bootstrap 95% CI excludes 1 and no
leave-one-literal-out variant flips the direction of the effect.

Run from the repo root:
  interp/.venv/bin/python scripts/16_c5_tail_lexicon_corrected.py
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))  # noqa: E402

SEED = 42
WINDOW = 400          # chars per event-aligned window
N_OFFSETS = 5         # windows on each side of T_stab
JUMP_SPAN = 2         # jump statistic uses +-2 windows (800 chars)
MIN_SIDE_CHARS = 200  # published usability filter, kept for continuity

INK = "#333333"
GRID = "#e6e6e6"
BLUE = "#2a78d6"
ORANGE = "#eb6834"
GREEN = "#1baf7a"
PURPLE = "#4a3aa7"
GOLD = "#c9a227"
NEUTRAL = "#9aa0a6"

# ----------------------------------------------------------------------------
# Published lexicons (script 13, verbatim) -- used only for the sanity gate and
# the continuity table.
# ----------------------------------------------------------------------------
ORIGINAL_LEXICONS = {
    "verification": re.compile(
        r"\b(verify|verif\w+|check|checks|checking|confirm\w*|double-?check|re-?check|ensure|"
        r"valid\w*|consistent|correct\b|makes sense|indeed|so the answer|therefore|thus|hence)\b",
        re.IGNORECASE),
    "correction": re.compile(
        r"\b(wait|but|actually|however|hmm|oops|mistake|wrong|instead|re-?consider|scratch that)\b",
        re.IGNORECASE),
    "action_word": re.compile(r"\b(up|down|left|right)\b", re.IGNORECASE),
    "spatial": re.compile(r"\b(key|door|wall|grid|row|col(?:umn)?|room|cell|position|coordinate)\b"
                          r"|\(\s*\d+\s*,\s*\d+\s*\)", re.IGNORECASE),
    "plan": re.compile(r"\b(plan|path|route|then|first|next|need to|go to|move)\b", re.IGNORECASE),
}

# ----------------------------------------------------------------------------
# Revised classes. Each is a list of alternation pieces so leave-one-out can
# recompile the class without a piece. Pieces are ordered specific-first so
# per-piece attribution (top-literal share) is well defined.
# ----------------------------------------------------------------------------
REVISED_PIECES: dict[str, list[str]] = {
    "checking": [r"double-?check\w*", r"re-?check\w*", "checking", "checks", "check",
                 r"verif\w+", r"confirm\w*", "ensure", r"valid\w*", "consistent",
                 r"correct\b", "makes sense"],
    "concluding": ["therefore", "thus", "hence", "so the answer", "indeed"],
    "self_correction": ["wait", "actually", "however", "hmm", "oops", "mistake",
                        "wrong", "instead", r"re-?consider", "scratch that"],
    "contrast_but": ["but"],
    "sequencing": ["then", "first", "next"],
    "planning": ["plan", "path", "route", "need to", "go to", "move"],
    "grid_nouns": ["grid", "row", r"col(?:umn)?", "cell", "room", "wall",
                   "position", "coordinate"],
    "objects": ["key", "door", "goal"],
}
COORD_PATTERN = re.compile(r"\(\s*\d+\s*,\s*\d+\s*\)")
ACTION_PATTERN = re.compile(r"\b(up|down|left|right)\b", re.IGNORECASE)

# Action-word context rules (precedence: answer > movement > descriptive > bare).
ANSWER_BEFORE = re.compile(r"(?:answer|action|final|output)\W{0,12}$", re.IGNORECASE)
MOVE_BEFORE = re.compile(r"(?:\bmove\w*|\bgo(?:es|ing)?|\bgone|\bwent|\bhead\w*|\bstep\w*|\bwalk\w*)\s*$",
                         re.IGNORECASE)
TO_THE_BEFORE = re.compile(r"\b(?:to|on|at|in)\s+the\s*$", re.IGNORECASE)
DESC_AFTER = re.compile(r"^\s*(?:column|row|side|of|from|wall|cell|edge|half|corner|neighbor)\b",
                        re.IGNORECASE)
QUOTES = "\"'`"

REVISED_CLASS_NAMES = [*REVISED_PIECES.keys(), "coordinates", "action_total",
                       "action_answer", "action_movement", "action_descriptive",
                       "action_bare"]

# Semantic classes scored by embedding similarity to frozen seed phrases.
EMBED_SEEDS = {
    "checking": [
        "Let me double-check that.",
        "Verify that this is correct.",
        "Check the position again to be sure.",
        "Make sure the path is valid.",
        "Confirm that the move is allowed.",
        "Yes, that matches what I computed earlier.",
    ],
    "concluding": [
        "Therefore the answer is clear.",
        "Thus the move is decided.",
        "So the final answer is up.",
        "Hence we should go down.",
        "In conclusion, the action is left.",
    ],
    "self_correction": [
        "Wait, that is wrong.",
        "Actually, I made a mistake.",
        "Hmm, let me reconsider.",
        "Oops, that cell is a wall.",
        "No wait, that is not correct.",
    ],
    "planning": [
        "First I need to plan a route to the key.",
        "The path goes through the door.",
        "I need to reach the goal from here.",
        "Plan: go to the key, then to the door.",
        "We must find a way around the wall.",
    ],
    "spatial_description": [
        "The agent is at row 3, column 5.",
        "The wall is on the left side.",
        "The key is in the top room.",
        "The door is below the agent.",
        "The grid is nine by nine.",
    ],
    "answer_naming": [
        "The action is right.",
        "Answer: down.",
        "The move to output is left.",
        "Final action: up.",
        "I will answer with the action up.",
    ],
}

FW_ALPHA0 = 500.0
FW_MIN_COUNT = 10
WORD_RE = re.compile(r"[a-z']+")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=Path,
                        default=REPO_ROOT / "outputs/conjectures/c5_tail/steps.parquet")
    parser.add_argument("--published", type=Path,
                        default=REPO_ROOT / "outputs/conjectures/c5_tail/head_tail_lexicon.csv")
    parser.add_argument("--prefixes", type=Path, default=REPO_ROOT / "outputs/prefixes.parquet")
    parser.add_argument("--out_dir", type=Path,
                        default=REPO_ROOT / "outputs/conjectures/c5_tail_corrected")
    parser.add_argument("--figure_dir", type=Path, default=REPO_ROOT / "figures/c5_corrected")
    parser.add_argument("--paper_figure_dir", type=Path, default=REPO_ROOT / "paper/figures")
    parser.add_argument("--n_boot", type=int, default=2000)
    parser.add_argument("--n_boot_curves", type=int, default=500)
    parser.add_argument("--n_boot_fw", type=int, default=500)
    parser.add_argument("--n_placebo", type=int, default=200)
    parser.add_argument("--skip_embedding", action="store_true",
                        help="skip the sentence-embedding stage (dev runs)")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_steps(args: argparse.Namespace) -> pd.DataFrame:
    import pyarrow.parquet as pq

    steps = pd.read_parquet(args.steps, columns=[
        "trajectory_id", "step_id", "agent_action", "num_tokens",
        "stab_prefix_id", "final_prefix_id", "stab_progress", "late_reversal"])
    needed = sorted(set(steps.stab_prefix_id) | set(steps.final_prefix_id))
    texts = pq.read_table(args.prefixes, columns=["prefix_id", "analysis_prefix_text"],
                          filters=[("prefix_id", "in", needed)]).to_pandas()
    text_of = dict(zip(texts.prefix_id, texts.analysis_prefix_text))

    heads, fulls, violations = [], [], 0
    for row in steps.itertuples():
        head = text_of.get(row.stab_prefix_id) or ""
        full = text_of.get(row.final_prefix_id) or ""
        if not full.startswith(head):
            violations += 1
        heads.append(head)
        fulls.append(full)
    steps = steps.assign(head_text=heads, full_text=fulls)
    steps["stab_pos"] = steps.head_text.str.len()
    steps["full_len"] = steps.full_text.str.len()
    steps["tail_len"] = steps.full_len - steps.stab_pos
    steps["usable"] = (steps.stab_pos > MIN_SIDE_CHARS) & (steps.tail_len > MIN_SIDE_CHARS)
    steps.attrs["prefix_violations"] = violations
    steps.attrs["n_stab_zero"] = int((steps.stab_pos == 0).sum())
    if violations:
        raise AssertionError(f"{violations} steps where full text does not start with head text")
    return steps


def sanity_gate(steps: pd.DataFrame, published_csv: Path) -> pd.DataFrame:
    """Reproduce the published per-step ratios exactly (slice counting, as in 13)."""
    usable = steps[steps.usable]
    rows = []
    for name, pattern in ORIGINAL_LEXICONS.items():
        head_d = np.array([1000 * len(pattern.findall(h)) / len(h) for h in usable.head_text])
        tail_text = [f[len(h):] for h, f in zip(usable.head_text, usable.full_text)]
        tail_d = np.array([1000 * len(pattern.findall(t)) / len(t) for t in tail_text])
        rows.append({"lexicon": name,
                     "head_density": float(head_d.mean()),
                     "tail_density": float(tail_d.mean()),
                     "tail_over_head": float(tail_d.mean() / head_d.mean())})
    got = pd.DataFrame(rows).set_index("lexicon")
    want = pd.read_csv(published_csv).set_index("lexicon")
    for name in got.index:
        for col in ("head_density", "tail_density", "tail_over_head"):
            if not np.isclose(got.loc[name, col], want.loc[name, col], rtol=0, atol=1e-9):
                raise AssertionError(
                    f"sanity gate: {name}.{col} = {got.loc[name, col]!r} != published "
                    f"{want.loc[name, col]!r}")
    return got.reset_index()


# ---------------------------------------------------------------------------
# Match positions for the revised classes (positions on the FULL text; a hit
# belongs to the head iff its start position < stab_pos)
# ---------------------------------------------------------------------------

def compile_class(pieces: list[str]) -> re.Pattern:
    return re.compile(r"\b(?:" + "|".join(pieces) + r")\b", re.IGNORECASE)


def classify_action_hit(text: str, start: int, end: int) -> str:
    before = text[max(0, start - 18):start]
    after = text[end:end + 18]
    quoted = (start > 0 and text[start - 1] in QUOTES
              and end < len(text) and text[end] in QUOTES)
    if quoted or ANSWER_BEFORE.search(before):
        return "action_answer"
    if MOVE_BEFORE.search(before):
        return "action_movement"
    if DESC_AFTER.match(after) or TO_THE_BEFORE.search(before):
        return "action_descriptive"
    return "action_bare"


def collect_positions(steps: pd.DataFrame) -> dict[str, list[np.ndarray]]:
    """Per class, one sorted array of match start positions per step (full text)."""
    class_patterns = {name: compile_class(pieces) for name, pieces in REVISED_PIECES.items()}
    positions: dict[str, list[np.ndarray]] = {name: [] for name in REVISED_CLASS_NAMES}
    for full in steps.full_text:
        for name, pattern in class_patterns.items():
            positions[name].append(np.fromiter(
                (m.start() for m in pattern.finditer(full)), dtype=np.int64))
        positions["coordinates"].append(np.fromiter(
            (m.start() for m in COORD_PATTERN.finditer(full)), dtype=np.int64))
        subs: dict[str, list[int]] = {k: [] for k in
                                      ("action_answer", "action_movement",
                                       "action_descriptive", "action_bare")}
        total = []
        for m in ACTION_PATTERN.finditer(full):
            total.append(m.start())
            subs[classify_action_hit(full, m.start(), m.end())].append(m.start())
        positions["action_total"].append(np.asarray(total, dtype=np.int64))
        for k, v in subs.items():
            positions[k].append(np.asarray(v, dtype=np.int64))
    return positions


def spot_check_actions(steps: pd.DataFrame, rng: np.random.Generator) -> list[str]:
    """The one documented tuning round for the action-context rules."""
    examples = []
    sample = steps.sample(n=min(6, len(steps)), random_state=SEED)
    for full in sample.full_text:
        hits = list(ACTION_PATTERN.finditer(full))
        if not hits:
            continue
        for m in rng.choice(len(hits), size=min(4, len(hits)), replace=False):
            match = hits[int(m)]
            label = classify_action_hit(full, match.start(), match.end())
            ctx = full[max(0, match.start() - 40):match.end() + 40].replace("\n", " ")
            examples.append(f"[{label:18s}] ...{ctx}...")
        if len(examples) >= 20:
            break
    return examples[:20]


# ---------------------------------------------------------------------------
# Head/tail statistics with cluster bootstrap and LOO
# ---------------------------------------------------------------------------

def head_tail_counts(positions: dict[str, list[np.ndarray]],
                     steps: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    n = len(steps)
    names = REVISED_CLASS_NAMES
    head = np.zeros((n, len(names)), dtype=np.float64)
    tail = np.zeros((n, len(names)), dtype=np.float64)
    stab = steps.stab_pos.to_numpy()
    for j, name in enumerate(names):
        for i, pos in enumerate(positions[name]):
            k = int(np.searchsorted(pos, stab[i]))
            head[i, j] = k
            tail[i, j] = len(pos) - k
    return head, tail


def traj_groups_of(steps: pd.DataFrame) -> list[np.ndarray]:
    codes = pd.Categorical(steps.trajectory_id).codes
    return [np.flatnonzero(codes == c) for c in range(codes.max() + 1)]


def bootstrap_ratios(head_hits: np.ndarray, tail_hits: np.ndarray,
                     head_len: np.ndarray, tail_len: np.ndarray,
                     groups: list[np.ndarray], n_boot: int,
                     rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Returns (n_boot, n_classes) arrays of per-step and pooled ratios."""
    n_traj = len(groups)
    per_step = np.empty((n_boot, head_hits.shape[1]))
    pooled = np.empty_like(per_step)
    head_d = 1000.0 * head_hits / head_len[:, None]
    tail_d = 1000.0 * tail_hits / tail_len[:, None]
    for b in range(n_boot):
        take = rng.integers(0, n_traj, n_traj)
        idx = np.concatenate([groups[t] for t in take])
        per_step[b] = tail_d[idx].mean(axis=0) / head_d[idx].mean(axis=0)
        pooled[b] = ((tail_hits[idx].sum(axis=0) / tail_len[idx].sum())
                     / (head_hits[idx].sum(axis=0) / head_len[idx].sum()))
    return per_step, pooled


def revised_stats(steps: pd.DataFrame, positions: dict[str, list[np.ndarray]],
                  args: argparse.Namespace, rng: np.random.Generator
                  ) -> tuple[pd.DataFrame, pd.DataFrame]:
    from scipy.stats import wilcoxon

    usable_idx = np.flatnonzero(steps.usable.to_numpy())
    sub = steps.iloc[usable_idx]
    pos_u = {name: [positions[name][i] for i in usable_idx] for name in REVISED_CLASS_NAMES}
    head_hits, tail_hits = head_tail_counts(pos_u, sub)
    head_len = sub.stab_pos.to_numpy(dtype=float)
    tail_len = sub.tail_len.to_numpy(dtype=float)
    groups = traj_groups_of(sub)

    head_d = 1000.0 * head_hits / head_len[:, None]
    tail_d = 1000.0 * tail_hits / tail_len[:, None]
    boot_step, boot_pool = bootstrap_ratios(head_hits, tail_hits, head_len, tail_len,
                                            groups, args.n_boot, rng)

    rows = []
    for j, name in enumerate(REVISED_CLASS_NAMES):
        hd, td = head_d[:, j], tail_d[:, j]
        diffs = td - hd
        nonzero = diffs != 0
        stat = wilcoxon(td, hd) if nonzero.any() else None
        per_step = float(td.mean() / hd.mean()) if hd.mean() > 0 else float("nan")
        pooled = (float((tail_hits[:, j].sum() / tail_len.sum())
                        / (head_hits[:, j].sum() / head_len.sum()))
                  if head_hits[:, j].sum() > 0 else float("nan"))
        rows.append({
            "class": name,
            "head_density": float(hd.mean()),
            "tail_density": float(td.mean()),
            "per_step_ratio": per_step,
            "per_step_ci_lo": float(np.nanquantile(boot_step[:, j], 0.025)),
            "per_step_ci_hi": float(np.nanquantile(boot_step[:, j], 0.975)),
            "pooled_ratio": pooled,
            "pooled_ci_lo": float(np.nanquantile(boot_pool[:, j], 0.025)),
            "pooled_ci_hi": float(np.nanquantile(boot_pool[:, j], 0.975)),
            "frac_tail_higher": float(np.mean(td > hd)),
            "frac_ties": float(np.mean(diffs == 0)),
            "wilcoxon_p": float(stat.pvalue) if stat is not None else float("nan"),
            "head_hits": int(head_hits[:, j].sum()),
            "tail_hits": int(tail_hits[:, j].sum()),
        })
    stats = pd.DataFrame(rows)

    # Leave-one-literal-out (recompile the alternation without each piece) plus
    # per-piece attribution for the top-literal share.
    loo_rows = []
    stab = sub.stab_pos.to_numpy()
    for name, pieces in REVISED_PIECES.items():
        if len(pieces) < 2:
            continue
        full_ratio = float(stats.loc[stats["class"] == name, "per_step_ratio"].iloc[0])
        base_sign = np.sign(np.log(full_ratio))
        piece_pat = [re.compile(r"\b(?:" + p + r")\b", re.IGNORECASE) for p in pieces]
        attrib_head = Counter()
        attrib_tail = Counter()
        for i, full in enumerate(sub.full_text):
            for p_label, pat in zip(pieces, piece_pat):
                for m in pat.finditer(full):
                    (attrib_head if m.start() < stab[i] else attrib_tail)[p_label] += 1
        total_tail = sum(attrib_tail.values()) or 1
        for drop in pieces:
            kept = [p for p in pieces if p != drop]
            pat = compile_class(kept)
            hh = np.array([sum(m.start() < stab[i] for m in pat.finditer(full))
                           for i, full in enumerate(sub.full_text)], dtype=float)
            tt = np.array([sum(1 for _ in pat.finditer(full)) for full in sub.full_text],
                          dtype=float) - hh
            hd = 1000 * hh / head_len
            td = 1000 * tt / tail_len
            ratio = float(td.mean() / hd.mean()) if hd.mean() > 0 else float("nan")
            loo_rows.append({
                "class": name, "dropped_literal": drop, "loo_ratio": ratio,
                "sign_flip": bool(np.isfinite(ratio) and np.sign(np.log(ratio)) != base_sign),
                "literal_tail_share": attrib_tail[drop] / total_tail,
                "literal_head_hits": attrib_head[drop],
                "literal_tail_hits": attrib_tail[drop],
            })
    loo = pd.DataFrame(loo_rows)

    flip = loo.groupby("class").sign_flip.any() if len(loo) else pd.Series(dtype=bool)
    robust = []
    for name, lo, hi in zip(stats["class"], stats.per_step_ci_lo, stats.per_step_ci_hi):
        ci_excludes_1 = bool(lo > 1.0) or bool(hi < 1.0)
        robust.append(bool(ci_excludes_1 and not bool(flip.get(name, False))))
    stats["robust"] = robust
    if len(loo):
        top_share = loo.groupby("class").literal_tail_share.max()
        stats["top_literal_share"] = stats["class"].map(top_share)
        loo_range = loo.groupby("class").loo_ratio.agg(["min", "max"])
        stats["loo_ratio_min"] = stats["class"].map(loo_range["min"])
        stats["loo_ratio_max"] = stats["class"].map(loo_range["max"])
    return stats, loo


# ---------------------------------------------------------------------------
# Fightin' Words (Monroe et al. 2008)
# ---------------------------------------------------------------------------

def fw_tokenize(text: str) -> list[str]:
    text = COORD_PATTERN.sub(" coordpair ", text.lower())
    text = re.sub(r"\d+", " num ", text)
    return WORD_RE.findall(text)


def fightin_words(steps: pd.DataFrame, args: argparse.Namespace,
                  rng: np.random.Generator) -> pd.DataFrame:
    sub = steps[steps.usable]
    traj_codes = pd.Categorical(sub.trajectory_id).codes
    n_traj = traj_codes.max() + 1
    head_counts: list[Counter] = [Counter() for _ in range(n_traj)]
    tail_counts: list[Counter] = [Counter() for _ in range(n_traj)]
    for code, head, full in zip(traj_codes, sub.head_text, sub.full_text):
        for counts, text in ((head_counts[code], head), (tail_counts[code], full[len(head):])):
            tokens = fw_tokenize(text)
            counts.update(tokens)
            counts.update(f"{a} {b}" for a, b in zip(tokens, tokens[1:]))

    pooled: Counter = Counter()
    for c in head_counts:
        pooled.update(c)
    for c in tail_counts:
        pooled.update(c)
    vocab = sorted(t for t, c in pooled.items() if c >= FW_MIN_COUNT)
    index = {t: i for i, t in enumerate(vocab)}
    A_head = np.zeros((n_traj, len(vocab)), dtype=np.int64)
    A_tail = np.zeros_like(A_head)
    for k, (hc, tc) in enumerate(zip(head_counts, tail_counts)):
        for t, c in hc.items():
            if t in index:
                A_head[k, index[t]] = c
        for t, c in tc.items():
            if t in index:
                A_tail[k, index[t]] = c

    def z_scores(head_vec: np.ndarray, tail_vec: np.ndarray) -> np.ndarray:
        pool = head_vec + tail_vec
        alpha = FW_ALPHA0 * pool / pool.sum()
        n_head, n_tail = head_vec.sum(), tail_vec.sum()
        with np.errstate(divide="ignore", invalid="ignore"):
            lh = np.log((head_vec + alpha) / (n_head + FW_ALPHA0 - head_vec - alpha))
            lt = np.log((tail_vec + alpha) / (n_tail + FW_ALPHA0 - tail_vec - alpha))
            z = (lt - lh) / np.sqrt(1.0 / (head_vec + alpha) + 1.0 / (tail_vec + alpha))
        z[pool == 0] = 0.0  # term absent from a bootstrap resample: no evidence
        return z

    head_tot = A_head.sum(axis=0).astype(float)
    tail_tot = A_tail.sum(axis=0).astype(float)
    z = z_scores(head_tot, tail_tot)

    order = np.argsort(z)
    top_rising = set(order[-50:])
    top_falling = set(order[:50])
    stability = np.zeros((len(vocab), 2))
    for _ in range(args.n_boot_fw):
        take = rng.integers(0, n_traj, n_traj)
        zb = z_scores(A_head[take].sum(axis=0).astype(float),
                      A_tail[take].sum(axis=0).astype(float))
        ob = np.argsort(zb)
        stability[list(set(ob[-50:]) & top_rising), 0] += 1
        stability[list(set(ob[:50]) & top_falling), 1] += 1
    stability /= args.n_boot_fw

    class_pats = {name: compile_class(p) for name, p in REVISED_PIECES.items()}
    def term_class(term: str) -> str:
        if "coordpair" in term:
            return "coordinates"
        if ACTION_PATTERN.fullmatch(term):
            return "action_total"
        for name, pat in class_pats.items():
            if pat.search(term):
                return name
        return ""

    table = pd.DataFrame({
        "term": vocab,
        "kind": ["bigram" if " " in t else "unigram" for t in vocab],
        "head_count": head_tot.astype(int),
        "tail_count": tail_tot.astype(int),
        "z": z,
        "top50_rising_stability": stability[:, 0],
        "top50_falling_stability": stability[:, 1],
    })
    table["revised_class"] = [term_class(t) for t in vocab]
    return table.sort_values("z", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Embedding soft categories
# ---------------------------------------------------------------------------

SENT_SPLIT = re.compile(r"(?<=[.?!])\s+|\n+")


def split_sentences(text: str) -> list[tuple[int, int, str]]:
    spans = []
    start = 0
    for m in SENT_SPLIT.finditer(text):
        spans.append((start, m.start()))
        start = m.end()
    spans.append((start, len(text)))
    out = []
    for a, b in spans:
        raw = text[a:b]
        stripped = raw.strip()
        if len(stripped) >= 8:
            lead = len(raw) - len(raw.lstrip())
            out.append((a + lead, a + lead + len(stripped), stripped))
    return out


def embedding_scores(steps: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    import torch
    from sentence_transformers import SentenceTransformer

    torch.manual_seed(SEED)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=device)
    model.eval()

    sub = steps[steps.usable]
    records = []
    for step_row, (traj, sid, full, stab) in enumerate(zip(
            sub.trajectory_id, sub.step_id, sub.full_text, sub.stab_pos)):
        for a, b, text in split_sentences(full):
            mid = (a + b) / 2.0
            records.append((traj, int(sid), step_row, a, b, mid,
                            "tail" if mid >= stab else "head", text))
    frame = pd.DataFrame(records, columns=[
        "trajectory_id", "step_id", "step_row", "start", "end", "mid", "side", "text"])

    unique_texts = frame.text.drop_duplicates().tolist()
    with torch.no_grad():
        emb = model.encode(unique_texts, batch_size=512, normalize_embeddings=True,
                           show_progress_bar=False)
    emb_of = {t: e for t, e in zip(unique_texts, emb)}

    for name, seeds in EMBED_SEEDS.items():
        with torch.no_grad():
            seed_emb = model.encode(seeds, normalize_embeddings=True, show_progress_bar=False)
        centroid = seed_emb.mean(axis=0)
        centroid /= np.linalg.norm(centroid)
        frame[f"score_{name}"] = [float(emb_of[t] @ centroid) for t in frame.text]
    return frame


def embedding_stats(frame: pd.DataFrame, steps: pd.DataFrame,
                    args: argparse.Namespace, rng: np.random.Generator) -> pd.DataFrame:
    from scipy.stats import mannwhitneyu, wilcoxon

    sub = steps[steps.usable].reset_index(drop=True)
    groups = traj_groups_of(sub)
    rows = []
    score_cols = [c for c in frame.columns if c.startswith("score_")]
    per_step = frame.groupby(["step_row", "side"])[score_cols].mean().unstack("side")
    complete = per_step.dropna()
    step_rows = complete.index.to_numpy()
    keep = np.isin(np.arange(len(sub)), step_rows)
    row_of = {s: i for i, s in enumerate(step_rows)}

    for col in score_cols:
        name = col.removeprefix("score_")
        head_m = complete[(col, "head")].to_numpy()
        tail_m = complete[(col, "tail")].to_numpy()
        diff = tail_m - head_m
        boot = np.full(args.n_boot_curves, np.nan)
        for b in range(args.n_boot_curves):
            take = rng.integers(0, len(groups), len(groups))
            idx = [row_of[s] for t in take for s in groups[t] if keep[s]]
            if idx:
                boot[b] = diff[idx].mean()
        auc = mannwhitneyu(frame.loc[frame.side == "tail", col],
                           frame.loc[frame.side == "head", col])
        n_sent = (len(frame[frame.side == "tail"]), len(frame[frame.side == "head"]))
        rows.append({
            "class": name,
            "head_mean": float(head_m.mean()),
            "tail_mean": float(tail_m.mean()),
            "mean_diff": float(diff.mean()),
            "diff_ci_lo": float(np.nanquantile(boot, 0.025)),
            "diff_ci_hi": float(np.nanquantile(boot, 0.975)),
            "rank_biserial": float((np.sum(diff > 0) - np.sum(diff < 0)) / len(diff)),
            "sentence_auc": float(auc.statistic / (n_sent[0] * n_sent[1])),
            "wilcoxon_p": float(wilcoxon(tail_m, head_m).pvalue),
            "n_steps": int(len(diff)),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Event alignment + placebo (shared machinery)
# ---------------------------------------------------------------------------

def placebo_positions(sub: pd.DataFrame, n_draws: int,
                      rng: np.random.Generator) -> np.ndarray:
    """(n_draws, n_steps) placebo stabilization positions: each step gets the
    relative stab position of another step from the same chain-length quintile."""
    lens = sub.full_len.to_numpy()
    rel = (sub.stab_pos / sub.full_len).to_numpy()
    quint = pd.qcut(lens, 5, labels=False, duplicates="drop")
    draws = np.empty((n_draws, len(sub)), dtype=np.int64)
    for q in np.unique(quint):
        members = np.flatnonzero(quint == q)
        for d in range(n_draws):
            donor = members[rng.integers(0, len(members), len(members))]
            same = donor == members
            if same.any():
                donor[same] = members[(np.flatnonzero(same) + 1) % len(members)]
            draws[d, members] = np.round(rel[donor] * lens[members]).astype(np.int64)
    return draws


def window_counts(pos: np.ndarray, lo: int, hi: int) -> int:
    return int(np.searchsorted(pos, hi) - np.searchsorted(pos, lo))


def event_aligned(steps: pd.DataFrame, positions: dict[str, list[np.ndarray]],
                  frame: pd.DataFrame | None, args: argparse.Namespace,
                  rng: np.random.Generator
                  ) -> tuple[pd.DataFrame, pd.DataFrame]:
    usable_idx = np.flatnonzero(steps.usable.to_numpy())
    sub = steps.iloc[usable_idx].reset_index(drop=True)
    groups = traj_groups_of(sub)
    stab = sub.stab_pos.to_numpy()
    lens = sub.full_len.to_numpy()
    n = len(sub)
    classes = REVISED_CLASS_NAMES
    pos_u = {name: [positions[name][i] for i in usable_idx] for name in classes}

    offsets = np.arange(-N_OFFSETS, N_OFFSETS)
    dens = np.full((n, len(classes), len(offsets)), np.nan)
    for i in range(n):
        for k, off in enumerate(offsets):
            lo = stab[i] + off * WINDOW
            hi = lo + WINDOW
            if lo < 0 or hi > lens[i]:
                continue
            for j, name in enumerate(classes):
                dens[i, j, k] = 1000.0 * window_counts(pos_u[name][i], lo, hi) / WINDOW

    curve_rows = []
    boot_idx = [np.concatenate([groups[t] for t in rng.integers(0, len(groups), len(groups))])
                for _ in range(args.n_boot_curves)]
    for j, name in enumerate(classes):
        for k, off in enumerate(offsets):
            col = dens[:, j, k]
            valid = ~np.isnan(col)
            if valid.sum() < 20:
                continue
            boots = [np.nanmean(col[idx]) for idx in boot_idx]
            curve_rows.append({
                "class": name, "offset": int(off), "mean_density": float(np.nanmean(col)),
                "ci_lo": float(np.nanquantile(boots, 0.025)),
                "ci_hi": float(np.nanquantile(boots, 0.975)),
                "n_steps": int(valid.sum()),
            })
    curves = pd.DataFrame(curve_rows)

    # Jump statistic vs shuffled-stabilization placebo.
    span = JUMP_SPAN * WINDOW
    draws = placebo_positions(sub, args.n_placebo, rng)

    def mean_jump(stab_vec: np.ndarray, j: int) -> float:
        vals = []
        pos_list = pos_u[classes[j]]
        for i in range(n):
            p = stab_vec[i]
            if p < span or p + span > lens[i]:
                continue
            post = window_counts(pos_list[i], p, p + span)
            pre = window_counts(pos_list[i], p - span, p)
            vals.append(1000.0 * (post - pre) / span)
        return float(np.mean(vals)) if vals else float("nan")

    def empirical_p(real: float, placebo: np.ndarray) -> float:
        hi = (1 + np.sum(placebo >= real)) / (len(placebo) + 1)
        lo = (1 + np.sum(placebo <= real)) / (len(placebo) + 1)
        return float(min(1.0, 2.0 * min(hi, lo)))

    jump_rows = []
    for j, name in enumerate(classes):
        real = mean_jump(stab, j)
        placebo = np.array([mean_jump(draws[d], j) for d in range(args.n_placebo)])
        p_two = empirical_p(real, placebo)
        jump_rows.append({
            "class": name, "kind": "regex", "real_jump": real,
            "placebo_mean": float(np.nanmean(placebo)),
            "placebo_lo": float(np.nanquantile(placebo, 0.025)),
            "placebo_hi": float(np.nanquantile(placebo, 0.975)),
            "p_two_sided": p_two,
        })

    # Same jump for embedding soft scores (prefix-sum over sentence mids).
    if frame is not None:
        score_cols = [c for c in frame.columns if c.startswith("score_")]
        by_step: dict[int, tuple[np.ndarray, dict[str, np.ndarray]]] = {}
        for step_row, g in frame.sort_values("mid").groupby("step_row"):
            mids = g.mid.to_numpy()
            sums = {c: np.concatenate([[0.0], np.cumsum(g[c].to_numpy())]) for c in score_cols}
            by_step[int(step_row)] = (mids, sums)

        def soft_jump(stab_vec: np.ndarray, col: str) -> float:
            vals = []
            for i in range(n):
                if i not in by_step:
                    continue
                p = stab_vec[i]
                if p < span or p + span > lens[i]:
                    continue
                mids, sums = by_step[i]
                a, b_, c_ = (np.searchsorted(mids, p - span), np.searchsorted(mids, p),
                             np.searchsorted(mids, p + span))
                if b_ == a or c_ == b_:
                    continue
                pre = (sums[col][b_] - sums[col][a]) / (b_ - a)
                post = (sums[col][c_] - sums[col][b_]) / (c_ - b_)
                vals.append(post - pre)
            return float(np.mean(vals)) if vals else float("nan")

        for col in score_cols:
            name = col.removeprefix("score_")
            real = soft_jump(stab, col)
            placebo = np.array([soft_jump(draws[d], col) for d in range(args.n_placebo)])
            p_two = empirical_p(real, placebo)
            jump_rows.append({
                "class": name, "kind": "embedding", "real_jump": real,
                "placebo_mean": float(np.nanmean(placebo)),
                "placebo_lo": float(np.nanquantile(placebo, 0.025)),
                "placebo_hi": float(np.nanquantile(placebo, 0.975)),
                "p_two_sided": p_two,
            })
    return curves, pd.DataFrame(jump_rows)


# ---------------------------------------------------------------------------
# Position-matched decile control
# ---------------------------------------------------------------------------

def decile_control(steps: pd.DataFrame, positions: dict[str, list[np.ndarray]],
                   args: argparse.Namespace, rng: np.random.Generator) -> pd.DataFrame:
    usable_idx = np.flatnonzero(steps.usable.to_numpy())
    sub = steps.iloc[usable_idx].reset_index(drop=True)
    groups = traj_groups_of(sub)
    stab = sub.stab_pos.to_numpy()
    lens = sub.full_len.to_numpy()
    classes = REVISED_CLASS_NAMES
    pos_u = {name: [positions[name][i] for i in usable_idx] for name in classes}
    n = len(sub)

    dens = np.zeros((n, len(classes), 10))
    post = np.zeros((n, 10), dtype=bool)
    for i in range(n):
        bounds = np.round(np.linspace(0, lens[i], 11)).astype(int)
        for d in range(10):
            lo, hi = bounds[d], bounds[d + 1]
            width = max(hi - lo, 1)
            post[i, d] = lo >= stab[i]
            for j in range(len(classes)):
                dens[i, j, d] = 1000.0 * window_counts(pos_u[classes[j]][i], lo, hi) / width

    boot_idx = [np.concatenate([groups[t] for t in rng.integers(0, len(groups), len(groups))])
                for _ in range(args.n_boot_curves)]
    rows = []
    for d in range(10):
        pre_mask = ~post[:, d]
        post_mask = post[:, d]
        if pre_mask.sum() < 30 or post_mask.sum() < 30:
            continue
        for j, name in enumerate(classes):
            col = dens[:, j, d]
            diff = col[post_mask].mean() - col[pre_mask].mean()
            boots = []
            for idx in boot_idx:
                pm, qm = post_mask[idx], ~post_mask[idx]
                if pm.sum() and qm.sum():
                    boots.append(col[idx][pm].mean() - col[idx][qm].mean())
            rows.append({
                "decile": d, "class": name,
                "pre_density": float(col[pre_mask].mean()),
                "post_density": float(col[post_mask].mean()),
                "post_minus_pre": float(diff),
                "ci_lo": float(np.quantile(boots, 0.025)),
                "ci_hi": float(np.quantile(boots, 0.975)),
                "n_pre": int(pre_mask.sum()), "n_post": int(post_mask.sum()),
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Precision sample
# ---------------------------------------------------------------------------

def precision_sample(steps: pd.DataFrame, positions: dict[str, list[np.ndarray]],
                     frame: pd.DataFrame | None,
                     rng: np.random.Generator) -> pd.DataFrame:
    usable_idx = np.flatnonzero(steps.usable.to_numpy())
    sub = steps.iloc[usable_idx].reset_index(drop=True)
    stab = sub.stab_pos.to_numpy()
    rows = []
    for name in REVISED_CLASS_NAMES:
        if name == "action_total":
            continue
        # positions were collected on the full steps frame; re-index via usable_idx
        hits = [(i, int(p)) for i in range(len(sub))
                for p in positions[name][int(usable_idx[i])]]
        for side in ("head", "tail"):
            pool = [(i, p) for i, p in hits if (p < stab[i]) == (side == "head")]
            if not pool:
                continue
            take = rng.choice(len(pool), size=min(20, len(pool)), replace=False)
            for t in take:
                i, p = pool[int(t)]
                text = sub.full_text.iloc[i]
                lo, hi = max(0, p - 60), min(len(text), p + 70)
                snippet = text[lo:p] + "[[" + text[p:p + 12] + "]]" + text[p + 12:hi]
                rows.append({"method": "regex", "class": name, "side": side,
                             "snippet": snippet.replace("\n", " "), "label": ""})
    if frame is not None:
        for col in [c for c in frame.columns if c.startswith("score_")]:
            name = col.removeprefix("score_")
            for side in ("head", "tail"):
                part = frame[frame.side == side]
                cut = part[col].quantile(0.9)
                top = part[part[col] >= cut]
                take = top.sample(n=min(20, len(top)), random_state=SEED)
                for text in take.text:
                    rows.append({"method": "embedding", "class": name, "side": side,
                                 "snippet": text.replace("\n", " ")[:200], "label": ""})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def style(axis, xlabel: str = "", ylabel: str = "") -> None:
    axis.grid(True, color=GRID, linewidth=0.8)
    axis.set_axisbelow(True)
    axis.tick_params(colors=INK, labelsize=9)
    for spine in ("top", "right"):
        axis.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        axis.spines[spine].set_color(INK)
    if xlabel:
        axis.set_xlabel(xlabel, color=INK, fontsize=10)
    if ylabel:
        axis.set_ylabel(ylabel, color=INK, fontsize=10)


CLASS_COLORS = {
    "checking": BLUE, "concluding": PURPLE, "self_correction": ORANGE,
    "coordinates": GREEN, "grid_nouns": GOLD, "sequencing": GOLD,
    "planning": GOLD, "objects": GOLD, "contrast_but": NEUTRAL,
    "action_total": GREEN, "": NEUTRAL,
}

RATIO_PLOT_CLASSES = ["concluding", "checking", "action_answer", "action_movement",
                      "sequencing", "planning", "coordinates", "objects",
                      "action_descriptive", "grid_nouns", "contrast_but",
                      "self_correction"]

RATIO_LABELS = {
    "concluding": "concluding (thus/therefore)",
    "checking": "checking (verify/confirm)",
    "action_answer": "action word: answer-naming",
    "action_movement": "action word: movement",
    "sequencing": "sequencing (then/first/next)",
    "planning": "planning (plan/path/route)",
    "coordinates": "coordinate pairs (r, c)",
    "objects": "objects (key/door/goal)",
    "action_descriptive": "action word: descriptive",
    "grid_nouns": "grid nouns (grid/row/column)",
    "contrast_but": "'but' alone",
    "self_correction": "self-correction (wait/actually)",
}


def ratio_figure(stats: pd.DataFrame, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    view = stats.set_index("class").loc[RATIO_PLOT_CLASSES].reset_index()
    view = view.sort_values("per_step_ratio")
    y = np.arange(len(view))
    cap = 20.0
    figure, axis = plt.subplots(figsize=(7.0, 4.6))
    axis.axvline(1.0, color=INK, linewidth=1.1, linestyle=":")
    for k, row in enumerate(view.itertuples()):
        color = BLUE if row.per_step_ratio > 1 else ORANGE
        if row.per_step_ratio > cap:
            axis.annotate(f"$\\rightarrow$ {row.per_step_ratio:.0f}x "
                          f"({row.head_hits} vs {row.tail_hits} hits)",
                          (cap * 0.9, k), color=color, fontsize=8.5,
                          ha="right", va="center")
            continue
        axis.plot([row.per_step_ci_lo, min(row.per_step_ci_hi, cap)], [k, k],
                  color=color, linewidth=1.8, solid_capstyle="round", zorder=2)
        axis.plot(row.per_step_ratio, k, "o", color=color, markersize=7,
                  markeredgecolor="white", markeredgewidth=0.8, zorder=3)
        axis.plot(row.pooled_ratio, k, "o", color=color, markersize=5.5,
                  markerfacecolor="none", markeredgewidth=1.4, zorder=3)
        tag = " *" if not row.robust else ""
        axis.annotate(f"{row.per_step_ratio:.2f}x{tag}",
                      (min(max(row.per_step_ci_hi, row.per_step_ratio), cap), k),
                      xytext=(7, -3), textcoords="offset points",
                      color=INK, fontsize=8.5)
    axis.set_xscale("log")
    axis.set_yticks(y, [RATIO_LABELS[c] for c in view["class"]], fontsize=9)
    axis.set_xticks([0.5, 1, 2, 4, 8], ["0.5x", "1x", "2x", "4x", "8x"])
    axis.set_xlim(0.4, cap)
    style(axis, xlabel="tail density / head density (log scale)")
    axis.annotate("filled = per-step ratio (95% cluster CI)   hollow = character-pooled\n"
                  "* = not robust (CI includes 1 or a leave-one-word-out sign flip)",
                  (0.02, 1.01), xycoords="axes fraction", color=INK, fontsize=8.2)
    figure.tight_layout()
    figure.savefig(path, dpi=220)
    plt.close(figure)


def fightin_figure(table: pd.DataFrame, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    uni = table[table.kind == "unigram"]
    rising = uni.head(25).iloc[::-1]
    falling = uni.tail(25)
    figure, axes = plt.subplots(1, 2, figsize=(9.6, 5.6), sharex=False)
    for axis, part, title in ((axes[0], falling, "falls after stabilization"),
                              (axes[1], rising, "rises after stabilization")):
        colors = [CLASS_COLORS.get(c, NEUTRAL) for c in part.revised_class]
        axis.barh(np.arange(len(part)), part.z, color=colors, height=0.62)
        axis.set_yticks(np.arange(len(part)),
                        [t if len(t) <= 14 else t[:13] + "…" for t in part.term],
                        fontsize=8.4)
        axis.set_title(title, color=INK, fontsize=10)
        style(axis, xlabel="log-odds z (tail vs head)")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in
               (BLUE, PURPLE, ORANGE, GREEN, GOLD, NEUTRAL)]
    figure.legend(handles, ["checking", "concluding", "self-correction",
                            "coordinates/actions", "plan/grid/objects", "no class"],
                  loc="lower center", ncol=6, frameon=False, fontsize=8.2)
    figure.suptitle("Data-driven contrast vocabulary (Fightin' Words, unigrams)",
                    color=INK, fontsize=11)
    figure.tight_layout(rect=(0, 0.05, 1, 1))
    figure.savefig(path, dpi=220)
    plt.close(figure)


EVENT_PANELS = ["checking", "concluding", "self_correction",
                "coordinates", "grid_nouns", "action_answer"]


def event_figure(curves: pd.DataFrame, jumps: pd.DataFrame, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(2, 3, figsize=(10.6, 5.6))
    for axis, name in zip(axes.ravel(), EVENT_PANELS):
        part = curves[curves["class"] == name].sort_values("offset")
        x = (part.offset + 0.5) * WINDOW
        axis.axvline(0, color=INK, linewidth=1.0, linestyle=":")
        axis.fill_between(x, part.ci_lo, part.ci_hi, color=BLUE, alpha=0.18, linewidth=0)
        axis.plot(x, part.mean_density, color=BLUE, linewidth=1.8, marker="o", markersize=3.5)
        jrow = jumps[(jumps["class"] == name) & (jumps.kind == "regex")].iloc[0]
        axis.set_title(f"{RATIO_LABELS.get(name, name)}\n"
                       f"jump {jrow.real_jump:+.2f}/1k vs placebo "
                       f"[{jrow.placebo_lo:+.2f},{jrow.placebo_hi:+.2f}], p={jrow.p_two_sided:.3f}",
                       color=INK, fontsize=8.4)
        style(axis, xlabel="chars from stabilization" if name in EVENT_PANELS[3:] else "",
              ylabel="hits / 1000 chars" if name in (EVENT_PANELS[0], EVENT_PANELS[3]) else "")
    figure.suptitle("Lexical densities aligned to the moment the answer settles "
                    "(shaded: 95% cluster CI)", color=INK, fontsize=11)
    figure.tight_layout()
    figure.savefig(path, dpi=220)
    plt.close(figure)


def combined_paper_figure(stats: pd.DataFrame, path: Path) -> None:
    """Replacement for paper Fig 9: left = early exit (unchanged data), right =
    corrected ratio plot."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    exits = pd.read_csv(REPO_ROOT / "outputs/conjectures/c5_tail/early_exit.csv")
    figure = plt.figure(figsize=(11.6, 4.4))
    ax0 = figure.add_subplot(1, 2, 1)
    ax0.scatter(exits.loc[~exits.on_frontier, "mean_token_savings"],
                exits.loc[~exits.on_frontier, "accuracy"],
                s=20, color=NEUTRAL, alpha=0.55, edgecolors="white", linewidths=0.4)
    frontier = exits[exits.on_frontier].sort_values("mean_token_savings")
    ax0.plot(frontier.mean_token_savings, frontier.accuracy, color=BLUE,
             linewidth=2, marker="o", markersize=5)
    ax0.axhline(0.95, color=INK, linewidth=1.2, linestyle=":")
    ax0.annotate("95% agreement (never reached)", (0.30, 0.955), color=INK, fontsize=9.5)
    best = frontier.iloc[frontier.accuracy.argmax()]
    ax0.annotate(f"best online rule:\n{best.accuracy:.3f} at {best.mean_token_savings:.0%} saved",
                 (best.mean_token_savings, best.accuracy), xytext=(14, -30),
                 textcoords="offset points", color=INK, fontsize=9.5,
                 arrowprops=dict(arrowstyle="-", color=NEUTRAL, linewidth=0.9))
    ax0.set_ylim(0.38, 1.0)
    style(ax0, xlabel="fraction of tokens saved", ylabel="agreement with full-chain action")

    ax1 = figure.add_subplot(1, 2, 2)
    view = stats.set_index("class").loc[RATIO_PLOT_CLASSES].reset_index()
    view = view.sort_values("per_step_ratio")
    cap = 24.0
    ax1.axvline(1.0, color=INK, linewidth=1.1, linestyle=":")
    for k, row in enumerate(view.itertuples()):
        color = BLUE if row.per_step_ratio > 1 else ORANGE
        if row.per_step_ratio > cap:
            ax1.annotate(f"$\\rightarrow$ {row.per_step_ratio:.0f}x "
                         f"({row.head_hits} vs {row.tail_hits} hits)",
                         (cap * 0.92, k), color=color, fontsize=8.2,
                         ha="right", va="center")
            continue
        ax1.plot([row.per_step_ci_lo, min(row.per_step_ci_hi, cap)], [k, k], color=color,
                 linewidth=1.8, solid_capstyle="round")
        ax1.plot(row.per_step_ratio, k, "o", color=color, markersize=6.5,
                 markeredgecolor="white", markeredgewidth=0.8)
    ax1.set_xscale("log")
    ax1.set_xlim(0.3, cap)
    ax1.set_yticks(np.arange(len(view)), [RATIO_LABELS[c] for c in view["class"]], fontsize=8.2)
    ax1.set_xticks([0.5, 1, 2, 4, 8, 16], ["0.5x", "1x", "2x", "4x", "8x", "16x"])
    style(ax1, xlabel="tail / head density ratio (per-step, 95% cluster CI)")
    figure.tight_layout()
    figure.savefig(path, dpi=220)
    plt.close(figure)


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def to_md(frame: pd.DataFrame, floatfmt: str = ".3f") -> str:
    return frame.to_markdown(index=False, floatfmt=floatfmt)


def write_report(args, steps, gate, stats, loo, fw, embed, curves, jumps,
                 deciles, spot_examples) -> None:
    sub = steps[steps.usable]
    lines = [
        "# Corrected C5 tail lexicon (script 16)",
        "",
        f"- {len(steps)} steps; {len(sub)} usable (>{MIN_SIDE_CHARS} chars both sides); "
        f"{steps.attrs['n_stab_zero']} steps with stabilization at prefix 0 (empty head); "
        f"{steps.attrs['prefix_violations']} head/full prefix violations.",
        f"- Trajectory-cluster bootstrap: {args.n_boot} reps (ratios), "
        f"{args.n_boot_curves} (curves/deciles), {args.n_boot_fw} (Fightin' Words), "
        f"{args.n_placebo} placebo draws; seed {SEED}.",
        "",
        "## Sanity gate — published numbers reproduced exactly",
        "", to_md(gate, ".6f"), "",
        "## Revised classes: head vs tail",
        "",
        "Filled per-step ratio = ratio of mean per-step densities (published convention);",
        "pooled = character-weighted. ROBUST = 95% cluster CI excludes 1 AND no",
        "leave-one-literal-out sign flip.",
        "", to_md(stats.round(4)), "",
        "## Leave-one-literal-out",
        "", to_md(loo.sort_values(["class", "literal_tail_share"],
                                  ascending=[True, False]).round(4)), "",
        "## Fightin' Words — top 30 rising / falling unigrams",
        "",
        to_md(fw[fw.kind == "unigram"].head(30)[
            ["term", "head_count", "tail_count", "z",
             "top50_rising_stability", "revised_class"]].round(3)), "",
        to_md(fw[fw.kind == "unigram"].tail(30)[
            ["term", "head_count", "tail_count", "z",
             "top50_falling_stability", "revised_class"]].round(3)), "",
        "## Fightin' Words — top 15 rising / falling bigrams",
        "",
        to_md(fw[fw.kind == "bigram"].head(15)[
            ["term", "head_count", "tail_count", "z", "revised_class"]].round(3)), "",
        to_md(fw[fw.kind == "bigram"].tail(15)[
            ["term", "head_count", "tail_count", "z", "revised_class"]].round(3)), "",
    ]
    if embed is not None:
        lines += ["## Embedding soft categories (per-step mean sentence score)",
                  "", to_md(embed.round(4)), ""]
    lines += [
        "## Event-aligned jumps at T_stab vs shuffled-stabilization placebo",
        "",
        "Jump = density in [0, +800) minus [-800, 0) chars around stabilization;",
        "placebo redraws each step's stabilization point from another step in the",
        "same chain-length quintile. Interpretation rule (stated in advance): a jump",
        "outside the placebo band = stabilization-linked; inside = positional drift.",
        "", to_md(jumps.round(4)), "",
        "## Position-matched decile control (post-stab minus pre-stab, same decile)",
        "", to_md(deciles[deciles["class"].isin(EVENT_PANELS)].round(4)), "",
        "## Action-word context rules: the 20-example spot check (one round, then frozen)",
        "", *[f"    {e}" for e in spot_examples], "",
    ]
    (args.out_dir / "report.md").write_text("\n".join(lines))


def main() -> int:
    args = parse_args()
    rng = np.random.default_rng(SEED)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.figure_dir.mkdir(parents=True, exist_ok=True)

    print("loading steps + texts ...", flush=True)
    steps = load_steps(args)
    print(f"  {len(steps)} steps, usable {int(steps.usable.sum())}, "
          f"stab@0 {steps.attrs['n_stab_zero']}")

    print("sanity gate vs published ...", flush=True)
    gate = sanity_gate(steps, args.published)
    print("  gate PASSED")

    print("collecting match positions ...", flush=True)
    positions = collect_positions(steps)
    spot_examples = spot_check_actions(steps[steps.usable], rng)

    print("revised class stats + LOO + bootstrap ...", flush=True)
    stats, loo = revised_stats(steps, positions, args, rng)
    stats.to_csv(args.out_dir / "revised_lexicon_stats.csv", index=False)
    loo.to_csv(args.out_dir / "loo_robustness.csv", index=False)

    print("fightin' words ...", flush=True)
    fw = fightin_words(steps, args, rng)
    fw.to_csv(args.out_dir / "fightin_words.csv", index=False)

    frame = None
    embed = None
    if not args.skip_embedding:
        print("embedding sentences ...", flush=True)
        frame = embedding_scores(steps, args)
        frame.drop(columns=["text"]).to_parquet(
            args.out_dir / "embedding_sentence_scores.parquet", index=False)
        embed = embedding_stats(frame, steps, args, rng)
        embed.to_csv(args.out_dir / "embedding_class_stats.csv", index=False)

    print("event alignment + placebo ...", flush=True)
    curves, jumps = event_aligned(steps, positions, frame, args, rng)
    curves.to_csv(args.out_dir / "event_aligned.csv", index=False)
    jumps.to_csv(args.out_dir / "placebo_jumps.csv", index=False)

    print("decile control ...", flush=True)
    deciles = decile_control(steps, positions, args, rng)
    deciles.to_csv(args.out_dir / "position_matched.csv", index=False)

    print("precision sample ...", flush=True)
    sample = precision_sample(steps, positions, frame, rng)
    sample.to_csv(args.out_dir / "precision_sample.csv", index=False)

    print("figures ...", flush=True)
    ratio_figure(stats, args.figure_dir / "tail_corrected_ratios.png")
    fightin_figure(fw, args.figure_dir / "tail_fightin_words.png")
    event_figure(curves, jumps, args.figure_dir / "tail_event_aligned.png")
    combined_paper_figure(stats, args.figure_dir / "tail_v2.png")
    if args.paper_figure_dir.exists():
        import shutil
        for name in ("tail_fightin_words.png", "tail_event_aligned.png", "tail_v2.png",
                     "tail_corrected_ratios.png"):
            shutil.copy(args.figure_dir / name, args.paper_figure_dir / name)

    write_report(args, steps, gate, stats, loo, fw, embed, curves, jumps,
                 deciles, spot_examples)
    print(f"done -> {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
