#!/usr/bin/env python
"""Full-corpus 4-class sentence-function annotation + head/tail distributions.

Taxonomy = coarse4 v2, frozen after two documented pilot rounds
(outputs/conjectures/c5_sentence_functions/coarse4_pilot/):
  W world-reading   grid text, object locations, rules, DERIVED world facts
  S solving         proposing/weighing routes, tracing moves, counting steps
  M meta-checking   hesitation/doubt, re-verifying something already derived
  A concluding/answering (solution-level)  settled path/step/action statements

Instrument quality on the 152-sentence blind gold sample (author-labelled,
Reasoning: low): precision A 0.91 / S 0.78 / W 0.73 / M 0.32. M is exploratory
at every tested effort; verification claims stay with the validated regex
class (scripts 13/16). The chosen effort is recorded in the labels filename.

Annotator: local MLX gpt-oss-20b (pinned weights, greedy, fixed prompt,
checkpointed jsonl, resumable). Data lives in the primary checkout; this
script resolves DATA_ROOT accordingly so it can run from a worktree.

Run:
  interp/.venv/bin/python scripts/18_coarse4_full_annotation.py --phase classify
  interp/.venv/bin/python scripts/18_coarse4_full_annotation.py --phase analyze
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = (REPO_ROOT if (REPO_ROOT / "outputs" / "conjectures").exists()
             else Path("/Users/wws/SPAR_code"))
sys.path.insert(0, str(DATA_ROOT))  # noqa: E402

SEED = 42
WINDOW = 400
JUMP_SPAN = 2
FUNCTIONS = list("WSMA")
FUNCTION_NAMES = {"W": "world-reading", "S": "solving",
                  "M": "meta-checking", "A": "concluding/answering"}
LINE4 = re.compile(r"^\s*(\d+)\s*[:.)-]\s*([WSMA])\b", re.IGNORECASE | re.MULTILINE)

TASK_HEADER = """Classify each numbered sentence from a maze-solving reasoning transcript into exactly one category:

W = world-reading: reading the grid text, listing rows/columns/cells, locating agent/key/door/goal, quoting the environment's rules, or stating a fact about the world's structure, including DERIVED facts ("So we cannot cross column 4.", "So the door is blocking the left side.", "row3 col2 '_' yes.").
S = solving: proposing or weighing a route or subgoal, tracing moves or cells, counting steps or distances ("(2,3)->(2,2)->(3,2)", "Which is better?", "maybe go around via row 1").
M = meta-checking: hesitation or doubt ("wait", "hmm", "did I make a mistake?"), or verifying something ALREADY derived, such as re-checking a computed path's legality or a step count. "Check" aimed at reading the map for the first time is W, not M.
A = concluding/answering about the SOLUTION: a settled statement about the path, step count, or chosen action ("Thus 10 steps total.", "So that path is blocked.", "So the action is UP.", "{"action": "RIGHT"}").

Worked example:
1. Row 3: "# _ _ A # _ K _ #" so A at (3,3).            -> 1: W
2. So we cannot cross column 4 in rows 1-4.              -> 2: W
3. We need to pick up the key before the door.           -> 3: S
4. From (3,3) up to (2,3) open, that's 1 step.           -> 4: S
5. Wait, is (2,4) a wall?                                -> 5: M
6. Let me double-check that the 7-step path is legal.    -> 6: M
7. Thus the shortest path is 7 steps.                    -> 7: A
8. So the action is "UP".                                -> 8: A

Reply in the final channel with one line per sentence, formatted exactly "<number>: <letter>". No other text.

Sentences:
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("classify", "analyze", "all"), default="all")
    parser.add_argument("--effort", choices=("low", "medium", "high"), default="low")
    parser.add_argument("--n_steps", type=int, default=0,
                        help="0 = all usable decisions")
    parser.add_argument("--batch", type=int, default=0,
                        help="0 = auto (25 for low, 5 otherwise)")
    parser.add_argument("--max_tokens", type=int, default=0,
                        help="0 = auto (1400 for low, 4000 otherwise)")
    parser.add_argument("--out_dir", type=Path,
                        default=DATA_ROOT / "outputs/conjectures/c5_sentence_functions_coarse4")
    parser.add_argument("--figure_dir", type=Path, default=DATA_ROOT / "figures/c5_corrected")
    parser.add_argument("--n_boot", type=int, default=1000)
    parser.add_argument("--n_placebo", type=int, default=200)
    return parser.parse_args()


def load_s17():
    spec = importlib.util.spec_from_file_location(
        "s17", DATA_ROOT / "scripts/17_sentence_function_classification.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse4(s17, raw: str, n: int) -> dict[int, str]:
    parts = s17.FINAL_SPLIT.split(raw)
    final = parts[-1] if len(parts) > 1 else raw
    final = final.split("<|end|>")[0].split("<|return|>")[0]
    out: dict[int, str] = {}
    for m in LINE4.finditer(final):
        k = int(m.group(1))
        if 1 <= k <= n and k not in out:
            out[k] = m.group(2).upper()
    return out


def classify(args, s17, frame: pd.DataFrame) -> None:
    from mlx_lm import load as mlx_load, generate as mlx_generate
    from mlx_lm.sample_utils import make_sampler

    batch = args.batch or (25 if args.effort == "low" else 5)
    max_tokens = args.max_tokens or (1400 if args.effort == "low" else 4000)
    system = s17.SYSTEM_TEMPLATE.replace("Reasoning: low", f"Reasoning: {args.effort}")
    ckpt = args.out_dir / f"labels_{args.effort}.jsonl"
    done = {}
    if ckpt.exists():
        for line in ckpt.read_text().splitlines():
            rec = json.loads(line)
            done[rec["sent_id"]] = rec["label"]
    todo = frame[~frame.sent_id.isin(done)]
    print(f"classify[{args.effort}]: {len(frame)} sentences, {len(done)} done, "
          f"{len(todo)} to go", flush=True)
    if not len(todo):
        return
    model, tokenizer = mlx_load(s17.MODEL)
    sampler = make_sampler(temp=0.0)
    handle = ckpt.open("a")
    chunks = [todo.iloc[i:i + batch] for i in range(0, len(todo), batch)]
    for ci, chunk in enumerate(chunks):
        listing = "\n".join(f"{k + 1}. {t[:240]}" for k, t in enumerate(chunk.text))
        raw = mlx_generate(model, tokenizer,
                           prompt=system.format(task=TASK_HEADER + listing),
                           max_tokens=max_tokens, sampler=sampler, verbose=False)
        got = parse4(s17, raw, len(chunk))
        if len(got) < len(chunk):
            raw2 = mlx_generate(model, tokenizer,
                                prompt=system.format(
                                    task=TASK_HEADER + listing +
                                    "\n\nAnswer with EXACTLY one '<number>: <letter>' line per sentence, nothing else."),
                                max_tokens=max_tokens, sampler=sampler, verbose=False)
            got = {**parse4(s17, raw2, len(chunk)), **got}
        for k, (_, row) in enumerate(chunk.iterrows()):
            handle.write(json.dumps({"sent_id": int(row.sent_id),
                                     "label": got.get(k + 1, "?")}) + "\n")
        handle.flush()
        if (ci + 1) % 50 == 0:
            print(f"  batch {ci + 1}/{len(chunks)}", flush=True)
    handle.close()


def analyze(args, s17, sub: pd.DataFrame, frame: pd.DataFrame) -> None:
    s16 = s17.load_s16()
    rng = np.random.default_rng(SEED)
    ckpt = args.out_dir / f"labels_{args.effort}.jsonl"
    labels = {json.loads(l)["sent_id"]: json.loads(l)["label"]
              for l in ckpt.read_text().splitlines()}
    frame = frame.assign(label=frame.sent_id.map(labels))
    n_missing = int((frame.label.isna() | (frame.label == "?")).sum())
    frame = frame[frame.label.isin(FUNCTIONS)].copy()
    print(f"analyze: {len(frame)} labelled ({n_missing} unparsed)", flush=True)
    frame.drop(columns=["text"]).to_parquet(args.out_dir / "sentence_labels.parquet",
                                            index=False)

    groups = s16.traj_groups_of(sub)
    stab = sub.stab_pos.to_numpy()
    lens = sub.full_len.to_numpy()
    n = len(sub)

    share = (frame.groupby(["step_row", "side"]).label
             .value_counts(normalize=True).unstack(fill_value=0.0)
             .reindex(columns=FUNCTIONS, fill_value=0.0))
    both = share.unstack("side").dropna()
    boot_idx = [np.concatenate([groups[t] for t in rng.integers(0, len(groups), len(groups))])
                for _ in range(args.n_boot)]
    present = set(both.index)
    row_of = {s: i for i, s in enumerate(both.index)}
    rows = []
    for fn in FUNCTIONS:
        head = both[(fn, "head")].to_numpy()
        tail = both[(fn, "tail")].to_numpy()
        diff = tail - head
        boots = []
        for idx in boot_idx:
            sel = [row_of[s] for s in idx if s in present]
            if sel:
                boots.append(diff[sel].mean())
        rows.append({"function": FUNCTION_NAMES[fn], "code": fn,
                     "head_share": float(head.mean()), "tail_share": float(tail.mean()),
                     "diff": float(diff.mean()),
                     "ci_lo": float(np.quantile(boots, 0.025)),
                     "ci_hi": float(np.quantile(boots, 0.975)),
                     "n_steps": int(len(both))})
    stats = pd.DataFrame(rows)
    stats.to_csv(args.out_dir / "function_stats.csv", index=False)

    by_step = {}
    for step_row, g in frame.sort_values("mid").groupby("step_row"):
        mids = g.mid.to_numpy()
        ind = {fn: np.concatenate([[0.0], np.cumsum((g.label == fn).to_numpy(float))])
               for fn in FUNCTIONS}
        by_step[int(step_row)] = (mids, ind)

    def window_share(i, lo, hi, fn):
        if i not in by_step:
            return None
        mids, ind = by_step[i]
        a, b = np.searchsorted(mids, lo), np.searchsorted(mids, hi)
        if b == a:
            return None
        return float((ind[fn][b] - ind[fn][a]) / (b - a))

    offsets = np.arange(-5, 5)
    dens = np.full((n, len(FUNCTIONS), len(offsets)), np.nan)
    for i in range(n):
        for k, off in enumerate(offsets):
            lo = stab[i] + off * WINDOW
            hi = lo + WINDOW
            if lo < 0 or hi > lens[i]:
                continue
            for j, fn in enumerate(FUNCTIONS):
                v = window_share(i, lo, hi, fn)
                if v is not None:
                    dens[i, j, k] = v
    boot_small = boot_idx[: min(400, len(boot_idx))]
    curve_rows = []
    for j, fn in enumerate(FUNCTIONS):
        for k, off in enumerate(offsets):
            col = dens[:, j, k]
            if np.isfinite(col).sum() < 15:
                continue
            boots = [np.nanmean(col[idx]) for idx in boot_small]
            curve_rows.append({"function": FUNCTION_NAMES[fn], "offset": int(off),
                               "mean_share": float(np.nanmean(col)),
                               "ci_lo": float(np.nanquantile(boots, 0.025)),
                               "ci_hi": float(np.nanquantile(boots, 0.975)),
                               "n_steps": int(np.isfinite(col).sum())})
    curves = pd.DataFrame(curve_rows)
    curves.to_csv(args.out_dir / "event_aligned.csv", index=False)

    span = JUMP_SPAN * WINDOW
    draws = s16.placebo_positions(sub, args.n_placebo, rng)

    def mean_jump(stab_vec, fn):
        vals = []
        for i in range(n):
            p = stab_vec[i]
            if p < span or p + span > lens[i]:
                continue
            pre = window_share(i, p - span, p, fn)
            post = window_share(i, p, p + span, fn)
            if pre is None or post is None:
                continue
            vals.append(post - pre)
        return float(np.mean(vals)) if vals else float("nan")

    def empirical_p(real, placebo):
        hi = (1 + np.sum(placebo >= real)) / (len(placebo) + 1)
        lo = (1 + np.sum(placebo <= real)) / (len(placebo) + 1)
        return float(min(1.0, 2.0 * min(hi, lo)))

    jump_rows = []
    for fn in FUNCTIONS:
        real = mean_jump(stab, fn)
        placebo = np.array([mean_jump(draws[d], fn) for d in range(args.n_placebo)])
        jump_rows.append({"function": FUNCTION_NAMES[fn], "real_jump": real,
                          "placebo_mean": float(np.nanmean(placebo)),
                          "placebo_lo": float(np.nanquantile(placebo, 0.025)),
                          "placebo_hi": float(np.nanquantile(placebo, 0.975)),
                          "p_two_sided": empirical_p(real, placebo)})
    jumps = pd.DataFrame(jump_rows)
    jumps.to_csv(args.out_dir / "placebo_jumps.csv", index=False)

    plot(stats, curves, jumps, args.figure_dir / "coarse4_functions.png",
         n_decisions=int(stats.n_steps.iloc[0]))

    n_labeled_steps = int(frame.step_row.nunique())
    lines = [
        "# Coarse-4 sentence functions (script 18)",
        "",
        f"- effort {args.effort}; {len(frame)} labelled sentences covering "
        f"{n_labeled_steps} decisions (of {n} usable; annotation checkpointed, "
        f"stopped at user request once >half the corpus was covered); "
        f"{n_missing} sentences outside the checkpoint or unparsed.",
        "- Instrument (gold-152): precision A 0.91 / S 0.78 / W 0.73 / M 0.32.",
        "  M is exploratory only; verification claims rest on the regex class.",
        "",
        "## Head vs tail share", "", stats.round(4).to_markdown(index=False), "",
        "## Jump at T_stab vs placebo", "", jumps.round(4).to_markdown(index=False), "",
    ]
    (args.out_dir / "report.md").write_text("\n".join(lines))
    print(stats.round(3).to_string(index=False))
    print(jumps.round(3).to_string(index=False))


def plot(stats, curves, jumps, path: Path, n_decisions: int = 0) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    INK, GRID, BLUE, ORANGE = "#333333", "#e6e6e6", "#2a78d6", "#eb6834"

    def style(axis, xlabel="", ylabel=""):
        axis.grid(True, color=GRID, linewidth=0.8)
        axis.set_axisbelow(True)
        axis.tick_params(colors=INK, labelsize=8.5)
        for spine in ("top", "right"):
            axis.spines[spine].set_visible(False)
        if xlabel:
            axis.set_xlabel(xlabel, color=INK, fontsize=9.5)
        if ylabel:
            axis.set_ylabel(ylabel, color=INK, fontsize=9.5)

    figure = plt.figure(figsize=(11.8, 7.2))
    grid = figure.add_gridspec(2, 4, height_ratios=[0.9, 1.1], hspace=0.45, wspace=0.35)
    axis = figure.add_subplot(grid[0, :])
    order = stats.sort_values("diff")
    axis.axvline(0, color=INK, linewidth=1.0, linestyle=":")
    for k, row in enumerate(order.itertuples()):
        color = BLUE if row.diff > 0 else ORANGE
        faded = row.code == "M"
        axis.plot([row.ci_lo, row.ci_hi], [k, k], color=color, linewidth=1.8,
                  solid_capstyle="round", alpha=0.45 if faded else 1.0)
        axis.plot(row.diff, k, "o", color=color, markersize=7,
                  markeredgecolor="white", markeredgewidth=0.8,
                  alpha=0.45 if faded else 1.0)
        axis.annotate(f"{row.head_share:.2f} → {row.tail_share:.2f}",
                      (max(row.ci_hi, row.diff), k), xytext=(8, -3),
                      textcoords="offset points", color=INK, fontsize=8.5)
    axis.set_yticks(np.arange(len(order)),
                    [f"{r.function}{' *' if r.code == 'M' else ''}"
                     for r in order.itertuples()], fontsize=9.5)
    style(axis, xlabel="tail share − head share of sentences (95% cluster CI)"
          "      * precision 0.32 on blind gold — exploratory only")
    n_txt = f"{n_decisions} decisions" if n_decisions else "coarse-4"
    axis.set_title("Sentence functions before vs after the answer settles "
                   f"(coarse-4, {n_txt}, gpt-oss-20b annotator)", color=INK, fontsize=11)

    for p, fn in enumerate([FUNCTION_NAMES[c] for c in FUNCTIONS]):
        axis = figure.add_subplot(grid[1, p])
        part = curves[curves.function == fn].sort_values("offset")
        x = (part.offset + 0.5) * WINDOW
        axis.axvline(0, color=INK, linewidth=1.0, linestyle=":")
        axis.fill_between(x, part.ci_lo, part.ci_hi, color=BLUE, alpha=0.18, linewidth=0)
        axis.plot(x, part.mean_share, color=BLUE, linewidth=1.8, marker="o", markersize=3)
        jrow = jumps[jumps.function == fn].iloc[0]
        axis.set_title(f"{fn}\njump {jrow.real_jump:+.3f} vs "
                       f"[{jrow.placebo_lo:+.3f},{jrow.placebo_hi:+.3f}] "
                       f"p={jrow.p_two_sided:.2f}", color=INK, fontsize=8.6)
        style(axis, xlabel="chars from stabilization",
              ylabel="share of sentences" if p == 0 else "")
    figure.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.figure_dir.mkdir(parents=True, exist_ok=True)
    s17 = load_s17()
    n_steps = args.n_steps or 10 ** 9
    _, sub, frame = s17.build_sentences(argparse.Namespace(n_steps=n_steps))
    print(f"{len(sub)} decisions, {len(frame)} sentences", flush=True)
    if args.phase in ("classify", "all"):
        classify(args, s17, frame)
    if args.phase in ("analyze", "all"):
        analyze(args, s17, sub, frame)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
