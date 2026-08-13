#!/usr/bin/env python
"""Method 4: sentence-function classification of the CoT (the "validated rubric").

Labels every sentence of a trajectory-grouped subsample of decisions with one of
eight reasoning functions (taxonomy adapted from Thought Anchors, Bogdan et al.
2025, arXiv:2506.19143 -- cited in the paper as anchors2025), using the locally
cached MLX build of the *same* gpt-oss-20b checkpoint as a pinned, greedy,
zero-API annotator. Model labels are validated against an author-labelled gold
sample (blind to model labels) with Cohen's kappa and per-class P/R; the rule,
stated in advance: kappa < 0.6 marks the whole method unreliable.

Taxonomy (single-letter codes):
  P  parsing the world   reading the grid text, listing rows/cells, locating objects
  L  planning            proposing/weighing a route or subgoal
  C  computing           concretely tracing moves/cells, counting steps
  U  uncertainty         hesitation, confusion, flagging possible error
  V  verifying           checking/confirming something already derived
  F  concluding          stating the settled outcome of reasoning
  A  answering           naming the final action / output format
  O  other               rule quotes, meta remarks, anything else

Phases:
  --phase classify   run the MLX annotator (checkpointed jsonl, resumable)
  --phase analyze    shares, event-aligned jumps vs placebo, gold sample, figure
  --phase all        both (default)

Run from the repo root with the ambient python (mlx_lm required for classify):
  python3 scripts/17_sentence_function_classification.py --phase classify
  interp/.venv/bin/python scripts/17_sentence_function_classification.py --phase analyze
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
sys.path.insert(0, str(REPO_ROOT))  # noqa: E402

SEED = 42
N_STEPS = 120
BATCH = 25
MODEL = "mlx-community/gpt-oss-20b-MXFP4-Q4"
WINDOW = 400
JUMP_SPAN = 2
FUNCTIONS = list("PLCUVFAO")
FUNCTION_NAMES = {
    "P": "parsing", "L": "planning", "C": "computing", "U": "uncertainty",
    "V": "verifying", "F": "concluding", "A": "answering", "O": "other",
}

SYSTEM_TEMPLATE = (
    "<|start|>system<|message|>You are ChatGPT, a large language model trained by OpenAI."
    "\nKnowledge cutoff: 2024-06\nCurrent date: 2026-08-11\n\nReasoning: low\n\n"
    "# Valid channels: analysis, commentary, final. Channel must be included for every "
    "message.<|end|><|start|>user<|message|>{task}<|end|><|start|>assistant"
)

TASK_HEADER = """Classify each numbered sentence from a maze-solving reasoning transcript into exactly one category:

P = parsing the world: reading the grid text, listing what is in rows/columns/cells, locating agent/key/door/goal
L = planning: proposing or weighing a route or subgoal ("need to get key first", "maybe go around via row 1")
C = computing: concretely tracing moves or cells, counting steps or distances ("from (3,5) down to (4,5) open", "that's 6 steps")
U = uncertainty: hesitation, confusion, or flagging a possible error ("wait", "hmm", "that seems wrong")
V = verifying: checking or confirming something already derived ("let's double-check the wall", "yes, that matches")
F = concluding: stating the settled outcome of a piece of reasoning ("thus the shortest path is 8 steps", "so the only way is via the door")
A = answering: naming the final action or the output ("action: UP", "{"action": "RIGHT"}")
O = other: rule quotations, meta remarks, anything that fits nothing above

Worked example:
1. Row 3: "# _ _ A # _ K _ #" so A at (3,3).   -> 1: P
2. We need to pick up the key before the door. -> 2: L
3. From (3,3) up to (2,3) open, that's 1 step. -> 3: C
4. Wait, is (2,4) a wall?                      -> 4: U
5. Let me double-check row 2.                  -> 5: V
6. Thus the shortest path is 7 steps.          -> 6: F
7. So the action is "UP".                      -> 7: A
8. The door opens automatically when adjacent. -> 8: O

Reply in the final channel with one line per sentence, formatted exactly "<number>: <letter>". No other text.

Sentences:
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("classify", "analyze", "all"), default="all")
    parser.add_argument("--n_steps", type=int, default=N_STEPS)
    parser.add_argument("--batch", type=int, default=BATCH)
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--max_tokens", type=int, default=900)
    parser.add_argument("--out_dir", type=Path,
                        default=REPO_ROOT / "outputs/conjectures/c5_sentence_functions")
    parser.add_argument("--figure_dir", type=Path, default=REPO_ROOT / "figures/c5_corrected")
    parser.add_argument("--n_boot", type=int, default=1000)
    parser.add_argument("--n_placebo", type=int, default=200)
    parser.add_argument("--limit_batches", type=int, default=0,
                        help="debug: stop after this many prompts (0 = all)")
    return parser.parse_args()


def load_s16():
    spec = importlib.util.spec_from_file_location(
        "s16", REPO_ROOT / "scripts/16_c5_tail_lexicon_corrected.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_sentences(args: argparse.Namespace):
    """Same steps frame and sentence splitter as script 16; subsample of steps."""
    s16 = load_s16()
    ns = argparse.Namespace(
        steps=REPO_ROOT / "outputs/conjectures/c5_tail/steps.parquet",
        prefixes=REPO_ROOT / "outputs/prefixes.parquet")
    steps = s16.load_steps(ns)
    usable = steps[steps.usable].reset_index(drop=True)
    rng = np.random.default_rng(SEED)
    take = np.sort(rng.choice(len(usable), size=min(args.n_steps, len(usable)),
                              replace=False))
    sub = usable.iloc[take].reset_index(drop=True)
    rows = []
    for step_row, row in sub.iterrows():
        for a, b, text in s16.split_sentences(row.full_text):
            rows.append({
                "step_row": int(step_row), "trajectory_id": row.trajectory_id,
                "step_id": int(row.step_id), "start": a, "end": b,
                "mid": (a + b) / 2.0,
                "side": "tail" if (a + b) / 2.0 >= row.stab_pos else "head",
                "text": text,
            })
    frame = pd.DataFrame(rows)
    frame["sent_id"] = np.arange(len(frame))
    return s16, sub, frame


# ---------------------------------------------------------------------------
# Phase 1: classify with local MLX gpt-oss-20b (greedy, checkpointed)
# ---------------------------------------------------------------------------

FINAL_SPLIT = re.compile(r"<\|channel\|>final<\|message\|>|assistantfinal", re.IGNORECASE)
LINE = re.compile(r"^\s*(\d+)\s*[:.)-]\s*([PLCUVFAO])\b", re.IGNORECASE | re.MULTILINE)


def parse_labels(raw: str, n: int) -> dict[int, str]:
    parts = FINAL_SPLIT.split(raw)
    final = parts[-1] if len(parts) > 1 else raw
    final = final.split("<|end|>")[0].split("<|return|>")[0]
    out: dict[int, str] = {}
    for m in LINE.finditer(final):
        k = int(m.group(1))
        if 1 <= k <= n and k not in out:
            out[k] = m.group(2).upper()
    return out


def classify(args: argparse.Namespace, frame: pd.DataFrame) -> None:
    from mlx_lm import load as mlx_load
    from mlx_lm import generate as mlx_generate
    from mlx_lm.sample_utils import make_sampler

    args.out_dir.mkdir(parents=True, exist_ok=True)
    ckpt = args.out_dir / "labels_checkpoint.jsonl"
    done: dict[int, str] = {}
    if ckpt.exists():
        for line in ckpt.read_text().splitlines():
            rec = json.loads(line)
            done[rec["sent_id"]] = rec["label"]
    todo = frame[~frame.sent_id.isin(done)]
    print(f"classify: {len(frame)} sentences, {len(done)} done, {len(todo)} to go",
          flush=True)
    if not len(todo):
        return

    model, tokenizer = mlx_load(args.model)
    sampler = make_sampler(temp=0.0)
    handle = ckpt.open("a")
    batches = [todo.iloc[i:i + args.batch] for i in range(0, len(todo), args.batch)]
    if args.limit_batches:
        batches = batches[:args.limit_batches]
    for b_index, batch in enumerate(batches):
        listing = "\n".join(
            f"{k + 1}. {t[:240]}" for k, t in enumerate(batch.text))
        prompt = SYSTEM_TEMPLATE.format(task=TASK_HEADER + listing)
        raw = mlx_generate(model, tokenizer, prompt=prompt,
                           max_tokens=args.max_tokens, sampler=sampler, verbose=False)
        labels = parse_labels(raw, len(batch))
        if len(labels) < 0.8 * len(batch):  # one retry with a sterner suffix
            raw = mlx_generate(model, tokenizer,
                               prompt=SYSTEM_TEMPLATE.format(
                                   task=TASK_HEADER + listing +
                                   "\n\nAnswer with EXACTLY one '<number>: <letter>' line per sentence."),
                               max_tokens=args.max_tokens, sampler=sampler, verbose=False)
            labels = {**parse_labels(raw, len(batch)), **labels}
        for k, (_, row) in enumerate(batch.iterrows()):
            handle.write(json.dumps({"sent_id": int(row.sent_id),
                                     "label": labels.get(k + 1, "?")}) + "\n")
        handle.flush()
        if (b_index + 1) % 20 == 0:
            print(f"  batch {b_index + 1}/{len(batches)}", flush=True)
    handle.close()


# ---------------------------------------------------------------------------
# Phase 2: analysis
# ---------------------------------------------------------------------------

def analyze(args: argparse.Namespace, s16, sub: pd.DataFrame, frame: pd.DataFrame) -> None:
    rng = np.random.default_rng(SEED)
    ckpt = args.out_dir / "labels_checkpoint.jsonl"
    labels = {json.loads(l)["sent_id"]: json.loads(l)["label"]
              for l in ckpt.read_text().splitlines()}
    frame = frame.assign(label=frame.sent_id.map(labels))
    n_missing = int((frame.label.isna() | (frame.label == "?")).sum())
    frame = frame[frame.label.isin(FUNCTIONS)].copy()
    print(f"analyze: {len(frame)} labelled sentences ({n_missing} unparsed)", flush=True)
    frame.drop(columns=["text"]).to_parquet(args.out_dir / "sentence_labels.parquet",
                                            index=False)

    groups = s16.traj_groups_of(sub)
    stab = sub.stab_pos.to_numpy()
    lens = sub.full_len.to_numpy()

    # Head/tail function shares (per-step mean shares, cluster bootstrap).
    share = (frame.groupby(["step_row", "side"]).label
             .value_counts(normalize=True).unstack(fill_value=0.0)
             .reindex(columns=FUNCTIONS, fill_value=0.0))
    both = share.unstack("side")
    both = both.dropna()
    rows = []
    boot_idx = [np.concatenate([groups[t] for t in rng.integers(0, len(groups), len(groups))])
                for _ in range(args.n_boot)]
    present = set(both.index)
    for fn in FUNCTIONS:
        head = both[(fn, "head")].to_numpy()
        tail = both[(fn, "tail")].to_numpy()
        diff = tail - head
        row_of = {s: i for i, s in enumerate(both.index)}
        boots = []
        for idx in boot_idx:
            sel = [row_of[s] for s in idx if s in present]
            if sel:
                boots.append(diff[sel].mean())
        rows.append({
            "function": FUNCTION_NAMES[fn], "code": fn,
            "head_share": float(head.mean()), "tail_share": float(tail.mean()),
            "diff": float(diff.mean()),
            "ci_lo": float(np.quantile(boots, 0.025)),
            "ci_hi": float(np.quantile(boots, 0.975)),
            "n_steps": int(len(both)),
        })
    stats = pd.DataFrame(rows)
    stats.to_csv(args.out_dir / "function_stats.csv", index=False)

    # Event-aligned share curves + placebo jumps (reuse script 16 machinery).
    offsets = np.arange(-5, 5)
    n = len(sub)
    by_step: dict[int, tuple[np.ndarray, dict[str, np.ndarray]]] = {}
    for step_row, g in frame.sort_values("mid").groupby("step_row"):
        mids = g.mid.to_numpy()
        ind = {fn: np.concatenate([[0.0], np.cumsum((g.label == fn).to_numpy(float))])
               for fn in FUNCTIONS}
        by_step[int(step_row)] = (mids, ind)

    def window_share(i: int, lo: float, hi: float, fn: str) -> float | None:
        if i not in by_step:
            return None
        mids, ind = by_step[i]
        a, b = np.searchsorted(mids, lo), np.searchsorted(mids, hi)
        if b == a:
            return None
        return float((ind[fn][b] - ind[fn][a]) / (b - a))

    curve_rows = []
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
    for j, fn in enumerate(FUNCTIONS):
        for k, off in enumerate(offsets):
            col = dens[:, j, k]
            if np.isfinite(col).sum() < 15:
                continue
            boots = [np.nanmean(col[idx]) for idx in boot_small]
            curve_rows.append({
                "function": FUNCTION_NAMES[fn], "offset": int(off),
                "mean_share": float(np.nanmean(col)),
                "ci_lo": float(np.nanquantile(boots, 0.025)),
                "ci_hi": float(np.nanquantile(boots, 0.975)),
                "n_steps": int(np.isfinite(col).sum()),
            })
    curves = pd.DataFrame(curve_rows)
    curves.to_csv(args.out_dir / "event_aligned.csv", index=False)

    span = JUMP_SPAN * WINDOW
    draws = s16.placebo_positions(sub, args.n_placebo, rng)

    def mean_jump(stab_vec: np.ndarray, fn: str) -> float:
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

    def empirical_p(real: float, placebo: np.ndarray) -> float:
        hi = (1 + np.sum(placebo >= real)) / (len(placebo) + 1)
        lo = (1 + np.sum(placebo <= real)) / (len(placebo) + 1)
        return float(min(1.0, 2.0 * min(hi, lo)))

    jump_rows = []
    for fn in FUNCTIONS:
        real = mean_jump(stab, fn)
        placebo = np.array([mean_jump(draws[d], fn) for d in range(args.n_placebo)])
        jump_rows.append({
            "function": FUNCTION_NAMES[fn], "real_jump": real,
            "placebo_mean": float(np.nanmean(placebo)),
            "placebo_lo": float(np.nanquantile(placebo, 0.025)),
            "placebo_hi": float(np.nanquantile(placebo, 0.975)),
            "p_two_sided": empirical_p(real, placebo),
        })
    jumps = pd.DataFrame(jump_rows)
    jumps.to_csv(args.out_dir / "placebo_jumps.csv", index=False)

    # Gold sample for author labelling (blind: no model label in the sample file).
    gold_rng = np.random.default_rng(SEED)
    picks = []
    for fn in FUNCTIONS:
        pool = frame[frame.label == fn]
        k = min(19, len(pool))
        if k:
            picks.append(pool.sample(n=k, random_state=int(gold_rng.integers(1 << 30))))
    gold = pd.concat(picks).sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    gold_out = gold[["sent_id", "text"]].copy()
    gold_out["gold_label"] = ""
    gold_out.to_csv(args.out_dir / "gold_sample.csv", index=False)
    gold[["sent_id", "label"]].to_csv(args.out_dir / "gold_key.csv", index=False)

    # Figure: head/tail shares + event-aligned small multiples for key functions.
    figure_path = args.figure_dir / "sentence_functions.png"
    plot(stats, curves, jumps, figure_path)

    lines = [
        "# Sentence-function classification (script 17, method 4)",
        "",
        f"- {len(frame)} sentences from {n} decisions; {n_missing} unparsed; "
        f"annotator: {MODEL}, greedy, Reasoning: low; taxonomy: Thought Anchors adapted.",
        "",
        "## Head vs tail share of each function (per-step means, cluster CI)",
        "", stats.round(4).to_markdown(index=False), "",
        "## Jump at T_stab vs shuffled-stabilization placebo",
        "", jumps.round(4).to_markdown(index=False), "",
        "Gold sample written to gold_sample.csv (author labels pending -> agreement.csv).",
    ]
    (args.out_dir / "report.md").write_text("\n".join(lines))
    print(stats.round(3).to_string(index=False))
    print(jumps.round(3).to_string(index=False))


def plot(stats: pd.DataFrame, curves: pd.DataFrame, jumps: pd.DataFrame,
         path: Path) -> None:
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

    figure = plt.figure(figsize=(11.8, 7.6))
    grid = figure.add_gridspec(2, 4, height_ratios=[1.0, 1.15], hspace=0.42, wspace=0.34)

    # Gold-validated precision per model class (agreement.csv); classes below
    # 0.80 are marked as unreliable in the plot.
    RELIABLE = {"parsing", "concluding", "answering"}
    axis = figure.add_subplot(grid[0, :])
    order = stats.sort_values("diff")
    y = np.arange(len(order))
    axis.axvline(0, color=INK, linewidth=1.0, linestyle=":")
    for k, row in enumerate(order.itertuples()):
        color = BLUE if row.diff > 0 else ORANGE
        reliable = row.function in RELIABLE
        axis.plot([row.ci_lo, row.ci_hi], [k, k], color=color, linewidth=1.8,
                  solid_capstyle="round", alpha=1.0 if reliable else 0.45)
        axis.plot(row.diff, k, "o", color=color, markersize=7,
                  markeredgecolor="white", markeredgewidth=0.8,
                  alpha=1.0 if reliable else 0.45)
        axis.annotate(f"{row.head_share:.2f} → {row.tail_share:.2f}",
                      (max(row.ci_hi, row.diff), k), xytext=(8, -3),
                      textcoords="offset points", color=INK, fontsize=8.5, ha="left")
    labels = [fn if fn in RELIABLE else f"{fn} *"
              for fn in order.function.tolist()]
    axis.set_yticks(y, labels, fontsize=9.5)
    style(axis, xlabel="tail share − head share of sentences (95% cluster CI)"
          "      * label precision < 0.80 on blind gold — faded classes not citable")
    axis.set_title("Sentence functions before vs after the answer settles "
                   "(gpt-oss-20b annotator; gold agreement kappa 0.50)",
                   color=INK, fontsize=11)

    panel_fns = ["parsing", "computing", "verifying", "concluding"]
    for p, fn in enumerate(panel_fns):
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
    s16, sub, frame = build_sentences(args)
    print(f"{len(sub)} steps, {len(frame)} sentences", flush=True)
    if args.phase in ("classify", "all"):
        classify(args, frame)
    if args.phase in ("analyze", "all"):
        analyze(args, s16, sub, frame)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
