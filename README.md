# Gridworld Experiments: the Action–Belief Gap in gpt-oss-20b

Interpretability research on **action commitment along the chain of thought (CoT)** of the
language-model agent `openai/gpt-oss-20b` solving 2D key-door gridworlds (actions
`UP / DOWN / LEFT / RIGHT`).

At every point in the CoT we ask: *if the model had to answer now, what would it do, and how
committed is it?* We measure this behaviourally by appending a fixed forced-answer suffix to
each CoT prefix and reading the action distribution `q = (q_UP, q_DOWN, q_LEFT, q_RIGHT)`. We
then ask whether that belief can be **read** from, and **written** into, the residual stream.

This is a follow-up to the SPAR-Telos gridworld paper
([arXiv 2602.08964](https://arxiv.org/abs/2602.08964), code at
[SPAR-Telos/interp](https://github.com/SPAR-Telos/interp)).

Paper draft: [`paper/main.pdf`](paper/main.pdf), *Reading and Writing an Agent's Belief About
Its Own Next Action*.

## Headline results

All numbers use the canonical extracted labels (see "Label regime" below). The full evidence
is in [`gridworld_action_belief_gap_claims_conjectures.md`](gridworld_action_belief_gap_claims_conjectures.md)
and [`RUN_HISTORY.md`](RUN_HISTORY.md).

- **Commitment rises gradually along the CoT.** Commitment to the eventual action goes from
  about 0.35 at the empty prefix to about 0.92 at the end of the chain.
- **Much of the reasoning comes after the decision.** The median post-commitment tail is
  50.6% of the CoT. Three clocks tick in order: first mention of the action (median 0.25 of
  the CoT), argmax stabilization (0.49), then permanent commitment with q ≥ 0.8 (0.65).
- **Beliefs track the reasoning text, not the world state.** In mismatched prompt/CoT pairs,
  q follows the CoT.
- **Commitment is linearly readable from activations.** A 5-fold, trajectory-grouped
  probe on CoT-window activations reaches Spearman 0.77 ± 0.01, against a fair positional
  baseline of 0.37.
- **Readable but not easily writable.** Steering along the decoded directions mostly fails to
  move the model's action. A pre-registered activation-causality campaign
  (`scripts/30`–`42`, `src/activation_causality/`) tests this with frozen confirmation sets.

## Repository layout

| Path | What it is |
|---|---|
| `scripts/01`–`07` | Core behavioural pipeline: trajectories → CoT prefixes → forced-suffix action scores → commitment metrics → behaviour manifold |
| `scripts/08`–`29` | Behavioural follow-ups: commitment boundaries, mismatched prompts, belief-update anchors, conjecture tests, sentence-function annotation, text-level injection and patching |
| `scripts/30`–`42` | Activation causality campaign: discovery, frozen confirmation, audits, figures |
| `scripts/activation/01`–`12` | Activation-side pipeline: token indexing, alignment to behaviour, layerwise decoding, controls |
| `scripts/conceptual_belief_space/00`–`96` | Probe-development arc: predicting the model's forced action and commitment from activations alone, under increasingly strict leakage rules |
| `src/conceptual_belief_space/` | Shared library: beliefs/commitment transforms, probes, grouped CV splits, geometry, plotting, forced-suffix probe |
| `src/activation_causality/` | Engine and protocols for the activation-intervention campaign |
| `tests/` | pytest suite (forced-suffix probe, activation-causality engine and reports) |
| `notebooks/` | Exploratory reports (behaviour diagnostics, activation manifold, belief space) |
| `paper/` | NeurIPS-style draft (`main.tex`, `main.pdf`, figures) |
| `RUN_HISTORY.md` | Dated, append-only log of every run: command, metrics, artifacts. The project's audit trail |
| `RESEARCH_DIRECTIONS.md`, `research_directions_deep_review_2026-08-20.md` | Research plans and literature positioning |
| `CLAUDE.md` | Conventions for AI-assisted work in this repo |

Scripts are numbered and meant to be run **one at a time, in order, from the repo root**. Each
one inserts the repo root on `sys.path`, reads the previous stage's output from `outputs/`, and
writes a Markdown report next to its artifacts. Run `python <script> --help` to see a stage's
inputs and outputs. Existing stages are never renumbered; new experiments get a new number.

## Data (not in this repo)

Data and derived artifacts are **not tracked** (see `.gitignore`). The scripts expect them at
these paths, relative to the repo root:

- `trajectories_key_door_100/`, `bad_traj_natural_100/`: raw agent trajectories (JSON) and
  packed BF16 activations
- `together_ai_*/`: trajectories scored with the Together AI API
- `outputs/`: all intermediate parquet/npy artifacts, probe checkpoints and reports
- `figures/`: generated figures
- `interp/`: a checkout of [SPAR-Telos/interp](https://github.com/SPAR-Telos/interp), which
  provides activation gathering and the `telos_interp` package

To reproduce from scratch, gather trajectories and activations with `interp/`, then run
`scripts/01` onward.

### Label regime

Several scoring runs exist for q. The stage-03 MLX-cached labels are **quarantined**: 93.5%
of q values changed on re-extraction. The canonical labels are
`outputs/conceptual_belief_space/cot_window_layer_transition_probe/activations/extracted_beliefs.parquet`
(98,310 prefixes, 95 trajectories). Some older scripts still default to the quarantined file,
so pass the extracted labels explicitly.

## Probe contract

New activation probes must follow the anti-leakage contract set out in `RUN_HISTORY.md`:

- frozen backbone, with a readout under 1M parameters
- activation-only inputs: no grid/task features, no progress signal, no future tokens, no
  forced-answer tokens
- trajectory-grouped nested cross-validation, with no trajectory appearing in more than one fold

## Setup

There is no package manifest at the root. Use Python 3.12+ with `numpy`, `pandas`, `pyarrow`,
`scikit-learn`, `torch`, `matplotlib` and `pytest`. Model-running stages use `mlx` / `mlx-lm`
on Apple silicon, or the Together AI API.

```bash
pytest tests/test_forced_suffix_probe.py
ruff check .
```
