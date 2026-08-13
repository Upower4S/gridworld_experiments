# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

Interpretability research on **goal-directedness and "action commitment"** in the language-model agent `openai/gpt-oss-20b` as it solves 2D gridworld navigation tasks (key-door environments; actions `UP/DOWN/LEFT/RIGHT`). The core question: at each point in the model's chain-of-thought (CoT), how committed is it to the action it will eventually take, and can that commitment be read off the residual-stream activations?

There are **two distinct layers** in this tree, and they are separate projects:

- **Root `SPAR_code/`** — the active experiment. Loose scripts + a `src/` package, run directly with `python`. No packaging, no `pyproject.toml`. **This is where work happens.**
- **`interp/`** — a *nested independent git checkout* of the upstream paper repo (`SPAR-Telos/interp`, arXiv 2602.08964). It has its own `pyproject.toml`, `telos_interp` package, `interp-cli`, `uv` env, and Makefile. Treat it as read-only reference/upstream unless explicitly asked to work in it. Do not conflate its structure or commands with the root project.

## Commands (root project)

There is no build step and no dependency manifest at the root. Scripts run against the ambient Python (3.12/3.13 caches present) and import `src` via a `sys.path` insertion of the repo root at the top of each file. Always run from the repo root:

```bash
# Run a pipeline stage (scripts are numbered and ordered; run in sequence)
python scripts/conceptual_belief_space/02_build_action_beliefs.py --help

# Tests (single test file exists)
pytest tests/test_forced_suffix_probe.py
pytest tests/test_forced_suffix_probe.py -k <name>   # single test

# Lint / format (ruff is configured only inside interp/, but ruff runs here too)
ruff check .
ruff format .
```

Nearly every script uses `argparse` and defaults its inputs/outputs to concrete paths under `outputs/`, so `--help` is the fastest way to learn a stage's contract. Scripts are meant to be run individually and in numeric order, not orchestrated by a runner.

### Commands inside `interp/` (only if working there)
```bash
cd interp && make install-dev   # uv sync --all-extras + pre-commit + editable install
make test          # pytest -c pyproject.toml -vv  (has --doctest-modules, --strict-markers)
make check-style   # ruff format --check + ruff check
interp-cli <subcommand>   # gather_activations | prepare_activations_for_probing | train_*_probe | ...
```

## Architecture

### The behavioral pipeline (`scripts/01`–`07`, top-level)
Builds the core behavioral objects, each stage reading the previous stage's parquet from `outputs/`:

1. **`01_prepare_steps_from_trajectories.py`** — recursively read raw trajectory JSON (from `trajectories_key_door_100/`, `bad_traj_natural_100/`, etc.), reconstruct `prompt_text`, extract analysis/action tokens → `outputs/canonical_steps.parquet`. Raw files are never mutated; bad steps are quarantined to `quarantined_steps.jsonl`.
2. **`02_generate_prefixes.py`** — cut each step's CoT into token **prefixes** → `outputs/prefixes*.parquet`.
3. **`03_score_prefix_actions_*.py`** — re-run the model on each prefix (plus a **forced-answer suffix**) to get per-prefix action probabilities `q_UP…q_RIGHT`. Three interchangeable backends: `_mlx_cached` (local MLX, uses a KV prompt cache), `_together` (Together AI API), `_json_final`. `03b_score_prefix_actions_mc.py` is a Monte-Carlo variant.
4. **`04_compute_commitment_metrics.py`** — the central metric: `commitment_to_final = q(agent_action)`, plus entropy and transformed action distributions.
5. **`05`–`07`** — build behavior vectors and embed/plot the behavior manifold.

The `_mlx_full` suffix on output files marks the full-scale MLX run; unsuffixed files are earlier/smaller subsets. `scripts/activation/` (numbered `01`–`12`) is the parallel activation-side pipeline: token indexing, aligning activations to behavior, layerwise decoding, centroid basins, dynamics/switching, controls.

### The library (`src/conceptual_belief_space/`)
Shared, importable logic behind the numbered scripts. Key modules: `config.py` (`ACTIONS`, `EPS`, `DEFAULT_OUTPUT_DIR`/`DEFAULT_FIGURE_DIR`, `DEFAULT_SEED=42`), `beliefs.py` (commitment/entropy/CLR transforms of action-prob vectors), `io.py` (checked parquet read + markdown report helpers), `manifold.py`, `geometry.py`, `centroids.py`, `probes.py`, `controls.py`, `stats.py`, `splits.py` (trajectory-grouped CV), `plotting.py`, `forced_suffix_probe.py`, `mlp_branch_utils.py`. `src/activation_utils.py` and `src/probing.py` sit at the top level.

### The probe-development arc (`scripts/conceptual_belief_space/`, `00`–`77`)
A long, strictly append-only sequence of experiments trying to **predict the model's top forced action from activations alone**, under progressively stricter leakage rules. It moves through: action-belief vectors → manifolds → linear probes → fusion/stacking → sequence transformers (`22`–`33`) → nested OOF stackers → probe-guided activation embeddings (`54`–`66`) → last-token/sparse **attention** probes (`67`–`75`) → **forced-suffix layer probes** (`76`–`77`). Each numbered file is a self-contained stage that writes a subdirectory under `outputs/conceptual_belief_space/` plus a markdown report; later stages consume earlier outputs. Do not renumber or rewrite existing stages — add a new number.

### Current active direction (`prompt.txt`, scripts `76`/`77`, `forced_suffix_probe.py`)
Force the model to answer by appending a fixed suffix (`FORCED_SUFFIX` = `<|end|><|start|>assistant<|channel|>final<|message|>{"action": "`, exactly 12 known GPT-OSS token IDs — validated on load). Extract activations of those forced/auxiliary tokens **across all layers**, store as raw BF16 shards with JSON sidecars (`forced_suffix_projected_bf16_v1`). `prompt.txt` proposes a new transformer probe whose *sequence axis is the layer index* (token position = layer), to measure how information transitions layer-by-layer and whether high- vs. low-commitment cases differ in that transition. `forced_suffix_probe.py` deliberately keeps the MLX extraction runtime out of the module so shard validation, model tests, and training don't require MLX.

## Conventions to preserve

- **`RUN_HISTORY.md` is the project's memory/audit log** — a large, dated, append-only record of every run, its command, and its metrics/artifacts. When you complete a meaningful run or code change, append a new dated section at the bottom; never overwrite prior entries. Its latest sections define the *current* probe contract and note where earlier results were superseded (e.g., the 2026-07-16 "scope correction" relabeling a 0.70 attention probe as a leaky causal classifier).
- **Strict anti-leakage probe contract** (see RUN_HISTORY tail): frozen backbone, <1M readout params, activation-only inputs, trajectory-grouped nested splits with zero trajectory overlap across folds, and explicit exclusion of grid/task features, progress, future tokens, and forced-answer tokens from model inputs. New probes are expected to state and honor this contract.
- Each script inserts the repo root onto `sys.path` before `from src...` imports (with `# noqa: E402`). Match this pattern in new scripts.
- Data directories (`trajectories_key_door_100/`, `bad_traj_natural_100/`, `together_ai_*`, `lowDim/`, `outputs/`, `figures/`) are large and untracked; scripts reference them by relative path from the repo root.
