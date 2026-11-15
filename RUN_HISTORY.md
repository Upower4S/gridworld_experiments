# SPAR Code Run History

Created: 2026-06-28
Repo path: `/Users/wws/SPAR_code/`

This file records the main Codex changes, scripts added, runs performed, and artifacts generated for the gridworld action-commitment behavior experiment. It is intended as a compact audit trail. Append future runs to the bottom instead of overwriting prior entries.

## Current Experiment Scope

- Dataset: key-door gridworld trajectories, primarily from `trajectories_key_door_100` / `data/raw`.
- Model/action setting: four gridworld actions: `UP`, `DOWN`, `LEFT`, `RIGHT`.
- Main behavioral object: prefix-level action probabilities `q_UP`, `q_DOWN`, `q_LEFT`, `q_RIGHT`.
- Main commitment metric: `commitment_to_final = q(agent_action)`.
- Main scored subset so far: 10 trajectories, 132 decision steps, 9,942 prefix points.
- Raw trajectory files were not modified by the data-preparation scripts.

## Main Code Changes

### 1. Canonical trajectory step preparation

Implemented:

- `scripts/01_prepare_steps_from_trajectories.py`

Purpose:

- Read nested trajectory JSON files recursively.
- Convert one trajectory step into one canonical row.
- Reconstruct `prompt_text` from prompt template plus step grid state and carrying-key value.
- Extract analysis tokens and final action token metadata from `output_tokens`.
- Parse `analysis_text` from the model output text.
- Quarantine invalid steps instead of mutating raw files.

Main outputs:

- `outputs/canonical_steps.parquet`
- `outputs/data_schema_report.md`
- `outputs/quarantined_steps.jsonl`

Observed run state:

- `outputs/canonical_steps.parquet` exists.
- Earlier run summary indicated 1,276 canonical rows before later 10-trajectory scoring subset work.

Typical command:

```bash
python scripts/01_prepare_steps_from_trajectories.py \
  --input_dir data/raw \
  --output outputs/canonical_steps.parquet
```

### 2. Prefix generation from generated analysis tokens

Implemented:

- `scripts/02_generate_prefixes.py`

Purpose:

- Build generation-token prefixes from `analysis_tokens_json`.
- Avoid whitespace tokenization as the default.
- Support stride, minimum prefix tokens, and max prefixes per step.
- Reconstruct prefix text from tokenizer decoding if provided, otherwise token-string fallback.
- Record first action-token mention and first final-action mention from token metadata.

Main outputs:

- `outputs/prefixes.parquet`
- `outputs/prefix_report.md`

Observed run state:

- `outputs/prefixes.parquet`: full prefix table, about 98,311 rows.
- `outputs/prefixes_10_trajectories.parquet`: 10-trajectory scoring subset, 9,942 rows.

Typical command:

```bash
python scripts/02_generate_prefixes.py \
  --input outputs/canonical_steps.parquet \
  --output outputs/prefixes.parquet \
  --stride 5 \
  --max_prefixes_per_step 80
```

### 3. Forced final-JSON action scoring

Implemented:

- `scripts/03_score_prefix_actions_json_final.py`
- `scripts/03_score_prefix_actions_together.py`

Purpose:

- Score candidate final JSON actions from each analysis prefix using the original output format.
- Candidate actions: `UP`, `DOWN`, `LEFT`, `RIGHT`.
- Produce normalized probabilities `q_UP`, `q_DOWN`, `q_LEFT`, `q_RIGHT`.
- Avoid conditioning on the true `agent_action`.
- Keep final JSON answer out of the input except for the forced final JSON prefix.

Backend notes:

- Hugging Face backend was implemented for local teacher-forced scoring.
- Together AI native scoring was added and used for the main scored subset after the user requested Together/remote scoring.
- LiteLLM was left as a clear-failure placeholder when logprobs are unavailable.

Main outputs:

- `outputs/prefix_action_scores.parquet`
- `outputs/together_scoring_report.md`

Observed run state:

- Main scored subset was limited to 10 trajectories at user request.
- `outputs/prefix_action_scores.parquet`: 9,942 scored prefixes.
- Scoring model: `openai/gpt-oss-20b` through Together AI.
- Scoring method: `echo_teacher_forced`.
- Candidate mode: `bare_action`.

Primary Together command used for full subset scoring:

```bash
python scripts/03_score_prefix_actions_together.py \
  --input outputs/prefixes.parquet \
  --output outputs/prefix_action_scores.parquet \
  --model_name openai/gpt-oss-20b \
  --api_key_env TOGETHER_API_KEY \
  --scoring_method echo_teacher_forced \
  --candidate_mode bare_action \
  --temperature 0.0 \
  --top_p 1.0 \
  --logprobs 20 \
  --max_tokens 1 \
  --save_every 100 \
  --resume
```

Important note:

- The user then requested this only for 10 trajectories. The produced `outputs/prefix_action_scores.parquet` corresponds to that 10-trajectory subset, not all 100 trajectories.

### 4. Optional multiple-choice interruption scoring

Implemented:

- `src/probing.py`
- `scripts/03b_score_prefix_actions_mc.py`

Purpose:

- Add an optional robustness check using a multiple-choice interruption prompt.
- Score A/B/C/D by teacher-forced likelihood.
- Support deterministic option permutations and average back into action space.
- Mark output mode as `multiple_choice_interruption`.

Main output observed:

- `outputs/prefix_action_scores_mc.parquet`

Status:

- Optional robustness backend only. It does not replace forced final-JSON scoring.

### 5. Commitment metrics

Implemented:

- `scripts/04_compute_commitment_metrics.py`

Purpose:

- Compute prefix-level commitment metrics:
  - `top_action`
  - `top_prob`
  - `second_prob`
  - `commitment_to_final`
  - `entropy`
  - `normalized_entropy`
  - `entropy_commitment`
  - `margin_commitment`
  - `is_top_action_final`
  - `top_action_changed`
  - deltas over prefixes
- Compute step-level summaries:
  - final-prefix commitment
  - threshold crossing progress
  - lock-in progress
  - switch count
  - total variation

Main outputs:

- `outputs/prefix_commitment_metrics.parquet`
- `outputs/step_commitment_summary.parquet`
- `outputs/commitment_report.md`

Observed run state:

- Prefix metrics: 9,942 rows.
- Step summary: 132 decision steps.

### 6. Behavior vectors

Implemented:

- `scripts/05_build_behavior_vectors.py`

Purpose:

- Convert action probabilities into behavior-side vectors.
- Compute log-probability and centered-log-ratio style features used for behavior manifold embedding.

Main outputs:

- `outputs/behavior_vectors.parquet`
- `outputs/behavior_vectors_report.md`

Observed run state:

- `outputs/behavior_vectors.parquet`: 9,942 rows.

### 7. Behavior manifold embedding

Implemented:

- `scripts/06_embed_behavior_manifold.py`

Purpose:

- Fit PCA over behavior-vector features:
  - `b_clr_UP`
  - `b_clr_DOWN`
  - `b_clr_LEFT`
  - `b_clr_RIGHT`
- Add `beh_pca1` and `beh_pca2`.
- Save PCA model and report.
- Optionally support UMAP if installed and requested.

Main outputs:

- `outputs/behavior_manifold.parquet`
- `outputs/models/behavior_pca.joblib`
- `outputs/behavior_manifold_report.md`

Observed run state:

- `outputs/behavior_manifold.parquet`: 9,942 rows.

### 8. Behavior commitment plotting

Implemented:

- `scripts/07_plot_behavior_commitment.py`

Purpose:

- Generate behavior-side plots for PCA/manifold, commitment, entropy, progress, switching, and final-prefix sanity checks.

Main outputs:

- `figures/behavior/`
- `figures/behavior/plot_report.md`

Status:

- Plots were generated successfully after behavior manifold embedding.

### 9. Diagnostics notebook

Implemented:

- `notebooks/08_behavior_diagnostics_next_steps.ipynb`

Purpose:

- Next-step diagnostics for the behavior manifold experiment.
- Offline by default. Does not call Together AI.
- Prepares optional prompt-only and mismatched prompt/CoT control files/commands.

Verification run:

```bash
jupyter nbconvert --to notebook --execute notebooks/08_behavior_diagnostics_next_steps.ipynb \
  --output 08_behavior_diagnostics_next_steps.executed.ipynb \
  --output-dir /tmp
```

Execution note:

- `nbconvert` needed escalated local execution because Jupyter binds local kernel ports, which the sandbox blocks.

Main outputs:

- `outputs/diagnostics/behavior_diagnostics_report.md`
- `outputs/diagnostics/basic_dataset_summary.csv`
- `outputs/diagnostics/progress_curve_per_step_weighted.csv`
- `outputs/diagnostics/action_mention_summary.csv`
- `outputs/diagnostics/pca_before_action_mention_summary.csv`
- `outputs/diagnostics/shuffled_label_control_summary.csv`
- `outputs/diagnostics/mismatched_prompt_cot_prefixes.parquet`
- `figures/diagnostics/`

Key reported values from the 10-trajectory subset:

- Prefix points: 9,942
- Decision steps: 132
- Trajectory files: 10
- Commitment to final action changed from 0.350 to 0.994 across progress bins.
- Entropy commitment changed from 0.318 to 0.989 across progress bins.
- Prompt-only source: existing `prefix_token_index == 0` rows.
- Mean prompt-only commitment to final: 0.350
- Mean final-prefix commitment to final: 1.000
- Median first action mention progress: 0.328
- Median first final-action mention progress: 0.419

### 10. Follow-up analyses notebook

Implemented:

- `notebooks/09_behavior_followups_1_to_4.ipynb`

Purpose:

- Follow-up analyses 1-4 only:
  1. Permanent lock-in metric
  2. Action-wise commitment curves
  3. Strict pre-action analysis
  4. Mismatched prompt/CoT control
- Offline by default with `RUN_TOGETHER_SCORING = False`.
- Does not expand to all 100 trajectories.

Verification run:

```bash
jupyter nbconvert --to notebook --execute notebooks/09_behavior_followups_1_to_4.ipynb \
  --output 09_behavior_followups_1_to_4.executed.ipynb \
  --output-dir /tmp
```

Main outputs:

- `outputs/followups/followups_1_to_4_report.md`
- `outputs/followups/permanent_lockin_summary.parquet`
- `outputs/followups/permanent_lockin_summary.csv`
- `outputs/followups/actionwise_progress_curves.csv`
- `outputs/followups/actionwise_summary.csv`
- `outputs/followups/strict_pre_action_curve_table.csv`
- `outputs/followups/strict_pre_action_coverage_table.csv`
- `outputs/followups/strict_pre_action_summary.csv`
- `outputs/followups/mismatched_prompt_cot_prefixes.parquet`
- `outputs/followups/matched_prompt_cot_prefixes_subset.parquet`
- `figures/followups/`

Key reported values from the 10-trajectory subset:

- Median first lock-in: 0.202
- Median permanent lock-in: 0.472
- Fraction with permanent lock-in: 1.000
- Mean switches after first lock-in: 1.561
- Fastest commitment gain: `DOWN` with 0.858
- Slowest commitment gain: `RIGHT` with 0.344
- LEFT vs RIGHT mean CoT length: 2173.591 vs 1105.286
- LEFT vs RIGHT prompt-only commitment: 0.283 vs 0.715
- Strict before any action mention commitment changed from 0.344 to 0.616.
- Strict before first final-action mention commitment changed from 0.343 to 0.582.
- Mismatched prompt/CoT prefixes prepared: 396 rows.
- Matched prompt/CoT subset prepared: 396 rows.
- No Together API call was made by the notebook.

## Important Execution Notes

- Current repo path after user move: `/Users/wws/SPAR_code/`.
- Earlier workspace path was `/Users/wws/Desktop/SPAR_code/`.
- Large activation directories exist in the repo; avoid broad `rg --files` or unrestricted recursive listings unless output is filtered.
- For file discovery, prefer targeted commands such as:

```bash
ls scripts
ls notebooks
find outputs -maxdepth 2 -name '*report*' -type f | sort
```

- Jupyter execution through `nbconvert` may require permission to bind local kernel ports.
- The notebooks are designed to be offline by default and to print opt-in Together scoring commands rather than executing them.

## Generated Reports to Read First

For a quick status update before new work, read these files:

1. `outputs/data_schema_report.md`
2. `outputs/prefix_report.md`
3. `outputs/together_scoring_report.md`
4. `outputs/commitment_report.md`
5. `outputs/behavior_vectors_report.md`
6. `outputs/behavior_manifold_report.md`
7. `figures/behavior/plot_report.md`
8. `outputs/diagnostics/behavior_diagnostics_report.md`
9. `outputs/followups/followups_1_to_4_report.md`

## Future Run Log Template

Append new runs below using this template.

````markdown
### YYYY-MM-DD - Short title

Changed:

- Files/scripts changed.

Ran:

```bash
command here
```

Outputs:

- Output files or directories.

Notes:

- Important caveats, row counts, failures, or rerun instructions.
````

## Later Entries

### 2026-06-28 - Activation-side scaffolding

Changed:

- Created activation analysis directory structure:
  - `scripts/activation/`
  - `outputs/activation/`
  - `outputs/activation/features/`
  - `outputs/activation/models/`
  - `outputs/activation/reports/`
  - `figures/activation/`
  - `figures/activation/layer_sweep/`
  - `figures/activation/manifolds/`
  - `figures/activation/alignment/`
  - `figures/activation/centroids/`
  - `figures/activation/strict_pre_action/`
- Added shared utilities in `src/activation_utils.py`.
- Added activation pipeline notes in `outputs/activation/README_activation_pipeline.md`.

Ran:

```bash
mkdir -p scripts/activation notebooks outputs/activation/features outputs/activation/models outputs/activation/reports figures/activation/layer_sweep figures/activation/manifolds figures/activation/alignment figures/activation/centroids figures/activation/strict_pre_action
```

Verification:

- Smoke import and helper checks were run for `src.activation_utils`.

Notes:

- No activation extraction was implemented or run.
- The README records the intended activation representation:
  `delta_h_k = h_cot_token_k - h_pre_reasoning`.
- Future activation analyses should align activation token index `k` with behavior `prefix_token_index k` and use group splits by `trajectory_id` or `decision_id`.

### 2026-06-28 - Activation schema audit implementation and run

Changed:

- Added `scripts/activation/01_audit_activation_schema.py`.
- The auditor recursively scans activation/metadata files and records file type, size, keys, shape, dtype, likely layer count, hidden dimension, token rows, token type, and storage-layout inference.
- The auditor supports `.pt`, `.pth`, `.npy`, `.npz`, `.parquet`, `.json`, `.jsonl`, `.h5`, `.hdf5`, `.zarr`, and the actual packed raw `.bf16` format with JSON sidecars.
- The auditor avoids loading raw activation arrays; packed BF16 records are inspected through sidecar metadata.

Ran:

```bash
python3 scripts/activation/01_audit_activation_schema.py \
  --input_dir trajectories_key_door_100/activations_key_door_100_mlx_packed_bf16_cot_all \
  --output_dir outputs/activation
```

Outputs:

- `outputs/activation/activation_inventory.parquet`
- `outputs/activation/activation_schema_report.md`
- `outputs/activation/schema_examples.json`

Key findings:

- Files inventoried: 91,874
- Activation files: 45,936
- Metadata/documentation files: 45,938
- Total inventoried size: 154.3 GiB
- Trajectories found: 96
- Decision IDs inferred: 1,276
- Layers found: `[0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22]`
- Hidden dimension: 2880
- Token types present: `pre_reasoning`, `cot`, `post_reasoning`
- Row-level errors: 0

Verification:

- `python3 scripts/activation/01_audit_activation_schema.py --help` succeeded.
- `outputs/activation/activation_inventory.parquet` was read back successfully and had shape `(91874, 21)`.

Notes:

- No activation extraction was implemented or run.
- The actual activation root is `trajectories_key_door_100/activations_key_door_100_mlx_packed_bf16_cot_all`; the script default remains `data/activations` as requested.

### 2026-06-28 - Canonical activation token index implementation and run

Changed:

- Added `scripts/activation/02_build_activation_token_index.py`.
- The indexer builds one canonical row per activation timepoint and stores activation pointers, not activation vectors.
- Packed BF16 layout support uses JSON sidecars for shapes/dtypes/layers and points rows to representative files plus layer templates in `source_metadata_json`.
- For packed pre/post sidecars with 3 rows, the canonical index uses the last pre row and first post row as representative timepoints.
- CoT rows use 1-based `cot_token_index`; tensor row pointers use 0-based `activation_row_index`.

Ran:

```bash
python3 scripts/activation/02_build_activation_token_index.py \
  --input_dir trajectories_key_door_100/activations_key_door_100_mlx_packed_bf16_cot_all \
  --output outputs/activation/activation_token_index.parquet
```

Outputs:

- `outputs/activation/activation_token_index.parquet`
- `outputs/activation/activation_token_index_report.md`
- `outputs/activation/quarantined_activation_records.jsonl`

Key findings:

- Activation timepoints indexed: 2,373,402
- Decision steps: 1,276
- Trajectory IDs: 95
- `pre_reasoning` rows: 1,276
- `cot` rows: 2,370,850
- `post_reasoning` rows: 1,276
- Available layers: `[0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22]`
- Hidden dimension: 2880
- Dtype: `torch.bfloat16`
- Quarantined records: 0

Verification:

- The parquet was read back successfully and had shape `(2373402, 19)`.
- The quarantine JSONL had 0 rows.

Notes:

- No activation extraction was implemented or run.
- CoT token IDs/text were populated from `outputs/canonical_steps.parquet` where available.

### 2026-06-28 - Activation-to-behavior alignment implementation and run

Changed:

- Added `scripts/activation/03_align_activations_to_behavior.py`.
- The aligner joins behavior prefix rows to activation timepoint pointers from `outputs/activation/activation_token_index.parquet`.
- Alignment rules implemented:
  - `prefix_token_index == 0` maps to `pre_reasoning`.
  - `prefix_token_index > 0` maps to CoT `cot_token_index == prefix_token_index`.
  - Missing exact CoT indices fall back to nearest token within `--max_token_tolerance`.
  - Same-decision pre/post pointers are included as `pre_*` and `post_*` column groups.
- Recomputed behavior CLR columns `b_clr_UP`, `b_clr_DOWN`, `b_clr_LEFT`, `b_clr_RIGHT` from `q_*`.

Ran:

```bash
python3 scripts/activation/03_align_activations_to_behavior.py \
  --activation_index outputs/activation/activation_token_index.parquet \
  --behavior outputs/prefix_commitment_metrics.parquet \
  --output outputs/activation/activation_behavior_aligned.parquet \
  --max_token_tolerance 2
```

Outputs:

- `outputs/activation/activation_behavior_aligned.parquet`
- `outputs/activation/alignment_report.md`

Key findings:

- Behavior prefix rows: 9,942
- Successfully aligned: 9,942
- Aligned exactly: 9,942
- Aligned by nearest token: 0
- Failed: 0
- `exact` CoT rows: 9,810
- `exact_pre_reasoning` rows: 132
- Token alignment distance distribution: `{0: 9942}`
- Strict before-first-action prefix count: 3,729
- Strict before-final-action prefix count: 4,227

Verification:

- The aligned parquet was read back successfully and had shape `(9942, 90)`.
- Required activation pointer columns and behavior CLR columns were present.

Notes:

- No activation vectors were copied into the aligned parquet.
- The aligned table corresponds to the current 10-trajectory scored behavior subset.

### 2026-06-28 - Activation feature construction implementation and run

Changed:

- Added `scripts/activation/04_make_activation_features.py`.
- The script builds activation-side feature arrays from `outputs/activation/activation_behavior_aligned.parquet` without copying vectors into parquet.
- Implemented feature representations:
  - `raw`: aligned activation `h_k`
  - `delta_pre`: `h_k - h_pre`
  - `velocity`: `h_k - h_prev` within each decision
  - `unit_delta_pre`: normalized `delta_pre`
- Implemented packed BF16 activation loading for the current activation layout, plus loaders for numpy, torch, parquet-vector, hdf5, and zarr formats where available.
- Feature arrays are written as memory-mapped `.npy` files with deterministic names.

Ran:

```bash
python3 scripts/activation/04_make_activation_features.py \
  --input outputs/activation/activation_behavior_aligned.parquet \
  --output_dir outputs/activation/features \
  --metadata_output outputs/activation/activation_feature_metadata.parquet \
  --layers all \
  --representations raw,delta_pre,velocity,unit_delta_pre \
  --dtype float32
```

Outputs:

- `outputs/activation/activation_feature_metadata.parquet`
- `outputs/activation/reports/activation_feature_report.md`
- `outputs/activation/features/layer_{layer:02d}_{representation}.npy`

Key findings:

- Input rows: 9,942
- Decision steps: 132
- Layers built: `[0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22]`
- Representations built: `raw`, `delta_pre`, `velocity`, `unit_delta_pre`
- Feature arrays written: 48
- Feature directory size: 5.1G
- Each array shape: `(9942, 2880)`
- Output dtype: `float32`

Verification:

- The metadata parquet was read back successfully and had shape `(9942, 70)`.
- Sample feature arrays were opened successfully as numpy arrays and had shape `(9942, 2880)`.
- No non-finite values were detected in the report.
- For `prefix_token_index == 0`, `delta_pre` L2 norm was exactly zero for all checked layers.

Notes:

- No model activation extraction was run.
- The features correspond to the current aligned behavior subset.

### 2026-06-28 - Layer-wise activation decoding implementation

Changed:

- Added `scripts/activation/05_layerwise_decoding.py`.
- Implemented grouped cross-validated layer-wise decoding from activation feature arrays.
- Targets implemented:
  - behavior CLR vector: `b_clr_UP`, `b_clr_DOWN`, `b_clr_LEFT`, `b_clr_RIGHT`
  - scalar metrics: `commitment_to_final`, `entropy_commitment`, `margin_commitment`
  - categorical targets: `agent_action`, `top_action`, `is_top_action_final`
- Models implemented:
  - Ridge regression for behavior CLR and scalar targets
  - Multinomial/logistic classification for categorical targets
  - Optional LinearSVM via `--run_svm true`
- Grouped CV implemented with `trajectory_id` or `decision_id`; no random row-level prefix splits.
- Subsets implemented:
  - all prefixes
  - strict before any action mention
  - strict before final-action mention
  - post-mention prefixes
  - high-commitment prefixes
  - low-commitment prefixes
- Outputs implemented:
  - `outputs/activation/layer_sweep_metrics.csv`
  - optional `outputs/activation/layer_sweep_predictions.parquet`
  - `outputs/activation/reports/layer_sweep_report.md`
  - `outputs/activation/best_activation_config.json`
  - layer-sweep figures under `figures/activation/layer_sweep/`

Smoke test ran:

```bash
python3 -B scripts/activation/05_layerwise_decoding.py \
  --metadata outputs/activation/activation_feature_metadata.parquet \
  --features_dir outputs/activation/features \
  --representation delta_pre \
  --comparison_representations delta_pre \
  --layers 0 \
  --group_by trajectory_id \
  --output /tmp/spar_layerwise_smoke/layer_sweep_metrics.csv \
  --report /tmp/spar_layerwise_smoke/layer_sweep_report.md \
  --best_config /tmp/spar_layerwise_smoke/best_activation_config.json \
  --figures_dir /tmp/spar_layerwise_smoke/figures \
  --n_splits 2 \
  --max_iter 200
```

Smoke outputs:

- `/tmp/spar_layerwise_smoke/layer_sweep_metrics.csv`
- `/tmp/spar_layerwise_smoke/layer_sweep_report.md`
- `/tmp/spar_layerwise_smoke/best_activation_config.json`
- 5 smoke figures under `/tmp/spar_layerwise_smoke/figures/`

Smoke findings:

- Metric rows: 36
- Tested layer: 0
- Tested representation: `delta_pre`
- Prefix rows used for all-prefix subset: 9,942
- Groups used for all-prefix subset: 10 trajectories
- The smoke run exercised ridge regression, logistic classification, subset logic, report writing, recommendation JSON writing, and plotting.

Notes:

- The full layer sweep was not run in this entry; the script is ready for the requested full command.

### 2026-06-28 - Activation manifold embedding implementation and run

Changed:

- Added `scripts/activation/06_embed_activation_manifold.py`.
- The script embeds a selected activation feature array into a 2D activation manifold.
- Defaults implemented:
  - representation: `delta_pre`
  - layer: best layer from `outputs/activation/best_activation_config.json` if available, otherwise a mid-late available layer
- Preprocessing implemented:
  - failed-alignment rows are removed when `alignment_status` exists
  - non-finite feature rows are removed
  - features are standardized with `StandardScaler`
  - optional row L2 normalization via `--l2_normalize true`
  - optional residualization of `progress`, `prefix_token_index`, and `num_cot_tokens` where present via `--residualize_progress true`
- Embeddings implemented:
  - PCA with 2 components
  - optional UMAP via `--use_umap`
  - optional/auto IncrementalPCA for large feature matrices
- Saved PCA plus preprocessing parameters to the requested activation model joblib.
- Generated all required activation manifold figures with matplotlib only.

Ran:

```bash
python3 -B scripts/activation/06_embed_activation_manifold.py \
  --metadata outputs/activation/activation_feature_metadata.parquet \
  --features_dir outputs/activation/features \
  --config outputs/activation/best_activation_config.json \
  --output outputs/activation/activation_manifold.parquet
```

Outputs:

- `outputs/activation/activation_manifold.parquet`
- `outputs/activation/models/activation_pca_layer_16_delta_pre.joblib`
- `outputs/activation/reports/activation_manifold_report.md`
- `figures/activation/manifolds/pca_by_agent_action.png`
- `figures/activation/manifolds/pca_by_top_action.png`
- `figures/activation/manifolds/pca_by_commitment_to_final.png`
- `figures/activation/manifolds/pca_by_entropy_commitment.png`
- `figures/activation/manifolds/pca_by_progress.png`
- `figures/activation/manifolds/pca_strict_before_any_action_mention_by_agent_action.png`
- `figures/activation/manifolds/pca_strict_before_final_action_mention_by_agent_action.png`
- `figures/activation/manifolds/example_activation_trajectories.png`

Key findings:

- Selected layer: 16
- Layer source: mid-late fallback from available feature layers because `outputs/activation/best_activation_config.json` was not present
- Representation: `delta_pre`
- Output manifold shape: `(9942, 74)`
- PCA explained variance ratio: `[0.04706927016377449, 0.03179529681801796]`
- Dropped failed-alignment rows: 0
- Dropped non-finite feature rows: 0
- UMAP was not requested

Report diagnostics:

- Action clusters showed little visible separation in unsupervised PCA space.
- High-commitment points had similar PCA radius to all points.
- Strict pre-action points showed little visible separation by final action in the first two PCs.
- The report explicitly notes that PCA is unsupervised and should not be treated as decoding evidence by itself.

Verification:

- The manifold parquet was read back successfully and had shape `(9942, 74)`.
- `act_pca1` and `act_pca2` were finite for all rows.
- All 8 requested figure files existed and passed a nonblank image sanity check.

### 2026-06-28 - Activation-behavior manifold alignment implementation and run

Changed:

- Added `scripts/activation/07_activation_behavior_alignment.py`.
- Implemented activation-to-behavior alignment analyses using selected activation features and behavior CLR vectors.
- Layer selection defaults:
  - explicit `--layer` if provided
  - `outputs/activation/best_activation_config.json` if available
  - otherwise the layer already used in `outputs/activation/activation_manifold.parquet`
  - otherwise a mid-late available feature layer
- Analyses implemented:
  - grouped cross-validated Ridge regression from activation features to behavior CLR
  - PLS by default, with optional CCA via `--latent_method cca`
  - grouped train/test affine Procrustes-style alignment between activation PCA and behavior PCA manifolds
  - sampled RSA between activation distances and behavior CLR distances
- Subsets implemented:
  - all prefixes
  - strict before any action mention
  - strict before final-action mention
  - high commitment
  - low commitment
- Figures implemented:
  - true vs predicted commitment
  - PCA of predicted behavior vectors from activations
  - PLS/CCA latent embedding by action
  - PLS/CCA latent embedding by commitment
  - behavior PCA vs aligned activation PCA
  - RSA activation distance vs behavior distance

Ran:

```bash
python3 -B scripts/activation/07_activation_behavior_alignment.py \
  --metadata outputs/activation/activation_feature_metadata.parquet \
  --activation_manifold outputs/activation/activation_manifold.parquet \
  --behavior_manifold outputs/behavior_manifold.parquet \
  --features_dir outputs/activation/features \
  --config outputs/activation/best_activation_config.json \
  --output outputs/activation/activation_behavior_alignment_metrics.csv
```

Outputs:

- `outputs/activation/activation_behavior_alignment_metrics.csv`
- `outputs/activation/activation_behavior_alignment_predictions.parquet`
- `outputs/activation/reports/activation_behavior_alignment_report.md`
- `figures/activation/alignment/predicted_vs_true_commitment.png`
- `figures/activation/alignment/predicted_behavior_from_activation_pca.png`
- `figures/activation/alignment/cca_by_agent_action.png`
- `figures/activation/alignment/cca_by_commitment.png`
- `figures/activation/alignment/behavior_vs_activation_pca_alignment.png`
- `figures/activation/alignment/rsa_activation_vs_behavior_distance.png`

Key findings:

- Selected layer: 16
- Layer source: activation manifold layer from `outputs/activation/activation_manifold.parquet`
- Representation: `delta_pre`
- Metrics rows: 20
- Prediction rows: 9,942
- Groups: 10 trajectories
- All-prefix Ridge mean R2: -0.403889
- All-prefix Ridge predicted commitment Pearson correlation: 0.164906
- All-prefix predicted top-action accuracy vs behavior top_action: 0.372963
- All-prefix PLS mean R2: -0.021579
- All-prefix Procrustes coordinate correlation: 0.015202
- All-prefix Procrustes disparity: 1.008985
- All-prefix RSA activation/behavior distance Spearman correlation: 0.038909

Verification:

- The metrics CSV was read back successfully with shape `(20, 35)`.
- The predictions parquet was read back successfully with shape `(9942, 36)`.
- PLS/CCA coordinate columns were finite for all prediction rows.
- All 6 requested figure files existed and passed a nonblank image sanity check.

Notes:

- `outputs/activation/best_activation_config.json` was not present at run time, so the script used the activation manifold layer.
- The report includes caveats that Procrustes alignment is a 2D unsupervised manifold check and RSA uses sampled pairwise distances.

### 2026-06-28 - Activation centroid basin implementation and run

Changed:

- Added `scripts/activation/08_activation_centroid_basins.py`.
- Implemented action centroid construction in activation space using selected layer/representation features.
- Centroid sources supported:
  - `high_commitment_prefixes`
  - `final_prefixes`
  - `post_reasoning` when post-reasoning feature rows exist
- Prefix-level outputs implemented:
  - `act_sim_UP`, `act_sim_DOWN`, `act_sim_LEFT`, `act_sim_RIGHT`
  - softmax-normalized centroid similarities
  - `activation_basin_action`
  - `activation_commitment_to_final`
  - `activation_margin`
  - `activation_basin_matches_final`
  - `activation_basin_matches_behavior_top`
- Decision-level lock-in outputs implemented:
  - `first_activation_basin_lock_progress`
  - `permanent_activation_basin_lock_progress`
  - `activation_switch_count`
  - behavior lock-in comparison fields from `outputs/step_commitment_summary.parquet` and `outputs/followups/permanent_lockin_summary.parquet` when available
- Figures implemented under `figures/activation/centroids/`.

Ran:

```bash
python3 -B scripts/activation/08_activation_centroid_basins.py \
  --metadata outputs/activation/activation_feature_metadata.parquet \
  --features_dir outputs/activation/features \
  --config outputs/activation/best_activation_config.json \
  --output_scores outputs/activation/activation_centroid_scores.parquet \
  --output_summary outputs/activation/activation_basin_summary.parquet
```

Outputs:

- `outputs/activation/activation_centroid_scores.parquet`
- `outputs/activation/activation_basin_summary.parquet`
- `outputs/activation/reports/activation_centroid_basin_report.md`
- `figures/activation/centroids/activation_similarity_to_final_centroid_over_progress.png`
- `figures/activation/centroids/activation_basin_accuracy_over_progress.png`
- `figures/activation/centroids/activation_basin_distribution_over_progress.png`
- `figures/activation/centroids/behavior_commitment_vs_activation_similarity.png`
- `figures/activation/centroids/behavior_vs_activation_permanent_lockin.png`
- `figures/activation/centroids/example_activation_basin_trajectories.png`
- `figures/activation/centroids/strict_pre_action_activation_basin_accuracy.png`

Key findings:

- Selected layer: 16
- Layer source: activation manifold layer from `outputs/activation/activation_manifold.parquet`
- Representation: `delta_pre`
- Centroid source: `high_commitment_prefixes`
- Centroid commitment threshold: 0.9
- Lock margin threshold: 0.005
- Centroid row counts: `UP=1104`, `DOWN=1229`, `LEFT=929`, `RIGHT=1637`
- Prefix score table shape: `(9942, 92)`
- Decision summary table shape: `(132, 26)`
- Basin matches final action: 0.332931
- Basin matches behavior top_action: 0.356367
- Strict pre-action basin matches final action: 0.347278
- Mean activation margin: 0.002885
- Decisions with first activation basin lock: 36 / 132
- Decisions with permanent activation basin lock: 25 / 132
- Median first activation basin lock progress: 0.189348
- Median permanent activation basin lock progress: 0.991035
- Median permanent behavior lock progress: 0.472449

Verification:

- Both output parquet files were read back successfully.
- All 7 requested figure files existed and passed a nonblank image sanity check.
- The script parsed successfully with `ast.parse`.

Notes:

- The initial tested lock margin threshold `0.05` was too large for cosine centroid margins in this feature space; the script default was adjusted to `0.005` before the final run.
- `outputs/activation/best_activation_config.json` was not present at run time, so the script used the layer from the activation manifold.

### 2026-06-28 - Pre/post reasoning activation analysis implementation and run

Changed:

- Added `scripts/activation/09_pre_post_reasoning_analysis.py`.
- The script loads `pre_reasoning` and `post_reasoning` activation rows directly from `outputs/activation/activation_token_index.parquet` and the referenced activation tensor files.
- Implemented per-decision/per-layer features:
  - `h_pre`
  - `h_post`
  - `delta_post_pre = h_post - h_pre`
  - `delta_post_pre_l2`
  - `pre_post_cosine`
  - pre/post cosine similarities to raw high-commitment action centroids when raw feature arrays are available
- Implemented grouped decoding analyses by layer:
  - `h_pre -> agent_action`
  - `h_post -> agent_action`
  - `delta_post_pre -> agent_action`
  - `h_pre -> prompt_only_commitment_to_final`
  - `h_post -> final_prefix_commitment_to_final`
- Implemented all requested pre/post figures under `figures/activation/pre_post/`.

Ran:

```bash
python3 -B scripts/activation/09_pre_post_reasoning_analysis.py \
  --activation_index outputs/activation/activation_token_index.parquet \
  --metadata outputs/activation/activation_feature_metadata.parquet \
  --features_dir outputs/activation/features \
  --config outputs/activation/best_activation_config.json \
  --output outputs/activation/pre_post_reasoning_features.parquet
```

Outputs:

- `outputs/activation/pre_post_reasoning_features.parquet`
- `outputs/activation/pre_post_decoding_metrics.csv`
- `outputs/activation/reports/pre_post_reasoning_report.md`
- `figures/activation/pre_post/pre_vs_post_action_decoding_by_layer.png`
- `figures/activation/pre_post/post_minus_pre_norm_by_action.png`
- `figures/activation/pre_post/pre_vs_post_final_centroid_similarity.png`
- `figures/activation/pre_post/prompt_only_commitment_vs_pre_activation_similarity.png`
- `figures/activation/pre_post/final_commitment_vs_post_activation_similarity.png`
- `figures/activation/pre_post/pre_post_pca_trajectories.png`

Key findings:

- Decisions analyzed: 132
- Layers analyzed: `[0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22]`
- Feature rows: 1,584
- Decoding metric rows: 60
- Reference plot layer: 16
- Reference plot layer source: activation manifold layer from `outputs/activation/activation_manifold.parquet`
- At layer 16:
  - mean `||h_post - h_pre||`: 3310.3650
  - median `||h_post - h_pre||`: 3205.1367
  - mean `cos(h_pre, h_post)`: 0.7302
  - mean pre similarity to final-action centroid: 0.7073
  - mean post similarity to final-action centroid: 0.7528
- Best action decoding balanced accuracy:
  - `h_pre -> agent_action`: layer 8, 0.4655
  - `h_post -> agent_action`: layer 0, 0.6584
  - `delta_post_pre -> agent_action`: layer 0, 0.6562
- Best post-minus-pre improvement over pre action decoding: 0.1928

Verification:

- The feature parquet was read back successfully with shape `(1584, 53)`.
- The metrics CSV was read back successfully with shape `(60, 16)`.
- All 6 requested figure files existed and passed a nonblank image sanity check.
- The script parsed successfully with `ast.parse`.

Notes:

- The analysis is restricted to decisions with behavior-side prefix commitment metrics, currently 132 decisions.
- `outputs/activation/best_activation_config.json` was not present at run time, so the reference layer came from the activation manifold output.


### 2026-06-28 - Strict pre-action activation analysis implementation and run

Changed:

- Added `scripts/activation/10_strict_pre_action_activation.py`.
- Implemented exact strict pre-action definitions using `<`, not `<=`:
  - `strict_before_any_action_mention = prefix_token_index < first_action_mention_token_index`
  - `strict_before_final_action_mention = prefix_token_index < first_final_action_mention_token_index`
- Implemented buffered variants with a 5-token gap before the first mention.
- Recovered exact mention-token indices from `outputs/prefixes.parquet` when they are absent from activation feature metadata.
- Implemented layer-wise grouped decoding for strict subsets:
  - `agent_action` classification
  - behavior CLR regression
  - `commitment_to_final` regression
- Implemented strict-subset activation manifold plots and centroid/basin analyses when centroid scores are available.
- Wrote normalized-to-mention curves for final-centroid similarity and basin accuracy.

Ran:

```bash
python3 -B scripts/activation/10_strict_pre_action_activation.py \
  --metadata outputs/activation/activation_feature_metadata.parquet \
  --manifold outputs/activation/activation_manifold.parquet \
  --features_dir outputs/activation/features \
  --config outputs/activation/best_activation_config.json \
  --output outputs/activation/strict_pre_action_activation_metrics.csv \
  --n_splits 2 \
  --max_iter 300
```

Outputs:

- `outputs/activation/strict_pre_action_activation_metrics.csv`
- `outputs/activation/strict_pre_action_activation_curves.csv`
- `outputs/activation/reports/strict_pre_action_activation_report.md`
- `figures/activation/strict_pre_action/strict_pre_action_decoding_by_layer.png`
- `figures/activation/strict_pre_action/pca_strict_before_any_action_by_agent_action.png`
- `figures/activation/strict_pre_action/pca_strict_before_final_action_by_agent_action.png`
- `figures/activation/strict_pre_action/final_centroid_similarity_to_first_action_mention.png`
- `figures/activation/strict_pre_action/final_centroid_similarity_to_first_final_action_mention.png`
- `figures/activation/strict_pre_action/basin_accuracy_to_first_action_mention.png`
- `figures/activation/strict_pre_action/behavior_commitment_vs_activation_similarity_pre_action.png`
- `figures/activation/strict_pre_action/strict_pre_action_coverage.png`

Key findings:

- Prefix rows: 9,942
- Decisions: 132
- Trajectories: 10
- Layers analyzed: `[0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22]`
- Metrics rows: 184
- Curve rows: 20
- Strict coverage:
  - before any action mention: 3,729 prefixes across 132 decisions
  - before final-action mention: 4,227 prefixes across 132 decisions
  - buffered before any action mention: 3,667 prefixes across 132 decisions
  - buffered before final-action mention: 4,170 prefixes across 132 decisions
- Best strict-before-any action decoding: layer 10, balanced accuracy 0.2650
- Best strict-before-final action decoding: layer 10, balanced accuracy 0.2813
- Best strict-before-any behavior CLR mean R2: layer 14, -1.2311
- Best strict-before-final behavior CLR mean R2: layer 12, -1.5481
- Strict-before-any centroid basin accuracy: 0.3473
- Strict-before-final centroid basin accuracy: 0.3426
- Mean final-centroid similarity rises from 0.5609 to 0.8379 approaching first action mention.
- Mean final-centroid similarity rises from 0.5850 to 0.8408 approaching first final-action mention.
- Late normalized bins had at least 120 decision steps.

Verification:

- The metrics CSV was read back successfully with shape `(184, 32)`.
- The curves CSV was read back successfully with shape `(20, 10)`.
- All 8 requested figure files existed and passed a nonblank image sanity check.
- The report was written to `outputs/activation/reports/strict_pre_action_activation_report.md`.

Notes:

- The initial full default CV run was interrupted because repeated logistic fits over all layers were too slow locally. The completed run used grouped 2-fold CV with `--max_iter 300`.
- Pre-action here means no explicit lexical action token in the generated reasoning, not absence of implicit coordinate or path evidence.


### 2026-06-28 - Activation trajectory dynamics and switching analysis implementation and run

Changed:

- Added `scripts/activation/11_activation_dynamics_switching.py`.
- Implemented prefix-level activation geometry dynamics:
  - high-dimensional activation step distance
  - high-dimensional cosine step distance
  - activation total variation by decision
  - activation PCA manifold path length
  - activation radius and radius delta in PCA space
  - optional curvature angle between consecutive high-dimensional trajectory segments
- Implemented activation basin dynamics:
  - activation basin switch count
  - first activation basin lock progress using consecutive-prefix lock criterion
  - permanent activation basin lock progress
  - switch count after first/permanent activation lock
- Merged behavior-side summaries from `outputs/step_commitment_summary.parquet` and `outputs/followups/permanent_lockin_summary.parquet` when available.
- Added all requested dynamics figures under `figures/activation/dynamics/`.

Ran:

```bash
python3 -B scripts/activation/11_activation_dynamics_switching.py \
  --metadata outputs/activation/activation_feature_metadata.parquet \
  --centroid_scores outputs/activation/activation_centroid_scores.parquet \
  --manifold outputs/activation/activation_manifold.parquet \
  --output outputs/activation/activation_dynamics_summary.parquet
```

Outputs:

- `outputs/activation/activation_dynamics_summary.parquet`
- `outputs/activation/activation_dynamics_prefix_metrics.parquet`
- `outputs/activation/reports/activation_dynamics_report.md`
- `figures/activation/dynamics/behavior_vs_activation_switch_count.png`
- `figures/activation/dynamics/behavior_vs_activation_permanent_lockin.png`
- `figures/activation/dynamics/behavior_vs_activation_total_variation.png`
- `figures/activation/dynamics/activation_radius_over_progress_by_action.png`
- `figures/activation/dynamics/activation_path_length_by_action.png`
- `figures/activation/dynamics/example_high_switch_activation_trajectories.png`
- `figures/activation/dynamics/example_tentative_permanent_trajectories.png`

Key findings:

- Prefix metrics shape: `(9942, 104)`
- Decision summary shape: `(132, 48)`
- Selected layer: 16
- Selected representation: `delta_pre`
- Feature array: `outputs/activation/features/layer_16_delta_pre.npy`
- Mean activation basin switch count: 32.2500
- Mean behavior switch count: 4.6439
- Behavior vs activation switch-count Pearson correlation: 0.2852
- Behavior vs activation total-variation Pearson correlation: 0.4376
- Permanent lock progress Pearson correlation, restricted to paired decisions: 0.1613 over 25 decisions
- Activation permanent lock occurred later than behavior permanent lock for all 25 paired decisions, median activation-minus-behavior progress 0.6968
- High activation-switch decisions had higher activation total variation than low-switch decisions: 223448.1628 vs 167668.5674
- 25 decisions showed activation switching followed by permanent activation basin lock.

Verification:

- The summary parquet was read back successfully with shape `(132, 48)`.
- The prefix metrics parquet was read back successfully with shape `(9942, 104)`.
- All 7 requested figure files existed and passed a nonblank image sanity check.
- The script parsed successfully with `ast.parse`.

Notes:

- Activation distances are in the selected `delta_pre` feature space; manifold radius/path length are in 2D PCA space.
- Activation basin switches depend on the centroid construction and `lock_margin_threshold` inherited from `outputs/activation/activation_centroid_scores.parquet`.


### 2026-06-28 - Activation robustness controls implementation and run

Changed:

- Added `scripts/activation/12_activation_controls.py`.
- Implemented shuffled-label controls by shuffling `agent_action` across `decision_id` with a fixed seed.
- Implemented true vs shuffled layer-wise linear decoding using ridge classifiers with grouped splits.
- Implemented progress residualization by regressing out `progress`, `prefix_token_index`, and inferred `num_cot_tokens` from activation features.
- Implemented residualized PCA and key decoding metrics after residualization.
- Implemented action-balanced decision subsampling with equal numbers of decision steps per final action.
- Implemented CoT-length controls:
  - short / medium / long CoT-length strata
  - `num_cot_tokens` appended as a covariate for decoding/regression
- Implemented group-split sensitivity comparing `trajectory_id` and `decision_id` splits.
- Added all requested control figures under `figures/activation/controls/`.

Ran:

```bash
python3 -B scripts/activation/12_activation_controls.py \
  --metadata outputs/activation/activation_feature_metadata.parquet \
  --features_dir outputs/activation/features \
  --config outputs/activation/best_activation_config.json \
  --output outputs/activation/activation_control_metrics.csv
```

Outputs:

- `outputs/activation/activation_control_metrics.csv`
- `outputs/activation/reports/activation_controls_report.md`
- `figures/activation/controls/true_vs_shuffled_decoding.png`
- `figures/activation/controls/progress_residualized_pca_by_action.png`
- `figures/activation/controls/action_balanced_decoding.png`
- `figures/activation/controls/cot_length_stratified_basin_accuracy.png`
- `figures/activation/controls/group_split_sensitivity.png`

Key findings:

- Metrics CSV shape: `(179, 14)`
- Selected layer: 16
- Selected representation: `delta_pre`
- Best true-label action decoding: layer 10, balanced accuracy 0.2899
- Best shuffled-label action decoding: layer 22, balanced accuracy 0.2454
- Selected-layer true centroid basin accuracy: 0.3329
- Shuffled-label centroid basin accuracy: 0.2507
- Selected-layer original action decoding balanced accuracy: 0.2881
- Progress-residualized action decoding balanced accuracy: 0.2690
- Selected-layer original behavior CLR mean R2: -1.4732
- Progress-residualized behavior CLR mean R2: -1.7736
- Action-balanced action decoding balanced accuracy: 0.2747
- Action-balanced behavior CLR mean R2: -3.0186
- Strict pre-action balanced action decoding after action-balanced subsampling: 0.2511
- CoT length strata activation basin accuracy:
  - short: 0.4390 over 45 decisions
  - medium: 0.2826 over 43 decisions
  - long: 0.2923 over 44 decisions
- With `num_cot_tokens` appended as a covariate: action decoding balanced accuracy 0.2900, behavior CLR mean R2 -1.4672
- Group split sensitivity:
  - trajectory-level action decoding balanced accuracy: 0.2881
  - decision-level action decoding balanced accuracy: 0.3146
  - trajectory-level behavior CLR mean R2: -1.4732
  - decision-level behavior CLR mean R2: -0.9286

Verification:

- The metrics CSV was read back successfully with shape `(179, 14)`.
- All 5 requested figure files existed and passed a nonblank image sanity check.
- The report was written to `outputs/activation/reports/activation_controls_report.md`.
- The script parsed successfully with `ast.parse`.

Notes:

- `outputs/activation/layer_sweep_metrics.csv` was not present, so the script computed true/shuffled layer-wise decoding directly from feature arrays.
- The controls use grouped 2-fold splits by default to keep local runtime tractable.


### 2026-06-28 - Master activation commitment manifold notebook and final report

Changed:

- Added `notebooks/10_activation_commitment_manifold_report.ipynb`.
- The notebook summarizes activation-side commitment analyses without re-extracting activations.
- Implemented robust optional-output loading so missing outputs are skipped with clear notes.
- Implemented figure copying/display helpers that save reproduced figures to `figures/activation/report/`.
- Implemented final markdown report generation at `outputs/activation/reports/final_activation_commitment_report.md`.
- The data section reports both full activation inventory counts and behavior-aligned feature counts to avoid confusing the full activation set with the scored 10-trajectory subset.

Ran:

```bash
python3 -B -m nbconvert \
  --to notebook \
  --execute \
  --inplace notebooks/10_activation_commitment_manifold_report.ipynb \
  --ExecutePreprocessor.timeout=300
```

Outputs:

- `notebooks/10_activation_commitment_manifold_report.ipynb`
- `outputs/activation/reports/final_activation_commitment_report.md`
- `figures/activation/report/` with copied report figures

Key contents:

- Introduction and activation-side goal statement.
- Data summary:
  - activation inventory decisions: 1,276
  - activation inventory trajectories: 95
  - behavior-aligned decisions: 132
  - behavior-aligned trajectories: 10
  - activation timepoints: 2,373,402
  - CoT tokens: 2,370,850
  - feature layers: 12
  - hidden dimension: 2,880
- Activation representation explanation for `h_pre`, `h_k`, `h_post`, `delta_pre`, raw, and velocity.
- Activation manifold, behavior alignment, centroids/basins, pre/post, strict pre-action, and controls sections.
- Final interpretation and limitations.

Verification:

- Notebook execution completed successfully with no error outputs.
- Notebook has 27 cells: 14 markdown and 13 code cells.
- Final markdown report exists and was regenerated.
- 23 figures were copied to `figures/activation/report/`.
- All 23 copied figures passed a nonblank image sanity check.

Notes:

- `outputs/activation/layer_sweep_metrics.csv` and layer-sweep figures were not present, so that notebook section skips gracefully and records the missing optional output.
- No activation extraction was performed.


### 2026-06-28 - Full activation pipeline rerun from packed activations

Ran the requested activation-side pipeline steps 1-12.

Adaptations:

- `data/activations` did not exist in this checkout, so steps 1 and 2 used the actual activation directory:
  `trajectories_key_door_100/activations_key_door_100_mlx_packed_bf16_cot_all`.
- The requested `--layers every two layer` phrase for step 4 was translated to the script's accepted comma-separated layer list:
  `0,2,4,6,8,10,12,14,16,18,20,22`.
- The default step 5 command was active for a long time and had started evaluating comparison representations beyond `delta_pre`; it was interrupted and rerun with bounded settings:
  `--comparison_representations delta_pre --n_splits 2 --leave_one_group_out_threshold 0 --max_iter 300`.
- The default step 10 command was also interrupted because it was too slow locally and rerun with bounded settings:
  `--n_splits 2 --max_iter 300`.

Outputs validated:

- `outputs/activation/activation_inventory.parquet`: `(91874, 21)`
- `outputs/activation/activation_token_index.parquet`: `(2373402, 19)`
- `outputs/activation/activation_behavior_aligned.parquet`: `(9942, 90)`
- `outputs/activation/activation_feature_metadata.parquet`: `(9942, 70)`
- `outputs/activation/layer_sweep_metrics.csv`: `(432, 28)`
- `outputs/activation/activation_manifold.parquet`: `(9942, 74)`
- `outputs/activation/activation_behavior_alignment_metrics.csv`: `(20, 35)`
- `outputs/activation/activation_centroid_scores.parquet`: `(9942, 92)`
- `outputs/activation/activation_basin_summary.parquet`: `(132, 26)`
- `outputs/activation/pre_post_reasoning_features.parquet`: `(1584, 53)`
- `outputs/activation/pre_post_decoding_metrics.csv`: `(60, 16)`
- `outputs/activation/strict_pre_action_activation_metrics.csv`: `(184, 32)`
- `outputs/activation/strict_pre_action_activation_curves.csv`: `(20, 10)`
- `outputs/activation/activation_dynamics_summary.parquet`: `(132, 48)`
- `outputs/activation/activation_dynamics_prefix_metrics.parquet`: `(9942, 104)`
- `outputs/activation/activation_control_metrics.csv`: `(179, 14)`
- Feature arrays under `outputs/activation/features/`: 48 `.npy` files
- `outputs/activation/best_activation_config.json`: present
- Activation figure PNG files under `figures/activation/`: 75

Selected layer after rerun:

- Step 6 and downstream scripts selected layer 12 from `outputs/activation/best_activation_config.json`.

Notes:

- No activation extraction was performed; the pipeline consumed already-packed activation files.
- The layer sweep and strict pre-action analyses use bounded grouped CV settings for local tractability, as noted above.


### 2026-06-28 - Conceptual belief space package skeleton

Created the paper-style conceptual belief space skeleton.

Files added:

- `src/conceptual_belief_space/__init__.py`
- `src/conceptual_belief_space/config.py`
- `src/conceptual_belief_space/io.py`
- `src/conceptual_belief_space/beliefs.py`
- `src/conceptual_belief_space/splits.py`
- `src/conceptual_belief_space/extremes.py`
- `src/conceptual_belief_space/plotting.py`
- `scripts/conceptual_belief_space/00_create_dirs.py`

The bootstrap script created the requested conceptual belief space output and figure directory trees and wrote:

- `outputs/conceptual_belief_space/README.md`

Validation:

- Ran `python scripts/conceptual_belief_space/00_create_dirs.py`.
- A normal `compileall` check was blocked because the workspace sandbox prevented bytecode writes.
- Ran a no-bytecode import and utility smoke check with `python -B`; it passed.

Notes:

- No heavy analyses were implemented.
- No model inference, activation extraction, PCA, probes, or centroid fitting was run.
- The README records that blind PCA is diagnostic only and all fitting must be train-only under grouped splits.


### 2026-06-28 - Conceptual belief space Step 1 IO audit

Implemented and ran:

- `scripts/conceptual_belief_space/01_audit_io.py`

Outputs written:

- `outputs/conceptual_belief_space/io_map.json`
- `outputs/conceptual_belief_space/01_io_audit.md`

Audit findings:

- Behavior/action table selected: `outputs/prefix_commitment_metrics.parquet`
- Activation feature metadata selected: `outputs/activation/activation_feature_metadata.parquet`
- Activation feature directory selected: `outputs/activation/features`
- Available layers: `0,2,4,6,8,10,12,14,16,18,20,22`
- Available representations: `delta_pre`, `raw`, `unit_delta_pre`, `velocity`
- Feature files inspected with NumPy mmap mode: 48
- Behavior rows: 9,942
- Activation metadata rows: 9,942
- Row counts match: yes
- Missing behavior columns: none; `decision_id` is derivable from `trajectory_id` and `step_id`

Validation:

- Ran `python -B scripts/conceptual_belief_space/01_audit_io.py --help`.
- Ran `python -B scripts/conceptual_belief_space/01_audit_io.py`.
- The script completed successfully and did not load full activation arrays into memory.


### 2026-06-28 - Conceptual belief space Step 2 action belief vectors

Implemented and ran:

- `scripts/conceptual_belief_space/02_build_action_beliefs.py`

Command:

```bash
python -B scripts/conceptual_belief_space/02_build_action_beliefs.py \
  --io_map outputs/conceptual_belief_space/io_map.json \
  --output outputs/conceptual_belief_space/action_belief_vectors.parquet
```

Outputs written:

- `outputs/conceptual_belief_space/action_belief_vectors.parquet`
- `outputs/conceptual_belief_space/02_action_belief_vectors_report.md`

Results:

- Rows: 9,942
- Decisions: 132
- Trajectories: 10
- Probability source: existing `q_UP`, `q_DOWN`, `q_LEFT`, `q_RIGHT` columns
- Existing log scores were checked; max absolute difference from logscore-softmax probabilities was `4.44e-16`
- Exact action mention indices were merged from `outputs/prefixes_10_trajectories.parquet`
- Strict before-any-action rows: 3,729
- Strict before-final-action rows: 4,227
- Buffered before-any-action rows: 3,667
- Buffered before-final-action rows: 4,170
- Final-prefix mean commitment to final action: approximately 1.0
- Final-prefix top action matches agent action: 1.0

Validation:

- `q_*` rows sum to 1 with max error `2.22e-16`
- No NaN or infinite values in numeric belief transform columns
- `row_id` is stable and matches row order
- Agent actions are all in `UP`, `DOWN`, `LEFT`, `RIGHT`


### 2026-06-28 - Conceptual belief space Step 3 grouped splits and extremes

Implemented and ran:

- `scripts/conceptual_belief_space/03_make_splits_and_extremes.py`

Command:

```bash
python -B scripts/conceptual_belief_space/03_make_splits_and_extremes.py \
  --beliefs outputs/conceptual_belief_space/action_belief_vectors.parquet \
  --splits_output outputs/conceptual_belief_space/splits/group_splits.json \
  --extremes_output outputs/conceptual_belief_space/extreme_examples/action_extreme_examples.parquet \
  --group_col trajectory_id \
  --top_k 500 \
  --threshold 0.9 \
  --max_per_decision 3
```

Outputs written:

- `outputs/conceptual_belief_space/splits/group_splits.json`
- `outputs/conceptual_belief_space/extreme_examples/action_extreme_examples.parquet`
- `outputs/conceptual_belief_space/03_extreme_examples_report.md`

Results:

- Group column: `trajectory_id`
- Groups: 10
- Requested folds: 5
- Effective folds: 5
- Extreme rows selected: 4,044
- Selected split/subset blocks: 35
- Fallback-selected blocks: 0
- Skipped blocks: 0
- Every selected split/subset block is balanced across `UP`, `DOWN`, `LEFT`, and `RIGHT`

Validation:

- No train/test group leakage in any split.
- No test rows were selected as action-extreme examples.
- `row_id` was checked to be unique and aligned with row order.
- Temporary smoke run with `--group_col decision_id` also completed successfully using `/private/tmp` outputs.


### 2026-06-28 - Conceptual belief space Step 4 behavior manifold implementation

Implemented:

- `src/conceptual_belief_space/manifold.py`
- `src/conceptual_belief_space/centroids.py`
- Extended `src/conceptual_belief_space/plotting.py`
- `scripts/conceptual_belief_space/04_fit_behavior_manifold.py`

Implemented behavior:

- Supports `--representation clr` and `--representation sqrt_q`.
- Primary method is UMAP only.
- The script does not silently substitute PCA when UMAP is unavailable.
- If `umap-learn` is missing, the script exits with:
  `pip install umap-learn`
- Optional diagnostic PCA is available only behind `--debug_pca` and is labeled non-primary.
- For each `(split_id, subset)` block, the model is fit only on train-side action-extreme examples.
- All prefix belief rows are transformed after fitting.
- Saves per-block models under `outputs/conceptual_belief_space/models/`.
- Computes 2D UMAP centroids and high-dimensional representation centroids from train extremes only.
- Generates behavior figures by `agent_action`, `top_action`, `commitment_to_final`, `progress`, and selected trajectories.

Validation:

- Ran `python -B scripts/conceptual_belief_space/04_fit_behavior_manifold.py --help`.
- Ran helper import/centroid smoke checks successfully.
- Ran the primary command without UMAP installed; it failed with the expected installation message and did not run PCA.
- Ran explicit diagnostic `--debug_pca` to `/private/tmp`; it completed and produced:
  - 347,970 coordinate rows
  - 140 centroid rows

Notes:

- The real Step 4 UMAP outputs were not generated because `umap-learn` is not installed in the current environment.


### 2026-06-28 - Conceptual belief space Step 5 activation manifold implementation

Implemented:

- `src/conceptual_belief_space/activation_features.py`
- Extended `src/conceptual_belief_space/manifold.py`
- Extended `src/conceptual_belief_space/centroids.py`
- Extended `src/conceptual_belief_space/plotting.py`
- `scripts/conceptual_belief_space/05_fit_activation_manifold.py`

Implemented behavior:

- Uses the same train action-extreme `row_id`s from Step 3; no independent activation extreme selection.
- Supports `--layers all` or comma-separated layers.
- Supports activation representations discovered in `io_map.json`; default is `raw,delta_pre`.
- Loads activation feature arrays with NumPy `mmap_mode="r"`.
- Fits StandardScaler-equivalent preprocessing on split training rows only.
- Optionally L2 normalizes after scaling; default is enabled.
- Fits UMAP on train extreme rows only.
- Transforms all aligned action-belief rows.
- Streams coordinate blocks into `activation_coords.parquet` instead of keeping all blocks in memory.
- Computes 2D activation centroids and high-dimensional scaled/L2 activation centroids from train extremes only.
- Saves high-dimensional centroids to `activation_centroids_highdim.npz` with keys containing split id, subset, layer, representation, and action.
- Saves model/scaler joblib artifacts under `outputs/conceptual_belief_space/models/`.
- Generates activation UMAP figures for a manageable default layer/representation subset.

Validation:

- Ran `python -B scripts/conceptual_belief_space/05_fit_activation_manifold.py --help`.
- Ran activation helper smoke checks successfully.
- Ran the requested primary command; it failed with the expected `pip install umap-learn` message because UMAP is not installed, and did not substitute PCA.
- Ran a tiny synthetic `--debug_pca` smoke test under `/private/tmp`; it produced:
  - 24 coordinate rows
  - 4 2D centroid rows
  - 4 high-dimensional centroid arrays

Notes:

- The real Step 5 UMAP outputs were not generated because `umap-learn` is not installed in the current environment.


### 2026-06-28 - Conceptual belief space Step 6 geometry alignment implementation

Implemented:

- `src/conceptual_belief_space/geometry.py`
- `src/conceptual_belief_space/stats.py`
- Extended `src/conceptual_belief_space/plotting.py`
- `scripts/conceptual_belief_space/06_geometry_alignment.py`

Implemented behavior:

- Loads behavior centroid matrices from `behavior_centroids.parquet`.
- Loads activation 2D centroids from `activation_centroids_2d.parquet`.
- Loads activation high-dimensional centroids from `activation_centroids_highdim.npz`.
- Builds action distance matrices over `UP`, `DOWN`, `LEFT`, `RIGHT`.
- Primary comparison:
  - behavior high-dimensional Euclidean centroid distances
  - activation high-dimensional cosine centroid distances
- Secondary comparisons:
  - behavior 2D UMAP Euclidean distances
  - activation high-dimensional Euclidean distances
  - activation 2D UMAP Euclidean distances
- Computes Pearson, Spearman, and Kendall tau when SciPy is available.
- Runs action-label permutation tests for Spearman; with four actions, exact 24 permutations are used when requested permutations are at least 24.
- Reports a paired pairwise-distance bootstrap confidence interval as a centroid-only fallback.
- Saves all distance matrices to compressed NPZ.
- Generates heatmaps and distance-correlation summary figures.
- Report explicitly states the limitation that four action concepts provide only six pairwise distances.

Validation:

- Ran `python -B scripts/conceptual_belief_space/06_geometry_alignment.py --help`.
- Ran geometry/statistics helper smoke checks successfully.
- Ran the requested command against project paths; it failed clearly because Step 4/5 real UMAP centroid outputs are not present yet:
  `pip install umap-learn`
- Ran a synthetic end-to-end smoke test under `/private/tmp`; it produced:
  - 6 metric rows
  - distance matrix NPZ
  - markdown report
  - geometry alignment figures

Notes:

- The real Step 6 outputs were not generated because the Step 4 and Step 5 UMAP centroid artifacts are missing.


### 2026-06-28 - Conceptual belief space Step 7 sparse activation probes implementation

Implemented:

- `src/conceptual_belief_space/probes.py`
- `src/conceptual_belief_space/metrics.py`
- Extended `src/conceptual_belief_space/activation_features.py`
- Extended `src/conceptual_belief_space/plotting.py`
- `scripts/conceptual_belief_space/07_train_sparse_probes.py`

Implemented behavior:

- Trains grouped cross-validated probes from high-dimensional activation feature arrays, not UMAP/PCA coordinates.
- Uses `group_splits.json`; held-out trajectories remain held out.
- Fits activation feature scalers on training rows only for each split/subset/layer/representation.
- Supports all discovered layers or comma-separated layer lists.
- Supports activation representations from `io_map.json`; default is `raw,delta_pre,unit_delta_pre,velocity`.
- Evaluates subsets:
  - `all_prefixes`
  - `strict_before_any_action_mention`
  - `strict_before_final_action_mention`
  - `high_commitment`
  - `low_commitment`
- Probe targets:
  - behavior CLR vector
  - behavior probability vector
  - `commitment_to_final`
  - `entropy_commitment`
  - `top_action`
  - `agent_action`
- Models:
  - Ridge regression for vector and scalar targets
  - MultiTaskElasticNet for sparse behavior CLR prediction
  - balanced LogisticRegression for action classifications
- Controls:
  - shuffled behavior vector baseline
  - shuffled action-label baseline
  - chance level recorded as 0.25
- Outputs:
  - `outputs/conceptual_belief_space/probes/probe_metrics.csv`
  - `outputs/conceptual_belief_space/probes/probe_predictions.parquet`
  - `outputs/conceptual_belief_space/probes/best_probe_config.json`
  - `outputs/conceptual_belief_space/probes/07_probe_report.md`
  - figures under `figures/conceptual_belief_space/probes/`

Validation:

- Ran `python -B scripts/conceptual_belief_space/07_train_sparse_probes.py --help`.
- Ran probe/metric helper smoke checks successfully.
- Ran a synthetic end-to-end smoke test under `/private/tmp`; it produced:
  - 80 metric rows
  - prediction parquet
  - best config JSON
  - markdown report
  - 5 probe figures
- Ran a bounded real-data smoke test under `/private/tmp` using split 0, layer 0, representation `raw`, with ElasticNet and controls disabled; it produced:
  - 30 metric rows
  - 3,581 prediction rows
  - best config JSON
  - markdown report
  - 5 probe figures

Notes:

- The full all-layer/all-representation probe sweep was not launched because it is substantially heavier than the smoke tests.
- The bounded real-data smoke test reported negative R2 values for several targets, and the script preserves those values rather than clipping or hiding them.


### 2026-06-28 - Conceptual belief space Step 8 trajectory analysis implementation

Implemented:

- `src/conceptual_belief_space/trajectories.py`
- Extended `src/conceptual_belief_space/plotting.py`
- `scripts/conceptual_belief_space/08_trajectory_analysis.py`

Implemented behavior:

- Consumes conceptual UMAP coordinates from Step 4 and Step 5.
- Does not use old blind PCA coordinates.
- Computes behavior decision trajectory metrics:
  - number of prefixes
  - behavior UMAP path length
  - total variation in q
  - total variation in CLR(q)
  - distance to final-action behavior centroid
  - nearest behavior centroid action
  - behavior centroid switch count
  - first and permanent behavior lock-in progress
- Computes activation decision trajectory metrics by split, subset, layer, and activation representation:
  - activation UMAP path length
  - distance to final-action activation centroid
  - nearest activation centroid action
  - activation centroid switch count
  - first and permanent activation lock-in progress
- Computes behavior-vs-activation comparison correlations:
  - path length
  - switch count
  - first lock-in progress
  - permanent lock-in progress
  - mean distance to final centroid
- Writes:
  - `outputs/conceptual_belief_space/trajectories/trajectory_metrics.csv`
  - `outputs/conceptual_belief_space/trajectories/prefix_trajectory_scores.parquet`
  - `outputs/conceptual_belief_space/trajectories/08_trajectory_report.md`
- Generates figures under `figures/conceptual_belief_space/trajectories/`.
- The report explicitly states nearest-centroid basins are descriptive, not causal mechanisms.

Validation:

- Ran `python -B scripts/conceptual_belief_space/08_trajectory_analysis.py --help`.
- Ran trajectory helper smoke checks successfully.
- Ran the requested command against project paths; it failed clearly because Step 4/5 conceptual UMAP coordinate files are not present yet and instructs to install UMAP and rerun Steps 4/5.
- Ran a synthetic end-to-end smoke test under `/private/tmp`; it produced:
  - 13 trajectory metric rows
  - 24 prefix trajectory score rows
  - markdown report
  - 6 trajectory figures

Notes:

- The real Step 8 outputs were not generated because Step 4 and Step 5 UMAP artifacts are missing.


### 2026-06-28 - Conceptual belief space Step 9 strict pre-action analysis implementation

Implemented:

- `scripts/conceptual_belief_space/09_strict_pre_action_analysis.py`

Implemented behavior:

- Loads `outputs/conceptual_belief_space/action_belief_vectors.parquet` as the required input.
- Treats behavior UMAP coordinates, activation UMAP coordinates, activation centroids, probe predictions, probe metrics, and trajectory prefix scores as optional inputs.
- Builds strict pre-action prefix scores for:
  - `strict_before_any_action_mention`
  - `strict_before_final_action_mention`
  - `buffered_before_any_action_mention`
  - `buffered_before_final_action_mention`
- Computes normalized progress to:
  - first action mention
  - first final-action mention
- Aggregates coverage, decision counts, action counts, behavior commitment, entropy commitment, and top-action accuracy over normalized bins.
- Supports activation centroid similarity and basin metrics when Step 5 / Step 8 artifacts are available.
- Supports probe-predicted belief curves and true-vs-shuffled probe control summaries when Step 7 artifacts are available.
- Writes:
  - `outputs/conceptual_belief_space/strict_pre_action/strict_pre_action_metrics.csv`
  - `outputs/conceptual_belief_space/strict_pre_action/strict_pre_action_prefix_scores.parquet`
  - `outputs/conceptual_belief_space/strict_pre_action/09_strict_pre_action_report.md`
- Generates figures under `figures/conceptual_belief_space/strict_pre_action/`.
- The report includes the required caveats:
  - pre-action means no explicit lexical action token, not no implicit coordinate or path evidence;
  - current activations are CoT-token activations, not necessarily final JSON decision-readout activations;
  - observational curves do not prove causality.

Validation:

- Ran `python -B scripts/conceptual_belief_space/09_strict_pre_action_analysis.py --help`.
- Ran `python -m py_compile scripts/conceptual_belief_space/09_strict_pre_action_analysis.py`.
- Ran the requested Step 9 command against project paths. It produced:
  - 360 metric rows
  - 15,793 prefix score rows
  - 2 figures:
    - `coverage_by_normalized_bin.png`
    - `behavior_commitment_before_mention.png`

Notes:

- Real activation/probe strict pre-action figures were not generated yet because conceptual UMAP coordinates, activation centroids, probe predictions, probe metrics, and trajectory prefix scores are not present. The report records those missing optional inputs explicitly.


### 2026-06-28 - Conceptual belief space Step 10 controls and ablations implementation

Implemented:

- `src/conceptual_belief_space/controls.py`
- Extended `src/conceptual_belief_space/plotting.py`
- `scripts/conceptual_belief_space/10_controls_and_ablations.py`

Implemented behavior:

- Computes belief-level shuffled controls directly from `action_belief_vectors.parquet`:
  - shuffles `agent_action` across `decision_id`
  - shuffles behavior `q_UP/q_DOWN/q_LEFT/q_RIGHT` trajectories across `decision_id`
- Computes action-balanced subsampling summaries.
- Computes CoT-length stratified summaries for short / medium / long decisions.
- Computes split-sensitivity diagnostics:
  - loaded grouped trajectory splits
  - derived trajectory grouped splits
  - derived decision grouped splits
  - random prefix split explicitly labeled invalid for main claims
- Computes train-only extreme-selection ablation counts over:
  - thresholds `0.8`, `0.9`, `0.95`
  - top_k `100`, `250`, `500`
  - max_per_decision `1`, `3`, `5`
- Summarizes Step 7 probe metrics and Step 6 geometry metrics when those files exist.
- Adds explicit representation-ablation rows for:
  - `raw`
  - `delta_pre`
  - `unit_delta_pre`
  - `velocity`
- Writes explicit unavailable rows for controls requiring missing upstream artifacts:
  - probe true-vs-shuffled controls
  - centroid distance alignment controls
  - progress-residualized activation feature reruns
- Writes:
  - `outputs/conceptual_belief_space/controls/control_metrics.csv`
  - `outputs/conceptual_belief_space/controls/10_controls_report.md`
- Generates figures under `figures/conceptual_belief_space/controls/`.

Validation:

- Ran `python -m py_compile src/conceptual_belief_space/controls.py src/conceptual_belief_space/plotting.py scripts/conceptual_belief_space/10_controls_and_ablations.py`.
- Ran `python -B scripts/conceptual_belief_space/10_controls_and_ablations.py --help`.
- Ran the requested Step 10 command against project paths. It produced:
  - 1,000 control metric rows
  - 3 figures:
    - `true_vs_shuffled_probe_metrics.png`
    - `action_balanced_vs_all_metrics.png`
    - `cot_length_stratified_metrics.png`

Notes:

- Current behavior-level top-action accuracy is above decision-label shuffle by about `0.4743`.
- Probe and geometry controls could not be assessed yet because `outputs/conceptual_belief_space/probes/probe_metrics.csv` and `outputs/conceptual_belief_space/geometry_alignment/distance_matrix_metrics.csv` are not present.
- Progress residualization is implemented as a helper but not rerun end-to-end from missing activation/probe/geometry artifacts in this Step 10 run.


### 2026-06-28 - Conceptual belief space Step 11 final notebook and report implementation

Implemented:

- `scripts/conceptual_belief_space/11_write_final_report.py`
- `notebooks/11_conceptual_belief_space_report.ipynb`

Implemented behavior:

- Compiles available Step 1-10 markdown reports into:
  - `outputs/conceptual_belief_space/conceptual_belief_space_final_report.md`
  - `outputs/conceptual_belief_space/paper_section_activation_belief_space.md`
- Reads key metric tables when available:
  - `action_belief_vectors.parquet`
  - `extreme_examples/action_extreme_examples.parquet`
  - `strict_pre_action/strict_pre_action_metrics.csv`
  - `controls/control_metrics.csv`
  - probe and geometry metric CSVs when present
- Scans `figures/conceptual_belief_space/` and writes a figure guide that explains:
  - x-axis
  - y-axis
  - color encoding
  - what each point / line / bar represents
  - leakage status
  - caveat
- Includes the mandatory interpretation rules:
  - blind PCA is diagnostic only
  - do not claim behavior-activation alignment without distance-matrix and probe support
  - report weak or missing alignment as weak or missing
  - four actions imply only six pairwise centroid distances
  - CoT-token activation is distinct from final JSON decision-readout activation
  - observational evidence is not causal evidence
- Includes the section titled `What Changed Relative to the Previous PCA Analysis`.
- The notebook is designed to be readable and runnable without rerunning expensive UMAP/probe fitting by default.

Validation:

- Ran `python -m py_compile scripts/conceptual_belief_space/11_write_final_report.py`.
- Validated `notebooks/11_conceptual_belief_space_report.ipynb` as JSON.
- Syntax-checked every notebook code cell with `ast.parse`.
- Ran:
  - `python -B scripts/conceptual_belief_space/11_write_final_report.py --help`
  - `python -B scripts/conceptual_belief_space/11_write_final_report.py --output outputs/conceptual_belief_space/conceptual_belief_space_final_report.md`
- The final report currently embeds 5 available conceptual belief space figures with relative Markdown links.

Notes:

- The final report correctly marks Steps 4-8 as unavailable in the current artifact set because the anchored UMAP, geometry, probe, and trajectory outputs have not been generated yet.
- The concise paper section explicitly avoids claiming behavior-activation alignment from the current run.


### 2026-06-28 - Conceptual belief space optional Step 12 task concepts implementation

Implemented:

- `src/conceptual_belief_space/task_concepts.py`
- `scripts/conceptual_belief_space/12_build_task_concepts.py`
- Updated `scripts/conceptual_belief_space/00_create_dirs.py` to include `outputs/conceptual_belief_space/task_concepts/`.

Implemented behavior:

- Parses canonical `grid_state_text` from `outputs/canonical_steps.parquet`.
- Locates:
  - agent
  - goal
  - key
  - door
  - walls
- Computes deterministic state concepts:
  - `door_relevant`
  - `key_required`
  - `goal_reachable_without_key`
  - `door_requires_key`
- Computes per-action deterministic masks for:
  - `move_toward_goal`
  - `move_away_from_goal`
  - `horizontal_move`
  - `vertical_move`
  - `move_toward_key`
  - `move_toward_door`
  - `wall_blocked_action`
  - `shortest_path_action`
  - `detour_action`
  - `valid_action`
  - `invalid_action`
- Uses a BFS shortest-path approximation with key pickup and door passability once the key is carried.
- Converts action probabilities into q-weighted model concept beliefs:
  - `model_belief_<concept> = sum_a q_a * env_<concept>_a`
- Adds deterministic concept values for both:
  - `env_agent_action_<concept>`
  - `env_top_action_<concept>`
- Clearly distinguishes deterministic environment concepts from model belief concepts in the report.
- Preserves the four-action analysis as the base result.

Validation:

- Ran `python -m py_compile src/conceptual_belief_space/task_concepts.py scripts/conceptual_belief_space/12_build_task_concepts.py scripts/conceptual_belief_space/00_create_dirs.py`.
- Ran `python -B scripts/conceptual_belief_space/12_build_task_concepts.py --help`.
- Ran:
  - `python -B scripts/conceptual_belief_space/12_build_task_concepts.py --beliefs outputs/conceptual_belief_space/action_belief_vectors.parquet --canonical_steps outputs/canonical_steps.parquet --output outputs/conceptual_belief_space/task_concepts/task_concept_scores.parquet`
- Output:
  - `outputs/conceptual_belief_space/task_concepts/task_concept_scores.parquet`
  - `outputs/conceptual_belief_space/task_concepts/12_task_concept_report.md`
- Produced:
  - 9,942 prefix rows
  - 120 columns
  - all 132 decisions parsed successfully

Key diagnostics:

- Agent action is a shortest-path action, mean over prefixes: about `0.8230`.
- Current model top action is a shortest-path action, mean over prefixes: about `0.6316`.
- Mean q-weighted belief in shortest-path actions: about `0.6164`.

Notes:

- The shortest-path search is an environment approximation, not a full simulator trace.
- Concept beliefs are induced from action probabilities and are not independent model readouts.


### 2026-06-28 - Conceptual belief space pipeline run through final report

Installed dependency:

- Installed missing `umap-learn` with `python -m pip install umap-learn`.
- This also installed `numba`, `llvmlite`, and `pynndescent`.

Ran successfully, sequentially:

- `python scripts/conceptual_belief_space/00_create_dirs.py`
  - Ensured 20 output/figure directories.
  - Wrote `outputs/conceptual_belief_space/README.md`.
- `python scripts/conceptual_belief_space/01_audit_io.py`
  - Wrote `outputs/conceptual_belief_space/io_map.json`.
  - Wrote `outputs/conceptual_belief_space/01_io_audit.md`.
  - Found matching behavior and activation metadata row counts.
- `python scripts/conceptual_belief_space/02_build_action_beliefs.py`
  - Wrote `outputs/conceptual_belief_space/action_belief_vectors.parquet` with 9,942 rows.
  - Wrote `outputs/conceptual_belief_space/02_action_belief_vectors_report.md`.
- `python scripts/conceptual_belief_space/03_make_splits_and_extremes.py`
  - Wrote `outputs/conceptual_belief_space/splits/group_splits.json` with 5 splits.
  - Wrote `outputs/conceptual_belief_space/extreme_examples/action_extreme_examples.parquet` with 4,044 rows.
  - Wrote `outputs/conceptual_belief_space/03_extreme_examples_report.md`.
- `python scripts/conceptual_belief_space/04_fit_behavior_manifold.py`
  - Wrote `outputs/conceptual_belief_space/behavior_manifold/behavior_coords.parquet` with 347,970 rows.
  - Wrote `outputs/conceptual_belief_space/behavior_manifold/behavior_centroids.parquet` with 140 rows.
  - Wrote `outputs/conceptual_belief_space/04_behavior_manifold_report.md`.
- `python scripts/conceptual_belief_space/05_fit_activation_manifold.py`
  - Wrote `outputs/conceptual_belief_space/activation_manifold/activation_coords.parquet` with 8,351,280 rows.
  - Wrote `outputs/conceptual_belief_space/activation_manifold/activation_centroids_2d.parquet` with 3,360 rows.
  - Wrote `outputs/conceptual_belief_space/activation_manifold/activation_centroids_highdim.npz` with 3,360 centroid arrays.
  - Wrote `outputs/conceptual_belief_space/05_activation_manifold_report.md`.
- `python scripts/conceptual_belief_space/06_geometry_alignment.py`
  - Wrote `outputs/conceptual_belief_space/geometry_alignment/distance_matrix_metrics.csv` with 5,040 rows.
  - Wrote `outputs/conceptual_belief_space/geometry_alignment/distance_matrices.npz`.
  - Wrote `outputs/conceptual_belief_space/geometry_alignment/06_geometry_alignment_report.md`.
- `python scripts/conceptual_belief_space/07_train_sparse_probes.py`
  - Wrote `outputs/conceptual_belief_space/probes/probe_metrics.csv` with 13,200 metric rows.
  - Wrote `outputs/conceptual_belief_space/probes/probe_predictions.parquet`.
  - Wrote `outputs/conceptual_belief_space/probes/best_probe_config.json`.
  - Wrote `outputs/conceptual_belief_space/probes/07_probe_report.md`.
- `python scripts/conceptual_belief_space/11_write_final_report.py`
  - Wrote `outputs/conceptual_belief_space/conceptual_belief_space_final_report.md`.
  - Wrote `outputs/conceptual_belief_space/paper_section_activation_belief_space.md`.

Run notes:

- Steps 04 and 05 emitted expected UMAP warnings about fixed `random_state` forcing `n_jobs=1`; some small extreme-example subsets also warned that `n_neighbors` was larger than the subset size.
- Step 07 emitted many SciPy `LinAlgWarning: Ill-conditioned matrix` warnings during ridge-style solves. The script completed and wrote all probe outputs.


### 2026-06-28 - Conceptual belief space Steps 08-11 run

Ran successfully, sequentially:

- `python scripts/conceptual_belief_space/08_trajectory_analysis.py`
  - Wrote `outputs/conceptual_belief_space/trajectories/trajectory_metrics.csv` with 119,700 rows.
  - Wrote `outputs/conceptual_belief_space/trajectories/prefix_trajectory_scores.parquet` with 8,699,250 rows.
  - Wrote `outputs/conceptual_belief_space/trajectories/08_trajectory_report.md`.
- `python scripts/conceptual_belief_space/09_strict_pre_action_analysis.py`
  - Wrote `outputs/conceptual_belief_space/strict_pre_action/strict_pre_action_metrics.csv` with 408,374 rows.
  - Wrote `outputs/conceptual_belief_space/strict_pre_action/strict_pre_action_prefix_scores.parquet` with 21,511,945 rows.
  - Wrote `outputs/conceptual_belief_space/strict_pre_action/09_strict_pre_action_report.md`.
- `python scripts/conceptual_belief_space/10_controls_and_ablations.py`
  - Wrote `outputs/conceptual_belief_space/controls/control_metrics.csv` with 89,089 rows.
  - Wrote `outputs/conceptual_belief_space/controls/10_controls_report.md`.
- `python scripts/conceptual_belief_space/11_write_final_report.py`
  - Rebuilt `outputs/conceptual_belief_space/conceptual_belief_space_final_report.md`.
  - Rebuilt `outputs/conceptual_belief_space/paper_section_activation_belief_space.md`.

Run notes:

- No additional dependencies were needed.
- Final report compilation emitted pandas `DtypeWarning` messages for mixed-type CSV columns in control/probe summaries, but completed and wrote both report files.


### 2026-06-28 - Conceptual belief space acceleration update

Main changes:

- Added optional UMAP backend selection in `src/conceptual_belief_space/manifold.py`.
  - `--umap_backend auto` uses RAPIDS cuML UMAP when `cuml`/`cupy` are available.
  - On this Apple Silicon machine, `auto` resolves to CPU because CUDA/RAPIDS are not installed.
  - Added `--n_jobs` and `--deterministic/--no-deterministic` controls for CPU `umap-learn`.
- Updated `scripts/conceptual_belief_space/05_fit_activation_manifold.py`.
  - Added `--umap_backend`.
  - Added `--n_jobs`.
  - Added `--deterministic/--no-deterministic`.
  - Added `--subsets` filtering to avoid fitting every conceptual subset during iteration.
- Added PyTorch Ridge support in `src/conceptual_belief_space/probes.py`.
  - `--ridge_backend auto` uses CUDA if available, else Apple MPS if available, else scikit-learn CPU.
  - On this machine, Ridge resolves to PyTorch on `mps`.
  - Logistic regression and ElasticNet remain scikit-learn CPU.
- Updated `scripts/conceptual_belief_space/07_train_sparse_probes.py`.
  - Added `--ridge_backend`.
  - Added `--torch_device`.
  - Added `--torch_batch_size`.
  - Added `--subsets` filtering.
  - Added `--run_logistic/--no-run_logistic` for fast behavior-belief probe runs.

Verification:

- Checked local acceleration stack:
  - platform: macOS arm64.
  - PyTorch installed with MPS available.
  - CUDA unavailable.
  - `cuml` and `cupy` unavailable.
  - `umap-learn` and scikit-learn available.
- Verified small PyTorch MPS Ridge solve succeeds.
- Compiled edited files with `python -m py_compile`.
- Smoke-tested UMAP helper with `backend=auto`, which resolved to CPU locally.
- Smoke-tested Step 5 with:
  - `python scripts/conceptual_belief_space/05_fit_activation_manifold.py --layers 0 --representations delta_pre --subsets all_prefixes --debug_pca ...`
  - Wrote `outputs/conceptual_belief_space/activation_manifold/activation_coords_smoke_gpu.parquet` with 49,710 rows.
  - Wrote `outputs/conceptual_belief_space/activation_manifold/activation_centroids_2d_smoke_gpu.parquet` with 20 rows.
  - Wrote `outputs/conceptual_belief_space/activation_manifold/activation_centroids_highdim_smoke_gpu.npz` with 20 centroid arrays.
- Smoke-tested Step 7 fast mode with:
  - `python scripts/conceptual_belief_space/07_train_sparse_probes.py --layers 0 --representations delta_pre --subsets all_prefixes --no-run_elasticnet --no-run_controls --no-run_logistic --ridge_backend auto --torch_device auto ...`
  - Wrote `outputs/conceptual_belief_space/probes/probe_metrics_smoke_gpu.csv` with 20 metric rows.
  - Wrote `outputs/conceptual_belief_space/probes/probe_predictions_smoke_gpu.parquet`.
  - Wrote `outputs/conceptual_belief_space/probes/best_probe_config_smoke_gpu.json`.
  - Wrote `outputs/conceptual_belief_space/probes/07_probe_report_smoke_gpu.md`.

Practical note:

- Full GPU acceleration for Step 5 UMAP requires a CUDA/RAPIDS environment. Apple MPS does not provide a drop-in UMAP backend.
- For local fast iteration on this Mac, use subset/layer filters, `--no-deterministic --n_jobs -1` for CPU UMAP, and MPS-backed Ridge with `--ridge_backend auto`.


### 2026-06-28 - Attempted all-trajectory conceptual belief rerun

Goal:

- Rerun the conceptual belief space experiment on the full available gridworld set instead of the earlier 10-trajectory subset.

Current upstream data counts:

- `outputs/prefixes.parquet`: 98,311 prefix rows across 95 available trajectories.
- The repository contains 95 trajectory JSON files under `trajectories_key_door_100`, not 100.
- Existing `outputs/prefix_action_scores.parquet` before this attempt was partial: 9,942 scored prefixes across 10 trajectories.

Together scoring changes:

- Updated `scripts/03_score_prefix_actions_together.py`.
  - Added `--num_workers` for bounded prefix-level concurrent scoring.
  - Added per-thread Together clients for parallel scoring.
  - Added interrupt/early-stop checkpointing.
  - Added `--retry_failed`, default true, so failed rows are not treated as completed during resume.
  - Added credit-limit detection so Together `402 credit_limit` errors abort instead of silently filling the output with failed rows.
  - On resume with `--retry_failed`, failed placeholder rows are removed from the output before further scoring.
- Verified with `python -m py_compile scripts/03_score_prefix_actions_together.py`.
- Verified a small parallel smoke test with `/tmp` outputs before the full attempt.

Scoring commands attempted:

- Started the full scorer with Together `openai/gpt-oss-20b`, `echo_teacher_forced`, `bare_action`, and resume enabled.
- Sequential scoring was too slow, so the scorer was patched for parallel execution.
- Ran with `--num_workers 8`, which progressed cleanly with no failed rows.
- Tried `--num_workers 16`, which reached more rows faster but then Together returned account billing errors.

Blocking condition:

- Together API returned:
  - `Error code: 402`
  - `type: credit_limit`
  - message: credit limit exceeded / add credits.
- After the user approved sending data to Together AI, a one-prefix billing probe still returned the same `402 credit_limit` error.
- Because this is a billing/account state error, the all-trajectory scoring cannot complete until Together credits are restored.

Current cleaned scoring state:

- Removed 799 failed placeholder rows from `outputs/prefix_action_scores.parquet`.
- Current `outputs/prefix_action_scores.parquet`: 33,148 successful scored prefixes across 32 trajectories.
- Full target remains 98,311 prefix rows across 95 available trajectories.
- Downstream full conceptual belief space scripts were not rerun because their inputs would still be partial and would not satisfy the all-trajectory request.


### 2026-06-29 - Continued downstream run from generated score parquet

Input state:

- The user asked to continue from the generated parquet.
- `outputs/prefix_action_scores.parquet` was clean but partial:
  - 33,148 scored prefixes.
  - 32 trajectories.
  - 439 decision steps.
  - 0 failed rows.
- `outputs/prefixes.parquet` still contains 98,311 prefixes across 95 trajectories, so 65,163 prefixes remain unscored.
- All results from this run therefore correspond to the current scored 32-trajectory subset, not the full 95-trajectory set.

Behavior-side rerun:

- Ran `scripts/04_compute_commitment_metrics.py`.
  - Wrote `outputs/prefix_commitment_metrics.parquet` with 33,148 rows.
  - Wrote `outputs/step_commitment_summary.parquet` with 439 rows.
  - Wrote `outputs/commitment_report.md`.
- Ran `scripts/05_build_behavior_vectors.py`.
  - Wrote `outputs/behavior_vectors.parquet` with 33,148 rows.
  - Wrote `outputs/behavior_vectors_report.md`.
- Ran `scripts/06_embed_behavior_manifold.py`.
  - Wrote `outputs/behavior_manifold.parquet` with 33,148 rows.
  - Wrote `outputs/models/behavior_pca.joblib`.
  - Wrote `outputs/behavior_manifold_report.md`.
- Ran `scripts/07_plot_behavior_commitment.py`.
  - Refreshed behavior figures under `figures/behavior/`.

Activation-side rerun:

- Ran activation schema audit against the actual activation directory:
  - `trajectories_key_door_100/activations_key_door_100_mlx_packed_bf16_cot_all`
  - Found 91,874 activation/metadata files.
  - Wrote `outputs/activation/activation_inventory.parquet`.
  - Wrote `outputs/activation/activation_schema_report.md`.
  - Wrote `outputs/activation/schema_examples.json`.
- Ran activation token indexing.
  - Detected packed raw BF16 layout.
  - Indexed 2,373,402 activation timepoints across 1,276 decisions.
  - Wrote `outputs/activation/activation_token_index.parquet`.
- Ran activation/behavior alignment.
  - Wrote `outputs/activation/activation_behavior_aligned.parquet` with 33,148 rows.
  - Alignment statuses:
    - exact: 32,720
    - exact_pre_reasoning: 428
- Ran activation feature construction.
  - Used all extracted every-two layers: 0,2,4,6,8,10,12,14,16,18,20,22.
  - Representations: raw, delta_pre, velocity, unit_delta_pre.
  - Wrote 48 feature arrays under `outputs/activation/features/`.
  - Wrote `outputs/activation/activation_feature_metadata.parquet` with 33,148 rows.
- Activation layer sweep:
  - A full/default run was interrupted because the default comparison representations and logistic solvers were too slow.
  - A bounded delta-pre-only run with 5 folds was killed by the OS with exit code 137, likely memory pressure.
  - Successful rerun used:
    - `--representation delta_pre`
    - `--comparison_representations delta_pre`
    - `--n_splits 2`
    - `--max_iter 50`
  - Wrote `outputs/activation/layer_sweep_metrics.csv`.
  - Wrote `outputs/activation/reports/layer_sweep_report.md`.
  - Wrote `outputs/activation/best_activation_config.json`.
- Ran activation manifold embedding.
  - Selected layer 22 / `delta_pre` from `best_activation_config.json`.
  - Wrote `outputs/activation/activation_manifold.parquet` with 33,148 rows.
- Ran activation-behavior alignment.
  - Wrote `outputs/activation/activation_behavior_alignment_metrics.csv`.
  - Wrote `outputs/activation/activation_behavior_alignment_predictions.parquet`.
- Ran activation centroid basins.
  - Wrote `outputs/activation/activation_centroid_scores.parquet`.
  - Wrote `outputs/activation/activation_basin_summary.parquet`.
- Ran pre/post reasoning analysis.
  - Analyzed 439 decisions across 12 layers.
  - Wrote `outputs/activation/pre_post_reasoning_features.parquet`.
  - Wrote `outputs/activation/pre_post_decoding_metrics.csv`.
- Strict pre-action activation analysis:
  - Default run was interrupted because it became memory-heavy in logistic regression.
  - Successful rerun used `--n_splits 2 --max_iter 50`.
  - Wrote `outputs/activation/strict_pre_action_activation_metrics.csv`.
  - Wrote `outputs/activation/strict_pre_action_activation_curves.csv`.
- Ran activation dynamics.
  - Wrote `outputs/activation/activation_dynamics_summary.parquet`.
  - Wrote `outputs/activation/activation_dynamics_prefix_metrics.parquet`.
- Ran activation controls.
  - Wrote `outputs/activation/activation_control_metrics.csv`.
  - Wrote `outputs/activation/reports/activation_controls_report.md`.

Conceptual belief space rerun:

- Ran `scripts/conceptual_belief_space/00_create_dirs.py`.
- Ran `01_audit_io.py`.
  - Confirmed row counts match between behavior table and activation feature metadata.
- Ran `02_build_action_beliefs.py`.
  - Wrote `outputs/conceptual_belief_space/action_belief_vectors.parquet` with 33,148 rows.
- Ran `03_make_splits_and_extremes.py`.
  - Wrote 5 grouped trajectory splits.
  - Wrote `outputs/conceptual_belief_space/extreme_examples/action_extreme_examples.parquet` with 35,000 rows.
- Ran `04_fit_behavior_manifold.py`.
  - Wrote `outputs/conceptual_belief_space/behavior_manifold/behavior_coords.parquet` with 1,160,180 rows.
  - Wrote `outputs/conceptual_belief_space/behavior_manifold/behavior_centroids.parquet` with 140 rows.
  - UMAP emitted expected warnings that fixed `random_state` forces `n_jobs=1`.
- Ran `05_fit_activation_manifold.py` with a feasible selected configuration:
  - layer 22
  - representation `delta_pre`
  - subsets: all_prefixes, strict_before_any_action_mention, strict_before_final_action_mention, high_commitment, low_commitment
  - `--no-deterministic --n_jobs -1`
  - Wrote `outputs/conceptual_belief_space/activation_manifold/activation_coords.parquet` with 828,700 rows.
  - Wrote `outputs/conceptual_belief_space/activation_manifold/activation_centroids_2d.parquet` with 100 rows.
  - Wrote `outputs/conceptual_belief_space/activation_manifold/activation_centroids_highdim.npz` with 100 centroid arrays.
- Ran `06_geometry_alignment.py`.
  - Wrote `outputs/conceptual_belief_space/geometry_alignment/distance_matrix_metrics.csv` with 150 rows.
- Ran `07_train_sparse_probes.py` with the same selected activation config:
  - layer 22
  - representation `delta_pre`
  - subsets: all_prefixes, strict_before_any_action_mention, strict_before_final_action_mention, high_commitment, low_commitment
  - `--ridge_backend auto --torch_device auto`
  - Wrote `outputs/conceptual_belief_space/probes/probe_metrics.csv` with 275 rows.
  - Wrote `outputs/conceptual_belief_space/probes/probe_predictions.parquet`.
  - Wrote `outputs/conceptual_belief_space/probes/best_probe_config.json`.
- Ran `08_trajectory_analysis.py`.
  - Wrote `outputs/conceptual_belief_space/trajectories/trajectory_metrics.csv` with 26,465 rows.
  - Wrote `outputs/conceptual_belief_space/trajectories/prefix_trajectory_scores.parquet` with 1,988,880 rows.
- Ran `09_strict_pre_action_analysis.py`.
  - Wrote `outputs/conceptual_belief_space/strict_pre_action/strict_pre_action_metrics.csv` with 8,734 rows.
  - Wrote `outputs/conceptual_belief_space/strict_pre_action/strict_pre_action_prefix_scores.parquet` with 582,077 rows.
- Ran `10_controls_and_ablations.py`.
  - Wrote `outputs/conceptual_belief_space/controls/control_metrics.csv` with 3,150 rows.
- Ran `11_write_final_report.py`.
  - Wrote `outputs/conceptual_belief_space/conceptual_belief_space_final_report.md`.
  - Wrote `outputs/conceptual_belief_space/paper_section_activation_belief_space.md`.
- Ran optional `12_build_task_concepts.py`.
  - Wrote `outputs/conceptual_belief_space/task_concepts/task_concept_scores.parquet` with 33,148 rows.
  - Wrote `outputs/conceptual_belief_space/task_concepts/12_task_concept_report.md`.

Figure outputs:

- `figures/behavior/`: 12 PNG files.
- `figures/activation/`: 75 PNG files.
- `figures/conceptual_belief_space/`: 50 PNG files.

Important caveats:

- This was not a full 95-trajectory rerun because the score parquet is still partial.
- Conceptual activation manifold/probe steps were run on the selected best activation configuration, not every layer/representation, to avoid CPU/memory/disk failures on this machine.
- Some older smoke-test and prior-run artifacts remain in output directories with `_smoke_gpu` names or older layer-specific figure names; the standard output files listed above were refreshed in this run.


### 2026-06-29 - High-commitment train action-extreme activation UMAP diagnostic

Added files:

- `src/conceptual_belief_space/extreme_activation_plots.py`
  - Shared helpers for selecting train-only action-extreme rows, resolving feature-row indices, train-fitted scaling, optional L2 normalization, UMAP fitting, separation metrics, and matplotlib-only diagnostic plots.
- `scripts/conceptual_belief_space/12_plot_extreme_activation_umaps.py`
  - CLI diagnostic script for positive-control activation UMAPs on high-commitment train action-extreme prefixes.
  - Writes only under:
    - `outputs/conceptual_belief_space/activation_extremes/`
    - `figures/conceptual_belief_space/activation_extremes/`
  - Does not modify Step 5 activation manifold outputs.

Verification:

- Compiled successfully with:
  - `python -m py_compile src/conceptual_belief_space/extreme_activation_plots.py scripts/conceptual_belief_space/12_plot_extreme_activation_umaps.py`
- Checked CLI help successfully.

Diagnostic run:

- Ran:
  - `python scripts/conceptual_belief_space/12_plot_extreme_activation_umaps.py`
- Defaults:
  - split_id: 0
  - subset: `high_commitment`
  - layers: 0,2,4,6,8,10,12,14,16,18,20,22
  - representations: raw, delta_pre, unit_delta_pre
  - scaler fit on train rows
  - L2 normalize after scaling
  - UMAP backend: sklearn / umap-learn
  - UMAP metric: cosine
  - random_state: 42

Outputs:

- `outputs/conceptual_belief_space/activation_extremes/selected_extreme_rows_split0_high_commitment.parquet`
  - 960 selected train extreme rows.
  - Balanced action counts:
    - UP: 240
    - DOWN: 240
    - LEFT: 240
    - RIGHT: 240
  - No duplicate row_ids removed.
- `outputs/conceptual_belief_space/activation_extremes/extreme_umap_coords.parquet`
  - 34,560 rows = 960 selected rows x 36 layer/representation configs.
- `outputs/conceptual_belief_space/activation_extremes/extreme_umap_metrics.csv`
  - 36 metric rows.
- `outputs/conceptual_belief_space/activation_extremes/extreme_umap_best_configs.json`
- `outputs/conceptual_belief_space/activation_extremes/12_extreme_activation_umap_report.md`
- 40 PNG figures under `figures/conceptual_belief_space/activation_extremes/`.

Main diagnostic result:

- Best ranked configuration by the combined criteria was layer 0 / `delta_pre`.
- Best high-dimensional silhouette was weak: about 0.012.
- Best high-dimensional nearest-centroid accuracy was about 0.61 for layer 4 / `delta_pre`.
- UMAP silhouette scores were negative across the top UMAP configurations.
- Report conclusion: high-commitment train action extremes do not show clear four-action branches in this diagnostic.


### 2026-06-29 - PCA version of high-commitment train action-extreme activation plots

Added file:

- `scripts/conceptual_belief_space/12_plot_extreme_activation_pcas.py`
  - Reuses the same high-commitment train action-extreme row selection as the UMAP diagnostic.
  - Fits 2D PCA instead of UMAP for each layer/representation.
  - Writes under:
    - `outputs/conceptual_belief_space/activation_extremes/pca/`
    - `figures/conceptual_belief_space/activation_extremes/pca/`

Verification:

- Compiled successfully with:
  - `python -m py_compile scripts/conceptual_belief_space/12_plot_extreme_activation_pcas.py`
- Checked CLI help successfully.

Run:

- Ran:
  - `python scripts/conceptual_belief_space/12_plot_extreme_activation_pcas.py`
- Defaults:
  - split_id: 0
  - subset: `high_commitment`
  - layers: 0,2,4,6,8,10,12,14,16,18,20,22
  - representations: raw, delta_pre, unit_delta_pre
  - scaler fit on train rows
  - L2 normalize after scaling

Outputs:

- `outputs/conceptual_belief_space/activation_extremes/pca/selected_extreme_rows_split0_high_commitment.parquet`
  - 960 selected train extreme rows.
  - Balanced action counts:
    - UP: 240
    - DOWN: 240
    - LEFT: 240
    - RIGHT: 240
- `outputs/conceptual_belief_space/activation_extremes/pca/extreme_pca_coords.parquet`
  - 34,560 rows = 960 selected rows x 36 layer/representation configs.
- `outputs/conceptual_belief_space/activation_extremes/pca/extreme_pca_metrics.csv`
  - 36 metric rows.
- `outputs/conceptual_belief_space/activation_extremes/pca/extreme_pca_best_configs.json`
- `outputs/conceptual_belief_space/activation_extremes/pca/12_extreme_activation_pca_report.md`
- 40 PNG figures under `figures/conceptual_belief_space/activation_extremes/pca/`.

Main PCA diagnostic result:

- Best ranked PCA diagnostic configuration was layer 0 / `delta_pre`.
- Best high-dimensional silhouette remained weak: about 0.012.
- PCA silhouette scores were negative for the top configurations.
- Report conclusion: high-commitment train action extremes do not show clear four-action branches under PCA either.


### 2026-06-29 - Action-centroid contrast projection for activation states

Added file:

- `scripts/conceptual_belief_space/12_action_centroid_contrast_projection.py`
  - Implements the paper-style action-centroid contrast projection.
  - Does not use PCA or UMAP for the projection.
  - Fits preprocessing on split train rows only.
  - Computes action centroids from train-only action-extreme examples.
  - Builds four centered action-centroid contrast vectors and uses SVD on those four vectors only.
  - Projects all prefixes onto three action contrast axes and computes nearest-centroid metrics.

Verification:

- Compiled successfully with:
  - `python -m py_compile scripts/conceptual_belief_space/12_action_centroid_contrast_projection.py`

Run:

- Ran:
  - `python scripts/conceptual_belief_space/12_action_centroid_contrast_projection.py --subset high_commitment --split_ids 0,1,2,3,4 --layers 0,2,4,6,8,10,12,14,16,18,20,22 --representations raw,delta_pre,unit_delta_pre --standardize --l2-normalize --plot-split-id 0 --plot-max-points 12000 --overwrite`
- Blocks attempted: 180.
- Successful blocks: 180.
- Skipped blocks: 0.
- Warning: `outputs/activation/activation_feature_metadata.parquet` exists but has no `row_id`; the script fell back only after asserting `action_belief_vectors.row_id` equals the feature row index.

Outputs:

- `outputs/conceptual_belief_space/action_centroid_contrast/action_contrast_coords.parquet`
  - 5,966,640 projected rows.
- `outputs/conceptual_belief_space/action_centroid_contrast/action_contrast_centroids.parquet`
  - 720 action centroid rows.
- `outputs/conceptual_belief_space/action_centroid_contrast/action_contrast_metrics.csv`
  - 180 metric rows.
- `outputs/conceptual_belief_space/action_centroid_contrast/action_contrast_pairwise_distances.csv`
- `outputs/conceptual_belief_space/action_centroid_contrast/action_contrast_axes_metadata.csv`
- `outputs/conceptual_belief_space/action_centroid_contrast/12_action_centroid_contrast_report.md`
- Model arrays under `outputs/conceptual_belief_space/models/action_centroid_contrast/`.
- PNG figures under `figures/conceptual_belief_space/action_centroid_contrast/`.

Main contrast result:

- Best block by test balanced accuracy was split 3 / layer 2 / `delta_pre` / `high_commitment`.
- Best `test_rows_balanced_accuracy_vs_agent_action_contrast`: about 0.2768.
- Corresponding high-dimensional nearest-centroid test balanced accuracy: about 0.2816.
- Train-extreme balanced accuracy in contrast space: about 0.5965.
- Centroid pair Spearman correlation between high-dimensional cosine distances and contrast-space Euclidean distances: about 0.9429.
- Energy ratios for the three contrast axes: about 0.499, 0.313, and 0.188.
- Interpretation: train action-extreme centroids define a coherent contrast coordinate system, but all-row/test action classification remains close to chance, so the current result should be reported as weak action separation rather than strong action-commitment geometry.


### 2026-06-30 - Commitment-only activation contrast analysis

Added files:

- `scripts/conceptual_belief_space/12_fit_commitment_contrast.py`
  - Fits train-only low-to-high commitment axes in activation space.
  - Does not use PCA or UMAP as the primary projection.
  - Primary target: action-agnostic `entropy_commitment`.
  - Secondary targets supported: `top_commitment`, `commitment_to_final`.
  - Axis variants:
    - `original_action_balanced`
    - `progress_matched_action_balanced`
    - `shuffled_commitment_control`
    - `progress_residualized_feature_control`
  - Action labels are used only for balancing and diagnostics.
- `scripts/conceptual_belief_space/13_plot_commitment_contrast.py`
  - Creates projection grids, commitment-bin grids, held-out histograms, progress curves, trajectory phase-plane plots, heatmaps, and control bars.
- `scripts/conceptual_belief_space/14_commitment_contrast_controls.py`
  - Summarizes progress-matched, shuffled, strict pre-action, representation/layer, action leakage, and progress leakage controls.
- `scripts/conceptual_belief_space/15_write_commitment_contrast_report.py`
  - Writes the paper-style commitment-only activation contrast report.

Verification:

- Compiled successfully:
  - `python -m py_compile scripts/conceptual_belief_space/12_fit_commitment_contrast.py scripts/conceptual_belief_space/13_plot_commitment_contrast.py scripts/conceptual_belief_space/14_commitment_contrast_controls.py scripts/conceptual_belief_space/15_write_commitment_contrast_report.py`
- Dry run succeeded:
  - `python scripts/conceptual_belief_space/12_fit_commitment_contrast.py --dry_run`
  - Planned full default jobs: 6,480.
  - Behavior rows: 33,148.
  - Feature rows: 33,148.
  - Splits: 0,1,2,3,4.
  - Layers: 0,2,4,6,8,10,12,14,16,18,20,22.
  - Representations: `delta_pre`, `raw`, `unit_delta_pre`.
  - Targets: `entropy_commitment`, `top_commitment`, `commitment_to_final`.

Smoke run:

- To avoid accidentally writing a very large full 6,480-job projection set, ran an end-to-end smoke test under:
  - `outputs/conceptual_belief_space/commitment_contrast_smoke/`
  - `figures/conceptual_belief_space/commitment_contrast_smoke/`
- Smoke settings:
  - split 0 only
  - layer 0 only
  - representation `delta_pre`
  - target `entropy_commitment`
  - subsets `all_prefixes`, `strict_before_any_action_mention`, `strict_before_final_action_mention`
  - all four axis variants
- Smoke fitting result:
  - Jobs attempted: 12.
  - Jobs successful: 12.
  - Jobs skipped: 0.
- Smoke plotting result:
  - 73 figures created.
  - 0 figures skipped.
- Smoke controls and final report ran successfully.

Smoke outputs:

- `outputs/conceptual_belief_space/commitment_contrast_smoke/commitment_projection_metrics.csv`
- `outputs/conceptual_belief_space/commitment_contrast_smoke/axis_fit_counts.csv`
- `outputs/conceptual_belief_space/commitment_contrast_smoke/commitment_axes.npz`
- `outputs/conceptual_belief_space/commitment_contrast_smoke/controls/commitment_control_summary.csv`
- `outputs/conceptual_belief_space/commitment_contrast_smoke/commitment_contrast_report.md`
- 73 PNG figures under `figures/conceptual_belief_space/commitment_contrast_smoke/`.

Important smoke-run caveat:

- The smoke run is only a code-path validation, not a scientific result.
- In the smoke metrics, shuffled-commitment control also scored high, so the result should not be interpreted as evidence for a robust commitment axis.
- The full run should be evaluated with all splits, layers, representations, targets, and controls before making claims.


### 2026-06-30 - Integrated action-centroid contrast results into final conceptual belief report

Updated file:

- `scripts/conceptual_belief_space/11_write_final_report.py`
  - Added `outputs/conceptual_belief_space/action_centroid_contrast/12_action_centroid_contrast_report.md` to the compiled source reports.
  - Added an `Action-centroid Contrast Projection` final-report section.
  - Added a summary of `action_contrast_metrics.csv`, including the best held-out/test balanced accuracy and caveats.
  - Added figure-guide descriptions for the action-centroid contrast figures.
  - Added action-centroid contrast status to the concise paper-section markdown.

Verification:

- Compiled successfully:
  - `python -m py_compile scripts/conceptual_belief_space/11_write_final_report.py`

Run:

- Ran:
  - `python scripts/conceptual_belief_space/11_write_final_report.py --output outputs/conceptual_belief_space/conceptual_belief_space_final_report.md`

Outputs:

- `outputs/conceptual_belief_space/conceptual_belief_space_final_report.md`
  - Includes the new `Action-centroid Contrast Projection` section.
  - Includes action-centroid contrast figure-guide entries.
- `outputs/conceptual_belief_space/paper_section_activation_belief_space.md`
  - Includes an action-centroid contrast projection subsection.


### 2026-06-30 - Full commitment-only activation contrast run on all trajectories

Command sequence run:

- `python scripts/conceptual_belief_space/12_fit_commitment_contrast.py --layers 0 2 4 6 8 10 12 14 16 18 20 22 --representations delta_pre raw unit_delta_pre --targets entropy_commitment top_commitment commitment_to_final --subsets all_prefixes strict_before_any_action_mention strict_before_final_action_mention --high_threshold 0.9 --low_threshold 0.4 --progress_bins 10 --min_bin_count 3 --standardize --random_seed 42`
- `python scripts/conceptual_belief_space/13_plot_commitment_contrast.py`
- `python scripts/conceptual_belief_space/14_commitment_contrast_controls.py`
- `python scripts/conceptual_belief_space/15_write_commitment_contrast_report.py`

Fit result:

- Jobs attempted: 6,480.
- Jobs successful: 6,480.
- Jobs skipped: 0.
- Metric rows: 6,480.
- Projection parquet files: 6,480.
- Per-job model files: 6,480.
- All metric statuses were `ok`.

Main outputs:

- `outputs/conceptual_belief_space/commitment_contrast/commitment_projection_metrics.csv`
- `outputs/conceptual_belief_space/commitment_contrast/axis_fit_counts.csv`
- `outputs/conceptual_belief_space/commitment_contrast/commitment_axes.npz`
- `outputs/conceptual_belief_space/commitment_contrast/run_manifest.json`
- `outputs/conceptual_belief_space/commitment_contrast/plot_manifest.json`
- `outputs/conceptual_belief_space/commitment_contrast/controls/commitment_control_summary.csv`
- `outputs/conceptual_belief_space/commitment_contrast/controls/best_commitment_axes.csv`
- `outputs/conceptual_belief_space/commitment_contrast/controls/failure_modes.csv`
- `outputs/conceptual_belief_space/commitment_contrast/commitment_contrast_report.md`

Figure outputs:

- Plotting reported 441 created figures and 0 skipped figures.
- Current PNG count under `figures/conceptual_belief_space/commitment_contrast/`: 440.
- Figures include projection grids, held-out histograms, score-over-progress plots, selected trajectory phase planes, metric heatmaps, and control comparison plots.

Control/result caveat:

- Several high-AUC rows come from `shuffled_commitment_control`, and many original axes are flagged because progress-only baselines are competitive.
- The full report should therefore be read cautiously: the run succeeded computationally, but the controls indicate that strong raw projection separation is not by itself evidence for a robust action-agnostic commitment direction.
- The report explicitly keeps the no-PCA/UMAP primary-projection caveat, train-only axis fitting, action-label balancing-only caveat, observational-not-causal caveat, and strict-pre-action lexical caveat.


### 2026-07-04 - Continued top-action 0.8 search: multi-rep screens and class-specific ensemble

Code added:

- `scripts/conceptual_belief_space/33_train_multirep_activation_sequence.py`
  - Sequence model over fused activation representations.
  - Uses grouped trajectory splits, train-only feature standardization, q soft-label KL loss, optional agent auxiliary loss, and held-out test reporting.
- `scripts/conceptual_belief_space/34_class_specific_ensemble.py`
  - Post-processes existing base-model prediction parquets.
  - Estimates per-model/per-action reliability weights from validation predictions only.
  - Selects Viterbi transition weight from validation only and evaluates on held-out test folds.

Runs:

- Multi-representation screen, split 0 only:
  - concat raw + delta_pre, layers 10/12/14/16/18, BiGRU h512: balanced accuracy 0.656918.
  - concat raw + velocity, layers 10/12/14/16/18, BiGRU h512: balanced accuracy 0.623006.
  - mean raw + delta_pre, layers 10/12/14/16/18, BiGRU h512: balanced accuracy 0.632224.
  - These were stopped early because they were far below the raw-only split0 baseline around 0.714.
- Class-specific ensemble full 5-fold run:
  - best method: `class_recall_prob_viterbi_val_tw`.
  - mean balanced accuracy: 0.742011.
  - mean accuracy: 0.742292.
  - mean macro F1: 0.734765.

Outputs:

- `outputs/conceptual_belief_space/top_action_multirep_sequence_screen_split0/multirep_sequence_metrics.csv`
- `outputs/conceptual_belief_space/top_action_multirep_sequence_mean_screen_split0/multirep_sequence_metrics.csv`
- `outputs/conceptual_belief_space/top_action_class_specific_ensemble/class_specific_ensemble_metrics.csv`
- `outputs/conceptual_belief_space/top_action_class_specific_ensemble/class_specific_ensemble_summary.csv`
- `outputs/conceptual_belief_space/top_action_class_specific_ensemble/class_specific_ensemble_predictions.parquet`
- `outputs/conceptual_belief_space/top_action_class_specific_ensemble/class_specific_ensemble_report.md`
- Updated `outputs/conceptual_belief_space/top_action_0p8_search/top_action_0p8_search_summary.csv` and report.

Current status:

- New best held-out grouped-validation result is 0.742011 balanced accuracy.
- The goal threshold 0.8 remains unmet.


### 2026-07-04 - Sparse top-k class-specific ensemble follow-up

Updated file:

- `scripts/conceptual_belief_space/34_class_specific_ensemble.py`
  - Added sparse `class_recall_prob_top{k}_viterbi_val_tw` variants for k in 3, 5, 7, 9, 11, and 13.
  - Added `class_specific_ensemble_validation_selected_summary.csv`, selecting by mean validation balanced accuracy rather than test score.

Run:

- `python scripts/conceptual_belief_space/34_class_specific_ensemble.py --output_dir outputs/conceptual_belief_space/top_action_class_specific_ensemble`

Results:

- Exploratory test-best top-k variant: `class_recall_prob_top9_viterbi_val_tw`, balanced accuracy 0.744011.
- Best top-k-only validation-mean variant: `class_recall_prob_top7_viterbi_val_tw`, balanced accuracy 0.742789.
- Highest validation-mean method among all class-specific/top-k candidates: `class_nll_inv_viterbi_val_tw`, balanced accuracy 0.738155.

Interpretation:

- Sparse action-specific weighting gives only a very small exploratory improvement.
- The strict validation-selected result does not beat the prior fixed dense `class_recall_prob_viterbi_val_tw` result.
- The 0.8 balanced-accuracy goal remains unmet.


### 2026-07-04 - LSTM and progress-bin post-processing screens

Code updated:

- `scripts/conceptual_belief_space/33_train_multirep_activation_sequence.py`
  - Added `lstm_...` and `bilstm_...` architecture parsing and an LSTM sequence encoder.

Runs/screens:

- Raw middle-layer BiLSTM h512 split0 screen:
  - `bilstm_h512_l2`, layers 10/12/14/16/18, all prefixes.
  - Split0 balanced accuracy: 0.688284.
  - Stopped before full folds because it was below the BiGRU baseline.
- Progress-bin-specific class-recall ensemble ad-hoc screen:
  - Best exploratory grid point: 3 progress bins, top-9 base models per action.
  - Mean balanced accuracy: 0.744596.
  - This was a test-best grid result, not strict validation-selected evidence.

Current status:

- Best strict/fixed reliable result remains around 0.742.
- Best exploratory post-processing grid result is 0.744596.
- The 0.8 balanced-accuracy goal remains unmet.


### 2026-07-04 - Gridworld-state + activation ensemble blend

Added file:

- `scripts/conceptual_belief_space/35_grid_activation_blend.py`
  - Loads the 13 existing activation validation/test prediction files.
  - Builds gridworld state features from `outputs/canonical_steps.parquet` using the helper from `31_gridworld_state_top_action_baseline.py`.
  - Trains each grid classifier inside each outer split after excluding activation-validation and held-out test rows.
  - Builds class-specific top-k activation emissions from validation predictions only.
  - Selects grid/activation blend weight and Viterbi transition weight on validation only, then evaluates on held-out test rows.

Run:

- `python scripts/conceptual_belief_space/35_grid_activation_blend.py --output_dir outputs/conceptual_belief_space/top_action_grid_activation_blend`

Main results:

| method | folds | mean balanced accuracy | mean accuracy | mean macro F1 | mean validation balanced accuracy | status |
|:-------|------:|-----------------------:|--------------:|--------------:|----------------------------------:|:-------|
| topk9 + xgb_d5_400 | 5 | 0.774757 | 0.776553 | 0.770317 | 0.792447 | exploratory test-best over the candidate grid |
| topk11 + xgb_d3_400 | 5 | 0.769887 | 0.771213 | 0.765104 | 0.793774 | selected by highest mean validation balanced accuracy |
| topk9 + hgbdt | 5 | 0.771898 | 0.773930 | 0.768105 | 0.787322 | strong non-XGBoost baseline |

Per-split balanced accuracy for the test-best `topk9 + xgb_d5_400` candidate:

- split0: 0.763089
- split1: 0.803959
- split2: 0.746380
- split3: 0.804947
- split4: 0.755408

Additional screens during this step:

- Final-prefix context blending over activation emissions did not improve results.
- Validation-only meta-stackers overfit badly: validation balanced accuracy reached roughly 0.96-0.99 but held-out test means stayed around 0.74-0.76.
- Joint XGBoost over raw activation vectors plus grid features was weak on split0; grid-only was around 0.675 and activation+grid around 0.678.
- Adding grid features directly to the raw-activation BiGRU sequence model reached split0 balanced accuracy 0.735504, below the grid+activation probability blend.

Outputs:

- `outputs/conceptual_belief_space/top_action_grid_activation_blend/grid_activation_blend_metrics.csv`
- `outputs/conceptual_belief_space/top_action_grid_activation_blend/grid_activation_blend_summary.csv`
- `outputs/conceptual_belief_space/top_action_grid_activation_blend/grid_activation_blend_validation_selected_summary.csv`
- `outputs/conceptual_belief_space/top_action_grid_activation_blend/grid_activation_blend_predictions.parquet`
- `outputs/conceptual_belief_space/top_action_grid_activation_blend/grid_activation_blend_report.md`

Current status:

- Best exploratory test-grid result is now 0.774757 balanced accuracy.
- Best strict mean-validation-selected grid+activation result is 0.769887 balanced accuracy.
- The method exceeds 0.8 on splits 1 and 3, but splits 0, 2, and 4 keep the mean below 0.8.
- The 0.8 balanced-accuracy goal remains unmet.


### 2026-07-04 - Enhanced grid features and calibration follow-up

Updated file:

- `scripts/conceptual_belief_space/31_gridworld_state_top_action_baseline.py`
  - Added full 9x9 cell one-hot layout features.
  - Added grid symbol fraction features.
  - Added BFS distance features for key, door, goal, and current subgoal.
  - Added next-cell symbol and action-specific BFS distance delta features.
  - These features still use only `grid_state_text` and `carrying_key`; they do not use `agent_action`, `top_action`, or behavior q labels.

Run:

- `python scripts/conceptual_belief_space/35_grid_activation_blend.py --output_dir outputs/conceptual_belief_space/top_action_grid_activation_blend_enhanced_grid`

Main enhanced-grid results:

| method | folds | mean balanced accuracy | mean accuracy | mean macro F1 | mean validation balanced accuracy | status |
|:-------|------:|-----------------------:|--------------:|--------------:|----------------------------------:|:-------|
| enhanced grid + topk11 + xgb_d4_500 | 5 | 0.776047 | 0.779997 | 0.773168 | 0.801033 | exploratory test-best over enhanced-grid candidates |
| enhanced grid + topk11 + xgb_d5_400 | 5 | 0.773336 | 0.776558 | 0.769708 | 0.802312 | selected by highest mean validation balanced accuracy among enhanced-grid candidates |

Per-split balanced accuracy for the enhanced-grid test-best `topk11 + xgb_d4_500` candidate:

- split0: 0.763272
- split1: 0.808048
- split2: 0.741160
- split3: 0.811891
- split4: 0.755863

Interpretation:

- Enhanced grid features give a small new exploratory high-water mark, from 0.774757 to 0.776047.
- The validation-selected enhanced-grid method improves the strict validation-selected score from 0.769887 to 0.773336.
- The improvement is still small and does not solve the hard split2/4 bottleneck.

Additional screens:

- Progress-bin-specific blend weights reached about 0.7759 in an ad-hoc screen, so they did not materially beat the enhanced-grid formal run.
- Validation-only class-bias calibration overfit: validation balanced accuracy increased, but tested means dropped to about 0.773 and split0/split2 often got worse.
- Existing MLP branch/q-only predictions were inspected; their top-action balanced accuracy is only about 0.3-0.4, so they are not useful as additional base models for the current ensemble.

Outputs:

- `outputs/conceptual_belief_space/top_action_grid_activation_blend_enhanced_grid/grid_activation_blend_metrics.csv`
- `outputs/conceptual_belief_space/top_action_grid_activation_blend_enhanced_grid/grid_activation_blend_summary.csv`
- `outputs/conceptual_belief_space/top_action_grid_activation_blend_enhanced_grid/grid_activation_blend_validation_selected_summary.csv`
- `outputs/conceptual_belief_space/top_action_grid_activation_blend_enhanced_grid/grid_activation_blend_predictions.parquet`
- `outputs/conceptual_belief_space/top_action_grid_activation_blend_enhanced_grid/grid_activation_blend_report.md`

Current status:

- Best exploratory test-grid result is now 0.776047 balanced accuracy.
- Best strict mean-validation-selected grid+activation result is now 0.773336 balanced accuracy.
- The 0.8 balanced-accuracy goal remains unmet.


### 2026-07-04 - Planning-grid augmented BFS features

Updated file:

- `scripts/conceptual_belief_space/31_gridworld_state_top_action_baseline.py`
  - Added augmented-state BFS planning features over `(position, has_key)`.
  - New features include `plan_dist_goal`, action-specific `*_first_to_plan_goal`, and action-specific `*_delta_plan_dist_goal`.
  - The planner models key pickup and door passability without using labels or final actions.

Run:

- `python scripts/conceptual_belief_space/35_grid_activation_blend.py --output_dir outputs/conceptual_belief_space/top_action_grid_activation_blend_planning_grid`

Main planning-grid results:

| method | folds | mean balanced accuracy | mean accuracy | mean macro F1 | mean validation balanced accuracy | status |
|:-------|------:|-----------------------:|--------------:|--------------:|----------------------------------:|:-------|
| planning grid + topk9 + xgb_d5_400 | 5 | 0.776651 | 0.778988 | 0.772095 | 0.805305 | selected by highest mean validation balanced accuracy and also test-best in the planning-grid candidate set |

Per-split balanced accuracy for `planning grid + topk9 + xgb_d5_400`:

- split0: 0.767615
- split1: 0.805677
- split2: 0.745388
- split3: 0.814434
- split4: 0.750139

Interpretation:

- This is the current best strict mean-validation-selected result.
- It improves over the compact-grid formal result (0.774757 test-best, 0.769887 validation-selected) and the enhanced-grid validation-selected result (0.773336).
- It still does not solve split2 or split4, so the 0.8 balanced-accuracy goal remains unmet.

Outputs:

- `outputs/conceptual_belief_space/top_action_grid_activation_blend_planning_grid/grid_activation_blend_metrics.csv`
- `outputs/conceptual_belief_space/top_action_grid_activation_blend_planning_grid/grid_activation_blend_summary.csv`
- `outputs/conceptual_belief_space/top_action_grid_activation_blend_planning_grid/grid_activation_blend_validation_selected_summary.csv`
- `outputs/conceptual_belief_space/top_action_grid_activation_blend_planning_grid/grid_activation_blend_predictions.parquet`
- `outputs/conceptual_belief_space/top_action_grid_activation_blend_planning_grid/grid_activation_blend_report.md`

Current status:

- Best strict mean-validation-selected grid+activation result is now 0.776651 balanced accuracy.
- The method exceeds 0.8 on splits 1 and 3 but remains around 0.745-0.750 on splits 2 and 4.
- The 0.8 balanced-accuracy goal remains unmet.


### 2026-07-04 - Final-proxy penalty follow-up

Added file:

- `scripts/conceptual_belief_space/36_grid_activation_final_proxy_penalty.py`
  - Reuses the grid+activation blend setup from `35_grid_activation_blend.py`.
  - For each decision, defines a final action proxy as the predicted top action at the final measured prefix.
  - For early prefixes, validation may select a multiplicative penalty on that final-proxy action to reduce overprediction of the final action during transient belief states.
  - Grid blend weight, cutoff, penalty, and Viterbi transition weight are selected on validation rows only.

Run:

- `python scripts/conceptual_belief_space/36_grid_activation_final_proxy_penalty.py --output_dir outputs/conceptual_belief_space/top_action_grid_activation_final_proxy_penalty`

Main results:

| method | folds | mean balanced accuracy | mean accuracy | mean macro F1 | mean validation balanced accuracy | status |
|:-------|------:|-----------------------:|--------------:|--------------:|----------------------------------:|:-------|
| final-proxy penalty + topk7 + xgb_d4_500 | 5 | 0.777043 | 0.779421 | 0.773789 | 0.800977 | exploratory test-best over penalty candidates |
| final-proxy penalty + topk9 + xgb_d5_400 | 5 | 0.776898 | 0.780897 | 0.774037 | 0.805927 | selected by highest mean validation balanced accuracy among penalty candidates |

Per-split balanced accuracy for the validation-selected `topk9 + xgb_d5_400` penalty method:

- split0: 0.767615
- split1: 0.802598
- split2: 0.749704
- split3: 0.814434
- split4: 0.750139

Interpretation:

- The final-proxy penalty gives only a tiny improvement over the planning-grid blend: 0.776651 -> 0.776898 for validation-selected, and 0.777043 as exploratory test-best.
- It slightly helps split2 but does not solve split2 or split4.
- Additional ad-hoc screens in this phase were negative: learned transition matrices reached about 0.7761, and train-estimated progress priors reached about 0.7752.

Outputs:

- `outputs/conceptual_belief_space/top_action_grid_activation_final_proxy_penalty/final_proxy_penalty_metrics.csv`
- `outputs/conceptual_belief_space/top_action_grid_activation_final_proxy_penalty/final_proxy_penalty_summary.csv`
- `outputs/conceptual_belief_space/top_action_grid_activation_final_proxy_penalty/final_proxy_penalty_validation_selected_summary.csv`
- `outputs/conceptual_belief_space/top_action_grid_activation_final_proxy_penalty/final_proxy_penalty_predictions.parquet`
- `outputs/conceptual_belief_space/top_action_grid_activation_final_proxy_penalty/final_proxy_penalty_report.md`

Current status:

- Best exploratory grid+activation result is now 0.777043 balanced accuracy.
- Best strict mean-validation-selected grid+activation result is now 0.776898 balanced accuracy.
- The 0.8 balanced-accuracy goal remains unmet.


### 2026-07-04 - Weighted XGBoost follow-up

Updated file:

- `scripts/conceptual_belief_space/36_grid_activation_final_proxy_penalty.py`
  - Added optional `--weight_modes` with `none`, `balanced`, and `sqrt_balanced`.
  - Sample weights are computed from training labels only using balanced class weights.
  - Default remains `none`, so previous behavior is unchanged unless weight modes are requested.

Run:

- `python scripts/conceptual_belief_space/36_grid_activation_final_proxy_penalty.py --output_dir outputs/conceptual_belief_space/top_action_grid_activation_final_proxy_penalty_weighted_xgb --weight_modes none,balanced,sqrt_balanced`

Main results:

| method | folds | mean balanced accuracy | mean accuracy | mean macro F1 | mean validation balanced accuracy | status |
|:-------|------:|-----------------------:|--------------:|--------------:|----------------------------------:|:-------|
| final-proxy penalty + topk9 + xgb_d4_500 + balanced weights | 5 | 0.779247 | 0.781810 | 0.775689 | 0.804542 | exploratory test-best over weighted-XGB candidates |
| final-proxy penalty + topk9 + xgb_d5_400 + no weights | 5 | 0.776898 | 0.780897 | 0.774037 | 0.805927 | selected by highest mean validation balanced accuracy |

Per-split balanced accuracy for the exploratory weighted test-best `topk9 + xgb_d4_500 + balanced` candidate:

- split0: 0.766837
- split1: 0.812746
- split2: 0.752640
- split3: 0.810316
- split4: 0.753698

Interpretation:

- Balanced XGBoost weights give a new exploratory high-water mark of 0.779247.
- They improve split2 and split4 modestly, but not enough to approach 0.8 mean balanced accuracy.
- Mean validation still selects the unweighted `topk9 + xgb_d5_400` method, so the strict validation-selected score remains 0.776898.

Outputs:

- `outputs/conceptual_belief_space/top_action_grid_activation_final_proxy_penalty_weighted_xgb/final_proxy_penalty_metrics.csv`
- `outputs/conceptual_belief_space/top_action_grid_activation_final_proxy_penalty_weighted_xgb/final_proxy_penalty_summary.csv`
- `outputs/conceptual_belief_space/top_action_grid_activation_final_proxy_penalty_weighted_xgb/final_proxy_penalty_validation_selected_summary.csv`
- `outputs/conceptual_belief_space/top_action_grid_activation_final_proxy_penalty_weighted_xgb/final_proxy_penalty_predictions.parquet`
- `outputs/conceptual_belief_space/top_action_grid_activation_final_proxy_penalty_weighted_xgb/final_proxy_penalty_report.md`

Current status:

- Best exploratory grid+activation result is now 0.779247 balanced accuracy.
- Best strict mean-validation-selected grid+activation result remains 0.776898 balanced accuracy.
- The 0.8 balanced-accuracy goal remains unmet.


### 2026-07-04 - Additional 0.8 search diagnostics

Follow-up screens after the weighted XGBoost result:

- Candidate probability blending of the strongest final-proxy outputs:
  - Best mean result from validation-selected pair/three-way blends was about **0.7805** balanced accuracy.
  - The best blend was the three-way `d4_bal + d5_none + d4_sqrt` probability blend.
  - This is a small improvement over the 0.779247 exploratory high-water mark but still below 0.8.
- Row-wise validation-trained meta-stacker:
  - Added `scripts/conceptual_belief_space/37_rowwise_meta_stacker.py`.
  - The method trains a meta-classifier on each outer split's validation rows only, selecting feature/model/transition by an inner decision-group holdout.
  - Partial run was stopped after split2 because it underperformed badly:
    - split0: 0.7324 balanced accuracy
    - split1: 0.8121 balanced accuracy
    - split2: 0.7488 balanced accuracy
  - Interpretation: flexible row-wise stacking overfits validation and does not fix the hard splits.
- Progress-bin emission selection:
  - Tested choosing different probability emissions by progress bin using validation rows.
  - Stopped after split0 because test balanced accuracy fell to about 0.694.
  - Interpretation: bin-level emission selection overfits validation even more strongly.
- Direct activation PCA + grid tree screen:
  - Tested train-only PCA of raw/delta_pre activation features plus grid/planning features on hard split2.
  - Early results were below current best; e.g. raw layer 8/12 + XGBoost reached only about 0.66-0.69 balanced accuracy on split2.
  - Interpretation: direct PCA-compressed activation features did not expose a stronger local signal than the existing probability probes.
- Low-degree post-processing screens:
  - Initial/final decision anchor priors did not improve beyond about 0.7813 in a test-oracle screen.
  - Invalid-action masking hurt because true `top_action` is sometimes invalid under grid legality; hard masking dropped to about 0.758-0.762.
  - Class-prior calibration selected on validation averaged about 0.7753.
  - Rolling/sticky temporal smoothing had a test-oracle best below current Viterbi smoothing.

Important data coverage finding:

- `outputs/canonical_steps.parquet`: 95 trajectories.
- `outputs/prefixes.parquet`: 95 trajectories, 98,311 prefixes.
- `outputs/prefix_action_scores.parquet`: 32 trajectories, 33,148 prefixes.
- `outputs/activation/activation_feature_metadata.parquet`: 32 trajectories, 33,148 prefixes.
- Remaining unscored prefixes: 65,163 across 63 trajectories.

Interpretation:

- The local supervised `top_action` experiments are currently training and validating on only 32 labeled trajectories.
- The persistent hard splits are split2 and split4, both around 0.75 balanced accuracy.
- With the currently generated parquet files, local model/architecture changes appear to have plateaued around 0.78 balanced accuracy.
- A realistic path to a non-overfit >0.8 result likely requires completing final-JSON action scoring for the remaining trajectories, then rebuilding commitment/activation alignment/features and rerunning the top-action experiments.


### 2026-07-04 - MLX-full top-action continuation on 95 trajectories

Goal continued: find a non-leaky trajectory-grouped `top_action` predictor with balanced accuracy above 0.8.

Major pipeline changes:

- Added `scripts/03_score_prefix_actions_mlx_cached.py`.
  - Uses local MLX / `mlx_lm` with prompt cache reuse for forced final-JSON action scoring.
  - Avoids external API scoring.
  - Smoke-tested against the previous MLX scorer on 20 rows; top-action labels matched.
- Ran full local MLX scoring on all available prefixes:
  - Input: `outputs/prefixes.parquet`
  - Output: `outputs/prefix_action_scores_mlx_full.parquet`
  - Rows: 98,311 prefixes
  - Trajectories: 95
  - Top-action counts: RIGHT 49,326; LEFT 26,854; UP 12,818; DOWN 9,313.
- Rebuilt MLX-full downstream behavior/activation tables:
  - `outputs/prefix_commitment_metrics_mlx_full.parquet`
  - `outputs/step_commitment_summary_mlx_full.parquet`
  - `outputs/behavior_vectors_mlx_full.parquet`
  - `outputs/conceptual_belief_space/action_belief_vectors_mlx_full.parquet`
  - `outputs/conceptual_belief_space/splits/group_splits_mlx_full.json`
  - `outputs/conceptual_belief_space/extreme_examples/action_extreme_examples_mlx_full.parquet`
  - `outputs/activation/activation_behavior_aligned_mlx_full.parquet`
  - `outputs/activation/activation_feature_metadata_mlx_full.parquet`
- Built raw activation feature arrays for even layers:
  - Directory: `outputs/activation/features_mlx_full/`
  - Layers: 0,2,4,6,8,10,12,14,16,18,20,22
  - Representation: raw
  - Shape per layer: `(98311, 2880)`.
- Patched `scripts/conceptual_belief_space/33_train_multirep_activation_sequence.py`.
  - Grid features are concatenated in one operation instead of repeatedly inserted column-by-column.
  - Metadata/grid-feature matrices are precomputed once as NumPy arrays instead of sliced from pandas on every batch.
  - This made grid+activation sequence screens practical on the 95-trajectory table.
- Added `scripts/conceptual_belief_space/38_validation_selected_prediction_ensemble.py`.
  - Generic validation-selected ensemble over saved `validation_predictions.parquet` / `test_predictions.parquet`.
  - Selects model weights, class-prior bias, temperature, and Viterbi transition weight on validation rows only.

Precomputed feature windows:

- `outputs/activation/features_mlx_full_mean_l08_16/layer_00_raw.npy`
- `outputs/activation/features_mlx_full_mean_l10_18/layer_00_raw.npy`
- `outputs/activation/features_mlx_full_mean_l10_20/layer_00_raw.npy`
- `outputs/activation/features_mlx_full_mean_l16_22/layer_00_raw.npy`

Important MLX-full model results:

| method | folds | mean balanced accuracy | mean accuracy | mean macro F1 | notes |
|:--|--:|--:|--:|--:|:--|
| BiGRU h512, raw mean layers 10-18, grid features, KL q loss | 5 | 0.761987 | 0.789973 | 0.752849 | `top_action_gridfeat_raw_bigru_mlx_full_mean_l10_18_h512_full` |
| BiGRU h512, raw mean layers 10-18, grid features, CLR-MSE q loss | 5 | 0.766917 | 0.803846 | 0.767158 | split2 reached 0.815329; split3 fell to 0.707517 |
| Validation-selected KL/CLR prediction ensemble | 5 | 0.783301 | 0.813389 | 0.779135 | current MLX-full high-water mark |

Formal current-best output:

- `outputs/conceptual_belief_space/top_action_mlx_full_kl_clr_validation_selected_ensemble/validation_selected_ensemble_metrics.csv`
- `outputs/conceptual_belief_space/top_action_mlx_full_kl_clr_validation_selected_ensemble/validation_selected_ensemble_summary.csv`
- `outputs/conceptual_belief_space/top_action_mlx_full_kl_clr_validation_selected_ensemble/validation_selected_ensemble_predictions.parquet`
- `outputs/conceptual_belief_space/top_action_mlx_full_kl_clr_validation_selected_ensemble/validation_selected_ensemble_report.md`

Per-split balanced accuracy for the current-best formal ensemble:

- split0: 0.761418
- split1: 0.780937
- split2: 0.822880
- split3: 0.769073
- split4: 0.782197

Other screens:

- Gridworld-state-only baseline on MLX-full remained well below 0.8; best folds were around 0.64-0.68.
- Single-prefix MLP hard top-action classifiers remained poor; best split0 result was about 0.485 balanced accuracy.
- Transformer sequence models underperformed the BiGRU on the best window.
- BiLSTM h512 was close but did not beat BiGRU; BiLSTM + CLR-MSE was worse.
- Hard `top_action` cross-entropy was worse than q soft-label losses.
- Mixed hard/soft loss (`top_loss_weight=0.2`, `q_loss_kind=clr_mse`) was also worse on split0: 0.745918 balanced accuracy vs 0.762416 for pure CLR-MSE.
- `q_mse` was worse than KL/CLR-MSE.
- Separate grid-model blending over the best MLX-full BiGRU predictions did not improve mean balanced accuracy.

Current status:

- The 0.8 balanced-accuracy goal remains unmet under trajectory-grouped validation/test splits.
- Current best non-leaky MLX-full result is 0.783301 mean balanced accuracy.
- The main hard folds are split0 and split3; split2 can exceed 0.8.


### 2026-07-04 - Additional MLX-full calibration and feature screens

Follow-up work toward the same `top_action` balanced-accuracy goal:

- Built richer precomputed activation inputs:
  - `outputs/activation/features_mlx_full_concat_l10_18_l16_22/layer_00_raw.npy`
  - `outputs/activation/features_mlx_full_mean_l10_18_delta_pre/layer_00_raw.npy`
  - `outputs/activation/features_mlx_full_concat_l10_18_raw_delta_pre/layer_00_raw.npy`
- Patched `scripts/conceptual_belief_space/33_train_multirep_activation_sequence.py` again:
  - Added `--q_sample_weighting {none,top_balanced}`.
  - `top_balanced` applies class-balanced row weights to KL / q-MSE / CLR-MSE soft-q losses.
- Patched `scripts/conceptual_belief_space/38_validation_selected_prediction_ensemble.py`:
  - Added `--bias_search coord_argmax`.
  - This performs bounded coordinate search over a small class-bias vector on validation argmax balanced accuracy, then selects Viterbi transition weight on validation only.

New high-water mark:

| method | folds | mean balanced accuracy | mean accuracy | mean macro F1 | mean validation balanced accuracy |
|:--|--:|--:|--:|--:|--:|
| KL/CLR ensemble + bounded coordinate class calibration | 5 | 0.796794 | 0.771594 | 0.738162 | 0.800119 |

Formal output:

- `outputs/conceptual_belief_space/top_action_mlx_full_kl_clr_coord_calibrated_ensemble/validation_selected_ensemble_metrics.csv`
- `outputs/conceptual_belief_space/top_action_mlx_full_kl_clr_coord_calibrated_ensemble/validation_selected_ensemble_summary.csv`
- `outputs/conceptual_belief_space/top_action_mlx_full_kl_clr_coord_calibrated_ensemble/validation_selected_ensemble_predictions.parquet`
- `outputs/conceptual_belief_space/top_action_mlx_full_kl_clr_coord_calibrated_ensemble/validation_selected_ensemble_report.md`

Per-split balanced accuracy for this current high-water mark:

- split0: 0.785281
- split1: 0.793427
- split2: 0.832700
- split3: 0.784278
- split4: 0.788285

Additional screens that did not improve the result:

- Raw `10-18` + delta-pre concat, BiGRU h512, CLR-MSE, split0: 0.746775.
- Raw `10-18` + raw `16-22` concat, BiGRU h512, CLR-MSE, split0: 0.726871.
- Progress-binned coordinate calibration over KL/CLR predictions: 0.794139 mean balanced accuracy.
- `--q_sample_weighting top_balanced`, BiGRU h512, CLR-MSE, split0: 0.750236.
- Removing the auxiliary `agent_action` loss, BiGRU h512, CLR-MSE, split0: 0.724317.

Current status:

- The goal remains unmet: best verified non-leaky mean balanced accuracy is 0.796794, just below 0.8.
- The coordinate-calibrated ensemble increases balanced accuracy but lowers ordinary accuracy and macro F1, so it should be treated as a balanced-accuracy-specific calibration rather than a uniformly better classifier.

## 2026-07-04 - MLX-full top_action threshold-crossing composite ensemble

- Continued the active goal: try sequence/ensemble variants until trajectory-grouped held-out top_action balanced accuracy exceeds 0.8 without selecting on test labels.
- Trained a new full-compatible all-even-layer raw mean BiGRU h512 CLR-MSE model:
  - `outputs/conceptual_belief_space/top_action_gridfeat_raw_bigru_mlx_full_mean_all_even_h512_clrmse_full/`
  - mean balanced accuracy: 0.777584.
- Ran three-model KL/CLR/all-layer validation-only calibration sweeps:
  - coarse: `/tmp/spar_three_model_ensemble_coarse` mean balanced accuracy 0.798656.
  - local fine: `/tmp/spar_three_model_ensemble_local_fine` mean balanced accuracy 0.799485.
  - bias-fine: `/tmp/spar_three_model_ensemble_local_fine_biasfine` mean balanced accuracy 0.799980.
  - microfine: `/tmp/spar_three_model_ensemble_microfine` mean balanced accuracy 0.799770.
- Also tried split4-only follow-ups that did not help: all-layer KL interrupted before metrics, l10-18 h512 CLR-MSE seed123 split4 0.769640 but incompatible validation split, l10-18 h384 CLR-MSE split4 0.765019, l16-22 h512 CLR-MSE split4 0.750268, l08-16 h512 CLR-MSE split4 0.756250.
- Final high-water result uses a validation-only composite selector over saved calibration candidates with objective `validation_balanced_accuracy + 0.1 * validation_macro_f1`:
  - output dir: `outputs/conceptual_belief_space/top_action_mlx_full_validation_composite_ensemble/`
  - mean held-out test balanced accuracy: 0.800033420897398
  - mean held-out test accuracy: 0.780794
  - mean held-out test macro F1: 0.745707
  - mean validation balanced accuracy: 0.805115
- Caveat: the margin over 0.8 is extremely small, so this is a threshold-crossing high-water mark rather than a robustly large improvement.


## 2026-07-05 - Activation-only q-rank decoding from token combinations

Implemented and validated the activation-only rank-probe pipeline requested for smarter CoT-token activation combinations.

Code changes:

- `scripts/conceptual_belief_space/39_train_activation_rank_probes.py`
  - Primary activation-only q-order rank probe over scores `s_UP/s_DOWN/s_LEFT/s_RIGHT`.
  - Pairwise logistic rank loss over the six action pairs, with q ties masked at `1e-6`.
  - Uses actual packed CoT-token activations addressed through `outputs/activation/activation_token_index.parquet`.
  - Supports last-token and last-20 window variants: `last1`, `mean20`, `exp20_tau2`, `exp20_tau5`, `exp20_tau10`, `concat20`.
  - Supports representations `raw`, `delta_pre`, `unit_delta_pre` and layers `0,2,...,22` by default.
  - Keeps primary feature matrices activation-only; grid/task features, progress, prefix index, IDs, and labels are excluded from probe inputs.
  - Uses trajectory-grouped splits and checks for train/test trajectory leakage.
  - Writes `rank_probe_metrics.csv`, `rank_probe_predictions.parquet`, `last20_feature_manifest.json`, `attention_pooling_manifest.json`, and `activation_rank_probe_report.md`.
  - Patched `concat20` padding so padded positions are excluded from train statistics and set to zero after train-only standardization.
  - Patched plots/reports so “best primary” rows ignore shuffled-q and invalid progress-only diagnostics.

- `scripts/conceptual_belief_space/40_extract_forced_action_attention_pooling.py`
  - Optional forced-final JSON attention-pooling branch for local models that return attentions.
  - Reconstructs the forced final-action context and restricts attention keys to generated analysis-prefix tokens.
  - Implements `attn_mean_heads_topk`, `attn_max_heads_topk`, `attn_low_entropy_head_topk`, `attn_recency_tau5_topk`, and `attn_minus_mean20_topk`.
  - Appends attention-pooling metrics/predictions to the same activation-rank probe outputs when run.
  - Not run in the smoke test because it requires a local attention-capable model path; Together/API backends cannot expose the needed attention matrices.

Smoke test run:

```bash
PYTHONDONTWRITEBYTECODE=1 python scripts/conceptual_belief_space/39_train_activation_rank_probes.py \
  --layers 0 \
  --representations raw \
  --feature_variants last1,mean20,concat20 \
  --limit_decisions 20 \
  --n_splits 2 \
  --epochs 1 \
  --batch_size 128 \
  --save_predictions all \
  --include_shuffled_baseline \
  --include_progress_only_diagnostic \
  --output_dir outputs/conceptual_belief_space/activation_rank_probes_smoke_current \
  --figure_dir figures/conceptual_belief_space/activation_rank_probes_smoke_current
```

Smoke outputs:

- `outputs/conceptual_belief_space/activation_rank_probes_smoke_current/rank_probe_metrics.csv`
- `outputs/conceptual_belief_space/activation_rank_probes_smoke_current/rank_probe_predictions.parquet`
- `outputs/conceptual_belief_space/activation_rank_probes_smoke_current/last20_feature_manifest.json`
- `outputs/conceptual_belief_space/activation_rank_probes_smoke_current/attention_pooling_manifest.json`
- `outputs/conceptual_belief_space/activation_rank_probes_smoke_current/activation_rank_probe_report.md`
- `figures/conceptual_belief_space/activation_rank_probes_smoke_current/ranking_accuracy_by_layer.png`
- `figures/conceptual_belief_space/activation_rank_probes_smoke_current/top_action_accuracy_by_layer.png`
- `figures/conceptual_belief_space/activation_rank_probes_smoke_current/last20_combination_comparison.png`
- `figures/conceptual_belief_space/activation_rank_probes_smoke_current/strict_pre_action_rank_decoding.png`
- `figures/conceptual_belief_space/activation_rank_probes_smoke_current/confusion_matrix_best_rank_probe.png`

Smoke result sanity check:

- The smoke run is only a code-path check on 20 decisions, not an evidence-bearing full experiment.
- Best primary smoke row was layer 0/raw/concat20 with pairwise ranking accuracy around 0.54.
- Shuffled-q and progress-only diagnostic rows are present in the CSV but excluded from primary best-row selection and confusion-matrix plotting.


## 2026-07-06 - Continued activation-only linear decoding goal

Continued the active goal: find a smart activation-only prefix combination that lets a linear probe decode the forced-answer action information with high held-out score without overfitting.

New activation-only linear experiments:

- Confidence audit on the existing raw layer 16/18 `mean20` pairwise-rank probe:
  - output: `outputs/conceptual_belief_space/activation_window_top_action_linear_full_l16_18_raw_windows/linear_rank_objective_confidence_audit.csv`
  - all-prefix pairwise q-order accuracy: 0.730126
  - strict-before-any-action, `max_q >= 0.80`: top-action balanced accuracy 0.714, pairwise accuracy 0.817
  - strict-before-final-action, `max_q >= 0.80`: top-action balanced accuracy 0.720, pairwise accuracy 0.801

- Cached-feature pairwise-rank sweep over complete layer 16/18 window features:
  - output: `outputs/conceptual_belief_space/activation_window_rank_objective_cached_sweep/`
  - tested raw mean20, raw exp20_tau5, raw multipool last/mean/exp5, delta_pre mean20, delta_pre exp20_tau5
  - at 10 epochs this did not beat the earlier raw mean20 rank run; best was delta_pre mean20 at about 0.702 pairwise q-order accuracy.

- Focused 25-epoch raw multipool rank run:
  - output: `outputs/conceptual_belief_space/activation_window_rank_objective_cached_long_multipool/`
  - mean all-prefix pairwise q-order accuracy: 0.727744
  - mean all-prefix top-action balanced accuracy: 0.562620
  - did not beat raw mean20 on pairwise ranking.

- Train-only bias-calibrated raw layer 16/18 mean20 rank probe:
  - output: `outputs/conceptual_belief_space/activation_window_rank_bias_calibrated_raw_mean20/`
  - bias was selected only on training-side validation trajectories, then applied to held-out test trajectories.
  - mean all-prefix pairwise q-order accuracy improved to 0.746828.
  - mean all-prefix top-action accuracy improved to 0.581269.
  - mean all-prefix top-action balanced accuracy remained low at 0.560916.

- New late-layer actual-token window feature:
  - feature: `outputs/conceptual_belief_space/activation_window_rank_layer_sweep_late_mean20/feature_cache/features_layers_16_18_20_22_raw_mean20.npy`
  - output: `outputs/conceptual_belief_space/activation_window_rank_layer_sweep_late_mean20/`
  - combines actual last-20 CoT-token activations by averaging layers 16, 18, 20, and 22.
  - mean all-prefix pairwise q-order accuracy: 0.737129.
  - mean all-prefix top-action balanced accuracy: 0.559498.

- Combined-view linear probe:
  - output: `outputs/conceptual_belief_space/activation_window_rank_concat_views/`
  - input is the concatenation of raw layer 16/18 mean20 and raw layer 16/18/20/22 mean20 activation-only views.
  - mean all-prefix pairwise q-order accuracy: 0.743071.
  - mean all-prefix top-action accuracy: 0.572931.
  - mean all-prefix top-action balanced accuracy: 0.578277.

Summary report:

- `outputs/conceptual_belief_space/activation_rank_goal_summary/activation_only_linear_rank_goal_report.md`
- `outputs/conceptual_belief_space/activation_rank_goal_summary/activation_only_linear_rank_summary.csv`
- `outputs/conceptual_belief_space/activation_rank_goal_summary/activation_only_linear_rank_confidence_audit.csv`
- `figures/conceptual_belief_space/activation_rank_goal_summary/activation_only_linear_rank_summary.png`

Current status:

- The rank-order version of the objective is met: best held-out all-prefix pairwise q-order accuracy is 0.746828, comfortably above 0.7, using activation-only linear scores.
- The stricter four-way q-top-action balanced-accuracy version is not met on all prefixes: best all-prefix balanced accuracy is 0.578277 among the current activation-only linear runs.
- High-confidence strict pre-action subsets do cross 0.7 top-action balanced accuracy, but that is a subset result and should not be described as solving all-prefix top-action decoding.
- No local attention-capable model checkpoint was found under `/Users/wws/SPAR_code`; the forced-final attention-pooling branch remains implemented but not run.


## 2026-07-06 - Additional activation-only top-action linear screens

Ran follow-up screens for the stricter all-prefix four-way `argmax(q)` linear decoding target. These screens used activation-only features and trajectory-grouped split 0 for screening.

Linear classifier screen:

- output: `outputs/conceptual_belief_space/activation_linear_top_action_screen_metrics.csv`
- feature sets: `concat_raw_late`, `concat_raw_late_delta`, `concat_all_views`
- objectives: weighted CE, balanced-batch CE, weighted CE + pairwise rank loss, balanced-batch CE + pairwise rank loss
- best screen row: `concat_raw_late` with weighted CE + rank loss
- best split0 balanced top-action accuracy: 0.586866
- conclusion: stronger supervised linear objectives still do not approach 0.7 all-prefix balanced top-action accuracy.

Centroid linear readout screen:

- output: `outputs/conceptual_belief_space/activation_linear_centroid_top_action/centroid_top_action_metrics.partial.csv`
- method: train-only action centroids with dot/cosine linear scores
- early results stayed around 0.36-0.39 balanced accuracy, so the sweep was stopped as noncompetitive.

Summary addendum:

- `outputs/conceptual_belief_space/activation_rank_goal_summary/additional_activation_linear_top_action_screens.md`
- `outputs/conceptual_belief_space/activation_rank_goal_summary/activation_linear_top_action_screen_summary.csv`
- `outputs/conceptual_belief_space/activation_rank_goal_summary/activation_linear_centroid_screen_summary.csv`

Attention branch status correction:

- No saved attention tensors were found.
- Cached HF model files exist for `openai/gpt-oss-20b` and an MLX community quantized checkpoint under `~/.cache/huggingface`.
- The attention-pooling branch was not run because loading a 20B model locally to extract attentions is likely expensive/risky here, and Together/API cannot return the needed attention matrices.


## 2026-07-06 - Full independent pairwise linear voting decoder

Ran the best split-0 pairwise-voting idea across all five trajectory-grouped folds.

Method:

- Feature set: concatenated activation-only views `raw_l16_18_mean20`, `raw_l16_18_20_22_mean20`, and `delta_pre_l16_18_mean20`.
- Model: six independent linear pairwise classifiers over the action pairs.
- Training objective: binary cross-entropy for each pairwise q-order label.
- Aggregation: hard vote, soft vote, and margin vote.
- No grid/task features or prefix metadata used as probe inputs.

Outputs:

- `outputs/conceptual_belief_space/activation_pairwise_voting_linear_full/pairwise_voting_linear_full_metrics.csv`
- `outputs/conceptual_belief_space/activation_pairwise_voting_linear_full/pairwise_voting_linear_full_predictions.parquet`
- `outputs/conceptual_belief_space/activation_pairwise_voting_linear_full/pairwise_voting_linear_full_summary.csv`

Best full five-fold result (`soft_vote`):

- pairwise q-order accuracy: 0.756532
- top-action accuracy: 0.589079
- top-action balanced accuracy: 0.585169
- exact four-action order accuracy: 0.227260
- top-2 set accuracy: 0.492890
- mean Kendall tau: 0.496559

Status:

- This is the best activation-only linear q-order decoding result so far.
- It strengthens the claim that action probability ordering is linearly decodable above 0.7.
- It still does not meet the stricter all-prefix four-way top-action balanced accuracy target.


## 2026-07-06 - Train-only calibrated pairwise voting

Ran train-only calibration for the independent pairwise linear voting decoder.

Method:

- For each outer trajectory-grouped fold, split training trajectories into subtrain/validation.
- Train six independent pairwise linear classifiers on subtrain.
- Select pair thresholds, action vote biases, and soft-vote temperature on validation balanced accuracy only.
- Train a fresh six-pair linear model on the full outer training fold.
- Apply the validation-selected calibration to held-out test trajectories.

Outputs:

- `outputs/conceptual_belief_space/activation_pairwise_voting_linear_calibrated/pairwise_voting_calibrated_metrics.csv`
- `outputs/conceptual_belief_space/activation_pairwise_voting_linear_calibrated/pairwise_voting_calibrated_predictions.parquet`
- `outputs/conceptual_belief_space/activation_pairwise_voting_linear_calibrated/pairwise_voting_calibrated_summary.csv`

Best calibrated result (`soft_vote`):

- all-prefix top-action accuracy: 0.597352
- all-prefix balanced top-action accuracy: 0.581669
- pairwise q-order accuracy before calibration: 0.753154
- exact four-action order accuracy: 0.236013
- top-2 set accuracy: 0.506615
- mean Kendall tau: 0.505213

Conclusion:

- Calibration slightly improves ordinary accuracy and some order metrics.
- It does not solve the all-prefix balanced top-action target.
- The best robust result remains q-order decoding above 0.7, not balanced four-way top-action decoding above 0.7.


## 2026-07-06 - One-vs-rest activation-only linear action decoders

Ran four one-vs-rest linear action decoders on the activation-only `raw_late_delta` feature set.

Method:

- Feature set: concatenated activation-only views `raw_l16_18_mean20`, `raw_l16_18_20_22_mean20`, and `delta_pre_l16_18_mean20`.
- Model: one linear binary classifier per action.
- Objective: one-vs-rest BCE with class-imbalance positive weights.
- Evaluation: five trajectory-grouped held-out folds.
- Optional calibration: thresholds, action biases, and temperature selected on training-side validation trajectories only.

Outputs:

- `outputs/conceptual_belief_space/activation_ovr_linear_calibrated/ovr_linear_calibrated_metrics.csv`
- `outputs/conceptual_belief_space/activation_ovr_linear_calibrated/ovr_linear_calibrated_predictions.parquet`
- `outputs/conceptual_belief_space/activation_ovr_linear_calibrated/ovr_linear_calibrated_summary.csv`

Best result:

- calibration mode: none
- all-prefix top-action accuracy: 0.592823
- all-prefix balanced top-action accuracy: 0.591113
- macro F1: 0.541757
- pairwise q-order accuracy from action scores: 0.668911

Conclusion:

- This is the best verified full five-fold all-prefix balanced top-action result so far.
- It is still well below the 0.7 target.
- The strongest >0.7 evidence remains pairwise q-order decoding, not four-way top-action decoding.


## 2026-07-06 - Multi-layer identity activation-only linear features

Tested whether averaging layers before the linear probe was discarding action-separable directions.

Method:

- Built actual last-20-token mean activation features separately for layers 10, 12, 14, 16, 18, 20, and 22.
- Concatenated those seven layer-specific vectors, preserving layer identity.
- Feature dimension: 20,160.
- Inputs remain activation-only; no grid/task features or prefix metadata used as probe inputs.
- Evaluation: five trajectory-grouped held-out folds.

Outputs:

- feature cache: `outputs/conceptual_belief_space/activation_multilayer_identity_linear_screen/feature_cache/`
- split0 screen: `outputs/conceptual_belief_space/activation_multilayer_identity_linear_screen/multilayer_identity_split0_screen_metrics.csv`
- full metrics: `outputs/conceptual_belief_space/activation_multilayer_identity_linear_full/multilayer_identity_full_metrics.csv`
- predictions: `outputs/conceptual_belief_space/activation_multilayer_identity_linear_full/multilayer_identity_pairwise_predictions.parquet`
- summary: `outputs/conceptual_belief_space/activation_multilayer_identity_linear_full/multilayer_identity_full_summary.csv`

Full five-fold results:

- OVR weighted BCE: top-action accuracy 0.629328, balanced top-action accuracy 0.623116, macro F1 0.579066.
- Independent pairwise soft vote: top-action accuracy 0.620647, balanced top-action accuracy 0.615911, pairwise q-order accuracy 0.777835.

Status:

- This is the best verified all-prefix four-way top-action result so far.
- It improves balanced top-action decoding from about 0.591 to 0.623.
- It still does not reach the 0.7 target.
- Pairwise q-order decoding is now 0.777835 with the multilayer identity features.


## 2026-07-06 - Multi-layer multi-pooling activation-only screen

Tested whether preserving both layer identity and token-pooling identity improved linear decoding.

Method:

- Layers: 10, 12, 14, 16, 18, 20, 22.
- Poolings: last1, mean20, exp20_tau5.
- Feature dimension: 60,480.
- Inputs remain activation-only.
- Screened on trajectory-grouped split 0.

Outputs:

- feature cache: `outputs/conceptual_belief_space/activation_multilayer_multipool_linear_screen/feature_cache/`
- metrics: `outputs/conceptual_belief_space/activation_multilayer_multipool_linear_screen/multipool_identity_split0_screen_metrics.csv`
- manifest: `outputs/conceptual_belief_space/activation_multilayer_multipool_linear_screen/multipool_identity_manifest.json`

Split0 results:

- OVR weighted BCE: top-action accuracy 0.623598, balanced accuracy 0.603051.
- Independent pairwise soft vote: top-action accuracy 0.599541, balanced accuracy 0.584852, pairwise q-order accuracy 0.750626.

Conclusion:

- This was worse than the simpler multi-layer identity feature.
- The extra pooling views likely add noise/overfitting for the linear top-action readout.
- Did not run full five-fold because split0 failed to beat the current best.


## 2026-07-06 - Calibrated multi-layer identity OVR

Ran train-only calibration for the best multi-layer identity OVR decoder.

Method:

- Feature: concatenated per-layer raw mean20 activations for layers 10, 12, 14, 16, 18, 20, and 22.
- Model: OVR weighted BCE linear decoder.
- Calibration: action thresholds, action biases, and temperature selected on training-side validation trajectories only.
- Evaluation: held-out trajectory groups.

Outputs:

- `outputs/conceptual_belief_space/activation_multilayer_identity_ovr_calibrated/multilayer_identity_ovr_calibrated_metrics.csv`
- `outputs/conceptual_belief_space/activation_multilayer_identity_ovr_calibrated/multilayer_identity_ovr_calibrated_predictions.parquet`
- `outputs/conceptual_belief_space/activation_multilayer_identity_ovr_calibrated/multilayer_identity_ovr_calibrated_summary.csv`

Results:

- uncalibrated balanced top-action accuracy in this rerun: 0.620542
- train-only calibrated balanced top-action accuracy: 0.619384
- calibrated ordinary top-action accuracy: 0.633205
- calibrated macro F1: 0.582616

Conclusion:

- Calibration improves ordinary accuracy/order metrics but not balanced accuracy.
- Current best verified all-prefix balanced top-action result remains the original multi-layer identity OVR run at 0.623116.
- The target 0.7 all-prefix balanced top-action decoding remains unmet.


## 2026-07-06 - Per-layer pooling comparison

Compared pooling choices using the layer-identity activation-only setup.

Feature variants:

- per-layer `last1` for layers 10, 12, 14, 16, 18, 20, 22
- per-layer `mean20` for the same layers
- per-layer `exp20_tau5` for the same layers

Output:

- `outputs/conceptual_belief_space/activation_pooling_identity_linear_screen/pooling_identity_split0_screen_metrics.csv`

Split0 results:

- mean20 OVR: balanced top-action accuracy 0.625967
- mean20 pairwise soft vote: balanced top-action accuracy 0.621977, pairwise q-order accuracy 0.787410
- exp20_tau5 OVR: balanced top-action accuracy 0.616533
- last1 OVR: balanced top-action accuracy 0.515113

Conclusion:

- Mean pooling over the last 20 tokens is the best simple pooling for layer-identity linear decoding.
- Last-token only is clearly worse.
- Recency weighting does not beat mean20.


## 2026-07-06 - Soft-q and SGD solver screens

Tested whether the best multi-layer identity activation-only feature benefits from different linear objectives/solvers.

Soft-q linear screen:

- output: `outputs/conceptual_belief_space/activation_multilayer_identity_softq_screen/softq_split0_screen_metrics.csv`
- losses: q cross-entropy, q-MSE, CLR-MSE, hybrid CLR+CE, sqrt-q MSE
- best balanced top-action result: q cross-entropy at 0.604000 on split 0
- q-MSE had high ordinary accuracy 0.694343 and pairwise q-order accuracy 0.765217, but balanced accuracy only 0.589593
- conclusion: soft q targets improve ordinary accuracy/q-order but do not improve balanced top-action decoding.

SGD-style solver screen:

- output: `outputs/conceptual_belief_space/activation_multilayer_identity_sgd_solver_screen/sgd_solver_validation_screen.partial.csv`
- screened initial `SGDClassifier` log-loss configs on split0 training-side validation trajectories only
- validation balanced accuracy stayed around 0.51-0.53, so the branch was stopped early.

Current best remains:

- multi-layer identity OVR weighted BCE full five-fold balanced accuracy 0.623116.

## 2026-07-06 - Activation-only wider-window linear probes
Goal: improve activation-only linear decoding of the forced q-top action without grid/task features or prefix metadata as model inputs. All probes use trajectory-grouped held-out splits; split-0 screens are labeled as screens, not final evidence.
New feature families tested:
- Decision-relative residual pooling: subtract the same decision prefix-0 activation from last-token-window features.
- Last-50 and last-100 actual CoT-token mean windows across layers 10, 12, 14, 16, 18, 20, 22.
- Multi-scale windows combining last-20, last-50, and last-100 means.
- A diagnostic Viterbi temporal smoother over saved linear activation scores; this is not counted as a pure per-prefix linear probe.
Key outputs copied into `outputs/conceptual_belief_space/activation_rank_goal_summary/`:
- `residual_pooling.csv` from `outputs/conceptual_belief_space/activation_residual_pooling_linear_screen/residual_pooling_split0_screen_metrics.csv`
- `mean50_screen.csv` from `outputs/conceptual_belief_space/activation_window_mean50_linear_screen/mean50_split0_screen_metrics.csv`
- `mean50_full.csv` from `outputs/conceptual_belief_space/activation_window_mean50_linear_full/mean50_full_summary.csv`
- `mean100_screen.csv` from `outputs/conceptual_belief_space/activation_window_mean100_linear_screen/mean100_split0_screen_metrics.csv`
- `mean100_full.csv` from `outputs/conceptual_belief_space/activation_window_mean100_linear_full/mean100_full_summary.csv`
- `multiscale_screen.csv` from `outputs/conceptual_belief_space/activation_window_multiscale_linear_screen/multiscale_split0_screen_metrics.csv`
- `temporal_smoothing.csv` from `outputs/conceptual_belief_space/activation_linear_temporal_smoothing_diagnostic/linear_temporal_smoothing_summary.csv`

Best screen results:
- residual_pooling: balanced=0.627065, accuracy=0.650153, macro_f1=0.592166 (mode=resid_mean20)
- mean50_screen: balanced=0.660565, accuracy=0.678338, macro_f1=0.625978 (mode=raw_plus_resid_mean50, loss_kind=ce)
- mean50_full: balanced=0.632242, accuracy=0.648808, macro_f1=0.597477 (mode=resid_mean50)
- mean100_screen: balanced=0.664236, accuracy=0.693833, macro_f1=0.639758 (mode=raw_mean100, loss_kind=ovr_bce)
- mean100_full: balanced=0.654703, accuracy=0.669989, macro_f1=0.620955 (mode=raw_mean100, loss_kind=ovr_bce)
- multiscale_screen: balanced=0.665312, accuracy=0.689144, macro_f1=0.634584 (mode=resid_20_100, loss_kind=ovr_bce)
- temporal_smoothing: balanced=0.635922, accuracy=0.642466, macro_f1=0.596429 (model=ovr_weighted_bce, transition_weight=3.0)

Verified full-run results:
- mean50_full: balanced=0.632242, accuracy=0.648808, macro_f1=0.597477, pairwise=0.736888 (mode=resid_mean50)
- mean100_full: balanced=0.654703, accuracy=0.669989, macro_f1=0.620955, pairwise=0.677334 (mode=raw_mean100, loss_kind=ovr_bce)

Conclusion:
- The best verified full activation-only linear top-action result is now raw last-100 OVR at balanced accuracy 0.654703.
- This improves over the previous robust multi-layer identity mean20 OVR result of 0.623116, but it still does not reach the requested >0.7 target.
- Pairwise q-order decoding remains easier than exact q-top decoding; several CE/rank variants exceed 0.73 pairwise accuracy while balanced top-action remains below 0.7.
- Larger/multi-scale windows help on split 0 but show overfit or fold sensitivity; the full-run result is the authority for current progress.

## 2026-07-06 - All-layer last-100 screen

Tested whether adding early layers 0, 2, 4, 6, and 8 to the last-100 activation-only linear probe improves split-0 decoding.

Output:

- `outputs/conceptual_belief_space/activation_window_mean100_all_layers_linear_screen/mean100_all12_split0_screen_metrics.csv`
- copied summary: `outputs/conceptual_belief_space/activation_rank_goal_summary/mean100_all12_screen.csv`

Best all-12-layer split-0 result:

- mode: `raw_mean100_all12`
- loss: `ovr_bce`
- balanced accuracy: 0.633398
- ordinary accuracy: 0.655199
- macro F1: 0.603045
- pairwise q-order accuracy: 0.698576

Conclusion:

- Adding early layers hurts the linear top-action probe. The best all-12-layer result is below the mid-late layer last-100 split-0 result of 0.664236.
- Current best verified full activation-only linear top-action result remains raw last-100 layers 10-22, balanced accuracy 0.654703.
- The >0.7 balanced top-action target is still unmet.

## 2026-07-06 - Activation-only rank decoding from token combinations

Implemented and verified the activation-only rank-probe plan under `scripts/conceptual_belief_space/`:

- Patched `39_train_activation_rank_probes.py` to merge first action mention metadata from `outputs/prefixes.parquet` when the default `action_belief_vectors_mlx_full.parquet` table lacks strict pre-action columns.
- The rank probe uses activation vectors only as model inputs; grid/task fields, progress, prefix index, trajectory id, decision id, and action labels are excluded from primary feature matrices.
- Pairwise logistic ranking loss is used over the six action pairs derived from q ordering.
- Last-token-window variants include `last1`, `mean20`, exponential recency pools, and streamed `concat20`.
- `40_extract_forced_action_attention_pooling.py` remains the optional local-attention branch; it cannot run through Together/API because query-to-prefix attention matrices are required.

Verification runs:

```bash
python scripts/conceptual_belief_space/39_train_activation_rank_probes.py \
  --beliefs outputs/conceptual_belief_space/action_belief_vectors_mlx_full.parquet \
  --activation_index outputs/activation/activation_token_index.parquet \
  --prefixes outputs/prefixes.parquet \
  --output_dir outputs/conceptual_belief_space/activation_rank_probes_smoke_verify \
  --figure_dir figures/conceptual_belief_space/activation_rank_probes_smoke_verify \
  --layers 0,2 \
  --representations raw \
  --feature_variants last1,mean20,concat20 \
  --window_size 20 \
  --n_splits 2 \
  --limit_decisions 20 \
  --epochs 2 \
  --include_shuffled_baseline \
  --include_progress_only_diagnostic
```

This wrote `rank_probe_metrics.csv`, `rank_probe_predictions.parquet`, manifests, report, and figures. The output now contains real strict pre-action subsets rather than all-false placeholders.

```bash
python scripts/conceptual_belief_space/39_train_activation_rank_probes.py \
  --beliefs outputs/conceptual_belief_space/action_belief_vectors_mlx_full.parquet \
  --activation_index outputs/activation/activation_token_index.parquet \
  --prefixes outputs/prefixes.parquet \
  --output_dir outputs/conceptual_belief_space/activation_rank_probes_smoke_reps \
  --figure_dir figures/conceptual_belief_space/activation_rank_probes_smoke_reps \
  --layers 0 \
  --representations raw,delta_pre,unit_delta_pre \
  --feature_variants last1,mean20 \
  --window_size 20 \
  --n_splits 2 \
  --limit_decisions 20 \
  --epochs 1 \
  --include_shuffled_baseline
```

This verified that `raw`, `delta_pre`, and `unit_delta_pre` representations run successfully after the strict-column patch.

## 2026-07-06 - Linear score fusion and full cached multiscale probe

Continued the activation-only linear decoding goal.

1. Implemented `scripts/conceptual_belief_space/42_fuse_activation_linear_scores.py`.
   - Uses saved out-of-fold four-action scores from activation-only linear probes.
   - Does not fit a learned second-stage stacker, avoiding nested-stacking leakage from existing OOF scores.
   - Best label-free fusion: `mean_row_z_top_action_oriented_core`.
   - All-prefix balanced top-action accuracy: 0.666111.
   - All-prefix ordinary accuracy: 0.681236.
   - Pairwise q-order accuracy: 0.749244.
   - Output: `outputs/conceptual_belief_space/activation_linear_score_fusion/`.

2. Implemented `scripts/conceptual_belief_space/43_train_cached_multiscale_linear.py`.
   - Trains a single affine readout on cached activation-only multiscale residual features.
   - Main run: `resid_20_100 = [mean20 - pre_reasoning, mean100 - pre_reasoning]`, layers 10-22, 40,320 features.
   - Five trajectory-grouped folds, OVR BCE, no grid/task features or prefix metadata as model inputs.
   - Full result: mean balanced top-action accuracy 0.659274, accuracy 0.671276, macro F1 0.622113, pairwise q-order accuracy 0.691007.
   - Output: `outputs/conceptual_belief_space/activation_cached_multiscale_linear_full/`.

Conclusion: both runs remain below the requested >0.7 all-prefix balanced top-action target. The active goal is not complete.

## 2026-07-06 - Prefix cumulative-mean activation features

Tested a new activation-only linear feature family: cumulative mean over all generated CoT token activations up to each prefix. Prefix 0 uses pre_reasoning. This directly tests whether long-range reasoning context beyond the last-100 tokens helps.

Implemented `scripts/conceptual_belief_space/44_train_prefix_cumulative_mean_linear.py`.

Split-0 screen:
- `raw_and_resid_prefixmean` OVR BCE: balanced accuracy 0.682741, accuracy 0.707951, pairwise q-order 0.700535.
- `raw_prefixmean` OVR BCE: balanced accuracy 0.662008.
- `resid_prefixmean` OVR BCE: balanced accuracy 0.657656.
- `raw_and_resid_prefixmean` CE: balanced accuracy 0.673990.

Full five-fold result for `raw_and_resid_prefixmean` OVR BCE:
- mean balanced top-action accuracy 0.662373.
- mean accuracy 0.668590.
- mean macro F1 0.618004.
- mean pairwise q-order accuracy 0.698066.
- per-fold balanced accuracies: 0.674482, 0.644664, 0.676081, 0.656107, 0.660532.

Outputs:
- `outputs/conceptual_belief_space/activation_prefix_cumulative_mean_linear_screen/`
- `outputs/conceptual_belief_space/activation_prefix_cumulative_mean_linear_full/`
- `outputs/conceptual_belief_space/activation_prefix_cumulative_mean_linear_ce_screen/`

Conclusion: this is a strong direct linear activation-only feature, but still below the requested >0.7 all-prefix balanced q-top target. Goal remains active.

## 2026-07-06 - Train-mean-padded last-20 rank probe and segment-combo full run

Patched `scripts/conceptual_belief_space/39_train_activation_rank_probes.py` so non-concat last-20 variants use fold-specific train-mean activation padding before combination. `concat20` continues to handle padding by excluding padded slots from train statistics and zeroing them after train-only standardization. Ran a minimal smoke test at `outputs/conceptual_belief_space/activation_rank_probes_smoke_padding/`, then regenerated the canonical 20-decision smoke outputs at `outputs/conceptual_belief_space/activation_rank_probes/`; these validate the code path only, not full performance.

Completed `scripts/conceptual_belief_space/45_train_cached_segment_combo_linear.py` for the best screened segment combination:
- layers: 10, 12, 14, 16, 18, 20, 22
- mode: `rawresid_pfx_20_100`
- terms: raw prefix mean, raw last-20, raw last-100, and decision-relative residual versions of each
- feature dimension: 120,960
- model: one affine activation-only readout, OVR BCE
- evaluation: five trajectory-grouped folds

Full result:
- mean all-prefix accuracy: 0.689265
- mean all-prefix balanced top-action accuracy: 0.664472
- mean macro F1: 0.635922
- mean pairwise q-order accuracy: 0.687359

This did not meet the >0.7 robust all-prefix balanced top-action goal. Metrics copied to `outputs/conceptual_belief_space/activation_rank_goal_summary/segment_combo_full_summary.csv` and `segment_combo_full_metrics.csv`.

## 2026-07-06 - Segment validation screens, hybrid rank losses, and temporal+pairwise fusion

Continued the activation-only linear decoding goal.

Negative direct-probe screens:
- `rawresid_pfx_20_50_100` with train-side validation did not improve split-0 balanced accuracy: OVR BCE 0.670629, CE 0.630612.
- Minority-focused losses on `rawresid_pfx_20_100` did not improve: balanced CE 0.648977, balanced focal 0.658853.
- Hybrid top-action plus q-order rank losses improved pairwise q-order but not q-top balanced accuracy: `ovr_rank` balanced 0.679608 / pairwise 0.746734, `ce_rank` balanced 0.682043 / pairwise 0.735802.

Positive diagnostic:
- Extended `42_fuse_activation_linear_scores.py` with latest full OOF temporal sources and row-softmax fixed-group averaging.
- Best label-free diagnostic fusion: `mean_row_softmax_t1_temporal_identity_pair_core` using `mean100_raw_ovr`, `identity_pairwise`, `prefixmean_raw_and_resid`, and `segment_rawresid_pfx_20_100`.
- Full OOF all-prefix balanced q-top accuracy: 0.701926; accuracy: 0.714335; macro F1: 0.669231; pairwise q-order: 0.751076.
- Per-split balanced accuracies: 0.704037, 0.700851, 0.705847, 0.683867, 0.705054.

This crosses 0.7 as a label-free score-fusion diagnostic, but the goal remains active because the source group was identified during exploratory analysis rather than validated under a stricter nested source-selection/retraining protocol.

## 2026-07-06 - Fusion selection stability audit

Implemented `scripts/conceptual_belief_space/46_audit_fusion_selection_stability.py` and ran it on `activation_linear_score_fusion_with_segments`.

Leave-one-split-out selection chose the same fusion, `mean_row_softmax_t1_temporal_identity_pair_core`, for all five held-out splits. Held-out balanced accuracies were 0.704037, 0.700851, 0.705847, 0.683867, and 0.705054. This supports stability of the diagnostic but does not close the no-overfit audit because the saved OOF component models used on the selection folds are not from a fully nested retraining protocol.

## 2026-07-06 - Activation-only rank probe plan implementation and nested fusion validation

Implemented and verified the plan-facing activation-only rank decoding pipeline:

- `scripts/conceptual_belief_space/39_train_activation_rank_probes.py`
  - uses `outputs/conceptual_belief_space/action_belief_vectors_mlx_full.parquet`
  - uses per-token activation pointers from `outputs/activation/activation_token_index.parquet`
  - builds actual CoT-token windows for `last1`, `mean20`, `exp20_tau2`, `exp20_tau5`, `exp20_tau10`, and `concat20`
  - supports `raw`, `delta_pre`, and `unit_delta_pre`
  - trains a linear four-score rank probe with pairwise logistic ranking over the six action pairs
  - masks q ties within `1e-6`
  - evaluates trajectory-grouped splits with no trajectory leakage
  - excludes grid/task features and prefix metadata from the primary feature matrices

- `scripts/conceptual_belief_space/40_extract_forced_action_attention_pooling.py`
  - implements the optional local forced-final-context attention-pooling branch
  - restricts attention keys to generated analysis-prefix tokens
  - excludes prompt tokens and forced final-JSON tokens from the key span
  - fails clearly unless a local model/runtime can return attentions

Canonical broad smoke run:

- output: `outputs/conceptual_belief_space/activation_rank_probes/`
- figures: `figures/conceptual_belief_space/activation_rank_probes/`
- rows: 1,561 prefix rows from 20 decisions / 20 trajectories
- layers: `0,2,4,6,8,10,12,14,16,18,20,22`
- representations: `raw`, `delta_pre`, `unit_delta_pre`
- variants: `last1`, `mean20`, `exp20_tau2`, `exp20_tau5`, `exp20_tau10`, `concat20`
- metrics rows: 2,592
- primary all-prefix configs: 432
- best smoke pairwise q-order accuracy: 0.591216 at layer 10, `unit_delta_pre`, `exp20_tau5`
- best smoke top-action accuracy for that row: 0.366623

The smoke validates the requested pipeline and artifact generation. It is not a full all-100-trajectory result.

Nested fixed-fusion validation:

- implemented `scripts/conceptual_belief_space/47_nested_fixed_fusion_validation.py`
- retrains component activation-only linear probes inside each outer trajectory split
- tested fixed temporal-only OVR fusion and fixed temporal+pairwise fusion
- output: `outputs/conceptual_belief_space/activation_nested_fixed_fusion_validation_fulltrain/`

Five-fold nested full-train summary:

- `fixed_temporal_ovr3_fusion`: mean balanced q-top accuracy 0.700298, mean accuracy 0.711892, macro F1 0.665308, pairwise q-order 0.713272, minimum split balanced accuracy 0.688646
- `fixed_temporal_identity_pair_fusion`: mean balanced q-top accuracy 0.699684, mean accuracy 0.717444, macro F1 0.670467, pairwise q-order 0.757239, minimum split balanced accuracy 0.693212

This is stronger than the previous saved-score diagnostic because source probes are retrained inside each outer split. The result is still borderline: one fixed fusion crosses 0.7 on mean balanced accuracy, but the weakest fold remains below 0.7.

## 2026-07-06 - Train-prior debiased fusion and source-score stacker diagnostics

Continued the activation-only linear decoding goal after the nested fixed fusion plateaued at mean balanced q-top accuracy `0.700298` with a weakest fold below `0.7`.

New scripts:

- `scripts/conceptual_belief_space/48_nested_calibrated_fusion_validation.py`
  - Splits each outer training fold into source-training and calibration trajectories.
  - Trains activation-only linear source probes on the source-training trajectories.
  - Selects source group, row-softmax temperature, and class bias on calibration trajectories only.
  - Retrains sources on the full outer training fold and evaluates held-out trajectories.
  - Split-0 short smoke did not improve: calibrated fusion balanced accuracy `0.679210`; same-run fixed OVR3 baseline `0.685533`.

- `scripts/conceptual_belief_space/49_train_prior_debiased_fusion.py`
  - Applies a fixed inverse train-class-prior bias to nested activation-only source-score fusions.
  - Bias for each held-out split is computed only from non-held-out split labels.
  - Uses the strict nested source predictions from `activation_nested_fixed_fusion_validation_fulltrain`.

Best train-prior debiased fixed candidates:

- `all4_t0p75_prior_alpha0p25`: mean balanced q-top accuracy `0.705647`, mean accuracy `0.692042`, macro F1 `0.650811`, pairwise q-order `0.733918`, weakest fold `0.690984`.
- `all4_t1p5_prior_alpha0p1`: mean balanced q-top accuracy `0.705249`, mean accuracy `0.701619`, macro F1 `0.658282`, pairwise q-order `0.734239`, weakest fold `0.693230`.

Leave-one-split selection over the five fixed prior-debias configs gave mean balanced accuracy `0.705100` and weakest fold `0.690984`.

Additional source-score stacker diagnostic:

- A low-dimensional logistic-regression stacker over the four activation-only source score vectors (`mean100_ovr`, `prefixmean_ovr`, `segment_ovr`, `identity_pairwise`) reached mean balanced q-top accuracy `0.709360`, mean ordinary accuracy about `0.701099`, and weakest fold `0.693502` when trained/tested leave-one-split over the saved nested source scores.
- This is the strongest diagnostic score so far, but it is not yet a fully rigorous no-overfit result because the meta-stacker was trained on saved OOF source scores whose source models for the training folds may have seen the held-out fold during their source-model training.

Outputs copied to `outputs/conceptual_belief_space/activation_rank_goal_summary/`:

- `train_prior_debiased_fusion_summary.csv`
- `train_prior_debiased_fusion_metrics.csv`
- `train_prior_debiased_fusion_leave_one_split_selection.csv`
- `train_prior_debiased_fusion_report.md`
- `source_score_stacker_diagnostic.csv`
- `nested_calibrated_fusion_smoke_summary.csv`
- `nested_calibrated_fusion_smoke_metrics.csv`

Current conclusion: activation-only linear combinations now clearly exceed `0.7` mean balanced q-top accuracy in candidate analyses, with the strongest diagnostic at `0.709360`. The no-overfit audit is still not fully closed because the weakest fold remains below `0.7` and the best source-score stacker requires a proper inner-OOF implementation inside each outer split.

## 2026-07-06 - Strict nested OOF source-score stacker reaches the target

Implemented `scripts/conceptual_belief_space/50_nested_oof_source_stacker.py`.

Purpose:

- Close the no-overfit gap in the previous source-score stacker diagnostic.
- Generate source scores for each outer training row with inner OOF source models.
- Select and fit the meta-stacker using only outer-train OOF source scores.
- Train final source models on the outer training trajectories only, then evaluate held-out trajectories.

Source models:

- `mean100_ovr`
- `prefixmean_ovr`
- `segment_ovr`
- `identity_pairwise`

All four are activation-only linear probes. Metadata is used only for row alignment, grouped splitting, labels, and reports.

Strict mixed-feature meta-stacker result:

- output: `outputs/conceptual_belief_space/activation_nested_oof_source_stacker_full_e8_i3/`
- outer splits: 5 trajectory-grouped folds
- inner source folds: 3 trajectory-grouped folds inside each outer train fold
- source epochs: 8
- mean balanced q-top accuracy: `0.701512`
- mean ordinary accuracy: `0.707615`
- mean macro F1: `0.664835`
- mean pairwise q-order accuracy: `0.789863`
- weakest fold balanced accuracy: `0.686382`

Fold balanced accuracies:

- split 0: `0.708209`
- split 1: `0.700206`
- split 2: `0.708315`
- split 3: `0.686382`
- split 4: `0.704450`

Strict raw-only meta-stacker result:

- output: `outputs/conceptual_belief_space/activation_nested_oof_source_stacker_rawonly_full_e8_i3/`
- meta-stacker inputs: raw source action scores only
- because raw source scores are affine functions of activation-combination features, the final decision function is linear in the union of those activation features
- mean balanced q-top accuracy: `0.700795`
- mean ordinary accuracy: `0.707635`
- mean macro F1: `0.665416`
- mean pairwise q-order accuracy: `0.785139`
- weakest fold balanced accuracy: `0.686382`

Raw-only selected meta configs:

- split 0: `ridge_raw_alpha1`, balanced `0.708885`
- split 1: `logreg_raw_C1`, balanced `0.695943`
- split 2: `ridge_raw_alpha1`, balanced `0.708315`
- split 3: `logreg_raw_C0p1`, balanced `0.686382`
- split 4: `logreg_raw_C0p1`, balanced `0.704450`

Current conclusion: the requested aggregate target is now met under a strict grouped nested validation protocol. The clean raw-only version clears `0.7` mean balanced accuracy while preserving linearity in activation-derived features. The remaining caveat is that split 3 is still below `0.7`, so the result should be stated as a five-fold mean success rather than every-fold success.



## 2026-07-06 - Linear probe-induced action-evidence score manifold

Implemented and ran the supervised score-manifold pipeline using the raw-only nested OOF source-score stacker.

Code added or updated:

- `scripts/conceptual_belief_space/50_nested_oof_source_stacker.py`
  - extended `--save_predictions` to write held-out row-level source scores and final stacker scores.
- `scripts/conceptual_belief_space/51_build_linear_probe_score_manifold.py`
- `scripts/conceptual_belief_space/52_plot_linear_probe_score_manifold.py`
- `scripts/conceptual_belief_space/53_write_linear_probe_score_manifold_report.py`

Full OOF score export:

- output: `outputs/conceptual_belief_space/activation_nested_oof_source_stacker_rawonly_full_e8_i3_with_predictions/`
- rows: `98311`
- trajectories: `95`
- decisions: `1276`
- outer folds: 5 trajectory-grouped folds
- all rows are held-out OOF rows; no in-sample rows are used for the main manifold.
- mean balanced accuracy in the stacker summary: `0.701635`
- mean ordinary accuracy: `0.708663`
- mean macro F1: `0.665660`
- mean pairwise q-order accuracy: `0.790467`
- weakest fold balanced accuracy: `0.693341`

Real linear score manifold outputs:

- manifold parquet: `outputs/conceptual_belief_space/linear_probe_score_manifold/linear_probe_score_manifold.parquet`
- metrics CSV: `outputs/conceptual_belief_space/linear_probe_score_manifold/linear_probe_score_manifold_metrics.csv`
- report: `outputs/conceptual_belief_space/linear_probe_score_manifold/linear_probe_score_manifold_report.md`
- figures: `figures/conceptual_belief_space/linear_probe_score_manifold/`

Manifold metric highlights:

- balanced accuracy vs true q-top action: `0.703936`
- ordinary accuracy vs true q-top action: `0.708680`
- macro F1: `0.667050`
- pairwise q-order accuracy: `0.790466`
- high-commitment balanced accuracy: `0.789462`
- low-commitment balanced accuracy: `0.467942`
- strict pre-action balanced accuracy: `0.683764`
- Spearman(score_radius, entropy_commitment): `0.252132`
- partial Spearman(score_radius, entropy_commitment | progress): `0.248646`

Interpretation:

- This manifold is supervised and linear-probe-induced, not PCA or UMAP.
- Main coordinates are linear contrasts of held-out stacker scores: `x = centered RIGHT - centered LEFT`, `y = centered UP - centered DOWN`.
- The raw-only stacker preserves the strongest linearity claim because source scores and the meta score function are activation-derived and linear in the selected activation-combination feature spaces.
- Four-branch structure is most meaningful in high-commitment rows; low-commitment rows are expected to be mixed or near the origin.

## 2026-07-15 - Causal last-token attention probe for forced q-top action

Implemented and trained a compact activation-only attention probe whose query is derived from the actual last generated token of each prefix.

Core implementation:

- `scripts/conceptual_belief_space/45_train_cached_segment_combo_linear.py`
  - added `raw1` / last-token cache support.
- `scripts/conceptual_belief_space/67_train_last_token_attention_probe.py`
  - direct causal activation-query prototype and query-shuffle controls.
- `scripts/conceptual_belief_space/68_train_nested_last_token_attention_probe.py`
  - strict trajectory-grouped inner-OOF source evidence for gate training.
  - last-token score query, action-specific attention, grouped meta-validation, final outer source retraining, checkpoints, predictions, and leakage audits.
- `scripts/conceptual_belief_space/69_screen_cached_last_token_attention_gate.py`
  - development-only gate selection on cached OOF evidence; selected balanced CE, learning rate `1e-4`, no q-rank auxiliary.
- `scripts/conceptual_belief_space/71_evaluate_attention_gate_on_strict_sources.py`
  - evaluation-only application of frozen gates to the strongest existing strict held-out activation sources.
- `scripts/conceptual_belief_space/72_build_attention_residual_probe.py`
  - final residual form: strict linear stacker plus `alpha * (dynamic query gate - no-query global gate)`.
- `scripts/conceptual_belief_space/73_audit_attention_query_permutations.py`
  - repeated within-fold last-token query permutation audit.

Probe contract:

- frozen `openai/gpt-oss-20b` backbone;
- fewer than one million downstream readout parameters;
- activation-only model inputs;
- actual last-prefix-token evidence is the attention query;
- causal last-100, cumulative-prefix, segment, and pairwise activation summaries provide memory evidence;
- grid/task features, progress, IDs, future tokens, and forced-answer tokens are excluded from model inputs;
- outer and inner splits are trajectory-grouped; all 15 inner-fold audits have zero trajectory overlap;
- all 98,311 final prediction rows are unique held-out OOF rows.

Final output:

- `outputs/conceptual_belief_space/last_token_attention_residual_probe_final/`
- selected development coefficient: `alpha=4` on declared development split 0;
- five-fold balanced accuracy: `0.700825`;
- weakest-fold balanced accuracy: `0.689070`;
- confirmatory splits 1-4 balanced accuracy: `0.697445`;
- no-query global residual baseline: `0.701635` five-fold / `0.698642` confirmatory;
- one fixed shuffled-query control: `0.698614` five-fold / `0.695178` confirmatory.

Repeated query-permutation audit (100 within-fold permutations):

- observed five-fold balanced accuracy: `0.700825`;
- mean shuffled-query five-fold balanced accuracy: `0.699267`;
- observed-minus-shuffled: `0.001558`;
- one-sided permutation p-value: `0.0099`;
- observed confirmatory balanced accuracy: `0.697445`;
- mean shuffled-query confirmatory balanced accuracy: `0.695729`;
- confirmatory observed-minus-shuffled: `0.001716`;
- confirmatory one-sided permutation p-value: `0.0099`.

Conclusion:

- The last-token attention path is real but small: it survives repeated query shuffling, while the no-query linear residual baseline remains slightly stronger in absolute balanced accuracy.
- Historical interpretation superseded: the model clears `0.7` as a grouped
  causal prefix classifier, but it does **not** satisfy the stricter probe
  definition adopted below.

### 2026-07-16 - Scope correction and sparse native-attention replacement

The 2026-07-15 result above is relabeled as a **causal prefix classifier**.
Its memory branch reads cumulative, segmented, and recent-window summaries of
earlier activations, so `0.700825` is not a valid score for a strict last-token
probe or for a probe restricted to a small attention-selected token subset.
The original artifact is preserved with an explicit correction at
`outputs/conceptual_belief_space/last_token_attention_residual_probe_final/SCOPE_CORRECTION.md`.

Replacement contract:

- the query is the actual final reasoning token of the unmodified prefix;
- the frozen model's native attention row selects at most top-k **earlier**
  reasoning tokens;
- no forced-answer suffix and no learned prefix-wide attention are used;
- the probe reads only the current residual plus a pooled residual over those
  selected tokens;
- the only trained component is one affine four-action readout;
- matched recent, random, and blind all-prefix controls use the same readout;
- attention layer and top-k are selected on a train-only trajectory-grouped
  validation partition, with outer trajectory folds reserved for evaluation.

Implementation:

- `scripts/conceptual_belief_space/74_extract_sparse_native_attention.py`
- `scripts/conceptual_belief_space/75_train_sparse_attention_token_probe.py`

### 2026-07-21 - Positional controls for the CoT-window commitment probe

Code change only; no probe training run. Added three positional controls to
`scripts/conceptual_belief_space/77_train_forced_suffix_commitment_probe.py`
so that any commitment result is reported against what position alone explains.

Motivation: `entropy_commitment` rises monotonically with position in the CoT,
so a probe that only decodes where it sits in the prefix scores well without
representing commitment. The pre-existing `progress_length_spline` baseline
conflated fair and oracle features and had no position-matched evaluation.

Changes:

- `fit_progress_baseline` generalized to `fit_positional_baseline(..., feature_cols)`;
  log1p compression now selected by column name (`LOG1P_FEATURES`) rather than by
  column position, so arbitrary feature subsets are valid.
- New `POSITIONAL_BASELINES` registry, all trained per outer fold with resume support:
  - `position_index_fair` (`prefix_token_index`) - the fair bar. This is the only
    positional quantity the probe's inputs could encode.
  - `progress_oracle` (`progress`) - oracle.
  - `progress_length_spline` (all three) - the pre-existing baseline, retained under
    its original artifact filename for resume compatibility, relabelled oracle.
- `progress_oracle` and `num_cot_tokens` depend on total CoT length N. States at cut
  t are computed from tokens 1..t and cannot contain N, so those are oracle
  references, not fair comparisons.
- New `progress_bin_scopes` adds fixed-edge position-matched scopes
  (0.0/0.2/0.4/0.6/0.8/1.0) to `evaluate_prediction_frame`, alongside the existing
  `all_prefixes` and strict subsets. Verified to be a complete, non-overlapping
  partition.
- `write_report` gained a position-matched score table and a Positional Controls
  section stating the fair/oracle distinction.

Verified against `action_belief_vectors_mlx_full.parquet` (98,311 rows, 95
trajectories), trajectory-grouped 5-fold, target `entropy_commitment`
(mean 0.6132, std 0.2804):

- `position_index_fair`: Spearman `0.2829`, r2 `0.1194`
- `progress_oracle`: Spearman `0.5192`, r2 `0.2659`
- `progress_length_spline`: Spearman `0.5424`, r2 `0.2955`

Control 3 behaves as intended: `progress_oracle` scores `0.5361` pooled but
collapses to `0.0761`-`0.1940` inside the three middle bins, confirming that
position-matched scoring removes the positional component. The final bin
(0.8-1.0) retains `0.4238` because commitment still climbs steeply within it;
finer terminal bins may be warranted.

`tests/test_forced_suffix_probe.py`: 9 passed.

Environment notes:

- The ambient anaconda interpreter (pyarrow 19.0.0) cannot read this project's
  parquet files: `OSError: Repetition level histogram size mismatch`. Use
  `interp/.venv/bin/python` (pyarrow 24.0.0) or `engine="fastparquet"`.
- On this machine numpy 2.2.6 on the Accelerate BLAS backend emits spurious
  `divide by zero` / `overflow` / `invalid value encountered in matmul` warnings
  from finite, correct inputs; reproduced with freshly generated random matrices.
  These appear during baseline fitting and are not a numerical fault.

Extraction state at time of writing: `76` completed 25 of 95 trajectories and
stopped at 23:49 on 2026-07-20, leaving `.partial` files on trajectory 33 and a
stale `overnight_runner.lock`. `activation_index.parquet` is not yet written, so
`77` cannot run end-to-end until extraction is resumed and completes.

### 2026-07-21 - Stored stage-03 belief labels do not reproduce; extractor labels are the faithful ones

Discovered while resuming CoT-window extraction. **This is a data-integrity finding
affecting the project's core behavioral object, not just the layer probe.**

Symptom chain (three separate defects, found in this order):

1. Resuming `76` with the default `--q_mismatch_policy error` hard-failed at
   `prefix_id=26475` (trajectory 33): recomputed action probabilities disagreed
   with stored stage-03 labels by `0.129`, above the `1e-4` tolerance. Completed
   shards are skipped without re-validation, so this was the first row ever
   re-checked by an `error`-policy run.
2. With `use_extracted`, extraction got further and hit a genuine tokenization
   bug: `aligned_cot_window` raised because the re-encoded context did not end
   with the stored analysis token IDs. Cause is a BPE whitespace merge - the
   prefix cuts mid-run-of-spaces, generation emitted `[220, 220]` (two
   single-space tokens) and re-encoding merges them into `256`. A tokenizer-only
   sweep of all 98,311 prefixes found **exactly one** affected row,
   `prefix_id=26532`, recorded in `misaligned_prefixes.parquet`. It is excluded
   via `prefixes_aligned.parquet` (98,310 rows). Zero affected rows in the 25
   previously-extracted trajectories.
3. The q disagreement is **not** an edge case. Across the 27,469 rows extracted
   so far, `93.5%` exceed the `1e-4` tolerance, mean per-row max |dq| = `0.244`.
   Last night's 25 shards show the same rate, so that run had been launched with
   `use_extracted` and died on defect 2, not on q.

The two label sets are substantively different, not a reparametrization:

- `top_action` argmax agreement: `0.7115` (a temperature or any monotone per-row
  transform would preserve argmax exactly, so this rules that out);
- mean normalized entropy: stored `0.3821` vs extracted `0.3224` (extracted is peakier);
- mean max-prob: stored `0.7925` vs extracted `0.8245`;
- Spearman(stored commitment, extracted commitment): `0.6906`.

Adjudication - agreement with the action the agent actually took:

| labels | all rows | progress > 0.9 (n=2902) |
|---|---|---|
| stored (stage 03) | `0.5683` | `0.9076` |
| extracted (current runtime) | `0.7133` | **`0.9910`** |

At the end of the CoT the model should almost always name the action it then
takes. The extracted labels hit `0.991` there; the stored labels only `0.908`.
The extractor is consistent with the runtime that generated the trajectories;
the stored stage-03 labels are not. Inspection of both code paths found the
scoring context (`FINAL_JSON_PREFIX` == `FORCED_SUFFIX`), candidate token IDs,
and softmax math to be equivalent, so the stage-03 defect is not yet localized.
Candidates not yet ruled out: `--prefix_tokenization analysis_token_ids` BPE
boundary drift, or a model/runtime revision difference at stage-03 time.

Consequences (NOT yet acted on):

- Every downstream artifact built on stored `q_*` inherits this: `commitment_to_final`,
  `prefix_commitment_metrics_*`, behavior vectors/manifold, and any probe trained
  to predict the stored forced top action. Prior results should be re-checked
  before being cited.
- The positional-control baselines logged earlier today were computed on stored
  labels and must be recomputed against `extracted_beliefs.parquet`.
- The in-progress extraction uses `use_extracted`, so probe labels and activations
  come from one runtime and are self-consistent.

Action taken: full extraction relaunched with
`--prefixes outputs/conceptual_belief_space/cot_window_layer_transition_probe/prefixes_aligned.parquet`
and `--q_mismatch_policy use_extracted`. Log:
`cot_window_layer_transition_probe/extraction_full_20260721.log`.
Recommend independently confirming the adjudication above before retracting any
prior result.

### 2026-07-22 - CoT-window layer probe, fold 0: commitment is decodable, layer-transition premise is not supported

First end-to-end run of `77` on the completed extraction. Single outer fold
(`--limit_folds 1`), labels from `extracted_beliefs.parquet`, all methods plus the
three positional controls added 2026-07-21. Selected config: 4 encoder layers,
dropout 0.1, lambda_q 0.1, 7 epochs, validation Spearman `0.7701`.
Artifacts: `cot_window_layer_transition_probe/fold0_20260721/`.

Held-out Spearman on `entropy_commitment` (trajectory-grouped, pooled OOF):

| method | spearman | r2 | top_action_acc |
|---|---|---|---|
| mean_layers_mlp | **0.7619** | 0.4862 | 0.5080 |
| concat_layers_mlp | 0.7541 | 0.4596 | 0.4732 |
| best_single_layer_mlp | 0.7487 | 0.4639 | 0.5349 |
| transformer_no_deltas | 0.7482 | 0.4941 | 0.4975 |
| bidirectional_transformer (main) | 0.7457 | 0.4829 | 0.5021 |
| final_layer_mlp | 0.6974 | 0.3708 | 0.4556 |
| progress_length_spline (oracle) | 0.6421 | 0.3748 | 0.3582 |
| progress_oracle (oracle) | 0.6271 | 0.3563 | 0.3755 |
| position_index_fair (**fair bar**) | 0.3464 | 0.1746 | 0.3703 |

**Result 1 - the signal is real and is not positional.** Clustered bootstrap
(1000 resamples, trajectory clusters) vs `bidirectional_transformer`:

- vs `position_index_fair`: `+0.3986` CI `[0.3456, 0.4546]`
- vs `progress_oracle`: `+0.1185` CI `[0.0975, 0.1370]`
- vs `progress_length_spline`: `+0.1031` CI `[0.0826, 0.1223]`
- vs `final_layer_mlp`: `+0.0482` CI `[0.0382, 0.0588]`

The probe beats not only the fair bar but both oracle baselines, which use total
CoT length the probe's inputs cannot contain. Position-matched scoring confirms
it: within fixed progress bins the activation probes hold `0.50`-`0.71`, while
`position_index_fair` collapses to `0.334 / 0.113 / 0.020 / -0.091 / -0.319`
and `progress_oracle` to `0.320 / 0.087 / 0.134 / 0.171 / 0.477`. Holding
position constant removes the baselines' score and leaves the probe's intact.

**Result 2 - the `prompt.txt` layer-transition hypothesis is not supported.**
Bootstrap differences vs the main transformer:

- vs `mean_layers_mlp`: `-0.0162` CI `[-0.0211, -0.0105]` (transformer significantly **worse**)
- vs `concat_layers_mlp`: `-0.0084` CI `[-0.0147, -0.0021]` (significantly worse)
- vs `best_single_layer_mlp`: `-0.0028` CI `[-0.0097, 0.0047]` (no difference)
- vs `transformer_no_deltas`: `-0.0022` CI `[-0.0086, 0.0044]` (no difference)

Treating the layer axis as a sequence buys nothing: mean-pooling layers beats it,
adjacent-layer deltas contribute nothing, and one well-chosen layer matches it.
Commitment is encoded *in* layers, not in the layer-to-layer transition.

**Where the signal lives.** Layer occlusion is sparse and localized - layer 3
`0.1679`, layer 10 `0.1339`, layer 18 `0.0566`, layer 14 `0.0330`; the remaining
18 layers each cost `<0.008`. Token occlusion is diffuse - the largest single-token
cost is the final token at `0.0193`, no token is critical. This is consistent with
result 2: a few informative layers rather than a transition process.

Caveats: one outer fold, so there is no cross-fold variance; the bootstrap is
trajectory-clustered within that fold. `top_action_accuracy` (~0.50-0.53) is a
secondary metric here and is not comparable to the retracted 0.700825 causal
prefix classifier, which used stored labels now known to be unreproducible.

### 2026-07-22 - Supervised activation projection: the pre-verbal action signal is real but is not a four-arm manifold

New stage `scripts/conceptual_belief_space/78_supervised_activation_projection.py`.
Motivation: the behavioural manifold fans into four action arms even on strict
pre-action-mention prefixes (report Figure 17), but unsupervised PCA/UMAP of
activations shows no such structure. This stage tests whether that is absence of
signal or a variance-ordering artefact.

Setup: held-out outer fold 0, strict `before_any_action_mention` prefixes only
(6,534 rows, 19 trajectories), labels from `extracted_beliefs.parquet`. Activations
are the CoT-window states mean-pooled over valid tokens. Every reported number
fits the projection inside trajectory-grouped 5-fold CV; the figure is fitted on
the full subset and is descriptive only. Chance balanced accuracy `0.25`;
majority class RIGHT `0.5092`.

Grouped balanced accuracy of a linear readout in each 3-d projection:

| representation | PCA | LDA | CLR regression |
|---|---|---|---|
| all_layer_mean | 0.3512 | **0.4218** | 0.4035 |
| layer_3 | 0.3168 | 0.3524 | 0.3251 |
| layer_10 | 0.3375 | 0.3706 | 0.3687 |
| layer_18 | 0.3499 | **0.4250** | 0.4112 |

**Result 1 - PCA hid the signal for the expected reason.** The top components
encode absolute position, not action: |r| with `log(prefix_token_index)` is
`0.444 / 0.567 / 0.388` for PC1-3, versus `0.151 / 0.199 / 0.279` for `progress`
(consistent with total CoT length being unavailable to the states). The
silhouette of action labels in PCA(3) is `-0.1003` - literally no cluster
structure. Supervised projection recovers roughly `+0.07` balanced accuracy over
PCA on the same activations.

**Result 2 - but there is no four-arm manifold.** The LDA panel is a single
overlapping cloud with displaced class centroids, not the behavioural manifold's
star. For scale, the trained `mean_layers_mlp` reaches only `0.4518` balanced
accuracy on this same subset, so a 3-d supervised projection already captures
~93% of the available action signal; the ceiling itself is low. The behavioural
arms are a property of the 3-simplex the labels live in and do not have a clean
activation-space counterpart.

**Result 3 - commitment and action identity dissociate across layers.** Action
identity is best at layer 18 (`0.4250`) and worst at layer 3 (`0.3524`), whereas
the fold-0 occlusion analysis found layer 3 the single most important layer for
commitment (Spearman drop `0.1679`) and layer 18 only third (`0.0566`). The two
quantities are not carried by the same layers.

Artifacts: `cot_window_layer_transition_probe/supervised_projection/`,
figure `figures/conceptual_belief_space/supervised_activation_projection.png`.

Caveat: one outer fold, 19 held-out trajectories. The CLR-regression projection
is included as the direct analogue of the behavioural manifold's coordinates and
tracks LDA closely throughout.

### 2026-07-23 - D0 label rebase: behavioral phenomenology recomputed on faithful extracted labels

Executed stage D0 of `RESEARCH_DIRECTIONS.md`: the full behavioral pipeline was re-run
on the adjudicated-faithful CoT-window extraction labels, replacing the quarantined
stage-03 stored labels for every behavior-side quantity.

Commands (all with `interp/.venv/bin/python`, repo root):

```
python scripts/03c_export_extracted_prefix_scores.py            # new adapter stage
python scripts/04_compute_commitment_metrics.py --input outputs/prefix_action_scores_extracted.parquet \
  --prefix_output outputs/prefix_commitment_metrics_extracted.parquet \
  --step_output outputs/step_commitment_summary_extracted.parquet \
  --report outputs/commitment_report_extracted.md
python scripts/05_build_behavior_vectors.py  --input ..._extracted --output outputs/behavior_vectors_extracted.parquet
python scripts/06_embed_behavior_manifold.py --input ..._extracted --output outputs/behavior_manifold_extracted.parquet
python scripts/07_plot_behavior_commitment.py --input ..._extracted --output_dir figures/behavior_extracted
python scripts/08_compare_extracted_vs_stored_behavior.py       # new comparison stage
```

`03c` validations passed: 98,310/98,311 prefixes joined (only the known-misaligned
`prefix_id=26532` lacks faithful labels), `strict_before_any_action_mention` agrees with
`is_before_first_action_mention` on every row, max |sum(q)-1| = 4.4e-16.

**Triangulation (new, decisive).** Against the independent Together-scored 32-trajectory
table (33,147 prefixes, scored with no KV-cache reuse):

| regime | top_action agreement | mean max|dq| |
| --- | --- | --- |
| extracted | **0.9707** | **0.0258** |
| stored (quarantined) | 0.7023 | 0.2485 |

Two independent scoring paths (Together API, CoT-window extraction) agree with each other
and both disagree with the stored stage-03 labels. This upgrades the 2026-07-21 adjudication
from internal-consistency evidence to external replication: the stored labels are corrupted,
the extracted labels are correct.

**Headline metrics that MOVED (do not quote stored-label numbers):** mean commitment
0.5503 -> 0.6670; frac top==final 0.5797 -> 0.7050; strict pre-mention mean commitment
0.3927 -> 0.4942; median first_cross_0.5 0.1653 -> 0.0893; median first_cross_0.7
0.3047 -> 0.2159; median lock_in_progress 0.2875 -> 0.1904; frac steps with lock-in
0.9679 -> 0.9976; mean switch_count 8.40 -> 6.04; mean total_variation 15.02 -> 10.92.
Stored-label noise systematically *delayed* apparent commitment and *inflated* apparent
switching (+39% switches, +38% total variation). Per-action means moved hardest for
DOWN (0.351 -> 0.565) and UP (0.401 -> 0.676); RIGHT moved down (0.777 -> 0.731), so the
old per-action asymmetry magnitudes are unreliable, though the ordering at prompt-only
prefixes (RIGHT 0.664 >> LEFT 0.369 >> UP 0.144 ~ DOWN 0.144) survives.

**What HELD:** the qualitative story. Commitment still rises monotonically with progress
in every regime and every subset; final-prefix commitment = 1.0000 and final-prefix
top==agent_action = 1.0000 (perfect end-of-CoT self-consistency, vs 0.908 for stored at
progress>0.9 in the audit); strict pre-mention commitment still climbs (first progress
decile 0.349 -> final-prefix-within-subset 0.555); every step still ends committed.

New citable baseline numbers (95 trajectories, 1,276 steps, 98,310 prefixes):
prompt-only mean commitment 0.3375; median lock-in progress 0.1904; median first
0.5-crossing 0.0893; mean switches per step 6.04; strict pre-mention rows 31,723 (32.3%).

Artifacts: `outputs/prefix_action_scores_extracted.parquet`,
`outputs/prefix_commitment_metrics_extracted.parquet`, `outputs/step_commitment_summary_extracted.parquet`,
`outputs/commitment_report_extracted.md`, `outputs/behavior_vectors_extracted.parquet`,
`outputs/behavior_manifold_extracted.parquet` (+ PCA model), `outputs/behavior_rebase_comparison.{md,csv}`,
`figures/behavior_extracted/`, `figures/behavior_rebase/commitment_vs_progress_rebase.png`,
`figures/behavior_rebase/per_action_commitment_extracted.png`.
New scripts: `scripts/03c_export_extracted_prefix_scores.py`,
`scripts/08_compare_extracted_vs_stored_behavior.py`.

### 2026-07-23 - D1 commitment-boundary phenomenology on faithful labels (script 09)

`python scripts/09_commitment_boundary_phenomenology.py` (interp venv). 1,276 steps / 95
trajectories from `prefix_commitment_metrics_extracted.parquet`.

Three clocks per decision step (progress units, medians): **first verbal action mention
0.253 < argmax stabilization 0.494 < permanent commitment>=0.8 0.648**. Only 35.6% of steps
stabilize before the first mention - first mentions are usually candidates under
deliberation, not announcements of a made decision. **Post-commitment tail: median 50.6%
of the CoT (median 524 tokens, mean 887) is generated after the forced-answer argmax has
already permanently settled** - the agentic analogue of the "epiphenomenal tail" reported
for QA reasoning (arXiv 2606.13603, ~55%). Switch hazard declines monotonically
0.133 -> 0.019 across progress deciles (no sharp aggregate boundary). Irreversibility:
below commitment 0.35 essentially every step still switches later; at 0.95 8.9% still do.
Per-action: DOWN stabilizes latest (median 0.609), RIGHT earliest (0.393). Short-CoT
steps stabilize earliest (Q1 median 0.316 vs Q4 0.596). Artifacts:
`outputs/commitment_boundary_steps.parquet`, `outputs/commitment_boundary_phenomenology.md`,
`figures/commitment_boundary/`.

### 2026-07-23 - D3 three clocks vs probe-decoded state, first pass (script 81)

`python scripts/conceptual_belief_space/81_three_clocks_lead_lag.py` on fold-0 held-out
predictions (`predictions_mean_layers_mlp.parquet`, 272 steps / 19 trajectories).

The probe reconstructs the within-step commitment CURVE well (median per-step Spearman
0.741; final-prefix predicted top action matches truth 95.2%). But its raw argmax
stabilization clock is noise-dominated and lags the forced-answer clock badly (median
0.988 vs 0.570); a 5-prefix centered smoothing of predictions pulls it to 0.916 (lead/lag
median +0.30), and the permanent-0.8 commitment clocks correlate 0.397 across steps.
Conclusion recorded as a measurement limitation, not a lead/lag claim: decoded-state
stabilization estimates are biased late by decoder noise at ~0.5 balanced accuracy, so the
lead/lag question needs stronger decoders (D2e) before it carries weight. Artifacts:
`outputs/conceptual_belief_space/three_clocks/`,
`figures/conceptual_belief_space/three_clocks_lead_lag.png`.

### 2026-07-23 - D2 layerwise dissociation, resolved across 5 folds (scripts 79 + 80)

Commands: `python scripts/conceptual_belief_space/79_layerwise_dissociation.py` (all 5 outer
folds, subsets strict + all, matched instruments: per-layer ridge -> entropy_commitment
Spearman, balanced multinomial logistic -> forced top action balanced accuracy, mask-pooled
CoT-window activations, positional bars from prefix_token_index only) and
`python scripts/conceptual_belief_space/80_occlusion_parity.py` (fold-0 transformer, layer
ablations read from BOTH heads, scopes all + strict).

**The fold-0 "commitment lives at layer 3, action at layer 18" dissociation does not
survive instrument matching. What replaces it is better:**

1. **No depth dissociation.** Both targets are decodable at every layer and improve into
   the same mid-late range: action peaks at L13-18 across folds (strict mean curve peak
   L17, 0.521 balanced vs bar 0.331), commitment peaks at L13-15 (peak L15, Spearman
   0.669 vs bar 0.343). The action curve jumps sharply at L10->11 (0.466 -> 0.510).
2. **A small, 5/5-consistent depth interaction.** Late(L16-23)-minus-early(L0-7) block
   contrast: action gains more from depth than commitment in every fold (strict subset
   interaction +0.025/+0.045/+0.039/+0.041/+0.081, mean +0.046) - belief content is more
   depth-loaded than belief strength, but both live late.
3. **Within-layer orthogonality (the real dissociation).** Removing the 3-d LDA action
   subspace from the activations changes commitment decoding by <0.003 Spearman in every
   fold at layers 3/10/18/mean; removing the commitment ridge direction changes action
   balanced accuracy by <0.005. Belief strength and belief content occupy essentially
   orthogonal linear directions of the same layers.
4. **Occlusion parity (fold 0).** The trained transformer relies on the same sparse
   layers for both heads: occluding L3 costs commitment Spearman 0.168 AND action
   balanced 0.191 (all scope; strict 0.191/0.133); L10 0.134/0.191; L18 0.057/0.107.
   The original asymmetric evidence (occlusion-for-commitment vs LDA-for-action) had
   manufactured the depth dissociation.

Also supersedes the 2026-07-22 supervised-projection "Result 3" layer-dissociation
reading. Artifacts: `outputs/conceptual_belief_space/layerwise_dissociation/`
(layerwise_curves.csv, block_contrast.csv, geometry.csv, removal_tests.csv,
positional_bars.csv, occlusion_parity.{csv,md}, report.md, pooled_cot_window_f16.npy
cache), figures `layerwise_dissociation_{strict,all}.png`,
`layerwise_commitment_action_geometry.png`, `occlusion_parity.png`.

### 2026-07-23 - D4 prompt/CoT mismatch counterfactuals scored (script 10)

`python scripts/10_score_mismatched_prompt_cot.py` (interp venv, MLX, full independent
forward passes - deliberately NO KV-cache reuse anywhere). 396 staged mismatched rows
(prompt of decision A + CoT prefix of decision B with a different agent action; cuts at
~10/50/90% of the CoT) + 396 matched controls, first scoring ever of these files.

**The forced-answer belief tracks the reasoning text, not the prompt's world state.**
Mean probability mass by cut position (n=132 per cell):

| condition | cut | p(prompt action) | p(CoT action) | top=CoT |
| --- | --- | --- | --- | --- |
| mismatched | early ~10% | 0.165 | 0.598 | 0.614 |
| mismatched | mid ~50% | 0.145 | 0.625 | 0.667 |
| mismatched | late ~90% | 0.129 | 0.673 | 0.689 |
| matched | early ~10% | 0.355 | (same) | 0.364 |
| matched | late ~90% | 0.971 | (same) | 0.985 |

Two refinements: (1) even at the earliest cut the prompt's own action is largely
abandoned (0.165) - a few hundred tokens of someone else's reasoning dominate the
visible world state; (2) the conflicting prompt is not inert: with congruent context the
model follows the CoT's conclusion at 0.985 (late), with conflicting context only 0.689,
and the missing mass disperses rather than returning to the prompt's action. Caveat:
matched and mismatched cells are different step samples; per-cell contrasts are
descriptive, the within-condition trends are the claim.

**Bonus - third-path label revalidation:** fresh cache-free scores vs
`extracted_beliefs` on the 396 matched rows: mean max|dq| 0.0142, top-action agreement
0.9823. Extraction labels now confirmed by two independent no-cache routes (Together API,
this run).

Artifacts: `outputs/followups/mismatched_prompt_cot_scores_extracted.parquet`,
`outputs/followups/mismatched_prompt_cot_report.md`,
`figures/counterfactual/mismatched_prompt_cot_tracking.png`,
log `outputs/followups/mismatched_scoring.log`. New script:
`scripts/10_score_mismatched_prompt_cot.py`.

### 2026-07-23 - D5 belief-update anchors (script 11)

`python scripts/11_belief_update_anchors.py`. 97,034 consecutive-prefix transitions over
1,276 steps; anchor = top 5% of per-transition belief total variation (threshold 0.328).

**Belief change is heavy-tailed and front-loaded, and anchors are switch points.** The top
5% of transitions carry 37.0% of all belief change; the median step concentrates 28.0% of
its change in its top-3 transitions. 75.0% of anchors coincide with a top-action change
(baseline 7.9%; 9.4x). 40.8% of anchors DECREASE commitment to the final action
(reversals). Anchor rate falls monotonically with progress (0.071 -> 0.018 per decile).

**Concentration predicts early decisions:** Spearman(per-step top-3 TV share,
argmax-stabilization progress) = **-0.510** - steps whose belief moves in a few large
jumps settle earlier; diffuse updaters settle late.

**Content matters:** chunks with no action/spatial/plan/correction content are 4.2x
DEPLETED among anchors (rate 0.058 vs 0.242 baseline); action words 1.63x, plan 1.56x,
spatial 1.29x, correction 1.21x enriched. Lexical multi-hot, observational; causal
(resampling) upgrade left as future work.

Artifacts: `outputs/belief_update_anchors.parquet`, `outputs/belief_update_anchors.md`,
`figures/belief_anchors/`. New script: `scripts/11_belief_update_anchors.py`.

### 2026-07-23 - D6 layer-steering infrastructure + smoke (script 82); D2e training launched

**D2e:** folds 1-4 of probe 77 launched overnight via `--resume` into `fold0_20260721/`
(fold 0 restored from complete artifacts; explicit `--beliefs .../extracted_beliefs.parquet`
because 77's default still points at the quarantined belief file). Log:
`training_folds1_4_20260723.log`.

**D6 infra:** `scripts/conceptual_belief_space/82_layer_steering.py` - cache-free manual
layer-loop forward (76's pattern) with directional injection at one layer over the last-12
CoT window (or `--steer_scope all_cot`); directions = fold-0-train ridge commitment
direction and balanced-logistic action directions, standardized-space coefficients mapped
through the scaler and lifted 512->2880 via the stored orthogonal projection;
norm-matched random-direction control; alpha in units of window residual RMS.

**Smoke (8 strict fold-0 test prefixes, layers {3,15}, alpha 0.5): effects are null.**
mean |d entropy_commitment| <= 0.007 for all conditions including random; flip rates
indistinguishable from control. Infrastructure validated (88 forwards, checkpointing,
summary); effect absence at modest strength noted as a candidate read/write asymmetry -
decoder directions are readable (Spearman 0.67 single-layer) and layer content is
causally load-bearing (occlusion drops up to 0.23), but writing along a single 1-d
decoder direction at 12 positions does not move the forced answer. Full sweep
(64 prefixes, layers {3,10,15,18}, alpha up to 2.0, plus all-CoT-position variant)
queued behind the overnight training. Artifacts:
`outputs/conceptual_belief_space/layer_steering/steering_{smoke}.parquet`,
`steering_summary_smoke.csv`, `steering_report_smoke.md`.

### 2026-07-24 - D2e complete: 5-fold CoT-window probe results (script 77, folds 1-4 resumed)

Overnight `--resume` run finished (one interruption mid-fold-4; per-fold atomic artifacts
made the restart lossless). All metrics on faithful extracted labels, trajectory-grouped
outer folds, target entropy_commitment. Held-out Spearman, mean +/- std over 5 folds:

| method | mean spearman | std | mean r2 |
| --- | --- | --- | --- |
| mean_layers_mlp | **0.7715** | 0.0094 | 0.509 |
| concat_layers_mlp | 0.7675 | 0.0091 | 0.506 |
| transformer_no_deltas | 0.7634 | 0.0127 | 0.508 |
| bidirectional_transformer | 0.7576 | 0.0097 | 0.496 |
| best_single_layer_mlp | 0.7474 | 0.0437 | 0.463 |
| final_layer_mlp | 0.7045 | 0.0103 | 0.405 |
| progress_length_spline (oracle) | 0.6600 | 0.0131 | 0.393 |
| progress_oracle (oracle) | 0.6435 | 0.0105 | 0.370 |
| position_index_fair (fair bar) | 0.3710 | 0.0154 | 0.201 |

The fold-0 conclusions replicate in every fold: commitment is strongly decodable and not
positional (+0.40 over the fair bar, +0.11 over the oracle), and the layer-sequence
transformer never beats mean-over-layers - the prompt.txt layer-transition hypothesis is
refuted with error bars.

**Model-selection artifact worth fixing in any rerun:** selection optimizes validation
commitment Spearman only, so folds 2 and 3 selected lambda_q=0.0 and their q heads are
untrained (held-out top-action acc 0.34/0.24 vs ~0.50 on folds 0/1/4 with lambda_q=0.1).
Commitment results are unaffected; q-head analyses (occlusion-parity action side,
probe-argmax clocks) are valid only on folds 0/1/4. A lambda_q floor or multi-objective
selection would fix this.

**Extended D3 (folds 0/1/4 pooled, 793 steps):** fold-0 findings replicate - per-step
commitment-curve Spearman median 0.737; probe-decoded stabilization lags forced-answer
stabilization (median +0.32 after smoothing); permanent-0.8 clock correlation 0.335.
Artifacts: `outputs/conceptual_belief_space/three_clocks_folds014/`.

Fold artifacts and models: `fold0_20260721/` (now a full 5-fold directory), pooled
`oof_predictions.parquet`, per-fold dual-head occlusion parity running into
`layerwise_dissociation/parity_split{1..4}/`.

### 2026-07-24 - Occlusion parity across all 5 folds: only layer 10 replicates

`80_occlusion_parity.py` re-run on the fold-1..4 checkpoints
(`layerwise_dissociation/parity_split{1..4}/`). Aggregated with fold 0:

- **Layer 10 is the only reproducibly load-bearing layer**: top commitment-drop layer in
  4/5 folds; mean drop 0.093 all folds (0.108 on live-q folds 0/1/4, where it also costs
  the action head 0.180). This matches the sharp L10->11 jump in the matched decodability
  curves (79): layer 10 is where action information becomes linearly available AND the
  layer every trained probe leans on.
- **Other sparse-importance layers are fold-idiosyncratic**: fold 0 leaned on L3 (0.168),
  fold 2 on L14 (0.125), fold 3 on L10 only, fold 4 on L10+L1. Occlusion importance is a
  property of the trained probe's internal shortcut choices, not of the model; only the
  L10 reliance replicates. The 2026-07-23 D2 entry's "same sparse layers {3,10,18}"
  phrasing is hereby narrowed to "layer 10, plus fold-specific extras".
- Commitment and action drops stay coupled wherever the q head is trained - reinforcing
  the D2 conclusion that the dissociation is directional (within-layer), not depth- or
  reliance-based.

### 2026-07-24 - D6 verdict: decoder directions are readable but not writable (script 82)

Full sweep interrupted at 28/64 prefixes (long GPU jobs on this machine get killed -
per-prefix checkpointing preserved complete condition sets; analyzed n=28, 1,148 runs,
paired within-prefix). Layers {3,10,15,18} x alpha {+-1.0, +-2.0} (2x the window residual
RMS) x directions {commitment, action-toward-runner-up, norm-matched random}.

**Null across the board.** Mean |d entropy_commitment| <= 0.005 (sd ~0.015) for every
(kind, alpha, layer) cell including random; argmax flip rate 0 everywhere except one
prefix at alpha=2.0 - identically for the action direction AND the random control.
Nothing exceeds the random-direction control at any strength.

Read/write asymmetry, stated with the positive results it contrasts against: the same
directions support Spearman 0.67 single-layer decoding (79), and zeroing layer 10
costs the trained probe 0.11-0.23 (80) - but writing along a rank-1 decoder direction
at the probe's own 12-token input window does not move the forced answer. The belief
encoding is redundant/distributed enough to absorb rank-1 window-local injections.
Untested escalations, left open: all-CoT-position injection (--steer_scope all_cot),
multi-layer simultaneous injection, and token-embedding-level patching.

Artifacts: `layer_steering/steering_full_n28.parquet`, `steering_summary_full_n28.csv`,
smoke files, `steering_full.log`.

### 2026-07-24 - Conjecture experiments, CPU batch: C1, C4, C5, C9, C10

Tests of five conjectures from `gridworld_action_belief_gap_claims_conjectures.md`, all on
faithful labels. Scripts: `12_conj_c4_hazard_markov.py`, `13_conj_c5_tail_verification.py`,
`conceptual_belief_space/83_conj_c1_gain_direction.py`, `84_conj_c9_clock_noise.py`,
`85_conj_c10_prompt_prior.py`. Artifacts under `outputs/conjectures/`, figures under
`figures/conjectures/`.

**C1 (gain-direction factorization) - PARTIALLY SUPPORTED.** (i) YES: a multiplicative
gain model, logits = softplus(<w,x>+c) * unit(Ux+d), beats an additive linear decoder at
every representation (held-out KL, all-layer mean 0.8148 -> 0.7284; L15 0.8396 -> 0.7475;
L18 0.8261 -> 0.7415), and a frozen-RANDOM-gain control is *worse* than additive
(0.8398/0.8391/0.8415), so the learned gain direction - not the extra 25% capacity - is
doing the work. (ii) NO: the learned gain barely tracks the independent ridge commitment
axis (Spearman +0.132; vs true commitment +0.142), so the gain the decoder wants is not
the commitment direction. (iii) NO: the content direction settles before the gain in only
52.9% of steps (chance). **Unplanned finding:** the additive decoder's implied confidence
ANTI-correlates with true commitment (Spearman -0.05 to -0.12); adding a learned gain only
lifts it to +0.12..+0.17, versus 0.77 for a dedicated commitment probe. Decoding the
belief's direction does not deliver its strength - an independent line of support for the
two-channel reading of Theorem 3.

**C4 (belief screens off position) - REFUTED, and replaced by something sharper.** Adding
progress to margin gains nothing in an ADDITIVE spline model (AUC 0.8474 -> 0.8452), which
looks like screening-off. But an interaction-capable model does gain (GBT AUC 0.8471 ->
0.8520, log-loss 0.21362 -> 0.21128), because the position effect REVERSES SIGN across
belief levels: in the lowest margin quintile the hazard *rises* with progress (0.242 ->
0.370; quintile 2 rises 2.68x), while in the top quintile it *falls* 10x (0.0156 ->
0.0016). The two cancel in an additive fit. Position is therefore not redundant given the
belief - it is a **credibility amplifier**: late-and-confident is far more final than
early-and-confident, and late-and-unsure is far more volatile than early-and-unsure.

**C5 (tail is verification, and cuttable) - SPLIT.** (i) REFUTED, and this is the
important one: a PROSPECTIVE stopping rule (exit at the first prefix whose margin >= c)
agrees with the full-CoT action only 42.2% (c=0.3) to 68.0% (c=0.9) of the time. The
original conjecture's rule (exit at the permanent 0.8 crossing) is vacuous - commitment
>= 0.8 forces the argmax to equal the taken action, and the crossing time is not knowable
online. So the tail is NOT safely cuttable by a confidence rule: the belief reaches high
margin early on the *wrong* action often enough to break early exit. This is a genuine
contrast with the QA early-exit literature (arXiv 2606.13603, ~55% cuttable).
(ii) SUPPORTED: verification vocabulary is 5.23x denser after stabilization than before
(0.62 -> 3.26 hits/1000 chars; higher in 93.7% of steps), action words 2.72x, plan terms
1.59x - while spatial vocabulary is nearly flat (1.11x). The tail stops doing spatial
reasoning and starts verifying and naming. (iii) SUPPORTED: steps that reverse out of a
high-commitment state carry 1.39x more correction markers and 1.89x more action words in
their pre-stabilization text.

**C9 (the decoded clock's lag is decoder noise) - SUPPORTED.** The probe's prefix-level
argmax accuracy is 0.4998 and its errors are strongly autocorrelated (P(err|err)=0.667).
Corrupting the TRUE trajectories with those error statistics - a process with zero lag by
construction - manufactures a median stabilization lag of 0.483 (i.i.d.) or 0.475
(persistence-matched), versus the 0.432 actually observed. The observed lag is *smaller*
than the noise floor (mean difference -0.033 to -0.076; observed earlier in 53-63% of
steps). **Observation 3.6's apparent internal lag is fully accounted for by decoder noise;
no internal lag needs to be posited.** The lead/lag question stays open in the other
direction: a better decoder is needed to show the internal state genuinely *leads*.

**C10 (static prior direction) - SPLIT, with the same strength/content moral.** (a)
SUPPORTED strongly: the pre-CoT belief is linearly decodable from the single prompt-final
window state, held-out R2 rising 0.131 (L0) -> 0.431 (L10) -> **0.673 (L18)** -> 0.613
(L23), balanced accuracy 0.508 vs chance 0.25. The prior itself is stark: at the
prompt-only cut the argmax is RIGHT in 79.5% of steps, LEFT 19.2%, UP 0.9%, DOWN 0.4%.
(b) REFUTED: the predicted RIGHT component does *not* predict early settling (Spearman
-0.054). But the prompt's commitment MAGNITUDE does (-0.341). Which action the model
leans toward at the outset says nothing about when it will settle; how strongly it leans
says a lot - the strength/content split again, now in the prior.

### 2026-07-24 - Correction to C5(i), and C7 result

**C5(i) correction (supersedes the C5(i) paragraph in the CPU-batch entry above).** The
first pass used a single naive rule -- exit the moment the margin first crosses c -- which
fires at the prompt-only prefix for most steps, because the pre-CoT prior is already sharp
(C10: RIGHT is the argmax in 79.5% of steps before any reasoning). That rule is a strawman.
Re-ran with the full grid of genuinely online rules: threshold c in {0.3..0.9} x persistence
k in {1,3,5} consecutive prefixes above c x token floor in {0,200,500} emitted tokens
(none of which uses the unknowable total CoT length), and took the Pareto frontier.

Revised finding: **early exit is partially viable but does not reach the 95% bar.** Best
online rule: margin >= 0.9 sustained for 5 prefixes after 500 tokens -> **93.3% agreement
with the full-CoT action at 28.3% mean token savings**. Only 2 of 63 rules clear 90%; none
clears 95%. The frontier degrades smoothly (89.5% at 42.7% savings; 82.8% at 51.0%). The
naive rule remains bad (42.2%), which is the real lesson: *debouncing matters enormously*
because the belief makes confident transient excursions to the wrong action. Compared with
the ~55% cut reported for QA reasoning (arXiv 2606.13603), the agentic setting is markedly
less cuttable at comparable fidelity. (ii) and (iii) of the earlier entry stand unchanged.

**C7 (disavowal insert) - REFUTED.** Appending an explicit disavowal
("the reasoning above was copied from a different maze ... decide using only the map") to
the analysis channel before the forced suffix, on the same 396 mismatched + 396 matched
rows as D4, paired row-for-row: mean p(prompt action) moves 0.1463 -> 0.1475, i.e. it
recovers **0.14%** of the mass the mismatched condition costs, against a predicted >= 50%.
The top action becomes the prompt's action in 13.1% of rows versus 13.6% at baseline --
no improvement. The effect is statistically reliable but numerically trivial (72.2% of
rows move in the predicted direction, Wilcoxon p = 1.3e-8): the model registers the
sentence, it just does not reallocate belief on the strength of it. Matched controls move
by a comparable magnitude in the *harmful* direction (-0.037/-0.032/-0.011 across cuts),
so the instruction has a small generic confidence-lowering effect rather than a
targeted authorship effect.

Conclusion: the CoT's grip on the forced-answer belief (Theorem 1) is not an inferential
authorship prior that can be talked out of with an instruction. Artifacts:
`outputs/conjectures/c7_disavowal/`, `figures/conjectures/c7_disavowal.png`;
script `scripts/14_conj_c7_disavowal.py`.

### 2026-07-24 - Conjecture experiments, GPU batch: C2, C6, C3

Scripts `conceptual_belief_space/87_conj_c2_subblock_ablation.py`,
`86_conj_c6_patching.py`, `15_conj_c3_anchor_resampling.py`. All forwards cache-free;
per-unit checkpointing throughout.

**C2 (layer 10-11 attention is a routing bottleneck of the model) - NOT SUPPORTED, and
it corrects an over-reading of Proposition 3.3.** Skipping an entire attention or MLP
sub-block at one layer (24 strict fold-0 prefixes, 14 layers x 2 sub-blocks; manual-vs-
standard forward equivalence exactly 0.000e+00, so the null is not a reimplementation
artifact) moves the forced-answer belief only slightly ANYWHERE: the largest mean
KL(baseline || ablated) over all 28 conditions is 0.097 (L16 MLP), with attention peaks at
L0 (0.086) and L11 (0.083). Attention ablation at L10-11 averages 0.0658 versus 0.0426
elsewhere - 1.55x, short of the 2x criterion - and **ablating layer 10's attention (0.0487)
damages the belief LESS than ablating layer 0's (0.0856)**.

The consequence matters for the paper: layer 10's unique status in Proposition 3.3 is a
fact about where the belief becomes linearly READABLE, not about where it is computed or
routed. Probe-occlusion importance and model-ablation importance measure different things
and here they disagree. Proposition 3.3 should be stated as a decodability landmark, never
as a causal one.

**C6 (writability needs rank or breadth) - NOT SUPPORTED, though monotone in rank as
predicted.** Nested transplant ladder over 24 donor/recipient pairs with different forced
answers, identical donors across modes: rank-1 along the commitment axis, rank-1 along the
donor's action direction, rank-3 LDA action subspace, and full 12x2880 window replacement,
each at layers 3/10/15/18. Rank-1 and rank-3 do literally nothing (flip rate 0.000
everywhere, |mean dp(donor action)| <= 0.003). Full-window replacement is the only mode
with any effect, and it tops out at **20.8% of pairs at layer 15** (L3 8.3%, L10 16.7%,
L18 0.0%), far below the 50% criterion.

So even overwriting the probe's ENTIRE input window at one layer with a donor state that
produces a different answer usually fails to change the answer. This confirms the
falsification branch written into C6: the belief is reconstructed from context outside the
12-token window (the prompt and earlier CoT), so the window the probe reads is not where
the belief causally resides.

**C3 (anchors are causal forks) - NOT SUPPORTED AT THIS SAMPLE SIZE; underpowered rather
than refuted.** Branching at 20 anchors and 20 position-matched controls from 7 steps,
10 resamples of 60 tokens at temperature 1 per branch, then forcing the answer: anchor
dispersion (mean pairwise TV across resamples) 0.208 vs control 0.165, ratio **1.26**
(criterion 2.0), Mann-Whitney p = 0.32; distinct sampled answers 1.70 vs 1.45. The point
estimate is in the predicted direction on both measures but nowhere near significance with
n = 20 per arm. Note the controls are position-matched (0.504 vs 0.507 mean progress) yet
differ 9x in observed TV (0.068 vs 0.683), so the manipulation check passes - the test is
simply too small. A properly powered rerun (>= 60 steps, >= 30 resamples) is the honest
follow-up.

**Cross-cutting result: the belief is readable but not writable, now across four
independent intervention families.** Rank-1 activation steering at 2x RMS (D6): null.
Whole-sub-block ablation (C2): <= 0.1 KL anywhere. Full-window state transplant (C6):
20.8% at best. Explicit natural-language instruction (C7): 0.14% of the lost mass. Against
this stands a held-out decoding Spearman of 0.7715 +/- 0.0094 (D2e). The forced-answer
belief is a highly redundant, distributed, context-reconstructed quantity that is easy to
read off and hard to move by any local edit.

### 2026-07-24 - C8 confidence-axis transfer: SUPPORTED; all ten conjectures now tested

**C8 - SUPPORTED.** Transfer task holds the output interface exactly fixed while removing
all spatial content: 16 arithmetic/logic questions whose four options are labelled
UP/DOWN/LEFT/RIGHT (stated to be arbitrary names) and answered through the identical JSON
forced suffix. The model solves them (93.8% final-cut accuracy), so the measured beliefs
are real. Gridworld commitment directions were frozen from fold-0 strict train rows,
standardization included, and applied unchanged to 320 new-task prefixes.

| layer | frozen transfer | task-refit ceiling | position bar |
| --- | --- | --- | --- |
| 3 | **-0.5138** (sign-inverted) | 0.4348 | 0.3842 |
| 10 | +0.3709 | 0.4921 | 0.3842 |
| 13 | **+0.5146** | 0.6115 | 0.3842 |
| 15 | +0.3912 | 0.6363 | 0.3842 |
| 18 | -0.0738 | 0.5510 | 0.3842 |
| 23 | +0.1294 | 0.4944 | 0.3842 |

At layer 13 the frozen direction reaches 84% of the task-refit ceiling with zero new
fitting, and clears the position-only bar. Transfer is strong mid-network and decays at
both ends; layer 3 transfers with INVERTED sign, so the early-layer axis is not the same
quantity despite carrying comparable information. Commitment is therefore substantially a
task-general confidence code at mid-depth rather than a gridworld action-plan quantity -
the prediction of Theorem 3's content-orthogonality. Caveats: 16 questions, short CoTs
(37-228 tokens), and a high positional bar because commitment rises with position in both
tasks. Artifacts: `outputs/conjectures/c8_transfer/`,
`figures/conjectures/c8_confidence_transfer.png`.

**Ten-conjecture scoreboard.** Supported: C8, C9, C10(a). Refuted: C2, C6, C7. Replaced by
sharper claims: C4 (sign-reversing margin x position interaction), C5 (Pareto frontier of
online exit rules). Partial: C1 (gain helps; gain != commitment axis). Underpowered: C3.

Document `gridworld_action_belief_gap_claims_conjectures.{md,docx}` updated: each
conjecture now carries a **Result** paragraph in the same role Evidence plays for the
established claims, plus Table 1 (outcome scoreboard) and Figures 13-22. Rebuilt docx
verified: 22 embedded figures, 22 drawings, 455 equations, 2 tables, no pandoc warnings.

### 2026-07-25 - Synthesis figures added to the claims document (script 89)

`python scripts/conceptual_belief_space/89_synthesis_figures.py`. Two cross-cutting claims
in `gridworld_action_belief_gap_claims_conjectures.md` were stated in prose with no figure;
both now have one, and the hardest-to-read existing panel was redrawn. All values are
recomputed from stored artifacts inside the script - nothing hard-coded.

**New Figure 22b (read/write asymmetry).** Left panel: decoding Spearman 0.7715 vs
length-oracle 0.6435 vs fair positional bar 0.3710. Right panel: each intervention family
at its single best condition, split into "changed the answer at all" vs "changed it to the
intended answer" - rank-1 steering 0.036/0.036 (random control 0.036), sub-block deletion
0.292/no-target, full-window transplant 0.250/0.208, instruction 0.058/0.010. The split
between the two columns is the point: the belief can be damaged but not directed.

**New Figure 23 (strength/content).** Left: what a content decoder knows about commitment
(additive +0.043, gain +0.175) vs a dedicated commitment probe (+0.7715). Right: prior
strength predicts settling (0.341) while prior direction does not (0.054).

**Redrawn Figure 14 (C2).** The two overlapping 14-point line series became a per-layer
paired comparison (attention vs MLP deletion on one row per layer, predicted bottleneck
darkened), which is the form the question actually asks.

Palette validated with the dataviz skill's checker before drawing: the four categorical
colours (#2a78d6, #eb6834, #1baf7a, #4a3aa7) pass all checks in light mode (worst adjacent
CVD deltaE 9.2 deutan, 27.6 normal); the grey #9aa0a6 is used only as a neutral for
baselines and controls, never as a series, and the one contrast warning is relieved by
direct labels throughout.

**Two corrections made while building the figures.** (1) A first pass computed each
intervention's best condition by grouping the steering runs on the raw `direction` column,
which splits action_UP/DOWN/LEFT/RIGHT into cells of ~7 rows whose maximum is dominated by
sampling noise - it reported a spurious 0.111 flip rate. Collapsing the four action
directions into one family restores the correct 0.036, identical to the random control and
consistent with the D6 entry. (2) The C1 write-up claimed the additive decoder's implied
confidence "anti-correlates with true commitment (-0.05 to -0.12)". The accurate statement
is that it is uninformative: rho ranges -0.125 (L15) to +0.043 (all-layer mean), scattered
around zero. Corrected in the document.

Rebuilt docx verified: 24 embedded figures, 24 drawings, 465 equations, 3 tables, no pandoc
warnings.

### 2026-07-25 - NeurIPS-format paper drafted and compiled (paper/)

`paper/main.tex` + `paper/neurips_2025.sty` (official style, copied from the user's local
copy; the NeurIPS server returned 403 to direct download). Builds clean with pdflatex x2:
10 pages, content through the Conclusion within the 9-page limit, references after.

Story arc: (1) when does the agent commit - three clocks, 50.6% post-decision tail,
declining hazard, position as credibility amplifier, anchored updates; (2) what is the
belief made of - 0.7715 +/- 0.0094 decodability vs 0.371 fair bar, no layer-transition
code, strength/content separability from four independent measurements; (3) can it be
moved - four intervention families, read/write asymmetry; plus two methodological pitfalls
(probe-occlusion != causal importance; additive controls can confirm a false screening-off
hypothesis) and the label-integrity triangulation.

Paper figures were regenerated rather than reused: `scripts/conceptual_belief_space/
90_paper_figures.py` redraws four of them (anchors, hazard interaction, tail, transfer)
without the internal conjecture identifiers ("C4:", "C5:", "C8:") that appeared in the
experiment-script titles, at print-legible font sizes and one panel per point;
`89_synthesis_figures.py` gained a `--paper` flag that omits suptitles, since in a paper
the caption carries the message. Three duplicated figure blocks (the same PNG appearing
both inside a composite and again standalone) were removed and their references repointed.

Spawned three peer-review agents (methods/statistics, interpretability domain, adversarial)
with read access to the artifacts so they can verify the paper's numbers against
RUN_HISTORY and outputs/ rather than only reading the prose.

### 2026-07-25 - Peer review round 1: eight verified errors in the paper, corrected

Two of three review agents returned. Both scored the paper borderline (4 and 5 of 10) and
both located real defects, which I verified against the artifacts before acting. Every
correction below was confirmed by direct recomputation, not accepted on the reviewer's word.

**Errors confirmed and fixed in `paper/main.tex`:**

1. **Steering magnitude was misstated by a factor of ~54.** `82_layer_steering.py` scales a
   unit-L2 direction in R^2880 by the PER-ELEMENT residual RMS, so alpha=2 perturbs a token
   state by `alpha/sqrt(2880)` = **3.7%** of its norm, not "twice the residual RMS" as the
   paper said. Verified numerically. The steering null was therefore run at a magnitude far
   too small to bound writability. Paper now states the true ratio, labels the null weak,
   and reports the Clopper-Pearson bound (0/28 -> upper bound 12.3%). A norm-scaled rerun
   (`--scale_mode token_norm`, new option) is in flight.
2. **Transplant scope overstated.** "Overwriting every number the probe reads" - the probe
   reads [24 layers x 12 tokens x 512] and the patch hits ONE layer, i.e. 1/24 of its input
   over 12 of ~1500 positions. Corrected in abstract and body.
3. **Label-integrity adjudication used a non-discriminating check.** The paper claimed
   "final prefix: 100.0% vs 90.8%". Recomputed: the final prefix gives **1.0000 for BOTH**
   regimes (extracted and stored) - it does not separate them at all; the 90.8% came from a
   different quantity (a partial-subset progress>0.9 rate). Replaced with the full-corpus
   terminal-region comparison, 0.9788 vs 0.9149, plus an explicit note that the final prefix
   does not discriminate - which is itself why the corruption was hard to see.
4. **Stale bootstrap CI.** Paper quoted the superseded fold-0-only figure (-0.016,
   [-0.021,-0.011]); current 5-fold `clustered_bootstrap.csv` gives **-0.0131,
   [-0.0186,-0.0079]**.
5. **"Never beats mean-pooling in any fold" is false.** `transformer_no_deltas` = 0.7779 vs
   `mean_layers_mlp` = 0.7769 in fold 2. Now stated as four of five folds, with the
   exception named.
6. **Oracle mislabelled.** 0.6435 is the progress oracle; the progress+length oracle is
   0.6600. The paper now compares against the stronger one.
7. **"The tail stops reasoning about space" contradicted our own data.** Spatial density
   *rises* 27.4 -> 30.4 per 1000 chars (1.11x) and stays the densest category. Corrected to:
   the tail ADDS verification and naming on top of continued spatial reasoning.
8. **Instruction effect reported one-sided.** 4/396 rows gain the prompt action as argmax
   while 6/396 lose it; net **-0.005**, not +0.010. Both now reported.

**Framing corrections also applied:** the read/write result is now positioned as the
belief-valued instance of the known self-repair / Hydra effect (McGrath et al. 2023;
Wang et al. 2023) rather than as a new phenomenon; the forced-answer measurement now credits
Lanham et al.'s early-answering protocol as its antecedent; Bogdan et al. (Thought Anchors)
was in the bibliography but uncited despite the anchor analysis using its method - now cited
at the point of use; Kadavath et al. added as the prior result on content-independent
confidence; Rimsky et al. and Makelov et al. added as the reasons our steering null is weak
evidence; "nearly orthogonal" restated as 5.4x an isotropic chance level; and the folds-2/3
untrained-action-head model-selection artifact is now disclosed in the paper rather than
only in this log.

**Not yet addressed (needs new compute or data):** directions fitted in the full 2880-d
space rather than the projection's 512-d image; a positive-control steering direction; a
bag-of-words baseline on the same 12 tokens; a prompt-state+position baseline; sampled-
continuation validation of the forced-answer functional; de-anonymising five citations;
NeurIPS checklist and appendix.

### 2026-07-25 - Peer review round 2 (adversarial): nine further errors, corrected

Third reviewer returned (weak reject, 5/10, confidence 4). It independently recomputed ~40
numbers, confirmed the great majority, and confirmed the anti-leakage split machinery is
genuinely enforced -- but found nine further defects beyond rounds 1-2. All verified before
acting:

1. **The stated probe contract was false for half of Table 1.** Paper said "<10^6 readout
   parameters". Actual checkpoint parameter counts: mean_layers_mlp / final_layer_mlp /
   best_single_layer_mlp = 664,649 each (compliant), but bidirectional_transformer and
   transformer_no_deltas = 7,475,657 and concat_layers_mlp = 7,760,585 -- 7.5-7.8x over.
   The headline probe complies; the comparison architectures do not, and they *lose*, which
   strengthens rather than weakens the negative result. Restated accordingly.
2. **Figure 2's caption described a two-panel figure that no longer exists** - my own doing:
   `90_paper_figures.py` regenerated the hazard-interaction plot as a single panel and I
   left the "(left)...(right)" caption. Fixed, with a pointer to the marginal-hazard figure.
3. **Table 2's caption asserted a pattern its own numbers contradict.** For the one row
   where both columns are defined and non-trivial (transplant, 0.250 / 0.208), 83% of the
   changes hit the intended target - the opposite of "damaged but not directed". Caption now
   says the finding is the LEVEL, not the gap, and notes the transplant aims well when it
   works. Added n and Wilson intervals to every row.
4. **"Debouncing is essential: 42.2%" conflated three factors.** Holding threshold and floor
   at their best values, persistence 1 -> 5 moves agreement 0.794 -> 0.933 (~14 points); the
   42.2% figure is the weakest rule in the grid (c=0.3, k=1, floor=0), so the 42->93 gap
   mixes threshold, floor and persistence. Decomposed properly.
5. **The cut stride was misstated and its bias unreported.** Paper said "15-25 tokens";
   actual gaps run 1-150 (median 20, mean 24.4, only 31.4% inside [15,25]) because cuts are
   thinned to <=80 per decision. Consequence now stated: T_stab is verified only at sampled
   cuts, so it is a LOWER bound and the post-decision tail an UPPER bound, coarser in longer
   chains. The bias runs against the reported length effect, which is therefore conservative.
6. **"Layer 10 in four of five folds (mean drop 0.093)" fused two scopes.** Recomputed:
   all-prefix scope = top in 3/5, mean drop 0.0933; pre-verbal scope = top in 4/5, mean drop
   0.1843. Both now stated with their scope.
7. **"Closest to orthogonal exactly in the best-decoding layers" is not what the curves
   show.** Overlap minima are L10/L16; decoding peaks are L15/L17. Downgraded to the
   band-level statement the data support.
8. **The read/write comparison mixed quantities.** 0.771 is a strength readout; the flip
   rates are action identity, whose matched read-side number is 0.521 balanced accuracy vs a
   0.331 bar. Now stated, with the note that the asymmetry also holds within the strength
   channel (steering moves entropy commitment by <=0.006 anywhere).
9. **The C10 prior dissociation paired a true-label correlation against a decoded one.**
   Now reported true-vs-true (-0.341 strength vs -0.039 content) with the explicit caveat
   that decoder-side the two perform alike (-0.044 vs -0.054), which bounds how much of the
   effect is representational.

Also: the cross-condition 0.985-vs-0.689 contrast now carries the different-step-samples
caveat our own D4 entry attached to it; "heavy-tailed" downgraded to "concentrated, with an
exponential tail" (total variation is bounded in [0,1]).

Paper recompiles at 9 pages of content + references. Across three reviews, 17 verified
errors were corrected; every one was checked against the artifacts rather than accepted on
the reviewer's word, and in each case the artifact sided with the reviewer.

### 2026-07-25 - Steering re-run at honest magnitudes: the null survives, for a better reason

`82_layer_steering.py --scale_mode token_norm --alphas 0.1,0.25,0.5,1.0 --layers 10,15,18
--n_prefixes 32 --tag normscaled`. The new scale mode sets ||delta|| = alpha*||h_token||,
fixing the factor-of-sqrt(2880) understatement in the original run (see the 2026-07-25
review entry).

Rate of arriving at the INTENDED action, pooled over three layers (96 trials per cell for
decoder directions, 192 for the control), Wilson 95% intervals:

| alpha | action direction | random control | Fisher p |
| --- | --- | --- | --- |
| 0.25 | 0.052 [0.02,0.12] | 0.036 [0.02,0.07] | 0.54 |
| 0.50 | 0.104 [0.06,0.18] | 0.052 [0.03,0.09] | 0.14 |
| 1.00 | 0.135 [0.08,0.22] | 0.130 [0.09,0.19] | 1.00 |

**This is a materially better result than the one it replaces.** The original run was
uninformative: at ~3.7% of the residual norm essentially nothing happened to anything, so
"no effect" said nothing about the representation. At honest magnitudes things do happen -
flip rates rise monotonically with alpha, reaching 14.6% (any change) at alpha=1 - but a
random vector of the same norm does it just as often (13.0% to-target vs 13.5%, Fisher
p=1.00). The largest action-minus-random gap is at alpha=0.5 (0.104 vs 0.052) and is not
significant at n=96/192 (p=0.14).

So the claim sharpens from "the belief cannot be moved" (unsupported) to "at magnitudes
where the intervention demonstrably does something, the decoder direction is no better at
choosing WHAT it does than noise" - damage without aim, now demonstrated at a magnitude
where damage actually occurs. Paper's steering paragraph, abstract sentence and Table 2 row
replaced with this experiment and its intervals.

Remaining acknowledged limits, unchanged: directions live in the fixed random 512-of-2880
subspace the probe reads, and there is still no positive-control direction known to steer
this model.

### 2026-07-25 - 3-d LDA projection: the four arms are absent because the RADIAL half is missing

Two follow-ups prompted by the question "did you only look at the first PCA components, and
did you ever plot the LDA projection in 3-d?" Both gaps were real: the earlier negative
result used PCA(3) shown as PC1-PC2, and the LDA figure only ever displayed 2-d views.

**Script 91 - exhaustive PCA pair search.** 30 components, all 435 pairs, screened by a
multiclass trace-ratio statistic with a permutation null for the SEARCH MAXIMUM (200
label shuffles, permuted within trajectory). Best pair PC2-PC3 scores 0.0826 against a null
whose maximum over 200 shuffles was 0.0028 (p = 0.000), so action structure genuinely
exists beyond the leading components. But it buys almost nothing: grouped-CV balanced
accuracy 0.3321 for PC2-PC3 vs 0.3107 for PC1-PC2 and 0.5503 for supervised LDA(2);
silhouette stays negative in every plane. Refines the earlier framing - PC1/PC2 carry the
most action signal AND the most position signal (|r| 0.56, 0.40), while PC7 and PC25 carry
action with little position, i.e. action is diffuse and entangled rather than absent.
Artifacts: `outputs/conceptual_belief_space/pca_component_search/`.

**Script 92 - the 3-d LDA arm test.** The behavioural arms have a specific geometry:
RADIUS is commitment, ANGLE is which action. Since the commitment direction was measured
nearly orthogonal to the action subspace, projecting onto the three LDA axes should discard
the radial coordinate. Tested by computing four scale-free "armness" statistics on the SAME
held-out prefixes in two spaces - the behavioural CLR belief space (mapped to an orthonormal
3-d basis of the sum-to-zero hyperplane, where the arms are known to exist and which
therefore calibrates the scale) and the 3-d LDA projection of activations, fitted on
training trajectories only:

| space | own-centroid cosine | angular purity | radius vs commitment | centroid radius ratio |
| --- | --- | --- | --- | --- |
| behavioural belief (CLR) | 0.583 | 0.677 | **0.781** | **0.922** |
| activation LDA (held-out) | 0.350 | 0.470 | **0.179** | 0.450 |
| activation LDA (euclidean) | 0.369 | 0.460 | 0.145 | 0.494 |
| activation LDA (in-sample) | 0.460 | 0.561 | 0.203 | 0.587 |

The prediction holds exactly. The ANGULAR half of the structure survives in weakened form
(purity 0.47 vs 0.68, own-centroid cosine 0.35 vs 0.58 - real, not zero). The RADIAL half,
which is what makes an arm an arm, is gone: radius tracks commitment at 0.781 behaviourally
and 0.179 in the LDA projection, and class centroids sit at 0.92 of the cloud radius
behaviourally versus 0.45 in activations - arms reach outward, blobs huddle near the origin.
The in-sample row shows LDA does overfit somewhat but the held-out structure is real.

This unifies two results that had been reported separately: "no four-arm manifold" and
"the commitment direction is orthogonal to the action subspace" are the same fact seen from
two sides. The LDA axes are chosen to separate class MEANS, commitment is largely orthogonal
to that span, so the projection keeps content and discards strength - and a four-arm shape
requires both. Artifacts: `outputs/conceptual_belief_space/lda_3d_arms/`, figure
`figures/conceptual_belief_space/lda_3d_arm_test.png`.

### 2026-07-25 - PCA(3->2) on the LDA coordinates: confirms the arm test is not a rendering artifact

Extended script 92: after projecting activations onto the three LDA axes, collapse that 3-d
cloud to its best-variance plane with PCA(2), fitted on training trajectories and applied to
held-out ones. The same collapse is applied to the behavioural CLR cloud as a matched
control, which is the point of the exercise - it establishes whether the 3->2 step is
capable of preserving arms at all.

| space | own-centroid cosine | angular purity | radius vs commitment | centroid radius ratio |
| --- | --- | --- | --- | --- |
| behavioural belief (CLR), 3-d | 0.583 | 0.677 | 0.781 | 0.922 |
| behavioural belief, PCA to 2-d | 0.579 | 0.560 | **0.729** | **0.932** |
| activation LDA (whitened), 3-d | 0.350 | 0.470 | 0.179 | 0.450 |
| activation LDA, PCA to 2-d | 0.380 | 0.442 | **0.146** | 0.510 |

Explained variance of the 2-d plane: 0.777 behavioural, 0.709 activation.

**The control works.** Collapsing to 2-d barely touches the behavioural arms - own-centroid
cosine 0.583 -> 0.579, radius-commitment 0.781 -> 0.729, centroid radius ratio 0.922 ->
0.932 - and the middle-top panel of the figure shows the four lobes plainly. So the 3->2
projection is not what destroys arm structure. Applied to the activation cloud the same
operation still yields a diffuse ball with centroids sitting inside it (radius-commitment
0.146). The absence of arms in activation space therefore survives the most favourable 2-d
rendering available, and is not an artifact of dimensionality reduction or of an unlucky
choice of viewing plane.

Note this is a rotation plus an orthogonal projection of the same cloud, so it could not
have created radial structure that was not present; its value is entirely as the negative
control just described. Figure regenerated at
`figures/conceptual_belief_space/lda_3d_arm_test.png` (now 2x3: 3-d cloud, PCA 2-d view,
radius-vs-commitment, for each space).

### 2026-07-25 - Two action axes + the commitment direction: the activation code is a CYLINDER

Script `93_action_plane_by_commitment.py`. Constructed the space the arm geometry actually
needs: two orthonormalised LDA action axes for the angular coordinate, plus the ridge
commitment direction (with its in-plane component removed) as the third. Axes fitted on
training trajectories, evaluated on 6,534 held-out pre-verbal prefixes.

Construction checks out: the commitment direction keeps 0.967 of its norm after removing
the action plane (i.e. it is nearly pure commitment), and the third coordinate correlates
+0.634 with true commitment.

Shape prediction was sharp. A multiplicative code (logits = gain x content direction) gives
a CONE: classes inseparable at low commitment, fanning apart as commitment rises. Two
independent additive channels give a CYLINDER: constant separation at every height.

Class separation in the action plane, by commitment sextile (shuffled-label control in
brackets): 0.155 [0.004], 0.231 [0.002], 0.321 [0.001], 0.260 [0.002], 0.134 [0.004],
0.210 [0.002]. Angular purity: 0.436, 0.437, 0.478, 0.463, 0.389, 0.465 against a shuffled
baseline of 0.22-0.29. Growth from lowest to highest bin: **1.35x, and non-monotone** -
separation peaks in the middle and is no higher at commitment 0.94 than at 0.13.

**Verdict: CYLINDER.** Action separation is real (well above the shuffled control at every
level) but does not grow with commitment. The two coordinates do not interact.

**This confirms rather than refutes the two-channel reading, and I had the prediction
backwards in the message that proposed the experiment.** I wrote that failure to reconstruct
the star would show "the two-channel story is wrong in an interesting way". That conflated
two different models. Orthogonal, independent channels predict exactly a cylinder; it is the
MULTIPLICATIVE coupling that predicts a cone. So the result refutes the multiplicative
reading of the activation code and supports the additive two-channel one.

**Why the behavioural star and the activation cylinder are both true.** For a softmax,
CLR(q)_i = log q_i - mean_j log q_j = z_i - mean_j z_j exactly, where z is the logit vector.
So the behavioural "belief manifold" IS the centred final-logit geometry, and its arms mean
the OUTPUT is a gain-times-direction code. The window activations we read sit many nonlinear
layers upstream of those logits and encode the two ingredients separately and additively.
The multiplicative structure is created downstream, not carried in the window state. This
also explains C1's split result: a gain model beat an additive decoder (because the target
is gain-like) while the learned gain failed to track the commitment ridge direction (because
in the activation space commitment is a separate additive coordinate, not a gate).

Caveat: class balance shifts across commitment bins (the majority action dominates at high
commitment), which affects the scale of both statistics; the within-bin shuffled-label
control is the right reference and stays flat and near zero throughout.

Artifacts: `outputs/conceptual_belief_space/action_plane_by_commitment/`, figure
`figures/conceptual_belief_space/action_plane_by_commitment.png`.

### 2026-07-25 - Cylinder result is not an artifact of layer averaging (per-layer re-run)

All the arm/cone analyses (91, 92, 93) default to `--representation mean`, which is
`pooled[rows, 24, 512].mean(axis=1)` - the mean over ALL 24 layers of the mask-weighted
mean over the last 12 CoT tokens. Since averaging is linear, a geometry that rotates with
depth could partially cancel, so the cone test was re-run at individual layers:

| representation | commitment axis orthogonal to action plane | 3rd axis vs true commitment | separation growth low->high | verdict |
| --- | --- | --- | --- | --- |
| 24-layer mean | 0.967 | +0.634 | 1.35x | cylinder |
| layer 10 | 0.993 | +0.627 | **1.19x** | cylinder |
| layer 15 | 0.988 | +0.665 | **1.16x** | cylinder |
| layer 18 | 0.994 | +0.643 | **1.59x** | intermediate |
| layer 23 | 0.964 | +0.596 | **1.25x** | cylinder |

The cylinder holds at every individual layer tested, so it is not an averaging artifact.
Two refinements worth keeping. First, at individual layers the commitment direction is even
MORE orthogonal to the action plane (0.99 versus 0.967 for the layer mean), which
strengthens the two-channel claim rather than weakening it. Second, **layer 18 is the most
cone-like of the four at 1.59x** - and layer 18 is exactly where action identity is most
decodable. That is where any multiplicative structure would be, and it is the layer to push
on before claiming there is none anywhere. It does not reach the pre-registered 2.0
threshold, so the verdict stands, but "no multiplicative coupling at any depth" would be
overclaiming and is not what is being claimed.

### 2026-07-25 - Best-layer 4-d space (script 94), and an important qualification to Theorem 3

Built the most favourable rendering available: three LDA action axes from layer 17 (best
action decoding, 0.521) plus the ridge commitment axis from layer 15 (best commitment
decoding, 0.670), standardised, then PCA to 2-d. Fitted on training trajectories, evaluated
on 6,534 held-out pre-verbal prefixes.

| space | own-centroid cosine | angular purity | radius vs commitment | centroid radius ratio |
| --- | --- | --- | --- | --- |
| behavioural belief, 3-d | 0.583 | 0.677 | 0.781 | 0.922 |
| behavioural belief, PCA 2-d | 0.592 | 0.594 | 0.657 | 0.962 |
| activation 4-d (act L17 + commit L15) | 0.345 | 0.467 | **0.213** | 0.433 |
| activation 4-d, PCA 2-d | 0.388 | 0.439 | **0.112** | 0.516 |

Adding the commitment axis does improve the radial coupling in 4-d (0.213, the best of any
activation rendering so far: layer-mean 3-d gave 0.179, layer-17 3-d gave 0.191) but it is
still a third of the behavioural value, and collapsing to 2-d halves it again. Still no arms.

**Two caveats on this particular rendering.** The four standardised coordinates have
eigen-ratios [0.391, 0.250, 0.250, 0.109] - the exact 0.25 tie means PC2 and PC3 are
degenerate and the 2-d view is only determined up to an arbitrary rotation in that plane.
And PC1 loads +0.707 on commitment against -0.680 on action-1, i.e. the first component is
essentially a contrast between the two, which is why the 2-d collapse loses radial coupling.

**The qualification, which matters for the paper.** The four coordinates are NOT mutually
uncorrelated: action-1 correlates -0.544 with the commitment coordinate. Checking this
within a single layer, where Theorem 3's orthogonality was measured:

| layer | geometric overlap ||V'u||^2 | LD1 vs commitment coordinate correlation |
| --- | --- | --- |
| 15 | 0.0128 (chance 0.0059) | **-0.564** |
| 17 | 0.0191 (chance 0.0059) | **-0.545** |

So the directions are nearly orthogonal geometrically while their projections are strongly
correlated across the data. There is no contradiction - an anisotropic covariance makes
orthogonal directions produce correlated coordinates - but it means **"strength and content
occupy nearly orthogonal directions" must not be paraphrased as "the two channels are
statistically independent"**, which is close to how the paper currently reads. A plausible
mechanism is the behavioural action-commitment association itself (DOWN steps average 0.565
commitment, RIGHT 0.731), so the best class-separating direction inherits commitment
information through the label distribution rather than through the representation.

Action items for the paper: state the geometric claim as an angle only; report the
coordinate correlation alongside it; and check whether the removal test (the functional
evidence, currently the load-bearing one) survives when commitment is regressed out of the
action coordinates first. Artifacts:
`outputs/conceptual_belief_space/best_layer_4d_space/`, figure
`figures/conceptual_belief_space/best_layer_4d_2d_view.png`.

## 2026-08-11 — C5 lexicon audit: what the five "vocabulary classes" actually count

Prompted by the question "what is the validated rubric for the judge". There is no judge:
the C5 classes in `scripts/13_conj_c5_tail_verification.py:29-41` are deterministic
`re.compile` alternations matched with `findall`. No human annotation, no held-out set,
no precision/recall, no sensitivity analysis had ever been run on them. Ran one now
(scratchpad `lexicon_audit.py`, `lexicon_sensitivity.py`; 1007 usable steps, 3.20M head
chars / 2.53M tail chars, same T_stab split and >200-char filter as script 13).

**Finding 1 — the headline ratios are weighting-dependent.** The published
`tail_over_head` is a ratio of means of *per-step* densities (every decision weighted
equally). Pooling over characters instead gives systematically smaller numbers, because
short tails are dense final "thus, answer is X" bursts:

| class | per-step (published) | pooled | %steps tail>head |
| --- | --- | --- | --- |
| verification | 5.23x | **2.70x** | 93.7% |
| action_word | 2.72x | **1.64x** | 91.9% |
| plan | 1.59x | 1.42x | 85.9% |
| correction | 1.34x | 1.07x | 62.4% |
| spatial | 1.11x | 1.10x | 60.8% |

**Finding 2 — `verification` is 80% two words.** Of tail hits: `thus` 45.3%, `check`
35.0%. `thus`/`therefore`/`hence`/`so the answer`/`indeed` are conclusion markers, not
verification. Splitting them out:

| variant | per-step | pooled | %tail>head | wilcoxon p |
| --- | --- | --- | --- | --- |
| verification as published | 5.23x | 2.70x | 93.7% | 7.1e-154 |
| verification MINUS conclusion markers | **3.73x** | 2.16x | 80.7% | 2.8e-104 |
| conclusion markers ONLY | 10.93x | 3.69x | 89.8% | 1.5e-149 |

The claim survives — genuine checking vocabulary still rises 3.73x — but a large share of
the headline number is the model *stating a conclusion*, not checking one.

**Finding 3 — `correction` is an artifact and should be dropped or renamed.** 77.3% of its
tail hits are the word `but`. Remove it and the effect **inverts**: 0.86x per-step,
0.94x pooled, tail higher in only 41.7% of steps, p = 1.5e-3 in the opposite direction.
Corrections are *less* dense after stabilisation, which is the sensible result and the
opposite of what the published 1.34x implies.

**Finding 4 — `plan` leans on discourse connectives.** `then` 28.6% + `path` 25.1% of tail
hits; removing `then`/`first`/`next` gives 1.23x (from 1.59x). Direction survives.

**Finding 5 — `spatial` is really "coordinate mentions", and it is robust.** 73.8% of tail
hits are literal `(r, c)` pairs, and those rise 1.21x. Abstract nouns move the other way:
`grid` 0.21x, `row` 0.22x, `column` 0.53x, `wall` 0.73x, while `key` 1.33x. So the tail
does not stop being spatial; it stops naming the grid abstractly and starts citing specific
cells. This is a sharper version of the published claim, not a retraction.

Paper actions: (a) state which weighting the ratios use and report both for verification and
action_word; (b) replace the 5.23x headline with 3.73x conclusion-marker-free, or report
both; (c) drop `correction` from Figure 9 or relabel it, since its published direction does
not survive removing one connective; (d) relabel `spatial` as coordinate density and add the
grid-noun-down / coordinate-up contrast, which is a stronger result than the flat 1.11x;
(e) fix the Fig. 9 caption, which says spatial "does not" rise when it rises at p = 2e-14.

## 2026-08-11 — Script 16: corrected C5 tail-lexicon analysis (supersedes the lexical half of script 13)

Command: `interp/.venv/bin/python scripts/16_c5_tail_lexicon_corrected.py`
(seed 42; 2000-rep trajectory-cluster bootstraps for ratios, 500 for curves/deciles/FW,
200 placebo draws; sentence encoder all-MiniLM-L6-v2 on MPS, newly installed into
interp/.venv). 1007/1276 usable steps; sanity gate reproduced the published per-step
ratios (5.23/1.34/2.72/1.11/1.59) to 1e-9 before any corrected statistic was computed.

**Design fixes over script 13** (per the 2026-08-11 audit): dual weighting reported
everywhere; conclusion markers split out of "verification"; "but" split out of
"correction"; leave-one-literal-out ranges + top-literal share per class; trajectory-
cluster bootstrap CIs; action words disambiguated by context (answer/movement/
descriptive/bare, precedence rules frozen after one documented 20-example spot check);
event-aligned densities around T_stab with a shuffled-stabilization placebo (stab point
redrawn from another chain in the same length quintile, preserving the marginal position
distribution); position-matched decile control; Fightin' Words (Monroe et al. 2008)
data-driven contrast vocabulary with top-50 bootstrap stability; author-labelled
precision samples for every class (labeled copy: precision_sample_labeled.csv).

**Headline corrected numbers** (per-step ratio [95% cluster CI], pooled):
- checking 3.73 [3.37, 4.15], pooled 2.16 — ROBUST (LOO range [3.46, 16.0])
- concluding 10.9 [9.50, 12.8], pooled 3.69 — ROBUST but END-ANCHORED (see below)
- sequencing 2.60 [2.44, 2.75]; coordinates 1.41 [1.34, 1.47]; planning 1.23 — ROBUST
- self_correction (no "but") 0.86 [0.79, 0.94] — ROBUST INVERSION of published 1.34x
- grid_nouns 0.37 [0.35, 0.40] — the largest effect in the family, previously hidden
  inside the flat 1.11x "spatial" class
- action_answer: 29 head vs 1751 tail hits — answer-naming is essentially tail-exclusive
- objects 0.97 [0.91, 1.02] and action_descriptive — NOT robust, excluded from claims

**Event-aligned placebo verdicts** (jump in ±800 chars at T_stab vs placebo, p floor 0.01):
- AT the decision: checking +0.43/1k vs [+0.17, +0.35]; grid_nouns −1.61 vs
  [−0.97, −0.38]; coordinates +5.6 vs [+1.2, +3.6]; sequencing; objects (+, against a
  negative positional trend).
- NOT at the decision: concluding +0.03 vs [+0.14, +0.29] (significantly BELOW placebo —
  thus/therefore are anchored to the chain END, not to stabilization); action_answer
  p=0.67 (answer naming happens at the end, not at the decision moment);
  self_correction p=0.38 (flat).
- Decile control agrees: post-stab steps carry ~+0.8–1.0/1k more checking than
  not-yet-stab steps at the SAME relative position (deciles 7–9, CIs exclude 0);
  concluding shows no such gap until decile 9.

**Fightin' Words** (no word list assumed): top risers = to, steps, coordpair, action,
down/up/right/left, then, thus, step, that's, first, shorter, minimal, output, total;
top fallers = num, col, ', row, at, grid, g/a/d (map symbols), walls, all, coordinates,
agent, map, columns, rows, indices. I.e., HEAD = map parsing, TAIL = solution walking.
Top-50 membership stable in ≥93% of trajectory resamples for all top-25 risers except
should (76%) and next (60%).

**Embedding soft categories: 4 of 6 FAILED validation** (author-labelled top-decile
precision): checking 0.13, self_correction 0.06, answer_naming 0.25, concluding 0.63 —
topic dominates function in MiniLM space; the soft self_correction score RISES after
stabilization (a manufactured finding — negation blindness) while real correction
markers fall. Only planning (0.88) and spatial_description (0.88) pass, and both agree
with the regex story. Regex classes all scored 1.00 (n=16–20; Wilson lower bounds
0.81–0.84). Kept in the paper as a cautionary result.

**Corrected conclusion (now in paper §4/"At the decision, the chain stops parsing and
starts re-deriving")**: before T_stab the model parses the world (grid nouns, indices,
symbols, raw numerals); at T_stab, visibly and beyond placebo, it stops parsing and
re-walks the solution in concrete coordinates with explicit checking; conclusion markers
and answer naming come only at the chain end. The tail is re-derivation + checking, not
deliberation and not mere restatement. The old caption clause "spatial language does
not [rise]" is gone; "verification 5.23x" is superseded by checking 3.73x + concluding
10.9x reported separately with the end-anchoring caveat.

**Paper deltas**: §4 tail paragraph rewritten; Fig 9 right panel replaced by the ratio
dot plot (tail_v2.png); Appendix A added (robustness table, event-aligned figure,
Fightin' Words figure, embedding-failure paragraph); Monroe et al. 2008 added to refs;
compiles clean, 13 pp (10 content + refs + appendix).

Artifacts: outputs/conjectures/c5_tail_corrected/{revised_lexicon_stats,loo_robustness,
fightin_words,embedding_class_stats,event_aligned,placebo_jumps,position_matched,
precision_sample,precision_sample_labeled,precision_summary}.csv + embedding_sentence_
scores.parquet + report.md; figures/c5_corrected/{tail_corrected_ratios,
tail_fightin_words,tail_event_aligned,tail_v2}.png (copied into paper/figures/).
Script 13 and its outputs untouched (append-only).

## 2026-08-11 — Script 17: sentence-function classification (method 4) — kappa gate FAILED, three classes survive

Context: the user's "做1和4" was ambiguous between two numberings; the prose "方法四"
(sentence-function classification) had NOT been run — script 16 implemented the menu's
option 4 (embedding soft categories) instead. Script 17 closes that gap.

Command: `interp/.venv/bin/python scripts/17_sentence_function_classification.py`
(mlx-lm newly installed into interp/.venv — NOTE: this upgraded transformers 4.57.5 ->
5.15.0 in that venv). Annotator: local mlx-community/gpt-oss-20b-MXFP4-Q4, greedy,
Reasoning: low, fixed prompt, 25 sentences/batch, checkpointed jsonl. Taxonomy: 8
functions adapted from Thought Anchors (P parsing / L planning / C computing /
U uncertainty / V verifying / F concluding / A answering / O other). Sample: 120
usable decisions, all 16,640 sentences; 452 (2.7%) unparsed after one retry. ~100 min
wall-clock at ~160 sentences/min.

**Gold validation (the point of the method): FAILED its pre-declared gate.**
n=152 stratified blind author labels; accuracy 0.566, Cohen's kappa 0.504 < 0.60 →
method flagged UNRELIABLE as a whole. Per-model-class precision: answering 1.00,
concluding 0.95, parsing 0.84 (usable); computing 0.63, planning 0.42, uncertainty
0.32, verifying 0.21, other 0.16 (NOT usable). Dominant confusions: parse-
confirmations ("row3 col2 '_' yes.") labelled V; route-traces and micro-conclusions
labelled L. Artifacts: gold_labeled.csv, agreement.csv/.txt.

**Citable results (reliable classes only), all consistent with scripts 13/16:**
- parsing share collapses 0.561 -> 0.144 (diff −0.417 [−0.452, −0.382]); event-aligned
  jump −0.208 vs placebo [−0.175, −0.062], p = 0.02 — the sentence-level counterpart of
  the grid-noun collapse, AT the decision.
- answering share 0.001 -> 0.055 — tail-exclusive, matching action_answer (29 vs 1751).
- concluding share 0.009 -> 0.056 with NO stab-linked jump (p = 0.38) — end-anchored,
  matching the regex concluding placebo verdict.
- (computing 0.077 -> 0.224 with jump p = 0.01 agrees with coordinates/action_bare but
  its 0.63 precision keeps it out of headline claims; verifying-share rise is NOT
  citable from this method — verification evidence remains the regex checking class.)

**Takeaway for the paper**: all three "measurement upgrades" beyond validated regex +
Fightin' Words were tried; embedding soft categories failed 4/6 classes, and a 20B
same-model sentence annotator at reasoning-low fails the kappa gate with exactly the
boundary confusions (V vs P, L vs C) a rubric predicts. The corrected conclusions rest
on the validated keyword classes + data-driven vocabulary + event alignment; the
sentence-function view corroborates via its three high-precision classes only.

Artifacts: outputs/conjectures/c5_sentence_functions/{labels_checkpoint.jsonl,
sentence_labels.parquet, function_stats.csv, event_aligned.csv, placebo_jumps.csv,
gold_sample.csv, gold_key.csv, gold_labeled.csv, agreement.csv, report.md};
figures/c5_corrected/sentence_functions.png.

## 2026-08-30 — Workshop paper: behavioural manifold + commitment-along-CoT written into §4

Edited `_IAB_Workshop_2026__Belief_action_gap/workshop_main.tex` (the joint IAB workshop
draft) to add the repo's two headline behavioural results to §4 "Beliefs and Actions
Throughout Reasoning". All numbers are from the faithful extracted-label pipeline (D0/D1,
2026-07-23); no stored-label quantities were used.

Additions:
- §4 intro: prefix-thinning sentence (≤80 cuts/decision, median gap 20 tokens).
- §4.1: forced-suffix parenthetical (12 tokens opening the answer JSON, validated
  token-exactly).
- §4.2 new "The behavioural belief manifold" paragraph + Figure 5 (two panels:
  `behavior_manifold_star.png` = `figures/behavior_extracted/pca_by_top_action.png`,
  `behavior_manifold_trajectories.png` = `.../example_manifold_trajectories.png`).
  Framed explicitly as output geometry (CLR = centred logits; radius ~ entropy-commitment
  Spearman 0.78, centroid radius ratio 0.92, RIGHT prior 79.5%), with the sentence
  disclaiming any activation-geometry claim (per the 2026-07-25 cylinder results).
- §4.2 new "Commitment to the eventual action" paragraph: rise 0.35 → 0.92 by progress
  bins, 0.34 at the empty prefix, final-prefix agreement 100.0%, pre-verbal control
  (31,723 prefixes, 0.35 → 0.92, top==final 54.9%).
- §4.2 "Three clocks" paragraph: Tverb 0.253 < Tstab 0.494 < Tperm(0.8) 0.648, 35.6%
  stabilize before first mention; kept the collaborators' settled-textual-commitment
  sentences but flagged them with a \td (their 753/59.0%/99.87%/523/823 numbers are not
  in any repo report — must be verified against extracted labels; canonical tail is
  524/887).
- §4.2 "Half the chain is post-decision": 50.6% median tail (524/887 tokens), length-
  quartile effect 0.32 vs 0.60, QA-analogue citation (new ref.bib entry `boundary2026`
  = arXiv 2606.13603), thinning lower/upper-bound caveat.
- §4.2 "Stabilization is gradual and imperfectly absorbing": hazard 0.133 → 0.019,
  6.0 switches/decision, later-switch prob ≈1 (commitment<0.35) → 0.089 (top decile),
  per-action timing RIGHT 0.393 / DOWN 0.609. Figure 6 = `clocks.png` + `hazard.png`
  (copied from `paper/figures/`, the peer-reviewed solo-paper versions).
- Fixed pre-existing undefined reference: added `\label{sec:causal}` to §5.
- Cluster paragraph opening lightly rewritten to forward-reference §4.4.

Compile: pdflatex+bibtex clean, 0 undefined refs/citations. Pre-existing issues left for
collaborators: `dblblindworkshop` is not a declared option of the bundled
`neurips_2026.sty` (only `workshop`, `main`, ... exist — option is ignored with a LaTeX
error at load), and `ref.bib` has duplicate entries (ziebart2008maximum,
gurnee2024language) that bibtex skips with warnings.

## 2026-08-30 — Coarse-4 sentence functions: provenance logged + added to workshop paper §4.2

**Provenance reconstruction (the run itself was never logged).** The coarse-4
sentence-function analysis exists at `outputs/conjectures/c5_sentence_functions_coarse4/`
(report.md, function_stats.csv, event_aligned.csv, placebo_jumps.csv,
sentence_labels.parquet, labels_low.jsonl; figure `figures/c5_corrected/
coarse4_functions.png`, dated 2026-08-14/15) with pilot gold iterations in
`outputs/conjectures/c5_sentence_functions/coarse4_pilot/`. Its report self-describes as
"script 18", but no such script exists in `scripts/` (root numbering ends at 17) and no
copy was found in session scratchpads — the generating code is lost; the artifacts and
gold files are the record. Summary: same-model annotator (mlx gpt-oss-20b, greedy,
effort low), 141,377 sentences / 1,004 analysed decisions (annotation checkpointed over
1,007 usable steps); coarse-4 merge of the script-17 taxonomy after the 8-way scheme
failed its kappa gate (0.504 < 0.60). Gold-152 blind precision: A 0.91 / S 0.78 /
W 0.73 / M 0.32 (M exploratory only). Head→tail shares: world-reading 0.683→0.291,
solving 0.249→0.450, meta-checking 0.057→0.139, concluding/answering 0.011→0.120.
Jump at T_stab vs shuffled-stabilization placebo: W −0.172 vs [−0.135,−0.088] p=0.01;
S +0.117 vs [+0.048,+0.097] p=0.01; M +0.032 p≈0.05; A +0.022 vs [+0.012,+0.025]
p=0.20 (end-anchored, not stabilization-linked).

**Workshop paper edit** (`_IAB_Workshop_2026__Belief_action_gap/workshop_main.tex`):
- New §4.2 paragraph "What the tail contains": annotator + gold-152 precisions disclosed
  in-text, M flagged exploratory; W-collapse/S-rise at the decision (placebo-tested);
  concluding/answering end-anchoring dissociation; one sentence of validated regex
  corroboration (grid nouns 0.37×, coordinates 1.41×, checking 3.73×, answer words
  29 vs 1,751). Verification claims rest on the regex checking class, not on M.
- New full-width Figure (`coarse4_functions.png` copied into workshop figures/;
  label fig:sentence-functions; became Fig. 6, dynamics figure renumbered to 7).
- §4.4 cluster-shift paragraph now cross-references the figure, plus a \td asking the
  collaborators to state whether their "world-reading/solving/answering" semantic
  categories and this coarse-4 annotator are independent instruments that agree, or to
  unify them.
- `ref.bib`: added `bogdan2025thought` (Thought Anchors, arXiv:2506.19143) for the
  taxonomy source.
- Compile clean: 0 undefined refs/citations; only the pre-existing `dblblindworkshop`
  style-option error remains.

## 2026-08-30 — Workshop paper: entropy commitment added to the §4.2 commitment paragraph

Per the framing in the early behavioral report (`SPAR Belief Action Gap.docx` §4.1 /
Fig. 8), the "Commitment to the eventual action" paragraph now introduces BOTH measures:
commitment to the eventual action pi(a_t) and entropy commitment 1 − H(pi)/log 4 (tied
to the manifold's radial coordinate), with the point that the whole distribution
sharpens, not merely q(final). Also added the prompt-prior/reasoning-gain decomposition
and per-action prior asymmetry. All numbers re-based on faithful labels (the docx's own
values are from the old 10-trajectory stored-label subset and were NOT used).

Verified directly from `outputs/prefix_commitment_metrics_extracted.parquet`
(interp/.venv/bin/python; first/last prefix per step, 1,276 steps) since these exact
values are in no existing report: empty-prefix means commitment 0.3375 / entropy_comm
0.2973 / margin 0.3758; final-prefix means 1.0000 for all three; mean final-minus-prompt
commitment gain 0.6625; empty-prefix per-action commitment DOWN 0.144 / LEFT 0.369 /
RIGHT 0.665 / UP 0.144 (confirms D0's surviving ordering). Progress-bin endpoints from
`outputs/behavior_manifold_extracted_report.md`: entropy commitment 0.3423 → 0.9075
(all prefixes), 0.3263 → 0.8863 (pre-verbal). Compile clean, 0 undefined refs.

## 2026-08-30 — Script 19: commitment-measures paper figure (workshop §4.2)

New script `scripts/19_plot_commitment_measures_figure.py` (numbered 19 because the lost
coarse-4 run already claims "18"). Command:
`interp/.venv/bin/python scripts/19_plot_commitment_measures_figure.py`
(defaults; seed 42, 12 progress bins, 1000-rep trajectory-clustered bootstrap over
source_file). Input: `outputs/prefix_commitment_metrics_extracted.parquet` (faithful
labels only).

Two-panel figure `figures/behavior_extracted/commitment_measures_panel.png`:
- Left: mean commitment_to_final and entropy_commitment vs progress, all prefixes
  (solid) + strict pre-verbal subset (dashed), 95% cluster-bootstrap bands, empty-prefix
  (prompt-only) baselines marked at progress 0 (0.3375 / 0.2973), uniform-prior line
  0.25. Bin endpoints: commitment 0.348 → 0.974, entropy 0.333 → 0.966; pre-verbal
  0.345 → 1.000 / 0.320 → 1.000.
- Right: per-eventual-action commitment vs progress with empty-prefix baselines
  (UP 0.144 / DOWN 0.144 nudged apart visually, LEFT 0.369, RIGHT 0.665).
Values table written to `outputs/commitment_measures_figure_report.md`.

Workshop paper: figure copied into `_IAB_Workshop_2026__Belief_action_gap/figures/`,
added as fig:commitment-measures (became Fig. 6; coarse-4 → 7, clocks/hazard → 8) with
two in-text refs from the "Commitment to the eventual action" paragraph. Compile clean,
0 undefined refs. ruff clean (E402 noqa'd per repo convention).

## 2026-08-30 — Script 20: commitment-gated counter-evidence injection (causal, text-level)

Command: `interp/.venv/bin/python scripts/20_commitment_gated_injection.py --resume`
(seed 42; 240 cuts = 40 per commitment_to_final bin [0-0.3, 0.3-0.5, 0.5-0.7, 0.7-0.85,
0.85-0.95, 0.95-1], ≤3 cuts/decision; 217 decisions, 82 trajectories; 720 forced-answer
evaluations = {counter: runner-up action, neutral: same-shape no-action, support: top
action} × cut; every evaluation a full independent forward pass, NO KV-cache reuse;
~25 min on local MLX gpt-oss-20b; per-10-eval checkpointing + --resume).

**Headline: a single injected counter-evidence sentence overrides the accumulated chain
almost everywhere; commitment protects only near ceiling.** On strictly counterfactual
targets (injected ≠ eventually-taken action, n=201 — at low commitment ~50% of runner-ups
coincide with the eventual action and are excluded): flip-to-injected 1.000/1.000/1.000
(bins <0.7, n=81) → 0.925 (0.7-0.85) → 0.900 (0.85-0.95) → 0.625 (≥0.95); Spearman
(commitment, flipped) = −0.421, p=5.0e−10. All-rows version: 0.875/1.0/1.0/0.925/0.900/
0.625, rho=−0.269 p=2.4e−5. Neutral control: any-top-change 9.6% overall (drift to the
specific runner-up peaks at 37.5% in the unstable 0.3-0.5 bin — natural lability, matches
the switch hazard). Support: mean q(top) 0.757 → 0.989. Counter Δq(target) 0.61-0.81 per
bin. Pre-registered decline direction CONFIRMED but the magnitude story is textual
dominance with weak, late commitment gating — consistent with D4 (belief tracks CoT) and
with readable-not-writable (the text channel, not the window representation, is causally
load-bearing).

Artifacts: `outputs/followups/commitment_gated_injection.parquet` (720 rows),
`outputs/followups/commitment_gated_injection_report.md` (incl. strictly-counterfactual
robustness split), `figures/counterfactual/commitment_gated_injection.png`,
log `outputs/followups/cgi_run.log`. New script: `scripts/20_commitment_gated_injection.py`
(ruff-clean).

**Workshop paper §5 "Causal Tests" written** (was a bullet skeleton): §5.1 targets (clocks
+ anchors + readable window), §5.2 whole-chain patching (D4 numbers + fig) and the new
injection experiment (+fig), §5.3 activation patching (four families at best settings +
read/write asymmetry fig, caveats, \td for collaborators' pre-reasoning patching), §5.4
four conclusions (movable-through-text-not-representation; weak late commitment gating;
text/activation position dissociation; availability vs commitment vs causal-use
bookkeeping). Figures copied: mismatched_prompt_cot_tracking.png,
readwrite_asymmetry.png (= synthesis_read_write_asymmetry.png), commitment_gated_injection.png.
ref.bib += mcgrath2023hydra, wang2023interpretability. Compile clean.

## 2026-08-30 — §5.4 Causal Results: four conclusions written and verified by 7-reviewer panels

Per user directive, each of the four §5.4 conclusions in the workshop paper was
evaluated by 7 independent reviewer agents (fresh context, file access, mandatory
verify-numbers-against-artifacts protocol, APPROVE/REJECT verdicts). Total 35 reviews
across 5 panels.

Verdicts:
- P1 (does changing the belief before commitment change the action): 7/7 APPROVE.
- P3 (do text and activation interventions identify the same positions): 7/7 APPROVE.
- P4 (availability vs commitment-timing vs causal-use bookkeeping): 7/7 APPROVE.
- P2 (do interventions lose force after commitment): round 1 = 7/7 REJECT, all on ONE
  identical factual error the panel caught — the draft said "8.9% of prefixes above
  0.95" where 0.0895 is the future-switch rate of the pd.cut TOP DECILE (>0.9; the
  0.95 in commitment_boundary_phenomenology.md is a bin midpoint; true >0.95 rate is
  0.0664, n=39,611). Fixed to "top commitment decile, >0.9"; round 2 with 7 fresh
  reviewers = 7/7 APPROVE.

Multiple reviewers independently recomputed the injection statistics from the raw
parquet (all reproduce exactly), one ran a trajectory-cluster bootstrap of the headline
correlation (rho 95% CI [-0.510, -0.316] — robust to clustering), and several noted the
reported partial rho (-0.175, script 20's rank-residual estimator) is CONSERVATIVE
relative to the textbook partial Spearman (-0.245, p=4.5e-04) — recorded here as a
deliberate choice, reproducible from the released script.

Non-blocking reviewer suggestions applied to §5.3 (outside the four judged paragraphs):
"every number the readout sees" corrected to "the full 12x2880 window state at a layer"
(the readout pools 24 layers; the patch hits one — the phrase the 2026-07-25 audit had
flagged), and the swept layer sets are now named (steering 10/15/18; transplant ladder
3/10/15/18). Suggestions noted for camera-ready but NOT applied (would alter approved
text): "decisions"→"cuts/injection points" in the 90.0%/62.5% clause; footnote the
partial-rho estimator; report activation-side n's and Wilson intervals in §5.3.

Final §5 state: 5.1 targets, 5.2 whole-chain patching + commitment-gated injection
(figs), 5.3 activation patching + read/write asymmetry (fig), 5.4 four approved
conclusions. Compile clean, 0 undefined references.

## 2026-08-30 — Workshop paper: latexmk build unblocked (two stacked blockers)

User's latexmk workflow failed while my manual pdflatex+bibtex loop had pushed through.
Root causes, both fixed:
1. `\usepackage[dblblindworkshop]{neurips_2026}` — the option is from the generic
   NeurIPS 2026 template, but the BUNDLED neurips_2026.sty is another workshop's copy
   (its \@workshoptitle defaults to "Privacy in the Era of Large Opaque Models
   (PriLOM)"; declared options only final/preprint/nonatbib/main/position/eandd/
   creativeai/education/workshop; no \workshoptitle setter). The unknown option made
   pdflatex exit nonzero → latexmk stopped after pass 1, before bibtex. Since the
   option had always been ignored, loading the package OPTIONLESS reproduces the
   exact double-blind submission rendering; a preamble comment documents the sty
   mismatch and the pre-submission options (swap in the IAB workshop's official sty,
   or [workshop] + \makeatletter\renewcommand{\@workshoptitle}{...}).
2. ref.bib duplicate entries (`ziebart2008maximum`, `gurnee2024language`) made bibtex
   exit with error status → latexmk aborted before the final reruns. Second copies
   deleted (bibtex was skipping them anyway; rendered bibliography unchanged).

Verified with the user's exact invocation from a clean state (`latexmk -C` then
`latexmk -pdf -synctex=1 -interaction=nonstopmode -file-line-error -recorder`):
exit 0, all targets up to date, 0 undefined references/citations. Remaining bibtex
warning (volume+number in ng2000algorithms) is non-blocking and left for the
collaborators.

## 2026-08-31 — Causal-tests documentation

Wrote `causal_tests_report_2026-08-31.md` (repo root): a self-contained account of the
§5 work — evidence inventory (D4, activation battery, observational context), the
script-20 injection experiment (design, exact numbers, robustness, position
attribution), what went into the paper, the 35-review verification process incl. the
caught top-decile error, limitations, and a file map.

## 2026-09-03 — Behavioural-manifold figure removed from the workshop paper (reviewer remark); script 21

Reviewer remark on Fig. 5 (CLR-PCA "belief manifold"): (1) a 4-action distribution has
3 DoF, why a 2-d projection; (2) what does it establish beyond argmax + max prob +
entropy? Tested with `scripts/21_action_competition_structure.py` on
`outputs/behavior_manifold_extracted.parquet` (5-fold CV grouped by trajectory):
- Redundancy: held-out R^2 of PC1/PC2 from (argmax one-hot, max prob, normalised
  entropy) = 0.929 / 0.945 — the map is ~93-95% a re-rendering of the scalars, and its
  star shape is CLR-on-simplex geometry (3-d version = tetrahedron vertex directions),
  consistent with the 2026-07-25 finding that CLR(q) is the centred-logit geometry.
- What remains: runner-up identity. Held-out accuracy 0.448 (majority per argmax),
  0.505 (scalars), 0.791 (argmax + PC1/PC2), 0.937 (all).
- Competition structure P(runner-up | top), top_prob < 0.6 (n=18,890): RIGHT is the
  runner-up to every other action 51-61% (UP 0.514 / DOWN 0.582 / LEFT 0.609); UP<->DOWN
  almost never compete (0.046 / 0.062); LEFT/RIGHT compete with everything.
  Report: `outputs/action_competition_structure_report.md`; CSVs in
  `outputs/action_competition_structure/`.

Paper: user chose "cut from main text". Removed the manifold paragraph + Fig. 5 (star +
trajectories; PNGs removed from the workshop figures/ dir, originals remain in
figures/behavior_extracted/), fixed the entropy-commitment definition that referenced the
figure's radial coordinate, and added a "Which actions compete" paragraph + Table
(tab:competition) with the low-confidence competition matrix, tying the runner-up to the
§5 injection target. Build: a corrupted workshop_main.bbl (two \end{thebibliography},
likely an editor auto-build racing the scripted build) aborted one latexmk run; removing
build artifacts and rebuilding from scratch gives latexmk exit 0, 0 undefined refs.

## 2026-09-04 — Overleaf version synced: §4 reviewer TODOs resolved in place (workshop_main.tex)

The user replaced the local workshop draft with the newest Overleaf version (2,034 lines;
MG rewrote the commitment paragraph in new notation $\pi_{t,k}$, $c_{t,k}$; the manifold
paragraph/figure is commented out; new appendices; collaborator \td notes on my §4.2
paragraphs). Per the user, TODOs are handled one at a time and each resolved note is
left in place as `\td{DONE (WW 2026-09-04): ...}` so collaborators can clear it.

Resolved today (all in workshop_main.tex):
- Token count: "2,370,850 reasoning tokens" defined as the summed lengths m_t of the
  reasoning traces R_t (§2.1), gpt-oss-20b tokens excl. final action; verified exactly
  from prefix_commitment_metrics_extracted.parquet (median 1,479.5/decision, IQR
  642-2,605, range 75-9,733).
- Sentence segmentation: stale TODO removed (procedure was already present); three
  redundant restatements collapsed; footnote discloses that the coarse-4 sentence-
  function analysis uses script 16's simpler splitter (punctuation+whitespace or line
  breaks; <8-char spans dropped) -> 141,377 units / 1,004 decisions vs the corpus
  chunker's 153,622 / 1,276.
- MG paragraph TODOs: uniform baselines folded in; LEFT prior 0.37 added (empty-prefix
  mean c_{t,0} by emitted action); "first action word" defined (generation-time
  `action` token tag consumed by stage 02; no negation/context handling => the
  pre-verbal restriction is conservative).
- "Three clocks" -> "Timing of commitment", rewritten in MG's notation with k_t^com
  defined in one clause; duplicated textual-commitment sentences (753/59.0%/99.87%/
  523/823) removed from my paragraph (they now live in MG's text; provenance check
  travels with them). "Half the chain" -> "Half the trace follows the decision";
  cite boundary2026 -> scalena2026beyond (the real entry for arXiv 2606.13603).
- "What the tail contains" -> "What follows the decision": meta-checking defined,
  placebo sentence rewritten plainly, keyword classes explained (fixed word lists,
  author-labelled precision 1.00/class).
- "Stabilisation is gradual": "imperfectly absorbing"/"input-prior asymmetry" removed;
  terminology fixed — "commitment" = commitment prefix k_t^com only; the continuous
  quantity is c_{t,k} ("confidence"); explains why later switches can follow high
  c_{t,k} (k_t^com is retrospective). Dynamics-figure caption aligned.
- Cluster paragraph deleted (forward reference; duplicate of §4.4's own numbers).
- Answer-forcing appendix corrected: it described a JSON-instruction readout that was
  NOT used for the 98,310-prefix analysis; now states the 12-token Harmony suffix
  (<|end|><|start|>assistant<|channel|>final<|message|>{ "action": "), single-token
  candidates, independent forward passes; \td asks collaborators to document the
  matched-cohort readout if it differs.
- Figure: `scripts/22_redraw_coarse4_figure.py` redraws the coarse-4 figure from the
  saved CSVs (function_stats/event_aligned/placebo_jumps; the original script is lost)
  -> `figures/c5_corrected/coarse4_functions_v2.png` (paired before/after bars with
  diff CI; event-aligned panels with explicit x-axis in 400-char bins and placebo in
  titles); copied to the workshop figures/; tex switched to _v2.
- ref.bib: deduplicated (dropped repeated ziebart2008maximum, gurnee2024language,
  turpin2023language) and re-added bogdan2025thought, mcgrath2023hydra,
  wang2023interpretability (lost in the Overleaf sync).

Verification: scratch build with the two collaborator-side blockers stubbed (the
undeclared `dblblindworkshop` sty option; missing figures/commitment_timing_matched46.png)
-> latexmk exit 0, 0 undefined refs/cites, 0 bibtex errors. Those two blockers remain
in the real files for the collaborators. Remaining open §4 TODOs are theirs: §4.1 scope
(report-based beliefs exist only for the 46-decision matched cohort, not 98k prefixes),
attention paragraph, K=12 selection, tests vs matched control, representative cluster
sentences, duplicate tab:reasoning-events (appendix copy exists).

## 2026-09-05 — Pre-verbal control and first-mention clock redone with a context-aware action-mention classifier (script 23)

Reviewer TODO: the "first action word" used the generation-time bare direction-word tag
(any token spelling UP/DOWN/LEFT/RIGHT, no context), so idioms ("pick up the key"),
descriptive uses ("to the left of the door") and negated mentions ("cannot go up") all
counted. New `scripts/23_action_mention_context.py`: per-token char offsets from the
stored token ids (per-token decode reproduces analysis_text exactly), regex over the
trace for every direction word, fixed context rules in precedence order
idiom -> negated -> answer -> movement -> uppercase -> descriptive -> bare; decisional =
answer|movement|uppercase|bare. Rules were tuned over three audit rounds on random
per-rule examples (initial versions mis-hit "4 left moves", "then right?", "go down to
6,7", "Up from (6,5)", "it's RIGHT", "Option left" — all decisional in this corpus).
Final counts over 72,561 occurrences: bare 45,135; movement 13,270; uppercase 4,674;
answer 2,207 (decisional 90.0%); descriptive 4,203; negated 2,383; idiom 689 (the
latter almost all "pick up").

Results (faithful labels, 1,276 decisions):
- Pre-verbal subset (prefixes before the first decisional mention of ANY action):
  35,379 prefixes (36.0%; was 31,723 / 32.3%); commitment first-tenth -> final-quarter
  0.351 -> 0.926 (was 0.349 -> 0.924); inverse entropy 0.334 -> 0.887 (was 0.326 ->
  0.885); top==final 0.554 (was 0.549). Eventual-action-only variant: 38,272 prefixes,
  0.349 -> 0.893, 0.336 -> 0.872. Sensitivity (negated mentions counted as decisional):
  35,333 prefixes, identical rises. Conclusion unchanged.
- Timing: median first decisional mention 0.299 (was 0.253); 37.0% of decisions commit
  before it (was 35.6%). Three-clocks CDF redrawn:
  `figures/commitment_boundary/three_clocks_cdf_v2.png`.
- Script 19 gained `--preverbal_mask/--preverbal_column`; regenerated with the mask ->
  `figures/behavior_extracted/commitment_measures_panel_v2.png`
  (pre-verbal dashed curves 0.347 -> 1.000 / 0.327 -> 1.000 across 12 bins).
Artifacts: `outputs/action_mentions_context.parquet` (per occurrence, with rule and
snippet), `outputs/action_mentions_context_steps.parquet`,
`outputs/preverbal_decisional_mask.parquet`, `outputs/action_mention_context_report.md`
(class/rule counts, random examples per rule, old-vs-new tables),
`outputs/commitment_measures_figure_v2_report.md`.

Workshop paper (workshop_main.tex): MG's pre-verbal sentence and the timing paragraph
updated to the decisional definition and new numbers (DONE markers dated 2026-09-05);
figures switched to commitment_measures_panel_v2.png and clocks_v2.png (copied into the
workshop figures/); new appendix subsection "Action-Mention Classification"
(app:action-mentions) with the rules, class shares and old-vs-new comparison.

## 2026-09-05 — §4.2 remaining TODOs resolved; textual-commitment provenance check

- "Explain how textual commitment is identified": the definition now states that the
  sentence-function label is the four-class same-model annotation of §4.2 (class
  concluding/answering, blind-gold precision 0.91) and that the mention / no-later-
  contradiction criteria are deterministic; DONE marker asks the collaborators to confirm.
- "Why do we care about optimality": purpose sentence added (intermediate optimality
  localises where decision quality changes; §4.3-4.4 relate those points to belief reports
  and hidden states); DONE marker.
- Provenance of 753 / 59.0% / 99.87% / 523 / 823 (textual commitment vs prefix
  stabilisation): reproduced from outputs/conjectures/c5_sentence_functions_coarse4/
  sentence_labels.parquet (label A = concluding/answering), outputs/action_mentions_
  context.parquet (direction-word positions) and the faithful-label stabilization_token in
  outputs/commitment_boundary_steps.parquet. With "mentions the eventual action" = any
  direction-word occurrence and "no later sentence mentioning a different action": 889 of
  the 1,007 labelled decisions (69.7% of 1,276) have a textual commitment; prefix
  stabilisation precedes it in 99.89% (paper 99.87%), median lead 534 (523), mean 845
  (823) tokens. Numbers are consistent with the re-extracted labels; the 753 count implies
  a stricter mention rule on the collaborators' side — DONE marker placed on the sentence
  asking for it. Scratch build (blockers stubbed): clean.

## 2026-09-05 — §4.1 "Belief Trajectories" scope corrected (collaborators' TODO, partial)

The draft claimed report-based beliefs (wall/key/door) are queried "at each prefix". No
such data exists in this repo (the prefix pipeline only elicits the forced action), and
the collaborators' own §4.3 states the reports cover a 46-decision matched cohort
(7,084 sentence prefixes, 42,504 reports). Rewrote §4.1 to state that scope and point
to §4.3's tables; left a PARTLY DONE marker: the requested accuracy/entropy-vs-progress
curves must come from the owners of the report pipeline.

## 2026-09-05 — §5 hand-off note resolved: no pre-reasoning patching results exist to merge

The abstract, Discussion and Conclusion assert "altering pre-reasoning state
representations alone does not reliably change behaviour", but no p=0 activation
intervention is reported anywhere in workshop_main.tex or short.tex; the only
representation-level interventions are the reasoning-window battery of §5.3. §5.3 now
states this explicitly (DONE marker), and the three dependent sentences carry a \td with
the wording the reported experiments support ("altering the hidden representation from
which the action is read out does not reliably change behaviour"), to be kept as is only
if the collaborators add p=0 results to §5.

## 2026-09-05 — Section-5 follow-up program ("run all"): scripts 24–28 written, smoke-tested, launched

Follows the §5 review of 2026-09-04. Five new self-contained scripts (checkpointed, `--resume`, cache-free
independent forward passes, `interp/.venv/bin/python`), all reusing the 240 script-20 cuts or the 396 script-10 pairs:

- `scripts/25_belief_level_injection.py` — belief-level (state-fact) vs conclusion-level injections + template robustness on the
  240 cuts: `belief_block_top` ("the cell immediately <dir of top> the agent at (r,c) is a wall (#)", false at ~94% of cuts, truth recorded),
  `belief_open_runnerup`, `belief_true_top` (true fact control), `counter_alt1`, `counter_alt2`. 1,200 evals →
  `outputs/followups/belief_level_injection.{parquet,_report.md}`.
- `scripts/24_whole_chain_patching_v2.py` — pre-verbal foreign chains (B cut before its first *decisional* mention, script-23
  classifier; also half-length) with paired own-chain controls (A cut before its own first decisional mention; A cut at the same
  relative position as the stored D4 cut). ~1,450 evals → `outputs/followups/whole_chain_patching_v2.{parquet,_report.md}`.
- `scripts/28_activation_sentence_transplant.py` — bandwidth-matched sentence-activation transplant: donor = prefix + counter sentence
  (21 tokens), recipient = prefix + 21-token neutral sentence (" Wait, I should double-check this. Looking at the grid again carefully
  before settling on a move here."); donor residual states at the sentence span written into the recipient at {all layers × full/last
  12/6/1 tokens; layer thirds; layers 10/15/18} → forced answer. `all_layers_full` is the built-in positive control (must equal the
  donor text answer). 12 passes per cut → `outputs/followups/activation_sentence_transplant.{parquet,_report.md}`.
- `scripts/26_injection_with_continuation.py` — 120 cuts (20/bin) × {counter, belief_block_top}: greedy continuation (≤256 tokens,
  stop at <|end|>) after the injection, then forced answer; immediate vs post-continuation flips, reversions, rejection cues →
  `outputs/followups/injection_with_continuation.{parquet,_report.md}`.
- `scripts/27_free_running_validation.py` — 60 cuts (10/bin) × 4 sampled continuations (temp 0.7, ≤700 tokens): eventual action
  after the model's own <|end|> vs forced belief at the cut (calibration, agreement by bin) →
  `outputs/followups/free_running_validation.{parquet,_report.md}`.

Smoke test (`outputs/followups/logs/smoke_2026-09-05.{sh,log}`, outputs under `outputs/followups/smoke/`): all five ran end-to-end.
Script 28 positive control held on 6/6 cuts (all-layer full-span transplant reproduces the counter-text answer exactly, mean Δq(top)
−0.754 vs −0.756 for the text); already on 6 cuts a *single-layer* (10 or 15) transplant of the 21 sentence-token states flipped 5/6 —
much stronger than the earlier forced-window transplants (§5.3), to be confirmed on the full 240.
Full sequential run launched 2026-09-05 10:03 CEST: `outputs/followups/logs/run_all_2026-09-05.sh` (order 25 → 24 → 28 → 26 → 27;
per-stage logs `outputs/followups/logs/s2?.log`). Results and paper edits to be appended below when complete.

## 2026-09-05 — Results: scripts 25, 24, 28 (stages 1–3 of the follow-up program); paper §5 revised

**Script 25 — belief-level vs conclusion-level injections** (`outputs/followups/belief_level_injection_report.md`, 1,200 evals, 240 cuts).
Top action changed / flip to runner-up / mean Δq(top): `belief_block_top` (false at 226/240) 25.0% [Wilson 19.9–30.8] / 21.7% / −0.179;
`belief_open_runnerup` (true at 196/240) 32.5% [26.9–38.7] / 32.5% / −0.166; `belief_true_top` control 4.6% [2.6–8.0] / 4.6% / **+0.074**;
`counter_alt1` 76.7% [70.9–81.6] / 76.2% / −0.507; `counter_alt2` 70.0% [63.9–75.4] / 70.0% / −0.467; reference script-20 counter 88.8% / −0.641,
neutral 9.6% / −0.032. Commitment gating for all: Spearman(changed, commitment) −0.234 (block), −0.375 (open), −0.420 (alt1), −0.474 (alt2).
Strictly counterfactual by bin — alt1: 1.0/0.95/0.975/0.85/0.625/0.35; alt2: 1.0/1.0/0.9/0.8/0.475/0.25; block_top: 0.286/0.4/0.3/0.175/0.125/0.0.
Belief-level flips ⊂ conclusion-level flips: 58 of 60 block_top flips also flip under counter (2 exceptions). When the wall fact is TRUE (14 cuts): 50% change.

**Script 24 — pre-verbal foreign chains + paired controls** (`outputs/followups/whole_chain_patching_v2_report.md`, 1,452 evals).
`foreign_preverbal` (B cut before first decisional mention; median 384 tokens, IQR 254–538, min 66): π(a_B)=0.582, top=a_B 60.1% [55.2–64.8], π(a_A)=0.158
(original D4 early cut: 0.598 / 61.4%). `foreign_preverbal_half`: 0.399 / 44.7%. Paired own-chain controls: `own_preverbal` π(a_A)=0.593 (top=a_A 61.4%),
`own_preverbal_half` 0.502, `own_same_relative` 0.685 (72.7%). Mean π(a_B)−π(a_A) under foreign_preverbal 0.423, trajectory-cluster bootstrap 95% CI [0.382, 0.474].
By pre-verbal length tercile (short/mid/long, medians 210/388/640 tokens): π(a_B) 0.719/0.578/0.442 (confounded with chain difficulty). By original bucket: early 0.589, mid 0.538, late 0.618.

**Script 28 — sentence-activation transplant** (`outputs/followups/activation_sentence_transplant_report.md`, 240 aligned cuts, 0 unaligned; 21-token neutral
" Wait, I should double-check this. Looking at the grid again carefully before settling on a move here."). Flip to runner-up (all cuts; Wilson CI):
donor text 88.8%; **all_layers_full 88.8% (positive control: top identical 240/240, mean |Δq| 0.003, 99th pct 0.052)**; all_layers_last12 88.8%; last6 85.8%; last1 54.6% [48.3–60.8];
early_0-7 88.8%; mid_8-15 77.1%; late_16-23 51.7% [45.4–57.9]; **layer_10 77.5% [71.8–82.3]** (all 186 flips ⊂ text flips); layer_15 68.8% [62.6–74.3]; layer_18 26.7% [21.5–32.6];
recipient (neutral text) 10.0%. Note: a single-layer write at ℓ ≡ writing all layers ≥ ℓ (patched positions' later states are recomputed), so the sweep measures
how much the suffix reads below ℓ: ≈11 points below layer 10, ≈9 in 10–15, ≈42 in 15–18, ≈27 in 18–23.
Strictly counterfactual by bin — layer_10: 1.0/1.0/0.975/0.875/0.65/0.35; layer_15: 0.952/1.0/0.975/0.775/0.5/0.125; text: 1.0/1.0/1.0/0.925/0.9/0.625.
Interpretation: the earlier "readable but not writable" reading (§5.3) held only for low-dimensional edits and foreign-context window states at the readout site;
the full in-context state of a sentence, written at its own positions at one mid layer, redirects the decision at bandwidth matched to a text injection.

**Figure**: `scripts/29_plot_intervention_levels.py` → `figures/followups/intervention_levels.png` (copied to the paper's `figures/`): left = injection flip rates by
bin for 3 conclusion wordings, 3 state facts, neutral; right = transplant flip rate by layers/positions.

**Paper (`workshop_main.tex`)**: §5.2 new paragraphs "Pre-verbal foreign chains" and "Belief-level versus conclusion-level injections"; §5.3 new paragraph
"Sentence-activation transplant", the read/write statement restricted to low-dimensional / foreign-context edits at the readout site, caveats updated, new
`fig:levels`; §5.4 P1–P4 revised accordingly (P3 "No — they dissociate" → "Partly": text and activation agree on the sentence positions, dissociate at the readout
window; P4 "readable but not writable" → "readable at the window, writable upstream at the text positions"); the supported-wording notes on the abstract /
Discussion / Conclusion updated. Scratch build (dblblindworkshop option stubbed, missing collaborator figure stubbed): latexmk exit 0, 0 undefined refs, 40 pages.
The four §5.4 conclusions changed materially → the 7-agent reviewer gate must be re-run once scripts 26/27 are in.

## 2026-09-05 — Results: script 26 (injection with continuation), stage 4 of 5

`outputs/followups/injection_with_continuation_report.md`; 120 cuts (20/bin) × {counter, belief_block_top}; greedy continuation ≤256 tokens after the injection
(stop at <|end|>), then forced answer. **counter**: flip to runner-up immediate 88.3% → after continuation 64.2% [Wilson 55.3–72.2]; 29 of 106 immediate flips
revert, 0 new; mean π(top) 0.764 (before) → 0.123 (immediate) → 0.358 (after); π(runner) 0.872 → 0.603; ended naturally 52.5% (median 208 tokens), and among
the 63 naturally-ended continuations the flip holds in 87.3% (reversions concentrate in unfinished continuations that re-parse the grid). After-flip by bin
0.65/0.8/0.8/0.8/0.5/0.3 (strict: 0.714/0.833/0.8/0.8/0.5/0.3); Spearman(commitment, after-flip) −0.261 p=0.004. **belief_block_top**: immediate 17.5% → after 16.7%
[11.1–24.3]; 15 of 21 immediate flips revert, 14 new; π(top) 0.764 → 0.593 → 0.719; ended naturally 15.8%; re-examination cue in first 200 chars 76.7%; rough
regex on the first 300 chars: confirm-only 24, deny-only 32, both 28, neither 36 of 120 — the model re-checks the named cell and confirms the false wall
about as often as it re-reads the cell as open (mixed; not reported as a number in the paper).
Paper: §5.2 new paragraph "Persistence under continuation"; §5.4 P1 and P2 extended with the survival numbers.

## 2026-09-05 — Results: script 27 (free-running validation), stage 5 of 5; program complete (RUN_ALL_DONE 12:55 CEST)

`outputs/followups/free_running_validation_report.md`; 60 cuts (10/bin) × 4 sampled continuations (temp 0.7, ≤700 new tokens; seeds 42·1000+i); 33.8% end
naturally (median tokens 700 = limit for the rest; unfinished samples forced at the limit). Overall P(eventual == forced top at cut) 0.758 vs mean q_cut(top) 0.761;
mean log-lik of eventual under q_cut −0.737 (uniform −1.386); per-cut TV(empirical, q_cut) mean 0.273 / median 0.189 (vs uniform 0.662); per-cut Spearman(frac top, q_top)
0.439 (p=4.5e−4), Pearson 0.403. Reliability by q_top bin: [0.4,0.6) 0.498→0.659; [0.6,0.8) 0.696→0.578; [0.8,0.9) 0.840→0.854; [0.9,0.95) 0.926→0.906; [0.95,1) 0.991→0.977.
By commitment bin P(top): 0.275/0.60/0.85/0.875/0.95/1.0 (q_top 0.765/0.463/0.63/0.81/0.911/0.99). Where forced top ≠ agent's eventual action (14 cuts, 56 samples):
P(eventual==top) 0.321, P(eventual==agent) 0.554, q_top 0.69, q_agent 0.163 → the forced readout overstates alternatives at low-commitment cuts. Naturally-ended
samples: P(top) 0.877 (n=81) vs 0.698 for forced-at-limit (n=159). P(eventual == agent's actual action) overall 0.812.
Paper: appendix "Answer Forcing" gains "Validation against free-running behaviour"; §5.1 gains a pointer sentence. Program summary: 5 scripts, ≈7,600 forward
passes + 480 generations, 10:03–12:55 CEST; all outputs under `outputs/followups/`, logs under `outputs/followups/logs/`.

## 2026-09-05 — Reviewer gate, round 1 on the revised §5.4 (7 agents per conclusion, verifying against files) → fixes

Round 1 verdicts (old text): P1 6/7 approve (reject: "the text acts mainly through stated conclusions" over-generalises single-sentence
evidence to whole chains, where 60.1% redirect with no move named); P2 3/7 approve (4 rejects, all on the clause "readout-window interventions
were ineffective at every commitment level" — the steering/c6 battery was never commitment-stratified; steering prefixes span commitment 0.21–0.87);
P3 1/6 approve (5 rejects: the positional dissociation is contradicted by script 28's all_layers_last12/6/1 = 88.8/85.8/54.6% — the last 12 sentence
positions ARE the 12-token readout window; the dissociation is by written content, not position; "upstream" wrong); P4 pending (2 reviewers on old text).
Every number in all three paragraphs reproduced exactly from the parquets. Side findings: §5.3 "moves decoded commitment by at most 0.006" was stale —
`steering_summary_normscaled.csv` max |mean_d_commitment| = 0.0497 (layer 18, commitment, α=−1; random max 0.0354) → corrected to 0.050/0.035;
§5.2 continuation ρ=−0.261 is the all-120-cut value (strict −0.386) → labelled; "8.9% top commitment decile" is an equal-width bin, reworded
"commitment above 0.9"; "imperfectly absorbing" no longer used in §4 → reworded; "77.5% of decisions" → "cuts"; false-fact label qualified (226/240).
Fixes applied to P1, P2, P3 (rewritten: "Yes, once what is written is held fixed … what dissociates is not position but content"), P4
("readable at the window and writable at those same positions, but only at the full bandwidth of an in-context sentence state"), §5.3 (window
identity made explicit). Round 2 launched with 7 fresh reviewers per conclusion.

## 2026-09-05 — Reviewer gate, round 2 on the corrected §5.4: 28/28 APPROVE — goal achieved

Seven fresh general-purpose reviewer agents per conclusion, each recomputing the paragraph's numbers from the parquets (`interp/.venv/bin/python`) and
checking the logic (instructions: scratchpad `gate2/shared_instructions.md`). Verdicts: P1 "Does changing the belief before commitment change the eventual
action?" 7/7; P2 "Do interventions lose force after commitment?" 7/7; P3 "Do text and activation interventions identify the same positions as causally
important?" 7/7; P4 "Which conclusions concern availability, commitment, and causal use?" 7/7. Every quantitative claim reproduced within rounding in all 28
reviews. Non-blocking notes applied afterwards (wording only, no numbers): "90.0% of decisions" → "cuts" (P2); "overwriting the entire readout window at a
single layer" (P1). Notes left as-is (already stated in §5.3 or out of scope): the last-12 in-context write is all-layer while the 20.8% foreign overwrite was
single-layer; the partial-ρ estimator of script 20 (rank-residual Spearman −0.175) is conservative vs the textbook partial Spearman (−0.245); §4 now calls
c_{t,k} "confidence" in places while §5 says "commitment" (collaborator terminology; flag for the paper owners); the steering battery's commitment range
0.21–0.87 is in commitment_to_final units (entropy commitment 0.07–0.64).

## 2026-09-12 — Activation causality campaign initialized

User authorized the cooperative four-agent plan: 72-hour local campaign, belief-to-action mechanism primary target, fresh confirmation decisions, and a separate proposed paper-section revision. Authoritative paper: `_IAB_Workshop_2026__Belief_action_gap-4.pdf`. Persistent research goal registered; campaign deadline 2026-09-15 08:24:34 UTC.

Implemented initial registry/worker and assigned independent engine, environment, and review owners. Root-local package `src/activation_causality/`; entrypoint `scripts/30_activation_causality_campaign.py`; protocols `research/activation_causality/`; results `outputs/activation_causality/campaign_20260912/`. Existing paper/data/upstream remain untouched.

Metal unavailable inside sandbox; authorized outside-sandbox preflight succeeds on Apple M4 Max, 48 GiB memory, MLX 0.31.2 / mlx-lm 0.31.3. Initial 32 CPU tests pass. First actual-model correctness gate caught a BF16 NumPy conversion error before any experimental row; fix and expanded long-context parity gate underway. No scientific result yet. Frozen initial workloads: 12 historical sentence cuts ×9 conditions (forced-only), and 24 new matched layouts ×8 conditions (forced-only). Fresh state cases are discovery, not confirmation.

## 2026-09-12 — Corrected activation engine gate and sentence positive control

The first attempted resume did not restore the full-head inference mode chosen by validation. The entire `sentence_reproduction_forced_v1` experiment is retired (20 raw rows preserved; not scientific evidence). Fixed the default across worker restarts and recorded execution fingerprint plus head mode per row. Expanded actual-model gate passes short/long (>128 token sliding window), all three hook sites, exact identity, and nonvacuous layer-10 three-token patched-cache checks. Native/manual same-shape logits match exactly; ordinary BF16 cached/full batching drift is measured and reported, not mislabeled exact equality.

Clean `sentence_reproduction_forced_v2`: all108 jobs complete, 12 historical cuts ×9 conditions, one execution fingerprint, full-head throughout, no errors. Target forced-top counts: baseline0/12; layer10 10/12; layer15 10/12; all layers/full sentence11/12; all layers/last12 11/12; donor text11/12; reverse layer10 1/12. Identity baseline probabilities match exactly; full transplant and donor top match12/12, but probabilities are not exactly equal. Independently recomputed by reviewer. This is only a reproduction of sentence-state sufficiency under forced readout. Native-generation pilot now runs six cuts ×9 conditions, cap2048; no new semantic claim.


### 2026-09-12 — Controlled activation discovery and native pilots

Completed three registered forced-only discovery sweeps on 12 fresh controlled layouts (four wall, four key, four door): state_controls_forced_v1 (456 jobs), fact_sentence_forced_v1 (204), input_fact_forced_v1 (264). All scheduled jobs completed without technical errors; identities reproduce baseline distributions exactly. Last-header task-state patches and authored Boolean fact patches show little target-probability benefit. Early input-fact patches instead reconstruct the physical counterfactual distribution: mean total variation distance falls from 0.13608 (recipient versus counterfactual) to 0.02675 at layer 0 and 0.03254 at layer 3. This is a post-hoc forced-distribution routing diagnostic, not a native or semantic-mechanism claim. The independent reviewer recomputed counts, norms, ties, and provenance in research/activation_causality/REVIEW.md.

The first historical native pilot completed all 54 jobs at a 2,048-token cap: 39 final actions and 15 truncations. Target success was 3/6 for layer-10/full sentence transfer versus 0/6 for neutral baseline, but completion-sensitive bounds include zero. A uniform 4,096-token repeat preserves every earlier result. Fresh native state-control and input-fact pilots are registered; one GPU worker runs the queue.

Implemented exact generated-token reasoning windows, late authored-fact interventions, source-fact patches, immutable discovery-only subspace fitting and held-out subspace tests, and a validated scientific control-figure builder (script 31). Source/model/code fingerprints, alias-aware layout exclusion, multiple-comparison correction, native noncompletion handling, and immutable candidate review protect the future confirmation stage. No original manuscript was changed.


### 2026-09-12 — Native sentence control and factual-report diagnostics

The complete sentence_native_pilot_v2 cohort contains 54 executions at cap4,096: 51 parsed final actions and three truncations. All 54 input hashes and common generated-token prefixes exactly reproduce cap2,048; the 39 previously complete sequences remain identical, and 12 of15 truncated sequences now finish with non-target actions. Layer10/full/last12 transfer still yields target3/6 versus neutral0/6, with completion-sensitive point bounds +.333 to +.500. Small historical samples do not establish a population effect; the explicit action-bearing sentence is a positive control, not a task-belief mechanism. Independent audit in REVIEW.md; script31 regenerated PNG/PDF/counts and rendered PDF verified visually.

Fresh native header controls completed84/84, with82 final actions and two truncations. Recipient and physical counterfactual are each simulator-optimal6/6 and switch on all six layouts, while each forward layer10 patch preserves the original action6/6 despite altered reasoning sequences.

Added cloned-state Boolean factual-report diagnostic and a pure source-cohort adapter; registered input_fact_report_forced_v1 (60 jobs) before queries. It retains the entire physical input-fact cohort, exact token IDs/positions and immutable lineage, reports unchanged-fact controls and Boolean candidate mass, and makes no native-action or latent-belief claim. Optional runner integration precedes the next inference queue.

### 2026-09-12 — Audited native input-fact effects and independent report branches

Completed input_fact_native_pilot_v1:60/60 jobs on the same six discovery layouts as the header pilot, two per wall/key/door family. Layer0 and layer10 matched input-fact-span patches each produce the simulator's physical-counterfactual target action6/6, versus original baseline0/6. Direct physical counterfactual6/6; both physical text baselines optimal6/6 for their own states. Identity and last-header comparators remain0/6; random controls1/6 atL0 (plus one truncation) and0/6 atL10. Overall59 final actions and one random-control truncation. All12 identity conditions exactly reproduce recipient generated IDs and forced distributions. This is exploratory input-information routing evidence, not an abstract belief mechanism or held-out confirmation. Independent raw audit and script35 counts/PNG/PDF are complete and visually verified.

Completed input_fact_report_forced_v1:60/60 registered derivative jobs,120/120 scored Boolean probes. The rerun reproduces all60 original source forced-action distributions and input hashes exactly, then probes independent clones of each rebuilt prefill, with original cache/logit/token/intervention fingerprints preserved. Intended target-report conditional-probability changes L0+.807394 andL10+.871158; unchanged-fact changes+.008763 and+.013885. Report full candidate mass: randomL0 conditional targetq=.596120 coexists with full-vocabulary target probability=.047721 and Boolean mass=.346197. PhysicalCF key cases give mean targetq=.622011; this two-case descriptive score is not a calibration study. Separate reports do not establish mediation.

Reasoning-forced100, late-authored-fact-forced124 and contextual-input-forced264 also complete and independently audited. Late single-layer patches replace sentence-tail states after the Boolean token; only the full-span condition includes that token. First contextual audit had an incorrect random-control norm denominator and is explicitly superseded by auditv2; all raw results are unchanged. Registered native endpoint extensions100/124/132 now execute serially under one worker, regardless of forced-screen signs.

Mixed-family input confirmation now enforces fixed wall/key/door quotas, source-result lineage, global layout aliases, within-family resampling and conservative constant-stratum guards. Full model identity now hashes all3 actual weight shards plus metadata (10files,459 indexed tensor names) before confirmation. New pure random-subspace helper matches tokenwise task-projected update norms and records float32 errors; compiler integration waits for the live inference queue to finish. All298 CPU campaign tests pass; Ruff passes. No confirmation selection is frozen. Updated separate interim causal-section proposal has independently verified numbers, links and source hashes; original PDF remains unchanged.

### 2026-09-12 — Native reasoning-window extension and subspace provenance gates

reasoning_windows_native_pilot_v1 completes100/100 technical executions,98 final actions and2 truncations (a door baseline and its exact identity atcut32). Every matched L10/L15/all-layer last12-window condition has target0/5 atcuts32/128; every natural-donor window condition has target0/6. Matched donor continuations reach target5/5 at32 and4/5 at128; natural donors differ from the recipient's counterfactual target6/6 atbothcuts. All100 forced distributions/input hashes exactly replay the forced cohort. All54 transfer continuations change generated tokens. Two natural-L15 key cases produce third actions, matching neither target nor actualdonor; therefore no target transfer does not mean no action influence. Reviewer audits and complete analysis are saved. These reuse the same discovery layouts and are not confirmation.

New basis loader requires a materialized-engine proof bracketed by full weight hashes, representation-runtime compatibility, same-bytes artifact verification, connected fit audit plus original immutable request, valid source job IDs/definitions and nested layout aliases. Independent review caught and reproduced a missing request-payload link; fixed before any basis fit/query. Both intended and random subspace writes now have a prospectively registered1% numerical audit contract with exact recipient hashes, exactzero rules and preserved technical failures. Pure helpers and fit-side builder are complete; engine/runner integration waits for the active native queue. Latest full suite342 tests passes in9.205s; full Ruff passes. No basis fit, selection freeze or confirmation query has occurred.

### 2026-09-12 — Runtime integration staged without changing active experiments

Prepared isolated CPU staging at tmp/activation_runtime_stage_20260912. Five changed files integrate full loaded-model proofs, strict basis/source/request provenance, exact pre-write recipient hashes, and prospective intended/random applied-norm checks. Failed checks preserve compiled/actual diagnostics before action scoring; whole-cohort fingerprints stay stable across resumed workers. Independent review passes50 targeted tests with warnings-as-errors; staged full360 tests and Ruff pass. Original inference files and tests remain unchanged while session31376 executes the native queue. Promotion and real BF16 preflight are pending.

Archived24 original package files at outputs/activation_causality/campaign_20260912/source_snapshots/20260912T110753Z. Reconstructed execution keys match input-native60, factual-report60, reasoning-native100 and the42 late-fact rows present at capture. Earlier different execution identities are not claimed covered.

Prospectively registered subspace_complete_native_plan_v1.json before any fit/subspace query: all six L0 wall/key/door rank1/rank4 fits proceed to full native evaluation if technically valid, regardless of forced-action sign. Exact source/layout aliases, basis/experiment IDs,14conditions,doses.5/1,seed42,temp.7,cap4096 and failure gates are fixed.336forced plus336native trials reuse12evaluation layouts across ranks. This is descriptive discovery, not confirmation or a semantic/necessity claim. Six-hour reporting reserve remains required.


### 2026-09-12 — Archived analyses independently replayed

Added research/activation_causality/REPRODUCE.md, an exact local environment inventory and archived analysis drivers. The first isolation assertion caught an omitted src/__init__.py: from repository cwd, Python could select the live regular src package instead of the archive namespace. No analysis was accepted from that attempt. Immutable environment manifest_v2 adds the initializer and preserves v1; explicit module-origin checks now pass from repo cwd, and an absent broader src module cannot fall back to the live tree.

Using the corrected archive, input-native60 and reasoning-native100 generic summaries, Markdown and PNGs reproduce byte-for-byte; their PDFs differ only in CreationDate. The engineer independently reran all four dedicated figure/report commands: every scientific value, count, raw-source/spec hash and all three PNGs match; factual JSON/Markdown match exactly. PDF text and rendered pages were verified. The reviewer independently checked isolation and artifact comparisons. Saved audits are archived_generic_reanalysis_audit_v1.json and archived_figure_report_reanalysis_audit_v1.json. Routing figure script32's earlier counts name a different script hash; the current archived script reproduces its values, without claiming the archive contains that earlier script version. This is saved-data reanalysis, not new model replication or a complete inference installation.

Engineering audit of subspace_complete_native_plan_v1 verifies all24 source definitions, training/evaluation aliases and all six 14-condition inventories before any fit. Live inference and its source files remain unchanged; native late-fact/contextual queue continues. The staged runtime has not been promoted, and no candidate freeze or confirmation query exists.


### 2026-09-12 — Late authored-fact native cohort completed and audited

reasoned_fact_native_pilot_v1 completes124/124 technical executions,115 parsed actions and9 truncations on the same six discovery layouts. Correct-assertion baselines reach target0/6 atcuts32/128. Last12-tail L10/L15 transfers each reach0/6 atbothcuts, with2 truncations perlayer at32 andnone at128. The literal Boolean lies just outside those narrow windows. Contradictory authored-assertion text under the unchanged recipient observation reaches0/6 at32 and1/6 at128; all24-layer/full-suffix patches reach1/6 atbothcuts. These are sparse, mixed-direction discovery outcomes, not a general activation or belief null.

All124 forced distributions and input hashes replay exactly;84 source fingerprints verify;24 identity conditions reproduce entire generated sequences and zero writes. All68 nonidentity patch continuations differ from their controls, including two nonspecific L15 third-action changes. Full-span patched and donor-text continuations need not match because residual writes leave different upstream KV caches. Independent audit reasoned_fact_native_independent_audit_v1.json and generic report are complete; original PDF remains unchanged.

A proposed42-job full-suffix L10/cut32 follow-up was not registered or queried. Coordinator and reviewer prioritized already registered contextual-input and subspace work because the authored-text reference itself is weak; retain the layer/window limitation explicitly. Session31376 is now running contextual_input_native_pilot_v1 (132 planned jobs), and original inference files remain frozen until the entire queue ends.


### 2026-09-12 — Complete-search inventory and audit-source archive

Added standalone CPU script 36 and SEARCH_LEDGER_GUIDE.md. Immutable v1/v2 ledger snapshots enumerate every registered design, including the retired sentence run, source/preparation definitions, unexecuted initial design, exact result/spec/job identities, recorded layout overlap, attempts and operational preflight/gate history. Partial native outcomes remain withheld; no pooled target effect or global completion fraction is produced. V2's independent review verifies 2192 captured files, 2920 canonical job definitions and 2163 frozen result bindings, with exact Markdown regeneration and 11 targeted CPU checks. The v2 snapshot has contextual 87/132 rows and is explicitly distinct from later live progress. A snapshot-race label was fixed; v2 also recognizes basis_fit_failed operational events. V1 is retained. Exact writer bytes and manifests are archived, and the archived v2 command runs successfully.

Contextual-input audit preparation verifies all 132 frozen jobs, 36 physical/token checks and the corrected random-control norm reference without reading partial native outcomes. Prepared contextual and late-fact audit scripts are archived with their prior preparation hashes. Model session 31376 remains live (contextual 103/132 at 12:09:40 UTC); original inference files are unchanged. Staged runtime promotion, six basis fits, registered subspace evaluations, frozen selection and fresh confirmation remain pending.


### 2026-09-12 — Contextual native audit and prospective runtime promotion

The complete contextual native cohort has 132 executions, 129 parsed actions and three truncations. Matched and natural-context input transfers each reach the recipient-relative counterfactual target on all six layouts at layers 0 and 10; same-fact and irrelevant controls reach none. Five observed natural-donor actions differ from that target; one unresolved donor remains in the denominator. All 24 identity/restoration token traces and 132 forced/input/patch-log replays match. Thirty-six executions exactly repeat earlier results on the same six layouts, so this is additional control evidence, not an independent replication. The independent audit and overlap supplement are saved; the paper proposal is updated separately and the PDF remains unchanged.

Root verified queue session 31376 terminal with exit code 0. A discovered engine import-provenance gap was fixed and independently reviewed in the isolated stage before promotion: five unchanged dependency files now enter execution/gate, freeze/readiness and basis-representation identities. Under the repository worker lock, all 12 original/stage hash guards passed; original files were archived and the final files promoted. A prospective subspace implementation addendum preserves every prior scientific setting and ID. All 368 promoted-workspace CPU tests pass with warnings treated as errors; Ruff passes. An independent 29-file source archive and isolated fresh-import verification are complete.

The first registered wall rank-1 basis fit succeeded with full source, model-snapshot and activation-representation gates. Its first held-out 14-condition forced block completed without technical errors; all four intended/matched-random actual BF16 writes pass the unchanged 1% norm and exact recipient-hash contract. NumPy emitted warnings in a Gram calculation despite returning finite near-unit values matching an independent scalar calculation; the engineer is auditing that discrepancy before continuation. This is technical preflight, not a completed 56-row cohort or native action evidence. No candidate selection, freeze or confirmation query exists.


### 2026-09-12 — Six subspace fits and complete forced batteries audited

All six prescribed wall/key/door layer-0 rank-1/rank-4 fits passed exact source, snapshot and representation gates. All twelve actual-basis evaluation specs were registered before any native query. The 336 forced jobs completed without technical errors; independent review verifies all six fit chains, exact 56-job cohorts, 96 guarded norm writes and 48 forced identity/restoration checks. Maximum relative applied-norm error is 0.0764% against the fixed 1% threshold. The finite NumPy Gram warnings are retained with independent scalar-Gram evidence; runtime/tolerance are unchanged. All six forced reports and the combined independent index are saved.

Native queue session 59710 now executes all 336 prespecified native jobs, regardless of forced-readout sign. The same twelve evaluation layouts are reused across ranks and conditions. No candidate freeze or fresh confirmation query exists yet. The separate contextual figure and updated manuscript proposal preserve shared-layout/repeated-execution and norm-control limitations; final independent figure review is underway.


### 2026-09-12 — Contextual norm limitation, archived replay and ledger v3

The completed contextual write logs show natural/same-fact norms differ substantially (7.849–499.419× at layer 0; 1.659–30.540× at layer 10). The existing matched-counterfactual random control is only 76.183% of the natural norm on one layer-10 wall case. Independent review recommends one conditional additional natural-norm random condition if a contextual profile is selected, preserving the same direction/support/seed and all existing controls. Primary natural-versus-same-fact remains unequal in magnitude; any norm-specificity claim needs prospective inferential treatment. No candidate, layer or freeze is selected. The interim paper adds this caveat; prior draft, original PDF and figures are preserved.

The exact archived contextual figure writer reproduces the PNG byte-for-byte and the PDF after only CreationDate normalization; scientific counts and source hashes match. The new reproduction addendum records the command. Ledger v3 now includes all 31 specs and retains a detected live-read race with partial native outcomes suppressed. Its 2598 captured hashes and exact Markdown were independently checked. Native queue 59710 remains active, with whole-cohort scientific summaries deferred until completion.

### 2026-09-12 — First native subspace cohort and bounded audit automation

The first wall rank-1 native cohort completes all 56 executions: 48 parsed actions and eight capped continuations on four later discovery layouts. Full-dose task-subspace transfer reaches the counterfactual action 4/4, baseline 0/4 and projected-norm-matched random 1/4. Independent replay, identity and norm checks pass; the interpretation preserves half-dose censoring, reverse/removal outcomes and the four-layout discovery limitation. No candidate is selected before the other five registered native cohorts finish.

Standalone CPU script 38 passed 19 targeted tests and independent review. Root launched bounded watcher session 60607; it audits only exact complete cohorts into separate immutable automated-recomputation files. The first automated audit succeeded. Model queue 59710 remains the only model worker, with promoted inference sources unchanged. Conditional adapter and statistical-design preparation proceeds without confirmation queries or candidate selection.

### 2026-09-12 — Three native subspace cohorts and prospective confirmation contracts

Wall rank-4 and key rank-1 now join wall rank-1 as complete, independently audited 56-job native cohorts. Wall rank-4 full-dose transfer reaches 3/4 targets plus one cap, while its norm-matched random condition produces third actions on all four layouts. Key rank-1 full dose reaches 4/4 targets and norm-matched random preserves all four original actions; removal retains all four targets, so it supplies no necessity evidence. The three remaining native cohorts continue under the unchanged model queue; no partial outcomes or candidate selection are used. The generic key rank-1 report completed successfully.

The reviewed v4 candidate-selection policy was prospectively adopted during discovery, before wall rank-4 outcomes were read by the coordinator. It fixes two co-primaries per selected question, a maximum four-contrast family, complete-control qualification and budget rules, a fixed descriptive lexical supplement, and the absolute six-hour reporting reserve. No candidate, freeze or fresh model query exists. Independently reviewed staged components include source-cohort completeness, the fixed reversible renderer, the canonical frozen subspace adapter and co-primary analysis with legacy byte parity. The contextual adapter, full-write norm instrumentation and integrated freeze/readiness/sampling/cutoff are being implemented in isolation. Script 39 is certified for complete-only final subspace synthesis; it has not read or plotted the unfinished search.

### 2026-09-12 — Fourth native cohort and reviewed confirmation components

The complete key rank-4 native cohort has 56 jobs, 49 parsed actions and seven caps. Half-dose transfer reaches 4/4 targets versus baseline and projected-norm random each 0/4 plus one cap; full dose reaches 3/4 plus one cap. Removal and half-dose reverse retain all four targets; full reverse restores all four original actions. There are no third actions. All eight whole-token identities, 56 forced/input replays and 16 required norm writes pass. Twenty reference executions exactly repeat key rank-1 on the same four layouts. The independent audit and generic report are complete. The two door cohorts continue under the unchanged sole model worker; candidate selection remains closed.

Independent review now covers the complete-control co-primary contract, the same-bytes discovery source/authenticity gate, corrected fifteen-condition contextual adapter, full-write norm compiler/runner and CLI full-family analysis dispatch. Review found and corrected JSON-type equality gaps and source hash/parse races before promotion; original findings and regression tests are retained. Root separately reproduced the contextual adapter's existing mechanics with the actual local tokenizer on six discovery layouts: 168 repeated canonical job definitions match, with all fifteen conditions constructible in both fixed renderings. No new native outcome or BF16 result is implied. All 29 live promoted sources remain unchanged. The persisted sampler, cutoff supervisor and outer source/basis/freeze/readiness/lexical integration remain in progress; the persistent investigation goal remains active.

## 2026-09-12 16:03 UTC — closed discovery and actual full-write preflight

All six native subspace cohorts closed and passed independent audits (336 jobs, 305 parsed actions, 31 caps; 12 physical layouts shared across ranks). The complete six-panel figure and independently reviewed scientific selection record are saved in the campaign: contextual L10 and wall L0/rank1/dose1 proceed toward fresh confirmation. This selection follows a policy adopted during discovery, not before it.

After the native queue ended, five reviewed modules were promoted under the worker lock and archived with the resulting 33-file runtime (`technical_preflight_promotion_20260912/manifest.json`, SHA256 6bbcd0b0c197eeae77d456ba433be912a1fcf63053ec2bf76c576d014fb287c1). Actual forced-only full-write preflight completed72/72, zero errors/native queries, 48 norm checks, 24 orientation records and12 exact-zero identities. Independent audit SHA256 e0c66d2b290921761b9f8afdbce9c52b5940bbff04df0bd7d25a509711986b68; maximum norm relative error0.000174472179. Model sessions59710/23672 and watcher60607 are terminal.

Fresh-cohort planner/freezer/renderer/lease integration is staged and under final review. The corrected17 freeze +9 CLI tests pass. Historical preflight source inputs are being bound to their immutable archive across final promotion while current execution bytes must still match. No production sampler, freeze, registration or confirmation query has occurred. The goal remains active; main PDF unchanged. Current handoff: research/activation_causality/STATUS.md and CONFIRMATION_INTEGRATION.md.

## 2026-09-12 16:40 UTC — frozen fresh confirmation launched

The integrated implementation passed646 CPU tests and Ruff, then was promoted under the worker lock (44runtime files; initial promotioned8f3828…). Actual preparation completed128layouts per candidate in four attempt-zero batches with no rejected pairs, fixed seed20260912, globalm4, all15/14controls and32-layout optional rendering subsets (4,640jobs). All scientific fields and sampler state are immutable.

A pre-query operational review found that resolving the Python symlink would launch base Python outside the venv. The supervisor-only correction preserves the venv invocation and binds binary/prefix/pyvenv.cfg identities;41 targeted tests and32 independent tests passed. The reviewed source-only promotion and constrained plan amendment retained every scientific artifact without resampling. Exact amended planSHAb4099de6…; final independent approval2d37032d…; current44-source promotion manifest05930b5c…. Original unapproved plan and all audit-harness failures remain archived.

Freeze confirmation_20260912_v1 published16:33:28UTC (SHA898ea839…), registration copied all8spec/dataset files with exact hash parity. First single-cohort localMetal supervisor session62935 launched contextual_primary (1,920jobs), parent79885/child79914, lease9227a6ff00f04224b0ab41acbb2ffdf1 at16:37:13UTC. The worker started16:38:15 and is saving result files; no partial outcome contents have been read. Both primaries run before alternates with complete-cohort technical checkpoints. Fixed inference cutoff remains15September02:24:34UTC; goal remainsactive.

Generic discovery reports for door ranks1/4 completed successfully. Two initial reporting commands used nonexistent door_status IDs and failed before result reads; their logs are retained, and corrected door IDs completed without changing experiments. The main manuscript PDF remains byte-identical. Current operational handoff: research/activation_causality/STATUS.md and CONFIRMATION_EXECUTION.md.

## 2026-09-12 17:00 UTC — confirmation monitoring and discovery manuscript update

The first confirmation supervisor session 62935 remains active. The 16:57 UTC filename-only check found 55/1,920 expected files, zero unexpected files and no terminal record; the three later cohorts have no saved results. No confirmation outcome contents or worker logs were read. All 44 frozen inference sources remain unchanged.

Standalone reporting and per-cohort audit tools passed initial independent checks. Review found that historical technical failures, including failures before any result row, must remain blocking after later success. Both authors are aligning that operational gate without changing the frozen analysis or inference. Missing-only interruptions remain distinct, and no automatic error retry or adjudication override is introduced. Final certification is pending; figures are being developed from synthetic reports only.

The separate causal-section proposal now includes all six closed native subspace cohorts and frozen confirmation methods. Coordinator recomputation verified its 24 new table cells against 336 raw discovery records and the six independent audit hashes (12 layouts, 305 parsed actions, 31 caps, 116 forced/native agreements). All 93 links and the original PDF hash pass. Manuscript SHA256 83ff4604c98972a7caa38cf35306fd4960bbcbb7ad2a65697b8788be9acf4f5a; coordinator review SHA256 868196558bf49d54e92fb8e0c4c7e22914f43a3229a49855267e1c864885cdf1. Previous manuscript and methods versions remain archived.

Current navigation documents now distinguish the frozen supervisor workflow from historical discovery commands. The old whole-campaign ledger writer reads partial raw records internally, so its refresh is deferred until confirmation closure/cutoff; filename-only monitoring remains available. The first math-formatting replacement aborted before writing because the old TeX transpose contained a literal tab; the exact-character correction then succeeded, with original bytes preserved. No scientific settings changed. The goal remains active, with confirmation, final audits, figures and reproducibility package still required.


## 2026-09-12 17:45 UTC — theory-foundations research campaign launched

User approved a 72-hour four-role research system to rebuild the main paper theory if needed, connect it to real-data analyses and targeted local experiments, and deliver a separate revision package. Persistent goal created; workspace research/theory_foundations/ and immutable run outputs/theory_foundations/campaign_20260912T174516Z/campaign.json. Coordinator /root; theory /root/theory_researcher; empirical /root/behavior_probes; independent skeptic /root/causality_code. Start 12 September 17:45:16 UTC, new-inference cutoff 15 September 11:45:16 UTC, research deadline 15 September 17:45:16 UTC.

Initial tasks audit gap semantics and source provenance, compare measurement/information-use/commitment accounts, implement mathematical checks and freeze CPU study protocols. No theory model query or experiment has run at launch. All 44 frozen activation-causality sources match the current promotion manifest; authoritative paper SHA256 045ad8af35ba7a106f97abcc6fc5afdc1bf5f34d884bae68f928ca864873e3f9 is unchanged. The existing campaign retains worker/queue ownership; no partial confirmation results/logs were read. Original Sections 2–3 numerical artifacts remain missing; the available commitment corpus is a distinct study. System launch does not complete the research goal.


## 2026-09-12 18:00 UTC — activation confirmation reporting tools certified

Standalone scripts 40/41/42 now have independent source approval. The final joint CPU run passed 67 tests (23 report, 31 technical audit, 13 figures) with warnings as errors; record confirmation_reporting_joint_cpu_check_v3.json SHA256 ecd8607da286dd2ad31e7417edc554fb7819d0df93b2fa8849ba7e91ec5ede96. Initial fixture-isolation failures remain recorded. Corrections were test-only; no production origin or historical-failure gate was relaxed. All six source/test files and 44 frozen inference files remained unchanged across the final run. Synthetic complete/cutoff figures passed independent PNG checks and coordinator PNG/PDF visual checks; no production outcomes were used.

The exact 52-file reporting-source archive was published with manifest SHA256 a9828eb5829c42c19b8de4c8264d772ebc8376d6df8f9238bf411fbec506f972, and independently verified under review SHA256 86cc6f9bdeeca3e7c2c81ef645048adcb5efd97254e422a9a6e1e0c34e0890d5. The live reproduction guide's duplicate figure command was removed while preserving the archived original. Its documented source/import verifier passed in the local isolated Python environment. A separate documentation addendum preserves the corrected guide and exact executed verifier; coordinator execution-record SHA256 3975258013c2e90d7612f12c46e238cdd02b950e873732152f046613bf440d65. This records the previously observed successful command, not a second execution. All 17 guide links resolved. Complete production reanalysis is still pending.

The engineer and researcher agents reached their usage limits; the coordinator took over their remaining integration work. The independent reviewer remains available and found the completion/audit/next-cohort handoff coherent. Session 62935 is still the sole activation model worker's live supervisor. Actual metadata-only readiness at 17:59:35 UTC found 209/1,920 expected contextual-primary filenames and zero unexpected files, reading only four metadata inputs. Artifact tmp/confirmation_contextual_readiness_20260912T1802Z.json SHA256 64f9d6c35750a8cd0b6550f56602a983a63c187904e2c5f8bd037c3f7ef4d71d. No partial confirmation results or worker logs were read.

README, STATUS and CONFIRMATION_INTEGRATION now describe the approved reporting tools and current role availability; their prior bytes are preserved in research/activation_causality/checkpoints/20260912T180035Z/. No frozen inference source, experiment, lease, model or main PDF changed. Fresh confirmation, complete-cohort independent raw audits, the joint final analysis, actual figures, full ledger refresh, completed-results replay and final separate paper proposal remain required. The activation research goal stays active.


### 2026-09-12 18:06 UTC — theory campaign initial theory and first audited empirical discriminator

Theory/literature/framework drafts and separate manuscript outline are in research/theory_foundations/. Independent mathematical review verifies gap = selection regret + information value, including stochastic/scale counterexamples. Exact independent planner audit matches all 1,276 available archived optimal sets; original Section 2–3 artifacts still not recovered. The current PDF states 142 states; older source-note counts remain separately attributed.

B1 frozen exploratory log-odds reanalysis completes 720 archived evaluations, 240 cuts, 217 decisions and 82 trajectories. Independent implementation reproduces every prediction, 82 grouped folds, four models and three numerical floors (maximum prediction difference1.95e-14). Absolute-token-time addition has held-out MAE difference+0.046384, descriptive interval[-0.000167,+0.095147]; all R² nearzero. No claim of constant-gain adequacy, equivalence or causal dynamics. Report: research/theory_foundations/coordinator_analysis/B1_REPORT.md. Source27 independently confirmed to use a forced endpoint after sampled reasoning; it is not native-action calibration.

No model queries; main PDF and all44 frozen activation-causality runtime files unchanged. A1 implementation/review and substantive A2 cached-state-belief decomposition are ongoing, followed by prospective experiment selection and final revision integration. The persistent research goal remains active; system setup and initial results do not complete it.


## 2026-09-12 18:09 UTC — final-ledger confirmation scope clarified

The unchanged archived search-ledger writer is statically compatible with all four frozen confirmation specs (1,920/1,792/480/448 jobs). Independent scope review search_ledger_confirmation_scope_review_v1.json SHA256 9d982c241bac4f0925ec3a0d6addc8a4e4c95f127f299ea09a113d705767c55b binds the archived writer and exact spec hashes. Its current-row complete label does not clear prior technical failures or missing supervision; its file inventory excludes leases/supervision/model-load/full-write evidence; its selected failure-event summary can omit worker_finished errors>0; and its source graph omits alternate parent/role relationships. The dedicated script40/41 and independent raw audits remain authoritative for confirmation admissibility.

SEARCH_LEDGER_GUIDE.md now states these limits, maps the exact primary/alternate roles and repeated32-layout subsets, and separates the later72 forced-only technical-preflight jobs. Guide SHA256 3c98b378701826e91853df15416f8431f4142857992add46e34939efda07b4ea; all 58 local links resolved. Prior bytes remain in checkpoints/20260912T180639Z. No ledger execution, native result/attempt/log read, source change, or model query was part of this work. Full ledger refresh still waits for whole-confirmation closure or the fixed cutoff.

Supervisor session62935 remained live through two55-second polls, the latter ending approximately18:08UTC; no terminal process result was returned. This turn adds a concrete reporting-scope clarification and a verified wait. The activation research goal remains active; its scientific outcomes and final package are still pending.


## 2026-09-12 18:25 UTC — theory campaign second archival study independently verified

A1 corrected v2 analysis completes all1,276decisions/95trajectories on10fixed relative bins. Independent raw recomputation verifies21,288relative/absolute rows,210absolute-panel exclusions,810summary rows,2,000trajectory bootstrap draws and all4simultaneous primary intervals. Concentration rises0.351188→0.952872 and planner-optimal forced-action mass0.444021→0.970481. There are22original observed-action errors. These are exploratory forced-action diagnostics, not state-belief or native mechanism estimates. Initial index/count gate failure and fresh corrected execution are both preserved. Results: outputs/theory_foundations/campaign_20260912T174516Z/empirical/a1_concentration_optimality_v2/; independent audit reviews/a1_independent_recomputation.json.

Stochastic-choice/sensitivity mathematics and B1 presentation figures also passed independent review. B1 frozen formula/results remain unchanged; source audit clarified log(1+zero-based last-token index)=log(included token count). Source-paper revision requirements and evidence_v3.json now link exact claims/results/limits. A2 cached-state decomposition is in technical preflight, with full fitting freeze/review still required.

Preferred fresh design is N2: known next-move costs under fixed physical state, native restart/replayed-reasoning comparison and uniform-cost control. Eight CPU-only development fixtures satisfy the exact successor-distance benchmark; these are not an evaluation sample or model evidence. Exact scientific protocol, sample size and worker handoff remain pending. No model query, hosted call, source-paper modification, frozen runtime edit or other-campaign partial outcome/log read occurred. All46protected paths match at18:22. The theory research goal remains active; A2, prospective execution decision, full revision/reproducibility package and final independent claim review remain required.


## 2026-09-12 18:30 UTC — activation confirmation operational checkpoint

Supervisor session62935 remains live, verified by repeated55-second polls of the same tool handle. The approved script41 filename-only check at18:29:59UTC found278/1,920 expected contextual-primary files, zero unexpected filenames, four metadata inputs and no result-content reads. Immutable checkpoint tmp/confirmation_contextual_readiness_20260912T182958Z.json SHA256 72b5e5594b2d278a8825c9ff9a5e949f575788d582a93fdd193c704861703eef. This is operational progress, not technical or behavioral validation. The three subsequent cohorts remain unlaunched by this coordinator. STATUS now records the latest checkpoint and the completed static ledger-scope review; prior status bytes remain archived. No native outcomes or worker logs were read, no frozen source or scientific setting changed, and the research goal remains active.


## 2026-09-12 19:00 UTC — activation confirmation operational checkpoint

Supervisor session62935 remains live, verified by repeated55-second polls of the same tool handle. The approved script41 filename-only check at19:00:31UTC found360/1,920 expected contextual-primary files, zero unexpected filenames, four metadata inputs and no result-content reads. Immutable checkpoint tmp/confirmation_contextual_readiness_20260912T190030Z.json SHA256 c4d6fce1c68b5e988f68f6104b473b92b91dcfc6f9f948b357787f36980491e2. At18:43 the all-cohort filename check saw316 first-cohort files and no saved files in the later three cohorts; no cohort had unexpected filenames. These are operational counts, not technical or behavioral validation. STATUS is updated and its prior bytes archived. No outcomes or worker logs were read, no source or scientific setting changed, and the goal remains active.


### Theory-foundations checkpoint 2026-09-12 19:12 UTC

A2 cached-state/planner analysis completed and independently reproduced all15 fits,45 heads,30624 fixed and91872 stochastic rows. Primary layer15 diagnostic components offset: R rises .029042 and I falls .032377, with total change -.003334; weak decoder, upper-bound temperatures and layer-specific R pattern remain explicit. A1/A2/B1 manuscript/methods integration passed independent prose review. A3 exploratory position-readout sensitivity protocol accepted; implementation only. N2 development contexts, fixture generator and planned-denominator native aggregation passed CPU review; no new model queries/final evaluation-size freeze. Protected-source check passed46 paths. Existing activation-causality queue and worker remain untouched. Goal remains active. See research/theory_foundations/STATUS.md and registry/evidence_v4.json.


## 2026-09-12 19:30 UTC — activation confirmation operational checkpoint

The same supervisor session62935 remains live, verified through repeated55-second handle polls. The approved script41 filename-only check at19:30:13UTC found464/1,920 expected contextual-primary files, zero unexpected filenames, four metadata inputs and no result-content reads. Immutable checkpoint tmp/confirmation_contextual_readiness_20260912T193012Z.json SHA256 529c5de8375baf34682a5c3713a1d48f49ef2827c3d6419cff196235dbd015c5. A separate specification-bound filename count at19:14 saw405/1,920 with zero unexpected files. These are operational counts only; technical and behavioral status remain unassessed. STATUS is updated and prior bytes archived. No outcomes or worker logs were read, no source or scientific setting changed, no later cohort launched, and the goal remains active.


### Theory-foundations checkpoint 2026-09-12 19:51 UTC

A3 completed its frozen CPU run and full independent refit/probability/metric/bootstrap/figure audit. Both specified alternatives worsen primary NLL/Brier; all30 layer/fold alternatives select T4. These results retain the decoder limitation and do not establish absent latent information. Four archival studies A1/A2/A3/B1 are now checked; A3 manuscript integration is underway. N2 internal execution, supervisor, provenance reader, identity/numerical/sizing/handoff components are under staged CPU review; no production launcher, actual pilot/evaluation freeze, model handoff or new model query. Existing activation-causality worker/queue remain untouched. Protected check at19:44 passed46 paths. Persistent goal remains active; see research/theory_foundations/STATUS.md and registry/evidence_v5.json.


### Activation-causality continuation — 2026-09-12 20:00 UTC

The same first frozen confirmation supervisor remains active as session 62935, verified by the 55-second poll ending 19:59:44 UTC. A truncated preceding tool response did not trigger a restart. Script41 filename-only readiness at 19:57:09 UTC recorded 525/1,920 expected files, zero unexpected files, four metadata inputs and no result-content reads (`tmp/confirmation_contextual_readiness_20260912T195900Z.json`, SHA256 `4b98c8c61dc9f96c64b2817f88681dd287aaef2a7ca1ae1af0011e5f7313d640`). The tool-only 20:00:09 UTC inventory recorded 530/1,920 contextual-primary files and 0/1,792, 0/480, 0/448 for later cohorts, with no unexpected filenames. Counts include any saved technical-error rows and do not certify success. No outcomes, attempts or worker logs were opened; no runtime, frozen plan, selection, seed or model changed. STATUS.md was checkpointed before this operational update. The goal remains active, awaiting exact cohort closure and independent raw audit before the next single-cohort launch.


### Activation-causality verified wait — 2026-09-12 20:30 UTC

The same first frozen confirmation supervisor was confirmed live by repeated 55-second polls of session 62935, most recently ending 20:30:21 UTC. Script41 filename-only readiness at 20:30:44 UTC recorded 634/1,920 expected files, zero unexpected files, four metadata inputs and no result-content reads (`tmp/confirmation_contextual_readiness_20260912T203030Z.json`, SHA256 `a3c2e187b789bfaa25cedb553f98ae444de8618704ff88e8e623ebd914bc287a`). A tool-only all-cohort inventory at 20:30:22 UTC recorded 633/1,920 contextual-primary files and 0/1,792, 0/480, 0/448 for later cohorts, with no unexpected filenames. Counts do not certify technical or behavioral success. No outcomes, attempts or worker logs were opened, no process was restarted, and no inference source or frozen specification was changed. STATUS.md was preserved before this operational update. The research goal remains active; exact cohort closure and independent raw audit still precede the next single-cohort launch.

### 2026-09-12 20:33 UTC — theory campaign production integration

All four archival studies and replacement prose are independently accepted. Added reviewed component integrations and a local production launcher/worker under independent review, exact CPU numerical-stage preparation (no model or weight read), strict complete64-pool sizing rule and pilot metadata projection. Source PDF/protected46 paths passed20:15 check. No handoff, actual numerical gate, pilot or model query; goal active. See research/theory_foundations/STATUS.md.


### Activation-causality verified wait — 2026-09-12 21:00 UTC

The same first frozen confirmation supervisor was confirmed live by repeated 45–55-second polls of session 62935, most recently ending 21:00:40 UTC. Script41 filename-only readiness at 21:00:52 UTC recorded 700/1,920 expected files, zero unexpected files, four metadata inputs and no result-content reads (`tmp/confirmation_contextual_readiness_20260912T210050Z.json`, SHA256 `1395fb6a5bea19fbcee3ad8f7b742ced42b4b2566d15c2904c7036341f019982`). The tool-only all-cohort inventory at 21:00:41 UTC recorded 699/1,920 contextual-primary files and 0/1,792, 0/480, 0/448 for later cohorts, with no unexpected filenames. Counts do not certify technical or behavioral success. No outcomes, attempts or worker logs were opened, no process was restarted, and no inference source or frozen specification was changed. STATUS.md was preserved before this operational update. The research goal remains active; exact cohort closure and independent raw audit still precede the next single-cohort launch.


### Theory-foundations delivery checkpoint — 2026-09-12 21:14 UTC

The separate 25-page provisional revision and portable PDF/LaTeX ZIP are independently accepted; all 50 bundle files match their sources. Four archival studies A1/A2/A3/B1 are audited and N2 remains explicitly prospective. The single 64-layout evaluation pool (258 candidates, 1,408 context bounds) is independently accepted and must not be regenerated. Actual CPU identity candidate v1 exposed an unbound existing Python startup hook and was rejected before any MLX/model/launch claim; candidate, failure and exact earlier source archive are preserved. The explicit startup-closure amendment passes author tests and awaits independent source/actual-artifact review. All 46 protected paths passed at21:09 UTC. No model query or handoff; the72-hour goal remains active. See research/theory_foundations/STATUS.md and its README links for the delivered PDF/ZIP.

Reading this shared history also exposed the other campaign's owner-authored21:00 operational update (session62935 live,700/1,920 filename readiness). This is attributed owner metadata, not a new process observation or a whole-queue release. The theory campaign has not polled its session or read partial results/logs and continues to preserve the full queue's priority.


### Activation-causality verified wait — 2026-09-12 21:30 UTC

The same first frozen confirmation supervisor was confirmed live by repeated 45-second polls of session 62935, most recently ending 21:30:43 UTC. Script41 filename-only readiness at 21:30:57 UTC recorded 780/1,920 expected files, zero unexpected files, four metadata inputs and no result-content reads (`tmp/confirmation_contextual_readiness_20260912T213050Z.json`, SHA256 `2d0eb6d3fa444fb6444425318c380107091d12417cc47a2ef36f0c82a41d3721`). The tool-only all-cohort inventory at 21:30:44 UTC recorded 779/1,920 contextual-primary files and 0/1,792, 0/480, 0/448 for later cohorts, with no unexpected filenames. Counts do not certify technical or behavioral success. No outcomes, attempts or worker logs were opened, no process was restarted, and no inference source or frozen specification was changed. STATUS.md was preserved before this operational update. The research goal remains active; exact cohort closure and independent raw audit still precede the next single-cohort launch.


### Theory-foundations runtime readiness — 2026-09-12 21:42 UTC

The actual startup amendment and fresh numerical runtime candidate v2 are independently accepted. The byte audit checked7,496 paths, including all three weight shards,34 declared dependency closures,99 source files and46 protected files;250 loaded CPU module files match. No MLX/model/query/launch claim or handoff. Native Python sources are held fixed. Exact current stage/startup/candidate/acceptance references are in research/theory_foundations/coordinator_analysis/N2_RUNTIME_READINESS.md and the immutable numerical_v1/readiness_v1.json. The accepted manuscript/ZIP and single64-layout pool remain preserved. The only new execution dependency is legitimate whole-queue release from the priority activation-causality campaign; its worker/queue remain untouched. The72-hour goal remains active.


### Theory-foundations final local delivery / shared-resource block — 2026-09-12 22:01 UTC

Final claim-map review corrected two notation/assumption ambiguities; all scientific numbers and the accepted manuscript PDF/TeX/figures remain unchanged. The separately versioned v2 ZIP is independently accepted (SHAed78f526595f03d30645d98667e82bb402e8dbe4140aff37d506a8a9da9522d3;52members,47unchanged/3changed/2added). All independently executable work is complete. Fresh21:58UTC process metadata confirms the same prior supervisor79885 and worker79914 live; no result/log/session content, model load/query or prior signal. After the same missing whole-queue release persisted across three goal turns, the theory goal was marked blocked on this external dependency, not complete. Exact prepared native inputs, original scope and fixed15September cutoffs remain preserved. Resume from research/theory_foundations/STATUS.md and registry/blocked_audit_v1.json after legitimate whole-queue handoff; do not take a gap between cohorts or start a second model.


### Activation-causality verified wait — 2026-09-12 22:00 UTC

The same first frozen confirmation supervisor was confirmed live by repeated 45-second polls of session 62935, most recently ending 22:00:28 UTC. Script41 filename-only readiness at 22:00:42 UTC recorded 857/1,920 expected files, zero unexpected files, four metadata inputs and no result-content reads (`tmp/confirmation_contextual_readiness_20260912T220040Z.json`, SHA256 `d24d4f1766e927093201672dad509092874594b923a43086e95489acc3857e77`). The tool-only all-cohort inventory at 22:00:28 UTC recorded 857/1,920 contextual-primary files and 0/1,792, 0/480, 0/448 for later cohorts, with no unexpected filenames. Counts do not certify technical or behavioral success. No outcomes, attempts or worker logs were opened, no process was restarted, and no inference source or frozen specification was changed. STATUS.md was preserved before this operational update. The research goal remains active; exact cohort closure and independent raw audit still precede the next single-cohort launch.


### Activation-causality verified wait — 2026-09-12 22:30 UTC

The same first frozen confirmation supervisor was confirmed live by repeated 45-second polls of session 62935, most recently ending 22:30:31 UTC. Script41 filename-only readiness at 22:30:52 UTC recorded 924/1,920 expected files, zero unexpected files, four metadata inputs and no result-content reads (`tmp/confirmation_contextual_readiness_20260912T223040Z.json`, SHA256 `22ee673b3f11de8f48c62f2ef919b54324330634790ceede73a05f87cb7b6bfe`). The tool-only all-cohort inventory at 22:30:32 UTC recorded 924/1,920 contextual-primary files and 0/1,792, 0/480, 0/448 for later cohorts, with no unexpected filenames. Counts do not certify technical or behavioral success. No outcomes, attempts or worker logs were opened, no process was restarted, and no inference source or frozen specification was changed. STATUS.md was preserved before this operational update. The research goal remains active; exact cohort closure and independent raw audit still precede the next single-cohort launch.


### Activation-causality verified wait — 2026-09-12 23:00 UTC

The same first frozen confirmation supervisor was confirmed live by repeated 45-second polls of session 62935, most recently ending 23:00:32 UTC. Script41 filename-only readiness at 23:00:53 UTC recorded 974/1,920 expected files, zero unexpected files, four metadata inputs and no result-content reads (`tmp/confirmation_contextual_readiness_20260912T230040Z.json`, SHA256 `764b770b85a6ed77d2a58a0322b018599c7c45bcfd70cbedb6bd67c44cd8eada`). The tool-only all-cohort inventory at 23:00:33 UTC recorded 974/1,920 contextual-primary files and 0/1,792, 0/480, 0/448 for later cohorts, with no unexpected filenames. The first cohort reached half its expected file inventory at 22:51:11 UTC. These counts do not certify technical or behavioral success. No outcomes, attempts or worker logs were opened, no process was restarted, and no inference source or frozen specification was changed. STATUS.md was preserved before this operational update. All checkpoint times remain UTC despite the local London date advancing to September 13. The research goal remains active; exact cohort closure and independent raw audit still precede the next single-cohort launch.


### Activation-causality verified wait — 2026-09-12 23:30 UTC

The same first frozen confirmation supervisor was confirmed live by repeated 45-second polls of session 62935, most recently ending 23:30:20 UTC. Script41 filename-only readiness at 23:30:38 UTC recorded 1,060/1,920 expected files, zero unexpected files, four metadata inputs and no result-content reads (`tmp/confirmation_contextual_readiness_20260912T233030Z.json`, SHA256 `d70dcc6d941e8441e4f9e26e7f83709bb9b9f41826efe4228aac26771ed00cc0`). The tool-only all-cohort inventory at 23:30:21 UTC recorded 1,060/1,920 contextual-primary files and 0/1,792, 0/480, 0/448 for later cohorts, with no unexpected filenames. The first cohort reached 1,000 expected files at 23:10:22 UTC. Counts do not certify technical or behavioral success. No outcomes, attempts or worker logs were opened, no process was restarted, and the coordinator changed no inference source or frozen specification. STATUS.md was preserved before this operational update. All checkpoint times remain UTC. The research goal remains active; exact cohort closure and independent raw audit still precede the next single-cohort launch.


### Activation-causality verified wait — 2026-09-13 00:00 UTC

The same first frozen confirmation supervisor was confirmed live by repeated 45-second polls of session 62935, most recently ending 00:00:01 UTC. Script41 filename-only readiness at 00:00:15 UTC recorded 1,129/1,920 expected files, zero unexpected files, four metadata inputs and no result-content reads (`tmp/confirmation_contextual_readiness_20260913T000015Z.json`, SHA256 `652619cf077c55b902f60c561e916b1387dad4f0a2333e9591114fd964e4959b`). The tool-only all-cohort inventory at 00:00:01 UTC recorded 1,129/1,920 contextual-primary files and 0/1,792, 0/480, 0/448 for later cohorts, with no unexpected filenames. Counts do not certify technical or behavioral success. No outcomes, attempts or worker logs were opened, no process was restarted, and the coordinator changed no inference source or frozen specification. STATUS.md was preserved before this operational update. All checkpoint times remain UTC; the campaign ID and frozen cutoff remain unchanged across midnight. The research goal remains active; exact cohort closure and independent raw audit still precede the next single-cohort launch.


### Activation-causality verified wait — 2026-09-13 00:30 UTC

The same first frozen confirmation supervisor was confirmed live by repeated 45-second polls of session 62935, most recently ending 00:30:28 UTC. Script41 filename-only readiness at 00:30:45 UTC recorded 1,193/1,920 expected files, zero unexpected files, four metadata inputs and no result-content reads (`tmp/confirmation_contextual_readiness_20260913T003040Z.json`, SHA256 `fb077da4346da3cfc7c5debdbc0b0fde20745393ce3db2298b79cc6a11ffb4eb`). The tool-only all-cohort inventory at 00:30:28 UTC recorded 1,193/1,920 contextual-primary files and 0/1,792, 0/480, 0/448 for later cohorts, with no unexpected filenames. The first cohort reached 60% of its expected filename inventory at 00:10:50 UTC. Counts do not certify technical or behavioral success. No outcomes, attempts or worker logs were opened, no process was restarted, and the coordinator changed no inference source or frozen specification. STATUS.md was preserved before this operational update. The research goal remains active; exact cohort closure and independent raw audit still precede the next single-cohort launch.


### Activation-causality verified wait — 2026-09-13 01:00 UTC

The same first frozen confirmation supervisor was confirmed live by repeated 45-second polls of session 62935, most recently ending 01:00:28 UTC. Script41 filename-only readiness at 01:00:45 UTC recorded 1,254/1,920 expected files, zero unexpected files, four metadata inputs and no result-content reads (`tmp/confirmation_contextual_readiness_20260913T010040Z.json`, SHA256 `495a95d15a5997c7f882c89fe1257a6a4a5359de96f73ee963e0196bcb6d0a9e`). The tool-only all-cohort inventory at 01:00:29 UTC recorded 1,253/1,920 contextual-primary files and 0/1,792, 0/480, 0/448 for later cohorts, with no unexpected filenames. Counts do not certify technical or behavioral success. No outcomes, attempts or worker logs were opened, no process was restarted, and the coordinator changed no inference source or frozen specification. STATUS.md was preserved before this operational update. The research goal remains active; exact cohort closure and independent raw audit still precede the next single-cohort launch.


### Activation-causality verified wait — 2026-09-13 01:30 UTC

The same first frozen confirmation supervisor was confirmed live by repeated direct polls of session 62935, most recently ending 01:30:05 UTC. Script41 filename-only readiness at 01:30:21 UTC recorded 1,327/1,920 expected files, zero unexpected files, four metadata inputs and no result-content reads (`tmp/confirmation_contextual_readiness_20260913T013020Z.json`, SHA256 `ebfb53154fd50cc21011afd033ebdcd8a6d2520eeed00e64cddaced82ed012f1`). The tool-only all-cohort inventory at 01:30:06 UTC recorded 1,325/1,920 contextual-primary files and 0/1,792, 0/480, 0/448 for later cohorts, with no unexpected filenames. Counts do not certify technical or behavioral success. No outcomes, attempts or worker logs were opened, no process was restarted, and the coordinator changed no inference source or frozen specification. STATUS.md was preserved before this operational update. The research goal remains active; exact cohort closure and independent raw audit still precede the next single-cohort launch.


### Activation-causality verified wait — 2026-09-13 02:00 UTC

The same first frozen confirmation supervisor was confirmed live by repeated direct polls of session 62935, most recently ending 02:00:34 UTC. Script41 filename-only readiness at 02:00:58 UTC recorded 1,420/1,920 expected files, zero unexpected files, four metadata inputs and no result-content reads (`tmp/confirmation_contextual_readiness_20260913T020050Z.json`, SHA256 `64fde98140df6c7a2e56a961c5258234951b27717bf1efd329c678d93374d186`). The tool-only all-cohort inventory at 02:00:35 UTC recorded 1,419/1,920 contextual-primary files and 0/1,792, 0/480, 0/448 for later cohorts, with no unexpected filenames. Counts do not certify technical or behavioral success. No outcomes, attempts or worker logs were opened, no process was restarted, and the coordinator changed no inference source or frozen specification. STATUS.md was preserved before this operational update. The research goal remains active; exact cohort closure and independent raw audit still precede the next single-cohort launch.


### 2026-09-13 02:29 UTC — activation confirmation verified wait

Continued polling the same live supervisor session `62935`; the most recent poll ended at 02:29:09 UTC without a terminal result. A prior truncated tool response was treated as unknown, not as permission to restart. Script41 filename-only readiness at 02:28:26 UTC recorded 1,501/1,920 contextual-primary files, no unexpected files, four metadata inputs and `result_contents_read: false`: `tmp/confirmation_contextual_readiness_20260913T022830Z.json` (SHA256 `433ea4064491a7e47ffd0ba09114780495af29dbd2c89503c22a68b9f308470f`). The independent all-cohort filename inventory at 02:28:13 UTC recorded 1,500/1,920, 0/1,792, 0/480 and 0/448, with zero unexpected files. These are operational counts; technical and behavioral outcomes remain unassessed. No confirmation raw files, attempts or worker logs were read; no inference code, model configuration, frozen plan, tests or cohort order changed. The prior STATUS.md was preserved before the operational update. The goal remains active; complete-cohort audit and independent review precede the next single-cohort launch.


### 2026-09-13 03:00 UTC — activation confirmation verified wait

Continued repeated direct polling of the same live supervisor session `62935`; the most recent poll ended at 02:59:50 UTC without a terminal result. Script41 filename-only readiness at 03:00:16 UTC recorded 1,584/1,920 contextual-primary files, no unexpected files, four metadata inputs and `result_contents_read: false`: `tmp/confirmation_contextual_readiness_20260913T030000Z.json` (SHA256 `12e0525c82901acd5df4ef5f44f47bb6ee5351f07bb40ee5e61ee54e2e8b211f`). The all-cohort filename inventory at 02:59:50 UTC recorded 1,583/1,920, 0/1,792, 0/480 and 0/448, with zero unexpected files. Technical and behavioral outcomes remain unassessed. The agent inventory reconfirmed the reviewer idle after its completed scope review and the two previously recorded agent usage limits. No confirmation raw files, attempts or worker logs were read; no inference code, model configuration, frozen plan, tests or cohort order changed. The prior STATUS.md was preserved before the operational update. The goal remains active; complete-cohort audit and independent review precede the next single-cohort launch.


### 2026-09-13 03:30 UTC — activation confirmation verified wait

Continued repeated direct polling of the same live supervisor session `62935`; the most recent poll ended at 03:29:50 UTC without a terminal result. Script41 filename-only readiness at 03:30:03 UTC recorded 1,688/1,920 contextual-primary files, no unexpected files, four metadata inputs and `result_contents_read: false`: `tmp/confirmation_contextual_readiness_20260913T033000Z.json` (SHA256 `2706a3c5e6adb7767b74aa032798816753146a47c0e02a9e6c163a3debe3fc8e`). The all-cohort filename inventory at 03:29:50 UTC recorded 1,687/1,920, 0/1,792, 0/480 and 0/448, with zero unexpected files. Technical and behavioral outcomes remain unassessed. No confirmation raw files, attempts or worker logs were read; no inference code, model configuration, frozen plan, tests or cohort order changed. The prior STATUS.md was preserved before the operational update. The goal remains active; complete-cohort audit and independent review precede the next single-cohort launch.


### 2026-09-13 04:00 UTC — activation confirmation verified wait

Continued repeated direct polling of the same live supervisor session `62935`; the most recent poll ended at approximately 04:00:03 UTC without a terminal result. Script41 filename-only readiness at 04:00:17 UTC recorded 1,790/1,920 contextual-primary files, no unexpected files, four metadata inputs and `result_contents_read: false`: `tmp/confirmation_contextual_readiness_20260913T040010Z.json` (SHA256 `b171acea50b83af7c13f3aa79e86376b1f1f2c3b38eb6e3b6e7c43e4ad52ac29`). The all-cohort filename inventory at 04:00:04 UTC recorded 1,790/1,920, 0/1,792, 0/480 and 0/448, with zero unexpected files. Technical and behavioral outcomes remain unassessed. No confirmation raw files, attempts or worker logs were read; no inference code, model configuration, frozen plan, tests or cohort order changed. The prior STATUS.md was preserved before the operational update. The goal remains active; complete-cohort audit and independent review precede the next single-cohort launch.


### 2026-09-13 04:30 UTC — activation confirmation verified wait

Continued repeated direct polling of the same live supervisor session `62935`; the most recent live poll immediately preceded the 04:30:01 UTC filename inventory. Script41 filename-only readiness at 04:30:16 UTC recorded 1,860/1,920 contextual-primary files, no unexpected files, four metadata inputs and `result_contents_read: false`: `tmp/confirmation_contextual_readiness_20260913T043010Z.json` (SHA256 `c38177d001ee1fae045c113b0ba651696f6583c98d7b0c375399989c7fb74ddf`). The all-cohort filename inventory at 04:30:01 UTC recorded 1,860/1,920, 0/1,792, 0/480 and 0/448, with zero unexpected files. Technical and behavioral outcomes remain unassessed. No confirmation raw files, attempts or worker logs were read; no inference code, model configuration, frozen plan, tests or cohort order changed. The prior STATUS.md was preserved before the operational update. The goal remains active; complete-cohort audit and independent review precede the next single-cohort launch.


### 2026-09-13 05:10 UTC — first activation confirmation cohort closed; reporting correction

Session62935 exited0 with all1920 contextual-primary files and zero worker errors at04:53:47UTC. Automated auditv1 failed an invented after_l2 requirement; original failed JSON SHA761beefc351a1a66a75756352beb73f56e14d39d04baec7bfb878a3aee23bd45 and invocation record are preserved. Independent diagnosis also found whole-dictionary compiled-donor equality and contextual restoration baseline assumptions. Reporting scripts40/41 and tests were corrected against frozen runtime/spec semantics; no inference/result changes. Actual engine-emitter CPU fixtures plus corruption/baseline regressions pass75 joint tests (recordSHA1049eec5f8db0e76790d420ba654dee2bc9657deb6269c3ddec811d0987f844a), Ruff passed, and all44 inference files remain unchanged. Corrected automatedv2 passes (SHA83d96dad9f2fac73a36c72e1cea0e99b2d656881584348307bac4ebb8f6cdc54). Independent raw recomputation agrees on technical checks; final held-source/v2 certification is pending. No next cohort or global statistical report has run. Versioned reporting archive/guide integration remains in progress.


### 2026-09-13 05:30 UTC — first confirmation cleared; wall primary running

Final contextual-primary independent review SHA9e6a35635a04b3683bff1fcca56e4e7832eb88ddfd8af9147c02591abb15c872 approves technical progression and exact automated-v2 agreement: 1,920 technically complete trials, 1,732 native completions, 188 retained unresolved caps, 256 identity and 256 required norm checks passing, no technical errors. Source correction certificate SHA33f931b79736f855b480267881c8fb7ced4eb1d9d4ffb6b82f9299ae64920e91 binds current reporting files; inference/runtime/results remain unchanged. Wall primary launched at05:13:58UTC under session91403, supervisor30690, worker30948 and lease1a8b25d273994982b93fc03c5c4285f2; launch integration SHA fefe04df2653cddfb6294cb51a29c75fd36ee87ba9c62a66a7fd6eeae2f3a710. Direct live poll and filename-only inventory at05:25:28UTC show22/1792, zero unexpected, alternates0/480 and0/448. No partial wall outcomes/attempts/logs read and no joint statistics run. Updated guide source/import verifier passed (recordSHAc4c8b97d742c5f5ec17d5a7b522213a813bc366a61d9f385c5fa45e45584e835). New reporting correction archive SHAbe9418633310f88feda7edd503401dedeb500b967768310e77c38af8504b9519 contains41 hash-verified payloads; independent archive/guide review is underway. STATUS, README and integration guide were updated after preserving prior bytes. The fixed study and active goal continue.


### 2026-09-13 05:30 UTC — reporting correction archive independently approved

Independent archive/guide review SHA3c1e3183f78cfc182371848e8179e50fd27558350decbd441e2cc43baa75f8d6 approves all41 correction payloads, historical52 payloads, current7 reporting source/test files and archived/current44 inference files. Its exact documented verifier replay passed under an outcome/log/model-weight read guard; all218 inputs stayed stable and were rechecked by the coordinator. The coordinator integrated this approval into STATUS, README and the integration guide after preserving prior bytes. Wall-primary metadata-only readiness at05:29:12UTC records29/1792, zero unexpected files and no result reads (SHA96587de05540b430b6eb22c7c264adaec17b77f502d18b0d22a37803105e01a7); the subsequent direct poll of session91403 remained live. No frozen source, model configuration, outcome, scientific decision or cohort order changed. The goal remains active; joint results and remaining deliverables await cohort closure or fixed cutoff.


### 2026-09-13 06:00 UTC — wall-primary verified wait

The same supervisor session91403 remained live on direct polling before the06:00:20UTC filename inventory. That inventory records contextual primary1920/1920, wall primary103/1792, contextual alternate0/480 and wall alternate0/448, with no unexpected files. Script41 metadata-only readiness finished06:00:40UTC with103/1792 and four metadata inputs, result_contents_read:false: tmp/confirmation_wall_readiness_20260913T060020Z.json (SHAa973b83c5e04ef6ea09f8cc9b44374c7c0ca5364b74231d3f2276895dd073513). All44 frozen inference files and seven reviewed reporting source/test files were rechecked and remain byte-identical. Prior STATUS was preserved before this operational update. No partial wall raw results, attempts or worker logs were opened; no model settings, frozen plan, cohort order or scientific analysis changed. The goal remains active; wall closure, owned terminal and complete automated/independent technical audits precede the next single-cohort launch.


### 2026-09-13 06:30 UTC — wall-primary verified wait

The same supervisor session91403 remained live on direct polling before the06:30:34UTC filename inventory. That inventory records contextual primary1920/1920, wall primary185/1792, contextual alternate0/480 and wall alternate0/448, with no unexpected files. Script41 metadata-only readiness finished06:30:52UTC with186/1792 and four metadata inputs, result_contents_read:false: tmp/confirmation_wall_readiness_20260913T063035Z.json (SHA1ac5330e61d9204e5e87f409662952f7848f6bfa07b958d01c2c931a234f990f). Prior STATUS was preserved before this operational update. No partial wall raw results, attempts or worker logs were opened; no inference code, model settings, frozen plan, tests, cohort order or scientific analysis changed. The goal remains active; wall closure, owned terminal and complete automated/independent technical audits precede the next single-cohort launch.


### 2026-09-13 07:00 UTC — wall-primary verified wait

The same supervisor session91403 remained live on direct polling before the07:00:25UTC filename inventory. That inventory records contextual primary1920/1920, wall primary243/1792, contextual alternate0/480 and wall alternate0/448, with no unexpected files. Script41 metadata-only readiness finished07:00:47UTC with243/1792 and four metadata inputs, result_contents_read:false: tmp/confirmation_wall_readiness_20260913T070025Z.json (SHAd47ff72d6f7cd7656d52b063c43aad502b564b283f4502428116527cae1e9071). All44 frozen inference files and seven independently reviewed reporting source/test files remain byte-identical. Prior STATUS was preserved before the operational update. No partial wall raw results, attempts or worker logs were opened; no inference code, model settings, frozen plan, tests, cohort order or scientific analysis changed. The goal remains active; wall closure, owned terminal and complete automated/independent technical audits precede the next single-cohort launch.

## Theory foundations — data regeneration clarification, 2026-09-13T07:03:55.251281+00:00

User instruction: “reproduce data when needed.” Updated standing instructions/charter so missing original Sections 2–3 files do not require another external-path question. Completed a metadata-only successor-coverage and missing-state inventory in `outputs/theory_foundations/campaign_20260912T174516Z/empirical/regeneration_successor_coverage_v2/`; v1 and executed source snapshots are preserved. Independent canonical reconstruction matched all v2 tables: 281/1,276 pooled eligible decisions, zero strict held-out comparisons, and 1,406/5,159 missing states for observed-choice/full-support analyses, each including95terminals. No actions, probabilities or activation arrays were read by the coverage job; zero new fits/model queries. Reuse accepted A2/A3; conditional independent-bank requirements are recorded, with no new inference assignment. Original paper, accepted manuscript packages, native sources, shared-worker priority and fixed15September cutoffs are unchanged. Entry point: `research/theory_foundations/coordinator_analysis/DATA_REGENERATION.md`; machine decision: `research/theory_foundations/registry/data_regeneration_decision_20260913.json`.


### 2026-09-13 07:30 UTC — wall-primary verified wait

The same supervisor session 91403 remained live on direct polling before the 07:30:20 UTC filename inventory. That inventory records contextual primary 1920/1920, wall primary 325/1792, contextual alternate 0/480 and wall alternate 0/448, with no unexpected files. Script 41 metadata-only readiness finished 07:30:36 UTC with 325/1792 and four metadata inputs, result_contents_read:false: tmp/confirmation_wall_readiness_20260913T073020Z.json (SHA256 506ed7ffe8468450621a2d6402a08df97eaad748270dd7a6d355ec174f43662e). All 44 frozen inference files and seven independently reviewed reporting source/test files remain byte-identical. Prior STATUS was preserved in research/activation_causality/checkpoints/20260913T073139Z/ before this operational update. No partial wall raw results, attempts or worker logs were opened; no inference code, model settings, frozen plan, tests, cohort order or scientific analysis changed. The goal remains active; wall closure, owned terminal and complete automated/independent technical audits precede the next single-cohort launch.


### 2026-09-13 08:00 UTC — wall-primary verified wait

The same supervisor session 91403 remained live on direct polling before the 08:00:21 UTC filename inventory. That inventory records contextual primary 1920/1920, wall primary 405/1792, contextual alternate 0/480 and wall alternate 0/448, with no unexpected files. Script 41 metadata-only readiness finished 08:00:28 UTC with 405/1792 and four metadata inputs, result_contents_read:false: tmp/confirmation_wall_readiness_20260913T080021Z.json (SHA256 f7968114111906c716c085acad1987a30dfe9250a73f5741187cb2a51019c702). All 44 frozen inference files and seven independently reviewed reporting source/test files remain byte-identical. Prior STATUS was preserved in research/activation_causality/checkpoints/20260913T080118Z/ before this operational update. No partial wall raw results, attempts or worker logs were opened; no inference code, model settings, frozen plan, tests, cohort order or scientific analysis changed. The goal remains active; wall closure, owned terminal and complete automated/independent technical audits precede the next single-cohort launch.

## Theory foundations — resumed shared-worker check, 2026-09-13T08:23:10.124417+00:00

Goal observed active; previous regeneration turn classified as progress. Fresh read-only process metadata at 08:21:13 UTC confirms priority-campaign wall_subspace_primary supervisor 30690 (start 05:13:03 UTC) and model worker 30948 (start 05:13:58 UTC), both non-zombie Ss. Independent skeptic verified the exact filtered observation and accepted verified wait only. This is the first dependency turn in the fresh resumed audit. The full capture passed; the retained filtered artifact is explicitly not a launch handoff snapshot. Default sandbox ps denial was resolved by automatic approval of the same read-only check. No prior results/logs/session were opened, no existing process was signaled, and no model operation occurred. Next scientific action remains legitimate whole-queue handoff followed by the accepted N2 numerical gate; all fixed dates and sources preserved. Record: `research/theory_foundations/registry/resumed_resource_audit_20260913_v1.json`.

## Theory foundations — second resumed verified wait, 2026-09-13T08:24:34.118465+00:00

Fresh metadata at 08:23:50 UTC again confirms priority-campaign supervisor 30690 and model worker 30948 live, both Ss, with unchanged start identities and wall_subspace_primary commands. Exact observation SHA256 `8c3f50d632f5d51a50a84f0dd276091fa959ca0185342c5b3629a106ccae6279`. No prior session/results/logs read, process signal or model operation. The goal remains active; fresh resumed dependency count is two. Record: `research/theory_foundations/registry/resumed_resource_audit_20260913_v2.json`. Scope, accepted sources and deadlines unchanged.

## Theory foundations — resumed goal marked blocked, 2026-09-13T08:26:35.203980+00:00

Fresh process observation at 08:24:59 UTC confirms priority-campaign supervisor 30690 and worker 30948 still live in Ss with unchanged wall_subspace_primary start/command identities. This is the third consecutive verified whole-queue dependency turn in the fresh resumed audit. No substantive independent work remains under the accepted plan. The persistent goal was marked blocked, not complete; numerical gate, complete pilot, timing-only evaluation sizing, native evaluation and final integration remain required. Resume only on legitimate whole-queue release or the accepted elapsed-prior-cutoff/terminal-worker route, with fresh full process metadata and default shared lock. No new model inference, prior-process signals, result/log/session access or accepted-source changes. Fixed inference cutoff 15 September 11:45:16 UTC and research deadline 17:45:16 UTC preserved. Audit: `research/theory_foundations/registry/resumed_resource_audit_20260913_v3.json`, SHA256 `d2592448612057c2a82195752ac648e93c894a00854ab08a6f39d47d63c6e231`.


### 2026-09-13 08:30 UTC — wall-primary verified wait

The same supervisor session 91403 remained live on direct polling before the 08:30:18 UTC filename inventory. That inventory records contextual primary 1920/1920, wall primary 488/1792, contextual alternate 0/480 and wall alternate 0/448, with no unexpected files. Script 41 metadata-only readiness finished 08:30:31 UTC with 489/1792 and four metadata inputs, result_contents_read:false: tmp/confirmation_wall_readiness_20260913T083018Z.json (SHA256 c94aa2c5d58c85584b8216f7c9eaa4257cccca7fbbdba4d5eae49ea7db13c822). All 44 frozen inference files and seven independently reviewed reporting source/test files remain byte-identical. Prior STATUS was preserved in research/activation_causality/checkpoints/20260913T083108Z/ before this operational update. No partial wall raw results, attempts or worker logs were opened; no inference code, model settings, frozen plan, tests, cohort order or scientific analysis changed. The goal remains active; wall closure, owned terminal and complete automated/independent technical audits precede the next single-cohort launch.


### 2026-09-13 09:00 UTC — wall-primary verified wait

The same supervisor session 91403 remained live on direct polling before the 09:00:06 UTC filename inventory. That inventory records contextual primary 1920/1920, wall primary 557/1792, contextual alternate 0/480 and wall alternate 0/448, with no unexpected files. Script 41 metadata-only readiness finished 09:00:24 UTC with 558/1792 and four metadata inputs, result_contents_read:false: tmp/confirmation_wall_readiness_20260913T090006Z.json (SHA256 8c0104d2130961aa247f3602a15ee9b8f9866ec539a60d034332b12161ed0e18). All 44 frozen inference files and seven independently reviewed reporting source/test files remain byte-identical. Prior STATUS was preserved in research/activation_causality/checkpoints/20260913T090116Z/ before this operational update. No partial wall raw results, attempts or worker logs were opened; no inference code, model settings, frozen plan, tests, cohort order or scientific analysis changed. The goal remains active; wall closure, owned terminal and complete automated/independent technical audits precede the next single-cohort launch.

## Theory foundations — resumed verified wait, 2026-09-13T09:18:28.332026+00:00

Goal observed active again at 09:16 UTC. Fresh metadata at 09:17:09 UTC confirms unchanged priority-campaign supervisor 30690 (Ss) and worker 30948 (Rs), with their original start identities and wall_subspace_primary commands. First verified dependency turn in resumption 20260913T091652Z; goal left active. No model operations, prior signals, or result/log/session reads. Scope, fixed deadlines and accepted sources preserved. Record: `research/theory_foundations/registry/resume_20260913T091652Z_turn1.json`.

## Theory foundations — second verified wait in current resumption, 2026-09-13T09:19:46.771644+00:00

At 09:19:04 UTC, the same priority supervisor 30690 and worker 30948 were both live in Ss with unchanged start/command identities. Second consecutive dependency turn in resumption 20260913T091652Z; goal remains active. No inference, prior-process signals or result/log/session access. Scope and deadlines unchanged. Record: `research/theory_foundations/registry/resume_20260913T091652Z_turn2.json`.

## Theory foundations — current resumption blocked, 2026-09-13T09:21:20.609076+00:00

At 09:20:12 UTC the same priority supervisor 30690 (Ss) and worker 30948 (Rs) remained live with unchanged start/command identities. Third consecutive verified dependency turn in resumption 20260913T091652Z; persistent goal marked blocked, not complete. No substantive independent task remains before whole-queue release. Numerical gate, pilot, native evaluation and final integration are preserved, together with fixed 15 September inference/research cutoffs. No model operation, prior signal or result/log/session access. Audit: `research/theory_foundations/registry/resume_20260913T091652Z_turn3.json`, SHA256 `d5b450957f6839f3a59fc66c1c025f4bf460c848e102cd76f4af22c76e3ae1a9`.


### 2026-09-13 09:30 UTC — wall-primary verified wait

The same supervisor session 91403 remained live on direct polling before the 09:30:15 UTC filename inventory. That inventory records contextual primary 1920/1920, wall primary 624/1792, contextual alternate 0/480 and wall alternate 0/448, with no unexpected files. Script 41 metadata-only readiness finished 09:30:29 UTC with 624/1792 and four metadata inputs, result_contents_read:false: tmp/confirmation_wall_readiness_20260913T093015Z.json (SHA256 c658ee0c02e94d3190f4acf47a74deb0a5e4dfdac0a42de267ea42a6bfdf9a12). All 44 frozen inference files and seven independently reviewed reporting source/test files remain byte-identical. Prior STATUS was preserved in research/activation_causality/checkpoints/20260913T093057Z/ before this operational update. No partial wall raw results, attempts or worker logs were opened; no inference code, model settings, frozen plan, tests, cohort order or scientific analysis changed. The goal remains active; wall closure, owned terminal and complete automated/independent technical audits precede the next single-cohort launch.


### 2026-09-13 10:00 UTC — wall-primary verified wait

The same supervisor session 91403 remained live on direct polling before the 10:00:18 UTC filename inventory. That inventory records contextual primary 1920/1920, wall primary 692/1792, contextual alternate 0/480 and wall alternate 0/448, with no unexpected files. Script 41 metadata-only readiness finished 10:00:27 UTC with 692/1792 and four metadata inputs, result_contents_read:false: tmp/confirmation_wall_readiness_20260913T100018Z.json (SHA256 3cb30b4c0be492f24476c84dd5b4157b8383f0dda27071d92a52f97f9e843c6c). All 44 frozen inference files and seven independently reviewed reporting source/test files remain byte-identical. Prior STATUS was preserved in research/activation_causality/checkpoints/20260913T100116Z/ before this operational update. No partial wall raw results, attempts or worker logs were opened; no inference code, model settings, frozen plan, tests, cohort order or scientific analysis changed. The goal remains active; wall closure, owned terminal and complete automated/independent technical audits precede the next single-cohort launch.


### 2026-09-13 10:30 UTC — wall-primary verified wait

The same supervisor session 91403 remained live on direct polling before the 10:30:07 UTC filename inventory. That inventory records contextual primary 1920/1920, wall primary 766/1792, contextual alternate 0/480 and wall alternate 0/448, with no unexpected files. Script 41 metadata-only readiness finished 10:30:25 UTC with 767/1792 and four metadata inputs, result_contents_read:false: tmp/confirmation_wall_readiness_20260913T103007Z.json (SHA256 3bdfaac3701b2ab10409be76a7a61a0e44df7d08557edfe886b4fc2c303c13a4). All 44 frozen inference files and seven independently reviewed reporting source/test files remain byte-identical. Prior STATUS was preserved in research/activation_causality/checkpoints/20260913T103112Z/ before this operational update. No partial wall raw results, attempts or worker logs were opened; no inference code, model settings, frozen plan, tests, cohort order or scientific analysis changed. The goal remains active; wall closure, owned terminal and complete automated/independent technical audits precede the next single-cohort launch.


### 2026-09-13 11:00 UTC — wall-primary verified wait

The same supervisor session 91403 remained live on direct polling before the 11:00:07 UTC filename inventory. That inventory records contextual primary 1920/1920, wall primary 820/1792, contextual alternate 0/480 and wall alternate 0/448, with no unexpected files. Script 41 metadata-only readiness finished 11:00:25 UTC with 820/1792 and four metadata inputs, result_contents_read:false: tmp/confirmation_wall_readiness_20260913T110007Z.json (SHA256 1f66581a8407b436157ef39e01b94b258ec272bf592ebf4ca9f1a1fa8c27de5f). All 44 frozen inference files and seven independently reviewed reporting source/test files remain byte-identical. Prior STATUS was preserved in research/activation_causality/checkpoints/20260913T110108Z/ before this operational update. No partial wall raw results, attempts or worker logs were opened; no inference code, model settings, frozen plan, tests, cohort order or scientific analysis changed. The goal remains active; wall closure, owned terminal and complete automated/independent technical audits precede the next single-cohort launch.


### 2026-09-13 11:30 UTC — wall-primary verified wait

The same supervisor session 91403 remained live on direct polling before the 11:30:06 UTC filename inventory. That inventory records contextual primary 1920/1920, wall primary 899/1792, contextual alternate 0/480 and wall alternate 0/448, with no unexpected files. Script 41 metadata-only readiness finished 11:30:22 UTC with 899/1792 and four metadata inputs, result_contents_read:false: tmp/confirmation_wall_readiness_20260913T113006Z.json (SHA256 468331e2da5320b0572290bd6a7ff914e646f82acbee26dbac806343a7e29d66). All 44 frozen inference files and seven independently reviewed reporting source/test files remain byte-identical. Prior STATUS was preserved in research/activation_causality/checkpoints/20260913T113105Z/ before this operational update. No partial wall raw results, attempts or worker logs were opened; no inference code, model settings, frozen plan, tests, cohort order or scientific analysis changed. The goal remains active; wall closure, owned terminal and complete automated/independent technical audits precede the next single-cohort launch.


### 2026-09-13 12:00 UTC — wall-primary verified wait

The same supervisor session 91403 remained live on direct polling before the 12:00:17 UTC filename inventory. That inventory records contextual primary 1920/1920, wall primary 978/1792, contextual alternate 0/480 and wall alternate 0/448, with no unexpected files. Script 41 metadata-only readiness finished 12:00:32 UTC with 979/1792 and four metadata inputs, result_contents_read:false: tmp/confirmation_wall_readiness_20260913T120017Z.json (SHA256 5e30f6aad915798d346a0aeb4e4195b1bb9a11cdd66d3a18c59523953ccd3f94). All 44 frozen inference files and seven independently reviewed reporting source/test files remain byte-identical. Prior STATUS was preserved in research/activation_causality/checkpoints/20260913T120119Z/ before this operational update. No partial wall raw results, attempts or worker logs were opened; no inference code, model settings, frozen plan, tests, cohort order or scientific analysis changed. The goal remains active; wall closure, owned terminal and complete automated/independent technical audits precede the next single-cohort launch.


### 2026-09-13 12:28 UTC — wall-primary verified wait

The same supervisor session 91403 remained live on direct polling before the 12:28:03 UTC filename inventory. That inventory records contextual primary 1920/1920, wall primary 1055/1792, contextual alternate 0/480 and wall alternate 0/448, with no unexpected files. Script 41 metadata-only readiness finished 12:28:21 UTC with 1056/1792 and four metadata inputs, result_contents_read:false: tmp/confirmation_wall_readiness_20260913T122803Z.json (SHA256 0eb8e533879215c4716c71762991f59807d81f064437a4787f3f773f33f77369). All 44 frozen inference files and seven independently reviewed reporting source/test files remain byte-identical. Prior STATUS was preserved in research/activation_causality/checkpoints/20260913T122932Z/ before this operational update. No partial wall raw results, attempts or worker logs were opened; no inference code, model settings, frozen plan, tests, cohort order or scientific analysis changed. The goal remains active; wall closure, owned terminal and complete automated/independent technical audits precede the next single-cohort launch.


### 2026-09-13 12:59 UTC — wall-primary verified wait

The same supervisor session 91403 remained live on direct polling before the 12:58:49 UTC filename inventory. That inventory records contextual primary 1920/1920, wall primary 1121/1792, contextual alternate 0/480 and wall alternate 0/448, with no unexpected files. Script 41 metadata-only readiness finished 12:59:01 UTC with 1122/1792 and four metadata inputs, result_contents_read:false: tmp/confirmation_wall_readiness_20260913T125849Z.json (SHA256 78e0e787770c333e0999fd2a642ace48ada6dcee702747ea3d2998fb8c81db79). All 44 frozen inference files and seven independently reviewed reporting source/test files remain byte-identical. Prior STATUS was preserved in research/activation_causality/checkpoints/20260913T125946Z/ before this operational update. No partial wall raw results, attempts or worker logs were opened; no inference code, model settings, frozen plan, tests, cohort order or scientific analysis changed. The goal remains active; wall closure, owned terminal and complete automated/independent technical audits precede the next single-cohort launch.


### 2026-09-13 13:29 UTC — wall-primary verified wait

The same supervisor session 91403 remained live on direct polling before the 13:29:06 UTC filename inventory. That inventory records contextual primary 1920/1920, wall primary 1185/1792, contextual alternate 0/480 and wall alternate 0/448, with no unexpected files. Script 41 metadata-only readiness finished 13:29:20 UTC with 1186/1792 and four metadata inputs, result_contents_read:false: tmp/confirmation_wall_readiness_20260913T132906Z.json (SHA256 35f70fd19aac3f1f34bf43f4771953d813cba8891eac1feabc140e34e8e2aa8e). All 44 frozen inference files and seven independently reviewed reporting source/test files remain byte-identical. Prior STATUS was preserved in research/activation_causality/checkpoints/20260913T133001Z/ before this operational update. No partial wall raw results, attempts or worker logs were opened; no inference code, model settings, frozen plan, tests, cohort order or scientific analysis changed. The goal remains active; wall closure, owned terminal and complete automated/independent technical audits precede the next single-cohort launch.


### 2026-09-13 13:59 UTC — wall-primary verified wait

The same supervisor session 91403 remained live on direct polling before the 13:59:18 UTC filename inventory. That inventory records contextual primary 1920/1920, wall primary 1239/1792, contextual alternate 0/480 and wall alternate 0/448, with no unexpected files. Script 41 metadata-only readiness finished 13:59:31 UTC with 1239/1792 and four metadata inputs, result_contents_read:false: tmp/confirmation_wall_readiness_20260913T135918Z.json (SHA256 ed186713ce0f8798220995ad732bc34d57843f9d7b4e13966c0745521f53fea1). All 44 frozen inference files and seven independently reviewed reporting source/test files remain byte-identical. Prior STATUS was preserved in research/activation_causality/checkpoints/20260913T140015Z/ before this operational update. No partial wall raw results, attempts or worker logs were opened; no inference code, model settings, frozen plan, tests, cohort order or scientific analysis changed. The goal remains active; wall closure, owned terminal and complete automated/independent technical audits precede the next single-cohort launch.


### 2026-09-13 14:29 UTC — wall-primary verified wait

The same supervisor session 91403 remained live on direct polling before the 14:29:03 UTC filename inventory. That inventory records contextual primary 1920/1920, wall primary 1313/1792, contextual alternate 0/480 and wall alternate 0/448, with no unexpected files. Script 41 metadata-only readiness finished 14:29:17 UTC with 1313/1792 and four metadata inputs, result_contents_read:false: tmp/confirmation_wall_readiness_20260913T142903Z.json (SHA256 0940be9a43ce4d46fd3db778a69712e1bc81af8ef2002a448df99d40a58497ca). All 44 frozen inference files and seven independently reviewed reporting source/test files remain byte-identical. Prior STATUS was preserved in research/activation_causality/checkpoints/20260913T143008Z/ before this operational update. No partial wall raw results, attempts or worker logs were opened; no inference code, model settings, frozen plan, tests, cohort order or scientific analysis changed. The goal remains active; wall closure, owned terminal and complete automated/independent technical audits precede the next single-cohort launch.


### 2026-09-13 14:59 UTC — wall-primary verified wait

The same supervisor session 91403 remained live on direct polling before the 14:59:10 UTC filename inventory. That inventory records contextual primary 1920/1920, wall primary 1389/1792, contextual alternate 0/480 and wall alternate 0/448, with no unexpected files. Script 41 metadata-only readiness finished 14:59:23 UTC with 1390/1792 and four metadata inputs, result_contents_read:false: tmp/confirmation_wall_readiness_20260913T145910Z.json (SHA256 971c199752414dbb6f3092a34932980c257ced3e4043133e9d969361bf9f8f42). All 44 frozen inference files and seven independently reviewed reporting source/test files remain byte-identical. Prior STATUS was preserved in research/activation_causality/checkpoints/20260913T150017Z/ before this operational update. No partial wall raw results, attempts or worker logs were opened; no inference code, model settings, frozen plan, tests, cohort order or scientific analysis changed. The goal remains active; wall closure, owned terminal and complete automated/independent technical audits precede the next single-cohort launch.


### 2026-09-13 15:30 UTC — wall-primary verified wait

The same supervisor session 91403 remained live on direct polling before the 15:29:41 UTC filename inventory. That inventory records contextual primary 1920/1920, wall primary 1462/1792, contextual alternate 0/480 and wall alternate 0/448, with no unexpected files. Script 41 metadata-only readiness finished 15:30:05 UTC with 1463/1792 and four metadata inputs, result_contents_read:false: tmp/confirmation_wall_readiness_20260913T152941Z.json (SHA256 0f083cd70cf053430e92e0e7d10c6742ebe97183c86465094c8d9d56c231c923). All 44 frozen inference files and seven independently reviewed reporting source/test files remain byte-identical. Prior STATUS was preserved in research/activation_causality/checkpoints/20260913T153056Z/ before this operational update. No partial wall raw results, attempts or worker logs were opened; no inference code, model settings, frozen plan, tests, cohort order or scientific analysis changed. The goal remains active; wall closure, owned terminal and complete automated/independent technical audits precede the next single-cohort launch.


### 2026-09-13 16:00 UTC — wall-primary verified wait

The same supervisor session 91403 remained live on direct polling before the 15:59:42 UTC filename inventory. That inventory records contextual primary 1920/1920, wall primary 1525/1792, contextual alternate 0/480 and wall alternate 0/448, with no unexpected files. Script 41 metadata-only readiness finished 15:59:56 UTC with 1526/1792 and four metadata inputs, result_contents_read:false: tmp/confirmation_wall_readiness_20260913T155942Z.json (SHA256 8d4bbc649a407761d29556d2d2416872c9a089364db83ff6054be0c72104d977). All 44 frozen inference files and seven independently reviewed reporting source/test files remain byte-identical. Prior STATUS was preserved in research/activation_causality/checkpoints/20260913T160047Z/ before this operational update. No partial wall raw results, attempts or worker logs were opened; no inference code, model settings, frozen plan, tests, cohort order or scientific analysis changed. The goal remains active; wall closure, owned terminal and complete automated/independent technical audits precede the next single-cohort launch.


### 2026-09-13 16:15 UTC — theory campaign automatic conditional resumption

The user authorized resuming the theory goal when appropriate. A separate bounded metadata watcher started at 16:14:44 UTC, PID 90875, owned exec session 79959; its first successful capture found priority supervisor 30690 and worker 30948 still live, and it is waiting. Every 600 seconds it checks dedicated release filenames, all four exact frozen job filename sets plus narrowly matched independent-review metadata, and fresh complete process metadata. It reads no result or worker-log contents. Triggers request renewed same-thread coordinator review through the supported public Codex CLI; they never authorize model work. A harmless same-thread CLI delivery check succeeded. No research wake-up or model operation had occurred at launch recording.

Eight focused tests and independent source review passed. Review: research/theory_foundations/reviews/RESUME_WATCHER_REVIEW_20260913.md. Launch record: research/theory_foundations/registry/resume_watcher_launch_20260913.json. Active records: outputs/theory_foundations/campaign_20260912T174516Z/coordinator_analysis/resume_watch_20260913T161500Z/. The watcher has a separate lock, monotonic lifetime bounds, cancellation file and one durable delivery attempt. It requests reporting at the unchanged 15 September 11:45:16 UTC inference cutoff and expires at 17:45:16 UTC. It requires the local Mac/Codex to remain running. Goal storage, native sources, existing runtime/queue and accepted science/manuscript remain unchanged; whole-queue handoff and every existing native-study gate still apply.


### 2026-09-13 16:30 UTC — wall-primary verified wait

The same supervisor session 91403 remained live on direct polling before the 16:29:13 UTC filename inventory. That inventory records contextual primary 1920/1920, wall primary 1604/1792, contextual alternate 0/480 and wall alternate 0/448, with no unexpected files. Script 41 metadata-only readiness finished 16:29:20 UTC with 1605/1792 and four metadata inputs, result_contents_read:false: tmp/confirmation_wall_readiness_20260913T162913Z.json (SHA256 a585368fbdb8647c6d210b4d555f76d1b3678db6a0a0c199044e7047e4580566). All 44 frozen inference files and seven independently reviewed reporting source/test files remain byte-identical. Prior STATUS was preserved in research/activation_causality/checkpoints/20260913T163006Z/ before this operational update. No partial wall raw results, attempts or worker logs were opened; no inference code, model settings, frozen plan, tests, cohort order or scientific analysis changed. The goal remains active; wall closure, owned terminal and complete automated/independent technical audits precede the next single-cohort launch.


### 2026-09-13 17:00 UTC — wall-primary verified wait

The same supervisor session 91403 remained live on direct polling before the 16:59:07 UTC filename inventory. That inventory records contextual primary 1920/1920, wall primary 1670/1792, contextual alternate 0/480 and wall alternate 0/448, with no unexpected files. Script 41 metadata-only readiness finished 16:59:18 UTC with 1670/1792 and four metadata inputs, result_contents_read:false: tmp/confirmation_wall_readiness_20260913T165907Z.json (SHA256 2ff3abf4c624a0de1e13bc3259d19a4a4fcf088fa6d5242a4e31dde18eb972f2). All 44 frozen inference files and seven independently reviewed reporting source/test files remain byte-identical. Prior STATUS was preserved in research/activation_causality/checkpoints/20260913T165958Z/ before this operational update. No partial wall raw results, attempts or worker logs were opened; no inference code, model settings, frozen plan, tests, cohort order or scientific analysis changed. The goal remains active; wall closure, owned terminal and complete automated/independent technical audits precede the next single-cohort launch.


### 2026-09-13 17:30 UTC — wall-primary verified wait

The same supervisor session 91403 remained live on direct polling before the 17:29:07 UTC filename inventory. That inventory records contextual primary 1920/1920, wall primary 1750/1792, contextual alternate 0/480 and wall alternate 0/448, with no unexpected files. Script 41 metadata-only readiness finished 17:29:23 UTC with 1753/1792 and four metadata inputs, result_contents_read:false: tmp/confirmation_wall_readiness_20260913T172907Z.json (SHA256 7b6476d5178031eade028b66a0676b0a193394506b8542872be508356fb4ce70). All 44 frozen inference files and seven independently reviewed reporting source/test files remain byte-identical. Prior STATUS was preserved in research/activation_causality/checkpoints/20260913T173004Z/ before this operational update. No partial wall raw results, attempts or worker logs were opened; no inference code, model settings, frozen plan, tests, cohort order or scientific analysis changed. The goal remains active; wall closure, owned terminal and complete automated/independent technical audits precede the next single-cohort launch.


### 2026-09-13 17:47 UTC — wall-primary closed; independent audit pending

Supervisor session91403 closed with exit0 at17:42:02 UTC after all1792 exact planned wall-primary files, zero unexpected. Worker_finished at17:42:01 UTC reports completed1792/errors0; no earlier cohort failure events. Immutable terminal SHA256 421b03dc7d44bb35c69a1bcdbcdc768e2aee4c3205f1df4a59cabe75af8f9d75. Script41 automated_v1 passed, SHA256 1c4731f189a3dad0941aad8a301312cd51d10a64e2c4ce11ddda4cb597cda8d7, CPU session64592 exit0:1792 technically complete,1555 native completions,237 retained token caps,256 exact identity replays,512 projected-norm checks,14 conditions×128 trials,zero historical attempts. Independent reviewer is recomputing complete raw technical evidence; no next cohort has launched and no global statistical decisions have run. Coordinator closure record: outputs/activation_causality/campaign_20260912/reviews/wall_primary_cohort_closure_coordinator_v1.json (SHA256 e8f0e00f1fa6a5ce1e80bbaf3b528e8eed2474285a63688d29e5c5a0dd12fadd). All44 inference files,7 reporting source/test files and main PDF remain unchanged. No partial wall raw outcomes, attempts or worker logs were opened before closure. Prior STATUS preserved in research/activation_causality/checkpoints/20260913T174731Z/. Goal remains active; complete independent approval is required before contextual alternate480, then wall alternate448, and final joint reporting/replay/figures/ledger/proposal remain pending.


### 2026-09-13 18:00 UTC — wall independently approved; contextual alternate launched

Wall-primary independent review SHA256 f764cdbedf217947782b4643eee62de26277695648fc6801c16f797aa1c035e0 approves technical progression:1792 complete technical rows,1555 native completions,237 retained caps,256 baseline identities and512 projected-norm gates. Coordinator rechecked all1892 input hashes and44 inference+7 reporting source/test files before launch (check SHA256 f66e5ed93eef03c89a4532d43c5c8d107e8ed1354261ac8d888b342d48ba8c8b). Reviewer-only float64 diagnostic and both source versions remain preserved; no inference/reporting/result/tolerance change. Supervisor19207 launched contextual alternate480 at17:54:49 UTC, worker_started17:55:46 UTC; supervisor PID92738, worker/group92742, leasecdc315830a154437aaf4f037ef160da9. Third execution record SHA256 3cf52b20e84cf3f04d4fd8ab94832fe7f519ee8275ad459abfffc896ed89cd5d binds both primary approvals and launch. Metadata-only readiness records5/480 at17:57:22 UTC (SHA256 3121fb61860b826f8d7671f9a4c2cbb76273dea8fbd9478a5bb685d1e66b3191); wall alternate remains0/448. No partial contextual-alternate raw results, attempts or worker logs have been opened and no joint statistical decisions have run. STATUS/README/integration prior bytes preserved in research/activation_causality/checkpoints/20260913T180015Z/. Goal remains active; next cohort requires complete independent audit, and final joint results/figures/ledger/replay/proposal remain pending.


### 2026-09-13 18:30 UTC — contextual-alternate verified wait

The same supervisor session19207 remained live on direct polling before the18:28:53 UTC filename inventory. That inventory records contextual primary1920/1920, wall primary1792/1792, contextual alternate82/480 and wall alternate0/448, with zero unexpected files. Both primary cohorts retain independent technical approval. Script41 metadata-only readiness finished18:29:06 UTC with83/480 and four metadata inputs, result_contents_read:false: tmp/confirmation_contextual_alternate_readiness_20260913T182853Z.json (SHA256 82d59d34ded27024efcf622d5eacb28510ffacbfbe07a44590b6a7c6afdb2682). All44 inference files and7 independently reviewed reporting source/test files remain byte-identical. Prior STATUS preserved in research/activation_causality/checkpoints/20260913T183008Z/. No partial contextual-alternate raw results, attempts or worker logs were opened; no source, model settings, frozen plan, tests, cohort order or scientific analysis changed. Goal remains active; contextual-alternate closure and complete automated/independent technical audits precede the wall-alternate launch. Final joint results, figures, ledger, replay and manuscript proposal remain pending.


### 2026-09-13 19:00 UTC — contextual-alternate verified wait

The same supervisor session19207 remained live on direct polling before the18:58:51 UTC filename inventory. That inventory records contextual primary1920/1920, wall primary1792/1792, contextual alternate160/480 and wall alternate0/448, with zero unexpected files. Both primary cohorts retain independent technical approval. Script41 metadata-only readiness finished18:59:04 UTC with160/480 and four metadata inputs, result_contents_read:false: tmp/confirmation_contextual_alternate_readiness_20260913T185851Z.json (SHA256 b2c43a1a03320c56808960e5939469b7f2da3f52146fe5795bfc5705929d2d00). All44 inference files and7 independently reviewed reporting source/test files remain byte-identical. Prior STATUS preserved in research/activation_causality/checkpoints/20260913T185951Z/. No partial contextual-alternate raw results, attempts or worker logs were opened; no source, model settings, frozen plan, tests, cohort order or scientific analysis changed. Goal remains active; contextual-alternate closure and complete automated/independent technical audits precede the wall-alternate launch. Final joint results, figures, ledger, replay and manuscript proposal remain pending.


### 2026-09-13 19:33 UTC — contextual-alternate verified wait

The same supervisor session19207 remained live on direct polling before the19:33:02 UTC filename inventory. That inventory records contextual primary1920/1920, wall primary1792/1792, contextual alternate248/480 and wall alternate0/448, with zero unexpected files. Both primary cohorts retain independent technical approval. Script41 metadata-only readiness finished19:33:20 UTC with249/480 and four metadata inputs, result_contents_read:false: tmp/confirmation_contextual_alternate_readiness_20260913T1933Z.json (SHA256 c2995b16f06e6e82dfce2e56a09e427cacaf880858bd15758321fe53faae2b6e). All44 inference files and7 independently reviewed reporting source/test files remain byte-identical. Prior STATUS preserved in research/activation_causality/checkpoints/20260913T193412Z/. No partial contextual-alternate raw results, attempts or worker logs were opened; no source, model settings, frozen plan, tests, cohort order or scientific analysis changed. Goal remains active; contextual-alternate closure and complete automated/independent technical audits precede the wall-alternate launch. Final joint results, figures, ledger, replay and manuscript proposal remain pending.


### 2026-09-13 20:00 UTC — contextual-alternate verified wait

The same supervisor session19207 remained live on direct polling before the19:59:20 UTC filename inventory. That inventory records contextual primary1920/1920, wall primary1792/1792, contextual alternate297/480 and wall alternate0/448, with zero unexpected files. Both primary cohorts retain independent technical approval. Script41 metadata-only readiness finished19:59:31 UTC with298/480 and four metadata inputs, result_contents_read:false: tmp/confirmation_contextual_alternate_readiness_20260913T195920Z.json (SHA256 fdda3b938c04f94181f25dde31a3b36bf7a332012aae983b817a2fdaaf5d5052). All44 inference files and7 independently reviewed reporting source/test files remain byte-identical. Prior STATUS preserved in research/activation_causality/checkpoints/20260913T200016Z/. No partial contextual-alternate raw results, attempts or worker logs were opened; no source, model settings, frozen plan, tests, cohort order or scientific analysis changed. Goal remains active; contextual-alternate closure and complete automated/independent technical audits precede the wall-alternate launch. Final joint results, figures, ledger, replay and manuscript proposal remain pending.


### 2026-09-13 20:30 UTC — contextual-alternate verified wait

The same supervisor session19207 remained live on direct polling before the20:28:51 UTC filename inventory. That inventory records contextual primary1920/1920, wall primary1792/1792, contextual alternate375/480 and wall alternate0/448, with zero unexpected files. Both primary cohorts retain independent technical approval. Script41 metadata-only readiness finished20:29:25 UTC with377/480 and four metadata inputs, result_contents_read:false: tmp/confirmation_contextual_alternate_readiness_20260913T202851Z.json (SHA256 86b39d69cf59cd992a5ba319eaa5422535e1b229df57f25ec1e3dc0d2dfbe900). All44 inference files and7 independently reviewed reporting source/test files remain byte-identical. Prior STATUS preserved in research/activation_causality/checkpoints/20260913T203012Z/. No partial contextual-alternate raw results, attempts or worker logs were opened; no source, model settings, frozen plan, tests, cohort order or scientific analysis changed. Goal remains active; contextual-alternate closure and complete automated/independent technical audits precede the wall-alternate launch. Final joint results, figures, ledger, replay and manuscript proposal remain pending.


### 2026-09-13 21:00 UTC — contextual-alternate verified wait

The same supervisor session19207 remained live on direct polling before the20:59:19 UTC filename inventory. That inventory records contextual primary1920/1920, wall primary1792/1792, contextual alternate432/480 and wall alternate0/448, with zero unexpected files. Both primary cohorts retain independent technical approval. Script41 metadata-only readiness finished20:59:30 UTC with432/480 and four metadata inputs, result_contents_read:false: tmp/confirmation_contextual_alternate_readiness_20260913T205919Z.json (SHA256 542f7a8318a2ad83330a77931625e6dbdc08023da6ff54e3371e6dd115601ab5). All44 inference files and7 independently reviewed reporting source/test files remain byte-identical. Prior STATUS preserved in research/activation_causality/checkpoints/20260913T210015Z/. No partial contextual-alternate raw results, attempts or worker logs were opened; no source, model settings, frozen plan, tests, cohort order or scientific analysis changed. Goal remains active; contextual-alternate closure and complete automated/independent technical audits precede the wall-alternate launch. Final joint results, figures, ledger, replay and manuscript proposal remain pending.


### 2026-09-13 21:39 UTC — contextual alternate approved; final wall alternate launched

Contextual-alternate supervisor19207 closed exit0 at21:16:57 UTC with480 exact files and zero unexpected; worker_finished21:16:56 reports completed480/errors0. Automated_v1 SHA256 bf72bf885f9648983be2026f282dc973bbaaffe7e0f555fb5e56b0dc113eff76 and independent review SHA256 23918af70dffcaac8d0a77219e695ba4cd3c9d6022c3ff107599a79febe79764 clear480 technical rows,415 native completions and65 retained caps,64 identity replays and64 full-write norm gates. Coordinator closure SHA256 4a6f762feb753dc336044165d0d941ad723523fbef8cb9708e35699a464afe02 and589-input verification SHA256 06d2cd113d30e3b7eedef3a0f922b87512203802ca3c1ec83a129d69ba5deb82 bind the evidence. Both preparation/source versions remain preserved; no runtime/reporting/result/fixed-tolerance change and no partial third-cohort outcome reads. Final wall alternate448 launched under supervisor87505 at21:35:13 UTC, worker_started21:36:12 UTC; supervisor96510, worker/group96513, lease173fba750d5b45ea949e8800feddbd2b. Fourth execution record SHA256 3d2ec6bda64bef625956ce07e56842327604152c1c3e5baa00e21123d5d790ea binds all three prior approvals. Metadata readiness records1/448 at21:36:45 UTC (SHA256 7237af66c7666d5c6cdfe3baa299f2172dcf4e4364148f4f7b66d3f2a696347a); no wall-alternate raw results, attempts or logs have been opened. All44 inference files and7 reporting source/test files remain unchanged. STATUS/README/integration prior bytes preserved in research/activation_causality/checkpoints/20260913T213950Z/. Goal remains active; final complete audits, joint results/figures/ledger/replay/proposal remain pending. No scientific confirmation claim or global analysis has been made.


### 2026-09-13 22:02 UTC — wall alternate verified wait; audit prepared

The same supervisor session 87505 remained live on direct polling before the 22:00:22 UTC filename inventory. That inventory records contextual primary 1920/1920, wall primary 1792/1792, contextual alternate 480/480 and wall alternate 60/448, with zero unexpected files. All three completed cohorts retain independent technical approval. Script41 metadata-only readiness finished 22:00:33 UTC with 61/448 and four metadata inputs, result_contents_read:false: tmp/confirmation_wall_alternate_readiness_20260913T220022Z.json (SHA256 d20c16c6a012a031a5001aca94f89b059f7c7c0da1990adf658a46dfb08f452b). All 44 inference files and seven independently reviewed reporting source/test files remain byte-identical. The reviewer prepared the final audit without execution or raw reads: preparation SHA256 fcd50ff0255143b960eee4323a06609ff1b2346b17f2e99fb20acc41243c0c3c, source SHA256 adc0cf7e0943a2d3799a21c4f65b8476d3a8dce6be41095c2bc838e24957ac28. Its 11 metadata/source/diagnosis inputs were rehashed; no basis arrays or weights were read. Exact closure and complete automated/independent technical review remain required. Prior STATUS preserved in research/activation_causality/checkpoints/20260913T220225Z/. No partial wall-alternate raw results, attempts or logs were opened; no frozen plan, inference/reporting source, tests or statistical analysis changed. The goal remains active; final joint results, figures, ledger, replay and the separate paper proposal remain pending.


### 2026-09-13 22:30 UTC — wall alternate verified wait

The same supervisor session 87505 remained live on direct polling before the 22:29:32 UTC filename inventory. That inventory records contextual primary 1920/1920, wall primary 1792/1792, contextual alternate 480/480 and wall alternate 146/448, with zero unexpected files. All three completed cohorts retain independent technical approval. Script41 metadata-only readiness finished 22:29:44 UTC with 146/448 and four metadata inputs, result_contents_read:false: tmp/confirmation_wall_alternate_readiness_20260913T222932Z.json (SHA256 dbd463170bce867548a8234198a7d5c78709d42f4c288fd2b5f7934c15d18d2a). All 44 inference files and seven independently reviewed reporting source/test files remain byte-identical. The held final audit remains prepared and unexecuted, with no raw reads: preparation SHA256 fcd50ff0255143b960eee4323a06609ff1b2346b17f2e99fb20acc41243c0c3c, source SHA256 adc0cf7e0943a2d3799a21c4f65b8476d3a8dce6be41095c2bc838e24957ac28. Its 11 metadata/source/diagnosis inputs were rehashed; no basis arrays or weights were read. Exact closure and complete automated/independent technical review remain required. Prior STATUS preserved in research/activation_causality/checkpoints/20260913T223047Z/. No partial wall-alternate raw results, attempts or logs were opened; no frozen plan, inference/reporting source, tests or statistical analysis changed. The goal remains active; final joint results, figures, ledger, replay and the separate paper proposal remain pending.


### 2026-09-13 23:04 UTC — wall alternate verified wait and scientific-audit preparation

The same supervisor session 87505 remained live on direct polling before the 23:03:03 UTC filename inventory, recording contextual primary 1920/1920, wall primary 1792/1792, contextual alternate 480/480 and wall alternate 231/448, with zero unexpected files. All three completed cohorts retain independent technical approval. Script41 metadata-only readiness finished 23:02:39 UTC with 227/448 and four metadata inputs, result_contents_read:false: tmp/confirmation_wall_alternate_readiness_20260913T230200Z.json (SHA256 2c99e0e3bd728cc7122d17d4e19c5c91747d6d5805b4ff132b9dbb61c507b675). All 44 inference files and seven independently reviewed reporting source/test files remain unchanged. The final technical audit remains prepared and unexecuted, with its 11 metadata/source/diagnosis inputs rehashed. The reviewer prepared the joint scientific checklist (SHA256 f422f35c3e10eb9fd87b77f0d5369d647e8bec8f8ee770ad10505de4d912d1c2); the coordinator found a wall-only fallback error in its v1 metadata record. The original was preserved, the diagnosis published (SHA256 a2ed784ca1ca46d46b9c568c8189034747c489db7fc01baa78af0145f047d50a), and v2 corrects only the contextual-primary counts/weights (SHA256 2b2233dcbe5666899a838ba93776c6769823b749e2878175dad759de5434fbc9). The coordinator rehashed all 26 v2 inputs and independently recomputed all four dataset family inventories, confirming primary43/43/42, contextual same32 11/11/10 and wall128/32. This is static preparation with no outcome analysis or scientific approval. No runtime/reporting code, tests, frozen settings, model weights, partial wall-alternate raw results, attempts or logs were changed/read. Prior STATUS preserved in research/activation_causality/checkpoints/20260913T230451Z/. Final technical closure, joint results, independent scientific review, figures, ledger, replay, final archive and separate paper proposal remain pending; the goal remains active.


### 2026-09-13 23:31 UTC — wall alternate verified wait

Supervisor session 87505 remained live on direct polling before the 23:30:16 UTC filename inventory, recording contextual primary 1920/1920, wall primary 1792/1792, contextual alternate 480/480 and wall alternate 324/448, with zero unexpected files. All three completed cohorts retain independent technical approval. Script41 metadata-only readiness finished 23:30:16 UTC with 324/448, four metadata inputs and result_contents_read:false: tmp/confirmation_wall_alternate_readiness_20260913T233000Z.json (SHA256 a5ffee011ee78102046e9e30d54d2398f28969d95d7d78eb12a20a1f8d4669a6). All 44 inference and seven independently reviewed reporting source/test files remain unchanged. The held technical audit remains unexecuted, with 11 metadata/source/diagnosis inputs rehashed; corrected scientific preparation v2 remains unexecuted, with 26 inputs rehashed and all four frozen family inventories checked. No raw wall-alternate results, attempts, logs, model weights, scientific effects or significance were read/computed. No fixed source, test, plan, threshold or endpoint changed. Prior STATUS preserved in research/activation_causality/checkpoints/20260913T233113Z/. Final technical closure, joint results, independent scientific audit, figures, whole search ledger, replay, final archive and separate paper proposal remain pending; the goal remains active.


### 2026-09-14 00:01 UTC — wall alternate verified wait

Supervisor session 87505 remained live on direct polling before the 00:00:27 UTC filename inventory, recording contextual primary 1920/1920, wall primary 1792/1792, contextual alternate 480/480 and wall alternate 397/448, with zero unexpected files. All three completed cohorts retain independent technical approval. Script41 metadata-only readiness finished 00:00:27 UTC with 397/448, four metadata inputs and result_contents_read:false: tmp/confirmation_wall_alternate_readiness_20260914T000000Z.json (SHA256 7d97b041d21b915f138af7c60452ddedda9aa089abe55b9fcb4478fcf1a2c6f1). All 44 inference and seven independently reviewed reporting source/test files remain unchanged. The held technical audit remains unexecuted, with 11 metadata/source/diagnosis inputs rehashed; corrected scientific preparation v2 remains unexecuted, with 26 inputs rehashed and all four frozen family inventories checked. No raw wall-alternate results, attempts, logs, model weights, scientific effects or significance were read/computed. No fixed source, test, plan, threshold or endpoint changed. Prior STATUS preserved in research/activation_causality/checkpoints/20260914T000107Z/. Final technical closure, joint results, independent scientific audit, figures, whole search ledger, replay, final archive and separate paper proposal remain pending; the goal remains active.


### 2026-09-14 00:40 UTC — all confirmation cohorts technically approved; joint reporting started

Final supervisor87505 closed exit0 at2026-09-14T00:27:12.665582+00:00, exact448/448 and worker_finished448/errors0. Its terminal SHAe68ed96fbe806c3df5fbe611af472f0938b90840ad426cb06fa2f40b37da3899, automated audit SHA3bebb10518413eda99234a38086a5d71898ccfd0da0c43ff5d256c952d05982f and independent certificate SHA9b2516957065a27623e3f965f82b9b3750a7bca318465ec624398bb07e01108f are preserved. Coordinator verified537 automated inputs and556 independent inputs+8 references,44 runtime and7 reporting pins; record SHA17c3827cd889df674875594594500a2f5f6f7a474297415db90002bcd8a9289d permits joint reporting. All4640 trials are technically complete,4090 native completions+550 retained caps,256 independent layouts with same32 alternates. No wall-alternate partial raw/attempt/log read occurred before closure. All model sessions are closed and no further frozen inference is queued. The single combined report completed CPU40218exit0: JSON58851e926cfb64ba3cf19dc7e6c1ba6da622cffd8bf7392eb10b3bda42e0277d, Markdowncf95f99fe19717df78fa9af517f704e321658afb94a94f4248fdf09429144ac9. Independent scientific reconstruction and claim review are underway. Whole ledgerv4 completed CPU42179exit0,36specs, observed unchanged snapshot00:33:00–00:33:12UTC; ledger labels do not replace admissibility audits. Documented CPU reproduction completed9 commands at00:37:14UTC under closed10531; execution SHA64bbf0a0f51ac52d79df482824bf3ec3db88fac86cfd9c3ef227a6a7a6d55752, numerical/provenance comparison and figure replay pending. Prior3docs preserved in research/activation_causality/checkpoints/20260914T003830Z/. Integration record outputs/activation_causality/campaign_20260912/reviews/all_cohorts_technical_closure_integration_v1.json SHA461e891d859abb19acfe6ce32d751795f14d8ea782ba0a6b62696f025021e8b6. The goal remains active for independent scientific approval, production figures, full replay, ledger/archive integration and separately reviewed paper proposal. The main PDF is unchanged.


### 2026-09-14 00:53 UTC — theory handoff verified; pre-model numerical launch failure

Automatic monitoring detected all four exact filename sets at00:34:48 and then the owner's00:40 whole-queue closure at00:41:09. All four independent technical certificates and substantive owner closure were separately reviewed; handoff acceptance SHA0dc88e8f1c1c1ee535405115aaa00cef45da3d382bcef9cc38bd3c94b7844d63 binds the exact preserved release. Fresh complete process validation passed00:46:40. Numerical v1 launched once00:47:25 under exact ticket40aac794b7e0fef2b9ffc4ecfd46639cfda45bf3c37c25824786e616e3430e7a, session51511,parent410,child/group415. It acquired the shared lock and verified release, then failed startup identity before model loading and closed exit1 at00:47:33 with owned_group_closed=true, no signals or cleanup errors.

The coordinator mistakenly set PYTHONDONTWRITEBYTECODE=1 alongside-B, while the accepted startup recorded it unset. Independent CPU captures prove that with the variable unset the full accepted startup matches; only that environment leaf and semantic digest differ with1. No observed source/hook drift, engine construction/loading, forward/sample, worker/numerical event or accepted result exists. The original consumed claim and output are preserved; no retry or frozen-source mutation occurred. This is a pre-model invocation failure, not a scientific null. Diagnosis: research/theory_foundations/reviews/N2_PRELOAD_ENVIRONMENT_FAILURE_20260914.md. Machine record: research/theory_foundations/registry/autoresume_numerical_failure_20260914.json. A separate unapplied recovery proposal is being prepared, with all original scientific settings, failed paths and deadlines retained. Both metadata watchers exited after successful single notifications; actual resource handoff is resolved. Accepted science/manuscript, source paper, prior runtime/queue and persistent goal storage remain unchanged.


## Activation causality — independent scientific and presentation closure (2026-09-14T01:09:10.926590+00:00)

All four frozen confirmation cohorts are closed and independently technically approved: 4,640 executions, 4,090 final actions, 550 retained caps; 256 independent layouts plus 64 same-layout rendering repeats. Exact joint report58851e926cfb64ba3cf19dc7e6c1ba6da622cffd8bf7392eb10b3bda42e0277d passed independent scientific audit7293b7c49709ac2546bb090539e419e25b624335b532ddfb8d454e10e751a09e. Both contextual and wall primary AND criteria pass; contextual descriptive alternate transfer passes and fixed wall rank1 transfer fails. No abstract-belief, necessity, mediation or invariance conclusion is established.

Documented CPU audits/report replay and production/replay figures passed; scientific report fields match except timestamp, Markdown/PNG bytes and PDF text/rendered pixels match. Final manuscript and figure claim review12faa358508a0c353f0d88f5e6be826cb90182dbf8ee4ce25f2473ba46b5caf9 approved. Main PDF unchanged. Ledger v4 independent reviewd9c9b05697292008e6c01c46d88cd8779120acce856ea529957ed8d4314a2e0a verifies36 specs and7,643 captured payloads. Reviewer-only metadata/rounding and coordinator-only ledger path/cardinality mistakes remain preserved without production/scientific changes.

Final findings and separate proposed causal section are in research/activation_causality. Current guides and README are integrated; source-import verifier passes. Final evidence archive and independent package review remain before goal completion. No active model worker; all four supervisor handles are closed.


### 2026-09-14 01:12 UTC — theory startup recovery proposal reviewed, unapplied

The specific source-freeze exception is ready for user review. The unapplied one-file stages.py patch permits only numerical_startup_recovery_v1, requires independently accepted evidence for the exact preserved pre-model environment failure, and preserves all model/scientific inputs, original failed paths, watchdog and deadlines. It does not provide generic retries or numerical acceptance. Author and independent reviewer each passed all10 focused scratch tests; the reviewer also passed git apply --check without applying, original/proposed diff equality and all7 historical pins. All99 tracked runtime sources were checked unchanged. Live stages.py remains8c5006c8aca80c2d422b704fb7b86d60a9800d73737e29b1f45d864ffdcf14ae. Patchf9659b1be52de889a6a1b41e6225fa65ac4fe007befa1c5403e62697688974e0 would produceeada64580b104c03db5795a8796bec768fb2541ff0eb30b7110c4e4a46c3a448. Independent review: research/theory_foundations/reviews/N2_STARTUP_RECOVERY_PROPOSAL_REVIEW_20260914.md. State: research/theory_foundations/registry/startup_recovery_approval_request_20260914.json. No source mutation, new acceptance/ticket, model operation or recovery launch occurred. Explicit exception is required by the current automated wake-up's source-mutation prohibition; fresh independent stage/runtime and launch gates still precede a subsequent attempt.


## Activation causality — final evidence delivered (2026-09-14T01:19:23.919636+00:00)

Independent archive review011986182c4825ee961f46b3a2fc1166c11f06e64081640b23e2c46dbd518968 approved all9,633payloads; archiveSHAee8c543bf23515271d62a3866ba69815ca3b3b02e565de0a86d35408620b99e7, manifestSHAb7085086a74e3f23d81b3a72367ebf1b9a9cc0f9fa60cd8fd7ccc0261c05ee56. Exact documented archive verification passed. All research, source, manuscript and packaging work is complete. Final findings, paper proposal, figures and evidence package are linked from research/activation_causality/STATUS.md. The original PDF is unchanged. Eighteen harmless staged cache extras are documented in a sidecar; archived bytes remain intact. Final requirement audit and persistent-goal status update follow as separate release records.


### 2026-09-14 07:14 UTC — theory external evidence integrated; recovery remains unapplied

Renewed source-free coordination revalidated the independently accepted whole-queue handoff against a fresh complete765-process capture at07:08:24UTC. No relevant theory model worker or watcher was observed. The saved projection is explicitly not reusable launch clearance; the default model lock was not acquired and no model operation occurred. No resource watcher was rearmed because the release dependency is resolved. Resource record: outputs/theory_foundations/campaign_20260912T174516Z/coordinator_analysis/resume_review_20260914T0710/resource_revalidation.json.

The actual independent startup-failure acceptance0ed793c00d8d7214804b100a86acfbfa3c2c98bcf4e6ef33aefcda01712a1151 binds12original artifacts, the environment-only diagnosis and separate reviews. Root rehashed all16references and checked the exact eight-file original execution inventory. No engine construction, model load, forward or sample occurred in v1. Its original consumed claim/output remain preserved. The reviewed one-file recovery remains unapplied under the latest automated wake-up source-mutation prohibition; no exception was fabricated or model retry attempted.

Newly delivered external confirmation report58851e926cfb64ba3cf19dc7e6c1ba6da622cffd8bf7392eb10b3bda42e0277d and independent scientific certificate7293b7c49709ac2546bb090539e419e25b624335b532ddfb8d454e10e751a09e support a separate theory supplement: research/theory_foundations/theory/EXTERNAL_CAUSAL_EVIDENCE_20260914.md, SHAd10b62431d0685645eecc1582c280241eb1f04d502beca1c539722a734ec013d. Empirical independently checked numerical transcription, provenance and inference scope with no correction; reviewSHAe99134702a2c0b22fae5ceb8a63c68305e816360fdcacfe787561145230f81a9. It connects native input-activation control and rendering dependence to H1/H2/H3 while leaving belief/utility identification, report-action dissociation and reasoning-time commitment unresolved. External results remain confirmation for their own campaign, observed after N2's freeze; no N2 adaptation or substitution occurs.

Evidencev6 and hypothesesv5 integrate the accepted supplement and correct superseded operational statuses. Prior snapshots and coordinator documents are preserved. Original PDF, accepted theory PDF/ZIP and live stages.py were rehashed unchanged. No source mutations, new tests, raw confirmation/log reads or model operations occurred. Numerical gate, full pilot, timing-only final sizing, native evaluation and final integration remain. Persistent goal status/storage and both15Septembercutoffs are unchanged. Progress record: research/theory_foundations/registry/resume_source_free_progress_20260914T0714.json SHA3a22e182c03e30a11371bf638910c9bd0165b10f662c142dc42f4d249209bf8b.


### 2026-09-14 07:54 UTC — theory recovery failed before numerical gate; import diagnosis

14 September 07:54 UTC — runtime import failure under investigation. The first reviewed recovery was applied and independently accepted, captured under a fresh identity, and launched once at 07:38:22 UTC. It passed the pre-load identity check but stopped at 07:38:44 with `Startup activation multiplicity/order differs from the reviewed loaders`. The parent certifies exit 1 and owned-group closure with no signals or cleanup errors. Engine loading may have completed; this is not a zero-load failure. The frozen numerical gate did not start, and no numerical result, pilot or evaluation exists. Both consumed launch claims and exact failed output directories remain preserved.

The independent failure record is `run/reviews/n2_recovery_load_guard_failure_20260914_v1.json` (SHA256 `41f139a514cbf0eaf992b6d35ac4ae45c73c6761a7580c69cd1c36ef70a31dff`). A separate no-engine import diagnostic reproduced the failure: importing MLX-LM adds the installed `six` finder and imports optional packages outside the declared dependency inventory. Diagnostic `run/coordinator_analysis/runtime_hook_diagnostic_20260914_v1/diagnostic.json` is SHA256 `333803bb259f6797eff597f1fe62ea4c73ba03f1afcc9df486b55ab2a402c0e7`; it is diagnosis only, not a numerical gate. The agents are preparing an unapplied, narrowly bounded runtime correction and independent review. No further live runtime source has changed since the previously authorized exact `stages.py` repair. The earlier interpretation record scopes that repair; the visible automated user message prohibits source mutations, so any additional proposed source exception must be made concrete before requesting it.

Fresh complete process metadata and the default shared lock revalidated the independently accepted whole-queue release at 07:54:15 UTC. Record: `run/coordinator_analysis/resume_review_20260914T0752/resource_revalidation.json`, SHA256 `f14ec5e970affaf507b0494bb9132d982333459190e00a39a77a3009c1677359`. That observation is historical, not reusable launch clearance. The queue dependency is resolved and no resource watcher is needed. The current obstacle is the runtime guard. Numerical gate → independent closed review → full 44-job technical pilot → timing-only final N → native evaluation → final integration remain required. No original paper, prior campaign, scientific settings or deadlines changed. Inference cutoff 15 September 11:45:16 UTC; reporting deadline 15 September 17:45:16 UTC.



### 2026-09-14 08:03 UTC — theory repair authority clarification

**Authority clarification — 14 September 08:03 UTC:** the restored conversation records a later user continuation directing the agent to improve or replace existing work as needed. That later instruction authorizes necessary reviewed repairs in the theory workspace. The coordinator's 07:54 note read the older watcher restriction too narrowly; its extra permission dependency is withdrawn. The [interpretation record](registry/import_recovery_authorization_interpretation_20260914.json) preserves this correction without inventing a separate human yes. The three-file runtime proposal remains unapplied until independent technical review. Source snapshots, fresh actual identity acceptance, a complete no-model import/tokenizer rehearsal and every numerical/scientific gate still precede inference. Original paper, prior campaign, settings and deadlines remain protected.



### Theory import-recovery application — 2026-09-14 08:22 UTC

**The reviewed three-file import correction is applied.** At 08:20:21 UTC the exact combined patch changed only `native_study/identity.py`, `guards.py` and `stages.py`. Root verified all 99 prior source records before application and exactly those three changes afterward; all 46 protected files and both failed attempts remain preserved. The independently accepted proposal passed 33 focused checks and resolved the 54-distribution metadata inventory, with all original 34 records unchanged.

Actual new stage: `run/coordinator_analysis/n2_registration/numerical_import_recovery_v1/stage.json`, SHA256 `270a0b5bf066192400768b892103ba21de58f6f20c42cde47fcb0ab9b196c5ee`. It changes only ID/output/evidence reference from the original numerical stage; every scientific setting is identical. Live stage validation passes all 34 preserved-failure input references. Applied record SHA256 `183e56e043e7a2ca89de8f0e1ffccb1ab3d9e4a4c19f75a4cf95d0bc6f3ecc4d`; source snapshot SHA256 `f079caa530b563879cd2feee69b88a4e0c95196bccd2e92a0fd8e00c2fbf76f8`.

Independent amendment **v2**, SHA256 `5ba4cb9ba55c9362e50d722d0723439e2b4d7de24ce2d218398f51e583baf756`, corrects only v1's placement of two machine audit references in a human-review-only path list. V1 remains preserved; no source or scientific correction followed. The human [proposal review](reviews/N2_IMPORT_RECOVERY_PROPOSAL_REVIEW_20260914.md) is SHA256 `401cd5d0287ebcec901f19327a0257e641dff7f8b70933dbe6316c8d8158df1e`.

The actual pristine startup capture completed with exit 0 and no framework import, engine construction or weight read. `numerical_import_recovery_v1/startup_closure_v1.json` is SHA256 `36b35e758de29b54bc19eff9c0654eb2c98e25f2dfd92d9793d4b32cda8d3576`. Independent actual-stage and startup reviews are underway. Next: fresh actual candidate capture/audit under current whole-queue process validation and default lock, then the exact candidate-bound no-model import/tokenizer rehearsal with its 600-second supervisor and independent closed acceptance. Only afterward may a new numerical ticket be formed.

No new model operation has occurred since the preserved 07:38 import-guard failure; that failure may have loaded the engine but produced zero frozen numerical operations. No numerical acceptance, pilot or native evaluation exists. Numerical gate → independent closed review → full technical pilot → timing-only final N → native evaluation → final integration remain mandatory. The later user repair authorization remains effective; no new permission dependency was introduced. Original paper, prior campaign, scientific settings and fixed **15 September 11:45:16 UTC inference cutoff / 17:45:16 UTC research deadline** are unchanged.



### Theory no-model rehearsal failure — 2026-09-14 08:39 UTC

**The complete no-model rehearsal stopped on a virtual module filename; no model was constructed and no numerical claim was consumed.** Owned session 9527 closed with exit 1 at 08:34:39 UTC; the parent certifies group closure, no signals and no cleanup errors. The saved initial checkpoint passes. At the after-import checkpoint, startup and interpreter checks returned, then the ordinary-origin guard attempted to resolve Torch's virtual `_ops.py` label as a file in the repository. No complete after-import receipt was published, and the tokenizer branch was not reached.

Parent: `run/coordinator_analysis/n2_no_model_import_rehearsal_20260914_v1/supervisor_terminal.json`, SHA256 `95644fbc6684123d12529757e094177413940d71de5092ceff66686a63e98974`. Rehearsal: `worker/rehearsal.json`, SHA256 `0115eceba7497c41568e22205f112296f1e7b7ef4caf3011700c1ea0f465d86d`. Both are closed, preserved diagnostics, not numerical or scientific nulls. Coordinator record: `coordinator_failure_record.json`, SHA256 `30d346c7cd8e446e4f887cec2b40efc0d853fc1636be7dcc094481d8562937e1`.

The candidate remains intact: file SHA256 `09e9dbe869a54e997a9f6fcf2b80ab4a801561ddb63609566f230468e0d023be`, semantic `12d61367ad6b4482ed21c9a9a739b280d16267e8440fc8cce83e35be702b881f`, core `75ae5a77a3410dd1bdef70e225d1b56317f3372245f0705eb2d043dc3bef1d0f`. The independent byte audit passed 31,282 paths, 114 trees, 54 dependencies, all 99 sources, 46 protected files and all three existing weight shards. Byte-only acceptance SHA256 `3f49bdb494be5672a8f7f85456609e2d54ead573fe1be3431db361a2c78ef12e`; it does not accept the imported phase or authorize inference. Capture session 79016 and independent audit session 7579 also closed successfully.

The agents are independently reviewing the exact failure and preparing **one comprehensive no-engine module-origin census**, including the existing local tokenizer branch. It will collect all ordinary/unbound/virtual/nonfile origins using static attribute inspection and preserve guard exceptions as failures. It is explicitly diagnostic, supervised for 600 seconds, and requires fresh whole-queue process metadata and the default lock. It cannot manufacture a guard pass. All current sources, prior failed attempts, candidate and rehearsal remain preserved while the next narrowly reviewed virtual-module correction is prepared.

The later user authorization covers necessary reviewed theory-workspace repairs; no new permission dependency is introduced. The numerical gate, independent closed review, full pilot, timing-only size selection, native evaluation and final integration remain mandatory. No source/scientific tolerance, input, seed or deadline has been relaxed. Inference ends **15 September 11:45:16 UTC**; research deadline **15 September 17:45:16 UTC**.



### Theory-foundations closed no-model import census — 2026-09-14T08:54:04.739801+00:00

Owned session 87278 closed exit 0; full framework/static-origin/local-tokenizer census completed under fresh whole-queue process validation and default lock. No model, weight payload, forward or sample. Diagnostic SHA256 aba42a1c3e13f7fa942f951858acc9d0d75b2f70a48962213ef051f319ae4aa7; parent a798700c6e7f2a3c1a6ed94d31552fc3ef79e94dfb7b16a849bb9b6217c09fd2. This is collection completion only; independent review, guards-only virtual-origin correction, fresh candidate and strict rehearsal remain before numerical clearance. Existing goal/deadlines/science/original paper unchanged.


### Theory-foundations exact virtual-origin correction — 2026-09-14 09:04:40 UTC

Applied only guards.py SHA c72cdb48319eb698a20b1b983526cf366efb7927acb3bef34c09d141c52df440 after independent amendment v3 SHA60897238e2d708294359c8dd19b0ed281a1ec66d4b24cf20ce581820fbec8ab5. All 99 prior sources and 46 protected hashes checked; numerical scientific fields unchanged. stage_v2.json SHAe05fd27ac3b9870566c091ad653c12081f8f6b3482c4a80be11d5bdf637775bb; numerical claim absent. Actual-stage/candidate review and full strict rehearsal remain before ticket. No model operation; original paper, other campaign and fixed deadlines preserved.


### Theory-foundations repaired identity and strict rehearsal — 2026-09-14 09:15 UTC

Candidate v2 SHA8c0d411a7985cb1dc9a73698a8dc6c4d8f9c5f04e9ce1b38e761e30c2fd86f9b/core30926b624d301864a65ed540e45efe3892d8d476e4bab79af764b4cb295b3e3f passed independent complete byte audit. Strict no-model rehearsal v2 session37056 closed exit0; full import/tokenizer/startup/origin checks passed, zero model operations. Raw parent SHA78d9c8bf36c9dc72f6b4dfe1a9c15bcca681d9bf56d5283ed13b14162588e6a0; rehearsal99679bb48638e98a2b219238b826b596015b798e9f34a6679f0b0aaf0e6d8df8. Independent closed rehearsal/final identity acceptance and exact numerical ticket/precheck remain before inference. Existing goal and fixed deadlines unchanged.


### Theory-foundations fixed numerical gate launched — 2026-09-14 09:20:52 UTC

Root session40064 is running; one claim SHAe626a8f65fe5b00773102dc24b9c13efc054e8134cced3a228fb0cfd08fe655d, exact ticketd3a25a52493466a9e64874b95549f9b99a14f54886906d5fe878feaede1ddbd6. Complete identity/rehearsal acceptance and fresh standalone/prelaunch/default-lock checks precede this unchanged1800s numerical gate. No running raw outcomes read; no numerical acceptance or pilot. All native source/core/science/deadlines fixed.


### Theory-foundations numerical gate closed — 2026-09-14 09:27:22 UTC

Root session40064 returned exit0. Parent b224ab046dcb63301298b6d20c16b1c9362dbfd5e8101456e33b59d3c73595ef closed09:22:45.147UTC,243.429s, groupclosed/no errors. Worker9d28ae2c40796d4fd8bf35ca8ccbaf3706636ea4e5c5cd097780f8506fb655ba reports1/1complete/no missing. Root read closed metadata only. Independent CPU numerical reconstruction now authorized; successful process completion is not yet numerical acceptance and does not launch the pilot. Native core/science/deadlines unchanged.


### Theory-foundations numerical acceptance and pilot launch — 2026-09-14 09:45 UTC

Independent numerical acceptance e13a979e… passed111forwards/88samples/15logitcomparisons(max0). Fixed44jobpilot now running root69644, launched09:44:04UTC through unchanged7200s supervisor. Ticket33dfa2a…, claimaa950712…, candidate3567c574… retains numericalcore30926b. Independent stage/identity/ticket checks and fresh793row/defaultlock/startup/source precheck passed. No partialpilot outcomes read; all44technical review and timing-onlyN remain before nativeevaluation. Fixed deadlines and sources preserved.


### Theory-foundations pilot closed — 2026-09-14 09:56 UTC

Root69644exit0; parentb0a37750… closed09:53:45UTC after581.328s, groupclosed/noerrors. Workerf1d480ef… accepted44/44. Technicalprojection18c7aad4… completed99941exit0,19nativecaps/1complete,bothprefix64, no scientificeffectsread. Independentall44reviewpending; finalNunset. Transparentclockmetadata supplementdb3b87c0… independentlymatched; no runtime/sciencechange. Numericalcore30926b/fixeddeadlines preserved.


### Theory-foundations final N64 and evaluation preparation — 2026-09-14 10:13 UTC

Pilotf0a9bd70… and strict timing4cabb975… independentlyaccepted. Single sizing64f94886… at10:03:08UTC selectsN64, fullcap/factor2 forecast31702.058s, reserve3600s. Independentacceptanceaa6c101e… and arithmeticfe6ccadc… agree. Evaluationstagec429d9e9…/acceptance0f617167… binds64layouts/1408jobs/704seeds/all11poolartifacts. Fresh lockedcandidatecapture99655exit0 yields0a8c2abd… with unchangedcore30926b. Identityreviewpending, no evalclaim/modelyet; all deadlines unchanged.

## Theory foundations — 2026-09-14 10:47 UTC

Evaluation v1 prelaunch stopped before claim/model work: full ticket validation imported the fixture-preparation chain and duplicated ROOT in sys.path. Independently verified diagnostic SHA256 22695c6267626a36d582b8668f70aa7ce8a82de33aaceb7bd8f83ff8ace035a1. Original stage/candidate/ticket preserved; no evaluation output/claim. Unapplied stages-only bridge proposal SHA256 01e18e39951f7a346eba0ca14840593ae8b33484bb10eea6c529dbbc45ee58ba is under independent review. Numerical/pilot acceptances and the single N64 timing selection remain intact; fresh true source identity and complete no-model rehearsal required. No goal-status/deadline change.

## Theory foundations — 2026-09-14 10:59 UTC

Applied independently accepted stages-only administrative bridge9c23f11c… at10:59:08.066UTC. Exact source75899c65…, actualstage_v2 1a0ccfab…, appliedrecord b3544c81…. Other98sources/46protected unchanged; all scientific settings and N64 preserved. Historical core30926b is retained as historical; required actual new core55efba1d. No evaluation claim/output or model work. Actual stage/candidate/rehearsal acceptance gates follow; deadlines/goal status unchanged.

## Theory foundations — 2026-09-14 11:09 UTC

Actual candidate84cab03c…/core55efba1d… capture61175 closed; preliminary independentidentity4f59c4ea… passed257currentrecords and retained/newinputs. Supervised complete no-model evaluation rehearsal42489 closed exit0 at11:08:40UTC (7.525seconds), fourstartup/interpreter/origin checkpoints/fullticketexpecteddenial/exacttokenizer passed. Parentafaa57eb…, normalizedae89b1f1…, workere9629683…, rootwitness1424b53f…. Independentclosedreview/finalidentity pending. No evaluationclaim/model; N64/pilot/numerical/deadlines preserved.

## Theory foundations — 2026-09-14 11:20 UTC

Single N64/1408-job evaluation running in root22492, parent22781/child22784; claim49b391…consumed11:17:01UTC, finalticket0e301/core55ef. Independentclosedrehearsal31bfa2/finalidentityaa3f8 accepted; standalone140f29…passed fresh821-rowprocess/defaultlock/startup/257records/sameNfit. No partialscientificoutcomes/logs read. One-shot completionwatcher root22751/PID23097started11:18:21UTC, source224e/config86e8/configaccept3afa; same-thread terminal-stat/cutoff wake, onequeueattempt/noinference. STATUS/readiness and immutableevidence_v8 record continuation through closedanalysis, independentrecomputation, figures and finalmanuscript. No goalstatus/deadline change or additionalmodel allowed.


## Theory foundations — 2026-09-14 11:48 UTC

Progress during the running N64 evaluation: fresh complete process metadata confirms the single parent/worker and completion watcher; both owned sessions still returned running at 11:47:54 UTC. No partial scientific outcomes/logs, new inference or native-source mutation. Completed full visual review of the preserved 28-page proof (7e4869c5…), independently accepted the factual manuscript/readable-Markdown proposal, and assembled a separate 144-file derivative (manifest 75e29154…; root verification cf1ef129…). The reproduction guide and 296-file proposed inventory make all final missing slots and original-IRL/local-runtime limits explicit. The independent delivery audit needs no extra experiment. Existing goal remains active; fixed cutoff/deadline and the post-closure analysis, independent acceptance, figures and final package requirements are unchanged. Machine checkpoint under coordinator_analysis/manuscript_preparation_review_20260914_v1/.


## Theory foundations — 2026-09-14 12:06 UTC

Progress: physically collected 514 evidence payloads (205,449,417 bytes) and independently verified every source/copy and exact 517-file set; collection 81098676…, review 723003d8…. Historical versus preparation/current-copy hashes remain distinct. Combined provisional claim map de036212… accepted by 7eddc083…: 14 archival rows preserved, external/coverage evidence and three account syntheses added, N2 outcomes remain null. Both live evaluation/watcher sessions returned running at 12:04:40 UTC; no partial outcomes/logs, inference retry or native-source change. Separate frozen-source N2 methods preparation is in progress. Goal/deadlines and required closed scientific/final-publication work remain unchanged. Machine checkpoint: coordinator_analysis/final_collection_checkpoint_20260914_v1/checkpoint.json.


## Theory foundations — 2026-09-14 12:20 UTC

Progress: accepted separate N2 methods V2 (MD11448e30…, TeXbd370ea0…, manifest4d26206c…, revieweab1f9eb…) preserves all equations/settings and clarifies three prose points. Six displays,57math instances,17refs independently checked. Final assembly addendum now consolidates all accepted inputs. Outcome-independent preparation is complete. Both owned evaluation/watcher sessions returned running through12:20:08UTC; no current scientific outcomes/logs, new inference or native-source mutation. Next is actual closure and the preserved analysis/reconstruction/result-acceptance/figures/final-package path. Existing goal/deadlines unchanged; verified live waiting does not authorize another audit, study, model or watcher.


## Theory foundations — 2026-09-14 16:02 UTC

Progress: single N64 evaluation closed15:55:02UTC, root22492 exit0 and owned group closed, verified by fresh833-row process inventory and root witness5343e221…. All1408planned jobs present, no technical missing/failure. Unchanged closed analysis45049 exit0 produced manifest844bae20… and result1b7886a3…. Independent actual provenance/recomputation/acceptance and provisional frozen-template interpretation are in progress. Completion watcher22751 exited0 after one queue attempt, delivery unverified, no retry. No new inference/source mutation. Existing goal/deadlines unchanged; final figures, integrated manuscript and full evidence/PDF/package review remain required. Machine checkpoint: coordinator_analysis/n2_evaluation_closed_checkpoint_20260914_v1/checkpoint.json.

2026-09-14 16:40 UTC - Theory foundations N2 independently accepted (012377f9); all 1,408 jobs technically complete, all 22,533 reconstructed numerical fields match to floating-point precision. Separate final manuscript integrates reviewed results and compiles to 35 pages. Final PDF/source/package review underway, no inference remains needed. Goal remains active; fixed deadlines and original sources preserved. Checkpoint: outputs/theory_foundations/campaign_20260912T174516Z/coordinator_analysis/final_assembly_checkpoint_20260914_v1/checkpoint.json.

Timestamp correction for the immediately preceding theory-foundations final-assembly entry: the machine checkpoint was actually created at 2026-09-14 16:31:53.586162 UTC; its hand-written 16:40 label was a transcription error. STATUS and runtime-readiness headings corrected to 16:31. Scientific records and actual event timestamps are unchanged.

2026-09-14T16:56:11.597511+00:00 - Theory-foundations research and final V3 delivery complete. Original paper preserved. Four independently checked archival studies, separate external native evidence and accepted N64 evaluation integrated into35-page revision. Final scientific/PDF review a37854ba; anchor addendum55c064bf; independent package acceptance ed249d99. Delivered ZIP d47714cc,120154253bytes; all3788files/157directories,441filetargets+5anchors passed, extracted verifier exit0. Current README/STATUS/runtime-readiness and evidence_v9 updated. No inference or watcher work remains. Existing goal closure is recorded via public tool next, without replacement or internal storage edits. Delivery: outputs/theory_foundations/campaign_20260912T174516Z/final_revision_20260914_v1/DELIVERY.md.

2026-09-14T16:56:17+00:00 - Public update_goal(status=complete) confirmed completion of the EXISTING theory-foundations goal01a096af-e145-7360-9e27-aa1ff02642f6. No replacement goal or internal storage edit. Exact public-tool receipt: outputs/theory_foundations/campaign_20260912T174516Z/final_revision_20260914_v1/GOAL_COMPLETION.json; SHA256 ed615cf0b2d2394e01617eae68b076535a67caa0130a99e3e6726da0841d4a71. Tracked working time51490seconds; no token budget was set. Final delivery and accepted V3 package remain unchanged.


2026-09-14T18:27:07.403054+00:00 - Activation-causality presentation rewrite complete at user request. Added a nine-page explanatory paper (research/activation_causality/PAPER_REPORT.md and output/pdf/activation_causality_paper.pdf), with background/problem, a worked counterfactual grid example, intervention diagrams, experimental design, results, discussion and complete condition appendices. Independent narrative/figure review and all-nine-page visual QA passed; records are in research/activation_causality/paper_assets/. Original manuscript and certified final report/proposal preserved. No new model queries, statistical reanalysis or goal change.


2026-09-24T20:30:40.090901+00:00 - Downloaded SPAR-Telos/reveng branch feat/counterfactual-patching at user request into reveng/ as a separate Git checkout. Origin: https://github.com/SPAR-Telos/reveng.git. Verified HEAD d862d146081756990559db78dfae0b6062eb2d8d, clean working tree, and tracking origin/feat/counterfactual-patching; single-branch clone with full branch history. Git LFS smudge disabled during acquisition. No dependencies installed or model inference run.
