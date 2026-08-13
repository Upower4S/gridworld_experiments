# Research Directions: The Action-Belief Gap Along the Chain of Thought

Date: 2026-07-23. Synthesized from a full RUN_HISTORY.md audit, the on-disk artifact inventory,
and a literature search (July 2026). This is a planning document, not a run log; runs still go to
RUN_HISTORY.md.

---

## 1. Where the project actually stands

### 1.1 The label regime (read this first)

Four scoring regimes exist for the core object `q_UP..q_RIGHT`. Trust order was reversed on
2026-07-21 by the label-integrity audit:

| regime | rows/traj | status |
|---|---|---|
| Together 10-traj | 9,942 / 10 | superseded (coverage) |
| Together 32-traj | 33,148 / 32 | superseded (partial, billing-capped) |
| MLX-full stored (stage-03) | 98,311 / 95 | **quarantined** — 93.5% q-mismatch on re-extraction; argmax agreement only 0.71 |
| `extracted_beliefs.parquet` | 98,310 / 95 | **canonical** — adjudicated faithful (end-of-CoT agreement with agent_action 0.991 vs 0.908 stored) |

Consequence: **every behavioral headline currently in the draft report (commitment 0.35→0.99,
lock-in medians, action asymmetries, the manifold) rests on superseded or quarantined labels.**
Only the 2026-07-22 fold-0 CoT-window probe and the supervised projection use faithful labels.
`outputs/conceptual_belief_space/cot_window_layer_transition_probe/activations/extracted_beliefs.parquet`
already carries `commitment_to_final`, `entropy_commitment`, `margin_commitment`, `clr_*`,
`strict_before_*` for all 98,310 rows — the recompute is a straight swap that has not been done.

### 1.2 Established (faithful-label) results

- Commitment (entropy_commitment) is strongly decodable from CoT-window activations:
  held-out Spearman 0.762 (`mean_layers_mlp`), R² 0.49, high/low AUC 0.957 — vs the fair
  positional bar 0.346 and even the oracle progress baselines 0.63–0.64. Survives within
  fixed progress bins (0.52–0.71 while the positional baseline collapses to ≤0.33 and goes
  negative). Fold 0 only.
- The `prompt.txt` layer-transition hypothesis is NOT supported: mean-over-layers beats the
  layer-sequence transformer (Δ Spearman −0.016, CI excludes 0); deltas add nothing.
- No four-arm activation manifold: the behavioral star is simplex geometry of the labels.
  Supervised LDA on strict pre-mention rows gives one overlapping cloud with displaced
  centroids (balanced acc 0.42–0.43 vs chance 0.25); PCA nulls are a variance-ordering
  artifact (top PCs encode log position, |r| up to 0.57).
- ~~Preliminary layer dissociation (fold 0)~~ **RESOLVED 2026-07-23 (see D2 status):** the
  depth dissociation was an instrument artifact. Matched per-layer probes put both targets
  in the same mid-late layers (peaks L13–18, 5 folds); the real dissociation is
  within-layer orthogonality of the commitment direction and the action subspace
  (removal tests flat in 5/5 folds), plus a small consistent depth interaction
  (action more depth-loaded, +0.046).

### 1.3 Retracted / not-supported (do not cite)

- The 0.7008 "last-token attention" probe → scope-corrected to a causal prefix classifier;
  attention-query effect is real but tiny (+0.0016, p=0.0099). Replacement (scripts 74/75)
  half-done: selectors extracted for layers 11/15/19/23, probe metrics never produced.
- Robust top-action balanced-acc > 0.8 never met (best fragile ensemble 0.80003).
- Stage-03 defect not yet localized (KV-cache path suspected; BPE boundary drift and model
  revision are the candidates).

---

## 2. Literature positioning (searched 2026-07)

Closest neighbors — all recent, none does what this repo can do:

- **Belief state geometry** (Shai et al., arXiv 2405.15943, NeurIPS 2024): belief simplices are
  linearly embedded in the residual stream of small transformers on HMM data. We extend to
  *agentic action beliefs* of a 20B production model along real CoTs; our CLR-regression
  projection is their methodology, and our "no four-arm manifold" result is an informative
  negative against naive geometric transfer.
- **When Does a Language Model Commit?** (arXiv 2605.06723): finite-answer log-odds
  stabilization on Qwen3-4B, *binary* delayed-verdict QA; stabilization 17–31 tokens before
  the answer is parseable. Behavioral only, no activation probes, no agent, no positional-
  confound framework. Our forced-answer q is the multi-class agentic generalization.
- **Beyond the Commitment Boundary** (arXiv 2606.13603): sharp commitment boundary; up to
  55% of CoT after it is epiphenomenal (early-exit safe). QA tasks; attention probes decode
  answer-formation stage. We can measure the agentic analogue (post-commitment tail per
  decision step) and go further: layer localization + belief distribution dynamics.
- **Exploring belief states in LLM chains of thought** (LessWrong, Gemma-3 27B, MMLU binary):
  probe accuracy near chance until the last ~40 of ~260 tokens; strongest mid-layers. Sharp
  contrast with our setting, where commitment is decodable throughout and rises 0.34→0.62
  *before any action mention* — task structure (continuous evidence accumulation in a spatial
  task vs discrete fact lookup) plausibly determines belief-dynamics shape. Good contrast
  point for the paper.
- **Broken Links Between Observations, Beliefs, and Actions** (arXiv 2605.00226): owns the
  term "belief-action gap" in strategic games (beliefs decodable but causally weak on
  actions). Cite and differentiate: our gap is *temporal* (belief trajectory vs eventual
  action along the CoT), theirs is *causal* at a single decision.
- **Temporal Predictors of Outcome in Reasoning LMs** (arXiv 2511.14773) and **Catching
  rationalization in the act** (arXiv 2603.17199): outcome/motivated-reasoning probes along
  the CoT; both acknowledge but do not systematically solve the position/length confound —
  our fair-vs-oracle positional-baseline framework is a methodological contribution.
- **Two-stage decide-then-bind** (arXiv 2601.03914) and confidence-probing work (e.g.
  arXiv 2603.17839, 2602.08159): confidence/correctness is decodable *earlier in depth* than
  answer content ("cached retrieval"). Directly anchors our layer-3-commitment vs
  layer-18-action dissociation.
- **Thought Anchors** (arXiv 2506.19143) / Thought Branches (2510.27484): sentence-level
  counterfactual importance via resampling — the method template for our belief-update
  attribution direction.
- Upstream paper of this repo: arXiv 2602.08964 (behavioral + representational
  goal-directedness; coarse spatial map; reasoning shifts from spatial cues to action
  selection). The present work is its mechanistic sequel.

**Unique assets nobody else has combined:** (1) full 4-class belief distribution per CoT
prefix (98,310 prefixes, 95 trajectories, 1,276 decisions) with adjudicated-faithful labels;
(2) aligned all-24-layer × 12-token residual activations per prefix; (3) strict
pre-verbalization masks; (4) fair/oracle positional-control framework; (5) trajectory-grouped
nested splits.

---

## 3. Directions, prioritized

> **Status 2026-07-24: all directions executed.** D0 (03c/08; triangulation 0.971 vs
> 0.702). D1 (09; three clocks, 50.6% post-commitment tail). D2a–e (79/80 + 77 folds 1–4:
> mean_layers_mlp 0.7715 ± 0.0094; depth dissociation refuted; within-layer orthogonality;
> only layer 10's occlusion reliance replicates across folds). D3 (81; fold 0 + folds
> 0/1/4 pooled: decoded clocks noise-limited). D4 (10; beliefs track CoT text 0.60–0.67 vs
> prompt 0.13–0.17). D5 (11; top-5% transitions carry 37% of belief change, concentration
> predicts early stabilization ρ=−0.51). D6 (82; readable-not-writable null at α≤2.0,
> n=28 paired; all-CoT injection left open). Details: RUN_HISTORY.md 2026-07-23/24.

### D0. Re-base the behavioral phenomenology on faithful labels (mandatory hygiene)
All draft-report behavior numbers must be recomputed from `extracted_beliefs.parquet`
(straight swap; metrics columns already present). Re-derive: commitment/entropy curves vs
progress, strict pre-mention curves (0.34→0.62 claim), lock-in and switch statistics
(scripts/04 logic), per-action asymmetries, and the behavior manifold figures. Anything that
moves gets flagged in the paper; anything stable becomes citable again. Cost: hours, CPU.

### D1. "When does the agent commit?" — agentic commitment-boundary phenomenology
The behavioral core of the paper, on faithful labels:
- Per decision step: retrospective stabilization time of argmax(q) and of the full q
  trajectory (à la 2605.06723), first/permanent lock-in, switch hazard vs progress,
  post-commitment tail fraction (agentic analogue of 2606.13603's 55%).
- Relate to task events (key pickup, door, turns) and to per-action asymmetries
  (LEFT vs RIGHT CoT length 2173 vs 1105 tokens on old labels — recheck).
- Deliverable: distribution of commitment times; how much of an agent's CoT is
  post-decision. Reuses scripts/04 + notebook 09 machinery. Cost: 1–2 days, CPU.

### D2. Commitment vs action identity live in different layers (the mechanism claim)
Upgrade the fold-0 observation to a defensible claim. The current evidence is methodologically
asymmetric: action = single-layer LDA decodability, commitment = transformer occlusion. Fix by
measuring both targets with the same instrument, then add the causal-parity and geometry views:

- **2a. Matched per-layer decoding curves.** For each layer L ∈ 0..23, pooled CoT-window
  features `pooled[:, L]` (512-d): ridge → entropy_commitment (Spearman, R²) and logistic →
  top_action (balanced acc). All 5 outer folds, subsets {all, strict_before_any_action_mention},
  with positional bars for both targets (position-only spline for commitment; position-only
  classifier for action). Reuse: `load_pooled_activations` (78), `prepare_table` /
  `load_outer_splits` / `clustered_bootstrap` (77), scoring from `mlp_branch_utils.scalar_metrics`.
  Cost: ~1 day CPU (pooled features for 98k rows ≈ 4.8 GB f32, loads once).
- **2b. Statistics.** Per-fold peak layers; early-vs-late block contrast (e.g. L0–7 vs L16–23)
  per target with fold-level paired tests + trajectory-clustered bootstrap; report the
  interaction (target × depth), which is the actual dissociation claim.
- **2c. Within-layer geometry.** Principal angles between the commitment ridge direction and
  the 3-d LDA action subspace per layer; projection-removal test (remove action subspace →
  does commitment decoding survive, and vice versa). Distinguishes "different layers" from
  "different directions everywhere".
- **2d. Occlusion parity.** Re-run 77's occlusion reading BOTH heads of the same trained
  probe: commitment Spearman drop AND q-head top-action balanced-acc drop per occluded layer.
  Same model, same rows → apples-to-apples occlusion curves.
- **2e. Cross-fold.** Train folds 1–4 of probe 77 (≈110 s/epoch, ~7 epochs each) so occlusion
  and headline Spearman get error bars.
- Anchor to two-stage decide/bind and confidence-before-content literature (§2).
  Cost: 2a–2c ~1 day CPU; 2d–2e ~1 day MLX/GPU.

### D3. Does the internal state commit before the words do? (the gap, white-box × time)
Apply the frozen fold-trained probe along every prefix of held-out trajectories to get an
activation-side commitment/belief trajectory; compare three clocks per decision step:
(i) activation-probe stabilization time, (ii) forced-answer behavioral stabilization time,
(iii) first verbal action mention. Lead/lag distributions define the action-belief gap as a
measurable time series. Old-label centroid-basin work found activation lock *lags* behavior
lock (median +0.697 progress) — recheck with the probe and faithful labels; if the probe
*leads* verbalization but lags forced-answer beliefs, that is a clean, publishable dissociation
of the three channels. Cost: ~1 day (probe inference is cheap; needs fold-trained checkpoints
from 2e for full coverage).

### D4. Score the 396 mismatched prompt/CoT counterfactuals (staged, never run)
`outputs/followups/mismatched_prompt_cot_prefixes.parquet` (+ matched controls) swaps prompts
and CoTs between steps; scoring them answers whether beliefs track the *prompt state* or the
*reasoning text* — the causal reading of "belief-action gap", and the direct bridge to
2605.00226. Run through the (now-trusted) extraction-style MLX scorer, not the quarantined
stage-03 path. Optionally extract activations for the same rows → does the probe track the
CoT or the prompt? Cost: hours.

### D5. What moves the belief? Token/sentence-level belief-update attribution
Consecutive prefixes differ by one token, so per-token Δq is already computable from
`extracted_beliefs.parquet` alone. Identify high-|Δq| tokens/sentences ("belief anchors"),
categorize their content (spatial reasoning, plan statements, corrections, backtracking),
and test whether anchor density predicts commitment speed. Optional causal upgrade on a
subset via Thought-Anchors-style resampling (2506.19143). Cost: analysis-only start.

### D6 (stretch). Layer-targeted causal patching
MLX intervention at layer 3 vs layer 18 during CoT (76's extractor shows the hook points):
patch/steer and test whether L18 edits flip the predicted action while L3 edits move
commitment magnitude. Turns D2's correlational dissociation causal. Cost: new infra, ~days.

### Hygiene backlog (not paper-blocking, note in appendix)
- Finish script 75 (sparse-native-attention probe) or formally close the remediation.
- Localize the stage-03 scorer defect (KV-cache prefix-trim path is the suspect).
- Finer terminal progress bins for the positional controls (0.8–1.0 bin still holds 0.42
  positional signal).

---

## 4. Suggested paper skeleton

Working title: *"When does an LLM agent commit? Belief dynamics along the chain of thought
and their layerwise substrate."*

1. Behavioral: multi-class agentic commitment dynamics with faithful labels (D0+D1) —
   first agentic, full-distribution version of the commitment-boundary phenomenon.
2. Decodability: commitment is linearly present in the residual stream and is not positional
   (existing fold-0 result + 2e cross-fold), with the fair/oracle positional-control
   framework as a methods contribution.
3. Mechanism: belief *strength* and belief *content* occupy different depths (D2), echoing
   decide-vs-bind; no naive geometric transfer of the behavioral simplex (existing negative).
4. The gap: three clocks — internal probe, forced-answer belief, verbalization — and their
   lead/lag structure (D3); counterfactual prompt/CoT dissociation (D4).

Recommended order: D0 → D2a–c + D1 (parallel, CPU) → D2d–e → D3 → D4 → D5/D6 as time allows.
