---
title: "The Action–Belief Gap Along the Chain of Thought: Precise Claims, Evidence, and Conjectures"
subtitle: "Theorem–proof style working notes for the gpt-oss-20b key-door gridworld corpus"
date: "2026-07-24"
---

**Scope.** These notes state every established result of the project as a precise claim
followed by its evidence, in the style of a mathematics paper: *Definitions* fix the
objects, *Lemmas/Propositions/Theorems* state what is known, *Evidence* blocks play the
role of proofs (they are empirical and cite the artifacts that reproduce them), *Remarks*
delimit scope, and *Conjectures* state what we believe but have not shown, each with a
proposed test and a falsification condition. All artifact paths are relative to the
repository root; dated entries refer to `RUN_HISTORY.md`.

# List of Figures

* **Figure 1** — Commitment versus CoT progress under faithful and quarantined labels, all and strict prefixes (Corollary 1.1, Proposition 2.1).
* **Figure 2** — Cumulative distributions of the three per-step clocks: verbalization, argmax stabilization, permanent commitment (Proposition 2.2).
* **Figure 3** — Switch hazard along the CoT, and irreversibility as a function of current commitment level (Propositions 2.4, 2.5).
* **Figure 4** — Heavy-tailed distribution of per-transition belief change, and where anchors occur along the CoT (Proposition 2.6).
* **Figure 5** — Content-category enrichment among anchor transitions (Proposition 2.6).
* **Figure 6** — Per-action commitment curves and the RIGHT-first prior asymmetry (Proposition 2.8).
* **Figure 7** — Mismatched prompt and CoT: the belief tracks the reasoning text, not the world state (Theorem 1).
* **Figure 8** — Matched per-layer decoding of top action and commitment, strict subset (Proposition 3.2).
* **Figure 9** — Overlap of the commitment direction with the action subspace, per layer (Theorem 3).
* **Figure 10** — PCA, LDA and CLR projections of pre-verbal activations: no four-arm manifold (Theorem 3).
* **Figure 11** — Dual-head layer-occlusion importance on a single trained probe (Proposition 3.3).
* **Figure 12** — Decoded probe clock versus behavioral clock on held-out steps (Observation 3.6).
* **Figure 13** — Gain versus additive decoders, and content-direction versus gain clocks (Conjecture C1).
* **Figure 14** — Belief damage from deleting whole attention or MLP sub-blocks, by layer (Conjecture C2).
* **Figure 15** — Belief dispersion after resampling at anchors versus position-matched controls (Conjecture C3).
* **Figure 16** — Switch hazard versus progress, marginally and within margin quintiles (Conjecture C4).
* **Figure 17** — Early-exit accuracy/savings frontier, and vocabulary before versus after settling (Conjecture C5).
* **Figure 18** — Belief transplantation as a function of how many dimensions are overwritten (Conjecture C6).
* **Figure 19** — Effect of an explicit disavowal on prompt- versus CoT-action probability (Conjecture C7).
* **Figure 20** — Frozen gridworld commitment direction applied to arithmetic chains of thought (Conjecture C8).
* **Figure 21** — Observed versus noise-simulated stabilization clocks (Conjecture C9).
* **Figure 22** — Decodability of the pre-reasoning belief, and its relation to settling time (Conjecture C10).
* **Figure 22b** — Synthesis: one decodability result against four intervention failures — the read/write asymmetry.
* **Figure 23** — Synthesis: two independent measurements of the strength/content dissociation.

# 1. Setting and Notation

**Definition 1 (Environment and agent).** The agent is the language model
$M$ = `openai/gpt-oss-20b` (24 transformer blocks, residual width 2880) acting in 2-D
key–door gridworlds. The action set is $\mathcal{A} = \{\mathrm{UP}, \mathrm{DOWN},
\mathrm{LEFT}, \mathrm{RIGHT}\}$, $|\mathcal{A}| = 4$. A *decision step* $s$ consists of a
prompt $x_s$ (the serialized world state), a chain of thought (CoT)
$c_s = (t_1, \dots, t_{N_s})$ of $N_s$ analysis tokens, and the action $a_s \in
\mathcal{A}$ the agent then takes. The corpus contains 95 trajectories, 1{,}276 decision
steps, and 98{,}310 CoT prefixes (median $N_s \approx 1{,}500$; mean per-step prefix count
$\approx 77$, cut stride $\approx 15$–$25$ tokens).

**Definition 2 (Forced-answer belief).** Let $\sigma$ denote the fixed 12-token forced
suffix (`<|end|><|start|>assistant<|channel|>final<|message|>{"action": "`, validated
token-exactly on load). For a cut $t \in \{0, \dots, N_s\}$, the *belief* is

$$q_s(t) \;=\; \mathrm{softmax}\big(\, \mathrm{logits}_M\!\left(x_s \,\|\, c_s[1{:}t] \,\|\, \sigma\right)\big|_{\mathcal{A}} \big) \;\in\; \Delta^3,$$

the softmax over the four single-token action candidates at the last position. The map
$t \mapsto q_s(t)$ is the *belief trajectory* of step $s$. The suffix $\sigma$ is used
**only** to evaluate this functional; it is never an input to any probe.

**Definition 3 (Commitment functionals).** For $q \in \Delta^3$:
$\kappa^{\mathrm{ent}}(q) = 1 - H(q)/\log 4$ (normalized-entropy commitment, the probe
target); $\kappa^{\mathrm{fin}}_s(t) = q_s(t)(a_s)$ (commitment to the final action);
$\kappa^{\mathrm{mar}}(q) = q_{(1)} - q_{(2)}$ (margin). *Progress* is $\tau = t / N_s \in
[0,1]$.

**Definition 4 (Clocks).** For each step $s$:

* $T^{\mathrm{stab}}_s$ (*argmax stabilization*): the progress of the first cut after
  which $\arg\max q_s(t)$ never changes again;
* $T^{\mathrm{perm}}_s(c)$ (*permanent crossing at level $c$*): the progress of the first
  cut after which $\kappa^{\mathrm{fin}}_s \ge c$ holds at every later cut;
* $T^{\mathrm{verb}}_s$ (*verbalization*): the progress of the first token index at which
  any action word appears in the CoT text;
* the *post-commitment tail* is $1 - T^{\mathrm{stab}}_s$ (also measured in tokens).

**Definition 5 (Strict pre-verbalization subset).** A prefix is *strict* iff its cut
precedes the first action mention of its step ($t < m_s$, vacuously true if no mention).
Strict prefixes number 31{,}723 (32.3%). All "pre-verbal" claims are evaluated on this
subset.

**Definition 6 (Activation objects).** $h^{(\ell)}_s(t) \in \mathbb{R}^{2880}$ is the
residual-stream state at the output of block $\ell \in \{0, \dots, 23\}$ (embedding not
stored; final norm applied only for logits). Stored activations are
$z^{(\ell)} = P^\top h^{(\ell)} \in \mathbb{R}^{512}$ for a fixed seeded scaled-orthogonal
$P \in \mathbb{R}^{2880 \times 512}$ ($P^\top P = \tfrac{2880}{512} I$; SHA-256 pinned).
The probe input for a prefix is the window of the last $W = 12$ genuine CoT tokens across
all 24 layers, $[24, 12, 512]$; $\bar z^{(\ell)}$ denotes the mask-weighted mean over the
window's valid tokens.

**Definition 7 (Anti-leakage probe contract).** Frozen backbone; activation-only inputs;
$< 10^6$ readout parameters; trajectory-grouped outer folds (5 folds, zero trajectory
overlap, verified at load); no grid/task features, no future tokens, no forced-suffix
states as inputs. Two positional baselines: the **fair bar** may use only the prefix token
index $t$ (decodable in principle from the inputs); **oracle bars** may additionally use
$N_s$ or $\tau$, which no causal state at cut $t$ can contain (the remaining tokens do not
yet exist).

**Definition 8 (Label regimes).** Three independent evaluations of Definition 2 exist:
(i) *stored* — the stage-03 MLX scorer with KV-cache reuse (98{,}311 rows);
(ii) *extracted* — recomputed during activation extraction, cache-validated per row
(98{,}310 rows); (iii) *Together* — independent API scoring, no cache reuse (33{,}147
rows / 32 trajectories). A fourth spot-check path scores rows with full, cache-free
forward passes.

# 2. The Foundation: Label Faithfulness

**Lemma 1 (Faithfulness of the extracted labels).** *The extracted labels are a correct
evaluation of the belief functional of Definition 2; the stored labels are not.*

**Evidence.** Three independent scoring paths agree with the extracted labels and
disagree with the stored ones. (a) Against Together (33{,}147 rows): extracted top-action
agreement $0.9707$, mean $\max_a |\Delta q_a| = 0.0258$; stored $0.7023$ and $0.2485$.
(b) Against cache-free full forwards (396 matched rows): extracted agreement $0.9823$,
mean $\max|\Delta q| = 0.0142$. (c) Internal consistency: at the final prefix ($t = N_s$)
the extracted top action equals the taken action $a_s$ in $100.0\%$ of the 1{,}276 steps
(a property any faithful evaluation must have, since the agent's decoded action was the
argmax continuation); in the near-terminal region $\tau > 0.9$ the original audit found
extracted agreement $0.991$ versus stored $0.908$. Overall, stored and extracted disagree
on $93.5\%$ of rows at tolerance $10^{-4}$, with mean row-wise $\max|\Delta q| = 0.235$
and top-action agreement only $0.719$. The stored scorer's KV-cache prefix-reuse path is
the suspected defect (not yet localized). Artifacts: `outputs/behavior_rebase_comparison.md`,
`outputs/followups/mismatched_prompt_cot_report.md`; entries 2026-07-21, -23. $\square$

**Corollary 1.1 (Quantitative corrections).** Under the faithful labels, stored-label
noise had *delayed apparent commitment and inflated apparent switching*. On identical
coverage (98{,}310 prefixes): mean $\kappa^{\mathrm{fin}}$ $0.550 \to 0.667$; fraction
$\arg\max q = a_s$ $0.580 \to 0.705$; median first crossing of $0.5$: $0.165 \to 0.089$;
median lock-in $0.288 \to 0.190$; steps with a lock-in $0.968 \to 0.998$; mean switches
per step $8.40 \to 6.04$ ($-28\%$); mean total variation $15.0 \to 10.9$. Every
qualitative claim (monotone rise, strict pre-verbal rise, terminal self-consistency)
survives the rebase. Rank stability across regimes: Spearman$(\kappa^{\mathrm{fin}}_{\mathrm{sto}}, \kappa^{\mathrm{fin}}_{\mathrm{ext}}) = 0.837$.

**Remark.** Any project that evaluates forced-answer beliefs through a KV-cache-reusing
scorer should replicate Lemma 1's triangulation before trusting its labels.

# 3. Behavioral Claims

**Proposition 2.1 (Monotone commitment with a pre-verbal component).** *Mean commitment
increases monotonically in progress, from $\approx 0.35$ at the prompt to $1.0$ at the end
of the CoT, and the rise begins before any action word is uttered.*

**Evidence.** Decile means of $\kappa^{\mathrm{fin}}$ rise strictly across all ten bins in
both the full set ($0.35 \to 0.97$) and the strict subset; at the prompt-only cut the mean
is $0.3375$; at the final prefix, $1.0000$. Within the strict subset, the first-decile mean
is $0.349$ and the mean at the last strict prefix of each step is $0.555$ with the argmax
already equal to $a_s$ in $59.1\%$ of steps — a genuinely pre-verbal drift toward the
eventual action. Artifacts: `outputs/commitment_report_extracted.md`,
`figures/behavior_rebase/commitment_vs_progress_rebase.png`. $\square$

![**Figure 1.** Mean commitment to the final action — the forced-answer probability of the action the agent actually took — against progress through the chain of thought, binned by decile over 95 trajectories, 1,276 decision steps and 98,310 prefixes. Left: all prefixes. Right: only the 31,723 strict prefixes (32.3%) cut before the first action word of their step. Solid line: the faithful "extracted" label regime; dashed line: the quarantined "stored" regime. Take-away: commitment rises monotonically from about 0.35 at the prompt to 1.0 at the end of the CoT, the rise is already underway before any action word is uttered, and the quarantined labels understate commitment at every progress level — mean 0.550 against 0.667 — without changing the shape of the curve.](figures/behavior_rebase/commitment_vs_progress_rebase.png)

**Proposition 2.2 (Clock ordering: the model speaks before it decides).** *The three
clocks of Definition 4 are ordered, in median, as*
$$T^{\mathrm{verb}} \;<\; T^{\mathrm{stab}} \;<\; T^{\mathrm{perm}}(0.8),$$
*with medians $0.253 < 0.494 < 0.648$, and the stabilization-before-verbalization event
$\{T^{\mathrm{stab}} < T^{\mathrm{verb}}\}$ has probability only $0.356$.*

**Evidence.** Per-step clocks over all 1{,}276 steps (every step has at least one action
mention). Median $T^{\mathrm{perm}}(0.5) = 0.521$. The median lead of stabilization over
mention is $-0.139$ progress (i.e., mention comes first). Interpretation fixed by the
data: early action mentions are predominantly *candidates under deliberation*, not
announcements of a made decision. Artifacts: `outputs/commitment_boundary_phenomenology.md`,
`figures/commitment_boundary/three_clocks_cdf.png`. $\square$

![**Figure 2.** Cumulative distributions, over all 1,276 decision steps, of three per-step clocks measured in progress through the chain of thought: the first verbal mention of any action word (median 0.253), the permanent stabilization of the forced-answer top action (median 0.494), and the permanent crossing of commitment level 0.8 (median 0.648). A curve lying higher and to the left means that event happens earlier. Take-away: verbalization comes first in the median step, so the model typically names an action well before its forced-answer choice stops changing — early action words are candidates under deliberation rather than announcements of a decision already made.](figures/commitment_boundary/three_clocks_cdf.png)

**Proposition 2.3 (The post-commitment tail is half the CoT).** *After the forced-answer
argmax has permanently settled, the model continues generating for a median $50.6\%$ of
its CoT (median $524$ tokens; mean $887$).*

**Evidence.** Distribution of $1 - T^{\mathrm{stab}}$ over steps: median $0.506$, IQR
$[0.276, 0.797]$. Short-CoT steps have the largest relative tails (quartile-1 median
stabilization $0.316$ vs quartile-4 $0.596$): when the answer is easy, most of the CoT is
post-hoc. This is the agentic, four-class analogue of the "epiphenomenal CoT" tail
reported for binary QA (arXiv 2606.13603, $\sim 55\%$), obtained with a belief functional
rather than early-exit accuracy. $\square$

**Proposition 2.4 (Declining hazard; no sharp aggregate boundary).** *The per-cut switch
hazard $\Pr[\arg\max q_s(t) \ne \arg\max q_s(t^-)]$ declines monotonically in progress,
$0.133 \to 0.019$ across deciles.*

**Evidence.** Pooled over 97{,}034 transitions; strictly decreasing across all ten
deciles. Note the compatibility with Proposition 2.6: aggregate smoothness coexists with
*individually* jumpy trajectories, because individual jump times are dispersed.
Artifact: `figures/commitment_boundary/switch_hazard_irreversibility.png`. $\square$

**Proposition 2.5 (Commitment is imperfectly absorbing).** *Conditional on current
commitment level, the probability of any later argmax switch decreases from $\approx 1$
(for $\kappa^{\mathrm{fin}} \le 0.35$) to $0.089$ (for $\kappa^{\mathrm{fin}} \in
[0.9, 1]$) — high commitment strongly predicts, but does not guarantee, finality.*

**Evidence.** Irreversibility curve over commitment deciles: $1.000, 1.000, 0.9996,
0.982, 0.893, 0.691, 0.572, 0.497, 0.374, 0.089$. $\square$

![**Figure 3.** Left: the per-cut probability that the forced-answer top action changes, by progress decile, pooled over 97,034 prefix-to-prefix transitions; it declines monotonically from 0.133 to 0.019, so there is no sharp aggregate decision boundary, only a steadily falling hazard. Right: the probability that at least one further top-action switch occurs later in the same step, as a function of the current commitment to the final action; it falls from 1.000 in the lowest commitment deciles to 0.089 in the highest. Take-away: commitment is strongly but imperfectly absorbing — even beliefs above 0.9 still reverse about 9% of the time — and aggregate smoothness on the left panel coexists with individually jumpy trajectories because jump times are dispersed across steps.](figures/commitment_boundary/switch_hazard_irreversibility.png)

**Proposition 2.6 (Belief change is heavy-tailed and anchored to switch events).** *The
top $5\%$ of transitions by total-variation change carry $37.0\%$ of all belief change;
$75.0\%$ of these "anchor" transitions coincide with an argmax switch (base rate
$7.9\%$); $40.8\%$ of anchors decrease commitment (reversals).*

**Evidence.** Per-transition $\mathrm{TV}_t = \tfrac{1}{2}\lVert q_s(t) - q_s(t^-)
\rVert_1$ over 97{,}034 transitions; anchor threshold $\mathrm{TV} \ge 0.328$ (95th
percentile); the median step concentrates $28.0\%$ of its belief change in its top-3
transitions; anchor rate declines with progress ($0.071 \to 0.018$ per decile). Content
is diagnostic: chunks containing none of {action word, spatial term, plan term,
correction marker} are $4.2\times$ depleted among anchors (rate $0.058$ vs $0.242$),
while action words are $1.63\times$, plan terms $1.56\times$, spatial terms $1.29\times$
enriched. Lexical, observational classification; causality deferred to Conjecture C3.
Artifacts: `outputs/belief_update_anchors.md`, `figures/belief_anchors/`. $\square$

![**Figure 4.** Left: complementary cumulative distribution, on a logarithmic vertical scale, of per-transition belief change measured as the total variation between consecutive forced-answer distributions, over 97,034 transitions; the dotted line marks the 95th-percentile "anchor" threshold of 0.328. Right: the anchor rate within each progress decile, against the uniform rate of 0.05 that a position-independent process would give. Take-away: belief change is heavy-tailed rather than incremental — the top 5% of transitions carry 37.0% of all belief movement — and anchors are front-loaded, their rate falling from 0.071 in the first decile to 0.018 in the last.](figures/belief_anchors/anchor_distribution.png)

![**Figure 5.** Enrichment of content categories among anchor transitions: how often each category appears in the chain-of-thought chunk added by an anchor transition, divided by how often it appears across all 97,034 transitions; the dotted line at 1.0 is no enrichment. Take-away: the largest belief updates are lexically diagnostic — chunks that mention an action word (1.63×), a plan term (1.56×) or a spatial term (1.29×) are over-represented among anchors, while chunks containing none of the four tracked categories are 4.2× depleted, a rate of 0.058 among anchors against 0.242 overall. The categorization is lexical and observational; it does not by itself show that the text causes the update.](figures/belief_anchors/anchor_category_enrichment.png)

**Proposition 2.7 (Update concentration predicts early settlement).** *Steps whose belief
change is concentrated in a few transitions stabilize earlier:*
$$\mathrm{Spearman}\big(\text{top-3 TV share}_s,\; T^{\mathrm{stab}}_s\big) = -0.510.$$

**Evidence.** Computed over 1{,}276 steps; concentration also anticorrelates with CoT
length ($\rho = -0.357$). Together with 2.4 and 2.6 this supports an *evidence-accumulation
picture with lumpy evidence*: a few decisive chunks settle the belief early when they
arrive early. $\square$

**Proposition 2.8 (Systematic per-action asymmetries).** *Actions differ in prior belief
and settlement time: at the prompt-only cut, mean $\kappa^{\mathrm{fin}}$ is RIGHT
$0.664 \gg$ LEFT $0.369 \gg$ UP $0.144 \approx$ DOWN $0.144$; median stabilization is
RIGHT $0.393 <$ LEFT $0.444 <$ UP $0.518 <$ DOWN $0.609$.*

**Evidence.** Per-action tables of the rebased corpus. The *ordering* is stable across
label regimes; stored-label magnitudes were unreliable (DOWN moved $0.351 \to 0.565$
under the rebase). RIGHT is also the majority action (26.9% of steps; class-imbalance is
why balanced accuracy is used everywhere). $\square$

![**Figure 6.** Mean commitment to the final action against chain-of-thought progress, computed on the faithful labels and split by which action the agent eventually took; left panel all prefixes, right panel only strict prefixes cut before the step's first action word. Take-away: the four actions do not start from the same prior — at the prompt-only cut mean commitment is RIGHT 0.664, LEFT 0.369, UP 0.144, DOWN 0.144 — and the ordering persists well into the CoT, with DOWN settling last (median stabilization 0.609, against 0.393 for RIGHT). RIGHT is also the majority action, taken at 26.9% of the 1,276 steps, which is why balanced accuracy is used for every action-decoding result in these notes.](figures/behavior_rebase/per_action_commitment_extracted.png)

**Theorem 1 (The belief is text-conditioned, not state-conditioned).** *Let $(x_A, c_B)$
be a mismatched pair: the prompt of step $A$ with the CoT prefix of step $B$, where
$a_A \ne a_B$. Then at every cut fraction the belief tracks the reasoning text:*
$$\mathbb{E}\big[q(a_B)\big] \in [0.598,\, 0.673] \quad\text{while}\quad \mathbb{E}\big[q(a_A)\big] \in [0.129,\, 0.165],$$
*already at cuts containing only $\sim 10\%$ of the foreign CoT. The conflicting prompt is
not inert: with congruent context the model follows the CoT's conclusion at rate $0.985$
(late cut), with conflicting context only $0.689$, and the missing mass disperses across
actions rather than returning to $a_A$.*

**Evidence.** 396 mismatched rows (cuts at $\sim 10/50/90\%$ of the CoT) plus 396
matched controls, all scored with full independent forward passes (no KV cache — Lemma 1
motivates the paranoia). Mean probabilities by cut: mismatched $q(a_B)$:
$0.598 / 0.625 / 0.673$; mismatched $q(a_A)$: $0.165 / 0.145 / 0.129$; matched control
$q(a_s)$: $0.355 / 0.684 / 0.971$; top-action rates analogous. Caveat: matched and
mismatched cells are different step samples, so cross-condition contrasts are
descriptive; the within-condition trends carry the claim. Relation to the literature:
this is the temporal/causal complement of the "belief–action gap" of arXiv 2605.00226 —
there, decodable beliefs fail to drive actions; here, the action-generating belief is
shown to be driven by the reasoning text rather than the world state. Artifacts:
`outputs/followups/mismatched_prompt_cot_report.md`,
`figures/counterfactual/mismatched_prompt_cot_tracking.png`. $\square$

![**Figure 7.** Mean forced-answer probability at three cut positions along the chain of thought, for 396 mismatched rows — the prompt of one decision step paired with the CoT prefix of another step whose action differs — and for 396 matched controls, all scored with independent cache-free forward passes. Take-away: at every cut the belief tracks the reasoning text rather than the world state described in the prompt: the probability of the CoT's action is 0.598 / 0.625 / 0.673 while the probability of the prompt's action is 0.165 / 0.145 / 0.129, already when only about 10% of the foreign CoT is present. The matched control rises 0.355 / 0.684 / 0.971. Matched and mismatched cells are different step samples, so the within-condition trends, not the cross-condition gap, carry the claim.](figures/counterfactual/mismatched_prompt_cot_tracking.png)

# 4. Representational Claims

**Theorem 2 (Commitment is linearly decodable and is not position).** *From
activation-only inputs (Definition 7), held-out entropy commitment is predicted at*
$$\mathrm{Spearman} = 0.7715 \pm 0.0094 \quad (5 \text{ folds}, R^2 = 0.509, \text{high/low AUC } 0.957),$$
*versus the fair positional bar $0.371 \pm 0.015$ and the oracle bars $0.644$–$0.660$; the
signal survives inside fixed progress bins ($0.52$–$0.71$) where the positional baseline
collapses ($0.334 \to -0.319$).*

**Evidence.** Best method `mean_layers_mlp` (learned per-layer encoder, mean-pooled over
layers; $<10^6$ readout parameters). Full 5-fold ladder: mean-pool $0.7715$ > concat
$0.7675$ > transformer-without-deltas $0.7634$ > layer-sequence transformer $0.7576$ >
best-single-layer $0.7474$ > final-layer $0.7045$ > length-oracle spline $0.6600$ >
progress oracle $0.6435$ > fair positional $0.3710$. Clustered-bootstrap CIs (fold 0)
separate probes from all baselines (e.g., vs fair bar $+0.399$, CI $[0.346, 0.455]$).
Artifacts: `.../fold0_20260721/summary.csv`, `report.md`; entries 2026-07-22, -24.
$\square$

**Proposition 3.1 (No layer-transition code).** *The hypothesis that commitment is
carried by layer-to-layer* transitions *(sequence structure over the depth axis) is
refuted: the depth-sequence transformer never beats mean-over-layers in any fold
($\Delta$Spearman $= -0.016$, CI $[-0.021, -0.011]$ on fold 0; ordering preserved in all
5 folds), and adjacent-layer delta features add nothing.*

**Evidence.** Theorem 2's ladder plus fold-0 bootstrap comparisons. This retires the
original `prompt.txt` premise of the layer-transition probe. $\square$

**Proposition 3.2 (Shared depth profile; content is slightly more depth-loaded).**
*Measured with the same instrument per layer (ridge for commitment, balanced logistic
for the top action, on $\bar z^{(\ell)}$), both quantities are decodable at every layer
and peak in the same mid-late band: action balanced accuracy peaks at layers 13–18
(strict-subset mean curve peak $0.521$ at L17 vs positional bar $0.331$; sharp jump
$0.466 \to 0.510$ from L10 to L11); commitment Spearman peaks at layers 13–15 (peak
$0.669$ at L15 vs bar $0.343$). The late-minus-early block contrast is positive for both
targets and larger for action in* every *fold: interaction $= +0.046$ on average
($+0.025/+0.045/+0.039/+0.041/+0.081$).*

**Evidence.** 24 layers $\times$ 5 folds $\times$ {strict, all}; `layerwise_curves.csv`,
`block_contrast.csv`, figure `layerwise_dissociation_strict.png`. This *replaces* the
earlier fold-0 reading "commitment at layer 3, action at layer 18", which arose from
comparing two different instruments (occlusion vs LDA) — see Proposition 3.4. $\square$

![**Figure 8.** Matched per-layer decoding from the same pooled chain-of-thought window activations at each of the 24 block outputs, on the strict pre-action-mention subset: the top action decoded by balanced logistic regression (left, balanced accuracy) and commitment decoded by ridge regression (right, Spearman correlation). The line is the mean over the five trajectory-grouped outer folds, the shaded band is the range across those folds, and the dotted line is the position-only baseline. Take-away: both quantities are decodable at every layer and peak in the same mid-late band — action 0.521 at layer 17 against a bar of 0.331, commitment 0.669 at layer 15 against a bar of 0.343 — with a sharp jump in action decodability from layer 10 to layer 11 (0.466 to 0.510). When the same instrument is used for both targets there is no depth-wise dissociation between belief strength and belief content.](figures/conceptual_belief_space/layerwise_dissociation_strict.png)

**Theorem 3 (Within-layer orthogonality of strength and content).** *At every layer, the
commitment direction and the action-identity subspace are almost orthogonal, and the two
signals are linearly separable: with $u_\ell$ the unit ridge direction for commitment and
$V_\ell$ the orthonormal 3-d LDA action subspace (both fitted on training folds),*
$$\lVert V_\ell^\top u_\ell \rVert^2 = 0.032 \text{ on average (chance } 3/512 = 0.006\text{)},$$
*minimized exactly in the best-decoding layers (L10: $0.010$; L16: $0.011$). Removing
$V_\ell$ from the features changes commitment decoding by $< 0.004$ Spearman; removing
$u_\ell$ changes action decoding by $< 0.005$ balanced accuracy — in every fold, at
layers 3, 10, 18 and the layer mean.*

**Evidence.** `geometry.csv`, `removal_tests.csv` (5 folds). Interpretation: belief
*strength* and belief *content* co-inhabit the same layers but occupy (linearly)
orthogonal directions — a two-channel code. The behavioral simplex geometry does not
transfer: supervised LDA of strict-subset activations yields one overlapping cloud with
displaced centroids (balanced accuracy $0.42$–$0.43$ vs chance $0.25$), not the
behavioral four-arm star, which is a property of $\Delta^3$ itself; the PCA null is a
variance-ordering artifact (top PCs track $\log t$, $|r|$ up to $0.57$; action silhouette
$-0.10$). $\square$

![**Figure 9.** Fraction of the commitment direction that lies inside the action-identity subspace, per layer: the squared cosine of the unit ridge commitment direction projected onto the orthonormal 3-dimensional LDA action subspace, both fitted on training folds only. The line is the mean and the band the range over the five outer folds; the dotted line is the chance level for a random direction in 512 dimensions, 3/512 = 0.006. Take-away: the overlap averages 0.032 and is smallest exactly at the layers where decoding is best (0.010 at layer 10, 0.011 at layer 16). Belief strength and belief content inhabit the same layers but occupy nearly orthogonal directions — a two-channel code rather than a depth-separated one.](figures/conceptual_belief_space/layerwise_commitment_action_geometry.png)

![**Figure 10.** Two-dimensional projections of held-out fold-0 activations from strict pre-action-mention prefixes, colored by the action the agent eventually took: unsupervised PCA (left), supervised four-class LDA (middle), and regression onto the CLR-transformed belief simplex (right); chance balanced accuracy is 0.25. Take-away: the four-arm star geometry of the behavioral belief simplex does not transfer to activation space. Even with supervision the four classes form a single overlapping cloud with merely displaced centroids (balanced accuracy 0.42 to 0.43), and the apparent structure in the unsupervised panel is a variance-ordering artifact whose leading components track token position (correlation up to 0.57 with log token index) rather than action identity (action silhouette -0.10).](figures/conceptual_belief_space/supervised_activation_projection.png)

**Proposition 3.3 (Layer 10 is the only reproducibly load-bearing layer).** *Reading
layer occlusions from both heads of the same trained probe: layer 10 is in the top-3
commitment-drop layers of* all five *folds and is the #1 layer in three; its mean drops
are $0.093$ (commitment Spearman) and $0.100$ (action balanced accuracy) over all folds
($0.108$ / $0.180$ on the folds with a trained q head). Every other salient layer is
idiosyncratic to its training run (fold 0: L3 with drop $0.168$; fold 2: L14 with
$0.125$).*

**Evidence.** Dual-head occlusion parity on all five fold checkpoints
(`occlusion_parity.csv`, `parity_split{1..4}/`). Two corollaries. (i) Occlusion
importance is largely a property of the *trained probe's* internal shortcut choices, not
of the model — only the L10 reliance replicates. (ii) The L10 reliance coincides with the
L10$\to$11 jump in linear decodability (Proposition 3.2): layer 10 is where action
information becomes linearly available, and it is what every probe leans on. Commitment
and action drops are *coupled* wherever measurable — occlusion produces no dissociation,
consistent with Theorem 3's claim that the dissociation is directional, not depth-wise.
$\square$

![**Figure 11.** Layer-occlusion importance read from both heads of a single trained probe (the fold-0 checkpoint): the drop in commitment Spearman (left) and in top-action balanced accuracy (right) when each of the 24 layers is zeroed in turn, shown for all held-out prefixes and for the strict pre-action-mention subset. Take-away: importance is spiky rather than diffuse, and the two heads spike at the same layers, so occlusion yields no commitment-versus-action dissociation. Across all five folds only layer 10 replicates — it is a top-3 commitment-drop layer in every fold, with mean drops of 0.093 in commitment Spearman and 0.100 in action balanced accuracy — whereas the other salient layers visible here, 3 and 18, are idiosyncratic to this one training run.](figures/conceptual_belief_space/occlusion_parity.png)

**Proposition 3.4 (Instrument-matching principle; a methodological caution).** *The
apparent depth dissociation ("commitment at L3, action at L18") was an artifact of
comparing occlusion importance for one target against single-layer decodability for the
other. Under instrument matching (same probe family per layer for both targets:
Proposition 3.2) and under occlusion parity (same trained model, both heads:
Proposition 3.3) the depth dissociation vanishes, and the real, replicable dissociation
is Theorem 3's orthogonality.*

**Evidence.** The refuting analyses are exactly 3.2 and 3.3; the superseded reading is
preserved, flagged, in the 2026-07-22 entries. $\square$

**Proposition 3.5 (Read/write asymmetry).** *The decoder directions are readable but not
writable: injecting $\alpha \cdot \mathrm{RMS} \cdot \hat d$ (for $\hat d$ the lifted
commitment or action direction, $P \hat d_{512} \in \mathbb{R}^{2880}$) into the last-12
CoT window at any single layer $\ell \in \{3, 10, 15, 18\}$, at strengths up to $\alpha =
2$ (twice the local residual RMS), produces no effect distinguishable from a norm-matched
random direction: $|\mathbb{E}[\Delta \kappa^{\mathrm{ent}}]| \le 0.005$ in every cell
(sd $\approx 0.015$), argmax flip rate $0$ except a single prefix at $\alpha = 2$ — for
the action direction* and *for the random control alike ($n = 28$ prefixes, paired
within-prefix, 1{,}148 steered forwards).*

**Evidence.** `layer_steering/steering_full_n28.parquet`, `steering_summary_full_n28.csv`.
The contrast that makes this informative: the same directions support Spearman $0.67$
single-layer decoding (Prop. 3.2), and *zeroing* layer 10 costs the trained probe
$0.11$–$0.23$ (Prop. 3.3). Hence the belief encoding is redundant/distributed enough to
absorb rank-1, window-local perturbations. Untested escalations are Conjecture C6.
$\square$

**Observation 3.6 (Decoded clocks are biased late; lead/lag is open).** Applying the
frozen probe along held-out trajectories reconstructs the within-step commitment *curve*
well (median per-step Spearman $0.74$; final-prefix decoded argmax matches truth $95\%$),
but the decoded argmax-stabilization clock lags the behavioral one grossly (median
$0.99$ vs $0.51$; still $0.92$ after 5-point smoothing) because decoder noise at
$\approx 0.5$ balanced accuracy produces spurious late flips — a *measurement* bias, not
a finding about the model. The permanent-crossing clocks correlate $0.34$–$0.40$ across
steps. Whether the internal state settles before the forced-answer channel is therefore
open; see Conjecture C9. (Evaluated on 272 fold-0 steps and replicated on 793 pooled
steps from the folds with trained q heads.)

![**Figure 12.** Clocks decoded by the frozen probe compared with the behavioral clocks, on 272 held-out fold-0 decision steps. Left: cumulative distributions over progress of the probe's top-action stabilization, the forced-answer top-action stabilization, the first verbal action mention, and the probe's permanent crossing of commitment 0.8. Right: per-step scatter of probe stabilization against forced-answer stabilization, with the identity line dotted. Take-away: the decoded stabilization clock is grossly biased late — median 0.99 against the behavioral 0.51, and nearly every step sits above the diagonal — because a decoder running at roughly 0.5 balanced accuracy manufactures spurious late flips. This is a measurement artifact of the decoder, not a property of the model, so whether the internal state settles before the forced-answer channel remains open.](figures/conceptual_belief_space/three_clocks_lead_lag.png)

**Remark (model-selection artifact, recorded for reruns).** Probe model selection
optimizes validation commitment Spearman only; folds 2 and 3 therefore selected
$\lambda_q = 0$ and shipped untrained q heads (held-out top-action accuracy $0.34/0.24$
vs $\approx 0.50$ on folds 0/1/4). Commitment results are unaffected; q-head-dependent
analyses use folds 0/1/4 only. Fix: a $\lambda_q$ floor or multi-objective selection.

# 5. Conjectures, and What Happened When We Tested Them

Each conjecture states a precise expectation, its motivation in the results above, a
concrete test executable with existing repo infrastructure, and what would falsify it.
**All ten were subsequently run** (2026-07-24); each is followed by a **Result** paragraph
reporting the outcome, in the same role that *Evidence* plays for the established claims.
Outcomes are summarized in Table 1 and detailed in §6.

: **Table 1.** Outcome of all ten pre-registered conjecture tests. "Split" means the
conjecture bundled several predictions that did not all survive.

| # | Claim in one line | Outcome |
|---|---|---|
| C1 | Belief code = content direction $\times$ scalar gain | **Partial** — gain helps, but it is not the commitment axis |
| C2 | Layer 10–11 attention is a routing bottleneck of the model | **Refuted** — no sub-block matters much anywhere |
| C3 | Anchors are causal forks | **Underpowered** — right direction (1.26$\times$), $p = 0.32$ |
| C4 | The belief screens off position in the switch hazard | **Refuted** — replaced by a sign-reversing interaction |
| C5 | The tail is verification, and is cheap to cut | **Split** — lexically yes; cuttable only to 93.3% at 28.3% savings |
| C6 | Writability needs higher rank or breadth | **Refuted** — full window swap flips only 20.8% |
| C7 | CoT dominance is a disruptable authorship prior | **Refuted** — 0.14% of the lost mass recovered |
| C8 | The commitment direction is a task-general confidence axis | **Supported** — frozen axis transfers to arithmetic at $\rho = 0.51$ |
| C9 | The decoded clock's lag is decoder noise | **Supported** — observed lag is below the noise floor |
| C10 | A static prior direction explains the RIGHT-first asymmetry | **Split** — prior is decodable ($R^2 = 0.673$); its *content* predicts nothing |

**Conjecture C1 (Gain–direction factorization; the two-channel code is multiplicative).**
*There exist per-action content directions $\{u_a\}_{a \in \mathcal{A}}$ and a scalar
gain $\beta_s(t) \ge 0$, decodable from the commitment direction, such that*
$$q_s(t) \;\approx\; \mathrm{softmax}\big(\beta_s(t) \cdot \langle u_a, h_s(t)\rangle_{a \in \mathcal{A}}\big),$$
*i.e., content fixes the direction of the logit vector and commitment is its gain.*
Motivation: Theorem 3 (orthogonal strength/content channels) plus the monotone growth of
$\kappa$ (Prop. 2.1) suggest annealing of an inverse-temperature-like gain along the CoT.
Test: fit the bilinear model on strict training rows (alternating least squares over
$\{u_a\}, \beta$); check (i) held-out fit beats the additive linear decoder at equal
parameter count, (ii) the fitted $\beta$ correlates with the projection onto the ridge
commitment direction ($\rho > 0.6$), (iii) $\beta$ rises along the CoT while the
direction $\langle u_a, h\rangle / \lVert \cdot \rVert$ stabilizes *earlier* than
$\beta$. Falsified if the bilinear fit does not beat additive, or if $\beta$ is
uncorrelated with the commitment direction.

**Result — PARTIAL.** (i) holds and cleanly: a gain model
$\mathrm{softmax}(\mathrm{softplus}(\langle w, x\rangle + c)\cdot \mathrm{unit}(Ux+d))$
beats an additive decoder at every representation (held-out $\mathrm{KL}$, all-layer mean
$0.815 \to 0.728$; L15 $0.840 \to 0.748$; L18 $0.826 \to 0.742$), and a **frozen random**
gain direction is *worse* than additive ($0.840/0.839/0.841$), so the learned gain
direction — not the 25% extra capacity — carries the improvement. (ii) fails: the learned
gain barely tracks the independent ridge commitment axis ($\rho = +0.132$; against true
commitment $+0.142$). (iii) fails: the content direction settles before the gain in
$52.9\%$ of steps, i.e. at chance. An unplanned observation strengthens Theorem 3 from a
new angle: the additive decoder's *implied* confidence is essentially uninformative about
true commitment ($\rho$ between $-0.125$ and $+0.043$ across representations, i.e. scattered
around zero), and the learned gain lifts it only to $+0.120$ to $+0.175$, versus $0.7715$
for a dedicated commitment probe — decoding the belief's direction does not deliver its
strength (Figure 23, left). Artifact: `outputs/conjectures/c1_gain_direction/`.

![**Figure 13.** Left: held-out KL of three decoders of the belief from pooled activations — additive, a gain model whose gain direction is frozen at random, and a gain model with a learned gain direction (lower is better; fold 0, all-layer mean). The learned gain is the only variant that improves on additive, so the gain direction carries information rather than capacity. Right: per held-out step, the progress at which the decoded content direction settles versus the progress at which the gain plateaus; points scatter around the diagonal, so content does not reliably settle before confidence.](figures/conjectures/c1_gain_direction.png)

**Conjecture C2 (Layer 10 is a routing bottleneck of the model, not just of the
probes).** *Ablating a small set of attention heads at layers 10–11 in the model itself
degrades the belief trajectory (flatter $\kappa$ curves, later $T^{\mathrm{stab}}$, more
switches) more than matched ablations at layers 6–9 or 12–15.* Motivation: the unique
cross-fold reliance on L10 (Prop. 3.3) and the L10$\to$11 decodability jump (Prop. 3.2)
suggest a genuine computational site where spatial evidence is compiled into an action
plan. Test: MLX head-masking at inference over $\sim 200$ prefixes; score forced-answer
beliefs (script-10 style, cache-free); compare curves per ablation site. Falsified if
belief degradation is flat in layer or head choice.

**Result — REFUTED, and it corrects an over-reading of Proposition 3.3.** Rather than
masking individual heads, the sharper version of the test skips an entire sub-block
(attention or MLP) at one layer during a manual forward, which strictly dominates
head-masking as an upper bound on that sub-block's contribution. Over 24 strict fold-0
prefixes and 14 layers (manual-versus-standard forward equivalence exactly $0$, so the
null is not a reimplementation artifact), *no* sub-block anywhere matters much: the
largest mean $\mathrm{KL}(\text{baseline} \| \text{ablated})$ over all 28 conditions is
$0.097$ (L16 MLP), and attention ablation at L10–11 averages $0.0658$ versus $0.0426$
elsewhere — $1.55\times$, short of the $2\times$ criterion. Decisively, **ablating layer
10's attention ($0.0487$) damages the belief less than ablating layer 0's ($0.0856$)**.

The methodological consequence is worth more than the conjecture was: layer 10's unique
status in Proposition 3.3 is a fact about where the belief becomes linearly **readable**,
not about where it is computed. Probe-occlusion importance and model-ablation importance
are different quantities and here they disagree, so Proposition 3.3 must be stated as a
decodability landmark and never as a causal one. Artifact:
`outputs/conjectures/c2_subblock/`.

![**Figure 14.** Cost of deleting one whole sub-block at a single layer, over 24 pre-verbal prefixes: each row pairs the damage from removing that layer's attention against removing its MLP, measured as divergence from the unablated belief. The layers predicted to be a routing bottleneck are darkened. Damage is small everywhere — the worst of 28 conditions costs under 0.1 nats — and the two largest effects sit at layer 16's MLP and layer 0's attention, not at the predicted site.](figures/conjectures/c2_subblock_paired.png)

**Conjecture C3 (Anchors are causal, not just coincident).** *Resampling the CoT from an
anchor transition changes the final-action distribution more than resampling from
TV-matched... non-anchor transitions of the same step: concretely, resampling divergence
(e.g., total variation of final-action distributions across 30 resamples) is at least
twice as large at anchors.* Motivation: Prop. 2.6–2.7; the Thought-Anchors methodology
(arXiv 2506.19143) provides the template. Test: 20 steps $\times$ (top-2 anchors + 2
matched controls) $\times$ 30 resamples at temperature 1 via the local MLX generator;
compare divergences. Falsified if anchor and control divergences are indistinguishable —
which would demote anchors to markers of *already-made* internal updates (itself
informative for C9).

**Result — NOT SUPPORTED AT THIS SAMPLE SIZE (underpowered, not refuted).** Two
specification repairs were needed. "TV-matched controls" is incoherent — a control with an
anchor's total variation *is* an anchor — so controls are **position-matched** non-anchor
transitions of the same step, which is what isolates pivotality from lateness. And rather
than generating to completion, each branch resamples a $60$-token continuation and then
forces the answer, measuring dispersion of the resulting beliefs. Over 20 anchors and 20
position-matched controls from 7 steps ($10$ resamples each), anchor dispersion (mean
pairwise total variation across resamples) is $0.208$ versus $0.165$ for controls — a ratio
of $1.26$ against the $2.0$ criterion, Mann–Whitney $p = 0.32$; distinct sampled answers
$1.70$ versus $1.45$. The manipulation check passes (controls match on progress, $0.504$ vs
$0.507$, while differing $9\times$ in observed TV, $0.068$ vs $0.683$), so this is a
power problem, not a design problem: both point estimates favour the conjecture and neither
approaches significance at $n = 20$ per arm. A properly powered rerun ($\ge 60$ steps,
$\ge 30$ resamples) is the honest follow-up. Artifact: `outputs/conjectures/c3_anchors/`.

![**Figure 15.** Dispersion of the forced-answer belief across resampled continuations, branching at anchor transitions versus position-matched non-anchor controls (20 each, 10 resamples per branch). Left: cumulative distributions; anchors are shifted toward higher dispersion but the distributions overlap heavily. Right: the same values against branch position, confirming the two groups are matched on where in the chain of thought they sit.](figures/conjectures/c3_anchor_resampling.png)

**Conjecture C4 (Markov-in-belief evidence accumulation).** *The switch hazard is
screened off from progress by the current belief: conditional on margin
$\kappa^{\mathrm{mar}}_s(t)$, the hazard is (approximately) independent of $\tau$,*
$$\Pr[\text{switch at } t \mid \kappa^{\mathrm{mar}}, \tau] \approx \Pr[\text{switch at } t \mid \kappa^{\mathrm{mar}}].$$
Motivation: the declining hazard (2.4), imperfect absorption ordered by level (2.5), and
lumpy updates (2.6) resemble a race/drift-diffusion process whose state is the belief
itself. Test: logistic hazard regression on 97k transitions with both predictors and an
interaction; compute partial dependence. Falsified if $\tau$ retains substantial hazard
information given margin (which would instead indicate an explicit position/schedule
signal in the dynamics — connecting to the positional confound literature).

**Result — REFUTED, and replaced by a sharper claim.** In an *additive* spline model,
adding progress to margin gains nothing (held-out AUC $0.8474 \to 0.8452$), which looks
like screening-off and would have been reported as confirmation had the test stopped
there. It does not, because an additive model cannot represent the effect that is present.
An interaction-capable model does gain (gradient boosting, AUC $0.8471 \to 0.8520$,
log-loss $0.21362 \to 0.21128$), and the stratified hazard shows why: **the position effect
reverses sign across belief levels.** In the lowest margin quintile the hazard *rises* with
progress ($0.242 \to 0.370$; the second quintile rises $2.68\times$), while in the top
quintile it *falls* roughly tenfold ($0.0156 \to 0.0016$). The two cancel in any additive
fit.

The replacement claim is more informative than the original: position is not redundant
given the belief, it is a **credibility amplifier**. Late-and-confident is far more final
than early-and-confident; late-and-unsure is far more volatile than early-and-unsure. A
step still undecided deep into its chain of thought is in a genuinely unstable regime.
Artifact: `outputs/conjectures/c4_hazard/`.

![**Figure 16.** Left: the marginal probability that the top action switches at the next prefix, against progress through the chain of thought (97,034 transitions). Right: the same hazard computed within margin quintiles; the curves fan in opposite directions, rising with progress for weakly-held beliefs and falling for strongly-held ones, which is why adding position to margin buys nothing in an additive model.](figures/conjectures/c4_hazard_markov.png)

**Conjecture C5 (The tail is verification, and it is cheap to cut).** *The
post-commitment tail (Prop. 2.3) is functionally a serialization/verification phase:
(i) forcing the answer at $T^{\mathrm{perm}}(0.8)$ reproduces $a_s$ in $\ge 95\%$ of
steps at $\sim 35\%$ token savings; (ii) tail text is enriched in verification/plan
vocabulary relative to pre-stabilization text; (iii) the $8.9\%$ late reversals
(Prop. 2.5) concentrate in steps whose tails contain correction markers.* Test: (i) is a
pure relabeling of existing per-step data (the belief at the crossing cut *is* the forced
answer there); (ii)–(iii) reuse the D5 lexicon on tail vs head segments. Falsified if
late reversals are unpredictable from tail content, or early forcing loses $> 10\%$
of actions.

**Result — SPLIT; part (i) as originally posed was vacuous and had to be rebuilt.**
Forcing the answer at $T^{\mathrm{perm}}(0.8)$ trivially reproduces $a_s$: at that cut
$\kappa^{\mathrm{fin}} \ge 0.8 > 0.5$, so the argmax *is* $a_s$ by construction, and the
crossing time is not knowable online in any case. The non-vacuous replacement sweeps
genuinely prospective rules — threshold $c \in \{0.3, \dots, 0.9\}$ $\times$ persistence
$k \in \{1,3,5\}$ consecutive prefixes above $c$ $\times$ token floor $\in \{0,200,500\}$,
none of which uses the unknowable $N_s$ — and takes the Pareto frontier over all 63.

The tail is **partially** cuttable. The best online rule (margin $\ge 0.9$ sustained for 5
prefixes after 500 tokens) reaches $93.3\%$ agreement with the full-CoT action at $28.3\%$
mean token savings; only 2 of 63 rules clear $90\%$ and **none clears $95\%$**. Debouncing
matters enormously — the naive "exit at first crossing" rule scores $42.2\%$, because the
belief makes confident transient excursions to the wrong action (the same phenomenon as
Proposition 2.5's imperfect absorption, now costed in decision terms). Compared with the
$\sim 55\%$ cut reported for QA reasoning (arXiv 2606.13603), the agentic setting is
markedly less cuttable at comparable fidelity.

Parts (ii) and (iii) hold decisively. Verification vocabulary is $5.23\times$ denser after
stabilization than before ($0.62 \to 3.26$ hits per 1000 characters; higher in $93.7\%$ of
steps, Wilcoxon $p$ below double-precision resolution), action words $2.72\times$ and plan
terms $1.59\times$, while **spatial vocabulary is nearly flat ($1.11\times$)** — the tail
stops reasoning about space and starts verifying and naming. And steps that reverse out of
a high-commitment state carry $1.39\times$ more correction markers and $1.89\times$ more
action words in their pre-stabilization text, so reversals are lexically foreshadowed.
Artifact: `outputs/conjectures/c5_tail/`.

![**Figure 17.** Left: accuracy against token savings for every prospective early-exit rule (grey), with the Pareto frontier in blue; the dotted line is the 95% agreement target, which no online rule reaches. Right: density of five vocabulary classes before versus after the belief's argmax permanently settles, per 1000 characters; verification and action language rise sharply while spatial language is unchanged.](figures/conjectures/c5_tail_verification.png)

**Conjecture C6 (Writability requires rank or breadth).** *The belief is writable, but
only by higher-rank or wider interventions than Prop. 3.5's rank-1 window-local
injection. Specifically, (a) swapping the entire $12 \times 2880$ window state at one
layer between a donor prefix (different top action) and a recipient moves
$q$ toward the donor's action in $> 50\%$ of pairs at some layer $\ell^* \in [10, 18]$;
(b) effect size is monotone in patch rank (direction $<$ 3-d action subspace $<$ full
window swap) and in breadth (12 tokens $<$ all CoT tokens).* Motivation: occlusion
sensitivity (3.3) proves layer content matters causally; the steering null (3.5) shows
rank-1 is absorbed; between them lies a rank threshold worth measuring. Test: extend
script 82 with `--mode patch` (donor/recipient pairs matched by step position);
$n = 32$ pairs $\times$ 4 layers. Falsified if even full-window swaps at every layer
leave the forced answer unchanged — which would imply the belief is reconstructed from
outside the probe's window (prompt tokens or earlier CoT), a finding that would itself
redraw the map.

**Result — REFUTED; the falsification branch is what happened, and the map is redrawn.**
The test was strengthened into a *nested transplant ladder*: a donor prefix whose forced
answer differs from the recipient's supplies the replacement content, and the four modes
differ only in how many of the $2880$ dimensions are overwritten — rank-1 along the
commitment axis, rank-1 along the donor's action direction, rank-3 along the LDA action
subspace, and the full $12 \times 2880$ window. Because the donor states are identical
across modes, rank is the only variable.

Over 24 pairs at layers $3/10/15/18$: rank-1 and rank-3 do **nothing** (flip rate exactly
$0$ everywhere, $|\Delta \bar q(\text{donor action})| \le 0.003$), and full-window
replacement — overwriting the probe's entire input — flips only $20.8\%$ of pairs at its
best layer ($15$), with $8.3\%$ at L3, $16.7\%$ at L10 and $0\%$ at L18. The ordering is
monotone in rank as predicted, but the level never approaches the $50\%$ criterion.

So transplanting everything the probe reads, at the layer where it reads it best, usually
fails to change the answer. The belief must therefore be reconstructed from context outside
the 12-token window — the prompt and the earlier chain of thought — and the window the
probe reads is where the belief is *legible*, not where it *lives*. Artifact:
`outputs/conjectures/c6_patching/`.

![**Figure 18.** Transplanting a donor prefix's activations into a recipient with a different forced answer, as a function of how many dimensions are overwritten. Left: mean change in the probability of the donor's action; right: fraction of pairs whose answer becomes the donor's. Rank-1 and rank-3 patches are inert at every layer; only replacing the entire window has any effect, and it succeeds in barely a fifth of pairs.](figures/conjectures/c6_patching.png)

**Conjecture C7 (Text dominance is self-conditioning, and it is disruptable).** *The
belief tracks the CoT (Theorem 1) because in-context reasoning acts as self-generated
evidence with an implicit authorship prior. Inserting an explicit disavowal between CoT
and suffix ("The above reasoning concerns a different maze; decide from the map alone.")
recovers at least half of the prompt-action mass lost in the mismatched condition
($\mathbb{E}[q(a_A)]$ rising from $\approx 0.15$ to $\ge 0.4$), while barely affecting
matched controls.* Test: rescore the 396 mismatched + 396 matched rows with the
disavowal insert (script-10 variant; $\sim$1 GPU-hour). Falsified if the disavowal
leaves $q(a_A)$ unmoved — evidence that text dominance is architectural (recency/attention
mass) rather than inferential (authorship attribution).

**Result — REFUTED.** Appending "the reasoning written above was copied from a different
maze … disregard it entirely and choose using only the map in the task description" to the
analysis channel, on the same 792 rows as Theorem 1 and paired row-for-row, moves mean
$q(a_A)$ from $0.1463$ to $0.1475$: it recovers **$0.14\%$** of the mass the mismatch
costs, against a predicted $\ge 50\%$. The prompt's action becomes the argmax in $13.1\%$
of rows versus $13.6\%$ at baseline — no improvement at all.

The effect is statistically reliable but numerically trivial ($72.2\%$ of rows move in the
predicted direction, Wilcoxon $p = 1.3 \times 10^{-8}$): the model registers the sentence
and does essentially nothing with it. Matched controls move by a comparable magnitude in
the *harmful* direction ($-0.037$, $-0.032$, $-0.011$ across the three cut positions), so
what the instruction produces is a small generic loss of confidence rather than a targeted
reattribution. Text dominance is therefore not an inferential authorship prior that can be
argued away; the architectural reading survives. Artifact:
`outputs/conjectures/c7_disavowal/`.

![**Figure 19.** Probability assigned to the prompt's action and to the foreign chain of thought's action, with and without an explicit instruction to disregard the foreign reasoning, at three cut positions (396 mismatched pairs). The disavowal leaves both curves essentially where they were: the model does not return belief to the world state it was told to rely on.](figures/conjectures/c7_disavowal.png)

**Conjecture C8 (The commitment direction is a task-general confidence axis).** *The
ridge commitment direction $u_\ell$ transfers zero-shot: frozen $u_{\ell^*}$ (fitted on
gridworld) decodes normalized-entropy confidence above the positional bar on a different
forced-choice task's CoTs (e.g., 4-way arithmetic or MCQ), at the same layers.*
Motivation: confidence-probing literature finds early-layer, content-independent
confidence codes (arXiv 2601.03914, 2603.17839); Theorem 3 shows our commitment channel
is content-orthogonal, which is what a generic confidence axis would look like. Test:
generate $\sim 20$ CoTs on a 4-option MCQ task with the same model, extract the same
window states, apply frozen $u_\ell$; compare to task-refit ceiling. Falsified if
transfer is at the positional bar — commitment would then be a task-specific quantity,
strengthening the action-plan reading instead.

**Result — SUPPORTED.** The transfer task holds the *output interface* exactly fixed while
removing all spatial content: sixteen arithmetic and logic questions whose four options are
labelled UP/DOWN/LEFT/RIGHT — arbitrary names, stated as such — answered through the
identical JSON forced suffix. The model solves them ($93.8\%$ final-cut accuracy), so the
beliefs being measured are real. Commitment directions were frozen from gridworld fold-0
strict training rows, including their standardization, and applied unchanged to 320
new-task prefixes.

At layer 13 the frozen gridworld direction attains Spearman $+0.515$ with the new task's
normalized-entropy commitment, against a position-only bar of $+0.384$ and a task-refit
ceiling of $+0.612$ — that is, **$84\%$ of what a probe trained on the new task itself
achieves, with zero new fitting**. Transfer is strong across the middle of the network
(L10 $+0.371$, L13 $+0.515$, L15 $+0.391$) and degrades at the ends (L18 $-0.074$, L23
$+0.129$). One anomaly deserves flagging rather than burying: at layer 3 the transfer is
large but **sign-inverted** ($-0.514$), so the early-layer axis is not the same quantity
despite carrying comparable information.

Commitment is therefore substantially a task-general confidence code at mid-depth, not a
gridworld-specific action-plan quantity — which is what Theorem 3's content-orthogonality
predicted and what the confidence-probing literature reports. Caveats: 16 questions with
short chains (37–228 tokens), and the positional bar is high because commitment rises with
position in both tasks. Artifact: `outputs/conjectures/c8_transfer/`.

![**Figure 20.** A commitment direction fitted only on gridworld activations, applied without refitting to chains of thought on four-option arithmetic questions that share the answer format but nothing else. Its correlation with the new task's confidence (solid) is compared against a position-only baseline (dotted) and against a probe refitted on the new task (dashed). Through the middle layers the frozen direction recovers most of the refitted ceiling.](figures/conjectures/c8_confidence_transfer.png)

**Conjecture C9 (The internal state leads the forced-answer channel; the observed lag is
decoder noise).** *For decoders with balanced accuracy $b$, the decoded stabilization
bias is a known increasing function of $1 - b$; extrapolating the smoothing/quality curve
to $b \to 1$ yields $T^{\mathrm{probe}} \le T^{\mathrm{stab}}$ in the majority of steps.*
Motivation: Obs. 3.6 showed the raw lag shrinks with every denoising step (0.99 raw,
0.92 smoothed); pre-verbal decodability (Prop. 2.1, 3.2) shows the information is present
early. Test without new training: noise-matched simulation — corrupt the *true* q
trajectories with the decoder's confusion matrix, recompute stabilization, and compare
the simulated lag to the observed lag; if observed $\le$ simulated, the internal state
leads (or matches) the forced-answer clock. Falsified if the observed lag exceeds the
noise-simulated lag — a genuine internal lag, i.e., the forced-answer channel integrates
information not yet linearly present in the window states.

**Result — SUPPORTED.** The probe's prefix-level argmax accuracy is $0.4998$ and its errors
are strongly autocorrelated ($\Pr[\text{err} \mid \text{err}] = 0.667$ against a $0.50$ base
rate), so two noise models were simulated: i.i.d. draws from the empirical confusion matrix
and a persistence-matched two-state error process. Both corrupt the **true** trajectories,
so both have zero lag by construction.

A zero-lag process carrying the probe's own noise manufactures a median stabilization lag
of $0.483$ (i.i.d.) or $0.475$ (persistence-matched). The lag actually observed is
$0.432$ — **smaller than the noise floor** (mean observed-minus-simulated $-0.033$ to
$-0.076$; the observed clock is earlier in $53$–$63\%$ of steps). Observation 3.6's
apparent internal lag is therefore fully accounted for by decoder noise and should not be
reported as a property of the model.

Note what this does and does not license. It removes the evidence for an internal *lag*; it
does not establish an internal *lead*, because a decoder at $0.50$ accuracy cannot resolve
one. The lead/lag question stays open in the one direction that would be interesting, and
needs a stronger decoder rather than a cleverer statistic. Artifact:
`outputs/conjectures/c9_clock_noise/`.

![**Figure 21.** Cumulative distributions of when the top action permanently settles, on held-out steps: the true forced-answer clock, the clock read off the probe, and two simulated clocks obtained by corrupting the true trajectories with the probe's own error statistics. The simulated clocks are later than the observed one, so the probe's apparent lateness is a noise artifact rather than evidence about the model.](figures/conjectures/c9_clock_noise.png)

**Conjecture C10 (A static default-action direction explains the prior asymmetry).**
*Prompt-only beliefs (before any CoT) are decodable from the prompt's final-token states
by a single linear map, and its RIGHT-component explains the RIGHT-first asymmetry of
Prop. 2.8: steps whose prompt states project high on the "default RIGHT" direction are
exactly those that settle early.* Test: linear decode of $q_s(0)$ from prompt-end
activations (already in the shards as the $t = 0$ window); correlate the projection with
$T^{\mathrm{stab}}$. Falsified if prompt-only beliefs are not linearly decodable
($R^2 \approx 0$) — which would locate the prior in the generation dynamics instead.

**Result — SPLIT, and the split repeats the theme.** The prompt-only cut carries exactly
one valid window token per step (verified, $1276/1276$), so its activation is literally the
last state before any chain of thought exists, and there is no positional confound to
control because every row sits at the same position.

Part (a) holds strongly. The pre-CoT belief is linearly decodable from that single state,
held-out $R^2$ rising $0.131$ (L0) $\to 0.431$ (L10) $\to \mathbf{0.673}$ (L18) $\to 0.613$
(L23), with balanced accuracy $0.508$ against a chance of $0.25$. The prior itself is
stark: before any reasoning, the argmax is RIGHT in $79.5\%$ of steps, LEFT in $19.2\%$,
UP in $0.9\%$ and DOWN in $0.4\%$.

Part (b) fails, and instructively. The decoded RIGHT component does **not** predict when a
step settles ($\rho = -0.054$). The prompt's commitment **magnitude** does ($\rho =
-0.341$). Which action the model leans toward at the outset says nothing about when it will
commit; how strongly it leans says a great deal. This is the strength/content dissociation
of Theorem 3 appearing a third time — now in the prior, before a single reasoning token has
been emitted. Artifact: `outputs/conjectures/c10_prompt_prior/`.

![**Figure 22.** Left: how well the belief held before any reasoning can be read off the single prompt-final residual state, by layer; it peaks near layer 18 at an R-squared of 0.67. Right: that decoded prior plotted against when the step's answer finally settles, coloured by the action eventually taken — the prior's direction is uninformative about settling time even though its strength is not.](figures/conjectures/c10_prompt_prior.png)

**Remark (what the ten tests changed).** Three conjectures survived (C8, C9, and part (a)
of C10); three were refuted outright (C2, C6, C7); two were replaced by sharper claims than
they proposed (C4's sign-reversing interaction, C5's Pareto frontier); one is partial (C1);
one is unresolved for want of power (C3). Two structural lessons recur.

First, **the belief is easy to read and hard to write** (Figure 22b). Against a held-out
decoding Spearman of $0.7715$ — comfortably above both the fair positional bar ($0.3710$)
and the length-oracle baseline ($0.6435$) — stand four independent intervention families,
each reported at its single best condition:

| intervention | changed the answer | changed it to the intended answer |
|---|---|---|
| rank-1 steering at $2\times$ residual RMS (Prop. 3.5) | $0.036$ | $0.036$ (random control: $0.036$) |
| delete a whole attention or MLP sub-block (C2) | $0.292$ | no intended target |
| transplant the entire 12-token window (C6) | $0.250$ | $0.208$ |
| instruct the model to ignore the reasoning (C7) | $0.058$ | $0.010$ |

The contrast that matters is between the two columns. Interventions can *damage* the
belief — deleting a sub-block perturbs the answer a quarter of the time — but they cannot
*direct* it: the only family that moves the answer toward a specified target does so in a
fifth of cases, and it is the one that overwrites all $12 \times 2880$ numbers the probe
reads. Rank-1 steering is indistinguishable from its random-direction control at every
strength tested. A quantity this legible and this immovable is best read as redundantly
encoded and continuously reconstructed from context, not as a value stored at a site.

Second, **strength and content are separable everywhere we look** (Figure 23): in the layer
geometry (Theorem 3), in what a content decoder fails to deliver (C1), in what transfers to
an entirely different task (C8), and in the prior before a single reasoning token exists
(C10). Four independent measurements, no shared methodology.

![**Figure 22b.** Left: how well the model's confidence in its own next action can be predicted from its internal state alone, compared against a baseline that sees only how far through the reasoning it is, and one that also sees the reasoning's total length. Right: how often four different interventions changed the answer at all (orange) versus changed it to the specific answer intended (blue), each shown at its most effective setting. The gap between the two columns is the finding: the belief can be damaged but not directed.](figures/conjectures/synthesis_read_write_asymmetry.png)

![**Figure 23.** Two measurements of the same dissociation, neither relying on the layer geometry of Theorem 3. Left: how much a decoder trained to recover *which* action the model favours knows about *how strongly* it favours it — almost nothing, whether or not it is given a multiplicative gain — against a probe trained directly on confidence. Right: before any reasoning has been generated, how strongly the model already leans predicts when its answer will settle, while which way it leans does not.](figures/conjectures/synthesis_strength_vs_content.png)

# 6. Reproducibility

| Object | Script | Key artifacts |
|---|---|---|
| Faithful labels, stage-03 contract | `scripts/03c_export_extracted_prefix_scores.py` | `outputs/prefix_action_scores_extracted.parquet` |
| Rebase + triangulation | `scripts/08_compare_extracted_vs_stored_behavior.py` | `outputs/behavior_rebase_comparison.{md,csv}` |
| Clocks, hazard, tails | `scripts/09_commitment_boundary_phenomenology.py` | `outputs/commitment_boundary_*.{parquet,md}` |
| Counterfactual scoring | `scripts/10_score_mismatched_prompt_cot.py` | `outputs/followups/mismatched_prompt_cot_*` |
| Anchors | `scripts/11_belief_update_anchors.py` | `outputs/belief_update_anchors.*` |
| 5-fold probe | `scripts/conceptual_belief_space/77_...py` | `.../fold0_20260721/summary.csv`, `oof_predictions.parquet` |
| Matched per-layer curves, geometry, removals | `scripts/conceptual_belief_space/79_layerwise_dissociation.py` | `.../layerwise_dissociation/*.csv` |
| Dual-head occlusion parity | `scripts/conceptual_belief_space/80_occlusion_parity.py` | `.../occlusion_parity.csv`, `parity_split{1..4}/` |
| Three clocks vs probe | `scripts/conceptual_belief_space/81_three_clocks_lead_lag.py` | `.../three_clocks*/report.md` |
| Steering | `scripts/conceptual_belief_space/82_layer_steering.py` | `.../layer_steering/steering_*` |
| C1 gain–direction factorization | `scripts/conceptual_belief_space/83_conj_c1_gain_direction.py` | `outputs/conjectures/c1_gain_direction/` |
| C2 sub-block ablation | `scripts/conceptual_belief_space/87_conj_c2_subblock_ablation.py` | `outputs/conjectures/c2_subblock/` |
| C3 anchor resampling | `scripts/15_conj_c3_anchor_resampling.py` | `outputs/conjectures/c3_anchors/` |
| C4 hazard screening | `scripts/12_conj_c4_hazard_markov.py` | `outputs/conjectures/c4_hazard/` |
| C5 tail and early exit | `scripts/13_conj_c5_tail_verification.py` | `outputs/conjectures/c5_tail/` |
| C6 transplant ladder | `scripts/conceptual_belief_space/86_conj_c6_patching.py` | `outputs/conjectures/c6_patching/` |
| C7 disavowal | `scripts/14_conj_c7_disavowal.py` | `outputs/conjectures/c7_disavowal/` |
| C8 confidence transfer | `scripts/conceptual_belief_space/88_conj_c8_confidence_transfer.py` | `outputs/conjectures/c8_transfer/` |
| C9 clock noise calibration | `scripts/conceptual_belief_space/84_conj_c9_clock_noise.py` | `outputs/conjectures/c9_clock_noise/` |
| C10 prompt prior | `scripts/conceptual_belief_space/85_conj_c10_prompt_prior.py` | `outputs/conjectures/c10_prompt_prior/` |
| Synthesis figures (22b, 23; redrawn 14) | `scripts/conceptual_belief_space/89_synthesis_figures.py` | `figures/conjectures/synthesis_*.png`, `c2_subblock_paired.png` |

All runs are logged with commands and dates in `RUN_HISTORY.md` (entries 2026-07-21
through 2026-07-24). Data reads require pyarrow $\ge 24$ (`interp/.venv/bin/python`).
Random seeds: 42 throughout; the activation projection is content-addressed
(SHA-256-pinned).
