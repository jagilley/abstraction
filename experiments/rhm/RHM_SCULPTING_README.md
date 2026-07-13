# RHM Sculpting & planning in latents: making RHM a control task that *needs* lookahead

**Status**: **WIP.** Stage 2 (latent-planning attempt — instructive dead-end), Stage 3a (learned beam captures the prize — clean positive), Stage 3b (planning in latents — latent ≈ token at ~8× lower cost on the *clean* channel), and **Stage 3c/3d (latent *beats* token once the token channel is lossy — partial observability + stochastic dynamics — clean positives)** all done, single seed. **Stage 4 (belief-quality axis — a deeper belief makes the block FM a better one-step ranker, monotone across 4 beliefs; a *non-privileged* masked-infilling belief captures it and is the best latent-planner substrate, beating even the privileged oracle; verified genuinely non-privileged by a fully-agnostic span-masking ablation)** done, single seed.
**Stage 5 (internalization — closing the planning loop onto the *belief*: a grounded differentiable planner reshapes the controller so its own latent is plannable → makes the belief FM-rollable + moves/expands the belief frontier and *flips* the clean-channel latent−token gap positive, where an endogenous "be-predictable" control **caps** — grounding is the pivot; and it *subsumes* the Stage-4 belief-quality lever on the plannability axis)** done, single seed.
**Date**: 2026-07-13
**Scripts**: [`rhm_latent_planner.py`](rhm_latent_planner.py) (Stage 2: faithful value + cerebellar FM), [`rhm_sculpt_precheck.py`](rhm_sculpt_precheck.py) + [`sculpting_control_task.md`](sculpting_control_task.md) (the perfect-simulator pre-check that validates the task), [`rhm_sculpt_planner.py`](rhm_sculpt_planner.py) (Stage 3a: the learned token-space beam), [`rhm_sculpt_latent.py`](rhm_sculpt_latent.py) (Stage 3b: the latent-space beam, head-to-head), [`rhm_sculpt_latent_po.py`](rhm_sculpt_latent_po.py) (Stage 3c: partial observability), [`rhm_sculpt_latent_stoch.py`](rhm_sculpt_latent_stoch.py) (Stage 3d: stochastic dynamics), [`rhm_sculpt_deepbelief.py`](rhm_sculpt_deepbelief.py) (Stage 4 gate: parser/oracle/data2vec/mlm belief conditions on the latent beam), [`rhm_sculpt_data2vec.py`](rhm_sculpt_data2vec.py) (Stage 4 isolation: non-privileged belief depth probed vs ground truth + anti-collapse), [`rhm_sculpt_internalize.py`](rhm_sculpt_internalize.py) (Stage 5: the internalization ladder + belief×internalization 2×2).
**Parent**: [RHM_EDIT_CONTROL_README.md](RHM_EDIT_CONTROL_README.md) (Parts 1–2: editing is plannable-in-belief but gameable off-manifold; a generator gives on-manifold moves + a cerebellar veto → true-success 0.006 → 0.65). The origin idea (plan in the model's own latents, cortex proposes / cerebellum forecasts) is Jasper's; see also [ideas/self_model_needs_a_loop.md](../../ideas/self_model_needs_a_loop.md).

---

## The question

Part 2 left a controller that reaches ~0.65 true success by greedy, token-space planning (materialize each move as tokens, re-encode, judge — the "CoT-like" regime where the model never plans in its own latents). The target payoff: **plan in latents** — a cerebellar forward model rolls the belief forward without decoding, so lookahead is cheap and carries the structured divergence that re-tokenizing discards. Does latent-space planning beat token-space? This is the RHM instance of the LLM "chain-of-thought never sees its own latents" gap.

Getting there took a **detour that turned out to be the main lesson**: the task we were on could not reward lookahead at all.

---

## Stage 2 — the latent-planning attempt, and why it kept failing (`rhm_latent_planner.py`)

We built the two learned pieces a latent planner needs and hit two walls, both instructive.

1. **A faithful *verifier* is honest but too sparse to plan with.** First value: "is this a valid r\* config?" (a true-parse head, ungameable — it says INVALID on 99% of the off-grammar starts, exactly where the controller's root head was gamed). But it is *silent* on the ~99% of half-edited configs the planner visits — a treasure detector that only beeps *on* the treasure. Greedy barely limps (0.27); **multi-step lookahead outright collapses (0.02)** — its internal rollout steers by the silent signal and wanders. *Faithfulness ≠ plannability.*

2. **A learned Monte-Carlo *value* fixes density — the value idea works.** The fix (Jasper's framing): don't fake density, *learn* it. Roll a cheap behaviour policy from many starts, label every visited state by whether that rollout eventually reached a true r\* config, regress a compact `V(z, r*)`. This is dense — the sparse reward smeared backward over experience — and it showed cleanly: mean predicted success falls monotonically with corruption distance (`c=0:0.70 → c=4:0.25`). The value function is real.

3. **But lookahead *still* collapsed — because the task was greedy-decomposable.** With the dense value, greedy (0.27) still beat depth-3 lookahead (0.02). The tell: the **ground-truth greedy oracle hit 1.000.** A one-step planner with a perfect ranker solves the corrupt-repair task *perfectly* — the corrupted blocks are essentially independent fixes, so there is **no sequential dependency for lookahead to exploit**; multi-step planning can only add noise. We had been trying to demonstrate a lookahead payoff on a task with no lookahead in it.

**The correction (important, and humbling).** My earlier "the task is greedy-solvable" reading was itself produced by a **witness-targeted oracle**: grading progress against the *specific* clean derivation we corrupted decomposes into independent per-block fixes and hides all coordination. The right ceiling is the **full-horizon DP optimum** (minimum edits to *any* valid r\* derivation), and the right baseline a reflex that must *choose* a globally-consistent derivation. Under that lens the task has real coordination — quantified next.

---

## The sculpting pre-check — the task *does* have a lookahead prize (`rhm_sculpt_precheck.py`, [full note](sculpting_control_task.md))

Pure combinatorics on the known grammar (no learning), **distinct (ambiguous) rules**, success = "r\* in the root's possible-set":

- **PLANNER** = exact DP giving `d*` (min token edits to any valid r\* derivation); solves 100% within budget by construction.
- **REFLEXES** = a weak parse-consistency hill-climber and a strong r\*-aware 1-level-lookahead editor — the best a myopic policy does.
- **Prize** = fraction of solvable instances the reflex fails.

At `L=4, m=2, c=3, budget=6` (our learned setting): **DP 1.00 | strong reflex 0.585 | weak reflex 0.242** (prize 0.42 / 0.76). The prize **scales with depth** (L=5: strong-reflex prize 0.63) and with **tightness** (`m`↓), and **collapses at m=4** (loose coupling → local repair chains to r\* → back to act ≈ plan). A clean dose-response confirming the coordination is real and knob-controlled — a property of the *task*, established before any learning.

---

## Stage 3a — a learned beam captures the prize (`rhm_sculpt_planner.py`)

Same machinery as Part 2 (generator on-manifold moves + a learned MC value), but a proper **beam** planner instead of a wander-prone greedy rollout: keep the top-`W` move sequences, decode to commit. **Beam width 1 = the myopic reflex; wider = coordination.** Grading by possible-set success; DP optimum = 1.0.

| planner (fully learned) | success | | reference | |
|---|---|---|---|---|
| beam width 1 (greedy reflex) | **0.291** | | weak hand-coded reflex | 0.242 |
| beam width 4 | 0.401 | | strong hand-coded reflex | 0.585 |
| beam width 16 | 0.506 | | DP optimum | 1.000 |
| **beam width 64** | **0.578** | | | |

**Perfectly monotone; the learned beam-64 matches the strong hand-coded reflex (0.578 ≈ 0.585)** with no hand-coding — a learned value + beam over on-manifold moves. Greedy (0.29) sits at the *weak* reflex, as a myopic policy should. **Lookahead helps — steeply and smoothly — the exact opposite of Stage 2's collapse.** The only thing that changed is the *task*: it now genuinely requires coordination.

**Headline lesson: the entire Stage-2 struggle was a task problem, not a method problem.** The identical generator + value + planner machinery that looked broken on the greedy-decomposable task works cleanly the instant the task rewards coordination.

**Limits + the learned coupling dose-response.** Pushing beam width and move budget, the learned planner **passes the strong reflex** (beam-256, budget-10: **0.611 > 0.585**), climbing toward the DP optimum — value quality (behaviour-policy success 0.34 caps the value) and search are the remaining levers. And the beam benefit tracks coupling, the learned mirror of the pre-check's negative control: greedy→beam-64 gain is **+0.29 (m=2) → +0.26 (m=3) → +0.13 (m=4)** — roughly halving as coupling loosens (m=4's learned greedy is weak so it does not fully collapse, but the prize shrinks as predicted).

---

## Stage 3b — planning in latents ≈ token-space, at ~8× lower cost (`rhm_sculpt_latent.py`)

The Stage-3a beam is token-space ("CoT-like"): it materializes every branch and re-encodes it. Stage 3b replaces the rollout with a **cerebellar FM over a richer per-block (non-pooled) latent**, keeping the value **identical** (it reads the pooled belief `b = mean_block(z)`, which equals the controller's own pooled state) so the head-to-head isolates exactly *FM-in-latents vs materialize-and-re-encode*.

**Open-loop latent rollout collapses** (FM drift over 6 un-grounded steps — the Stage-2 failure). The fix is Dreamer-style **re-grounding**: the FM only ever predicts *one* step from a *true* latent (used to rank candidates cheaply), and only the kept beam tips are materialized + re-encoded. So the FM never compounds error, and it saves the expensive op — the latent beam materializes `W` tips/step vs the token beam's `W·n_regions`.

Head-to-head (m=2, c=3; block FM delta-cos 0.49, per-move value-rank-corr 0.52):

| beam width | token beam | latent beam | latent / token |
|---|---|---|---|
| 1 (greedy) | 0.289 | 0.085 | 0.29 |
| 16 | 0.463 | 0.346 | 0.75 |
| 64 | 0.531 | 0.444 | 0.84 |
| **256** | 0.544 | **0.499** | **0.92** |

**Planning in latents works and closely approaches token-space** — 92% of the token beam at width 256, at ~8× fewer materializations/step. The mechanism is wide-beam convergence: latent-greedy is poor (the FM's top-1 move pick is only 36% right), but the beam keeps the true-best move among the FM's top-ranked candidates and re-grounding + the value recovers it, so the gap closes with width. **The honest reading: latents buy *cheaper* planning here, not *better* — the FM is a lossy-but-serviceable surrogate; no evidence it beats tokens by carrying divergence tokens can't.**

---

## Stage 3c/3d — when latents actually *beat* tokens: a lossy token channel (`rhm_sculpt_latent_po.py`, `rhm_sculpt_latent_stoch.py`)

Stage 3b's verdict — "an efficient surrogate, not superior" — was **honest but scoped to one corner**: the fully-observed, *deterministic* task, where the token sequence is a **sufficient statistic** (the parse is a deterministic function of the tokens, so re-encoding a materialized config is lossless, and a latent FM can only *approximate* that lossless map — no "structured divergence tokens can't carry" when tokens carry everything). Reasoning about *why* gave a falsifiable prediction: latent planning should **beat** token-space exactly when the token channel stops being sufficient — when the information the token beam must re-read is **missing** (partial observability) or **noisy** (stochastic dynamics). Both confirm, cleanly.

**Shared discipline (identical to Stage 3b, one knob added):** every instrument (controller, generator, MC value, block FM) is trained ONCE on the clean deterministic world and **frozen** — "an agent that learned the world, now deployed into a lossier one." The ONLY controlled variable is the channel knob, injected at *plan time*. Same eval instances, same value, same moves. Both variants reduce to the Stage-3b beams **exactly at knob=0** (asserted `|ref − variant| < 1e-6` before each sweep). Diagnostics shared by both (full run): block-FM `delta_cos=0.49`, `value_top1_agree=0.36`, `value_rank_corr=0.52`.

![Latent-vs-token crossover under two lossy channels](rhm_sculpt_latent_crossover.png)

*Figure ([`rhm_sculpt_latent_crossover.py`](rhm_sculpt_latent_crossover.py)): top row — gap (latent − token) vs the channel knob, one line per beam width; both channels cross from negative (token wins) to positive (latent wins), and the crossover is width-gated. Bottom row — absolute success at beam 256: the token beam collapses while the latent beam holds.*

### Missing information — partial observability (`rhm_sculpt_latent_po.py`)

A flickering per-block sensor occludes a fraction `p` of blocks at plan time. The token beam's only channel to a candidate's value is to **re-observe** it through the lossy sensor; the latent beam **carries a per-block belief** and rolls it forward with the FM (Kalman-style filter: observed block → re-encode, occluded block → keep the FM's prediction). Masking degrades the *evaluation* channel only (candidate construction held full-obs) to isolate the sufficiency claim; the flickering (vs permanent) occlusion is deliberate — permanent occlusion hides blocks from everyone equally (no belief to carry, guaranteed null).

Gap = latent − token success (+ = latent wins); anchor exact at p=0 (token 0.544 / latent 0.499 = Stage 3b):

| beam | p=0 | p=.25 | p=.5 | p=.75 | p=.9 |
|---|---|---|---|---|---|
| 64  | −.087 | −.023 | **+.021** | **+.030** | **+.029** |
| 256 | −.045 | **+.021** | **+.037** | **+.043** | **+.040** |

The gap rises monotonically and crosses zero. Mechanism (width-256 *levels*): the token beam **collapses** under occlusion (0.544 → 0.460 → 0.297 → 0.149 → 0.062) while the latent beam **decays gently** (0.499 → 0.480 → 0.334 → 0.192 → 0.102) — by p=0.9 it is **1.6× the token beam**. Latent planning carries belief the token channel destroys.

### Noisy information — stochastic dynamics (`rhm_sculpt_latent_stoch.py`)

A **slippery actuator**: an edit lands as intended with prob 1−q, else slips to a uniformly random feature. Now a move's consequence is a *distribution*, not a point. The token beam materializes ONE realization and re-encodes it — a single **noisy sample** of E[value | move] (and, like CoT, commits the sample it drew); the latent beam ranks by the FM's **stable** prediction from the true latent, reusing the *same deterministic FM* (its intended-outcome prediction is a sound, low-variance ranking, since the slip is exogenous and uncontrollable — Design 1). This is the exact **mirror of the active-query null**: there a mean-Δ FM was useless because the payoff lived in the *variance* the mean discards; here the payoff lives in the *mean* and the token beam is forced to sample it once — so the *same* mean-predicting instrument flips from "not superior" to superior, with **zero retraining**.

Gap = latent − token; anchor exact at q=0:

| beam | q=0 | q=.1 | q=.25 | q=.5 | q=.75 |
|---|---|---|---|---|---|
| 64  | −.087 | **+.055** | **+.036** | **+.048** | **+.024** |
| 256 | −.045 | **+.021** | **+.075** | **+.050** | **+.034** |

Flips at q=0.1; peak **+0.075** — a *larger* advantage than occlusion. The gap is **humped** (peaks at q=0.25, declines by q=0.75): at extreme noise even good selection cannot control the world, so both collapse — where PO instead *plateaus* (the ~10% still-visible blocks keep belief-carry useful). (A "Design 2" that retrains the FM on slippery targets — so it predicts the true k-dependent expectation E[Δz|z,k] rather than the intended outcome — can only help the latent beam further; Design 1 is the cheaper, instrument-reusing first cut.)

### The unifying signature — latent superiority is width-gated

In BOTH channels the latent advantage is **width-gated**: beam-1 (greedy) latent *never* wins (gap stays negative at every knob value), and the crossover moves to *lower* knob values as beam widens (256 crosses first, 64 next, 16 marginal, 1 never). Cause: the FM's weak top-1 move pick (`value_top1_agree = 0.36`). At width 1 the latent beam commits to often-wrong single moves and no belief-carry or stable-ranking rescues a bad greedy commitment; it takes beam ≥ 64 for the true-best move to survive *in the beam*, where the latent advantage can express. This is the same FM-top-1 bottleneck that made latent-greedy poor on the *clean* channel too (Stage 3b: latent-greedy 0.085 ≪ token-greedy 0.289) — one mechanism, visible across all three regimes.

### The learning (a scoped negative that pointed straight to the positive)

Stage 3b's "efficient surrogate, not superior" was **correct for the clean channel but would mislead if read as a property of latent planning per se.** Surrogate-vs-superior is not a fixed fact about latents — it is governed by whether the token channel is a **sufficient statistic**:

- **Clean channel** (full-obs, deterministic): tokens sufficient → latent is a cheaper *surrogate* (gap −0.045). *This is the special case.*
- **Lossy channel** (missing or noisy info): tokens insufficient → latent is *superior* (gap up to +0.075), because it carries belief / amortizes the expectation that a token round-trip destroys. *This is the common case* — real-world DGPs are partially observed and stochastic.

The honest negative, reasoned about rather than shrugged off, *predicted* the two knobs where it flips — which is the whole point of understanding a result before updating on it.

## Stage 4 — the belief is the bottleneck: a deeper (and *non-privileged*) belief lifts latent planning (`rhm_sculpt_deepbelief.py`, `rhm_sculpt_data2vec.py`)

A new axis, orthogonal to Stage 3c/3d's *channel* axis: not "when do latents beat tokens?" but "**what makes the latent beam good in the first place?**" Stage 3b's latent beam lags token on the clean channel because the block FM is a weak one-step ranker (`value_top1_agree = 0.36`, `delta_cos = 0.49`). The cross-experiment hypothesis (from the [RHM_LATENT_LOOP](RHM_LATENT_LOOP_README.md) ↔ sculpting discussion): the FM is weak because the controller is trained as a *parser* (root-CE only), so its per-block latent leaves the deep parse structure that governs an edit's consequence **tangled/unrecruited** — the same shallow-frontier failure RHM_LATENT_LOOP found for token targets. Prediction: a **deeper (DGP-aligned) belief → a more faithful FM → better latent planning**.

**Design — single controlled variable = the controller's training objective.** Every belief gets the same root-CE (so each keeps a functional head for the planner) + the same masking schedule; they differ ONLY in the aux term. Everything downstream — generator, eval instances, value collection, block FM, beams, all hyperparameters — is shared/frozen, so the encoder is the only thing that varies across conditions.
- **parser** (floor): root CE only (= Stage 3b's controller).
- **oracle** (privileged ceiling): + per-block **ground-truth ancestor-label** CE (uses the latent tree — a *diagnostic* of "is there headroom from a deeper belief," not a method).
- **data2vec** (non-privileged): + EMA self-distillation (predict the teacher's masked-position representation).
- **mlm** (non-privileged): + masked-infilling (predict the true masked leaf tokens from a masked view). Stage 2 uses subtree masking; a fully tree-agnostic span-masking variant recovers most of the depth (Stage 1 ablation), so the objective does **not** require knowing the DGP topology.

### Stage 1 — the non-privileged belief, validated *in isolation* (`rhm_sculpt_data2vec.py`)

Before any planning, probe each belief against **ground truth** — per-level linear recovery of a block's ancestor feature (only RHM lets you do this) — plus an anti-collapse check (participation ratio `PR` of the per-block latent; collapse → PR≈1, probe≈chance 1/v=0.125). "Working" = recruits deep structure, no collapse.

| belief | d2 | d3 | d4 | PR | d3 gap closed vs oracle |
|---|---|---|---|---|---|
| parser | 0.705 | 0.605 | 0.744 | 6.7 | — |
| data2vec | 0.743 | 0.688 | 0.752 | 7.2 | +23% |
| **mlm** | 0.781 | 0.762 | 0.852 | 7.9 | **+44%** |
| oracle | 0.939 | 0.963 | 0.984 | 12.7 | 100% |

- **data2vec barely recruits depth** (+16–23% of the gap). Principled: its EMA-teacher target is only as deep as root-CE, which stalls shallow — self-distillation *consolidates* the teacher's existing depth but cannot *invent* depth beyond it. This is RHM_LATENT_LOOP's core thesis reappearing from the planning side (endogenous/self-referential objectives don't move the frontier).
- **mlm recruits real depth** (+32–45%) with **no collapse** (PR 7.9 > 1, probes ≫ chance), because its target is the *true tokens* (not a teacher-capped representation), so predicting a masked span forces representing the ancestor. **Non-privileged-masking check (does subtree-aligned masking smuggle DGP topology?).** `_subtree_mask` masks spans whose sizes/boundaries are aligned to the true tree — it uses `s`, `L`, and constituent alignment, which a real DGP does *not* hand you. Re-running mlm with **fully agnostic contiguous-span masking** (`mask_mode="span"`: random length/position at leaf granularity, SpanBERT-style, *no* `s`/`L`/alignment) still recruits the large majority of the depth: d3 **0.722** vs subtree 0.762 (both ≫ parser 0.605 — ~75% of the subtree recruitment; d4 0.834 ≈ 0.852), no collapse (PR 7.6). So the alignment is a **modest booster** (d3 gap-closing 33% → 44%), **not** the mechanism — the mechanism is contiguous masking (a universal locality prior) forcing long-range inference against a grounded token target. **mlm is genuinely non-privileged.** Neither variant reaches the oracle: ~33–44%-of-ceiling deep. *(Stage 2 used the subtree variant; a span-variant planning confirmation is the remaining check — depth→planning is monotone, so the planning win is expected to survive.)*

### Stage 2 — plug each belief into the latent beam (`rhm_sculpt_deepbelief.py`)

One controlled 4-way run (parser/data2vec/mlm/oracle through the identical frozen pipeline):

| belief | d3 | FM top1 | FM rank-corr | token w256 | latent w256 | gap w256 |
|---|---|---|---|---|---|---|
| parser | 0.61 | 0.357 | 0.527 | 0.485 | 0.443 | −0.042 |
| data2vec | 0.69 | 0.378 | 0.532 | 0.497 | 0.466 | −0.031 |
| **mlm** | 0.76 | 0.411 | 0.576 | **0.550** | **0.535** | **−0.015** |
| oracle | 0.96 | 0.435 | 0.626 | 0.520 | 0.493 | −0.026 |

1. **Depth → FM-fidelity is a monotone dose-response.** Belief depth (d3) orders parser < data2vec < mlm < oracle, and the FM's one-step ranking follows the *exact same order* (top1 0.357→0.378→0.411→0.435; rank-corr 0.527→0.532→0.576→0.626). "Deeper belief → more faithful FM rollout" holds across four points — the mechanism as a dose-response, not a single contrast.
2. **The non-privileged mlm belief is the *best* planner substrate — beating even the privileged oracle.** mlm gives the highest beams of all four (latent 0.535 vs oracle 0.493 vs parser 0.443; token 0.550 vs 0.520 vs 0.485) and the *smallest* latent−token gap (−0.015 — latent nearly catches token). Against the original Stage-3b pipeline (parser belief, latent 0.499, gap −0.045), the mlm belief lifts latent planning to **0.535** and shrinks the gap to **−0.015** — a concrete, no-privilege improvement.
3. **The latent-*specific* advantage (Δlatent > Δtoken) is real and most skewed for the deepest belief.** Oracle helps the latent beam far more than the token beam (Δlatent/Δtoken = +0.069/+0.018 at w16) — pure FM-rollability, Stage 0's mechanism. mlm helps latent slightly more than token but lifts *both* a lot. So two distinguishable effects: **ancestor-probe depth (oracle) specifically buys latent-rollability; the grounded infilling objective (mlm) buys broad pipeline quality** — and the latter wins on absolute planning.

**The mlm > oracle puzzle.** mlm beats oracle on planning *despite less probe-depth and a slightly worse behaviour-policy/value* (value-buffer success mlm 0.328 < oracle 0.337) — so it is **not** value quality. Working hypothesis (untested): the oracle aux optimizes linear ancestor-decodability on *clean* sequences (what the probe measures), whereas the planner runs on *corrupted/off-manifold* configs; mlm's token-grounded objective sees masked/degraded inputs in training, plausibly yielding a belief geometry that **transfers better off-manifold** — exactly where the FM rolls.

**Caveats (Stage 4).** Single seed. `mlm > oracle` is modest (~0.03) and single-seed — the robust claim is "mlm ≈ oracle or better, both ≫ parser." The parser beams here (0.485/0.443) run ~0.06 below the Stage-3b published table (different training-pool seeding), so only the *within-run* belief ordering is load-bearing, not cross-run absolutes. The mlm>oracle mechanism is a hypothesis, not established.

## Stage 5 — Internalization: closing the planning loop onto the *belief* (`rhm_sculpt_internalize.py`)

A new axis, orthogonal to both the *channel* axis (3c/3d) and the *belief-quality* axis (Stage 4). Everything through Stage 4 trains each instrument once and **freezes** it, then bolts an external beam on top — the exact analog of the a2a reaching arc's *decoupled* apparatus (operator frozen, FM on frozen transitions, external `argmax` planner). [`a2a_forward/REACHING_INTERNAL`](../a2a_forward/REACHING_INTERNAL_README.md) took the next step there: make the forecast **endogenous** — co-train it, use it from inside the loop — and the operator reorganized to be dramatically more *plannable* (a fresh, independent external planner jumped +0.55 → +1.00 on the co-trained operator). This stage is that step for sculpting: **let the latent-planning objective reshape the controller's per-block belief `z`** — the thing the block FM rolls one step of, and the thing the beam searches.

**Design — single controlled variable = how much the planning loop touches the belief during training.** A nested 3-rung ladder on the *same* parser belief, matched init + matched controller-update budget; everything downstream (fresh MC value, fresh block FM, both beams, the depth probe) is **identical** across rungs, so the downstream **fresh-FM** latent beam is literally REACHING_INTERNAL's "fresh independent planner on the (maybe-internalized) operator" transferability test.
- **frozen**: root-CE only (= the Stage-3b parser). The loop never touches the belief.
- **fm_cotrain**: + the a2a asymmetric FM local-loss into the belief (FM learns the real *detached* dynamics; the controller is *pulled* toward the detached FM prediction). Endogenous "be-predictable" pressure, non-collapsing by construction.
- **planner**: + a differentiable **grounded** planner `logits_k = value((z + FM(z,k)).mean)/τ` imitation-trained against the **DP best move** `k* = argmin_k d*(regenerate(x,k))` (precomputed, controller-independent). The `int_plan` port — grounded plannability gradient flows into controller + FM + value.

Nested (`frozen ⊂ fm_cotrain ⊂ planner`), so `fm_cotrain − frozen` isolates *endogenous predictability* pressure and `planner − fm_cotrain` isolates what *grounding* adds on top.

| rung | d3 | d4 | PR | fresh-FM top1 | rank_corr | latent w256 | token w256 | gap w256 |
|---|---|---|---|---|---|---|---|---|
| frozen | 0.605 | 0.744 | 6.7 | 0.357 | 0.527 | 0.443 | 0.485 | −0.042 |
| fm_cotrain (endogenous) | 0.576 | 0.671 | 6.8 | 0.304 | 0.466 | 0.576 | 0.601 | −0.024 |
| **planner (grounded)** | **0.692** | **0.882** | **17.0** | **0.521** | **0.736** | **0.591** | 0.582 | **+0.009** |

Two results, and the pivot between them:

1. **Grounded internalization makes the belief plannable AND moves/expands the belief frontier.** Fresh-FM one-step ranking `top1 0.357→0.521`, `rank_corr 0.527→0.736` — attacking the exact bottleneck Stage 4 pinned (the FM's weak 0.36 top-1). The clean-channel latent−token gap **flips positive at w256 (−0.042 → +0.009)** and closes at w64 (−0.005), with **Δlatent > Δtoken at every width** (+0.222 vs +0.120 at greedy; +0.147 vs +0.097 at w256 — the latent-*specific* tell). And the belief deepens (d3 +0.087, d4 +0.138) and **expands** (PR 6.7 → 17.0 — recruiting idle belief dimensions). This is a **third route to latent-beats-token on the clean channel** — Stage 3b said only a *lossy* channel could do that (3c/3d); a plannability-shaped belief does it too, with tokens fully sufficient.
2. **The endogenous control caps — grounding is the pivot.** `fm_cotrain` (predictability pressure *without* grounding) does **not** deepen the belief (d3/d4 flat-to-down, PR flat 6.8) and actually *lowers* the transferable fresh-FM plannability (top1 0.357 → 0.304). Its beams *do* rise, but that is a **value/behaviour** artifact (value-buffer success 0.315 → 0.439), not belief plannability: both channels rise about equally, so its gap barely moves (−0.042 → −0.024). Only the *grounded* planner term moves the belief itself. Since `fm_cotrain ⊂ planner` differ **only** by that term, **grounding causes the reorganization, not merely closing the loop.**

**This is [RHM_LATENT_LOOP](RHM_LATENT_LOOP_README.md)'s frontier-moving-vs-capped dichotomy reproduced on the *planning* side.** There, endogenous/self-referential targets (data2vec's EMA teacher) *cap* while grounded targets (mlm's true tokens, oracle latents) *move* the frontier. The planning loop is structurally endogenous (the FM predicts the controller's own latents; the value is bootstrapped from its own rollouts), so the naive prediction is that it caps — and the endogenous `fm_cotrain` rung does. But the sculpting reward "reach a true r\* config" is **grounded by construction**, and the `planner` rung's DP-best-move target inherits that grounding — so it *moves* the frontier, exactly as mlm/oracle did on the prediction side.

**The dissociation (vs reaching).** Reaching found plannability rose while full-state veridicality *fell* (concentrate the value-relevant variable, let the rest get more complex). Here grounded internalization raised **both** (delta_cos +0.122 *and* top1 +0.163) alongside the big PR expansion — a genuine "frontier moved / new dimensions recruited" signature rather than reaching's "reorganize the same dimensions." The gain is still skewed value-relevant (Δtop1/Δrank_corr > Δdelta_cos), just *additive* to veridicality rather than orthogonal to it.

### Stage 5b — composing with the belief-quality axis: internalization *subsumes* Stage 4 on plannability (`sculpt_compose`)

The 2×2 {parser, mlm} belief × {frozen, planner} internalization, identical downstream. `parser_planner` reproduced Stage 5's `planner` rung to three digits (latent 0.591, gap +0.009, top1 0.521 — a cross-draw replication that validates the harness).

| condition | d3 | d4 | PR | FM top1 | latent w256 | token w256 | gap |
|---|---|---|---|---|---|---|---|
| parser_frozen | 0.605 | 0.744 | 6.7 | 0.371 | 0.446 | 0.478 | −0.031 |
| parser_planner | 0.692 | 0.882 | 17.0 | 0.521 | 0.591 | 0.582 | +0.009 |
| mlm_frozen | 0.762 | 0.852 | 7.9 | 0.411 | 0.535 | 0.550 | −0.015 |
| **mlm_planner** | 0.775 | 0.922 | 18.2 | 0.533 | **0.632** | 0.636 | −0.004 |

**Grounded internalization subsumes the static belief-quality lever on the plannability axis.** Decisive cell `mlm_planner − parser_planner`: Δtop1 **+0.013**, Δrank_corr **−0.004**, gap +0.009 → −0.004 — on FM-rollability and the clean-channel gap the two are **equal**. A plain parser belief + internalization already captures essentially all the *plannability* the hand-designed mlm objective provided. mlm_planner is the best *absolute* planner (latent 0.632), so mlm still adds something — but that increment is **not latent-specific** (token rises *more* than latent, +0.054 vs +0.041, and the gap actually dips), i.e. a general pipeline lift from mlm's extra depth, not more plannability.

**Mechanism dissociation — the two levers do different things, and only one produces plannability.** mlm buys **probe-depth** (d3 +0.157 over parser) but **not rollability** (ΔPR only +1.3, Δtop1 only +0.04 — linear-ancestor decodability rises, the FM still can't roll it). Grounded internalization buys **rollability** (ΔPR **+10**, ~2.6× the belief's effective dimension; Δtop1 **+0.15**) *identically on both bases* (mlm base: ΔPR +10.3, Δtop1 +0.12). The PR expansion — the RHM_LATENT_LOOP "grounded target feeds gradient to idle capacity" signature — is **entirely** an internalization effect; mlm does not expand PR. So **mlm makes the belief linearly deeper; internalization makes it *rollable*** (and deeper too). For latent planning, rollability is what matters — hence subsumption. The sharper form of the frontier-moving story: even Stage-4's best *static grounded* objective (mlm) caps on the plannability axis; only the *grounded planning loop* moves it.

**Caveats (Stage 5).** Single rule-seed (parser_planner replicated across two independent pipeline draws at that seed; magnitudes are directional, not hardened). The subsumption claim is on plannability-specific metrics (FM top1/rank_corr, the clean-channel gap); mlm_planner remains the best *absolute* cell. The endogenous `fm_cotrain` beam rise is a value/behaviour lift, disentangled from belief plannability by the gap + fresh-FM metrics but not separately ablated. Belief depth is probed on clean full-obs sequences (same scope as Stage 4).

## What this arc establishes (so far)

1. **Faithfulness ≠ plannability, and density must be *learned*, not faked.** A faithful verifier is too sparse; a Monte-Carlo value (dense-by-experience) is the right instrument (dense distance-to-goal, ungameable).
2. **A task must actually contain lookahead before a planner can demonstrate it.** The corrupt-repair task is greedy-decomposable (greedy oracle = 1.0); sculpting (choose a globally-consistent derivation, DP optimum vs choosing-reflex) is not. This is the reaching-arc lesson (obstacles/maze forced lookahead) reproduced in RHM.
3. **A learned beam captures the coordination prize** — monotone in width, up to and past the strong hand-coded reflex (0.611 with wide beam + budget) — where a greedy rollout collapsed. The task, not the method, gated it.
4. **On the *clean* channel, planning in latents is a viable *efficient surrogate*, not a superior one.** With a richer non-pooled latent + a re-grounded cerebellar FM, a latent beam reaches 92% of the token beam at ~8× fewer materializations. It approaches but does not beat token-space *here*; beam width compensates for the FM's imperfect per-move ranking. The open-loop version collapses — re-grounding (never rolling the FM more than one step from truth) is load-bearing. **But "surrogate, not superior" is channel-specific** — see point 5.
5. **On a *lossy* channel, planning in latents is genuinely *superior*.** Whether tokens are a sufficient statistic decides it. Partial observability (Stage 3c): the latent beam carries belief the sensor drops → gap flips to +0.02–0.04 for p ≥ 0.25. Stochastic dynamics (Stage 3d): the latent beam ranks by a stable FM estimate instead of one noisy sample → gap flips to up to +0.075 for q ≥ 0.1 (the mirror of the query null — payoff in the mean the token beam must sample). Both are width-gated (need beam ≥ 64 to beat the FM's 0.36 top-1). Full-obs-deterministic is the special corner; real DGPs are lossy, so *superior*, not *surrogate*, is the common case.
6. **The belief is a first-class lever on latent planning, and a *non-privileged* deep belief captures it (Stage 4).** The block FM's weak one-step ranking (top-1 0.36) — the bottleneck behind the clean-channel latent lag and all the width-gating — is downstream of belief *depth*: deepen the controller's belief and the FM's fidelity rises monotonically across four beliefs (top-1 0.36 → 0.44 parser→oracle). A privileged oracle-ancestor belief is the ceiling, but a **non-privileged masked-infilling (mlm) belief** recruits ~40% of that depth with no collapse and — plugged into the beam — is the **best latent-planner substrate tested, beating even the privileged oracle** (latent w256 0.535 vs 0.493 vs parser 0.443; gap −0.015 vs −0.045). A cheaper *self-supervised* objective, not a privileged one, is what the FM's rollability needed. data2vec under-delivers (teacher-capped → can't exceed the shallow frontier), independently re-deriving RHM_LATENT_LOOP's endogenous-frontier thesis from the planning side.
7. **Closing the planning loop onto the belief works — and *grounding* is the pivot (Stage 5).** Frozen-checkpoint-plus-bolt-on-beam is the sculpting analog of the reaching arc's *decoupled* apparatus; internalizing the forecast (a differentiable grounded planner backprops into the controller) makes the belief plannable (fresh-FM top1 0.36→0.52, rank_corr 0.53→0.74), moves *and* expands the belief frontier (d3 +0.09, PR 6.7→17), and **flips the clean-channel latent−token gap positive** — a third route to latent-beats-token that doesn't need a lossy channel. An *endogenous* "be-predictable" control caps (belief flat, fresh-FM plannability down); only the grounded planner term moves it — RHM_LATENT_LOOP's frontier-moving-vs-capped dichotomy, now on the planning side, with the DP-best-move reward supplying the grounding. And grounded internalization **subsumes the Stage-4 belief-quality lever on the plannability axis**: parser+internalize ≈ mlm+internalize on FM-rollability and the gap, because mlm buys probe-*depth* while internalization buys *rollability* (the PR expansion is entirely an internalization effect) — and rollability is what latent planning needs.

## What this does NOT establish / caveats

- **Single seed, single setting** (L=4, m=2, c=3); the learned beam *matches* the strong reflex but has not passed it or reached the DP optimum (1.0). Remaining headroom: value-iteration (recollect with the beam as behaviour policy — its 0.34 success rate caps the value), beam width, better moves.
- **Latent planning does not beat token-space *on the clean channel*, and lags at narrow beams** — the FM's per-move ranking (top-1 36%, rank-corr 0.52) is the bottleneck; wide beams hide it. It *does* beat token-space once the channel is lossy (Stage 3c/3d), but the narrow-beam lag persists across all channels (the width-gating). The efficiency claim (~8× fewer materializations) assumes the FM is meaningfully cheaper than generator+controller (it is smaller, but same order).
- **Stage 3c/3d crossover margins are modest (+0.02 to +0.08)** — single seed. The *pattern* is robust (monotone across 4 widths × 5 knob levels × 2 channels, exact knob=0 anchors, and it mirrors both the sufficiency prediction and the query null), but the per-cell magnitudes are small. Two principled, untested ways they should widen: a **belief-aware value** for PO (train the value on filtered beliefs, removing the full-obs-consumer cap) and **Design 2** for the FM under noise (train it on slippery targets).
- **Stage 3c masks the *evaluation* channel only** (candidate construction held full-obs) — the clean isolation of the sufficiency claim, but it does *not* test whether latents also help *propose* moves blind. Stage 3d's stochasticity, by contrast, hits construction (the actuator) directly.
- Distinct (ambiguous) rules make "success" = r\* derivable; at loose coupling (m=4) this is easy (prize collapses) — the sculpting *coordination* effect lives at tight coupling (m=2/3). The 3c/3d *channel* effects are a separate axis and were not re-swept over m.
- **Stage 4 is single-seed**, and `mlm > oracle` is a modest (~0.03) single-seed gap — the load-bearing claim is the *ordering* (parser < data2vec < mlm ≈/> oracle, on both depth-probe→FM-fidelity and planning), not the mlm-beats-oracle margin. Stage-4 belief depth is probed on *clean* full-obs sequences; whether it holds on the off-manifold configs the planner actually visits is untested — and is exactly the `mlm > oracle` hypothesis.

## Next steps

1. ~~**Stage 3b — plan in latents.**~~ *Done*: latent beam reaches 92% of token beam at ~8× lower cost (an efficient surrogate on the clean channel).
2. ~~**Is there *any* regime where latents *beat* tokens rather than approximate them?**~~ *Done* (Stage 3c/3d): **yes — whenever the token channel is lossy** (partial observability, stochastic dynamics). The sufficiency-of-tokens argument predicted exactly these two knobs; the clean-channel "surrogate, not superior" was the special case.
3. **Widen the 3c/3d margins with the two principled upgrades** — a **belief-aware value** (train the MC value on filtered/masked beliefs, not only full-obs ones — removes the full-obs-consumer cap) and **FM Design 2** (retrain the FM on slippery targets so it predicts the true k-dependent expectation). Both should turn a +0.05 into something less equivocal.
4. **The combined channel** — sweep partial-obs × stochastic together (real DGPs have both). Does the latent advantage compound?
5. **A better FM / value-iteration** — a stronger FM (capacity/training, or predicting the *value* directly) should lift the top-1 pick (0.36) and shrink the width-gating; value-iteration (recollect with the beam as behaviour policy, its 0.34 success caps the value) lifts both planners and would push the token beam past the strong reflex toward the DP optimum (also L=5 depth).
6. **The m=4 negative control in the learned setting** (beam should *not* help the *coordination* prize when coupling is loose — the learned analog of the pre-check's control; note the 3c/3d *channel* effects are a separate axis).
7. **Nail the `mlm > oracle` mechanism (Stage 4).** Probe belief depth on *corrupted/off-manifold* configs (not just clean full-obs), and test an mlm+oracle *combined* belief — does grounded off-manifold transfer explain why the non-privileged belief out-plans the privileged one?
8. **Seeds on the Stage-4 ordering** — confirm parser < data2vec < mlm ≈/> oracle before crystallizing a belief.
9. **Belief quality × lossy channel** — does the mlm belief *widen* the latent advantage under partial-obs / stochastic dynamics (Stage 3c/3d) more than on the clean channel? (The two positive axes composed.)
10. *(future work)* **Push the non-privileged belief past ~40%-of-ceiling depth** — more mlm steps / higher weight / mlm+data2vec, or a parser warm-start giving the EMA teacher a real floor. Deferred: mlm already suffices as a planner substrate.
11. **Internalization × lossy channel (Stage 5 × Stage 3c/3d).** Does the plannability-shaped belief *widen* the latent advantage under partial-obs / stochastic dynamics beyond the clean channel? Composes the two big positive axes of the whole arc — grounded internalization already flips the gap on the *clean* channel, so on a lossy channel it should push further.
12. **Seeds on the Stage-5 ordering** — confirm `frozen < fm_cotrain(caps) < planner` and the subsumption `parser_planner ≈ mlm_planner` (plannability axis) across 1–2 rule-seeds before crystallizing. (parser_planner already replicated across two pipeline draws at seed 0.)
13. **Continuous internalization (the "does it compound" question).** Stage 5 is one belief-training pass then freeze. The reaching/RHM_LATENT_LOOP analog is an interleaved loop (collect rollouts with the beam as behaviour policy → update value → update FM → update belief), i.e. value-iteration extended to the belief. Prediction (from RHM_LATENT_LOOP): compounds *while the belief frontier is still climbing toward the task's coordination-plannability ceiling*, then exhausts.
14. **Ablate-to-floor / map-model coexistence (Stage 5).** The current eval measures the belief's transferable plannability with a *fresh* FM (the "map" is available). The complementary reaching diagnostic — does the *co-trained* running loop *depend* on its own forecast (ablate → floor) while a linear readout still recovers a legible plan-map — was deferred; it would confirm map-and-model coexist here as in reaching.

## Reproduce

```bash
cd experiments/
# Perfect-simulator pre-check (validates the lookahead prize; CPU)
modal run rhm/rhm_sculpt_precheck.py::sculpt_precheck --m 2 --depths "4,5" --corrupt-blocks "2,3"
# Stage 3a: learned token-space beam (width 1 = reflex, wider = coordination)
modal run --detach rhm/rhm_sculpt_planner.py::sculpt_planner --m 2 --beam-widths "1,4,16,64"
# Stage 3b: latent-space beam vs token-space beam, head-to-head (clean channel)
modal run --detach rhm/rhm_sculpt_latent.py::sculpt_latent --m 2 --beam-widths "1,16,64,256"
# Stage 3c: latent BEATS token under partial observability (occlusion sweep)
modal run --detach rhm/rhm_sculpt_latent_po.py::sculpt_latent_po --m 2 --beam-widths "1,16,64,256" --p-masks "0.0,0.25,0.5,0.75,0.9"
# Stage 3d: latent BEATS token under stochastic dynamics (slippery-actuator sweep)
modal run --detach rhm/rhm_sculpt_latent_stoch.py::sculpt_latent_stoch --m 2 --beam-widths "1,16,64,256" --slips "0.0,0.1,0.25,0.5,0.75"
# Stage 4 isolation: non-privileged belief depth vs ground truth + anti-collapse (fast, no beams)
modal run --detach rhm/rhm_sculpt_data2vec.py::data2vec_isolation --m 2 --beliefs parser,oracle,data2vec,mlm
# Stage 4 non-privileged-masking ablation: fully-agnostic span masking (no s/L/alignment) still recruits depth
modal run --detach rhm/rhm_sculpt_data2vec.py::data2vec_isolation --m 2 --beliefs mlm --mask-mode span
# Stage 4 gate: belief conditions (parser floor / oracle ceiling / data2vec+mlm non-privileged) on the latent beam
modal run --detach rhm/rhm_sculpt_deepbelief.py::sculpt_deepbelief --m 2 --beliefs parser,oracle,data2vec,mlm
# Stage 2 (the instructive dead-end): faithful value + latent FM on corrupt-repair
modal run --detach rhm/rhm_latent_planner.py::latent_planner --m 2
# Stage 5: internalization ladder (frozen / fm_cotrain endogenous / planner grounded)
modal run --detach rhm/rhm_sculpt_internalize.py::sculpt_internalize --m 2
# Stage 5b: belief-quality × internalization 2×2 (parser/mlm × frozen/planner) — stack vs subsume
modal run --detach rhm/rhm_sculpt_internalize.py::sculpt_compose --m 2
```

Both Stage-3c/3d runs assert the knob=0 anchor reproduces Stage 3b exactly before sweeping. Results JSON on the `rhm-scaling-data` volume under `rhm_sculpt_planner/`, `rhm_sculpt_latent/`, `rhm_sculpt_latent_po/`, `rhm_sculpt_latent_stoch/`, `rhm_latent_planner/`, `rhm_sculpt_precheck/`, `rhm_sculpt_deepbelief/` (Stage 4 gate; output dir tagged by the belief set), `rhm_sculpt_data2vec/` (Stage 4 isolation), `rhm_sculpt_internalize/` (Stage 5 ladder tagged by rung set; Stage 5b tagged `compose_<mask_mode>`).
