# RHM Sculpting & planning in latents: making RHM a control task that *needs* lookahead

**Status**: **WIP.** Stage 2 (latent-planning attempt — instructive dead-end), Stage 3a (learned beam captures the prize — clean positive), Stage 3b (planning in latents — latent ≈ token at ~8× lower cost on the *clean* channel), and **Stage 3c/3d (latent *beats* token once the token channel is lossy — partial observability + stochastic dynamics — clean positives)** all done, single seed.
**Date**: 2026-07-12
**Scripts**: [`rhm_latent_planner.py`](rhm_latent_planner.py) (Stage 2: faithful value + cerebellar FM), [`rhm_sculpt_precheck.py`](rhm_sculpt_precheck.py) + [`sculpting_control_task.md`](sculpting_control_task.md) (the perfect-simulator pre-check that validates the task), [`rhm_sculpt_planner.py`](rhm_sculpt_planner.py) (Stage 3a: the learned token-space beam), [`rhm_sculpt_latent.py`](rhm_sculpt_latent.py) (Stage 3b: the latent-space beam, head-to-head), [`rhm_sculpt_latent_po.py`](rhm_sculpt_latent_po.py) (Stage 3c: partial observability), [`rhm_sculpt_latent_stoch.py`](rhm_sculpt_latent_stoch.py) (Stage 3d: stochastic dynamics).
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

## What this arc establishes (so far)

1. **Faithfulness ≠ plannability, and density must be *learned*, not faked.** A faithful verifier is too sparse; a Monte-Carlo value (dense-by-experience) is the right instrument (dense distance-to-goal, ungameable).
2. **A task must actually contain lookahead before a planner can demonstrate it.** The corrupt-repair task is greedy-decomposable (greedy oracle = 1.0); sculpting (choose a globally-consistent derivation, DP optimum vs choosing-reflex) is not. This is the reaching-arc lesson (obstacles/maze forced lookahead) reproduced in RHM.
3. **A learned beam captures the coordination prize** — monotone in width, up to and past the strong hand-coded reflex (0.611 with wide beam + budget) — where a greedy rollout collapsed. The task, not the method, gated it.
4. **On the *clean* channel, planning in latents is a viable *efficient surrogate*, not a superior one.** With a richer non-pooled latent + a re-grounded cerebellar FM, a latent beam reaches 92% of the token beam at ~8× fewer materializations. It approaches but does not beat token-space *here*; beam width compensates for the FM's imperfect per-move ranking. The open-loop version collapses — re-grounding (never rolling the FM more than one step from truth) is load-bearing. **But "surrogate, not superior" is channel-specific** — see point 5.
5. **On a *lossy* channel, planning in latents is genuinely *superior*.** Whether tokens are a sufficient statistic decides it. Partial observability (Stage 3c): the latent beam carries belief the sensor drops → gap flips to +0.02–0.04 for p ≥ 0.25. Stochastic dynamics (Stage 3d): the latent beam ranks by a stable FM estimate instead of one noisy sample → gap flips to up to +0.075 for q ≥ 0.1 (the mirror of the query null — payoff in the mean the token beam must sample). Both are width-gated (need beam ≥ 64 to beat the FM's 0.36 top-1). Full-obs-deterministic is the special corner; real DGPs are lossy, so *superior*, not *surrogate*, is the common case.

## What this does NOT establish / caveats

- **Single seed, single setting** (L=4, m=2, c=3); the learned beam *matches* the strong reflex but has not passed it or reached the DP optimum (1.0). Remaining headroom: value-iteration (recollect with the beam as behaviour policy — its 0.34 success rate caps the value), beam width, better moves.
- **Latent planning does not beat token-space *on the clean channel*, and lags at narrow beams** — the FM's per-move ranking (top-1 36%, rank-corr 0.52) is the bottleneck; wide beams hide it. It *does* beat token-space once the channel is lossy (Stage 3c/3d), but the narrow-beam lag persists across all channels (the width-gating). The efficiency claim (~8× fewer materializations) assumes the FM is meaningfully cheaper than generator+controller (it is smaller, but same order).
- **Stage 3c/3d crossover margins are modest (+0.02 to +0.08)** — single seed. The *pattern* is robust (monotone across 4 widths × 5 knob levels × 2 channels, exact knob=0 anchors, and it mirrors both the sufficiency prediction and the query null), but the per-cell magnitudes are small. Two principled, untested ways they should widen: a **belief-aware value** for PO (train the value on filtered beliefs, removing the full-obs-consumer cap) and **Design 2** for the FM under noise (train it on slippery targets).
- **Stage 3c masks the *evaluation* channel only** (candidate construction held full-obs) — the clean isolation of the sufficiency claim, but it does *not* test whether latents also help *propose* moves blind. Stage 3d's stochasticity, by contrast, hits construction (the actuator) directly.
- Distinct (ambiguous) rules make "success" = r\* derivable; at loose coupling (m=4) this is easy (prize collapses) — the sculpting *coordination* effect lives at tight coupling (m=2/3). The 3c/3d *channel* effects are a separate axis and were not re-swept over m.

## Next steps

1. ~~**Stage 3b — plan in latents.**~~ *Done*: latent beam reaches 92% of token beam at ~8× lower cost (an efficient surrogate on the clean channel).
2. ~~**Is there *any* regime where latents *beat* tokens rather than approximate them?**~~ *Done* (Stage 3c/3d): **yes — whenever the token channel is lossy** (partial observability, stochastic dynamics). The sufficiency-of-tokens argument predicted exactly these two knobs; the clean-channel "surrogate, not superior" was the special case.
3. **Widen the 3c/3d margins with the two principled upgrades** — a **belief-aware value** (train the MC value on filtered/masked beliefs, not only full-obs ones — removes the full-obs-consumer cap) and **FM Design 2** (retrain the FM on slippery targets so it predicts the true k-dependent expectation). Both should turn a +0.05 into something less equivocal.
4. **The combined channel** — sweep partial-obs × stochastic together (real DGPs have both). Does the latent advantage compound?
5. **A better FM / value-iteration** — a stronger FM (capacity/training, or predicting the *value* directly) should lift the top-1 pick (0.36) and shrink the width-gating; value-iteration (recollect with the beam as behaviour policy, its 0.34 success caps the value) lifts both planners and would push the token beam past the strong reflex toward the DP optimum (also L=5 depth).
6. **The m=4 negative control in the learned setting** (beam should *not* help the *coordination* prize when coupling is loose — the learned analog of the pre-check's control; note the 3c/3d *channel* effects are a separate axis).

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
# Stage 2 (the instructive dead-end): faithful value + latent FM on corrupt-repair
modal run --detach rhm/rhm_latent_planner.py::latent_planner --m 2
```

Both Stage-3c/3d runs assert the knob=0 anchor reproduces Stage 3b exactly before sweeping. Results JSON on the `rhm-scaling-data` volume under `rhm_sculpt_planner/`, `rhm_sculpt_latent/`, `rhm_sculpt_latent_po/`, `rhm_sculpt_latent_stoch/`, `rhm_latent_planner/`, `rhm_sculpt_precheck/`.
