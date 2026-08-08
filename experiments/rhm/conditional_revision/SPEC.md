# SPEC — Conditional revision: is "how much the token moved my beliefs" separable from "how surprising the token was"?

**Up**: [../README.md](../README.md) · **Idea**: [`ideas/revision_not_surprisal.md`](../../../ideas/revision_not_surprisal.md)
**Sibling / predecessor**: [`../endogenous_teacher/`](../endogenous_teacher/README.md) — this is the
measurement that cut needed before its interventions.
**Date**: 2026-08-07 · **Status**: Gate 0 **run and passed** (2026-08-07,
[`conditional_revision.py`](conditional_revision.py)); instrument self-check built and passing
([`oracle.py`](oracle.py)); Gates A+B implemented ([`gates_ab.py`](gates_ab.py)) and running.
`README.md` lands after the run and after discussing results (repo convention).

**Run order**: Gate 0 → A → B → C. Each is a kill point. Gate 0 is half a day and needs no new
machinery; do not build the oracle until it passes.

**Gate 0 result (2026-08-07).** Passed on the registered prediction, both sides of the kill clear.
Against the depth FM's `corr(rres, nll) = −0.336`, the temporal FM reads **+0.652** pooled on flat
windows with `R² = 0.425` (kill was >0.9) and **57.5%** of residual variance orthogonal to `nll`;
the level profile flips from **−0.786** (anti-localised) to **+0.484** (aligned). Two controls hold:
`depth_cotrain` reproduces `endogenous_teacher`'s Gate 0 to three decimals (−0.337 / 0.114 / 88.6%),
so the substrate is the same one, and a protocol-matched `depth_frozen` arm reads −0.328, so the
flip is the axis change and not the training protocol. A `temporal_direct` parametrisation control
reads +0.638, so the update-form readout is not load-bearing either. The base checkpoint is
persisted; **no gate retrains it**.

---

## The question

Token surprisal `nll` bundles two things: the token told me something about the world's latent
structure (**reducible**), and the token was one of several equivalent realizations of structure I
already had (**irreducible**). NTP teaches from the sum. This cut asks whether the reducible part is
separable and whether the model's own state tracks it:

> **Is belief revision a distinct signal from token surprisal, and does the model's own revision
> track the true one?**

**Why this is not a correlation hunt.** On RHM the split is an identity, not a hope. For latents `z`
truncated at abstraction level ℓ,

```
E[ B_ℓ ]  =  H(x_{t+1} | x_{≤t})  −  H(x_{t+1} | z_{≤ℓ}, x_{≤t})
             total surprisal        irreducible-given-structure
```

Both terms are exactly computable by sum-product belief propagation on the known parse tree with the
known rules. So the aleatoric/epistemic decomposition is **indexed by the DGP's own abstraction
ladder** — the same ladder every RHM readout in this repo uses.

---

## Where this came from, and what it actually claims

The origin is a phenomenological question about reading (2026-08-04, Jasper): *"When humans read a
book … each word comes in, and we register things like its innate surprise to us, what ideas it
provokes … Is consuming your own 'experience' of the text the same as consuming/predicting your own
latents?"* The claim that survived two rounds of scrutiny:

> **Felt surprise is how much a word changes your model, not how improbable it was.**

A rare synonym is high-surprisal and revises almost nothing; a quiet disambiguating word is
low-surprisal and can revise a lot. **No arity-1 objective can express that difference**, and NTP is
the arity-1 objective — which is why the *only* channel an LM has ever been trained on is the one
this claim says is the wrong one.

**Two things this retro-explains beyond itself**, both of which are reasons to care even if the
measurement here is null:

1. **Why the a2a self-knowledge line kept landing on legibility.** A depth FM's predictor and target
   have identical information sets, so its residual is the FM's failure to compress M — a property of
   the *(M, FM) pair*, not of M's relation to the world. `MNIST_LOCAL_LOSS`'s `LL` condition maximizes
   exactly that (FM cos **0.997**) and has **zero** self-knowledge (SK R² ≈ −0.03). Four results say
   the same thing.
2. **Why the Rao–Ballard / Friston local-loss port came back null.** With an endogenous target,
   "minimize prediction error" is definitionally "**be simpler**," not "model the world better" — and
   precision weighting is uninformative because there is no aleatoric component for it to weigh. See
   [idea doc](../../../ideas/revision_not_surprisal.md) §8. **The temporal FM specified below is the
   same object that would fix that arc**, which is the main reason to run this even at a low prior.

### Provenance — what is arithmetic, what is measured, what is argued

A future agent should weight these differently, and should feel free to attack the argued row.

| status | claim |
|---|---|
| **arithmetic** | the compression/revision split is an identity; `E[B_ℓ] = H(x\|prefix) − H(x\|z_{≤ℓ},prefix)` is exact; in belief space the arity-1 forecast is the identity (martingale/tower property) |
| **measured** | the depth residual is anti-localised against the hierarchy and anti-correlates with `nll` (−0.336); `res` ≈ `res_shuffled` under scalar weighting; ordering by `nll` is the *worst* assignment; endogenous targets cap in four independent settings |
| **argued** | that revision is *aleatoric-immune* and therefore a better epistemic allocator than surprisal. **This is the load-bearing untested claim and the entire point of the cut.** |

**Which parts are load-bearing.** Three claims stack: (1) the operative variable is the *conditioning
gap*, not depth-vs-time; (2) revision ≠ surprisal; (3) revision is aleatoric-immune, hence an
allocator. If (2) dies, (1) and (3) survive as theory but have **no instrument**, and the honest move
is to say so rather than to look for a different readout of the same claim. If (1) dies — see Gate 0's
`corr ≈ 0` branch — the whole framing is wrong, not just the measurement.

---

## The object — shift the target by one position, not six blocks

Everything downstream rests on one change to the existing FM:

```
depth FM   (existing):  FM( h₀[≤t] )  →  h₆[t]                    residual: 0% aleatoric
temporal FM (this cut): FM( h₆[≤t] )  →  Δ_t = h₆[t+1] − h₆[t]
```

One small causal FM against a **frozen** main model. No injection, no loop, no cancellation, no
architecture change, no base training.

**Why it is epistemically charged, in one line**: `h₆[t+1]` depends on `x_{t+1}`, which the corpus
supplies and the FM cannot hold — so for the first time in this program's activation-FM line the
residual has a **nonzero aleatoric component**. That is the whole change, and it is what the depth FM
lacks by construction ([idea doc](../../../ideas/revision_not_surprisal.md) §1).

**Prior work, scoped precisely.** RHM *has* had a temporal FM — [`RHM_SCULPTING`](../RHM_SCULPTING_README.md)
Stage 3b rolls the belief forward one step, as do `rhm_latent_planner.py` / `rhm_generative_planner.py`
— but on the **move axis** (t = edit-step in a control task), never the **position axis** (t = token).
Stage 3b's lesson (open-loop rollout collapses, needs Dreamer re-grounding) is about *multi-step*
rollout and does **not** transfer to one-step prediction. Do not import that pessimism.

### Four design choices, each forced by a prior result

1. **Predict at the deepest block, not the shallowest.** `h₀[t+1]` is essentially the embedding of
   `x_{t+1}`, so its residual would be token surprisal *by construction* — the kill built into the
   design. `h₆[t+1]` is what the new token did to the accumulated representation. That is the object.
2. **Predict the update `Δ_t`, not the state.** [`ACTIVE_VISION`](../../a2a_forward/reaching/ACTIVE_VISION_README.md):
   predicting `s_{t+1}` scored **0.94 by echoing carried state**. Residual streams carry state forward
   hard; this will be worse on the reading axis than it was there.
3. **Give the FM the full prefix.** A causal FM over all of `h₆[≤t]` holds everything *except* the new
   token, so the conditioning gap is **exactly one token** — clean, not approximate. This resolves the
   parent doc's open question 1 (*"`h[t+1]` depends on the whole prefix, not just `(h_t, x_{t+1})`"*).
4. **Open-loop, stop-grad on the main model.** The §8 collapse-toward-input-invariance degeneracy
   exists *only* if the FM's loss backprops into the main model. As a **measurement** there is nothing
   to collapse. The risk is a property of the intervention, not of the object.

**Honest caveat**: the depth and temporal FMs are not perfectly matched (different input altitude), and
a perfect match does not exist — `h₆[≤t] → h₆[t]` is the identity. The comparison is not "which FM is
better" but "does the residual have an aleatoric component," so this is a caveat, not a blocker.

---

## Gate 0 — did the axis change do anything? (half a day, no new machinery)

Frozen existing checkpoint + one `1L/8H/16d` FM. Run this before building anything else.

Reference: [`endogenous_teacher`](../endogenous_teacher/README.md) Gate 0 measured the **depth** FM's
residual at `corr(rres, nll) = −0.336`, `R²(rres ~ nll) = 0.113`, 88.7% of variance orthogonal to
`nll`, and **anti-localised** against the hierarchy (residual peaks leaf-adjacent at 0.436 where `nll`
is lowest at 0.802; residual minimal near root at 0.269 where `nll` peaks at 2.609).

**Measure the same statistics for the temporal FM's residual `r_temporal`.**

**Two-sided kill:**

| outcome | reading |
|---|---|
| `corr(r_temporal, nll) ≈ 0` and the level profile still anti-localised | the axis change did nothing. **Stop.** |
| `R²(r_temporal ~ nll) > 0.9` | the temporal residual is token surprisal re-expressed in state space. **This is the idea doc's primary kill, arriving cheap.** Stop. |
| positive, moderate correlation; level profile **aligns** with `nll` instead of inverting it; a substantial `nll`-orthogonal component remains | the residual is a genuine mixture and there is something to decompose. **Proceed to Gate A.** |

Report both pooled and within-position, and on aligned sequences (where the level map exists) as well
as flat concatenated windows (what training sees) — same as the predecessor Gate 0.

**Registered prediction**: `corr` flips from −0.336 to clearly positive, and the level profile inverts
to track `nll`'s (high near root, low leaf-adjacent). Magnitude unknown; the interesting outcome is
positive-but-not-saturating.

---

## Substrate and reference lines

Regime `v16 s2 L6 m4`, model `8L/8H/256D`, base at 12k steps — **identical to**
[`endogenous_teacher`](../endogenous_teacher/README.md) and
[`RHM_LATENT_LOOP`](../RHM_LATENT_LOOP_README.md), so their reference lines transfer (base d1 0.979,
d3 0.836, root 0.088; BP root 0.80; token-NTP frontier d1 0.98, d3 0.88, d4 0.51; greedy floor
d3 0.751, d4 0.708). No new base training is needed for Gates 0–B — reuse the existing checkpoint.

## Existing machinery this leans on

| piece | where | what it gives |
|---|---|---|
| exact BP over the parse tree | [`../rhm_bayes_entropy.py`](../rhm_bayes_entropy.py) — `_upward`, `_downward`, `bp_conditional_entropies`, `_position_levels`, plus a brute-force `_self_test` | node marginals `P(z_node \| evidence)` and exact conditional entropies. The addition landed in [`oracle.py`](oracle.py) rather than as an edit to that module, so prior results stay bit-identical: normalised messages, optional internal-node evidence (needed to clamp `z_D` for `H_irr`), node + clique marginals, the junction-tree joint KL, and an equivalence test against the incumbent. |
| the depth FM and its Gate-0 statistics | [`../endogenous_teacher/`](../endogenous_teacher/README.md) | the matched null-aleatoric comparison for Gate 0 |
| per-level ancestor probes | `endogenous_teacher` / [`RHM_SCULPTING`](../RHM_SCULPTING_README.md) Stage 4 readouts | the model's belief `b_t` = probe-decoded posterior over ancestors at each level |
| matched-arity idiom | [`../../mjc/arity_torque/`](../../mjc/arity_torque/README.md) | *"identical data — only the input differs"* — the control that makes `p⁺ − p⁻` attributable to the slot |
| capacity-invariance discipline | [`../residual_decomposition/`](../residual_decomposition/README.md) | β held to ±0.01 across a 4× FM-capacity sweep; the same sweep is the ε-control here |

## Quantities

Per position `t`, per level `ℓ ∈ 1..6`:

- `nll_t` — realized token surprisal under the model. Exogenous, the incumbent.
- `r_temporal` — the Gate-0 residual, `Δ_t − FM(h₆[≤t])`.
- `B_t^ℓ` — **oracle** belief revision: `KL( P(z_{≤ℓ} | x_{≤t+1}) ‖ P(z_{≤ℓ} | x_{≤t}) )`, exact by BP.
  Compression-free by construction.
- `M_t^ℓ` — the **model's** belief revision: `d( b_{t+1}^ℓ , b_t^ℓ )` on probe-decoded posteriors.
- `Δ_t^rev` — the **forecast** revision `p⁺ − p⁻` from a matched FM pair (Gate C only).
- `H_irr^ℓ = H(x_{t+1} | z_{≤ℓ}, x_{≤t})` — the irreducible-given-structure term, exact by BP.

**Instrument self-check, run before Gate A**: `B̂_ℓ + H_irr^ℓ` must reconstruct the BP conditional
entropy to within numerical tolerance at every ℓ. If it does not, the marginal extraction is wrong and
nothing downstream is interpretable. This is the calibrate-before-measure step
[`mjc/expansion`](../../mjc/expansion/README.md) is the precedent for.

---

## Gate A — does the model's belief revision track the oracle's, beyond surprisal? (no FM)

One existing checkpoint + probes + BP; no training.

**Measure**, pooled and per level, on aligned sequences *and* flat windows:

- partial `R²( M ~ B | nll )` — does the model's revision carry oracle structure surprisal doesn't explain?
- partial `R²( M ~ nll | B )` — how much of it is just surprisal?
- `R²( B ~ nll )` — how separable the two references are at all.

**Kill**: partial `R²(M ~ B | nll) < 0.05` at every level → the model's internal revision is a
restatement of token surprisal; §3–§4 of the idea doc are bookkeeping.
**Proceed**: > 0.15 at any level. Between: proceed and report underpowered.

## Gate B — the constructed contrast (the load-bearing test)

Gates 0 and A are contaminated by a confound the idea doc names explicitly: to first order
`E‖revision‖² ∝ H(x_{t+1}|prefix)`, which is what `nll` samples. **Magnitude cannot be the
discriminator.** So construct positions where `nll` is held fixed by design and only `B` varies. RHM
permits this exactly, because synonymy is a DGP primitive.

Two matched families, sampled to have overlapping `nll` distributions (report the overlap; discard
non-overlapping tails):

- **synonym positions** — the token is determined by *which of the `m` rules* realized an
  already-determined parent feature. Oracle: `B^ℓ ≡ 0` for ℓ above the rule's level, **by
  construction**. `nll` high.
- **disambiguating positions** — the token resolves ambiguity over a higher-level feature. Oracle:
  `B^ℓ` large at high levels. `nll` may be low.

**Readout**: AUC of each of `{nll, M, r_temporal, B}` at separating the two families, per level.
Report all four; `B` is the ceiling (it defines the families) and `nll` is the incumbent.

**Registered prediction**: `M` beats `nll` on AUC at levels 1–3 (where synonymy and disambiguation are
distinguishable), and does not at leaf-adjacent levels.

**Kill**: `M` ranks the families like `nll` and not like `B` → revision is surprisal re-expressed in
state space. **This is the primary kill for the whole idea doc.**

## Gate C — the forecast version, the martingale test, and the ε-control

Only if 0, A and B pass. Trains two FMs.

**Matched pair, `arity_torque` idiom**: identical architecture, capacity, data, optimizer, seed and
step budget. `FM₂` receives `x_{t+1}`'s embedding in a dedicated slot; `FM₁` receives a learned mask
token in the same slot. **Only the slot's contents differ.** `FM₁` is the Gate-0 temporal FM.

Three readouts:

1. **Martingale.** Idea doc §3 predicts the Bayes-optimal `p⁻` is the identity
   (`E[b_{t+1}|x_{≤t}] = b_t`). Measure `cos(p⁻, b_t)` and `‖p⁻ − b_t‖ / ‖b_t‖`. A learned `p⁻` that
   does *not* approach `b_t` is a direct readout of belief miscalibration, interesting either way.
2. **Capacity invariance (the ε-control).** Sweep FM capacity 4× on both arms. `Δ^rev`'s relationship
   to `B` must be flat. **If it moves with capacity we are measuring `ε₂ − ε₁`, not revision** — the
   `residual_decomposition` standard, where β held to ±0.01 across 4× while the level moved 1.8×.
3. **The compression term, reported not hidden.** `‖b_{t+1} − p⁺‖` and `B − Δ^rev` per level. The
   second is "how much of the true revision the model registered" — a DGP-legibility readout in its
   own right.

---

## Standing priors, what each null teaches, and when to stop running these gates

**The base rate is against us, and a future agent should read a marginal positive against it.** Four
independent replications say endogenous targets cap or invert: `LL` (FM cos 0.997, SK ≈ 0), `λ_local`
(ΔSK −0.06 → −1.54), `data2vec` (+23% against grounded MLM's +44%), `fm_cotrain` (*"grounding is the
pivot"*). None was conditional-revision, so the cap does not transfer directly — but note the shape of
the one exception: [`full_loop`](../directed_sculpting/full_loop/README.md) §5's expansion came from a
**grounded** evaluative grader (external, precomputed DP-best-move), and LP-alone was separately the
*weak* tap (18% vs relevance's 84%). The oracle `B_t` here is grounded in the same sense. **If revision
works only when read against the oracle and not from the model's own estimate, that is the same finding
a fifth time**, and it should be reported as such rather than as a new positive.

**Why the predecessor missed, in two lessons worth not repeating.**
[`endogenous_teacher`](../endogenous_teacher/README.md) ran two interventions and both fired the
pre-registered falsification. The causes were structural, not bad luck: (1) it **intervened before
establishing the signal was real** — this cut is measurement-only for that reason; (2) it **collapsed a
directional object to its norm** and used it as a loss multiplier, testing the part already known to
carry nothing (vector probe ΔR² +0.18 vs scalar +0.03). Anything built on top of this cut must keep the
revision object **directional**.

**What each null would teach.** Nulls here are informative, not wasted — record them.

| gate fires | what we learn |
|---|---|
| Gate 0, `corr ≈ 0` | the conditioning gap is **not** the operative variable — the strongest possible result, because it falsifies the reframe itself and not just the instrument. Idea doc §1 is wrong and the depth/temporal distinction is decorative. |
| Gate 0, `R² > 0.9` | the temporal residual is predictive entropy in state space. Revision is not a new signal; **the honest next question is whether it beats output entropy on language** (Exp A), where entropy is the strong incumbent rather than `nll`. |
| Gate A null, Gate 0 passed | the aleatoric component exists but the model's *belief* does not track the oracle's revision. This is a statement about the **base model's belief quality**, not about revision — go to the belief-depth lever (`RHM_SCULPTING` Stage 4) before concluding anything about the signal. |
| Gate A passes, **Gate B null** | the effect was magnitude all along — `E‖revision‖² ∝ H(x\|prefix)`. §4's aleatoric-immunity claim is dead; §1 and the §8 local-loss diagnosis both survive it. |
| Gate C: `Δ^rev` moves with FM capacity | we were measuring `ε₂ − ε₁`. Not a finding about revision; a finding about the instrument. Report and stop. |
| Gate C: `B − Δ` large at every level | the model registers little of the true revision. The result is about **DGP legibility**, not about revision as a signal — pivot to the belief-depth line rather than to the teaching intervention. |

**When to stop running these gates and do something else.** Explicit permission, because the gates are
one route to the claim and not the claim itself:

- **The martingale test (Gate C.1) fails badly** — a learned `p⁻` far from `b_t`. Then belief space is
  the wrong frame for *this* model, and the residual-stream version is worth running despite its
  echo/state-definition problems. Do not force the belief framing.
- **Everything passes but only at one level ℓ.** Then the finding is about that abstraction level, not
  about revision, and the cut should be rewritten around the level axis. The confound table already
  flags this; treat it as a redirect, not a caveat.
- **Language starts looking cheaper than Gate C.** The Exp-A probe swap runs on cached checkpoints
  against a *stronger* incumbent (output entropy, which beat every activation probe) than `nll` is.
  If Gates 0–B pass cleanly, jumping straight to language is defensible — Gate C is the careful version,
  not the necessary one.
- **The §8 local-loss test looks more decisive.** An LL-alone arm with the exogenous gap either
  reproduces the depth version's signature (FM cos → ~1, SK → 0, brittleness) or does not. That is a
  cheaper and sharper test of the *conditioning-gap* claim than anything here, and if the goal is to
  test the reframe rather than to build the allocator, it may be the better first cut.

---

## Confounds and handling

| confound | handling |
|---|---|
| revision magnitude ∝ `H(x\|prefix)` ≈ `nll` | Gate B holds `nll` fixed by construction; partial correlations everywhere else; Gate 0's kill is explicitly two-sided so a saturating correlation counts against us |
| temporal residual is trivially the token | predict at `h₆`, not `h₀` (design choice 1) |
| echo degeneracy | predict the *update* (design choice 2); belief space makes the echo the *correct* arity-1 forecast (§3), so it is structural rather than patched |
| depth/temporal FMs not perfectly matched | acknowledged; no perfect match exists (`h₆[≤t] → h₆[t]` is the identity). The claim is about presence of an aleatoric component, not relative FM quality |
| collapse toward input-invariance | impossible as specified — open-loop, stop-grad on the main model (design choice 4). Becomes live only in the follow-up |
| two capacity terms inside `Δ^rev` | matched FM pair (only the slot differs) + 4× capacity-invariance sweep |
| probe quality masquerading as belief revision | probes trained on held-out positions; report per-level probe R²; shuffled-probe control |
| position↔level coupling | run on aligned **and** flat windows, per the predecessor Gate 0 |
| coarse-graining choice dominates | every readout reported per ℓ; ordering across signals must be stable in ℓ or the finding is about ℓ |
| BP marginal extraction wrong | the `B̂_ℓ + H_irr^ℓ` reconstruction self-check, before Gate A. Built in [`oracle.py`](oracle.py) and passing: exact to machine precision (≤4e-16) against brute-force enumeration on tiny trees, agreeing with `rhm_bayes_entropy.bp_conditional_entropies` to 3e-15, and satisfying the identity on the real regime to Monte-Carlo error |
| **the latent graph is a hypertree, not a pairwise tree** | one rule emits a parent's whole `s`-tuple, so the children are dependent given the parent and a parent–child-edge factorisation is *wrong* — it reads H = 1.733 where the truth is ln 4 = 1.386 on the smallest test tree. The joint KL uses junction-tree cliques `{p} ∪ children(p)` with node separators. Caught only by brute force; keep that test |
| **matching strata leak at atoms of the matching variable** | exact BP surprisal piles mass on 0, ln 2, ln 3, ln 4…, so quantile edges repeat, `digitize` merges each atom with the continuous spread above it, and the matching variable separated the families at AUC **0.999 inside a stratum meant to hold it fixed** (self-matched 0.80 instead of 0.50). Strata are atom-aware, and every Gate-B block reports the matching variable's *self*-matched AUC as a guard |
| a signal that merely estimates surprisal better than the model does | matching on the model's `nll` alone does not close this: exact surprisal still scored 0.64–0.78 nll-matched, purely because model `nll` is a noisy proxy for it. Gate B therefore reports a second matching on `bp_surprisal`, where a pure-surprisal signal is pinned at 0.5 by construction |
| Gate 0's residual could be its target's norm | `delta_norm` (‖Δ_t‖, no forward model at all) is scored alongside `r_temporal` everywhere |
| rank-shaped instruments | none used. β is unusable below ~100 directions and RHM's `R_res_participation` is 7 — do not reach for it here |
| seed | single seed per repo convention; seeds added only if the effect is real and plausibly seed-sensitive |

## What this deliberately does not test

- **Any teaching intervention.** `endogenous_teacher` intervened before establishing the signal was
  real, and both cuts missed. Measurement first.
- The injection/cancellation forward path — see [`../endogenous_teacher/cancellation/`](../endogenous_teacher/cancellation/README.md).
- Multi-step rollout of the temporal FM. One step only; Stage 3b is the warning.
- Anything about a temporal *lead* in the causal-use sense. Per idea doc §2, measurement needs only
  conditioning; timing is a separate architectural fork.

## Follow-ups this gates (not part of this cut)

**1. Language.** No ground truth, so not first — but it has something RHM does not: a pre-existing null
with a pre-existing baseline. [`OOD_ROBUSTNESS`](../../a2a_forward/OOD_ROBUSTNESS_README.md) Experiment
A found competence probes transfer identically across all conditions and **output entropy beats every
activation probe** (ρ 0.49 vs ≤0.38 ID; retention 0.64–0.74 vs ≤0.65). The temporal residual is the
first residual here that could compete, because it contains a term governed by `H(x_{t+1}|prefix)` —
which is what output entropy measures. That makes the test unusually clean: **matching output entropy
is the kill** (it is predictive entropy in state space); **beating it is the finding**. Instruments,
corpora and baselines all exist; it is a probe swap on cached checkpoints.

**2. The teaching signal.** If 0–C pass: re-run the `endogenous_teacher` weighting cut with the right
quantity and the right sign. That cut weighted by the **instantaneous residual magnitude** — `|E|`, the
noisy TV — where the theory calls for **precision** `Π`, an *accumulated reducibility estimate over a
slower timescale* (idea doc §5; Ruffini et al. §2.3/§4.2: precision estimation requires pooling errors
over time). This family has produced three scalar-gating nulls (`EMOTION_INJECTION`; the
directional-vs-scalar belief node; `endogenous_teacher` itself), so the version worth building keeps
`Π` **directional** — a low-rank operator, not a coefficient. Note that this is also where the
input-invariance degeneracy becomes live (idea doc §8), so the anti-collapse machinery in
[`../rhm_sculpt_data2vec.py`](../rhm_sculpt_data2vec.py) is a prerequisite.

**3. The local-loss fix.** The same temporal target used as an auxiliary *training* signal is idea doc
§8's candidate non-degenerate local loss. Its own cheap kill is stated there: if LL-alone with the
exogenous gap reproduces the depth version's signature (FM cos → ~1, SK → 0, comparable brittleness),
the conditioning gap is not the operative variable.

## Reproduction (to be filled in on first run)

```bash
cd experiments
# smoke test (~2 min, attached)
modal run -m rhm.conditional_revision.conditional_revision::gate0 \
    --base-steps 400 --fm-steps 400 --pool-size 20000 \
    --n-eval-sequences 1024 --gate0-batches 4 --tag smoke

# Gate 0 proper — 12k-step base + 12k-step frozen-base FMs, ~1.5 h on an L4
modal run --detach -m rhm.conditional_revision.conditional_revision::gate0 \
    --base-steps 12000 --fm-steps 12000 --tag gate0

# instrument self-check + Gate A (CPU, no training) — NOT YET IMPLEMENTED
modal run -m rhm.conditional_revision.conditional_revision::gates_ab \
    --v 16 --s 2 --depth 6 --m 4 --tag gateAB
```

Results land on the `rhm-scaling-data` volume at
`/data/v16_s2_L6_m4_distinct/conditional_revision/results_<tag>_seed<seed>.json`.

**Volume path correction (2026-08-07).** This spec originally said `/data/v16_s2_L6_m4/`.
That directory does not exist. The regime is built with `generate_rules_distinct`, and both
`endogenous_teacher` and `temporal_lead` write under **`/data/v16_s2_L6_m4_distinct/`** — that is
the real path and the one the code uses.

**No reusable base checkpoint existed (2026-08-07).** The "half a day" estimate assumed one did.
`endogenous_teacher` keeps its 12k-step base in process memory and never writes it to the volume;
the only `.pt` on the volume for this regime is `rhm_latent_loop/.../levelfocus_ntp/ckpt_step0.pt`
(step 0, useless). Gate 0 therefore trains the base itself (`--base-steps 12000`, ~1 h on an L4)
and **does** persist it to
`/data/v16_s2_L6_m4_distinct/conditional_revision/base_8L8H256D_steps12000_seed42.pt`, so Gates A–C
can reuse it via `--base-ckpt`.
