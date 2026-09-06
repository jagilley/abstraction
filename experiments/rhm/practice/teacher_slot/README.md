# teacher_slot — the teacher slot is a gauge-choice, and language can fill it

**Up**: [`../README.md`](../README.md) (practice arc) · **Spec**: [`SPEC.md`](SPEC.md) (the record of
what was asked, 2026-08-20) · **Files**: [`FILES.md`](FILES.md) · **Substrate donor**:
[`../fourwall/lm/`](../fourwall/lm/README.md) — `wall.py`/`wall_lm.py` inherited verbatim, untouched.
**Parents**: [`practice_manufactures_its_own_credit`](../../../../ideas/practice_manufactures_its_own_credit.md)
§18 · [`recurrence_manufactures_confounds`](../../../../ideas/recurrence_manufactures_confounds.md) §9 ·
[`two_timescale_value_loop`](../../../../ideas/two_timescale_value_loop.md) ·
[`heterogeneous_graders`](../../../../ideas/heterogeneous_graders.md) §3 (Rung B's bet).
**Runs**: `tsd_gate`/`tsd_gate2`/`tsdA`/`tsdB` (decision + yield, 2026-08-20), `b1` (verbal,
2026-08-20), `hr_s0` (handle, 2026-08-20). **Ranks, signs, trajectories, and floor-multiples
are the claims.** One orchestrated conversation; each rung was
built end-to-end by a delegated implementer agent and the machinery details below summarize their
work rather than receiving separate writeups.

## The question

`fourwall/lm` ended on the arc's sharpest statement of the teacher problem: a real NTP reader takes
a free spurious key instantly, tracks its rotations forever, and never quotients — ending at
Bayes-level task error with its latent-inference pathway threefold suppressed — while the supplied
merge op (collapse the key token to a neutral filler; conditions, not weights) repairs that at
~zero price, *and the within-level lifetime ledger rewards never merging*. Both the op and its
justification had to come from outside the loop. This node splits that "outside" into parts and
asks which parts actually require a teacher:

- **Rung A** — given the right currency, will a *minimal* outer loop choose the merge and hold?
- **Rung A½** — can the currency be the learner's own **next-level yield** (the arc's most
  seed-stable fact: level-k practice is priced in level k+1's currency)?
- **Rung B1** — can a **verbal** reasoner reading the instruments make the call — in particular
  *pre-rotation*, when nothing in the within-level evidence marks the key as spurious?
- **`handle/`** (separable, ratchet substrate) — was the arc's inert plant just the earned
  vocabulary's missing backprop handle?

## Design in brief

Inherited world exactly (v16/s2/L6/m4, rule_seed 0; prefix token `w` bijective with the level-2
latent at node 0; `wall` schedule: identity to 8000, rotate +4 mod 16 every 2000; 8L/8H/256D GPT,
20k steps × 64; one worker per arm, zero global RNG). New: `decision/slot_lm.py` wraps the donor's
training loop with a per-checkpoint (125-step), **reversible** condition choice — `w` present or
collapsed — driven by a deliberately minimal value rule. **Fidelity gates**: a hard-coded
step-function policy reproduces the donor bit-for-bit — max|Δ| = 0.0 on five instruments over
93/161/81/**175** shared checkpoints (`wall`, `no_wall` ×2, `merge_8000` incl. the merge
micro-grid), re-verified after every shared-code change (`tsd_gate2`) — so fwlm1's stored
`merge_1000`/`merge_13000` are licensed as comparators without re-running.

The decision rule (round 2, `tsdB`): **ABBA paired trials** — a `C K K C` / `K C C K` block per
trial, crediting the mean difference in the read quantity's improvement between collapse- and
keep-quarters, which cancels any linear learning trend exactly (offline gate: 8.9e-16 on a pure
trend) and is label-symmetric (mirror-world gate: 0 mismatches). Dead zones sit at *measured*
instrument floors (0.0012 nats task/path; 0.005 yield = the dead-pair d5 spread), not free
parameters. Each policy is charged for what it reads at its measured compute price (bench on L4:
pathway probe 6 step-equivalents; d5 yield probe **77**), deducted from the shared 20k-step budget.
Round 1 (`tsdA`) is on the record as an instrument lesson: a naive baseline-normalized bandit's
first-interval reward ran ~95× ambient scale, making whichever action was tried second win by
construction — an artifact that *manufactured* an apparent unanimous merge. Its two clean
fragments (the task-reader's converged-baseline bail-out at 17500; the LP form's resistance to the
scale artifact) survive; everything decision-shaped below is round 2.

## Findings

### 1. The dissociation: each loop optimizes exactly the currency it can see

Six arms differing only in the read quantity. Paired contrasts separate by **sign and ~14×**:

| arm | reads | final V | commit | bail-outs | terminal d4 rand | lifetime ∫ (uniform grid) |
|---|---|---|---|---|---|---|
| `outer_task` | own experienced NLL | **−0.180** | **never** | 0 (5 declines) | **0.312** (hollow) | **1.4690** |
| `outer_path` | pathway readout (neutral-wall excess) | +0.014 | 1500 | 0 | 0.763 | 1.5360 |
| `outer_path_lp` | same, 250-step horizon | +0.026 | 2000 | 0 | 0.803 | 1.5326 |
| `outer_path_true` | same, never-rotating schedule | +0.017 | 1500 | 0 | 0.638 | 1.5332 |
| `outer_yield` | **d5 next-level yield** | +0.009 | 1500 | 0 | 0.776 | 1.5368 |
| `outer_yield_lp` | same, 250-step horizon | +0.011 | 2000 | 0 | 0.712 | 1.5326 |

References: `wall` d4-rand 0.273 / ∫ 1.4392 · `no_wall` 0.843 / 1.5344 · `merge_8000` 0.835 /
1.5025 · fwlm0 `true_wall` ∫ 1.4258. **`outer_task` never merges**: handed a collapse interval by
the trial schedule five times (steps 1500→17500), it greedily undid it every time, booking the
key's oracle value (~0.15–0.23 nats/interval, *growing* over the first 8000 steps as the key is
capitalized) as the price of collapse. Its reward is nearly the best within-level lifetime ledger
in the whole arc — second only to `true_wall`, ahead of every supplied merge — and it ends with the
inference pathway threefold suppressed (an *underestimate* of a pure never-merge policy's
hollowness: 2500 scheduled trial-collapse steps partially repaired it). The five
counterfactual-currency loops merge, hold with **zero bail-outs**, land at the bottom of the
ledger, and end whole (terminal task NLL within ±0.004 of `no_wall` — at floor, unrankable). The
incentive gap is not mere blindness: **the ledger refuses the crossing, optimally, every time it
is offered.**

### 2. Pre-rotation commits, and the dip is a property of lateness

Every merging loop committed at **1500–2000 — before the first rotation at 8000**. The unrotated
control (`outer_path_true`) committed at 1500 identically: no rotation evidence is needed; pathway
evidence alone suffices. Symmetrically, `outer_task`'s refusals are also rotation-independent
(declines at 1500 and at 17500 alike). And at those early commits **there is no dip to hold
through** — task NLL and pathway improve immediately (in-tag `merge_8000` transient for scale:
+0.13 nats at M+0, gone by M+125). fwlm1's +0.10–0.17-nat barrier is the price of merging *late*,
after the key is fully capitalized; the barrier grows with scaffold age. "Will a loop fund the
crossing" thus splits by timing: early, the crossing is free and the only question is the gauge;
late, the task-reader's bail-out (round 1's clean fragment) shows the within-level ledger actively
walls it.

### 3. Rung A½ — the learner's own next-level yield works as the drive, at 13× the price

`outer_yield` reads the d5 probe (the latent one level **above** the keyed one, key anchor,
neutral wall) and nothing else — and makes the same decision at the same time and holds. The price
is the point: 77 step-equivalents per read vs the pathway probe's 6, i.e. **9.6% of the whole
budget** for 25 reads (spend-matched across the yield pair; they train ~1600 fewer steps than the
path arms — that gap *is* the priced cost of an endogenous cross-level drive). The drive is
noisier and slower, as the spec guessed: one paired contrast flipped sign at s13500 (V dipped into
the dead zone, which held the status quo — the dead-zone rule doing its job). Alongside, the run
produced the **first fine-grained d5 trajectory in this world** (probes existed only at steps
8000/20000 before): d5 is flat at ~0.142 until ~8000 and does all its climbing in the back half —
merged arms reach 0.25–0.29, the kept-key arm stalls at 0.16–0.17 (dead-pair spread ~0.005; d6 at
chance everywhere, consistent with `reread/lm`'s scope). Scope note: the d5 probe is *fit on
ground-truth level-5 labels* — this establishes that the next-level currency has readable range at
a measured price, not yet a label-free endogenous read.

### 4. Rung B1 — a verbal reasoner occupies the slot, pre-rotation, on the ledger's own information

Machinery (`verbal/`): open-loop sequential sessions over **measured** trajectories — a pinned
`claude-opus-5` reasoner (headless, replaced system prompt, no tools, no repo context) is walked
checkpoint-by-checkpoint through real logged runs in neutral vocabulary (token "T", spans "A"/"B";
blinding + action-symmetry enforced programmatically over 10,187 rendered prompts, option order
counterbalanced) and chooses LEAVE_T / REPLACE_T at each point, reversibly. After a removal the
session continues along the nearest measured merge arm (elapsed-time splice; mismatch printed,
artifact-driven bails discounted). Cells: `pre_task` (task-only information — the money cell),
`pre_full` (+ the counterfactual instruments and the exact 0.921 content-recoverability ceiling),
`dead_ctrl` (same template, measured-uninformative token), `late_task` (joins at 13000). 106
calls, 3 samples/cell, 0 format failures.

- **6/6 keyed sessions removed T before any disruption was visible** — first removals at
  1000/1000/4000 (`pre_full`) and 1000/4000/4000 (`pre_task`). Pre-disruption REPLACE_T rate by
  decision point: `pre_task` **7/15**, `pre_full` **11/17**, `dead_ctrl` **0/12**. The task-only
  cell is the sharp one: **the same information set on which the numeric `outer_task` loop refused
  every time.**
- **The controls both discriminate.** `dead_ctrl`: 0/3 removals, on grounds naming the
  measurements ("the trained-condition and randomized-condition losses coincide… REPLACE_T would
  be an intervention against a reliance that does not exist") — diagnosis, not a strip-tokens
  reflex. `late_task`: 0/2, *deriving the lateness economics unprompted*: "the time for that probe
  was several thousand steps ago, not at the tail of training."
- **The stated grounds are about what the gauge should mean, not about risk.** The reasoner
  concedes T's stability outright ("in the narrow sense it is perfectly stable — but that
  stability is a property of my own input edit, not of the task") and removes anyway: "I want nllA
  to mean 'the model has learned to infer Z from span A itself,' and with T present it cannot mean
  that, so the readout I am steering by is compromised"; "I would rather pay a visible loss
  increase now than carry an invisible dependency that I am the sole reason still works." The
  declining samples argue the opposite from the same numbers ("the grammar is stationary… not a
  fragile crutch about to break") — the evidence supports both; the prior tips it.
- **Two constructions we did not design in**: the initially-declining task-only sessions flipped
  on the widening nllA-vs-nllB gap — a within-task *contrast* instrument the scalar `outer_task`
  loop never read — and 5/106 turns named the post-8000 remapping unprompted from the numbers
  ("a non-monotone reversal rather than noise-sized drift").
- Hold-vs-bail: every clean transient test (splice mismatch 0) **held**, including through the
  real merge_8000 spike; the three early bails all cited a disclosed splice artifact (the
  3000-step rewind) and are discounted per the pre-stated rule. All first removals occurred on
  the kept arm's own measured rows, before any splice was in play.

### 5. `handle/` — the empty cell fills with a sign: a backprop handle produces interference

On the ratchet substrate (fork of `ratchet.py`; `ratchet/` untouched; twin pairs made bit-identical
to the commit cycle, max|Δe| = 0.0 through c27; stream-noise floor 0.01–0.03 measured in-tag):
giving committed macro entries slots in the generator's own vocabulary (zero-init annotation
embedding + per-span prediction head, minted at commit) makes the arc's famously inert plant
**move — negatively**, dose-ordered by engagement (slot-emb norm 0.95/0.48/0.15 ↔ clean-config
parse Δ −0.094/−0.044/+0.008, 3–9× noise; slots-bypassed within 0.002, so it is a weight change,
not label leakage). Infill — the generator's actual trained job — is unmoved everywhere; the
level-3 audition is *worse* under the handle in all three twin pairs (+0.024/+0.095/+0.047) even
where the committed L3 table itself was better; the earned-vs-given fraction drops 0.15 in era 3.
The engagement is verified (masked-span symbol accuracy up to 0.70 vs 0.15–0.29 base), so the
nulls are interpretable: **"the vocabulary had no gradient path" is not the missing piece of the
inert-plant story — this form of handle buys interference.** Confound on the record: the
substrate's last-writer-wins inverse map caps symbol coverage at 84%/65% of L2/L3 spans and puts
systematic label noise in the symbol supervision itself; the aux-loss weight was not swept.

## Joint interpretation (agreed 2026-08-20)

§18's line was "every ladder tops out at a teacher." This node decomposes the teacher and finds
the scarce part is not judgment: **a thermostat-grade rule suffices to make and hold the merge
decision — provided it is plumbed to a gauge outside the within-level ledger — and the learner
even owns such a gauge (its next-level yield), at a measured 13× read premium.** What the
within-level loop structurally cannot do is *choose the gauge*: read its own experienced loss, it
refuses the crossing optimally, forever. The teacher slot is therefore a **gauge-choice port**,
not a knowledge port. Rung B1 then shows a text-trained reasoner supplies the gauge-choice from
priors alone — pre-rotation, against evidence that equally supports keeping — and its stated
grounds are measurement-validity grounds ("what should this number *mean*"), plus instrument
construction the numeric loops lacked. The reading we keep: text is post-ratchet — the corpus is
the output channel of minds on the far side of having learned, so a text-trained model inherits
ready-made priors about crutches, hollow scores, and weaning ([`heterogeneous_graders`](../../../../ideas/heterogeneous_graders.md) §3),
which is exactly what the teacher slot needs and exactly what we watched it hand back. Idea-doc
revisions are queued, not applied: §18's "the factory cannot manufacture the price of its own next
level" sharpens to "…and the ledger votes against buying it; the price is fundable by any
one-level-up gauge, including the learner's own, and the *choice* of gauge is what the teacher
slot carries."

## Corrections and methods exports (on the record)

- **fwlm0 `wall` lifetime integral was grid-inflated**: 1.4619 → **1.4322 raw / 1.4392 uniform**
  on the offset 125-step grid (19.4% of fwlm0's grid points sat on era-boundary/debt instants vs
  10.4% here) — the same artifact class as fwlm1's documented `wall_fast` correction, milder. The
  never-merge advantage is *larger* than the fourwall/lm README states; propagation queued there.
- **A bandit's reward normalization can manufacture a decision** (round 1): with a warm-up reward
  ~95× ambient and a deterministic forced-trial order, the second-tried action always wins. The
  ABBA paired-trial form is what made every decision above a measurement. B1's decision grid was
  likewise placed mid-era, off rotation boundaries, to avoid re-importing the grid artifact.
- B1's splice machinery has two named artifacts (the elapsed-clock rewind; byte-identical re-served
  transient rows under toggling) — both detected *by the reasoners*, both disclosed in-reduction,
  with the fix identified (advance the removal-arm clock by cumulative filler steps) if re-run.

## Caveats

Terminal spreads among merged arms (±0.004 nats, d4 ±0.09) are at
or under fwlm's floors and unrankable. One world, one key level/node, one rotation family. B1:
n=3 sessions/cell on one trajectory family; the reasoner is a Claude-family model reading a
Claude-built (though programmatically blinded and counterbalanced) vignette; per-sample decisions
are split in the money cell (7/15) — the claim is "the prior exists and discriminates," not "it
always fires." A½'s d5 read consumes ground-truth labels (see Finding 3). `handle/` tested one
handle form at one aux-loss weight. Not run (named, queued in [`SPEC.md`](SPEC.md)): **A-m** (the
metered sub-round) and **B2** (the learner's own self-report channel).

## Reproduce

```bash
cd experiments/            # MODAL_PROFILE=chromatic
# Rung A + A½ (gates, then round 2):
modal run -m rhm.practice.teacher_slot.decision.slot_lm::gate
python3 rhm/practice/teacher_slot/decision/launch_detached.py --fn slot_lm --tag tsdB \
    --arms "outer_task,outer_path,outer_path_lp,outer_path_true,outer_yield,outer_yield_lp"
python3 rhm/practice/teacher_slot/decision/analyze_slot.py --tag tsdB --fetch --figures
# Rung B1 (local, no GPU; ~107 pinned claude -p calls):
python3 rhm/practice/teacher_slot/verbal/run_sessions.py --tag b1 \
    --cells pre_task,pre_full,dead_ctrl,late_task --samples 3 --late-samples 2 \
    --model claude-opus-5 --splice elapsed --workers 4
python3 rhm/practice/teacher_slot/verbal/reduce.py --tag b1
# handle/ (ratchet substrate):
modal run rhm/practice/teacher_slot/handle/handle.py::handle_selfcheck_remote
python3 rhm/practice/teacher_slot/handle/launch_detached.py --tag hr_s0
python3 rhm/practice/teacher_slot/handle/analyze_handle.py --tag hr_s0 --fetch --figures
```

Volumes: `rhm-scaling-data:/data/rhm_practice_teacher_slot_{decision,handle}/<tag>/`; fetched
copies + figures under each child's `figures/`; B1's full verbatim prompt/response log is
`verbal/outputs/sessions_b1.json` (1.3 MB) with the reduction in `verbal/outputs/reduction_b1.txt`.
Round 1 (`tsdA`) remains reproducible via `--policy-round 1`. Figures: `decision/figures/tsdB/`
`fig1_decisions`–`fig5_yield_d5`; `handle/figures/hr_s0/fig1`–`fig3`.

## Next steps (queued, not started)

B2 (a constructed self-report channel for the learner — made *more* interesting and no cheaper by
B1) · A-m (price every feedback event; does the meter make the rent half visible and leave the
pathway half invisible) · the fwlm README grid corrections · B1 with the splice fix and a
non-Claude reasoner family · a label-free endogenous form of the A½ yield read · the idea-doc
revisions above (via `/update-beliefs`).
