# The full loop on RHM: inner + outer + forward model, and what drift actually buys

**Status**: built and run. Three conditions (allocation ladder, hierarchy climb, expansion 2×2), one certified
environment, two instrument bugs found and fixed mid-flight. Ladder is single rule-seed; the expansion 2×2 is
3 seeds × 3 grading worlds × 2 drift anchors. **Date**: 2026-07-28.
**Up**: [../README.md](../README.md) (directed_sculpting) · **Node**: [../../README.md](../../README.md) (rhm) · **Files**: [FILES.md](FILES.md)
**Idea doc**: [`ideas/adaptive_core_and_hierarchy_climb.md`](../../../../ideas/adaptive_core_and_hierarchy_climb.md) — this builds its §4 loop, tests §6/§11's climb and §5's missing cell.
**Direct parent**: [`mjc/on_policy/directed_on_policy/`](../../../mjc/on_policy/directed_on_policy/README.md) (E3, PR #4) — the loop being ported.
**Substrate**: [`../README.md`](../README.md) — the distractor DGP, support-fixed drift, and the repair-cost instrument this stands on.

---

## One-liner

The reward-free **relevance** signal ports and is a smoking gun — a value trained only on terminal task
success, never told which channel is which, separates real tokens from distractors **13–18×** and recovers
**76%** of a privileged oracle's advantage over uniform allocation. **Expansion turns out to be a property of
the grader's *type*, not of non-stationarity**: a grounded evaluative grader expands the belief ~1.7× in
effective dimension and ~15× in ballistic control while an endogenous dense grader caps *below the no-loop
floor*, and this holds identically in a static and a drifting world. Both of the idea doc's drift-dependent
predictions — that drift buys hierarchy-climbing, and that drift keeps expansion from exhausting — come back
**negative under instruments we had to repair twice**, each repair removing a bias that pointed toward the
positive result.

---

## 1. What got built

[`../README.md`](../README.md) left the substrate certified and the loop unbuilt: *"No arity-2 block FM, no DP
`k*` teacher, no six-policy ladder on this substrate. Everything here is the DGP and the instruments."* This is
the loop.

[`channel_env.py`](channel_env.py) teaches the sculpting apparatus (controller / generator / block FM / MC
value / beams, from [`../../rhm_sculpt_latent.py`](../../rhm_sculpt_latent.py)) to live on a multi-channel
sequence. Three changes were needed, and each is load-bearing:

1. **Per-block rendering tables.** Each channel edits through *its own* grammar. Noise blocks have none —
   regenerating one redraws its tokens iid, which is what makes them an irreducible target for the FM
   (`verify_distractors` P2, restated in the FM's currency).
2. **Mixture rendering.** The old move rendered a block's inferred feature as its *canonical* synonym. Under a
   frozen generator that is a fixed deterministic map, so support-fixed drift would be pure covariate shift and
   the FM would have nothing to re-adapt to. Drawing the synonym from the cell's *current drifted mixture*
   makes drift bite on the dynamics — `rhm_drift`'s "drift the encoding, not the content" taken literally. `d*`
   is unaffected: every synonym legally realises the same feature.
3. **Corruption confined to tree blocks.** Damage to a distractor cannot move `d*`, so it would be damage the
   task is structurally unable to see.

The **three taps** (§4) are one FM object read three ways, which is what makes this one system:
forward for control (`open_loop_beam`), residual-derivative to explore (`counterfactual_lp`), rolled forward
for relevance (`forecast_visits`).

### The back-compat gate

`env_check.py::fm_bisect` runs the same measurement through four conditions, one variable at a time:

| condition | `delta_cos` | `top1` | `rank_corr` |
|---|---|---|---|
| reference — the literal `rhm_sculpt_latent` pipeline, 8 blocks | 0.486 | 0.361 | 0.528 |
| **tree-only — *this* code path, same task** | **0.492** | **0.402** | **0.535** |
| + distractor channels (14 blocks) | 0.360 | 0.281 | 0.421 |
| + mixture rendering | 0.310 | 0.303 | 0.348 |

The reference reproduces the published 0.49 / 0.357 / 0.527, and this code path matches it. So the port is a
genuine back-compat pass, and the two design choices now carry price tags instead of hopes: the distractor
geometry costs ~27% of FM fidelity, making drift bite costs another ~14%. Both affordable.

## 2. The environment, certified before the loop ran on it

`verify_distractors` certified the DGP at the token-CE level and flagged that *"the FM-level LP the real loop
reads is a separate check."* [`env_check.py`](env_check.py) is that check (E1–E6, L=4 and L=5).

**E2 — the irreducible floor, measured rather than assumed.** Redraw the same `(x, k)` four times; the residual
around the conditional mean is the error no FM can beat. Acted-block, mixture rendering:

| | tree | structA | structB | noiseA | noiseB |
|---|---|---|---|---|---|
| **floor** | **0.189** | 0.272 | 0.461 | 0.502 | 0.500 |
| FM after 12k steps | 0.651 | 0.942 | 0.923 | 0.921 | 0.899 |
| headroom used | **+0.42** | +0.01 | +0.06 | +0.08 | +0.12 |

Ordered exactly as the ladder needs (noise > struct > tree), and **exactly 0.000** for the grammar channels
under deterministic rendering — E1 passes. The headroom row *is* the `lprog-only` trap: uniform collection
gives the tree 8/14 of the data and it burns 42% of its available headroom, while the struct channels sit
almost untouched with the most left to chase. Same asymmetry `verify_distractors` P4 found (tree satiates at
step 4000, structs at 7500), now in the currency the outer loop reads.

**E5 — the ballistic commit length.** Open-loop success is cleanly humped in commit length: `c1` 0.045,
**`c3` 0.092**, `c6` 0.043 — confirming the ~6-step veridicality bound from the length-gen arc, and setting
`chunk_len=3`. Reactive (re-grounded) control sits at 0.189 and is flat across policies: the near-blind grader
E3's own reactive control turned out to be.

## 3. E1 — the allocation ladder ([`ladder.py`](ladder.py))

Seven policies × 12 rounds under continuous drift, forking from one warm FM, on a bit-identical world sequence.
Monitor:collect **1.89×** against E3's 1.84×.

| policy | tree FM err ↓ | ballistic ↑ | →noise | →struct | →tree |
|---|---|---|---|---|---|
| oracle (privileged) | **0.676** | 0.110 | 2.0% | 2.0% | 96.0% |
| **visits_only** | **0.688** | **0.105** | 11.3% | 21.4% | 67.3% |
| value_satiety | 0.705 | 0.094 | 21.5% | 32.9% | 45.6% |
| lprog_only | 0.718 | 0.091 | 36.7% | 35.6% | 27.7% |
| value = lprog × visits | 0.720 | 0.086 | 21.5% | 34.0% | 44.4% |
| uniform | 0.727 | 0.083 | 40.0% | 40.0% | 20.0% |
| error_only | 0.736 | 0.078 | 39.8% | 42.8% | 17.4% |

### The smoking gun: the value system devalues distractors

The MC value is trained only on terminal possible-set success. It is never shown a channel label. It
nonetheless separates them:

| | tree | structA | structB | noiseA | noiseB |
|---|---|---|---|---|---|
| value sensitivity \|ΔV\| (in-loop, mean over rounds) | **0.431** | 0.024 | 0.027 | 0.029 | 0.033 |
| forecast visits (the `p` tap) | **0.673** | 0.164 | 0.065 | 0.064 | 0.034 |
| ground-truth best Δ`d*` | **+1.702** | 0.000 | 0.000 | 0.000 | 0.000 |

13–18× separation; the value's top-1 move is a tree block **94%** of the time; its move ranking correlates
**+0.31** with the exact DP's. Replicates at L=5 (0.61 vs 0.05, 94.5%, +0.30). And it *pays*: `visits_only`
recovers **76%** of the oracle's advantage over uniform on the sighted grader with zero DGP knowledge, matched
by the behavioural grader (0.105 vs oracle 0.110 vs uniform 0.083). `error_only` is the worst arm and pours 40%
of its budget into the irreducible channels — the noisy-TV trap, reproduced.

### The negative: our reducibility tap is inverted, and it costs `value` the ladder

E3's headline ordering does **not** reproduce — `value` loses to `visits_only`. The reason is diagnosable, and
it is not about relevance. Mean counterfactual LP per channel:

| tree | structA | structB | noiseA | noiseB |
|---|---|---|---|---|
| **+0.044** | +0.053 | +0.057 | +0.064 | **+0.066** |

Lowest on the target, highest on the irreducible channels. A fixed-budget counterfactual fit measures
*marginal improvement*, and at this point in training marginal improvement is dominated by **data starvation**,
not by reducibility: uniform warm-start leaves the 2-block noise channels at error 0.92 against a floor of 0.50
— a large predictable-mean component nobody has fitted — so probing there yields the biggest held-out drop. The
tap answers *"how starved am I here"*, not *"how reducible is this"*. The noisy-TV **filter** became the
noisy-TV **attractor**.

Downstream signature: tree-share standard deviation across rounds is 0.059 for `visits_only` and **0.309** for
`value`, which swings between 1% and 93%. Multiplying a clean signal by an inverted noisy one destroys it.

**This also makes the satiety rung untestable as run.** `value_satiety` edges `value` (0.705 vs 0.720) but
sends the *identical* 21.5% to the irreducible channels, so §10's specific prediction — that satiety reduces
the irreducible-region leak — gets no support. But it was being asked to suppress exactly the channels the
drive was wrongly attracted to. Confounded, not negative.

## 4. E2 — the hierarchy climb ([`climb.py`](climb.py)): no climb

Ten drift events with a *trainable* belief under the evaluative grader, swept over which level drifts, against
a `nodrift` matched-compute arm. Depth gain relative to that control, at per-event matched magnitude
(0.539 / 0.498 / 0.566 nats — within 7%):

| arm | Δd1 | Δd2 | Δd3 | Δd4 |
|---|---|---|---|---|
| surface | +0.005 | −0.005 | −0.011 | −0.026 |
| mid | −0.011 | −0.020 | −0.007 | +0.003 |
| root | −0.008 | +0.005 | −0.023 | −0.013 |

All within ±0.026, no ordering, no dose-response. The depth rise is real and large — `nodrift` alone goes
d3 0.645 → 0.711, d4 0.769 → 0.827, PR 9.1 → 11.2 — but it is caused by **the evaluative grader, not by
drift**. This is §11's adaptive-core-no-climbing, reproduced on a substrate that *does* have a hierarchy to
climb into, which makes it a stronger negative than `mjc/expansion`'s: there, nothing was available to climb.

**Repair-per-event never resolved** and is reported as out of range rather than as a number: 6–8 of 10 rounds
in range, per-arm slopes −0.004 / +0.015 / −0.257, individual values swinging −3.8 to +1.16. At these
magnitudes the instrument has no resolution.

## 5. E3 — expansion ([`expansion.py`](expansion.py)): grader type is everything, drift is orthogonal

§5's 2×2 with the missing cell filled. 3 seeds, entropy-matched base, graded in three worlds.

| condition | PR r1 → r10 | last-3 PR slope | d3 | d4 | ballistic (base) | fresh-FM top1 (base) |
|---|---|---|---|---|---|---|
| frozen_static | 6.99 → 6.93 | +0.003 | 0.583 → 0.572 | 0.660 → 0.619 | 0.024 | 0.173 |
| dense_static | 6.31 → 6.61 | +0.011 | 0.563 → 0.557 | 0.634 → 0.626 | 0.007 | ~0.20 |
| **evaluative_static** | **9.03 → 11.56** | +0.080 | 0.644 → 0.677 | 0.747 → **0.812** | **0.369** | **0.554** |
| frozen_drift | 6.93 → 6.84 | −0.006 | 0.584 → 0.571 | 0.671 → 0.628 | 0.027 | 0.177 |
| dense_drift | 6.32 → 6.62 | +0.041 | 0.569 → 0.546 | 0.645 → 0.618 | 0.008 | ~0.20 |
| **evaluative_drift** | **8.94 → 11.73** | +0.077 | 0.635 → 0.687 | 0.766 → **0.815** | **0.380** | **0.547** |

Round-0 belief: PR 7.04 ± 0.05, d4 0.665 ± 0.023.

1. **The grounded evaluative grader expands, hugely.** PR 7.0 → ~11.6 (1.7× the belief's effective dimension),
   d4 +0.15, transferable fresh-FM plannability 0.17 → 0.55 (**3×**), ballistic control 0.024 → 0.37
   (**~15× the no-loop floor, ~50× the dense grader**). Stable at 3 seeds, in all three grading worlds, under
   both drift anchors.
2. **The endogenous dense grader caps *below the floor* and degrades depth.** PR 6.6 against frozen's 6.9;
   d4 falls in both worlds; ballistic collapses to 0.007. This is Stage 5's `fm_cotrain` result (PR flat
   6.7 → 6.8, transferable plannability *down* 0.357 → 0.304) reproduced — and now with a drift arm showing
   drift does not rescue it.
3. **Drift is orthogonal.** Evaluative − frozen lift in the *same* world: ΔPR +4.63 ± 0.46 (static) vs
   +4.89 ± 0.73 (drift); Δd4 +0.193 ± 0.016 vs +0.187 ± 0.015. Statistically identical. §5's stated falsifier —
   *"falsified if the evaluative/drifting cell also exhausts in one pass"* — is met: both cells have the same
   PR slope at r10 (+0.080 vs +0.077) and land at the same place.

**The exhaustion law is about compression, and drift is not the escape.** The three prior discoveries of
"compounding requires a moving frontier" were all measured with a dense grader in the loop, and swapping the
grader type is what moves the frontier. Making the *world* move does not add to it.

## 6. Two instrument bugs, and why they matter more than the results

Both were found by chasing an apparent positive, and both had biased toward it.

### The own-world artifact

A drift-trained arm was being graded in the world it had just spent 20 OU steps fitting, while the static arm
was graded in a pristine one. Grading every arm, frozen, in a common world:

| grading world | evaluative_drift − evaluative_static, Δballistic | t |
|---|---|---|
| **own** | **+0.0243 ± 0.0196** | **+2.15** |
| pristine | +0.0081 ± 0.0199 | +0.71 |
| novel (held-out realisation) | +0.0098 ± 0.0214 | +0.79 |

~65% of the apparent advantage was the grading venue.

### The uniform anchor

The OU walk anchored at `θ = 0` — the uniform mixture, which is the **maximum-entropy point of the simplex**.
Since `KL(w‖uniform) = log m − H(w)` exactly, a drift event destroys precisely as much rendering entropy as the
KL it creates. So every drifted world has fewer synonyms in play per edit, a lower stochasticity floor, and is
**mechanically easier to plan in** — measured at a fixed controller, tree floor 0.160 (drifted) vs 0.195
(uniform), ~18%. A static arm parked at uniform is not a control; it is a handicap.

*The sign is an artifact of the anchor, not a property of drift.* Anchor at uniform and drift can only reduce
synonymity; anchor near a deterministic mixture and it could only increase it. Drawing the pre-drift world from
the same stationary law the walk lives in (`init="stationary"`, mechanically a σ=0 drift state) makes the two
arms' worlds exchangeable in entropy, so only the *motion* differs.

With both removed, at 3 seeds:

| grading world | Δballistic | Δtop1 |
|---|---|---|
| base | +0.0107 ± 0.0458 (t=0.40) | −0.0070 (t=−1.03) |
| novel | +0.0080 ± 0.0433 (t=0.32) | −0.0099 (t=−1.13) |

Per-seed ballistic deltas `[+0.017, +0.053, −0.038]` — the sign flips. Zero. And the symmetric question no
earlier design asked: a *static*-trained agent does not degrade in a novel drifted world either (base→novel
costs static −0.024, drift −0.027). **Drift-training bought no transfer in either direction.**

### The third appearance of the same identity

`KL(w‖uniform) = log m − H(w)` has now bitten this node three times: it is why raw CE is *exactly* blind to
support-fixed drift (`../README.md` Trap 1), why the drifted world is easier to control in, and why a
uniform-anchored control is not a control. It is worth treating as a standing property of this drift primitive
rather than a recurring surprise.

### And a third bug, in the readout rather than the design

`calibrate_sigma` matches KL from uniform *at stationarity* — accumulated displacement, not the size of an
event. The climb sweep, whose entire purpose was holding magnitude fixed, first ran at per-event magnitudes of
0.78 / 1.00 / **3.49** nats across levels. `calibrate_sigma_event` targets the event directly and lands all
four levels at 0.600. Both climb corrections removed biases pointing *toward* a climb, so §4's null survived a
thumb on the scale in its disfavour.

## 7. What this establishes

1. **A reward-free relevance signal ports from mjc to RHM and is strong.** The learned value devalues
   structurally irrelevant tokens 13–18× without ever being told which they are, agrees with the exact DP, and
   allocating by it alone recovers 76% of a privileged oracle.
2. **Expansion is a property of grader *type*.** Grounded-evaluative expands (PR 1.7×, plannability 3×,
   control 15×); endogenous-dense caps below the no-loop floor and degrades depth. Reproduced at 3 seeds in
   3 worlds under 2 anchors.
3. **Drift buys neither climbing nor sustained expansion.** Adaptive core only, on a substrate that has a
   hierarchy — with both instruments corrected in the direction that disfavours the null.
4. **A metered, on-policy loop is reproducible on RHM at E3's own subsidy ratio** (1.89× vs 1.84×), with the
   ballistic/reactive split behaving exactly as the length-gen arc predicts.

## 8. Caveats

- **The ladder's `value` rung is not a fair test of `lprog × visits`.** The LP tap is inverted by data
  starvation (§3). `value` vs `visits_only` should be re-run with a floor-relative-headroom LP before anyone
  concludes relevance-times-reducibility loses to relevance alone. `env_check` already measures the per-channel
  floor this needs.
- **The satiety rung is confounded**, for the same reason. §10's leak prediction is untested, not refuted.
- **Repair-per-event has no resolution** at these magnitudes and is reported as out of range.
- **The ladder and climb are single rule-seed**; only the expansion 2×2 is 3-seeded. The ladder's ordering is
  large relative to its round-to-round spread, but not hardened.
- **`own`-world grading remains favourable by construction** even after the entropy fix (the arm just fitted
  that world); `base` and `novel` are the fair columns and are the ones quoted.
- **L=4, not the L=5 the port was specced for.** L=5 was run through `env_check` and agrees on every DGP-level
  property (floors ordered, value separation 0.61 vs 0.05, top-1 share 94.5%); L=4 was chosen because the whole
  sculpting arc is calibrated there and the loop is ~3× cheaper.
- **PR is not certified as "the frontier"** — the idea doc's own warning. It is reported alongside depth
  throughout, and no PR rise with flat depth is read as expansion.

## 9. Open items

1. **Fix the LP tap** — normalise counterfactual improvement by floor-relative headroom, re-run the ladder.
   The one change that makes `value` vs `visits_only` decidable, and re-opens the satiety rung.
2. **The in-tree irreducible cell** is still unplaced (inherited from `../README.md` open item 1), so
   `visits_only` and `value` are not separable by construction here — which is exactly E3's own
   *"you only go where you reach"* limitation.
3. **Why does the dense grader land *below* the no-loop floor?** Stage 5 saw it cap; here it actively degrades
   depth and control in both worlds. Worth understanding before "endogenous predictability pressure is merely
   insufficient" is restated as "actively harmful".
4. **Does anything make the climb appear?** The negative is now well-instrumented, so the productive question
   is what the argument was missing — the idea doc's own answer (§12) is a satiety-gated preference for the
   next level up, which this loop has no mechanism for.

## Reproduce

```bash
cd experiments/            # MODAL_PROFILE=chromatic
python3 -c "from rhm.verify_backcompat import verify; verify()"                                   # the gate
modal run rhm/directed_sculpting/full_loop/env_check.py::env_check --quick                        # smoke
modal run --detach rhm/directed_sculpting/full_loop/env_check.py::fm_bisect --tag l4_v1           # back-compat
modal run --detach rhm/directed_sculpting/full_loop/env_check.py::env_check --tag l4_v2 \
    --tree-depth 4 --struct-depths 2,2 --noise-blocks 1,1 --edit-budget 6 --chunk-sweep 1,2,3,6
modal run --detach rhm/directed_sculpting/full_loop/ladder.py::ladder --tag l4_v1 \
    --rounds 12 --n-eval 1024 --mon-n 3000 --collect-budget 8192
modal run --detach rhm/directed_sculpting/full_loop/climb.py::climb --tag em_v1 --rounds 10
for s in 1 2 3; do                                    # launch each seed as its own client
  modal run --detach rhm/directed_sculpting/full_loop/expansion.py::expansion --tag em_s$s \
      --seed $s --rounds 10 --grade-every 3 --fresh-fm-steps 5000 --n-eval 1024
done
```

Results JSON on the `rhm-scaling-data` volume under `directed_sculpting/`, mirrored to
[`figures/`](figures/). `--entropy-matched-base false` and the default `calibrate="stationary"` reproduce the
pre-fix runs (`expansion_xw_s*`, `climb_l4_v1`) exactly.
