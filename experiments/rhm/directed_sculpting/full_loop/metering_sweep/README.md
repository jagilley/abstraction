# The metering sweep: what the meter is worth, and what it costs to look

**Status**: complete — **96 GPU runs, 32 operating points × 3 seeds**, two rounds each
pre-registered before its own launch and gated before its own read. **Date**: 2026-08-01.
**Up**: [../README.md](../README.md) (full_loop) · **Node**: [../../../README.md](../../../README.md) (rhm) · **Files**: [FILES.md](FILES.md)
**Idea doc**: [`ideas/meta_learning_under_metered_data.md`](../../../../../ideas/meta_learning_under_metered_data.md)
§Predictions bullet 1 — *"Falsified if the uniform→oracle gap does not shrink toward zero as the
monitor:collect ratio goes to zero. That is the whole content of the metering claim."*
**Design doc**: [DESIGN.md](DESIGN.md), written before the first run.

---


## One-liner

The idea doc's own load-bearing falsifier had never been run, because **the meter had never been a
variable** — every ladder in the repo reports exactly 1.78125×. Running it needed a design change
first: the `Meter` is **inert** (built with `budget=None`, read by nothing), so the "config change"
the doc calls cheap is a *provable no-op*; and `uniform`/`oracle` are both **tap-blind**, so the
prize can respond to monitoring only through the collection it displaces. With price routed through
the budget, the answer arrives in three pieces.

**The doc names the wrong variable.** Two matched-ratio pairs — identical monitor:collect, 3.6–4×
the budget — differ by **+0.023 ± 0.004** (t = +5.23) and +0.016 ± 0.005, while 4× the monitoring at
fixed budget leaves the prize **bit-identical to ten decimal places**. It is the *budget*, not the
ratio. Drift magnitude does nothing either: a clean 2×2 gives a drift effect of **±0.0009** across
30× while budget moves the prize +0.023 at both drift levels.

**The doc's prediction holds, but only at matched compute.** The loop **overfits within the round** —
16× the gradient steps on the *same* data costs **+0.117** tree error — so sweeping the budget at
fixed epochs silently scales compute too, and that confound made the prize look like it *grew* as
data got cheap. Hold gradient steps fixed and `E(d)` flattens normally toward a capacity floor while
the prize becomes **humped, turning over at the abundant end**: −0.0190 (t = −6.10) and −0.0221
(t = −21.67) at two independent compute levels, 3/3 seeds each. Iso-compute is also the scaling the
doc's own analogy implies — *"train on all of it"* vs curation is a choice made at a fixed **training**
budget over a corpus of varying size.

**And once a tap is priced, only the cheap one is worth buying.** Charging each arm for the
monitoring its own drive reads, `visits_only` pays 512 sequences/round — **2.2%** of the total — and
keeps **87%** of what a privileged oracle gets for free, while `reducible_only` pays 14080 and
recovers **4%**, falling to −0.0001 at the largest total. **~1200×** in cost-effectiveness. The whole
of the reducibility tap's apparent 13–18% was the equal-charge subsidy.

## What this establishes

1. **The metering claim survives, restated.** Curation's value does decline as data becomes
   abundant — at fixed training compute, the scaling the claim's own analogy requires. What fails is
   the *quantity* it names: monitor:collect ratio is not the operative variable, and neither is drift.
2. **The prize was never one thing.** It splits exactly into **interference** (training on junk
   damages the shared-parameter model) and **data quantity**. Interference falls from +0.047 to
   **+0.008** — 9% of the prize — by `B` = 262144, so the doc's intuition is fully vindicated *for
   that half* and the two halves move oppositely along the fixed-epoch diagonal.
3. **A priced tap is a different object from a subsidised one**, and this is the node's sharpest
   result. It also puts the first number on the idea doc §6 claim that the cheap curation is the weak
   curation — here the cheap tap is the **strong** one.
4. **This node's ladder is over-trained.** `fm_epochs=8` costs 0.03–0.07 per arm against 1–2 epochs
   and inflates the headline prize ~17%. **Every published contrast and the ordering survive** the
   re-run (§18), so the science holds and the levels should be requoted.
5. **A meter that is actually a variable**, a pointwise noise-floor arm, and a per-arm monitoring
   bill — all back-compat, all reusable on any ladder in this node.

## How to read this

**§1–§10 are round 1** and reach a headline that round 2 retracts: they swept the meter exactly as
the doc names it and found the prize *rising*. They are kept in full because the design argument in
§1, the gate discipline in §2, the matched-ratio result in §4 and the decomposition in §6 all stand
— only §3's direction and §5's "saturation is unreachable" scoping are superseded, and each says so
inline. **§11–§18 are round 2**: the confound, the controlled curve, the refuted premise, the priced
ladder, and the over-training re-run. If you want one section, read **§15**.

---

# Round 1 — the meter as the doc names it

## Round 1's one-liner, as written

*(Left as written, before E4 found the confound. The two paragraphs that survive round 2 are the
matched-ratio result and the interference/data decomposition; the direction of the headline does
not.)*

The metering claim's own load-bearing test had never been run because **the meter had never been a
variable** — every ladder in the repo reports exactly 1.78125×. Run properly, it comes back
**negative with a direction**: the uniform→oracle prize rises **+0.0043 ± 0.0008 per doubling of
collection budget** (t = +5.31, 3/3 seeds, +0.026 across the sweep against a measured floor of
~0.005), where the doc predicts it should fall to zero. Two **matched-ratio pairs** — identical
monitor:collect, 3.6–4× the budget — differ by +0.023 ± 0.004 (t = +5.23) and +0.016 ± 0.005
(t = +3.18), while 4× the monitoring at fixed budget leaves the prize **bit-identical to ten
decimals**. So the doc's stated quantity, the *ratio*, is not what the prize responds to; the
*collection budget* is. The mechanism is visible in one figure the sweep produces for free: pooling
all 66 (arm, budget) cells, tree error against tree-data-per-round **steepens** rather than flattens
(−0.011 → −0.025 per doubling over a 317× span), so the saturation shoulder the prediction lives on
is ~16 doublings away and **unreachable here**. Finally, the prize is not one quantity: it splits
exactly into **interference** (falling with budget, −0.004/doubling) and **data quantity** (rising,
+0.013/doubling). The doc's intuition is right about the first and wrong about the second.

---

## 1. Why this had never been run, and what had to change first

The idea doc calls this test *"cheap (the meter is a config change on an existing ladder)"*. It is
not, and the reason is the first finding.

**The meter is inert.** [`../ladder.py`](../ladder.py) builds `Meter()` with `budget=None`, so
`_check` never fires, and `meter.ratio` appears only in the results JSON. **Nothing in the loop
reads it.** A bare charge multiplier on monitoring would change the reported ratio and *nothing
else* — every arm bit-identical. Sweeping it alone is guaranteed to return an exactly flat curve
for reasons that have nothing to do with metering economics, and reading that as a falsification
would be an instrument defect pointing at a negative.

> **A price is only a price if paying it costs you something else.** For monitoring's price to be
> real it must *displace collection*.

**And `uniform` and `oracle` are both tap-blind.** `allocate` hands `uniform` a constant `ones(5)`
and `oracle` a constant one-hot; neither reads `errors`, `lprog`, `visits` or `reducible`. So the
prize **cannot** depend on monitoring quality — only on the collection the monitoring bill leaves
behind. That is what makes the price/information separation exact here rather than approximate, and
it turns the amount-sweep into a *control* on the design instead of a confounded variant of it.

So the headline sweep holds total per-round spend `T = λM + B = 22784` and monitoring quantity
`M = 14592` fixed, and lets the price `λ` buy itself out of the collection budget:
`B = T/(1+r)`, `λ = rB/M`. At `r = 1.78125` this gives `λ = 1.0` and `B = 8192` **exactly** — the
published anchor, reproduced from inside the new parameterisation rather than asserted.

| experiment | what varies | what is held | points |
|---|---|---|---|
| **E1-A** price at fixed total spend | `λ`, and `B = T − λM` | `T`, `M`, information | 7 (`r` 0.0625 → 22) |
| **E1-B** abundance | `B` | `λ = 1`, `M` | 2 (`B` 32768, 65536) |
| **E2** information *(control)* | `mon_n`/`floor_n`/`forecast_n` | `B`, `λ` | 2 (`M` ×0.25, ×4) |

Arms everywhere: `uniform`, `reducible_only`, `visits_only`, `value_red`, `oracle`, and
**`oracle_dup`** — a bit-for-bit duplicate of `oracle` that measures the instrument noise floor *at
each operating point*, since the floor is not budget-invariant.

## 2. Gates, all read before the curve

| gate | result |
|---|---|
| **C0** back-compat | `rhm.verify_backcompat.verify()` passes; `mon_price=1.0` takes the original `int(n)` path |
| **C1** the meter moves as intended | realised ratio hits target to **0.00%** at all 11 points; monitoring quantity **identical across every E1 point** (one distinct config) |
| **C2** the anchor reproduces | prize **+0.0633 ± 0.0031** (per seed +0.0695/+0.0594/+0.0611) against published `ladder_fix_s*` **+0.0629** — Δ **+0.0004** |
| **C3** pointwise noise floor | `\|oracle_dup − oracle\|` = **0.0018 – 0.0090**, mean ~0.005; `partial_hetero`'s 0.004 sits inside that range |
| **C4a** liveness | **11/11** points: 6-arm spread **7.8–43×** the pointwise floor |
| **C5** world intact | best Δ`d*` **exactly 0.000** off-tree, +1.5–1.8 on-tree, every point; warm-FM `delta_cos` 0.292–0.374 |

**C4's second clause was mis-specified, and this is recorded rather than quietly dropped.** It was
pre-registered as *"`oracle`'s tree error must fall from round 1 to round 12, else the readout is
dead"*. It is not a liveness test. This is a continuous-**drift-maintenance** task: error sits at a
steady state set by damage against repair, so a rising error means repair is losing to drift, not
that the instrument is blind — and the arms still separate by 0.06–0.09 at those points. It is
reported instead as a **regime diagnostic**, and it locates a real boundary:

| regime | points | `oracle` r1 → r12 |
|---|---|---|
| **drift-limited** (repair loses) | `B` ≤ 8192, and `r0p25` | −0.005 to −0.048 |
| **data-limited** (repair wins) | `B` ≥ 15189 | +0.000 to +0.043 |

The headline slope is reported below **both ways** — all 9 points and C4a-live points only — and is
numerically identical, so nothing turns on the reclassification.

## 3. The headline (round 1) — SUPERSEDED by §12

> This direction is a compute artefact; see §11. The absolute levels and the
> gate results below stand.

E1-A, price at fixed total spend. `uniform` and `oracle` absolute levels are shown because the
design doc makes them the saturation-vs-starvation discriminator.

| `r` | `B` | uniform | oracle | **prize** | sem | floor |
|---|---|---|---|---|---|---|
| 22.0 | 991 | 0.7688 | 0.7007 | **+0.0681** | 0.0047 | 0.0018 |
| 8.0 | 2532 | 0.7477 | 0.6892 | **+0.0585** | 0.0102 | 0.0071 |
| 4.0 | 4557 | 0.7477 | 0.6845 | **+0.0632** | 0.0074 | 0.0060 |
| **1.78125** | **8192** | 0.7371 | 0.6738 | **+0.0633** | 0.0031 | 0.0084 |
| 0.5 | 15189 | 0.7349 | 0.6585 | **+0.0764** | 0.0050 | 0.0033 |
| 0.25 | 18227 | 0.7305 | 0.6587 | **+0.0718** | 0.0036 | 0.0025 |
| 0.0625 | 21444 | 0.7241 | 0.6487 | **+0.0753** | 0.0020 | 0.0090 |

Pooling E1-A and E1-B gives 9 points over **66×** in budget (991 → 65536), and the abundance limb
extends the trend: `ab32768` **+0.0860**, `ab65536` **+0.0881**.

> **Prize vs log₂(collection budget): +0.00434 ± 0.00082 per doubling, t = +5.31, 3/3 seeds**
> (per-seed +0.00491 / +0.00537 / +0.00273), a change of **+0.0262** across the span against a
> pointwise floor of ~0.005.

Read in the doc's own currency, within E1-A: prize vs log₂(ratio) is **−0.00164 ± 0.00070,
t = −2.33, 0/3 seeds positive**; restricted to the doc's predicted limb (`r ≤ 1.78`) it is
**−0.00216 ± 0.00065, t = −3.31, 0/3 positive**. The prediction requires a *positive* slope there.

**Both absolute errors fall together on the low-`r` limb** (uniform 0.7688 → 0.6885, oracle 0.7007
→ 0.6004), which is the pre-registered signature of the saturation side, not the starvation side.
The learner really is getting less starved. The prize grows anyway.

## 4. The ratio is not the operative variable

The sharpest test, built in from the start: two points at (near-)identical monitor:collect and very
different budget, plus the reverse.

| contrast | `r` | `B` | prize | difference |
|---|---|---|---|---|
| `info0p25` vs `ab32768` | **0.4453 both** | 8192 vs 32768 | 0.0633 vs 0.0860 | **+0.0227 ± 0.0043, t = +5.23** |
| `r0p25` vs `ab65536` | 0.250 vs 0.223 | 18227 vs 65536 | 0.0718 vs 0.0881 | **+0.0163 ± 0.0051, t = +3.18** |
| `r1p78125` vs `info4` | 1.781 vs 7.125 | **8192 both** | 0.0633 vs 0.0633 | **exactly 0.0000** |

The third row is P5, the design's own control, and it holds **structurally rather than
statistically**: at ×0.25 and ×4 monitoring, `uniform` and `oracle` are **bit-identical to ten
decimal places** (0.7315856467 / 0.6621067673 on seed 1 in all three) while `visits_only` moves.
That is §1's tap-blindness argument confirmed in the data, and it certifies that E2 changed only
what it was meant to.

**So the ratio the doc names is not what the prize responds to.** Hold it fixed and quadruple the
budget: the prize moves by a third of itself. Hold the budget fixed and quadruple the monitoring:
nothing moves at all. This correction is independent of which way the curve goes.

## 5. Why the doc's limb looked unreachable — SUPERSEDED by §11–§12

> The steepening here is the fixed-epoch diagonal, not a learning curve. At
> matched compute the curve flattens and saturation *is* reached.

The sweep's arms overlap heavily in **tree transitions per round**, `d = share_tree × B`, across
budgets. Pooling all **66 (arm, budget) cells** gives tree error against `d` over a **317× span**
(198 → 62915). Since `oracle`/`uniform` is a fixed **4.8× data ratio at every point**,
`P(B) = E(0.20B) − E(0.96B)` exactly, so `E` determines the whole prize curve.

| slope of `E` per doubling of tree data | value |
|---|---|
| all 66 cells | −0.0175 |
| starved half (`d` ≤ 6240) | **−0.0106** |
| abundant half (`d` > 6240) | **−0.0254** |

**The curve steepens rather than flattens.** That is the opposite of approaching saturation: each
doubling buys *more* at the abundant end, which is exactly why the prize grows. At the abundant-end
slope, reaching `env_check` E2's measured aleatoric floor of **0.189** from the best cell (0.6004)
needs **~16 more doublings** of tree data (~7.6 × 10⁴×). **The saturation shoulder on which the
doc's prediction lives is not reachable in this apparatus, by a wide margin.**

The steepening has a mechanism already in hand: it tracks §2's regime boundary. In the
drift-limited regime the stale-model ceiling compresses every arm together; escaping it lets
allocation differences open up. `E` is plausibly S-shaped in log-data with compression at *both*
ends, and this sweep sits on the rising middle.

*(An earlier version of this check tested only |curvature| and reported "the shoulder is in range".
Sign matters; the bug is recorded in [FILES.md](FILES.md) and was caught by the gate ordering.)*

## 6. The prize is two effects moving in opposite directions

`E(d)` **does not collapse across arms** — the check the design doc built in as
self-falsifying. At matched tree data, `uniform` carries systematically more tree error:

| `d` | cell | err | vs `d` | cell | err | Δ |
|---|---|---|---|---|---|---|
| 911 | `uniform`@r4 | 0.7477 | 951 | `oracle`@r22 | 0.7007 | **−0.0470** |
| 4289 | `uniform`@r0p0625 | 0.7241 | 4375 | `oracle`@r4 | 0.6845 | **−0.0395** |
| 6554 | `uniform`@ab32768 | 0.7196 | 6398 | `value_red`@r1p78125 | 0.6845 | **−0.0351** |

So tree error is not a function of tree data alone: the other 80% of `uniform`'s budget trains the
*same shared-parameter FM* on distractor transitions, and that costs. The prize therefore splits,
with `E_oracle` interpolated per seed from that seed's oracle cells (residual **exactly 0.0000**):

```
P(B) = [E_uniform(0.20B) − E_oracle(0.20B)]  +  [E_oracle(0.20B) − E_oracle(0.96B)]
              interference                             data quantity
```

| `B` | prize | interference | data quantity | interference share |
|---|---|---|---|---|
| 8192 | +0.0633 | +0.0430 | +0.0203 | **68%** |
| 15189 | +0.0764 | +0.0475 | +0.0289 | 62% |
| 21444 | +0.0753 | +0.0394 | +0.0360 | 52% |
| 32768 | +0.0860 | +0.0425 | +0.0435 | 49% |
| 65536 | +0.0881 | +0.0274 | +0.0607 | **31%** |

> interference **−0.00402 per doubling of budget**; data quantity **+0.01290**.

**The doc's intuition is correct about one component and wrong about the other.** The value of
*not training on junk* does decay as data gets cheap, exactly as "the meter is off ⇒ train on all of
it" says. But over the reachable range the *data-quantity* component grows faster, and the sum
rises. What the ladder calls "the uniform→oracle prize" was never one thing, and the metering claim
was implicitly a claim about only one of its halves.

## 7. The taps

Recovery of the oracle prize, as a fraction, at each point:

| | `r`=22 | 8 | 4 | **1.78** | 0.5 | 0.25 | 0.0625 | ab32768 | ab65536 |
|---|---|---|---|---|---|---|---|---|---|
| `visits_only` | 97.0% | 92.5% | 79.0% | **77.6%** | 77.6% | 88.8% | 78.5% | 81.6% | 84.5% |
| `value_red` | 88.2% | 76.5% | 80.7% | **83.7%** | 83.6% | 86.9% | 80.6% | 85.0% | 83.3% |
| `reducible_only` | 33.3% | 16.2% | 21.7% | **13.5%** | 22.4% | 22.2% | 13.3% | 18.3% | 18.2% |

The anchor's 77.6% / 13.5% sit a little under the published 84% / 18%, within the arm-order
sensitivity the floor arm measures (0.0084 there). **E2 is where the taps behave as designed** (P6):
at fixed budget, scaling monitoring ×0.25 → ×1 → ×4 moves `reducible_only` **11.1% → 13.5% →
17.7%** while `visits_only` barely moves (75.4% → 77.6% → 77.8%). The **reducibility tap is
monitoring-resolution-limited and the relevance tap is not** — consistent with the floor
measurement being the expensive, noisy half (`floor_n × floor_draws` is 53% of the monitor bill).

## 8. What round 1 established (see the top of this file for the final list)

1. **The metering claim's load-bearing falsifier fires.** The uniform→oracle prize does not shrink
   toward zero as the monitor:collect ratio falls; it grows, t = +5.31, 3/3 seeds, over 66× in
   budget, with both absolute errors falling — i.e. on the saturation side, not the starved side.
2. **The named quantity is the wrong one.** Matched-ratio pairs differ by up to +0.023 (t = +5.23);
   matched-budget pairs at 4× monitoring differ by exactly zero, bit-for-bit. **Budget, not ratio.**
3. **The doc's predicted limb is not reachable on this ladder**, and now that is a *measured*
   statement: `E(d)` steepens over 317×, leaving ~16 doublings to the aleatoric floor. A flat or
   rising prize here is scoped by that, and the claim survives as an asymptotic one.
4. **"Curation's value" is two effects with opposite meter-dependence** — interference (−0.004/
   doubling) and data quantity (+0.013/doubling) — which the ladder had been reporting as one
   number. This is the most transferable output here.
5. **A meter that is actually a variable**, plus a pointwise noise-floor arm, both back-compat and
   reusable on any ladder in this node.

## 9. Caveats

- **The meter still does not bind inside the loop.** Price acts only through the budget the
  experimenter hands it. That is the honest implementation of the economics, but it means this
  measures *"what does a smaller collection budget do"* wearing the right label — which is the only
  channel through which price can act here at all (§1).
- **Optimizer steps scale with budget** (`n_steps = fm_epochs × B / batch_size`, `fm_epochs` fixed
  at 8), so "more data" is also "proportionally more gradient steps". Both arms get the same at
  every point, so the *prize* contrast is clean, but absolute levels move partly for a compute
  reason and §5's `E(d)` inherits that.
- **The `E(d)` decomposition rests on interpolating `E_oracle`**, and is reported as invalid at the
  three points where `uniform`'s `d` falls below the lowest oracle cell (`r4`, `r8`, `r22` —
  `np.interp` clamps rather than extrapolates). Those rows are flagged in the aggregator output.
- [`mjc/on_policy/metered_repair`](../../../../mjc/on_policy/metered_repair/README.md) **§4d bounds
  how far any ladder prize can be trusted**: there, A-error is not monotone in target budget share
  and no process column predicts the outcome column, raising the possibility that ladder outcomes
  track allocation *timing* rather than quality. `uniform` and `oracle` both have `tree_share_sd` =
  0.000, so timing is not in play for the prize itself — but it is fully in play for §7's tap arms,
  and those numbers should be read with that caveat.
- **3 seeds, one geometry, L=4, one drift calibration.** The regime boundary (§2) is located to
  within a factor of ~2 in budget, not sharply.
- **`partial_hetero` §5's learner/geometry confound is untouched here**, and the `oracle`-vs-arm-order
  sensitivity (0.008 at the anchor) is why the noise floor is measured per point rather than
  imported.

## 10. Open items

1. **Push the abundance limb until `E(d)` bends the other way.** The whole scoping in §5 rests on
   never reaching the shoulder; `B` = 262144 (4× the top point) is one command and would either
   find the turn or push the bound out another two doublings.
2. **Separate data from optimizer steps** by sweeping `fm_epochs` against `B` at matched product.
   This is the cleanest available check on §5's steepening, and it is cheap.
3. **Make the meter bind for real** (`meter_budget` is already plumbed) so an arm can *choose* its
   own monitor/collect split. That is the experiment the metering claim actually wants, and it is
   the one this ladder still cannot run: allocation between looking and collecting, rather than
   allocation of collection alone.
4. **Re-ask the metering question on the interference component alone**, which is the half the
   doc's intuition fits. It falls with budget here; whether it falls *to zero* is the sharpened
   version of the original prediction.

# Round 2 — what survives, and what the claim actually needed

**Status**: 63 further GPU runs (21 points × 3 seeds), five limbs, pre-registered in
[DESIGN_ROUND2.md](DESIGN_ROUND2.md). Round 2 **overturns round 1's
headline** (§11–§12), **refutes its own central premise** (§13), produces the node's sharpest
result on the value of a tap (§14), and re-runs the published ladder without the over-training
(§18). **Date**: 2026-08-01.

## 11. E4 — the loop overfits within the round, and that was round 1's confound

`n_steps = fm_epochs × collect_budget / batch_size`, so round 1's budget sweep moved distinct
data and gradient steps together at a fixed 8 epochs. Two orthogonal cuts separate them:

| cut | `B` | epochs | steps/rd | uniform | oracle | prize |
|---|---|---|---|---|---|---|
| **fixed DATA** | 8192 | 2 | 64 | 0.6921 | **0.6363** | +0.0559 |
| | 8192 | 8 | 256 | 0.7371 | 0.6738 | +0.0633 |
| | 8192 | 32 | 1024 | 0.8161 | **0.7533** | +0.0629 |
| **fixed COMPUTE** | 2048 | 32 | 256 | 0.8314 | **0.7961** | +0.0354 |
| | 8192 | 8 | 256 | 0.7371 | 0.6738 | +0.0633 |
| | 32768 | 2 | 256 | 0.6740 | **0.6221** | +0.0519 |

> **16× the gradient steps on the *same* data costs +0.117 in tree error.** The loop overfits
> the round's collected sample. The published `fm_epochs=8` is already well past optimal —
> 2 epochs gives oracle **0.6363** against 8 epochs' 0.6738 at identical data.

So round 1's `E(d)` ran along a **diagonal** where compute grew with data, mixing a real
benefit with a real harm. Its "steepening ⇒ saturation is ~16 doublings away" conclusion is an
artefact of that diagonal, and is withdrawn.

## 12. E6/E7 — at matched compute the prize turns over. This is the doc's limb.

Iso-compute is also the **right** scaling for the doc's own analogy: *"train on all of it"* vs
curation is a choice made at a **fixed training budget** over a corpus of varying size. More
corpus does not buy more gradient steps.

**256 gradient steps/round — 7 points, 64× in data:**

| `B` | epochs | `d` = 0.96·`B` | uniform | oracle | **prize** | sem | floor | slope of `E` |
|---|---|---|---|---|---|---|---|---|
| 1024 | 64 | 983 | 0.9001 | 0.8383 | +0.0618 | 0.0091 | 0.0100 | — |
| 2048 | 32 | 1966 | 0.8314 | 0.7961 | +0.0354 | 0.0030 | 0.0143 | −0.0423 |
| 4096 | 16 | 3932 | 0.7817 | 0.7260 | +0.0557 | 0.0076 | 0.0069 | −0.0701 |
| 8192 | 8 | 7864 | 0.7371 | 0.6738 | +0.0633 | 0.0031 | 0.0084 | −0.0522 |
| **16384** | 4 | 15729 | 0.7027 | 0.6387 | **+0.0640** | 0.0036 | 0.0028 | −0.0350 |
| 32768 | 2 | 31457 | 0.6740 | 0.6221 | +0.0519 | 0.0030 | 0.0034 | −0.0167 |
| 65536 | 1 | 62915 | 0.6604 | **0.6154** | **+0.0450** | 0.0010 | 0.0009 | **−0.0067** |

**1024 gradient steps/round — 3 points, 16× in data:** prize +0.0629 → **+0.0860** (`B`=32768)
→ **+0.0639** (`B`=131072).

| | peak → abundant end | t | seeds |
|---|---|---|---|
| 256 steps/rd | **−0.0190 ± 0.0031** (0.0640 → 0.0450, −30%) | **−6.10** | 3/3 |
| 1024 steps/rd | **−0.0221 ± 0.0010** (0.0860 → 0.0639, −26%) | **−21.67** | 3/3 |

1. **`E(d)` flattens toward a capacity floor** — slope −0.070 → **−0.0067** per doubling. That
   is ordinary diminishing returns, and it means **saturation is reached**, not 16 doublings
   away. The floor is the FM's *capacity* floor (~0.60 at 256 steps, ~0.586 at 1024), well
   above `env_check`'s 0.189 aleatoric floor — more compute lowers it, which is why the
   turnover point moves right with compute (`B`=16384 → 32768).
2. **The prize is humped**, exactly the shape round 1 pre-registered in §2(iv) and could not
   find. Both limbs are now measured, at two compute levels, 3/3 seeds each.
3. **So the doc's prediction is supported at the abundant end once compute is controlled.**
   Round 1's directional negative was the confound, not the claim.

## 13. E3 — a clean 2×2 that refutes this round's own premise

Round 2 was built on "damage is the denominator": the round is `advance_drift → collect →
refit`, so cutting drift 30× should be the cheap equivalent of raising the budget 30×.
**It is not.** Both budgets were run at both drifts:

| | drift 0.30 | drift 0.01 | drift effect |
|---|---|---|---|
| `B` = 8192 | +0.0633 | +0.0624 | **−0.0009** |
| `B` = 32768 | +0.0860 | +0.0868 | **+0.0008** |

with the budget effect **+0.0227 (t = +5.23)** at drift 0.30 and **+0.0244 (t = +5.37)** at
drift 0.01. C6 confirms the damage really moved (tree KL/round 5.02 → 0.11, 44×).

> **Budget moves the prize; drift does not touch it.** The repair-balance premise behind this
> limb is refuted by the limb itself, and round 1 §5's "the steepening tracks the drift-limited
> boundary" explanation goes with it.

The absolute errors get *slightly worse* as drift falls (oracle 0.6738 → 0.6871), which is this
node's entropy identity for the **fourth** time: `KL(w‖uniform) = log m − H(w)`, and the OU walk
is anchored **at** uniform, i.e. at maximum rendering entropy. Small σ keeps the world at its
hardest-to-predict point; large σ wanders to easier low-entropy corners (measured in
[`../README.md`](../README.md) §7 at tree floor 0.160 drifted vs 0.195 uniform). The prize
differences arms *within* one world, so this largely cancels there.

## 14. E5 — the honest ladder: priced, only the cheap tap is worth buying

The published ladder charges every arm the same monitor bill *on purpose*, "so the ladder
measures allocation only". That isolates the allocation rule and hides the economics: **a smart
allocator never pays for its own smartness.** E5 splits one shared total by what each policy's
drive actually reads. At the published total `T` = 22784, against `uniform`, which reads
nothing and collects all of it:

| arm | bill/round | `B` | tree err | **vs uniform** | % of the free ceiling |
|---|---|---|---|---|---|
| `oracle` (privileged, free) | 0 | 22784 | 0.6535 | +0.0738 | — (the ceiling) |
| **`visits_only`** | **512** (2.2% of `T`) | 22272 | **0.6632** | **+0.0642** | **87%** |
| `value_red` | 14592 (64%) | 8192 | 0.6891 | +0.0383 | 52% |
| `reducible_only` | 14080 (62%) | 8704 | 0.7246 | **+0.0028** | **4%** |

Pooled over three totals, per 1000 monitor sequences bought:

> **relevance +0.1286 tree-err · reducibility +0.0001 · a ~1200× difference in
> cost-effectiveness**, where the design doc predicted 25–30×.

At `T` = 45568 the reducibility tap recovers **−0.0001** — priced, it is not worth buying at
all, where the equal-charge ladder credited it with 13–18% of the prize. **The whole of its
apparent value was the subsidy.**

This confirms and sharpens [the idea doc](../../../../../ideas/meta_learning_under_metered_data.md)
§6, which argues the endogenous relevance tap is cheap (*"one round, zero environment
interaction, zero labels"*) while relevance in LLM training is expensive because it is done by
mixture ablation. E5 puts the first price on it — 512 vs 14080 sequences — and finds the cheap
tap keeps **87%** of what a privileged oracle gets for free.

## 15. Where this leaves the metering claim

| | fixed epochs (round 1, + `B`=262144) | **matched compute (round 2)** |
|---|---|---|
| prize vs budget | **rises**, +0.0046 ± 0.0004/doubling, t = +10.35 over 264× | **humped, falls** at the abundant end, t = −6.1 / −21.7 |
| `E(d)` | steepens (diagonal, confounded) | flattens to −0.0067/doubling — **saturated** |

The two scalings answer oppositely, and **which one the doc means is the whole question**. Its
own framing — a fixed training budget, a corpus you choose a mixture from — is iso-compute, and
there **the claim holds**: curation's value peaks and then declines as data becomes abundant.

Two things from round 1 survive unchanged and are independent of the confound:

- **The named quantity is still wrong.** Matched-ratio pairs differ by up to +0.023 (t = +5.23)
  while 4× the monitoring at fixed budget leaves the prize bit-identical to ten decimals. It is
  the **budget**, not the monitor:collect ratio.
- **The prize is two effects.** Interference now falls from +0.0470 to **+0.0082** (9% of the
  prize) at `B` = 262144 — the doc's intuition is *fully* vindicated for that half.

## 16. Round-2 caveats

- **E5 runs at the default 8 epochs**, so its three totals differ in gradient steps as well as
  data — the very confound §11 exposes. The **within-total arm contrasts are clean** (one world,
  one seed, one compute level per arm... except that each arm's own budget sets its own step
  count, so the paid arms also train less). The across-total trend is not clean.
- **The iso-compute lanes vary epochs from 64 down to 1** to hold steps fixed, so "distinct
  data" and "epochs" are perfectly anti-correlated within a lane. The turnover is therefore
  *"more unique data per gradient step stops paying"*, which is the right statement but is not
  the same as *"corpus size stops mattering"* in an infinite-data limit.
- **The capacity floor is compute-set (~0.60), not information-set (0.189)**, so "saturation"
  here means the model has extracted what it can at that compute — the doc's asymptotic claim
  about *information* saturation is still untested.
- The starved end of the 256-step lane is noisy (`ic1024` floor 0.0100, `fc2048` 0.0143); the
  **abundant end, where the claim lives, is clean** (floors 0.0009–0.0034).
- 3 seeds, one geometry, L=4 throughout; `metered_repair` §4d's timing caveat still bounds every
  tap-arm number.

## 17. Round-2 open items

1. **Push the iso-compute lane to the prize's zero crossing** — extrapolating the 256-step lane
   at −0.0095/doubling puts it near `B` ≈ 1.7M. That would turn "declines" into "vanishes".
2. **Fix the overfitting rather than working around it.** Replay across rounds, or an epoch
   count tuned per budget, would make the ladder's absolute numbers meaningfully better —
   2 epochs beats 8 by 0.037 at the published budget, for free.
3. ~~Re-run the published ladder at 2 epochs.~~ **Done — §18.** The science holds; the absolute
   levels and the headline prize do not.
4. **E5 with matched steps per arm**, so the paid arms are not also compute-starved.

---

## 18. E8 — the published ladder re-run at 1, 2 and 4 epochs

§11 found the loop over-trains, which puts every headline number on this node in question.
This is that question answered: the **full ten-arm published set** (`channel_env.POLICIES`, in
order, with `oracle_dup` appended *last* so the first ten consume the RNG stream exactly as
`ladder_fix_s*` did), at the published `B`, `mon_n`, `rounds` and `n_eval`, with **`fm_epochs`
the only thing that differs**. The ep8 column is the actual published runs, read from
[`../figures/ladder_fix_s{1,2,3}`](../figures/).

### Absolute levels: the whole ladder was over-trained by ~0.04

| arm | ep1 | **ep2** | ep4 | ep8 (published) |
|---|---|---|---|---|
| oracle | **0.6382** | 0.6401 | 0.6524 | 0.6742 |
| visits_only | **0.6476** | 0.6495 | 0.6653 | 0.6844 |
| value_red | **0.6476** | 0.6495 | 0.6592 | 0.6863 |
| reducible_only | 0.6754 | 0.6820 | 0.6987 | 0.7259 |
| uniform | 0.6826 | 0.6921 | 0.7089 | 0.7371 |
| value (old tap) | 0.7112 | 0.7120 | 0.7256 | 0.7406 |
| error_only | 0.6909 | 0.6930 | 0.7131 | 0.7425 |
| value_red_satiety | 0.6878 | 0.7008 | 0.7202 | 0.7554 |
| value_satiety | 0.7070 | 0.7219 | 0.7223 | 0.7562 |
| lprog_only | 0.7209 | 0.7330 | 0.7321 | 0.7524 |

Every arm is **0.03–0.07 better** at 1–2 epochs. The published tree-error levels are pessimistic
throughout, monotonically in epochs, and `fm_epochs=1` is best of the four everywhere.

### But every published *contrast* survives, and most get stronger

| claim (`../README.md` §3) | published (ep8) | **ep2** | verdict |
|---|---|---|---|
| the tap repair, `value_red − value` | −0.0542 | **−0.0625** | holds, **larger** |
| `value_red` ties `visits_only` | +0.0019 | **+0.0000** | holds, an *exact* tie |
| `oracle − value_red` | −0.0122 | −0.0094 | holds |
| `error_only − uniform` | +0.0055 | +0.0009 | holds, shrinks toward zero |
| satiety costs `value_red_satiety − value_red` | +0.0691 | +0.0513 | holds, ~25% smaller |
| `visits_only` recovery of the oracle prize | 83.8% | 82.3% | holds |
| `reducible_only` recovery | 17.7% | 20.0% | holds |
| allocation variance, `value` tree-share sd | 0.406 | 0.439 | holds |
| allocation variance, `value_red` tree-share sd | 0.059 | 0.044 | holds |

**The ordering is identical on the arms that carry the conclusions.** At ep1, ep2 *and* ep8 the
top five are `oracle < visits_only < value_red < reducible_only < uniform`; ep4 swaps
`visits_only`/`value_red`, which sit within 0.006 of each other and are a tie by construction
(§3). The reorderings are all in the **bottom half** — the satiety rungs, `value`, `error_only`
and `lprog_only` — which span only 0.02–0.04 and shuffle freely between epoch counts.

### The one number that does move

> **uniform→oracle prize: 0.0629 (ep8) → 0.0565 (ep4) → 0.0521 (ep2) → 0.0445 (ep1).**

The published prize is **~17% inflated** by over-training at ep2, ~29% at ep1. This is the same
effect as §12 seen from the other side: at fixed `B`, cutting epochs moves *along* the
iso-compute family toward more unique data per gradient step, and the prize shrinks — the
turnover again.

### What this settles

1. **This node's science is robust to the over-training.** The tap repair, the
   `value_red`/`visits_only` tie, the 84%/18% recovery split, the 8× allocation-variance drop
   and the satiety cost all reproduce at 4× and 8× less training, several of them more cleanly.
   Nothing that carries a conclusion depends on `fm_epochs=8`.
2. **The absolute levels should not be quoted as the ladder's capability.** They are ~0.04
   worse than the same configuration achieves at 1–2 epochs.
3. **The headline prize is inflated ~17%.** Anything downstream that treats 0.063 as *the*
   value of curation on this geometry should use ~0.052 at a non-over-trained setting — and
   §12's point stands that even that is a function of where you sit on the iso-compute curve.

*(One pre-existing inaccuracy, unrelated to epochs and present at ep8 too: `../README.md` §3's
*"`error_only` is the ladder's worst arm"* comes from the single-seed `l4_v1` table. In the
3-seed ten-arm data at **every** epoch count `error_only` ranks 6th–7th of 10, and the worst
arms are the satiety rungs and `lprog_only`.)*


---

## Reproduce

```bash
cd experiments/            # MODAL_PROFILE=chromatic
python3 -c "from rhm.verify_backcompat import verify; verify()"                # the standing gate
python3 rhm/directed_sculpting/full_loop/metering_sweep/sweep_plan.py          # the derived grid

modal run rhm/directed_sculpting/full_loop/ladder.py::ladder --quick --tag met_smoke \
    --policies uniform,reducible_only,visits_only,value_red,oracle,oracle_dup \
    --mon-price 1.494108 --collect-budget 8192                                 # smoke

# round 1 (33 runs), spawned in parallel from one detached orchestrator
modal run --detach rhm/directed_sculpting/full_loop/metering_sweep/run_sweep.py::sweep
# round 2 -- launch each limb separately so each reports on its own
for w in e1c e3 e4 e5 e6 e8; do
  modal run --detach rhm/directed_sculpting/full_loop/metering_sweep/run_sweep.py::sweep --which $w
done

python3 rhm/directed_sculpting/full_loop/metering_sweep/aggregate.py --mirror   # gates, then curve
```

Results JSON on the `rhm-scaling-data` volume under `directed_sculpting/ladder_met_*`, mirrored to
[`figures/`](figures/) along with [`figures/aggregate_output.txt`](figures/aggregate_output.txt).
`--mon-price 1.0` with the published `--collect-budget 8192` reproduces `ladder_fix_s*` bit-identically,
so every prior result on this node stays reachable from the same script.
