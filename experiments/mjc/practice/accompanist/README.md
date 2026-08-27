# accompanist — the forward model is the incumbent's free accompanist: `offbook/`'s negative was library construction stacked on a model the incumbent never paid for

**Up**: [`../README.md`](../README.md) (practice) · [`../../README.md`](../../README.md) (mjc)
**Files**: [`FILES.md`](FILES.md) · **Child**: [`presto/`](presto/FILES.md) (the fast-piece node,
design record and results) · **In place**: [`../offbook/FILES.md`](../offbook/FILES.md) Rounds 5–7b
(`d1`, `d2`, `d3`, `d3b` — additive flags on offbook's own delay gate, left there so `d0` stays
byte-reproducible).
**What this corrects**: [`../offbook/README.md`](../offbook/README.md) (PR #72), now marked
**suspect**. **What it licenses**: [`../acappella/SPEC.md`](../acappella/SPEC.md), the model-free
re-port.
**Runs**: `d1` (10 arms) · `d2` (7) · `d3` (8) · `d3b` (7 × 2 operators) on offbook's piece; `t0`
(3 × 3 ladder) · `p0` (8 strategies × 8 Δ × 2 operators) on the new piece. 2026-08-26 → 08-27,
single seed throughout; bit-identity controls across rounds are the claims. One orchestrated
conversation, two implementer agents.
**Attribution**: the "we shouldn't have run this with an FM" diagnosis is Jasper's; the rounds
below are what it took to see it.

## The question

`offbook/` ported RHM's re-internalization onto the arm and found the deep unit (a chain across the
whole piece) never adopted, concluding after a delay gate (`d0`) that "depth on this piece is
geometric, not economic — an environment property". This conversation asked where that setup
diverged from the conditions practice work needs, and ran the two fixes that the record suggested:
build the library the way `legato/` did, and build a piece that can only be played from memory.

## What was run

| run | where | one variable | outcome |
|---|---|---|---|
| `d1` | offbook piece | library content: harvest noise × selection × seam read, 10 arms | Δ\* = None everywhere; content closes 0–70% of the gap to the anchor; **vintage** (harvest after training) is the largest single move; keying and audition pay only together |
| `d2` | offbook piece | legato's **nesting** (candidates for seam k harvested in the configuration with seams < k committed) | reproduces legato's pool at every cell (per-state oracle 0.87–1.16× vs `d1`'s 4×); `nest_key` chain **0.1455** vs anchor 0.1460, flat over 0→320 ms (1.09×); Δ\* = None — misses the guard by 3.6% at 160 ms |
| `d3` | offbook piece | + the FM adapting through the nested practice (legato's plasticity) | **Δ\* = 160 ms — the pre-fixed gate passes** against the naive delayed incumbent; content ≥ legato at every cell; `e_react` 0.0098 → 0.0063 |
| `d3b` | offbook piece | + efference copy through the delay (the FM rolled Δ steps along issued commands), both operators | reactive at 80 ms **0.223 → 0.021**; Δ\* = None under both operators, every arm; the chain is best-of-three from 320 ms, not 80 |
| `t0` | new piece | tempo × damping ladder, 5 × 120 ms segments, no patch, both operators | efference copy removes the delay penalty in every cell (Δ = 80 ms: 0.025–0.062, all under the anchor); low damping collapses the horizon 23 → 8 and doubles stored-content error |
| `p0` | new piece | the Δ sweep at the design point, nested build, both operators | Δ\* = None under both; **naive**: ordering forms from 60 ms, chain < reactive from 40 ms inside playability, guard missed by 0.9%; **predicting**: reactive wins every row (2.3× degradation vs 19×) |

Full tables, identity gates and ledgers: `../offbook/FILES.md` §Rounds 5–7b and
[`presto/FILES.md`](presto/FILES.md).

## Findings

1. **`d0`'s "no playable niche" was library construction.** Offbook's tapes were noisy closed-loop
   renditions, uniformly drawn, selected by an audition blind at chain span, launched unkeyed; and
   — the part that mattered — harvested from purely reactive traversals, so candidates for later
   seams never launched from the states they would be scored and played on. Legato's own commit
   events show 142/144 of its candidates came from the already-committed configuration. With that
   nesting the pool matches legato's cell for cell (`d2`), and with the model adapting through it
   the content is at or better than legato's (`d3`). (`d1`–`d3`)
2. **Against a delayed incumbent that cannot predict, the deep unit is the best playable option in
   the human reflex band, on both pieces.** Offbook's piece: gate passes at 160 ms (`d3`). The
   fast piece: full ordering from 60 ms, chain beats reactive from 40 ms, guard missed by 0.0009 m
   (`p0`, naive). The nested keyed chain is essentially delay-invariant everywhere it was measured
   (0.96–1.09× over the sweep) while the reactive controller degrades 19–63×. (`d2`, `d3`, `p0`)
3. **Give the incumbent the forward model through the delay and the niche closes.** Efference copy
   is a Δ-step FM rollout; for Δ inside the composition horizon (~21 steps on both plants) it
   recovers 6–11× of the naive penalty, and reactive owns the whole playable region. `d2`'s
   "chain strictly best from 80 ms" moves to 320 ms; `p0`'s ordering never forms. The
   feedback-consumption ordering of degradation replicates under both operators — the operator
   changes magnitude, not mechanism. (`d3b`, `t0`, `p0`)
4. **The momentum axis, as a way to make a piece memory-only, fails.** Low damping degrades the
   model (horizon 23 → 8, one-step error 2.3×) far more than delayed control, and the stored
   content harvested under it is worse, not better. Lookahead across seams is worth 2.3–5× at
   120 ms seams (legato F3's null was measured at 400 ms). Seam posture spread sits below repeat
   noise in 44/45 seam-cells at this tempo — the keyed chain has content but no addressing. (`t0`)
5. **The audition's calibration tracks the model.** Under the frozen FM the launch key out-picks
   the FM audition at chain span by 1.28×; under the adapted FM the audition overtakes the key.
   Where the FM sits in the loop decides which selector is right. (`d2` vs `d3`)

## Interpretation (discussed with Jasper 2026-08-27 — argued, not measured)

The forward model plays a different role at every step of the mjc practice lineage, and none of
them is the role search played on RHM. In `etude/` (the one MuJoCo node where committed units beat
the incumbent, 3/3 seeds) the model was the **bottleneck** — capacity-limited, interfering across
segments — and executed traces won because they were stored outside it. From `fingering/` on, the
arm's h=256 model fits the plant essentially perfectly and re-grounds every 20 ms, so the reactive
incumbent plans in imagination **for free** and sits at the noise floor; `never` won every round.
`offbook/` then built its ports out of that model (seam-time audition, a span head on the trunk,
rent in rollout-steps), and its delay gate was bridged by it. On RHM the incumbent's search was
**real and priced** — every candidate materialised on the plant, a declared grounding budget — and
depth became unaffordable to primitives at the budget. That economy has never run on this
substrate. Finding 2 is the RHM condition seen in negative: the unit wins exactly when the
incumbent is denied the model. Finding 3 is why a delay knob cannot substitute for pricing the
incumbent's search: any delay the model can compose across, it will.

Held loosely: whether "denied the model" should be implemented as priced real rollouts (RHM's
structure) or as a reflex law with no search — [`../acappella/SPEC.md`](../acappella/SPEC.md)
proposes the first with the second as reference.

## What this does to `offbook/`

The π-level facts (trust tracks exposure; extinction with the forced window; the poison
quarantine under forced exposure) do not involve the model and stand. Findings 1–3 there (routing
"by selection hygiene under the audition scorer", audition blind at chain span, the rent table) are
statements about the FM audition; `d0`'s "geometric, not economic" conclusion is retracted by
`d2`/`d3` on its own criterion; and every niche claim in this node, including `d3`'s pass, is
against an incumbent with no predictor. `offbook/README.md` carries a suspect banner to this effect.

## Caveats

Single seed on every run; 24 shared eval geometries; means and medians disagree in places (all
reported). `d3`'s pass margin is 3.0%; `p0`'s naive miss is 0.9% — both are "at the bar", and
neither survives the predicting incumbent. The adapted-FM × efference-copy cell on offbook's piece
was not run. The predictor uses the same model the planner uses, so their qualities are not
separable in this design — which is the point of the next node, not a fix for this one.

## Reproduce

Offbook rounds: `../offbook/FILES.md` §Reproduce (Rounds 5–7b) — `delay_gate.py --arms …`,
`--nest-adapt`, `--dual-obs`, all `--spawn`. Presto: [`presto/FILES.md`](presto/FILES.md)
§Reproduce (`tempo.py`, `delay_gate.py`). Volume `mujoco-control-data`:
`/data/practice_offbook/{d1,d2,d3,d3b}/`, `/data/practice_presto/{t0,p0}/`.

## Next

[`../acappella/SPEC.md`](../acappella/SPEC.md): the port back on `etude/`'s substrate with no
forward model anywhere — a priced real-rollout incumbent under a declared budget, plant audition,
nested selection, routing π.
