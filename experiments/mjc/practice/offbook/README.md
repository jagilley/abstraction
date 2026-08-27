# offbook — the port back on the motor substrate: routing consolidates, trust tracks exposure, and the chain level is refused by the task's economics

**Up**: [`../README.md`](../README.md) (practice) · [`../../README.md`](../../README.md) (mjc)
**Files**: [`FILES.md`](FILES.md) — the full design record: the mapping table, every decision with
its measured reason, the per-round records, and the gotchas. **There is no `SPEC.md` by design**:
this node was run prompt-only (Jasper's call, 2026-08-25), so the orchestration conversation is the
record of what was asked and `FILES.md` is the record of what was decided.
**Substrate donor**: [`../legato/`](../legato/README.md) (`world.py` copy-forked verbatim, the
calibrated `l2` cost cell; donor untouched, gate G-F asserts bit-for-bit fidelity).
**The ports' shape**: [`../../../rhm/practice/native/`](../../../rhm/practice/native/README.md)
(routing + corridor + address book) · [`../../../rhm/practice/spiral/SPEC.md`](../../../rhm/practice/spiral/SPEC.md)
(the phase discipline followed here).
**Constraints carried from the mjc side**: [`../fingering/`](../fingering/README.md) (op taxonomy,
G1, priced deliberation) · [`../legato/`](../legato/README.md) F4/F5 · [`../span/`](../span/README.md)
F2/F4 (a live plan cannot be the consolidated object) · [`../README.md`](../README.md) (apparatus cautions).
**Idea docs**: [`practice_manufactures_its_own_credit`](../../../../ideas/practice_manufactures_its_own_credit.md)
§3½ (content/identity/use), §6 (rarity), §18, §20 ·
[`two_climbings`](../../../../ideas/two_climbings.md) §7 (the mjc row this node fills in).
**Runs**: `g0` (Phase A gates) · `O1` (the port, 6 arms × 120 cycles) · `O2` (the currency canary,
1 arm) · `O3` (the rehearsal canary, 1 arm) · `d0` (the `organist` delay gate), 2026-08-25→26.
**Single seed on every treatment; ranks, signs, bit-identity twins and cross-tag exact controls are
the claims.** One orchestrated conversation; one implementer agent built all four rounds,
canary-staged (one treatment arm per round after O1, each launched only after the previous round's
reduction was read).

## ⚠ Status: SUSPECT (2026-08-27) — read [`../accompanist/README.md`](../accompanist/README.md) first

**Jasper's note**: these findings may be erroneous; read with a grain of salt. We shouldn't have
run this with an FM; as of running we didn't understand how FM + practice/re-internalization play
together.

**Why, specifically** (six follow-up rounds, `d1`–`d3b` below and `../accompanist/presto/`):

- **This node's incumbent plans for free in a near-perfect forward model.** On RHM, the search the
  vocabulary competes with is *real and priced* (every candidate materialised on the plant, under a
  declared grounding budget) — that is what made depth unaffordable to primitives. Here the
  reactive incumbent re-plans every 20 ms in imagination at zero price. The RHM economy never ran
  on this substrate, so "the chain level is never adopted" is not a finding about the units.
- **The ports were built out of the same model.** Seam-time audition is an FM rollout (blind at
  chain span, G-C); the rent is denominated in FM rollout-steps (buys nothing, G-K(a)); Port 2 sits
  on the FM trunk. Findings 1–3 are statements about the FM audition, not about routing on a plant.
- **`d0`'s "no playable niche" was library construction, and is retracted on its own criterion.**
  Tapes here were noisy closed-loop renditions, uniformly drawn, harvested from purely reactive
  traversals. Built the way `legato/` built them (nested in the committed configuration, the model
  adapting), the chain reaches the anchor and the pre-fixed gate **passes at 160 ms** (`d3`) —
  against the naive delayed incumbent used in `d0`.
- **And that pass does not survive an incumbent allowed to predict through the delay** (`d3b`):
  efference copy is a Δ-step FM rollout, and every Δ inside the model's horizon is bridged by it.
  The "geometric, not economic" conclusion in the Interpretation below is superseded.

**What stands**: the π-level facts — trust tracks exposure (findings 4–5), the poison quarantine
under forced exposure (8), the frozen key's need for boundary information (9). **What is next**:
[`../acappella/SPEC.md`](../acappella/SPEC.md), the re-port with no forward model anywhere.

## The question

Every committed unit on this substrate had been an **external object** — a keyed tape
(`fingering/`) or a live CEM plan at launch. The RHM side ran the port back (`native/`, `spiral/`,
PR #69) and found routing consolidates into the planner, the corridor into the executor, the table
survives as the address book, and the deep-era value rides on **trust** — π's per-level proposal
mass, formed only by time-in-use. This node asks the same question where the units are motor, the
action space is continuous, and the model has a composition horizon:

> Does a committed motor vocabulary consolidate into the learner — proposed and run as one unit,
> its members no longer enumerated — and does trust in it form, on a plant?

## Design in brief

**The seam is the decision point; the library is the action set.** At each re-grounding point the
agent chooses among {segment tapes, chains to the end of the piece, the live plan}. The
enumeration analogue is a new op, **seam-time audition** — score every candidate's stored commands
by FM rollout from the *realized* seam state, priced per materialisation (licensed by legato G6 and
never previously run; fingering's audition only ever fired at commit time). Port 1 (**routing**):
π(unit-slot | seam posture) gates which slots get auditioned (top-k of 8 slots/cell), trained
online by self-imitation on traversals the **body** graded competent — never by the audition score.
Port 2 (**corridor**): a span head on the FM trunk emitting a unit's measured commands behind a
per-slot parity gate (identity is input, content is target — `handle/`'s lesson). Only measured
executed tapes enter the library (span F2/F4); the library table stays outside as the address book;
chain cells are grown over segment-slot spellings (legato F5's mechanism). K sits at the measured
competence plateau (72/cell — G-K(b)) so the reported rent is rent a competent agent actually
faces; the ratio is a property of the slot partition, not K. Full mapping table, the arm ladder
(`never` → `key_frozen` → `audit_all` → `audit_prop_k` → `route_native` → `fid`), the battery, and
the poison twin: [`FILES.md`](FILES.md).

Four rounds, each licensed by the previous one's reduction:
**O1** the port itself · **O2** π's imitation stream graded in the currency chunks pay in
(feedback events; ×8 importance weight, no invented exchange rate — fb is the only varying cost) ·
**O3** the exposure fix (the launch-ε moved *upstream* of π's gate at matched exploration budget —
`census/`'s "targeted rehearsal" on a plant) · **`organist`/d0** the world made to price feedback
(an observation delay Δ on every agent-side feedback consumer; instruments and FM training data
stay true-state), with a pre-fixed neutral criterion and a playability guard.

## Findings

1. **Routing consolidates on the motor substrate — better error at 5.3× less deliberation.**
   `audit_prop_k` 0.0460 vs `audit_all` 0.0576 at c120 (steady-state 2,448 vs 13,073
   materialisations/traversal); `route_native` 0.0540; `key_frozen` 0.4203; `never` 0.0085 at the
   flagged common horizon. `fid` ≡ `audit_all` exactly (0.000e+00) over the 69 cycles it reached;
   π at k=N ≡ full audition bit-for-bit (G-P). (`O1`)
2. **The mechanism is selection hygiene under a noisy scorer, not reinvested savings.** The freed
   budget has nothing to buy here — the CEM ladder is non-monotone past k=64 (G-K(a), `span/`'s
   exploitation gap) — so `native/` finding 1's cost half ports and its width half does not, for a
   measured reason. The scorer's optimism is now a direct instrument: realised ÷ predicted at the
   launched candidate runs **4.97×** (`O2`) and **5.52×** (`O3`) at ~22 cand/decision; and the
   enumeration−routing error gap flips sign as the argmax candidate set widens (corr +0.496,
   enumeration worse in 31/34 probes — étude E-3b's winner's curse at the seam). (`g0`, `O1`–`O3`)
3. **Seam-time audition is blind at chain span.** Segment-level ranking is calibrated (ρ=0.924);
   chain-level is not (ρ=0.337), truncation makes it worse (`aud_horizon=0` is the calibrated
   setting), and the audition committed a chain 0.00 of the time at every budget — the op routing
   replaces cannot rank the units that live beyond the model's horizon. (`g0` G-C)
4. **Trust tracks exposure, and credit cannot substitute for use.** Under a ×8 cheapness weight
   with exploration still downstream of π's gate, chain trust rose to ~6× the control exactly
   while the forced window was open and extinguished to the control's level within ~15 cycles of
   its closing; **3 chain traversals total** entered π's targets in 100 cycles (the two ε's
   multiply: ~0.002 chain launches/decision — `native/prop`'s lock-in one level out). The ×8
   weight had almost nothing to weight, and no feedback events were saved because the priced
   behaviour never occurred. (`O2`)
5. **Sustained exposure sustains and grows trust.** With the launch draw moved upstream of π's
   gate at matched exploration budget, chain mass rises monotonically across windows to 0.0230 in
   c91–120 — **25× O2's late mass** — with `argmax_chain` nonzero in 23/40 probes. O2's apparent
   currency effect was exposure in disguise. (`O3` vs `O2`/`O1`, cross-tag exact controls)
6. **The body refuses these chains — band-gating is ruled out.** 203 chain plays on the body
   (O1+O2 produced 3 between them): 10.8% pass the competence band; mean realised e_piece 0.2889
   against a band of 0.1044 — **2.77× worse than the median traversal**, stable across windows.
   `frac_chain` executed in performance is 0.000 at every probe of every arm in every round; the
   feedback saving chains carry (2 fb vs 4) is therefore never collected. Rehearsal costs ~30% on
   error and saves nothing. (`O3`, pre-band instruments)
7. **Delay sensitivity is ordered exactly by feedback consumption — and no playable delay creates
   a niche for the chain.** Under observation delay Δ=0→16 steps (0→320 ms): reactive (61
   fb/piece) degrades **63.1×**, live seg-planning (4 fb) 2.69×, the chain (2 fb) 1.69×, the
   segment tape 1.28×. The playable region (min ≤ `ref_stale` 0.1460) ends at Δ=2 (40 ms) and
   reactive wins all of it; the ordering inverts only at Δ=16, by universal collapse, which the
   pre-fixed guard refuses. **Δ\* = None; the round stopped at the gate as pre-committed.** One
   crossover does land inside playability: `seg_tape < live_seg` at 40 ms — frozen measured
   *segment* content overtaking the live segment plan, one level below the chain. (`d0`)
8. **The quarantine holds under adversarially generous exposure.** Uniform rehearsal force-plays
   the poison unit too: π's poison mass bumps (max 0.0061) and decays within a few probes, every
   time, with launch fraction 0.0000 throughout — `native/` finding 6 under forced exposure rather
   than under an enumeration it can ignore. Enumeration and the frozen key have no such channel
   (`key_frozen` launched the poison 2.5% of the time). (`O1`–`O3`)
9. **A frozen key needs boundary information.** `key_frozen` collapses exactly on the seam G-S
   measured at its noise floor (posture spread ÷ repeat-noise 1.12× at seam 2, vs 5.17×/2.27× at
   seams 0–1): by-segment error 0.895 there against 0.077/0.262 elsewhere. Realized-state
   audition and routing both survive the absence. (`g0`, `O1`)
10. **The library and the live planner are jointly load-bearing; the address-book claim is
    untested here.** `no_table` costs the routed arms only +0.005 — but with Port 2 shut, deletion
    falls back to the live plan (`frac_prim`→1.00), so that number measures the planner.
    `no_table_no_plan` leaves **no legal action at all** (O2, reproduced O3). `no_prim`: the
    library alone plays at ~2× error. Port 2 never cleared parity in-run (0/20 slots at τ=0.9,
    head served 0 of 5,482 calls) and was deferred as scoping from O2 on — the corridor port on
    continuous commands is **untested, not refuted**. Minability 12–28 throughout, no collapse.

## Interpretation (discussed with Jasper 2026-08-26 — argued, not measured)

- **(a) The consolidation story splits cleanly into a part that ports and a part the substrate
  refuses, and the refusal is located in the environment.** Routing ports, with a different
  mechanism than RHM's (protection from a 5×-optimistic scorer rather than reinvested width — the
  action set is the exposure, generalized from bad entries to audition noise). Adoption of the
  deep unit does not port, and rounds 2–4 assign why at three separate layers: credit cannot
  substitute for use (O2); use requires exposure the policy's own gate will not provide (O2's
  ε-arithmetic, fixed in O3); and exposure converts to adoption only when the content is worth
  adopting — which these chains, on this piece, are not (O3, d0).
- **(b) "Use is only earnable" gains motor-side structure.** On RHM, macros were worth adopting by
  construction (depth unaffordable to primitives), so the trust clock was the only visible
  constraint. Here the currencies dissociate: chains pay in feedback events, practice grades in
  error, and §6's rarity regime holds on the error axis in every round — so the arc's RHM
  mechanism (error-graded self-imitation funding chunks) never engages. Trust itself behaved
  lawfully throughout: it tracked time-in-use and nothing else — not credit weight, not content
  arrival.
- **(c) Depth on this piece is geometric, not economic — Jasper's framing, adopted.** A chain is
  60 steps against a ~21-step horizon, but flattening depth costs one feedback event per seam.
  `organist` made feedback expensive and confirmed the tax lands exactly in proportion to
  feedback consumption — yet the piece has no regime where the deep unit is the best playable
  option: it is easy enough to steer by feel at small delays, and unplayable for everyone at
  large ones. The tasks where ballistic chunks pay in animals are those *impossible* closed-loop
  and *possible* from memory; this piece is not one of them, and no knob on the learner or the
  senses makes it one. The constraint is the data-generating process, which we control — an
  environment issue, fixable in a future substrate.
- **(d) Where memory first pays under biological delay is one level down.** The segment tape is
  the most delay-robust strategy on the table (1.28×) and overtakes the live segment plan at
  40 ms, inside playability. Held loosely as the pointer it is: the delay-era niche on this plant
  belongs to the segment-span unit, not the phrase-span chain.

## Caveats

- **Single seed on every treatment.** Cross-round comparisons lean on bit-identity: `fid` and
  `prop_kN` twins in-tag; O2/O3's controls are cross-tag but code-path exact (additive flags whose
  defaults reproduce the donor round, asserted at 0.000e+00 each time).
- **O1's `never` and `fid` were truncated** (c57/90, c69/120) by a launch-path defect — a Modal
  local entrypoint blocking in `.map()` dies with the launching client; diagnosed, recorded as a
  Gotcha in `FILES.md`, and fixed with the `--spawn` path from O2's relaunch on. Never-relative
  claims are read at the common horizon c57 and flagged; within-committed claims at c120 are
  unaffected (all four treatment arms completed with their batteries).
- **Chain quality is measured at uniform-rehearsed states**, not at keyed launches chosen by an
  exploit policy (which never occurred). Legato's properly-keyed frozen phrase ran 0.10 against
  0.289 here, so state mismatch plausibly inflates the badness; the unnecessity layer (finding 7)
  does not depend on this.
- **Port 2 is untested** (0/20 parity at τ=0.9 under `span_min_hold`/`span_lam` unswept), and the
  O1 `route_native` − `audit_prop_k` gap (+0.008, the shape of spiral finding 9's pattern) sits
  near the substrate's measured arm-difference floor — filed as consistent-with, not a sighting.
- G-B's ordering was underpowered by design and never read; Phase B's probe ladder replaced it.

## Runs on disk

| tag | what |
|---|---|
| `g0` | Phase A: G-F/G-R/G-P exact; the rent table; G-K(a/b), G-S, G-C, G-B (underpowered), G-D |
| `O1` | the port: 6 arms × 120 cycles (reactive 90), battery + poison; `never`/`fid` truncated by the launch defect |
| `O2` | currency canary: `prop_priced` (×8 fb-importance), cross-tag control O1 `audit_prop_k`; + `no_table_no_plan` |
| `O3` | rehearsal canary: `prop_rehearse` (launch-ε upstream at matched budget), pre-band instruments |
| `d0` | `organist` delay gate: Δ ∈ {0,2,4,8,16} × {reactive, live_seg, seg_tape, chain}, pre-fixed criterion, Δ\* = None |
| smokes | `gsmoke`, `Osmoke`–`Osmoke9`, `dsmoke` — every fidelity gate run before every launch |

## Reproduce

```bash
cd experiments/            # MODAL_PROFILE=chromatic
# gates + smokes
modal run mjc/practice/offbook/gates.py::offbook_gates --quick --tag gsmoke
python3 mjc/practice/offbook/launch_detached.py --fn gates --tag g0 --seed 0
python3 mjc/practice/offbook/analyze_gates.py --tag g0 --fetch

# O1 — the port (NOTE: launch runners only via the --spawn path; see FILES.md Gotcha)
modal run --detach mjc/practice/offbook/offbook.py::offbook --spawn --tag O1 --seed 0 \
    --n-cycles 120 --n-cycles-react 90 --delib-budget 0 --aud-horizon 0 --n-slot 8 --k-prop 2 \
    --lib-add 24 --n-poison 8 --eps-act 0.10 --force-window 3 \
    --arms "never,key_frozen,audit_all,audit_prop_k,route_native,fid"
python3 mjc/practice/offbook/analyze_offbook.py --tag O1 --fetch
modal run mjc/practice/offbook/analyze_offbook.py::figs --tag O1

# O2 — currency canary (single arm; O1 flags plus:)
#   --arms prop_priced --pi-fb-rep 8
# O3 — rehearsal canary (O2 flags plus:)
#   --arms prop_rehearse --p-rehearse 0.10 --eps-act 0.0
# d0 — the organist delay gate
modal run --detach mjc/practice/offbook/delay_gate.py::delay_gate --spawn --tag d0 --seed 0
```

Full flag records per round, the decision table, and the launch-path Gotcha: [`FILES.md`](FILES.md).
Volume `mujoco-control-data`: `/data/practice_offbook/<tag>/…`; fetched copies, reports and figures
under `results/<tag>/`.

## Rounds 5–7b (2026-08-26 → 27): `d1`, `d2`, `d3`, `d3b`

Run after the banner above's diagnosis, on this node's own delay gate with additive flags (defaults
reproduce `d0` at 0.000e+00): the content ladder (`d1`), legato's nesting (`d2`), the model
adapting through it (`d3`), the efference-copy incumbent (`d3b`). Records in [`FILES.md`](FILES.md)
§Rounds 5–7b; reduced results under `results/d1`–`results/d3b`; the writeup and interpretation
live in [`../accompanist/README.md`](../accompanist/README.md).

## Next steps (superseded — see the banner; kept as the record of what was queued on 2026-08-26)

A piece that can only be played from memory — task difficulty co-designed with the reflex delay so
closed-loop control fails at realistic Δ while measured content still passes (the regime fast
motor sequences occupy in animals; the environment fix interpretation (c) names) · state-matched
rehearsal (play a unit only at its keyed seam — d0's `seg_tape` crossover and the state-mismatch
caveat both point here, at the segment span first) · Port 2 with a real training budget or a
simpler head (untested, not refuted) · the exposure-vs-credit control (`pi_fb_rep=1` rehearsal) if
trust dynamics ever need the decomposition · the arrival/rehearsal assay (`census/`'s queued
instrument) once a substrate exists where adoption can occur · `/update-beliefs` for the unit
(trust-tracks-exposure; the three-layer refusal; the quarantine under forced exposure; the
feedback-consumption ordering).
