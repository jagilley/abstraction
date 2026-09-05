# solo — re-internalization on the motor substrate with no forward model: routing, corridor and address book all port at segment span, and what learning buys is cost and delay-robustness, not error

**Up**: [`../README.md`](../README.md) (practice) · [`../../README.md`](../../README.md) (mjc)
**Contract**: [`SPEC.md`](SPEC.md) (2026-09-04, plus the 2026-09-05 amendment on the imitation
filter) · **Files**: [`FILES.md`](FILES.md) — every decision with its measured reason, the gate
record, the smoke record, the sizing table, both run records and their flags, unsmoothed.
**Why this node exists**: [`../acappella/README.md`](../acappella/README.md) found the model-free
niche (stored segment tapes beat a per-Δ re-fit reflex from 192 ms) and deferred Port 2 in one
sentence — "the corridor port has no trunk to sit on without a model." [`../offbook/`](../offbook/README.md)
ran re-internalization once on this substrate, under the forward-model confound, and Port 2 never
cleared parity there.
**Substrate donor**: acappella (world, piece, reflex law, nested library, plant audition, the naive
delay operator; donor untouched, gate S-F1 asserts bit-identity with `acappella/b1`, both delays).
**Port donor**: [`../offbook/nets.py`](../offbook/nets.py). **Trunk donor**: the behaviour-cloned
motor program of [`../../ballistic/`](../../ballistic/README.md) and
[`../../jacobian_teacher/`](../../jacobian_teacher/README.md), with the teacher swapped from a
planner under a model to the reflex law's own traversals.
**Roadmap**: `ROADMAP.md`[^private] §4.5 Track M. Supersedes acappella's queued B2
and its licensing condition (Jasper, 2026-09-04).
**Runs**: `s0` (8 arms × 120 cycles, 3336 s) · `s1q` (the imitation-filter probe, 30 cycles,
1035 s), seed 0, 16 CPUs, **no GPU and no forward model anywhere** — nothing in this folder
imports, trains or evaluates an `f(s,u)`. The trunk predicts no state; it reproduces what the
reflex law would command from the read it has.
**Attribution**: the framing is Jasper's — re-internalizing the macros into the policy is the goal,
the multi-level ladder is set aside on this environment, the FM's suspected role is as the
*certifier* of a macro, so the test is whether re-internalized macros go far with no FM,
consulting an oracle (the resettable plant) where required. The roles table, the trunk without a
model, and the tape-as-intention-reference reading came out of the exchange; the absolute-band
probe was Jasper's call on reading `s0`. One orchestrated conversation, one implementer agent.
**Single seed; ranks, signs, bit-identity twins, cross-tag exact controls and pre-fixed gates are
the claims.**

## The question, and what happened to it

Does [`rhm/practice/native/`](../../../rhm/practice/native/README.md)'s consolidation — a committed
motor vocabulary *proposed* by the learner's own routing head, *emitted* by a corridor head on its
own executor, the table deletable, trust formed by use — happen on a plant with nothing imaginary
in the loop, at the delay where committed content is measured to pay? It does. All three signatures
appear at segment span, model-free, and adoption tracks the meter in both directions. What the
learning did **not** buy is error: the routed learner's best selector was the prior it started
from, and a probe that starved the imitation filter left it there. Learning bought a fourfold cut
in groundings and priced time, a tenfold tighter poison quarantine, and a corridor head that plays
its units a little better than the tapes it was trained on.

## Findings

1. **The apparatus is clean, and the delay is the only variable against acappella.** S-F1
   reproduces `acappella/b1`'s library build and its Δ = 0 *and* Δ = 8 rows at 0.000e+00 on
   fourteen checks. The behaviour-cloned trunk clears its pre-fixed band in all eight cells of
   {trunk Δ} × {clean, rotated} × {eval Δ}; the operating cell is the loosest, 0.1869 against the
   PD law's 0.1697 (band 0.0424), so every "against the reflex" margin below is against the
   clone. `fid` (ports wired and shut) and π-at-k=N reproduce `audit_all` at 0.000e+00 over all
   120 metered cycles. (`s0`)
2. **Routing consolidates.** π's top-2 over slots reaches 0.1212 at 4 feedback events and 36
   groundings per traversal, against enumeration's 0.1142 at 8.2 and 148: a fourfold cut in
   groundings and a 3.8× cut in priced time (36.6 s vs 139.7 s) at a 6% error cost. The
   enumeration arm's extra feedback is the primitive it sometimes picks (`frac_prim` 0.03), which
   costs per-step feedback. (`s0`)

   | arm, Δ = 8, last cycle | e_piece | fb / traversal | groundings | priced s | frac_prim |
   |---|---|---|---|---|---|
   | reflex through the trunk | 0.1869 | 136 | 0 | 16.9 | — |
   | donor 8-tape `lib_seg` (acappella b1, re-derived in-run) | 0.0838 | 4 | 32 | 33.0 | 0.00 |
   | `key_seg` | 0.1312 | 4 | 0 | 3.7 | 0.00 |
   | `audit_all`, 36 members + primitive | 0.1142 | 8.2 | 148 | 139.7 | 0.03 |
   | `audit_prop_k`, Port 1 | 0.1212 | 4 | 36 | 36.6 | 0.00 |
   | `route_native`, Ports 1 + 2 | 0.1309 (battery base 0.1174) | 4 | 27 | 28.2 | 0.00 |
   | `prop_k_d0`, the Δ = 0 control | 0.0043 | 139 | 36 | 50.1 | **0.99** |

3. **Adoption tracks the meter in both directions.** At Δ = 8 the routed arms launch the primitive
   0.00 of the time by cycle 12, from 0.05 at cycle 0. At Δ = 0 the identical machinery routes to
   the primitive 0.99 of the time, with π's mass on it at 0.997 from the first probe. The learner
   plays by feel where feel wins and from memory where memory wins, and the switch is π's, not a
   rule's. (`s0`)
4. **Trust forms, and routing quarantines the poison tenfold.** Under `route_native`, π's segment
   mass goes 0.729 → 0.977, primitive mass 0.239 → 0.020, poison mass 0.032 → 0.003; entropy
   1.477 → 0.726. Poison launches over the run: 3.23% under enumeration, 2.81% under Port 1
   alone, **0.76%** under both ports, 0% at Δ = 0. `fid`'s untrained π stays exactly uniform
   (entropy ln 10) at every probe. The Port-1-only arm's poison mass barely decays (0.032 →
   0.025) while the two-port arm's falls tenfold; the only difference between those arms is that
   `route_native` consults the span head. (`s0`)
5. **The address book is deletable, and the trained heads carry the content.** Deleting the
   library at zero groundings *lowers* error in every trained arm — 0.1174 → 0.1142
   (`route_native`), 0.1219 → 0.1146 (`audit_prop_k`), 0.1142 → 0.1026 (`audit_prop_kN`) — while
   the untrained-head control reads 0.7176. `restored ≡ base` at 0.000e+00 everywhere. At Δ = 0
   the picture inverts as it should: ablating the primitive costs 0.0043 → 0.0516, and heads
   alone read 0.1103. This is the first mjc node where the battery is interpretable; offbook's
   `no_table` fell back to a live plan. (`s0`)

   | arm | base | no_prim | no_table | no_table_no_prim |
   |---|---|---|---|---|
   | `fid` (untrained heads) | 0.1142 | 0.1142 | **0.7176** | 0.7176 |
   | `audit_prop_kN` | 0.1142 | 0.1142 | 0.1026 | 0.1061 |
   | `audit_prop_k` | 0.1219 | 0.1224 | 0.1146 | 0.1146 |
   | `route_native` | 0.1174 | 0.1174 | 0.1142 | 0.1164 |
   | `prop_k_d0` (Δ = 0) | 0.0043 | 0.0516 | 0.0044 | 0.1103 |

6. **The corridor head beats the tape where parity opens.** Parity is execution reproduction on
   the plant (head vs the audition's chosen member, both executed from held-out seam states split
   by a deterministic state code; open iff the head is at least as good on 75% of them). Seven of
   23 slots are open under `route_native` at the last cycle, and on those the head's execution
   error is 0.005–0.04 below the tape's (slot 18: 0.1160 vs 0.1545). Together with finding 5 this
   is the SPEC's within-chunk question answered in the direction the piano analogy predicts: what
   internalizes is a posture-conditioned program, not a recording. Small margins. (`s0`)
7. **Learning bought cost and delay-robustness, not error — and the prior was the best selector.**
   Both routed arms start at 0.0955, where the untrained π amounts to the construction-order
   top-2, climb to ~0.15 over cycles 2–20 as π first moves off that prior, and settle at
   0.12–0.13; the donor's 8-tape audition reads 0.0838 on the same pool. The probe (`s1q`)
   replaced the relative imitation filter (`good = ep ≤ median(ep)`, which turns out to be
   offbook's "competence band" too) with two pre-fixed absolute bands. Both starve: at 0.1066
   (étude's playability guard) the pass fraction is 0.065 / 0.048 with 124 / 92 target rows over
   30 cycles; at 0.0838 (the donor audition) it is 0.013 / 0.015 with 24 / 28 rows. The arm that
   never trained sat at exactly 0.0955 through cycle 29, the best error of any routed arm in
   either run. The probe is exactly `s0` plus the filter (S-T2 and S-M at 0.000e+00), and the
   band-vs-median comparison is confounded with the *amount* of imitation, 8–40×. (`s0`, `s1q`)

   | filter | mean pass | π target rows / 30 cycles | `audit_prop_k` c29 | `route_native` c29 |
   |---|---|---|---|---|
   | median (s0's) | 0.500 | 960 | 0.1221 | 0.1176 |
   | band 0.1066 | 0.065 / 0.048 | 124 / 92 | 0.1138 | 0.1176 |
   | band 0.0838 | 0.013 / 0.015 | 24 / 28 | 0.0955 at every cycle | 0.0961 |

8. **The certifier is the body, consulted through the resettable plant.** Content is admitted by
   étude's compile op on the plant (uniformly drawn renditions from the reflex's own traversals,
   replayed from held-out hand-over states, best 32 of 64 kept). A slot is trusted because
   traversals that used it came out in the better half by realized error. The head replaces the
   tape because, executed, it does at least as well. Timing has no certifier at all — the library
   is harvested late on a fixed schedule, since the primitive never learns and there is no mastery
   to detect (crystallize's finding on a frozen plant). The tape-referenced δ-silence, the
   model-free analogue of étude's FM certificate, is logged per slot and consumed by nothing:
   0.44–0.60 under delay, 1.00 without it. (`s0`)
9. **Smaller banked instruments.** Seam information under the delayed read: per-state oracle gains
   2.37× / 1.75× / 1.52× / 2.40× with 5–8 of 9 distinct argmins, so keying is live at Δ = 8. The
   36-member treatment library is a worse object than the donor's 8 tapes (0.1142 vs 0.0838):
   more candidates auditioned from a Δ-old read is not the same object as fewer, better ones —
   acappella flag 6's mechanism at the level of candidate count. Pinning torch to one thread cut
   the per-cycle cost tenfold against a 16-process MuJoCo pool. Native's can't-decompose readout
   has no segment-span form here (a segment's "spelling" is the one primitive slot every segment
   shares, so it collapses to `mass_prim`, which is already the adoption readout). (`ssize2`, `s0`)

## Interpretation (discussed with Jasper 2026-09-04 → 05 — argued, not measured)

- **The genuine-automatization half that was missing on mjc is present.** Behavioural automaticity
  — a stored rendition executed open-loop that beats closed-loop control — has been on this
  substrate since étude and model-free since acappella. What no mjc node had shown is the
  representational half: a unit the learner's own planner proposes and executor emits, with trust
  formed by use, surviving deletion of the external table. That is findings 2–6, with nothing
  imaginary anywhere and the resettable plant as the only oracle. Under Jasper's mapping of RHM
  levels onto automatic execution span, the mjc row now has its second rung re-internalized; the
  chain rung is unread on this piece (acappella finding 5) and was set aside on purpose.
- **The value of the vocabulary here is cost and delay-robustness, not accuracy.** On RHM routing
  beat enumeration on error because enumeration paid for width at a grounding budget. Here
  groundings cost 0.87 ms and the plant audition is exact at Δ = 0, so there was no width to buy;
  what routing could buy was a posture read in place of a stale rollout, and the twofold feedback
  cut plus the tenfold poison quarantine are that purchase. The model-free regime turned out to be
  the natural one for re-internalization, not a handicap: chunks pay only where the incumbent
  cannot imagine (accompanist finding 3), RHM's port ran on an inert plant (ratchet), and nothing
  modelled should enter a library (span F2/F4).
- **Self-imitation on the body's grade under delay had little to teach π.** The construction-order
  prior was the strongest selector seen in either run, and every amount of imitation moved π away
  from it. The candidate mechanism — under delay the grade is only weakly a function of the slot
  chosen, so a relative filter admits posture luck as competence — survived the probe in the weak
  sense that less imitation was better, but the probe confounds selectivity with volume, and the
  bands were performance-tempo quantities applied to practice traversals under noise and
  exploration. A matched-fraction random filter is the control that separates them.
- **On the FM as certifier.** At segment span, on this piece, every certifying role the model
  played in the lineage was filled by executing on the plant. This does not say an FM would have
  certified worse; it says certification happened without one. Where a model would re-enter is
  specific: an intention reference for a *live* action the learner has not yet compiled, and a
  mastery detector for a primitive that learns — neither of which this node has, by design.

## Caveats

- **The reflex reference is the clone**, 0.1869 against the PD law's 0.1697 at the operating cell,
  inside the pre-fixed band and reported rather than tuned. Every committed arm also beats the law.
- **The treatment library differs from the donor's** (36 members vs 8 tapes, 0.1142 vs 0.0838 under
  full audition), so within-node comparisons are on the member library and finding 7's headroom
  is bounded by it.
- **The trunk is frozen** (the span loss is detached), so offbook's interference question is
  deferred and the plant guard is a control here, not a treatment readout. The implementer's call,
  recorded with its reason; an adapting trunk is the first variant to run.
- `route_native`'s last-cycle row (0.1309) sits above its own battery base (0.1174) because the
  parity set moves between cycles; its series is non-monotone in a way no other arm's is. Parity
  holds are uneven (4 to 64 states per slot), and poison slots at seams 1–3 never fill a hold
  buffer, so their parity is undefined rather than failed.
- The probe is 30 cycles and `s0`'s median arm was still descending at c29.
- `s1q` is complete in substance and `complete=False` on disk: a variable rebinding in the δ_perf
  block crashed the runner after every arm and battery had been saved, so S-M was recovered post
  hoc from the two saved records with the runner's arithmetic; `done.txt` was deliberately not
  written. The per-arm save is what saved the run.

## Runs on disk

| tag | what |
|---|---|
| `ssmoke` | `--quick` smokes of the full chain, every gate exercised before each launch |
| `ssize`, `ssize2` | sizing probes at full config; `ssize` carries the S-F1 / S-T1 gate record, `ssize2` the per-arm cost table |
| `s0` | the run: 8 arms × 120 cycles, Δ = 8 with the Δ = 0 adoption control, binding trunk gate, battery, poison twin, δ_perf instruments |
| `s1q` | the imitation-filter probe: two absolute bands vs the median twin, 30 cycles, S-T2 and S-M against `s0` |

## Reproduce

```bash
cd experiments/                      # MODAL_PROFILE=chromatic
modal run mjc/practice/solo/solo.py::solo_run --quick --tag ssmoke
modal run --detach mjc/practice/solo/solo.py::solo_run --spawn --tag s0 --seed 0
python3 mjc/practice/solo/analyze_solo.py --tag s0 --fetch
python3 mjc/practice/solo/analyze_solo.py --tag s0 --figures
modal run --detach mjc/practice/solo/solo.py::solo_run --spawn --tag s1q --seed 0 \
    --deltas 8 --n-cycles 30 --ref-tag s0 \
    --arms reflex,key_seg,audit_all,audit_prop_k,audit_prop_k_ref,route_native_ref,audit_prop_k_lib,route_native_lib
python3 mjc/practice/solo/analyze_solo.py --tag s1q --fetch --figures
```

Volume `mujoco-control-data`: `/data/practice_solo/{s0,s1q}/`; fetched copies, reports and
figures under `results/<tag>/` and `figures/<tag>/`.

## Next steps (queued, not started)

- **A matched-fraction random imitation filter** (the tacet/offbook idiom), to separate the
  selectivity of the band from the volume it removes; and a band frozen from cycle 0's
  practice-condition errors, which is absolute thereafter and actually admits traversals.
- **An adapting trunk**: does the head-vs-tape margin move, and does offbook's interference
  question have a model-free answer.
- **δ-silence against the tape as the parity or trust gate** — the direct test of the certifier
  hypothesis, on machinery that already logs it.
- **The presto 120 ms-segment piece under this node's discipline**, the prerequisite for any chain
  rung on this substrate; and Track F's trust instruments (targeted rehearsal of a received
  vocabulary, exposure vs credit at matched budget), which now have a model-free plant.
- `/update-beliefs` for the unit: re-internalization is FM-independent at segment span; the
  certifier is the body through the plant; the construction prior beats body-graded self-imitation
  under delay.

[^private]: Not mirrored: this link points to a document in the private lab repo (the roadmap, the queue, an unrun spec, reading notes, or a conversation). See the top-level README for what is held back and why.
