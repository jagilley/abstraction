# SPEC — merge: invariance-by-enumeration vs invariance-by-merge, and whether the unmetered monolith buys the quotient

**Status**: spec, unbuilt (2026-08-17). Operationalizes
[`recurrence_manufactures_confounds`](../../../../ideas/recurrence_manufactures_confounds.md) §8,
plus one arm added by a 2026-08-17 conversation with Jasper: the unmetered dense learner. Sibling
spec from the same conversation: [`../reread/SPEC.md`](../reread/SPEC.md).
**Up**: [`../README.md`](../README.md) (practice arc) · machinery donors:
[`../setlist/FILES.md`](../setlist/FILES.md) (OU demand drift, calibration discipline) ·
[`../crystallize/FILES.md`](../crystallize/FILES.md) (library/commit machinery, both ends of the
key-granularity axis)
**Attribution**: the challenge this spec tests is Jasper's — the suspicion that AlphaZero-style
learning from effectively unlimited data is a degenerate solution but not a *correct* one relative
to a hypothetical full-cognitive-loop player; that performers deliberately *seek out* drift
(line dancing's wall rotation) as a refinement technology, so metering and drift are instruments a
mature loop wields on itself rather than properties some worlds happen to have. The generalization
of the meter to any anti-enumeration pressure, the aimed-vs-diffuse compression framing, and the
enumeration-vs-merge dissociation came out of the exchange.

## Why this experiment

Two questions fused, one inherited and one new.

Inherited: the confounds doc's §8 instrument list has no experiment. The merge op — the first op in
the arc that acts on the library's **index** rather than its entries — is specified
(§5 there: coarsen the index, never the content; two evidence routes, audit-of-aliases and
forced-transfer-under-cache-miss) but unbuilt, and both ends of its key-granularity axis are
already priced ([`crystallize`](../crystallize/README.md): state-conditioning worth 1.8–3.0× where
the key carries information; per-key selection 3.3× more curse-prone).

New, with program-level stakes:
[`practice_manufactures_its_own_credit`](../../../../ideas/practice_manufactures_its_own_credit.md)
§18 calls pretraining "the correct degenerate solution when the meter is off." The challenge:
**"correct" was graded by a within-level signal** — in-distribution loss, the one currency the
monolith is built to win — and §16's own theorem (the value of level-k practice is denominated in
level k+1's currency, invisible to within-level signals) applies to the theory itself. The
confounds doc §4 already says the meter is what forces the flip from *track the key* (populate an
entry per wall — coverage by enumeration) to *quotient the key away* (merge). Remove the meter and
nothing forces the quotient: the theory's prediction is that you get enumerated coverage that is
benchmark-indistinguishable from invariance in-distribution and a different object everywhere else
— and that the difference is silent by construction
([`operators_not_footprints`](../../../../beliefs/trees/operators_not_footprints.md): the loss sees
quality, not character). So the question this experiment exists to ask: **is
invariance-by-enumeration the same object as invariance-by-merge?**

Two anchors worth holding while designing, neither of which the experiment depends on:

- *Recalled external instance (cite-check before leaning on it)*: adversarial policies against
  KataGo-class Go engines (Wang et al., ~2023) — a simple cyclic strategy, human-executable once
  learned, persisting through enormous self-play. In the
  [`question_model`](../../question_model/SPEC.md) vocabulary: self-play is training where the
  question model and the answer model are the same object — the demand distribution is the policy's
  own footprint — so channel (ii) collapses and holes sit exactly where demand never exercised an
  invariance.
- *The boundary case that sharpens the meter's definition*: grokking. Monoliths do find invariant
  circuits — canonically under weight decay, i.e. a **capacity** meter. This licenses the
  generalization the caveat in
  [`meta_learning_under_metered_data`](../../../../ideas/meta_learning_under_metered_data.md)
  records: the meter is *any* pressure against enumeration (samples, capacity, time, self-imposed
  constraint), with practice's meters aimed — per-dimension, scheduled, demand-weighted — and
  SGD's diffuse. The optional capacity-meter arm below is this case's probe.

## Reading list (ordered; first two load-bearing)

1. [`recurrence_manufactures_confounds`](../../../../ideas/recurrence_manufactures_confounds.md) —
   the parent. §5 is the merge op's design constraints; §8 is the instrument list this spec
   inherits; §9's risks transfer wholesale.
2. [`../typed_gaps/README.md`](../typed_gaps/README.md) + [`../setlist/FILES.md`](../setlist/FILES.md)
   — the demand machinery and its calibration discipline (separation-preserving loudness, measured
   before the main runs; the σ-vs-κ lesson: sweep, don't solve).
3. [`../crystallize/README.md`](../crystallize/README.md) — the library machinery and the two
   priced ends of the key-granularity axis.
4. [`practice_manufactures_its_own_credit`](../../../../ideas/practice_manufactures_its_own_credit.md)
   §16 and §18 — the clause under test and the theorem being applied to it.

## Substrate

Setlist-style RHM sculpting: grammar fixed, a **free observable context feature** attached to each
task, spuriously correlated with the true latent under a narrow initial demand distribution (the
manufactured confound), then decorrelated by rotating the demand distribution — paced, at rates
spanning slow-OU through fast-cyclic (§4's rate axis). The library may key on the observable
feature, the true latent, or any coarsening; the merge op acts on that key. Loudness of the spurious
feature is a swept, calibrated knob (§2's available-vs-taken gap). All drift knobs set by
measurement per the arc's convention, no-drift control in-run.

## Design (held loosely — the instrument list is the commitment, not the interpretation)

Arms:

- **Metered + rotation + merge**: the practice loop under priced feedback, manufactured
  decorrelation at a paced rate, merge op available.
- **Metered + rotation, track-only**: identical but the index never coarsens — populate an entry
  per key cell (the enumeration answer *inside* the metered regime; carries §8's storage/curse/
  coverage costs).
- **Unmetered dense, free variation** (the new arm — the monolith / AlphaZero analog): no feedback
  price, i.i.d. full variation of the context feature from the start. Free decorrelation, no
  manufactured recurrence, no library ops.
- *(optional, cheap)* **Unmetered dense + capacity meter**: same, with weight decay / capacity
  pressure — does diffuse anti-enumeration pressure buy the quotient slowly?

Readouts, in priority order:

1. **In-distribution parity check.** The unmetered arm should match or beat everything on the
   varied set — that is the point, and if it doesn't, the setup is miscalibrated, not the theory
   vindicated.
2. **Transforms outside the varied set** (the fifth wall): a frame transform never included in the
   rotation. This is the dissociator between a representation that *is* the quotient and one that
   enumerates its cells.
3. **Next-level minability**: run a ratchet-style mining pass on top of each arm's representation —
   what does each make representable next? (§16's currency, measured directly.)
4. **The parent doc's own instruments** (§8): early learning speed with vs without the spurious key
   (the scaffold's value); which frozen features bind as a function of loudness; key granularity
   over time, merge vs track, across the drift-rate sweep (the §4 track→merge threshold); the
   never-merge arm's carried costs.

The theory's predictions, recorded per repo norms as predictions and not as constraints: the
unmetered arm ties readout 1 and loses 2 and 3; the capacity-metered variant closes part of the gap,
slowly. If instead enumeration matches merge on every readout, then ordinary dense learning under
varied demand achieves the quotient with no discrete index event — confounds §9 already names this
outcome: §7's belief-limit claim collapses into "generalization," §18's clause survives as written,
and that is worth knowing at least as much. Bring the numbers back for discussion before writing
any README.

## Practicalities

- Invoke `/run-experiment-on-modal` first; profile `chromatic`; smoke-test before long runs; follow
  the halting procedure for anything >5 min.
- Single seed first, per `experiments/CLAUDE.md`; rank orderings over fractions; measure a noise
  floor if you need one (the arc's stream-position convention).
- Fidelity discipline: the no-drift, no-spurious-feature configuration must reproduce the setlist
  baseline; every loudness/rate knob set by a measurement, recorded in a calibration record
  ([`../setlist/FILES.md`](../setlist/FILES.md) is the form).
- Structure per `experiments/CLAUDE.md`: this folder gets its `README.md` post-discussion and a
  `FILES.md`; import `../setlist/` and `../crystallize/` machinery, don't copy or modify it.
- The merge op is the one genuinely new build. Design it against confounds §5's constraints before
  touching arms; a merge that blends content instead of coarsening the index is measuring the
  averaging catastrophe, not the quotient.
