# The temporal confabulation test: is the joint system's privileged access epistemically charged?

**Status**: designed, not run. Pre-registered here before any implementation.
**Date**: 2026-08-10
**Builds on**: [confabulation_test.md](confabulation_test.md) (the battery, run on the depth axis —
[rhm/confabulation/](../experiments/rhm/confabulation/README.md),
[a2a_forward/confabulation/](../experiments/a2a_forward/confabulation/README.md)) ·
[revision_not_surprisal.md](revision_not_surprisal.md) (the conditioning-gap taxonomy) ·
[`rhm/conditional_revision/`](../experiments/rhm/conditional_revision/README.md) (Gate B: the temporal
residual carries revision content, directionally) ·
[`rhm/conditional_revision/rule_family/`](../experiments/rhm/conditional_revision/rule_family/README.md)
(the readability law this design must respect)
**Evidence re-read**: [`aleatoric_fraction/`](../experiments/rhm/conditional_revision/aleatoric_fraction/README.md)
(the temporal residual is 0.665 aleatoric — the charge is real) ·
[`a2a_forward/conditional_revision/`](../experiments/a2a_forward/conditional_revision/README.md)
(residual *magnitude* is output entropy at R² 0.901; *direction* separates at 0.996) ·
[`local_loss/`](../experiments/rhm/conditional_revision/local_loss/README.md) (training on the temporal
target reproduces the depth signature — a live reason the answer here could be no)
**Reading**: Binz, Dasgupta, Jagadish, Botvinick, Wang & Schulz (2023), *Meta-Learned Models of
Cognition*[^private] — the canonical citation for this
doc's premise that a trained model's forward pass *is* amortized Bayesian inference (their Box 1:
the converged objective's optimum is the posterior predictive). Two of its threads anchor this design:
Argument 4's Wang et al. 2018 reading (a slow dopaminergic loop *builds* a fast in-activation
updater — the biological frame for reading revision off activations at all), and §4.1's lament that
meta-learned models offer "no underlying mathematical expression" to inspect — which is what running
the battery against an exact belief oracle answers: the composite's report is scored against the
analytical object their framework says the model approximates.
**Attribution**: the question is Jasper's, asked in two halves: *"if the main model + FM system is
introspective, would the token offset make that introspection epistemically-charged rather than
epistemically-inert?"* and *"the predictive-distribution revision does you no good if that signal is
real but cannot be consumed natively by the main model itself — would the temporal FM allow for
this?"* The observation that the observer ladder makes the advantage beyond-surprisal by construction
came out of the exchange.

---

## One-liner

The confabulation battery established that the composite (M + FM) has **privileged access** to the FM
residual — a first-person advantage no I/O-limited observer matches at any capacity — but the *content*
of the depth residual is computational (capacity shortfall), i.e. **privileged access to an
epistemically inert quantity**. The conditioning-gap taxonomy says the residual's content is set by
what fills the gap. Shift the FM's target by one token and the gap holds the world's next input: the
same decomposition now carries **what the token just taught the model**. The conjecture:

> **Depth FM: the joint system has privileged access to "how I compute" — charged with nothing.
> Temporal FM: the joint system would have privileged access to "what I just learned" — and it is
> private precisely because its I/O-visible shadow, surprisal, is exactly what it is not.**

## Why the two halves are independent, and why neither is settled

**Privacy is axis-independent.** Criterion (2)'s argument — reconstructing `r` from outside costs
rebuilding M and FM, so the residual is private in the precise sense that knowing it requires being
the system — nowhere depends on whether the target is six blocks down or one token ahead. Nothing
about the token offset should break the privilege. But this is an argument, not a measurement.

**Charge is measured, but access to it is not.** The temporal residual has genuine
aleatoric + epistemic composition (`aleatoric_fraction`: 0.665 of the ideal arity-1 residual), and its
*direction* carries revision content under full matching (Gate B: 0.53–0.61). What no experiment has
asked is whether the **composite's own report channel** reaches that content with a first-person
advantage — every revision readout so far has been an external probe trained on true latents.

**Nothing so far falsifies the conjunction.** `rule_family`'s Gate 1 null does not bear on it, for
three reasons: it read the state through an external probe (the polar opposite of a self-report riding
on `r`), it had no FM in the loop at all, and it was resolution-limited in a quantified way (probe at
11–17% of its Bayes ceiling, on the parent's own decodability curve). The story is untested, not dead.

## The elegant property: the observer ladder builds in the surprisal control

In the battery, `O_io` receives M's tokens **and full output distribution** — so it can compute
surprisal and entropy *perfectly*. On the depth axis this made `ENT` the sharpest control (the
observer beats the self-report on I/O facts, −0.287). On the temporal axis it does something better:

> **Any first-person advantage on a temporal-residual report is, by construction, the component of
> revision that surprisal cannot account for.**

Gate B needed atom-aware strata to pin surprisal; the observer ladder pins it architecturally. The
two disciplines are the same control implemented in different substrates, and this design gets it for
free.

## Why this and not the predictive-KL alone

The model's own predictive-distribution revision — `KL(p(future | x_{≤t+1}) ‖ p(future | x_{≤t}))` —
is the *operational* revision object, oracle-free and probe-free, and it remains the **calibration
standard** here. But it is an analyst's quantity: computing it requires holding the previous forecast
and differencing across time, which the model does not natively do — the comparison happens outside
the system, after the fact. A signal the system cannot touch cannot gate, teach, cancel, or allocate.

The temporal FM is the organ that fixes this: it **materializes `p⁻` in activation space, at the right
time**, so that when the token arrives the difference is physically present on a wire inside the
system — consumable by a gate, a report head, a salience signal. That is the cerebellar reading this
program has carried throughout (§2 of the parent idea doc: *timing governs use*), and the confabulation
results supply the missing empirical premise: composites demonstrably consume their FM residual
natively (the report rides on `r`; behaviour is causally sensitive under matched-KL steering).

Two constraints from the corrections, so this does not repeat known failures:

- **The consumable channel is the residual's direction.** Magnitude is output entropy
  (language sibling: R² 0.901 vs 0.0001 for revision) and is pre-registered here as the *negative
  control*, not a readout. The battery already reports direction (`IMPL_COS`).
- **Consumption means reading, never minimizing.** §8 of the parent idea doc and `local_loss`
  established that minimizing this prediction error is a simplicity/predictability objective with
  known degeneracies. Gating and reporting on the signal is a different operation, and it is the only
  one proposed here.

## Design

Fork the RHM battery ([`rhm_confabulation.py`](../experiments/rhm/confabulation/rhm_confabulation.py))
with the decomposition swapped to the temporal axis, on the **incumbent m4 substrate** — where the
charge is fat (Gate B fires at d2 0.690; the probe is at 82–90% of ceiling at d1–d2) — not on
`rule_family`'s 0.008-nat whisper. Suggested home when run:
`experiments/rhm/confabulation/temporal/`.

- **Decomposition**: `Δ_t = h6[t+1] − h6[t]`; `r_t = Δ_t − FM(h6[≤t])` with the conditional_revision
  Gate-0 temporal FM idiom (frozen M, matched heads). The depth battery's decomposition
  `a_j = FM(a_i) + r` is retained as the within-experiment comparison arm.
- **Report targets**: `TEMP-IMPL` / `TEMP-IMPL_COS` (direction cluster / direction of the temporal
  residual), against the battery's standing controls (`BEHAV`, `ENT`, `WORLD`) *plus* the depth-IMPL
  arm — the direct inert-vs-charged comparison at matched machinery.
- **The charge validation, which only this substrate offers**: the exact oracle gives per-position
  belief revision `B_t`. So we can ask not only *is access privileged* (the advantage) but **is the
  privileged part the revision part** — regress the self-report's advantage structure against oracle
  `B_t` with surprisal matched. The depth arm is the built-in null: its residual should carry no
  `B_t` structure beyond surprisal.
- **Guards, all inherited and mandatory**: the junk-residual trap (instrument capacity sweep with
  `ens_cos` and hierarchy-η² discriminators — the smoke-run artifact is documented and reproducible);
  matched heads; the observer data-budget control; permute-inputs-not-labels; and the battery's own
  finding that raw `self` numbers are not comparable across instruments — only margins and advantages.

## Predicted result table

| readout | prediction |
|---|---|
| `TEMP-IMPL_COS` first-person advantage | positive, surviving the capacity sweep |
| advantage's correlation with oracle `B_t` (surprisal-matched) | positive — the privileged part is the revision part |
| depth-IMPL arm, same regression | ~0 — privileged but inert, as established |
| `ENT` / `BEHAV` / `WORLD` | observer wins or ties, as on the depth axis |
| magnitude-based temporal report | tracks entropy; no advantage beyond `O_io` — the declared negative control |

## What would falsify this

- **`O_io` matches the self-report on `TEMP-IMPL`.** Then the temporal residual's private part is nil
  — everything the token did to the model is recoverable from the I/O map — and the charged-introspection
  claim fails at criterion (2). This is a real possibility: the token *is* visible to the observer,
  unlike the depth residual's drivers.
- **The advantage exists but is uncorrelated with oracle `B_t` given surprisal.** Privileged access to
  an epistemically inert temporal quantity — the depth result again, one axis over. `local_loss`'s
  finding that temporal training targets reproduce depth signatures makes this the most live negative,
  and it would be a genuine finding: the axis change moves the residual's *composition* but not what
  the composite can *report about it*.
- **The junk-residual signature** (advantage largest where `ens_cos` is lowest). The artifact is
  documented; if it appears, report it as such.
- **The depth-vs-temporal comparison is confounded by FM quality** — the two FMs sit at different
  points on their capacity curves. The instrument sweep must bracket both; if the comparison only
  holds at one capacity point, it does not hold.

## What this does not establish, stated up front

Everything the depth battery's scope notes say (trained report head; use ≠ report ≠ awareness; the
introspection belongs to the composite, not M alone) carries over unchanged. Two additions: a positive
result here is about *reportable access*, not about the signal being *used* for control or learning —
the consumability arm (gating on the wire) is a separate, later experiment; and the loop question
(does closing the loop amplify, as on RHM-depth, or not, as on language-depth) is inherited unresolved
and this design does not settle it — run OL first, CL as the follow-up arm.

## Open questions

- If the advantage is real and revision-correlated, can the report channel be read *by the model's own
  downstream computation* rather than by a trained head — i.e. does the wire get used when it is
  merely available? That is the full native-consumability question, and it is the bridge from
  measurement to the two-timescale program's teaching interventions.
- The martingale calibration: under the model's own sampling, the temporal forecast's revision should
  be drift-free; under corpus text its drift is a miscalibration readout. Cheap to add to any run.
- Does the observer-ladder shape (flat in capacity — access-limited) survive the axis change? On the
  depth axis this was the sharpest single result; if the temporal ladder *climbs*, the privacy
  argument needs revisiting for world-facing content.

[^private]: Not mirrored: this link points to a document in the private lab repo (the roadmap, the queue, an unrun spec, reading notes, or a conversation). See the top-level README for what is held back and why.
