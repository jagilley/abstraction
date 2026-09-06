# The temporal confabulation test: is the joint system's privileged access epistemically charged?

**Status**: **OL arm run** (2026-08-10, cloud agent) — the second pre-registered
falsifier **fired cleanly**, and the doc's conjecture is answered: privilege survives the axis change,
charge is present and reportable, and **their intersection is exactly empty**. Two pre-registered
predictions were contradicted by the run and are left standing below rather than edited out (the
magnitude negative control showed the *largest* advantage; the flat-ladder prediction held but for a
different reason than argued). Results and the reinterpretation are appended; `cr_base` (frozen
Gate A/B checkpoint) is the pending robustness arm; the CL arm is deliberately not run (§Results).
**Date**: 2026-08-10 · **Results appended**: 2026-08-10
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

**Post-run verdicts, kept against the predictions**: row 1 held (+0.014–0.027, surviving the sweep).
Row 2 **failed** — the advantage carries zero `B` structure; that failure is the headline finding.
Row 4 held (controls reproduce the depth battery's published row). Row 5 was **contradicted**:
`TEMP-MAG` shows the *largest* temporal advantage (+0.048 vs `TEMP-IMPL`'s +0.014; `DEPTH-MAG` +0.066
vs `IMPL`'s +0.059). Partly compression-near-ceiling (4-way vs 8-way targets; `TEMP-MAG` has more
headroom), but "no advantage beyond `O_io`" is falsified either way. The recorded diagnosis: the
pre-registration conflated two claims — "magnitude is entropy-like *as a revision readout*" (still
true; the magnitude advantage carries no `B` structure either) and "the magnitude *fact* is
I/O-public" (false: `|r|` includes an FM-capacity component, which is implementation-borne — so it is
privileged-but-inert, the depth result in miniature).

## What would falsify this

- **`O_io` matches the self-report on `TEMP-IMPL`.** Then the temporal residual's private part is nil
  — everything the token did to the model is recoverable from the I/O map — and the charged-introspection
  claim fails at criterion (2). This is a real possibility: the token *is* visible to the observer,
  unlike the depth residual's drivers.
  → **Did not fire outright, but the named exposure is the story.** The advantage survives at
  +0.014–0.027 (control band +0.002–0.011) — a third to half of depth's — and `O_input` reads **0.82
  against a self-report of 0.90** (depth axis: 0.57 vs 0.79). The temporal residual is overwhelmingly
  token-determined.
- **The advantage exists but is uncorrelated with oracle `B_t` given surprisal.** Privileged access to
  an epistemically inert temporal quantity — the depth result again, one axis over. `local_loss`'s
  finding that temporal training targets reproduce depth signatures makes this the most live negative,
  and it would be a genuine finding: the axis change moves the residual's *composition* but not what
  the composite can *report about it*.
  → **Fired, cleanly, with a sharper mechanism than the prediction imagined.** Partial
  `R²(advantage ~ B | surprisal)` = **0.0000** to four decimals at every capacity and level (shuffled
  0.0001–0.0002), while the same advantage carries 20–50× more surprisal structure. The decomposition
  is the finding: self-report and observer track oracle `B` **equally well** (0.0063–0.0078 vs
  0.0066–0.0074) and their difference is exactly zero, with 12/12 cells showing the advantage
  *shrinking* where `B` is large (depth arm: 6/12, a coin flip). So it is not a reportability failure —
  the revision content is present, in M's state, and reported fine; **it is simply not private**.
  Scale-matched steering agrees (0.96–0.98× vs depth's 1.32–1.71×).
- **The junk-residual signature** (advantage largest where `ens_cos` is lowest). The artifact is
  documented; if it appears, report it as such.
  → Did not fire: `ens_cos` flat at 0.939–0.945 across the sweep, cosine 0.77–0.85. One residue is
  unexplained: the temporal advantage *rises* with instrument capacity (+0.014 → +0.027) while depth
  is flat — not the junk signature, but not accounted for.
- **The depth-vs-temporal comparison is confounded by FM quality** — the two FMs sit at different
  points on their capacity curves. The instrument sweep must bracket both; if the comparison only
  holds at one capacity point, it does not hold.
  → The sweep bracketed both; the qualitative contrast (temporal advantage a fraction of depth's,
  charge-null at every point) holds across it.

## Results — OL arm (2026-08-10, cloud agent)

**The harness is a controlled fork and both anchors reproduce.** Wake val 1.5447 against the depth
battery's published 1.5447; the depth arm re-run at the temporal report positions reproduces the
published OL row (advantage +0.053–0.059 vs +0.06; margins, steer ratios, and all three control
targets within noise); the oracle independently reproduces `conditional_revision`'s DGP table
(`R²(B ~ exact surprisal)` 0.072/0.112/0.223 at d4/d3/d2 vs published 0.071/0.115/0.230). Regime
certified non-junk (temporal `ens_cos` 0.94, cosine 0.77–0.85). Guards clean throughout (surprisal
self-match 0.4995–0.5040, position-shuffled null 0.4972–0.5018).

**The answer to the conjecture: both conjuncts hold, their intersection is exactly empty.**

- *Privilege survives the axis change, diminished*: `TEMP-IMPL` advantage +0.014–0.027 against a
  +0.002–0.011 control band — a third to half of depth's — with the mechanism visible in `O_input`
  (tokens alone: 0.82 vs self 0.90; depth axis: 0.57 vs 0.79).
- *Charge is present and reportable*: the self-report tracks oracle `B` at partial `R²` 0.0063–0.0078
  given exact surprisal.
- *But the observer tracks it equally* (0.0066–0.0074), the advantage carries **0.0000** of it, and
  the advantage *shrinks* where `B` is large (12/12 cells) — because high-`B` positions are exactly
  where the token is most informative, and the token is what the observer has.

**The two flavors of ladder-flatness** (the diagnostic worth keeping): the ladder is flat on both
axes (`O_io` 0.871 → 0.868 from 1L/64D to 8L/256D; half-data 0.858), but on the depth axis it is flat
far *below* the self-report (the observer can't get more), while on the temporal axis it is flat just
*under* it (the observer already has nearly everything). Same signature, opposite meanings. The
`O_h6` ceiling ordered correctly for the first time in this line (0.905 > self 0.899 > `O_io` 0.884 >
`O_act` 0.853), so the ladder finally has a validated top.

**Arms**: `cr_base` (the exact frozen Gate A/B checkpoint, cached, no training) is running as the
wrong-substrate check. The CL arm is deliberately **not run**: loop closure changes how well M encodes
its residual, not whether the evidence stream is public — no plausible path from there to moving a
0.0000 partial `R²` — and the ~2h retrain is better spent on §Next below.

## The reinterpretation: privilege attaches to the manner, never the content

> **Introspective privilege attaches to *how* you process, never to *what* you learned from public
> evidence — because the world taught it to you in public.**

Revision driven by a shared evidence stream is public whenever the update map is learnable: the
observer holds the same tokens M does, M's belief update is a function of (prefix, token), and a
2-layer network suffices to learn that function here. What remains private is the
implementation-idiosyncratic manner of updating — which carries no `B`. Combined with the depth
battery, criterion-(2) introspection is now cornered into implementation facts from both sides: I/O
facts (entropy, behaviour) are third-person property, and evidence-borne belief revision is
I/O-*adjacent* property. The doc's own slogan inverted: surprisal was supposed to be revision's
public shadow — revision is public too; only the manner of updating casts no shadow.

Two scope limits keep this from over-generalizing, and both are live research directions rather than
caveats: the mechanism requires (a) a **shared evidence stream** — in generation the evidence partly
originates inside the model, exempting that regime — and (b) a **cheaply learnable update map** — at
frontier scale, simulating the update may genuinely approach the cost of being the model, which
reopens the design doc's "scaling curve, not point estimate" note as the live question for language.

Jasper's framing of where this points (recorded as the program-level lesson): introspection should pay
precisely where feedback is **sparse and evaluative** — where the public signal under-determines the
credit trace and the missing bits are implementation-borne — and not in pretraining-like regimes,
where dense public supervision determines credit completely. That retro-unifies the arc: every null
in this line tried to cash self-signals during the dense-public phase; the only self-knowledge
results with behavioral teeth (the bilevel learning gate at SK R² 0.83; MJC's corrected FM-error
teacher) were already sparse-feedback uses. One methodological law falls out immediately: **any
future "introspection helps X" claim must beat its observer-simulated twin** — the confabulation
battery is now the control-arm construction, not the experiment.

## The next thing to run: temporal FM epistemics, on this harness

**Built** (2026-08-11) as
[`rhm/confabulation/temporal/epistemics/`](../experiments/rhm/confabulation/temporal/epistemics/README.md).
Everything below is implemented as specified. Two things were added on top of it, both because the
counterfactual machinery the A/E target needs supplies them for free: an **exact `ε₁ = 0` source**
`r_mart = h[t+1] − E_{a∼p_M}[h[t+1] | x_{t+1}=a]` (the residual a forecaster with zero compression
error would leave, so §6's `ε₁` is *measured*, not bounded, and the concentration question acquires a
ceiling), and the exact form of the martingale add-on — under `p⁻_mart` the drift null holds *by
construction* on self-sampled continuations, so the corpus cell reads the model's calibration error
projected onto the state-update map. A one-hot-position **floor** was added alongside the shuffled
null, since on the RHM everything is position-coupled.

The privacy question is settled; the operational question this program actually cares about — **does
the temporal FM's signal carry genuine epistemic content, and does the FM do any epistemic work in
producing it?** — is now cleanly separated from it, and this harness is the right instrument: it
reads the temporal channel at 0.899 against a validated 0.905 ceiling, where Gate 1's belief probe
died at 11–17% of its ceiling. Publicity is a *feature* here: no privileged access is needed to
measure the channel.

Repoint the battery from privacy to composition — drop the report channel, steering, and CL; keep the
harness, guards, sweep, and oracle. A **source × target decode matrix**:

- **Sources**: the residual `r = Δ − FM(h≤t)` · the raw update `Δ` · the raw state `h[t+1]` · the
  forecast `p⁻ = FM(h≤t)` alone · (add-on) the explicit two-forecast pair `FM₂(h≤t, x_{t+1}) − FM₁(h≤t)`.
- **Targets**: graded `B` (partial, given exact surprisal) · the syn/dis contrast at matched surprisal
  (Gate B's aleatoric-null test, at this readout resolution) · `aleatoric_fraction`'s A/E class.
  Magnitude readouts run as the standing entropy negative control.
- **The pre-registered crux is `r` vs `Δ` at matched readout capacity.** The FM's epistemic claim as
  an organ is *concentration*: subtracting the forecast should cancel the predictable carried-state
  part and make `B`-content more accessible from `r` than from `Δ` (higher decode at small probe
  capacity). The language sibling's `r_dir ≈ h_after_dir` says the FM merely inherited there; this
  measures the concentration curve against an exact oracle. If `r` dominates `Δ` and is
  FM-capacity-invariant, the temporal FM is an SNR device for revision and downstream uses should be
  fed the residual. If `r ≈ Δ` everywhere, the FM is epistemically inert as a signal-former, and its
  remaining defensible role — materializing `p⁻` at the right *time* for in-loop consumption — is a
  claim about control topology, to be tested in sparse-feedback settings, not by more measurement.
- **Gate C folds in for free**: demanding the `B`-decode from `r` be flat across the four-point
  instrument sweep is the idea doc §6 capacity-invariance requirement (`revision` vs `ε₂ − ε₁`),
  pre-registered since the SPEC and never run.
- Cheap add-on: the martingale calibration — on self-sampled continuations the revision decodes
  should read zero-mean; drift under corpus text is a miscalibration readout.

Prior from this run: the report head — trained for cluster identity, not for `B` — already carried
`B` at ~0.007 partial `R²`; a `B`-trained head bounds that from below. The live range spans "weak but
real revision channel that the FM concentrates" to "surprisal plus capacity noise, concentration
nil," and either end is decision-relevant: it determines what signal the sparse-feedback
("density-dial") experiments should consume.

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
  → **Answered, with a twist**: flat (0.871 → 0.868, half-data 0.858), but flatness means the
  opposite thing on the two axes — depth's observer is flat because it *can't get more*, temporal's
  because it *already has nearly everything*. The ladder's asymptote relative to the self-report, not
  its slope, is the informative number.
- **From the run, unexplained**: the temporal advantage rises with instrument capacity
  (+0.014 → +0.027) while depth's is flat, with `ens_cos` flat throughout — not the junk signature,
  and not accounted for by any current reading.

[^private]: Not mirrored: this link points to a document in the private lab repo (the roadmap, the queue, an unrun spec, reading notes, or a conversation). See the top-level README for what is held back and why.
