# question_model — typing a token's news: structure, question, noise

**Up**: [../README.md](../README.md) (rhm) · **Spec**: [SPEC.md](SPEC.md) · **Files**: [FILES.md](FILES.md)
(machinery, both halves' calibration-by-measurement records, gates G-1…G-8 / H-1…H-2, the run
record, the noise floor, and the mis-read-instrument list)
**Parents**: [`../conditional_revision/`](../conditional_revision/README.md) (the revision identity
and the cached substrate this reuses) ·
[`../practice/setlist/`](../practice/setlist/FILES.md) (the OU demand machinery) ·
[`../practice/typed_gaps/`](../practice/typed_gaps/README.md) (the currency taxonomy this tests on
a second substrate) · idea docs:
[revision_not_surprisal](../../../ideas/revision_not_surprisal.md) §4 (the identity extended here),
[practice_manufactures_its_own_credit](../../../ideas/practice_manufactures_its_own_credit.md) §18–§19
**Status**: written up 2026-08-17 (runs 2026-08-16→17; interpretation discussed with Jasper
throughout). The Half-2 noise floor is a paired stream-position calibration, not a
replication. Implemented by one Opus subagent session from the SPEC.

## One-liner

A token's information decomposes **exactly** into structure-news + question-news + aleatoric residue
(residual 3e-4–1e-3 nats at every truncation, drift-off bit-identical to the incumbent oracle), and
the question channel — what the world is currently asking — is as large as the root-level structure
channel. A plain NTP learner builds an in-context question model **only above an evidence
threshold** (56 draws/window: 92% of the exact ceiling; 7 or 1: at chance despite headroom), and the
one place an explicit question-tracker pays next-token loss (~10× a measured noise floor) is
**warm-starting the fast component at context entry** — not the slow components it was first aimed
at. The demand-keyed self-knowledge readout is a **vacuous null**: this unmetered generalist's
competence is flat across the question axis (0.4975–0.5048, at the noise arms' own spread), so
there was nothing for a question-keyed ledger to predict — the premise, not the organ, failed to
appear.

## The question

[typed_gaps](../practice/typed_gaps/README.md) measured one-organ-per-currency at the level of a
practice loop's components. This node asks whether the same typing exists *inside a single
network*: does a token's question-channel (the mixture the world currently draws from — the object
`setlist` drifts, and the third term missing from `revision_not_surprisal` §4's identity) get
tracked as its own quantity by an NTP learner, and what does an explicit question-tracker buy?
Substrate: the cached `conditional_revision` LM (v16/s2/L6/m4, NTP — `tall`'s m-inadmissibility is
a sculpting result and does not apply) under calibrated OU drift on the root prior + rule mixtures
(per-event demand-KL 0.014–0.157 nats; σ ≥ 1.4 measured inadmissible — corpus difficulty degrades;
the admissibility-window pattern's third appearance).

## Half 1 — the instrument

1. **The three-way identity closes exactly.** Total 1.1521 = demand 0.0402 + structure + aleatoric
   at every truncation (d6: 0.0343 structure / 1.0774 aleatoric; d1: 0.5580 / 0.5532), residual
   2.6e-4–1.2e-3. Drift-off reproduces `conditional_revision/oracle.py` at **max|Δ| = 0.00e+00**
   with the demand channel identically zero; weighted BP vs brute force ≤1.8e-15; the new O(T·L)
   prefix filter vs BP ≤1.8e-14.
2. **The question channel is as large as the deepest structure channel** (0.0402 vs 0.0343
   nats/token; 3.5% of surprisal; per component lo 0.0234 > hi 0.0124 > root 0.0045).
3. **The aleatoric label is not drift-invariant**: knowing θ makes ~20% of "irreducible" synonym
   entropy predictable (H_irr d1 0.684 → 0.553). Part of what NTP treats as noise is question-news.
4. **Finite context manufactures the gap.** `Dm_window` is flat in κ while `Dm_history` scales with
   it: for a 64-token-window learner the question channel is kept alive by the **window**, not the
   drift — drift matters only for unbounded memory. (Exact conditioning ladder: knows-θ 1.1110 <
   history 1.1379 < window 1.1521 < no-θ 1.3600.)
5. **Laundering is gated by evidence density.** Against a properly-computed shuffled-label null
   (~0.25, not 1/7): the plain model's activations carry a θ-estimate for `lo` at **0.868 = 92% of
   the exact 0.943 ceiling** (unigram-count control 0.689; incoherent/uniform controls 0.51/0.53),
   with entry lag saturating ~12 positions — and for `hi` (7 draws) and `root` (1 draw) sit at the
   null *despite ceilings of 0.584 and 0.343*. A threshold between 7 and 56 draws per window, not a
   supply limit alone.
6. **The currency dissociation reproduces in activation space.** In the one exactly single-channel
   matched contrast (`demand_vs_syn`, structure guard 0.500 by construction), M_theta separates
   (0.553–0.560) while M_struct and r_temporal sit at ~0.50; on `demand_vs_struct` the two readouts
   move **opposite ways** with depth (M_theta 0.580→0.658 toward d1; M_struct 0.555→0.331). The
   structure-readout and question-readout of one network carry different currencies — `typed_gaps`'
   organ-level dissociation, one level in. (Also a retrodiction: `conditional_revision` measured
   the temporal residual as "mostly surprisal" — in a static world, where the question channel is
   identically zero and it had nothing distinctive to carry; under drift it is demand-specific,
   partial R² 0.022 vs 0.001–0.005 on structure.)

## Half 2 — the twin learner, 11 arms against a measured floor

**Noise floor** (duplicate arms differing only in stream position): |Δ val nll| = 0.00068 and
0.00094, two-sided. Both information-destroyed capacity controls sit **inside** it (+0.0002,
+0.0004); every informative tap sits 2–10× outside.

| arm | payload | val nll | vs plain | × floor | fraction of ceiling |
|---|---|---|---|---|---|
| `tap_oracle` | root+hi block-entry prior | 1.3998 | −0.0015 | 1.9 | 24% of 0.0063 |
| `tap_oracle_full` | root+hi+lo | 1.3966 | −0.0047 | 5.9 | 33% of 0.0143 |
| `tap_learned` | endogenous estimator, no θ label | 1.3942 | −0.0070 | 8.8 | 49% of 0.0143 |
| `tap_oracle_lo` | **lo only** | **1.3933** | **−0.0079** | **9.9** | ~93–105% of ~0.0075–0.0085 |

(The per-component prizes sum to 0.01379 vs joint 0.01426, so the lo ceiling is **soft** — recorded
as an estimate, not a bound.)

7. **The payload that pays is the fast component's entry lag.** The SPEC's original steer scoped
   the tap to the slow components ("the window launders them") — a recorded design error: 92%
   *within-window terminal* recovery is not *cross-context* redundancy, because the ~12-position
   entry lag repeats at every block. `lo` carries more than half the cross-context nats, the
   lo-only oracle is the best arm in the family, and the internalization ladder reads in the
   expected order (exact ≥ learned) once payloads are equal — the transient "endogenous beats
   oracle" inversion was payload restriction.
8. **The endogenous estimator's gain is a lo entry prior, attributed by profile.** `tap_learned`'s
   per-level profile correlates **+0.975** with `tap_oracle_lo`'s (max difference 0.55 pp, inside
   the per-level floor) vs +0.786 with the slow-component oracle's; the resolvable core is
   lvl1–lvl2 (5.0–5.5× floor), not lvl0. Suggested mechanism, unproven: bottom-up evidence
   sharpening — the leaf-level mixture makes each observed leaf pair sharper evidence about its
   parent, compounding upward.
9. **Open surprise**: `tap_oracle_lo` beats `tap_oracle_full`, *which contains it*, by 4× the floor
   — a 7-dim payload outperforming its 21-dim superset. Payload narrowness is doing unanticipated
   work (dilution of a zero-init projection vs exploitability at this budget — unresolved).
10. **E2, the SPEC's priority readout: a vacuous null, attributed to the world.** The
    demand-conditioned ledger is *identical* to family-only when frozen (complete backoff) and
    moves ≤0.0007 maintained, in every arm; the full across-arm range of every self-knowledge
    metric is at or below the paired noise delta. The `demand_dependence` diagnostic says why:
    competence spans 0.4975–0.5048 across θ buckets against 0.162→0.721 across position families —
    the drift moves nothing this learner is differentially good at (accuracy 0.5014 region A,
    0.5046 region B). **This is not the "structurally-external" outcome**: the premise (competence
    heterogeneity along the question axis) was never instantiated, because an unmetered generalist
    has no concentration for the question to be about. The surviving object is a precondition —
    calibration-as-question-tracking is a claim about **concentrated hosts** — and the
    four-months-prior framing of the epistemic-self-knowledge question is retired as stale
    (Jasper, 2026-08-17); the conditional is to be tested on a metered specialist only if it earns
    its place. Note the activation probe did **not** null here (0.8214 frozen-on-A transfers
    intact — this θ-shift moves no competence, unlike Exp A's domain shift), while Exp A's
    *ordering* replicates exactly: output entropy ≥ activation probe > every ledger.
11. **E3, two claims kept separate.** Slow-component θ-conditioning: 1.9× floor, with the ceiling
    arithmetic attached (0.017-nat Bayes prize vs 0.24 nats of optimization slack). The aleatoric
    weighting is a real, priced **reallocation lever**: 4.0–5.7% lower nll at lvl0–3 bought at
    +1.1–1.4% on lvl6, net worse in total because lvl6 dominates token count — worth it only if
    structural depth is what you value. The θ-awareness of the label bought nothing (within 0.003
    at every level but lvl6, labels differing at means 0.568 vs 0.694).
12. **E4: the delta_norm anomaly is a position confound** (0.088 → 0.0059 within-position, inside
    the shuffled band, and unstable across draws besides). The repo's directional-not-scalar
    pattern stands.

## Interpretation (discussed with Jasper, 2026-08-17)

- **The typed-currency picture reproduces inside one network**: distinct readouts of the same
  activations carry structure-news and question-news, moving opposite ways with depth.
- **The window is where the question channel lives.** A context-windowed learner has a live
  question channel even in a still world; it self-serves the fast half at 92% of ceiling and
  launders everything below the evidence threshold. What a monolithic learner lacks is small and
  specific — the slow components and the **entry prior** — and the portable LLM-shaped claim is
  that carrying question-state across context boundaries pays mostly as *warm-starting fast
  statistics after the boundary*, not as remembering slow ones.
- **The organ logic's degenerate-case clause, again**: an organ idles when its currency carries no
  news for its host. A generalist needs no question-keyed self-knowledge; the vacuous E2 cell is
  crystallize's "nothing to certify" in the epistemic domain.
- **Methodology export, third instance**: aim the news, check the loudness, and now *check the
  premise* — the world must instantiate the quantity (competence spread) the readout is about.

## Caveats

The noise floor is a paired calibration, not a replication. One substrate,
one drift magnitude, one window length (the evidence-density threshold is bracketed at 7–56
draws, not located). The lo ceiling is soft (per-component split approximate). The
bottom-up-sharpening mechanism (finding 8) and the narrowness effect (finding 9) are unattributed.
Five mis-read instruments are recorded in [FILES.md](FILES.md) so they are not rediscovered.

## Reproduce

```bash
cd experiments/            # MODAL_PROFILE=chromatic

modal run -m rhm.question_model.question_model::selfcheck        # gates G-1…G-8
python3 -m rhm.question_model.analyze_qm --tag dec0 --fetch --figures   # Half 1
python3 -m rhm.question_model.analyze_qm --tag h2 --fetch --figures     # Half 2 (report_h2)
```

Full launch commands with exact flags, the calibration records, and the volume layout
(`/data/v16_s2_L6_m4_distinct/question_model/` on `rhm-scaling-data`): [FILES.md](FILES.md).
