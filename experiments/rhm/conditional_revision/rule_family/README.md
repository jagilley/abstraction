# Rule-family RHM: in-context rule inference, and what a conditioning gap is worth to the full system

**Up**: [../README.md](../README.md) · **Files**: [FILES.md](FILES.md) · **Working notes / handoff**: [NOTES.md](NOTES.md)
**Idea**: [`ideas/revision_not_surprisal.md`](../../../../ideas/revision_not_surprisal.md) · follow-on design: [`ideas/temporal_confabulation_test.md`](../../../../ideas/temporal_confabulation_test.md)
**Siblings this converged with**: [`../synonym_retention/`](../synonym_retention/README.md) (whose
impossibility result this substrate lifts), [`../aleatoric_fraction/`](../aleatoric_fraction/README.md)
(the fixed-rules headroom measurement), [`a2a_forward/conditional_revision/`](../../../a2a_forward/conditional_revision/README.md)
(the language sibling whose entropy result reshaped Gate 2)
**Ancestry**: [`ratchet/RHM_META_LEARNING`](../../ratchet/RHM_META_LEARNING_README.md) (weight-space meta
across rule sets: collapse) · [`mjc/meta_adapt`](../../../mjc/meta_adapt/README.md) (the floor/conflict
lesson and the context-latent positive this design transposes to in-context inference)
**Date**: 2026-08-09/10 · **Status**: Gate −1, Gate 0 (+ 36k retrain), diagnostics, retention
(parts 1–2), and Gate 1 all run. Gate 2 not run — its pre-registered precondition was not met.
Single seed throughout.

## Goal

Everything else in the parent cut runs on **one fixed rule set**, where reducibility is static: which
positions are synonym-slots and which are disambiguators is fixed by the DGP, so an NTP-optimal model
can bake the split into its weights and never compute it. Under fixed rules RHM is also Markov in the
ancestor chain — resolved past constituents are conditionally independent of the future — so the
parent's probe findings ("close to a next-token-sufficient statistic", "does not summarise resolved
structure forward") describe a model that is *optimal*, not deficient, and the parent's headline
constraint ("the binding constraint is belief depth") may be regime-induced.

Here a context window is drawn from one of **R rule sets** (same `v/s/L/m`, different composition
tables), so the model must infer the active rules in-context. Reducibility becomes **state-dependent
within the context** — the same position is revision-heavy early (rules unknown; the token is
evidence) and synonym-noise late (rules identified) — and in-context learning on this DGP *is*
amortized Bayesian inference over rules. With finite R the exact oracle survives as a **mixture over R
junction trees**, and total surprisal decomposes exactly into parse-irreducible + parse-revision
(fast) + **rule-revision** (slow, decaying as the rule posterior concentrates). The incumbent
substrate is this design's **zero-conflict floor** (`differ_levels=[]`, bit-identical code path).

## Gate −1 — the budget constraint, and design selection without GPU time

The design space has a hard cap the original framing did not state: summed over a window,
`E[rule revision] = I(r; x_window) ≤ ln R`. **The entire slow component is budgeted at ln R nats
however the family is built**, so per-token strength and transient duration trade off along a fixed
budget. This is `RHM_META_LEARNING`'s "thin compositional signal" restated as an information-theoretic
identity. Gate −1 measures the trade-off exactly on CPU ([`gate_minus1.py`](gate_minus1.py), oracle
verified to 2.9e-15 against brute-force enumeration over (r, latents)), and two design constraints
imported from the MJC arc came back *measured* rather than assumed: differing the leaf-emitting table
identifies ~3× faster (justifying sharing it — the n-gram shortcut is real), and differing at d5 is
essentially invisible to this base model (0.0007 nats/token).

Two designs were selected: `d2_R128_nF4` (4.92 nats, ~77% spent in the first sequence — a strong
two-point contrast) and `d2_R64_nF2` (4.24 nats spread across all eight sequences — a graded
identification trajectory). One constructional trap is worth knowing: rule sets differing only in the
storage *order* of a feature's m tuples are distributionally identical, and unidentifiable duplicates
cap the posterior in a way that **mimics gradual identification**. Families are canonicalised, with a
guard refusing R above the partition count.

## Gate 0 — ICL is present, learnable-design-dependent, and was initially undertrained 3×

Matched depth-swap readout (same probe content, only preceding context varies), family arm net of its
compute/data/config-matched floor arm. The first pass ran 12k steps (the incumbent's budget); the
emergence trace showed both arms still descending, and a 36k retrain roughly **tripled** the effect:

| | `d2_R64_nF2` @12k | **`d2_R64_nF2` @36k (aligned)** | `d2_R128_nF4` @12k |
|---|---|---|---|
| net in-context decline (depth 7) | +0.0043 ± 0.0004 | **+0.0122** | +0.0013 ± 0.0004 |
| fraction of the exact oracle ceiling | 0.226 | **0.645** (still climbing → lower bound) | 0.022 |
| corr(net decline, oracle availability) | +0.992 | **+0.997** | +0.754 |
| rule decodability vs chance | 2.5× | **7.6×** (~19% of the 0.614 Bayes ceiling) | at chance |

`d2_R64_nF2` realizes in-context learning with the oracle's *shape*; `d2_R128_nF4` met its
pre-registered stopping criterion (ICL fraction < 0.10 with rule decodability at chance) and its
emergence trace is flat from step 2k — for that design the collapse is structural, not budgetary.
This is the first positive ICL-pressure reading on RHM; the 2026-06 weight-space meta-learning line
found ≈0.

**Held-out transfer replaced this cut's first mechanistic reading, and the correction is kept as a
correction** (NOTES §6.1). The R64 family model transfers to unseen rule sets at ~108% (held-out loss
1.4657 vs 1.4661 in-distribution; the floor model reads 1.877 on the same sequences) — it did not
memorise 64 tables, it learned transferable inference over partitions. So the initially-reported
"storage burden" reading is wrong; the variable that separates the designs is **inference
dimensionality** (pinning 8 tuples into 2 groups vs 16 into 4), with the capacity tax decomposing as
mostly unresolved-mixture penalty (failure-to-infer) rather than cost-to-store. Caveat: 6 held-out
sets, shared tuple pool — within-pool generalisation.

**Two protocol facts with teeth** (both from running matched arms rather than trusting one):
the phase-hidden training protocol, recommended at one point in this cut's own notes, is *worse* for
realized ICL (0.376 vs 0.645; arbitrary phase is a harder task, and its floor arm carries a +0.046
positional artifact ~80× the aligned one's) — Gates run on the **aligned** checkpoints, phase-hidden
retained as the discard-schedule control; and an unmatched single-arm ICL reading before ~6k steps is
an artifact (the floor arm alone shows +0.0048 at 2k that decays to zero).

## The discard schedule is a dependent variable, keyed to predictive relevance

Two findings that factor one mechanism, recorded as a pair (NOTES §7):

- **Relevance at fixed distance** ([`probe_positions.py`](probe_positions.py)): the floor arm reads
  the incumbent reference lines at positions where unresolved ancestors still depend on the current
  parse (d1 0.91–0.97, d3 0.72–0.89) and collapses where nothing does (d3 **0.880 vs 0.217** at the
  *same* distance-since-closure — the split is purely whether an unresolved ancestor pends). Gate 0's
  initially-alarming belief-depth numbers were read at the single worst position; the substrate was
  never damaged. Aligned-window training hands the model a phase cue, and it discards on schedule.
  [`../synonym_retention/`](../synonym_retention/README.md) supplies the other factor — distance at
  fixed (zero) relevance — so the schedule now has both axes measured.
- **The family regime re-prices retention, as replacement rather than insertion**
  ([`retention.py`](retention.py)): across sequence boundaries the family arm carries a compressed
  rule-sufficient statistic the floor arm does not (rule-probe excess +0.011 → +0.030, growing with
  accumulated evidence; floor pinned at chance), while carrying *less* decodable parse detail than the
  floor arm (−0.03 to −0.04). Deeper beliefs are bought, not added.

## `rule_retention` — the positive control fixed rules provably cannot have

`../synonym_retention/`'s impossibility section shows fixed-rules RHM has **no NTP-required-at-distance
content**, so its retention curve has no upper reference arm. This substrate lifts that: perturbing a
d2 constituent's rule choice (the differing level; leaf tables are shared and carry no rule
information) leaves prefix-identity intact while making the realisation evidence about `r`.

**Part 1 (DGP-side, model-free)**: the floor arm's exact next-token TV reaches **exactly zero in 100%
of cells from w=32** — the impossibility confirmed numerically — while the family arm stays non-zero
across three sequence boundaries (TV ~0.0006–0.0008 at w=157), where the rule posterior is the only
surviving channel. Only ~8% of cells carry it (perturbations landing on a differing feature), exactly
as the construction predicts, and the signal vanishes at read positions governed by the shared leaf
table — the internal consistency check.

**Part 2 (model-side)**: the first run (`p2c`) is **void** — its probe label was the rule's *storage
index*, which canonicalised pooling makes an arbitrary per-rule-set relabelling, so the family arm's
exact Bayes ceiling for the question was chance (measured: 0.25) while the floor arm's was 1.0. This
is the storage-order gotcha re-entering through the probe target, and the sibling's
"pin both ends of the probe" discipline is precisely the guard that catches it. The corrected run
(`p2d`, label = emitted tuple in the shared pool, exact mixture Bayes ceiling per distance; ceilings
now 0.980–0.987 vs 1.000) shows the two arms identical near the constituent (w ≤ 4, both ≈ +0.99),
then family−floor growing: **+0.11 (w=8), +0.21 (w=32), and +0.18/+0.27/+0.27 across the three
boundaries** (family holds +0.40–0.41 where the floor decays to +0.13–0.15). The excess is the
realisation bit that carries rule evidence; feature identity is common-mode. Note the floor arm is
*not* a reproduction of the sibling's λ=0 curve — the tuple label is feature-and-realisation jointly
and inherits the feature's slow decay; only the family−floor difference is load-bearing.

## Gate 1 — the belief-coordinate revision readout reads chance, for a quantified reason

Pre-registered design and full implementation record in NOTES §8. `M_rule` (KL between successive
probe-decoded rule posteriors) against the oracle's exact `rule_rev`, on the aligned 36k checkpoints,
under matching up to position × exact mixture surprisal, restricted to differing-ancestor cells:

- **`M_rule` never exceeds its guards at any depth** (.477–.519 vs `M_shuffled` .490–.507 and the
  self-matched surprisal column .513–.539; window-bootstrap SD .007–.015). Before-state readouts
  (`M_pointmass`, `negH_t`) are null too.
- **The pre-registered pooled column would have shipped a false positive**: unrestricted, the readout
  reads .552 → .510, above chance and decaying with depth — matching both registered predictions —
  and the entire effect is the differing-ancestor base rate (.422 → .033), decodable from the parse
  with zero rule inference. The confound restriction was added *before* any model numbers were seen
  (NOTES §8.1), which is what made this detectable as an artifact rather than reportable as a result.
- **The null is resolution-limited, not evidence the phenomenon is absent, and the two readings are
  distinguishable here.** The rule probe reads 11–17% of its exact Bayes ceiling — the parent's own
  Gate A/B established that this readout fires at ~82–90% of ceiling and reads chance at ~18%, so
  Gate 1 sits on the established null end of the program's decodability curve (third point on one
  line). Mean `M_rule` is 0.641 nats against mean oracle `rule_rev` of 0.008 nats — readout jitter
  ~80× the signal. Meanwhile the *function* is strong on the same checkpoints (ICL fraction 0.645,
  transfer ~108%), so the state demonstrably carries and uses what the probe cannot cleanly decode.
- **It is also not the language sibling's pattern**: there, the token created a distinction the belief
  readout couldn't express (`h_after` 0.99 vs `h_before` 0.55). Here `h_before_dir ≥ h_after_dir` at
  every depth and the floor arm reproduces most of the directional separation — prefix-determined
  parse role, not a hidden revision signal.
- One pre-registered discriminator did not survive contact with the oracle: "flat-in-depth ⇒ identity,
  not revision" assumes the truth's per-event magnitude decays with depth, and it does not — the
  oracle's depth decay is base rate (fraction of positive cells .451 → .025), not magnitude (.050
  flat). Recorded in NOTES §8.1 with the graded replacement (partial `R²` on positive cells).

**Gate 2 (the temporal FM readout) was not run**: its pre-registered precondition — a live
belief-space object — was not met, and it is independently pre-empted by the state decodes
(`h_before_dir ≥ h_after_dir` makes the FM-contribution comparison degenerate before it starts). The
design, including the entropy discipline the language sibling forces (in this regime the ICL decline
*is* an output-entropy decline, so a magnitude-based "FM residual decays over context" readout is an
artifact by construction), is preserved in NOTES §9 for a future substrate or a stronger probe.

## What this establishes, and what it does not

**Establishes** (single seed, one regime, one model scale):

- A rule-family RHM supports genuine in-context rule inference — the first positive ICL reading on
  this substrate — at 0.645 of an exact oracle ceiling (lower bound), with the oracle's shape, and
  transferring to unseen rule sets within the tuple pool.
- Realized ICL is not limited by information supply; the separating variable across designs is the
  dimensionality of the required inference, with incentive and supply held present. (Supply: `ln R`,
  exact. Realization: measured. The two had not previously been separable.)
- The model's retention schedule is keyed to predictive relevance and is re-priced by the regime:
  rule evidence is carried across boundaries exactly where it is the only surviving channel, paid for
  with parse detail. The NTP-required-at-distance content the fixed-rules substrate provably lacks
  exists here, categorically (floor TV exactly 0 vs family TV > 0 across three boundaries).
- The belief-coordinate revision readout on this substrate is chance at the current probe quality,
  consistent with the program's established decodability curve; the readout's failure and the
  phenomenon's presence are simultaneously measurable and were measured.

**Does not establish:**

- Anything about revision readable *from the composite's own channels* — every readout here is an
  external probe. That question is the follow-on design
  ([`ideas/temporal_confabulation_test.md`](../../../../ideas/temporal_confabulation_test.md)).
- That the belief-coordinate null is permanent: probe/ceiling roughly tracked training (~3× ICL, ~3×
  decodability from 12k → 36k, both still moving), so the checkpoint's position on the decodability
  curve is time-indexed. Untested whether longer training or a stronger (e.g. nonlinear) probe moves
  it into the readable zone.
- Whether the ICL fraction converges below 1 (the 36k curve is decelerating but not flat).
- Seed robustness, other regimes, or model scales. The `ln R` budget also caps what this substrate can
  ever say about a *dominant* slow channel — language's combinatorial latent space has no such
  poverty, so magnitude claims do not transfer in either direction.

## Reproduction

See NOTES §3 for the full command set (all tags, smoke variants, and the CPU-only subset). The core:

```bash
cd experiments
# CPU, no Modal: DGP self-test, oracle-vs-brute-force, design selection, retention part 1
python3 -m rhm.conditional_revision.rule_family.family
python3 -m rhm.conditional_revision.rule_family.oracle_mixture
python3 -m rhm.conditional_revision.rule_family.gate_minus1 --sweep --K 8 --n-windows 60
python3 -m rhm.conditional_revision.rule_family.rule_retention

# Modal (chromatic): Gate 0 + the 36k bundle, diagnostics, corrected retention, Gate 1
modal run --detach -m rhm.conditional_revision.rule_family.gate0_family::gate0 \
    --design d2_R64_nF2 --k-seqs 8 --phase aligned --base-steps 36000 \
    --ckpt-every 12000 --icl-trace-every 2000 --n-rule-windows 8192 --tag rt36k
modal run --detach -m rhm.conditional_revision.rule_family.rule_retention::part2b \
    --design d2_R64_nF2 --tag p2d
modal run --detach -m rhm.conditional_revision.rule_family.gate1::gate1 \
    --n-windows 3200 --n-probe-windows 6144 --oracle-shards 8 --tag g1
```

Artifacts land under `/data/v16_s2_L6_m4_distinct/rule_family/` on the `rhm-scaling-data` volume
(**chromatic** workspace); NOTES §4 is the full table. Build on the `_at36000` **aligned**
checkpoints.

## Gotchas worth not rediscovering

The complete list is NOTES §10; the four most expensive to relearn: storage order is not a DGP degree
of freedom (canonicalise, and never probe a storage index — it re-enters through labels after being
removed from the DGP); an unmatched single-arm ICL reading before ~6k steps is a positional artifact;
under random phase, later window positions carry more context on average (the +0.046 floor artifact);
and in this regime any magnitude-based "residual decays over context" readout is entropy by
construction — direction or nothing.
