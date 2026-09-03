# SPEC — committee_head: a committee of forward models as the reader, a trained head as the judge

**The question in one sentence**: can a head trained on the *second-moment signature* of a committee of
small forward models learn the three-way split — mastered / novel-and-aleatoric / novel-and-learnable —
together with relevance, and allocate a metered collection budget at least as well as the hand-composed
`lprog × visits` signal, on the substrate where every ingredient of that signal has already been
dissected separately?

**Status**: spec, 2026-09-02. Nothing run. **Parent**: [`../README.md`](../README.md) (mjc).
**Attribution**: the hypothesis is Jasper's (2026-09-02) — that reading the three-way split off the
forward model is a *value-shaped* thing that should guide the learner on frontier domains; that a
committee is welcome where a single FM is not, and can be read as cerebellar design; that the FM is an
*input to a value head* rather than a thing in itself; and that no oracle belongs in the mechanism,
RHM entering only as a substrate that provides one. The evidence assembly and the design came out of
the exchange that followed.

## What the record says the reader is, and what it never built

The split as stated is the repo's regime table — dark room / noisy TV / frontier
([`two_timescale_value_loop`](../../../ideas/two_timescale_value_loop.md) §regime table). Read the
following before designing anything; each is a fact about *who can read which leg*.

- **No single FM reads the middle leg.** The depth residual has zero aleatoric content by construction;
  the temporal residual has it (0.665 of variance, [`aleatoric_fraction/`](../../rhm/conditional_revision/aleatoric_fraction/README.md))
  but the FM does not separate it: its magnitude is the reader's output entropy (R² 0.90) and revision
  R² 0.0001, its direction is the raw state update's ([`a2a/conditional_revision`](../../a2a_forward/conditional_revision/README.md) Gate D;
  [`epistemics/`](../../rhm/confabulation/temporal/epistemics/README.md)). The FM's first moment is the
  unbiased conditional mean, Wiener gain ≈ 1; all the structure is in the error's second moment, 15×
  across directions ([`REPRESENTATIONAL_DIVERGENCE`](../../a2a_forward/REPRESENTATIONAL_DIVERGENCE_README.md)).
- **Every reader that worked pooled over repeats.** (i) Committee disagreement: rejects high-dim
  well-sampled noise by construction ([curiosity Phase 2b](../../a2a_forward/reaching/CURIOSITY_DRIVE_README.md), 3.7×→8.5×)
  but *chases* scarce low-dim noise on control ([`curiosity_control/`](../curiosity_control/README.md) Finding 3,
  patch-frac 0.00 on 3/3 noise seeds) and is blind to staleness and to a confident-prior shift
  ([`ballistic/directed/`](../ballistic/directed/README.md) S1; [`directed_readapt/`](../directed_readapt/README.md)).
  Which drive wins is set by the environment, not absolute. (ii) A context-conditional benchmark net
  `b(s)` on the FM's own error, differenced and agency-gated: reads the noisy TV as statistically zero,
  habituates to it, withdraws plasticity from it ([`benchmark_vs_cost/`](../curiosity_control/benchmark_vs_cost/README.md),
  [`plasticity_gain/`](../plasticity_gain/README.md), [`agency_gate/`](../agency_gate/README.md)) — but
  needs recurrence per context, has an interior optimum on its timescale, and is a *gain*, not an
  allocation score. (iii) A held-out trial of learning — the metered survey in
  [`on_policy/directed_on_policy/`](../on_policy/directed_on_policy/README.md) — inverts under data
  starvation unless floor-corrected ([`metered_repair/`](../on_policy/metered_repair/README.md)).
- **Relevance is not in the epistemic signature.** It came from rolling the plan through the FM
  (`fm_visits`, the p-tap) or from grounding with the exploit term; omitted, the drive tracks the
  frontier and pays nothing ([`curiosity_control/`](../curiosity_control/README.md) Finding 2). E3
  reproduced both halves under metered looking: `value` beats `lprog-only` 3/3 and `error-only` burns
  47% on the noisy TV.
- **The teacher must be sighted.** Control is a near-blind grader; the value-relevant FM error sees an
  interior explore/exploit optimum where control is flat 0.1% ([`drift_value_loop/`](../drift_value_loop/README.md) Cut 3).
  Fast per-transition and slow reward channels are non-redundant and need matched windows;
  sharing is destructive ([`two_clocks/`](../two_clocks/README.md)).
- **The untested atom.** [`ideas/activation_to_activation_forward.md`](../../../ideas/activation_to_activation_forward.md)
  §"Learned valence tagging" proposed *a small auxiliary classifier predicting whether a given error
  will prove reducible over subsequent training* in May, and it has never been built. Every
  composition in the record was hand-specified (`lprog × visits`, `grounded@b` swept, `b(s)` fixed in
  form); the one learned outer loop, REINFORCE over `b`, converged only directionally on a shallow bowl.
  [`heterogeneous_graders`](../../../ideas/heterogeneous_graders.md) §9 names the disagreement between a
  dense grader and an evaluative one as the fix for staleness-blindness, and it is unbuilt.

## The design

**Substrate: E3's, forked verbatim.** [`directed_on_policy.py`](../on_policy/directed_on_policy/directed_on_policy.py):
the arm, one on-reach reducible target A, three off-reach reducible distractors, two off-reach noise
regions, a reflecting OU walk on the curl gains so nothing stays repaired, on-policy metered collection
*and* survey through `Body`, sighted grader = region-A FM error, ballistic control as the cash-out.
Gate the fork with `verify_backcompat` and with bit-identity of E3's `value` arm at `K=1, rpf_beta=0`.
Keep E3's geometry gotchas as written. An on-reach noise region was unplaceable there (you only go
where you reach) — note it, do not fight it; that confound is what Phase C is for.

**Reader: a random-prior committee.** Replace the single `net` with the K-member RPF member idiom from
[`curiosity_control.py`](../curiosity_control/curiosity_control.py) (`rpf_beta`), re-fit on the same
buffer; the committee mean plans, so `fm_visits` and control are unchanged in kind. This is the
microzone reading in [`cerebellum_and_cognitive_architecture`](../../../beliefs/trees/cerebellum_and_cognitive_architecture.md):
many bands, one teacher.

**Signature: per region, per round, all reward-free, all from what the loop already computes.**
Committee-mean error `e` on the in-region survey subset; disagreement `d`; benchmarked error `b(s) − e`
with `b(s)` the `make_bench`/`bench_step` net from [`plasticity_gain.py`](../plasticity_gain/plasticity_gain.py)
trained online on the committee's own error stream; the agency gap `g` from
[`agency_gate.py`](../agency_gate/agency_gate.py) (near-inert here since every survey transition is
self-produced — carry it so the object is the full δ_perf and a playback control stays available);
plan occupancy `v` (`fm_visits`); and the last few rounds of each, so the head can form a derivative
if it needs one without being handed the derivative's known blindness at re-opening. The survey's
counterfactual `lprog` may be given as an input in one arm, so the ablation can say whether the head
adds anything beyond what the survey measures.

**Head: small, reads the signature, outputs the allocation.** A softmax over regions (and optionally a
per-sample plasticity gain, the [`bridge_assembly/`](../bridge_assembly/README.md) consumption). Two
training shapes, in this order:

1. **Supervised valence tag.** Targets are outcomes the loop logs one round later: the realized
   held-out error drop in region j after collecting there (the survey's `eb − ea` on the next round —
   *future* reducibility), and realized visitation. The head predicts "will collecting here pay, and
   will I be here"; allocation follows from the prediction. Dense, cheap, no RL. This is the atom.
2. **Reward-driven.** The head's parameters tuned by an outer loop whose only signal is the sighted
   teacher (the `online_front` idiom in [`online_value_loop.py`](../drift_value_loop/online_value_loop.py)).
   The record says REINFORCE is noisy on a shallow bowl; use a variance-reduced outer optimizer if this
   is attempted at all, and only after (1) has been read.

Two lessons to carry regardless of shape: a multiplicative signal needs an explicit no-op (S2's
`value-floor`), and if the head ever consumes both the fast committee channel and a slow reward, keep
them as separate channels with their own windows.

**The heterogeneous-disagreement arm** (§9 of heterogeneous_graders): allocation driven by where the
committee (dense grader) and the head trained on the sighted teacher (evaluative grader) *disagree*,
with plain committee disagreement as the honest homogeneous baseline. The OU drift on region gains is
local, which §9 requires. The discriminator it names: noise-disagreement does not *close* when you
collect there. The functional form is the implementer's call.

## Phase A — gates before any treatment

1. **Fork fidelity** as above.
2. **Committee health.** Members sub-saturation and genuinely disagreeing off-data: pairwise residual
   cosine (the `_ensemble_cos` guard in [`confabulation.py`](../../a2a_forward/confabulation/confabulation.py))
   and in-data vs off-data prediction variance. Record, with the head untrained, how the committee
   behaves on the two noise regions here — Finding 3 predicts members fit different realizations and
   disagree. Whatever it does is a substrate fact the head must learn around; put it in the README.
3. **Benchmark estimability.** Per-region recurrence per round against `bench_lr`; sweep around
   plasticity_gain's window; `b − e` should read ≈ 0 on the noise regions at steady state before the
   head sees it, as in benchmark_vs_cost's noisy-TV cell.
4. **Signature separability, read-only.** Plot the per-region signature over rounds by region class.
   If nothing in it separates A / off-reach reducible / noise, say so before building the head.

## Phase B — the ladder

`uniform` · `error-only` · `disagree-only` · `value = lprog × visits` (E3's incumbent, the hand-composed
head) · `oracle` (privileged) · `head_sup` · `head_rl` (optional) · `head_hetero` · the head's ablations
(drop `d`, drop `b − e`, drop `v`, drop histories, add survey `lprog`) — the ablation table is the
product of this cut, since it says which reader carries the split. Add
[`metered_repair/`](../on_policy/metered_repair/README.md) §7's scheduled-burst control (oracle's
target, `value`'s temporal profile) so the ladder is not measuring timing. Single seed first, then
seeds per [`experiments/CLAUDE.md`](../../CLAUDE.md).

**Readouts.** Region-A FM error, ballistic control, budget shares to A / off-reach reducible / noise,
leak, monitor:collect ratio (must stay O(1)). Split quality: predicted reducibility vs realized survey
`lprog`, predicted relevance vs realized visits, per region per round. Event-triggered response at
re-drift events, head vs committee disagreement — the §9 question.

## Phase C — calibration on RHM, the oracle scoring but never feeding

The same head shape on the frozen [`conditional_revision`](../../rhm/conditional_revision/README.md)
reader, with a committee of K temporal activation FMs in the `endogenous_teacher` idiom (one-token
conditioning gap, so the residual actually has an aleatoric leg). Per-position signature as above
over recurring contexts; head trained on the next window's realized error drop by position class;
scored against [`oracle.py`](../../rhm/conditional_revision/oracle.py)'s `irreducible_entropy` and
`revision_and_entropy` — the exact reducible/irreducible split, with ~49% of positions purely
irreducible while still carrying surprisal. This is what lets "the head learned the split" be a
measurement rather than an inference from allocation, and it is the only place RHM enters. The
Chronicle corpus in [`a2a/conditional_revision`](../../a2a_forward/conditional_revision/README.md)
carries an exact oracle too and is the optional language cell.

## Things to keep honest

E3's `value` is a strong incumbent and already matches the privileged oracle. On-policy has no
visited-but-irreducible cell, so reducibility and relevance are partly confounded on this arm — E3's
own finding — which is why the ablations and Phase C carry more than the headline. Control is
near-saturated; the ladder resolves on the sighted grader. One task family. Per repo norms, no
outcome is interpreted here.

## Housekeeping

Node at `experiments/mjc/committee_head/`; README + `FILES.md` at writeup, row in
[`../FILES.md`](../FILES.md). Results under `/data/committee_head/<tag>/` on `mujoco-control-data`.
On-policy collection is the convention for new cuts. Launch each seed as its own client. Invoke
`/run-experiment-on-modal`[^private] before running;
subagents follow `/subagent-instructions`[^private]
and do not touch git state.

[^private]: Not mirrored: this link points to a document in the private lab repo (the roadmap, the queue, an unrun spec, reading notes, or a conversation). See the top-level README for what is held back and why.
