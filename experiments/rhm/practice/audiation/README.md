# audiation — E1: the arity-2 self-model on practice. The forecastable part of the learner's own update is the δ already on the wire

**Up**: [`../README.md`](../README.md) (practice arc) · **Asked in**: `ROADMAP.md`[^private]
§4.2, first shape **E1** (this node is its record; the roadmap entry, the originating
conversation's E1 exchange, and one orchestrated conversation served as the spec — no separate
SPEC was written). The E1b extension was designed mid-conversation with Jasper after the E1
result. · **Files**: [`FILES.md`](FILES.md) (the instrument, the fitter's design calls, gates,
schema pointers, reproduce)
**Direct donors** (untouched): [`../conductor/`](../conductor/README.md) (substrate fork;
`cd_s0/anchor` the replay reference) ·
[`../../confabulation/temporal/epistemics/`](../../confabulation/temporal/epistemics/README.md)
(the `SlotFM`/`rev_pair` recipe, the matched-capacity decode discipline, the guards — and the
prior form of finding 2) ·
[`../../confabulation/temporal/temporal_confabulation.py`](../../confabulation/temporal/temporal_confabulation.py)
(the observer-rung structure) · `papers/forward_self_models_paper2.md`[^private]
(the definitions the interpretation is read against) ·
[`ideas/performance_error_is_the_bridge.md`](../../../../ideas/performance_error_is_the_bridge.md)
(δ, and the prior form of finding 3).
**Design exchange**: `conversations/955af184-f31a-4ada-9361-716c7662f284.md` lines 1670–1745 —
where E1's hypothesis was revised before any code existed: **E1 tests agency, not privacy.**
**Runs**: `au_smoke`/`au_s0` (E1: instrumented anchor, 0.56 GPU-h), `au_smoke_pd`/`au_s1`
(E1b: per-datum arm, 0.92 GPU-h both arms), 2026-08-29; all fits CPU-only, no GPU. **Single
seed on every treatment; ranks, signs, and the wiring-vs-finding separation are the claims.**
One orchestrated conversation; two delegated implementer agents (instrument, analysis), each
resumed for the E1b extension.

## The question

ROADMAP E1, verbatim: **is there a per-datum revision signal, is it gated by agency, and does
forming a residual against the forecast do any work the raw update does not?** The arity-2
self-model — FM₂(state, own-move-in-slot) vs FM₁(state, MASK), the `rev_pair` idiom with the
learner's *move* where NTP had the world's token — fitted over planner/value state logged per
decision on the practice stack. In teacher-forced NTP the arity gap g = ‖FM₂ − FM₁‖ is
identically zero (the world fills the slot); in practice the slot is the agent's own move and,
via explore-injection, not a deterministic function of state, so g > 0 and the agency gate has
something to gate for the first time. The observer ladder rides as the mandatory control, not
the point: both roadmap targets — the value head's error against the grade, and the change in
π's proposal at that state — are behavior-type facts in paper 2's taxonomy, and E1 is an
operational question about signal-forming, not a privacy question.

## Design in brief

**The instrument** (`audiation.py` forks `conductor.py` verbatim; additions `# [audiation]`-
marked, RNG-free, opt-in). Per decision in the practice beam: the frozen encoder state `z`, π's
logits in commit-invariant slot space, every scored candidate with provenance (proposed /
forced / **explore-injected**), the chosen move, lineage, and the terminal grade; per cycle,
snapshots of the two tiny trainable heads, so `f(heads_{c+1}, s) − f(heads_c, s)` is exactly
cycle c's revision at any logged state, recomputable offline in numpy with no GPU. The
instrumented anchor replays `cd_s0/anchor` at **0.000e+00 over all 116 cycles** (in-flight
hard assert every 5 cycles, plus offline cross-tag against `cd_s0` and `as_s0`), so `au_s0` is
a per-decision log *of a known run*: 881,280 tip-steps, 4.94M candidates, 117 head snapshots,
+7.8% wall.

**The fitter** (`fit.py`, three disjoint modes, all offline). The matched SlotFM pair on the
update (per the donor's discipline: shared init asserted, one loop, only the slot's content
differs), a source × target decode matrix at matched readout capacity with train-rows-only
stratum residualisation (cycle × step — the calendar confound is real here, finding 1), the
provenance-conditioned agency cut, practice-shaped observer twins with the half-data control,
and the guards (`ens_cos` ensembles, scalar-norm negatives, shuffled nulls, capacity
invariance, a strict-timing forward split).

**The two follow-ups**, designed with Jasper after the E1 result: `--ladder`, a
conditioning-completion ladder handing the pooled-run forecaster progressively more of the
batch's content; and **E1b**, a `perdatum` arm in which credit becomes per-datum — one graded
trajectory = one attributable update (`pd_n` = 128/cycle, per-cycle gradient budget matched to
the anchor as Σlr, replay separated into a pooled maintenance pass on history only so every
logged revision has exactly one cause, plant untouched), revisions logged in-run at the
datum's own states, the era probe, and a same-cycle reference set. Gates: with the mode OFF
the arm replays the anchor at 0.000e+00 (all new code inert when disabled); per-datum sums +
maintenance reproduce the boundary snapshots (4.8% median residual); a two-sided exact-zero
invariant on the π/value steps.

## Findings

1. **Under pooled credit the forecast target barely exists: the learner's update is mostly
   the calendar.** Held out by cycle, cos(forecast, Δ) = +0.003/−0.000/+0.008 at 8/33/130% of
   the heads' parameter count while in-sample climbs 0.13 → 0.44, and **|r|/|Δ| = 1.07–1.19 —
   the residual is *larger* than the raw update.** Mechanism: η²(Δ ~ cycle) = 0.438; a
   calendar-only forecaster reaches |r|/|Δ| = 0.731; even an FM shown the same cycle's update
   on other instances reaches only cos 0.141. The update is one batched step per cycle, and
   its cause — the batch — sits outside any prospective conditioning set: only 18.4% of choice
   states entered the value buffer at all, and π was supervised toward the agent's own
   top-ranked move on 3.6% of rows.

2. **r never beats Δ — the epistemics verdict, re-derived a third time, now with g > 0.**
   On the crux targets r ≤ Δ in every cell (0.052 vs 0.052 verr; 0.031 vs 0.029 dprop),
   capacity-invariant across the instrument sweep, shrinking further under the strict-timing
   split; scalar norms and the shuffled null at ≈ 0. The two large "self" cells in the matrix
   (`delta_u` → dprop 0.93–0.94) are wiring — the π target is mechanically inside (Δ, u) —
   and are labelled as such, not quoted.

3. **Grade-news lives at decision time, not in the update** — the third derivation of this
   fact (bridge-δ on mjc; state-not-residual on a2a language). The pre-update readout decodes
   the value-error target at 0.293 and the **deliberation state** — the unchosen candidates'
   own value scores, 12 dims — at 0.133, both flat across the capacity ladder, while every
   update-derived source sits ≤ 0.05 and decays. `delib` is decision-time and behavior-type
   (an observer with query access to the value head recovers it — paper 2 criterion 2), so
   this is a control-topology fact, not a privacy fact.

4. **Agency is visible in the geometry and does not rescue the residual.** |g|/|Δ| orders by
   provenance exactly as the efference-copy reading wants: 0.291 proposed / 0.352 explore /
   0.481 forced, and the dprop decode is ~3.7× stronger on explore-injected rows (0.098 vs
   0.026). But Δ ≥ r *inside* both provenance classes, and the efference copy improves the
   forecast only on proposed moves (|r₁|−|r| = +0.14) while worsening it on explore (−0.17)
   and forced (−0.96) — the slot functions as a typicality proxy, not as credit.

5. **The batch's content identifies the cycle and transfers nothing** (`--ladder`, au_s0).
   Handing the forecaster own-datum supervision flags changes nothing at three decimals;
   handing it the cycle's batch summaries and encodings moves in-sample cos 0.13 → 0.77–0.95
   while held-out cos goes *negative* and |r|/|Δ| inflates 1.07 → 1.6–2.1 — better-conditioned
   forecasters are more confidently wrong out of cycle, and the failure mode is confident
   misprediction, not shrinkage. Binding limit, stated in the file: rungs 3–5 are constant
   within a cycle, so their effective n is 84 training cycles.

6. **Under per-datum credit the revision becomes forecastable exactly insofar as it is the
   analytic δ — one fact, seen twice and counted once.** E1b's conditioning ladder on the
   value component: (s,a) 1.079 / cos +0.146 → **+grade 0.726 / +0.447** (pooled comparison:
   1.069 / +0.003). Q2 measures the same jump as wiring: BCE's logit gradient is σ(v₀) − y,
   and that scalar alone gives R²(Δv) = 0.588, sign agreement 94.8% (Adam breaks magnitude
   monotonicity; the reduction cross-references the two cells as the same fact). The π
   component's forecastable part is likewise the CE-toward-known-targets shape (lineage rung:
   cos 0.04 → 0.32 with |r|/|Δ| still > 1). On the joint 57-dim target |r|/|Δ| > 1 at every
   rung; `ens_cos` falls 0.956 → 0.635 (the per-datum residual is markedly less
   input-determined); and at whole-cycle granularity the per-datum arm is indistinguishable
   from the pooled one (1.059 / −0.017 vs 1.069 / +0.003).

7. **The vector form buys nothing over the scalars already free on the wire — the increment
   question, answered where every update has one cause.** With the per-state analytic δ, the
   grade, and log‖Δᵢ‖ as the covariate baseline, no vector source (Δᵢ, r, g, own Δπ) beats
   the baseline on any spillover target under cycle × order residualisation (e.g. 0.064
   baseline vs 0.057/0.042/0.034 on reference-spillover direction); the arity gap's raw-column
   gains vanish under residualisation — they were calendar.

8. **Two observations recorded, not interpreted.** (i) *The unsettled cell*: spillover
   direction — the one non-trivial forecast target left, since per-datum credit localized the
   update only partway (spillover RMS ratios 0.43–0.61) — is decoded at 0.289 by an observer
   holding the raw public configuration while every self source sits ≤ 0.062, including a
   dimension-matched twin holding the same information in the learner's own frozen encoding
   (0.036). The rungs are unstable (adding the action *lowers* the observer; the half-data
   control goes negative) and a one-hot-`x` vs fp16-`z` encoding asymmetry is unexcluded, so
   this cell is flagged, not claimed. (ii) *The arms' clocks* (single seed): `perdatum`
   commits L3 seven cycles earlier with a smaller table (8 vs 12 entries), fewer next-level
   observations (L3/L4 at support 51/55 vs 62/68), and π ends less consolidated (macro mass
   0.374 vs 0.484) — recorded for Track F.

## Interpretation (discussed with Jasper 2026-08-29 — argued, not measured)

- **(a) The self-update forecast has no signal-forming seat on this substrate.** Pooled: the
  target barely exists (finding 1). Per-datum: the target exists and its forecastable part is
  the δ already free on the wire as a scalar (findings 6–7). This is the fourth null for
  FM-as-signal-former (NTP twice, practice twice), and the E1b form — wiring explicitly
  separated from forecast skill, every update with one attributable cause — is the strongest
  form the negative takes here. Scoped to what was tested: signal *composition* on this
  substrate; not E2's object, and not a statement about other regimes.
- **(b) Under paper 2's taxonomy this is the law operating, not new evidence against
  self-models.** Both E1 targets are behavior-type; "privilege attaches to how you process,
  never to what public evidence taught you" predicts exactly the observed shape once the
  update's causes are named: NTP's conditioning gap was filled by the exogenous public token,
  pooled practice's by the batch, per-datum practice's by the grade — agency moves the *action*
  into the conditioning set, but the update's cause stays outside it until credit is per-datum,
  at which point the residual against the action-conditioned forecast *is* the grade's
  contribution, i.e. δ. The layer-span, implementation-fact object (paper 2's, and E2's) is
  untouched by any of this.
- **(c) Where it points.** The per-datum signal an outer loop or learning gate should consume
  is decision-time — δ and the deliberation state, both alive in this data (finding 3), both
  needing no FM. E3's gate should be built that way. The one temporal question this node
  *opens* rather than closes is whether an update's generalization (spillover) is cheaply
  knowable and by whom — finding 8(i), pending the symmetric re-encode check.
- **(d) The consolidation observation** (finding 8(ii)) reads as pooled/replayed credit being
  load-bearing for π's concentration — if it holds, "trust formation needs pooling" is a
  Track F instrument question, not an E question. Single seed; suggestive only.

## Caveats

- **Single seed, one arm per condition, one substrate.** Ranks and signs; the wiring/finding
  separations (Q1↔Q2, the dropped tautological target, the `delta_u` label) are design facts,
  not statistics.
- **The datum definition carries slack**: "the agent's own move" is the argmax-scored child
  (it is the sole survivor in the modal case; 44.3% of argmax children survived the topk);
  `u_kept`/`u_on_solved` are logged so the definition can be re-cut.
- **The observer ladder's low rungs are architecture-limited** (they must rebuild the frozen
  encoder through a probe stack and sit at ≈ 0): flat-at-zero licenses no privacy conclusion
  in either direction. No privacy claim is made anywhere in this node.
- **`e_did_pi == e_succ` exactly** in the per-datum arm, so grade-conditioning collides with
  π-step-occurrence; the refit's three target blocks exist for that reason.
- The maintenance pass is outside the revision tables (quantified: 4.8% median residual);
  π's realised Σlr came out 0.74× nominal (20.6% of sampled data solved); the ladder's
  effective n is 84 cycles; the E1b observer-inversion cell is unsettled (finding 8(i)).

## Runs on disk

| tag | what |
|---|---|
| `au_smoke` | E1 instrument mechanics + schema sanity at `--quick` (no commits at that scale) |
| `au_s0` | **E1 main**: instrumented anchor, bit-identical to `cd_s0/anchor`; phase-2 decode matrix + `--ladder` under `fit/` |
| `au_smoke_pd` | E1b mechanics: inverse gate, join, exact-zero invariants at `--quick` |
| `au_s1` | **E1b main**: `anchor,perdatum`; per-datum revision log; `--e1b` refit under `fit/` |

Volume `rhm-scaling-data:/data/rhm_practice_audiation/<tag>/`; fetched copies, `reduction.txt`
per tag and per fit, `schema.md` written by each run into its own outdir.

## Reproduce

Full command set in [`FILES.md`](FILES.md) §Reproduce (preflight, smoke, `au_s0`, the three
`fit.py` modes, `au_s1`). The fits are CPU-only and deterministic; the two big table caches
(`table_v2_*.npz`) are rebuildable and gitignored.

## Next steps (queued, not started)

**E2** — consolidation state off the depth residual: the implementation-fact gauge, the
paper-2-shaped practice target this node's nulls do not touch, and the clearly right next
E-node · **E3 reshaped** — the in-loop gate built on decision-time signals (δ + deliberation
state), no FM in the loop · the **symmetric re-encode check** on finding 8(i) before that cell
is cited anywhere · finding 8(ii) feeds Track F's F1 instrument · the `/update-beliefs` sweep
(ROADMAP §4.6) should carry this node's revision of the FM pillar's temporal half.

[^private]: Not mirrored: this link points to a document in the private lab repo (the roadmap, the queue, an unrun spec, reading notes, or a conversation). See the top-level README for what is held back and why.
