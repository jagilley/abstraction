# One Layer Deeper — repeated modular squaring as a ballistic forward-model substrate

**Up**: [../CLAUDE.md](../CLAUDE.md) (experiments) · **Files**: [FILES.md](FILES.md)
**Upstream**: [tilde-research/one-layer-deeper](https://github.com/tilde-research/one-layer-deeper) —
an architecture-and-optimizer competition from Core Automation × Tilde Research.
**Status**: four cuts complete plus the submission harness ([`dress_rehearsal/`](dress_rehearsal/README.md)). **Date**: 2026-08-25 (closing note on the practice arc added; last run 2026-08-22).

## What this is

The upstream benchmark asks for `y = x^(2^T) mod N` given `(N, x, T)`, with `N`'s factorisation
withheld — so without a shortcut the only route is `T` serial squarings. It fixes the data and
the outer loop and hands the participant the architecture, the optimizer, **and the training
loss**, with depth deliberately unconstrained.

We use it the way we use [RHM](../rhm/README.md) and [`mjc/`](../mjc/README.md): as a
**controllable DGP whose knobs we set and sweep**, not as a leaderboard to climb. None of the
science required a submission; one was nonetheless built and measured on the hosted tiers in
[`dress_rehearsal/`](dress_rehearsal/README.md) (2026-08-21/22), which is where the program's
findings meet the organizers' grader. What the substrate buys that ours do not:

- **It is purely ballistic.** The model commits at step 0 and never observes an intermediate
  state. Every controller in `mjc/` needed commitment installed as a knob, because reactive
  control re-grounds each step and is therefore a *near-blind grader* of forward-model quality
  (4b: transmission slope **+0.35** reactive vs **+1.07** ballistic). Here exact-match at
  held-out depth is a fully sighted grader by construction.
- **Ground truth is exact and free at every step**, so latent veridicality is measurable against
  a known `x_t` — and the per-step error amplification of the true dynamics is known
  *analytically* (squaring is `a ↦ 2a` in the discrete-log coordinate: the doubling map,
  Lyapunov exponent `ln 2`). Our other substrates had to manufacture the FM-quality axis and
  grade it with proxies.
- **Depth is an unbounded, scored axis** — which is the composition-horizon question with a
  scoreboard attached.

**The DGP knob that governs everything**: `x^(2^T)` is eventually *periodic* in `T`, so a badly
chosen modulus lets a model pass "depth extrapolation" by discovering a cycle rather than
iterating. `squaring_mod.py` computes the margin (`tail + period`, verified by brute force over
all units) and asserts against it.

### How our knobs sit against the upstream tiers

The manifests under `benchmark/manifests/` encode their config in the `data_root` path — e.g.
`..._easy_bidirectional_fixed_n_323_t123` is fixed `N=323`, train `T ∈ {1,2,3}`. The tiers vary
along **three** axes, not one: `fixed_n` (one modulus, vary `T`), `fixed_t` (one `T`, vary the
modulus — the arity axis), and `variable` (both). The `bNNNN` field is the *bit width* of `N`.

**The score is mean exact accuracy** — an example counts only when *every* target token is
correct. Easy/Medium average `test` (held-out **prompts**, i.e. held-out `(N,x,T)` tuples at
trained depths, so bases recur) with their merged `ood` split, equally. Hard equally averages a
hidden test split, a held-out-**depth** split, and a jointly held-out **modulus/depth** split.

**The depth ask on the public tiers is one doubling.** Read off `scripts/generate_datasets.sh`
(not the manifests, which only name the `data_root`): every public dataset has exactly *one* OOD
depth, and in all ten it is **2× the max trained depth** — E1 {1,2,3}→6, E2 {1,2,4}→7, E3/E4
{2}→4, E5 {1,2,3}→6, M1/M2 {4,8,16}→32, M3 {2}→4, M4 {8}→16, M5 {2,4,8}→16.

**The `T = 1,2,4,8,16,32,64` ladder is Hard's, and it is gated.** From the live leaderboard: each
cell shows the next `T` to certify and its accuracy, and *"once every example at that T is
correct, the target advances."* 100% exact, not an average. Two tracks are shown — in-distribution
number sizes and **out-of-distribution number sizes**. *(Corrected 2026-08-22: the **ranking** is
Hard's, the **readout** is not — upstream `4ceff95`'s README exposes the same `Max T` / `OOD N Max
T` fields on this ladder as diagnostics on Easy and Medium too, which is how
[`dress_rehearsal/`](dress_rehearsal/README.md) reads seven-rung profiles off Medium. Read the
2026-08-03 retraction below the same way: its "Hard-only" is about what gets ranked.)*

**Retracted (2026-08-03).** An earlier version of this section reported that ladder as the common
evaluation for *all* tiers and concluded the deep rungs were substantially degenerate (the "only
14.3% of Easy `e5`'s band clears 64" figure). Both are wrong: that ladder is Hard-only, and
against the real OOD depths upstream's moduli are clean. Recomputed with our own
`depth_first_repeat`:

| dataset | OOD `T` | first depth-repeat | clean? |
|---|---|---|---|
| E1 `N=323` / E2 `N=899` | 6 / 7 | 10 / 14 | ✅ |
| M1 `N=10403` / M2 `N=38021` | 32 / 32 | 42 / 48 | ✅ |
| E5 (10–11 bit population) | 6 | — | 99.0% of 308 |
| M4 (14/16-bit) / M5 (12–16 bit) | 16 | — | 96.3% / 95.2% |

Upstream chose its OOD depths carefully. What survives is the *design* point, not a criticism:
**our modulus sits at Easy's scale but is deliberately much deeper** — 10 bits, same as Easy's
variable range, with a periodicity margin of **67** against Easy `e1`'s 10, because we evaluate
to T=60 where upstream stops at 6.

| | modulus | bits | first depth-repeat | train `T` | eval `T` |
|---|---|---|---|---|---|
| Easy `e1` (fixed_n) | 323 = 17×19 | 9 | 10 | 1,2,3 | 6 |
| Medium `m1` (fixed_n) | 10403 = 101×103 | 14 | 42 | 4,8,16 | 32 |
| Easy `e5` (variable) | 10–11 bits | 10–11 | — | 1,2,3 | 6 |
| Medium `m5` (variable) | 12/14/16 bits | 12–16 | — | 2,4,8 | 16 |
| **ours** | **893 = 19×47** | **10** | **67** | 1..6 | **1..60** |
| ours, next rung | 9853 = 59×167 | 14 | 1149 | 1..6 | 1..120 |

**Hard may not be repeated squaring at all.** The problem statement: *"Hard is a hidden task
evaluation and may change aspects of the recurrence itself; do not assume it is repeated
squaring."* The repo agrees — `generator_family: str = "rsa_repeated_squaring"` is a *named*
field, `data/counting.py` is generic plumbing from a different task family, and
`tests/test_release.py` asserts no `h100_hard_*.json` manifest ships. Squaring-mod is the carrier,
chosen for three properties (inherently serial without the factorisation, exact cheap labels via
an evaluator-only trapdoor, monotone work in `T`); the competition's object is an architecture and
optimizer for *function composition*, and a public fixed recurrence would be memorisable — which
[`ballistic_depth/`](ballistic_depth/README.md) demonstrates by scoring 1.000 at T=60 having
learned only a 207-entry lookup table and how to compose it.

**The public leaderboard is stuck at `T=1` (checked 2026-08-03).** Every entry is an exact
multiple of 1/768: the best submission scores **6/768 = 0.78%** on in-distribution number sizes
and 3/768 on OOD ones; ranks 8–16 are at 0. Nobody has certified a *single* application of the
hidden map. That is the signature of a model banking only the free `x² < N` no-reduction cases —
the analytic floor [`variable_modulus/`](variable_modulus/README.md) §3 computes as `≈1/√N` — and
it independently reproduces that cut's central negative at Hard-tier compute with 16 participants.
**The benchmark's live frontier is rule acquisition, not depth.** *(Re-checked 2026-08-21 by [`dress_rehearsal/`](dress_rehearsal/README.md): still nobody has certified `T=1`, but rank 1 is now **16.54% = 127/768** on seen-`N` `T=1` with **0.00%** OOD, over 107 ranked entries. Ranks 2–6 sit at exactly 6/768 seen-`N` and 3/768 OOD — and that node measures the `[12,14,16]`-bit no-reduction floor as exactly 6/768, so the tier below rank 1 is the floor.)*

[`rule_acquisition/exact_atom/`](rule_acquisition/exact_atom/README.md) gives that a quantitative
form. Because the ladder gate is *every* example correct and
[`ballistic_depth/`](ballistic_depth/README.md) §9 turns a depth-`T` rollout into `T` depth-1
problems at test time for free, certifying rung `T` needs one-step error `eps <~ 9e-4/T` over
768 examples. So the rungs are a **log-error ladder**: each doubling of `T` costs one factor of
two in `eps`, `T=1` to `T=64` is 64×, and essentially all of the difficulty sits at rung zero.
That is why the leaderboard is stuck where it is, and it is the unit every number in that node
is reported in.

**The tier unit is wall-clock, not steps** (read off the manifests, 2026-08-02): `h100_easy_*`
allow **60 training seconds**, `medium` **600**, hard **3600**, on an H100 at bf16+AMP, batch 512,
under a `maximum_elements: 500,000,000` model-size cap. `ballistic_depth`'s 8000-step runs are
~1500 L4-seconds ≈ 190 H100-equivalent seconds at **2.1M** parameters — compute between Easy and
Medium, ~240× under the permitted model size. `variable_modulus`'s 60k-step runs at **5.8M**
parameters are Medium-to-Hard-scale compute (not logged). Any comparison should be in those units.

## Cuts

### [`ballistic_depth/`](ballistic_depth/README.md) — what actually extends a learned operator's composition horizon

A tied recurrent operator trained on terminal cross-entropy alone (train `T ≤ 6`) has a
composition horizon of **T ≈ 13**; a non-recurrent untied stack of matched depth is perfect at
every trained depth and **at chance one step past it** (1.000 → 0.009) — `operators_not_footprints`
with a scoreboard. Adding one **label-free** constraint — *the state you rolled into must be one
your own encoder could have produced* — moves the horizon to **T ≈ 51** (3.9×) at its swept
optimum, perfect out to T=20 and 0.98 at T=30, and lifts held-out-x generalisation from
**0.001 to 0.32**.

The mechanism is **manifold closure, not error suppression**: the base model's rolled state is
nearly orthogonal to the encoder's representation of the same residue (cos **0.19**) from step 1
while decoding it perfectly — a private trajectory, not a closed operator — and the constraint
takes that to **0.99**, with closure tracking accuracy across a 10× depth range. This reproduces
[`RHM_SCULPTING`](../rhm/RHM_SCULPTING_README.md) Stage 5b's *rollability ≠ depth* dissociation on
an unrelated substrate.

A follow-up micro-cut sharpens the mechanism into a behavioural claim. **Neither operator is
broken at depth**: hand the base model the true residue `x₅₅` and it rolls the last five steps of
a T=60 problem at **0.873**, while its own rollout to the same target scores **0.000**. What fails
is the state, not the map — the entire composition-horizon failure is the rollout leaving its own
input domain. The tightest statement is at zero rollout steps, where base cannot decode its own
encoder's output (**0.008**) and the constrained arm can (**1.000**). Across 72 runs closure
orders the horizon at Spearman **+0.94** — including through a knob reversal where the loss weight
stops predicting the horizon and closure does not — but it does *not* transfer on a calibrated
scale, and `re-entry-only` buys depth without buying closure at all.

Acting on that diagnosis works. Snapping the rolled state back onto the encoder manifold
*at test time*, using the model's own decode and **no retraining**, takes the ungrounded model
from 0.000 to 0.377 at T=60 and the grounded one to **1.000 at every depth out to T=60** — past
where `N = 893` can measure, since it repeats in depth at T=67. The intervention turns one rollout
into a chain of restarts whose accuracy goes as `p^⌈T/k⌉`, so what the training-time constraint
buys is the per-restart rate `p`: 1.000 for the grounded arm, 0.871 for the ungrounded one, which
decays exponentially in depth no matter how the period is tuned.

The cut's *predicted* mechanism — that the rollout fails by amplifying its own error, so
discretising the state is mandatory — is **falsified five ways**, with the instrument calibrated
along an on-manifold direction before the null was read. Hard quantisation failed to learn the
task twice while demonstrably achieving the error correction it was built for (per-step
contraction 0.889, accuracy 0.048): *error correction, achieved and useless.* Two further
results: the label-reusing half of the consistency term is **dose-dependently harmful** above
w≈0.1 (horizon 34.7 → 31.0 → 22.3), inverting the dense/evaluative complementarity the cut was
designed around; and the compute confound is dead in the useful direction — base at 2.5× the
gradient steps is *worse*, reproducing
[`metering_sweep`](../rhm/directed_sculpting/full_loop/metering_sweep/README.md)'s within-round
overfitting on a new substrate.

**Scaling to `N = 9853`** (11.6× the state set) breaks two things and taught us to distrust a
third. The cycle term at §7's optimum **collapses at this scale** (ID 0.001, closure 0.998),
capacity-independently — the cause is the *schedule*, not the weight, and repairing it restores
learning but not the advantage. That makes explicit that **closure is only interpretable
conditional on in-distribution competence**. The horizon numbers at that scale then turned out to
be **unconverged** — 2× the budget takes exact-match at T=7 from 0.176 to 0.996 with ID saturated
at 1.000 in both runs, while `N=893` is converged under a 2.5× control
([`horizon_convergence/`](ballistic_depth/horizon_convergence/README.md)). So `N=9853` horizons
are lower bounds, no cross-scale horizon comparison is quotable yet, and §1–§9 stand. And the
child node
[`rule_structure/`](ballistic_depth/rule_structure/README.md) reads the representation in the CRT
coordinate where squaring is exactly the doubling map: across six trained encoders there is **no
group organisation beyond a permutation null** — all below the weakest synthetic group signal the
instrument still detects — and scaling the state set moved it *down* rather than up, the opposite
of what "lookup is merely cheaper than the algorithm" predicts. **Scoped by
[`rule_acquisition/`](rule_acquisition/README.md) §5**: an arm that generalises at 0.920 reads at
the same null, so the statistic bounds *group-character-linear organisation*, not learnability —
that node's own caveats say as much, and its **unreachable** headline should be read accordingly.

### [`variable_modulus/`](variable_modulus/README.md) — the arity-2 cut: sampling `N` instead of fixing it

Fixed-`N` never forces the operator to be *conditioned on the rule*, so this cut samples `N` per
example (26 train / 7 held-out 3-digit moduli, first depth-repeat ≥ 29) and separates arms by
**where the rule lives**: carried in the rolled state (`fold`) vs re-injected into the operator
every step from `N`'s digits (`cond`).

`ballistic_depth`'s headline does not reproduce. The same label-free closure constraint installs
closure just as completely (cos **0.29 → 0.97**) and moves the composition horizon **10 → 12**,
against 13 → 51 at fixed `N`. Measuring why gives the cut's result: **two independent limits on
depth, with a double dissociation.** The cycle term multiplies closure 3× and leaves the
per-restart rate untouched (0.894 → 0.895); state coverage doubles that rate (0.47 → **0.98**,
`p ≈ coverage`) and leaves closure and the raw horizon flat. So the *unaided* rollout is
drift-limited while the *re-projected* rollout is coverage-limited, at `p^⌈T/k⌉` — 0.048 → 0.908
at T=28 across the coverage sweep, with identical closure. The oracle re-projection ceiling is
**0.894–0.903 across arms whose closure spans 0.15 → 0.97.**

Separately, **no arm learned modular squaring**: held-out `x` and held-out `N` sit at or below the
0.041 accuracy obtainable with no rule knowledge at all (the `x² < N` no-reduction cases), and a
500k-step probe with half the bases held out, constant LR and `wd ∈ {0.1, 1.0}` produced no phase
transition. Within these scales the model represents the task as per-modulus lookup tables, which
bounds what depth results on this substrate can mean — including the fixed-`N` 1.000 at T=60,
also measured on seen bases.

**Four of six pre-registered predictions failed**, recorded as such. The sharpest: re-projection
was expected to favour `fold` (the snap re-supplies the rule, which `cond` already has) and helps
both identically — *a static rule is cheap to carry*, a genuine disanalogy with `mjc`/RHM where
the command changes every step. An early "more moduli → shorter horizon" reading was **retracted**
— it was a fixed-depth readout on a steep curve; horizons are 8.6–11.6 across a 7× state-count
range. Three `ballistic_depth` results replicate unchanged: the flat oracle plateau, the chain
model, and the collapse at α=0.5.

### [`rule_acquisition/`](rule_acquisition/README.md) — why the one-step map is memorised

The atom, not its composition. `variable_modulus` §3 says the model cannot compute a *single*
modular squaring on an unseen input, and `rule_structure` read that as **unreachable**. Testing
that node's remedy list at depth 1 with half of every modulus's bases held out: **nine arms sit at
or below the no-reduction floor** — base-2 encoding, place-value embeddings, 8 encoder layers, 8
serial operator steps per task step, 4096-wide FFN, 17.75× more moduli, staged supervision on the
intermediate product, and a 21M-parameter arm stacking all of them (**0.051** against floor 0.049).

The **reduce** generalises once shown its own input space — `x^2 -> x^2 mod N` reads
**0.773–0.996** uniformly sampled against **0.022** on the inputs `x^2` actually supplies. Its
space is `N` times larger than the multiply's at equal sample count, so within these scales what
makes the atom memorisation-only is **coverage**, not the reachability of the algorithm — which
retro-explains the nine nulls, since none of those interventions manufactures coverage. Division
degrades gracefully with quotient range (0.996 → 0.982 → 0.773), not off a cliff. *An earlier
version of this section reported the multiply generalising at 0.920; that is **retracted** as a
per-modulus-split artefact — see the child below.*

The surviving barrier is the **rule** axis: held-out `N` is at or below floor at 8 moduli even
where held-out problems read 0.982, and reaches only 0.058 (3.8× floor) at 142 moduli, both arms
undertrained. Two pre-registered predictions failed.

Its child [`exact_atom/`](rule_acquisition/exact_atom/README.md) re-reads the atom against the
**exactness** target Hard gates on. Under `ballistic_depth` §9's re-projection at `k=1`, rung `T`
needs one-step error `eps <~ 9e-4/T`, so each rung is worth one factor of two and the whole
ladder `T=1..64` is 64× — depth is the cheap axis. Against it the atom decomposes into three
measured quantities: fed the true `Enc(DIV, N, x^2)` from inside a trained composed model the
reduce reads **0.953** on held-out `x` (converting the coverage inference into a measurement),
the multiply reads **0.068** on a clean split, and the composition reads 0.0035. The multiply's
failure is ordinary — at 3 digits `x` has under 1000 values; at 50k inputs and 27.4M parameters
it reaches **0.909** and is still improving steeply, and a pre-registered "generalisation
appears where memorisation becomes infeasible" mechanism **failed in both directions**. The two
widths tested bracket the problem: at 3 digits the reduce works and the multiply does not; at 4
digits the multiply improves and the *full-range* reduce collapses to 0.0004 while
bounded-quotient reads 0.986, so the reduce's limit is the **quotient range** — which `x^2`
spans by construction. Held-out `N` stays at or below floor everywhere, including under an
oracle state.

Its sibling [`staged_reduce/`](rule_acquisition/staged_reduce/README.md) acts on that
localisation and **moves the rule axis** — the one nothing in this program had moved. Schoolbook
long division makes every stage a bounded-quotient reduce, chained *at test time only* with the
remainder re-grounded through the model's own decode: §9's re-projection one level down, and
exact rather than a manifold projection because the remainder is a digit string. Step-matched at
600k it reads **0.9968** against a monolithic control's 0.4841 at 3 digits and **0.9899** against
**0.0005** at 4 digits — the width where the composed atom's reduce blade had failed — and
rejection-sampled chains none of whose stage queries was ever trained on read **0.9925**, so this
is generalisation rather than coverage. Held-out `N` goes from at-or-below floor to **0.9826 /
0.9790** across two seeds, per-modulus minimum 0.851 over 36 unseen moduli, against the 0.058 that
was the prior best.

The mechanism is an **interaction**, which is the transferable part: a family-matched 8-modulus
arm learns staged division essentially perfectly (0.9943 on held-out `y`) and still sits at floor
on held-out `N` (0.0042), while breadth without staging moves it only 0.0012 → 0.0025.
Suggestively, staging does not teach the rule — it changes the function into one that *has* a
shared form across moduli, and breadth supplies the evidence that the form is shared. Two scope
limits: the compute control is **not** null, so re-grounding dominates compute rather than being
the whole story; and no cell certifies rung `T=1` (best `eps` 3.16e-3 against 9.0e-4), so this
moves the axis that was stuck without clearing the ladder.

### [`dress_rehearsal/`](dress_rehearsal/README.md) — the organizers' evaluator, a submission port, and the first full-budget Hard measurements

The one node here that uses the benchmark *as* a benchmark. It runs the **organizers' evaluator
unmodified** (upstream `4ceff95`) on the organizers' data — our own Modal H100 for smoke only,
then the hosted Easy/Medium/Hard tiers, which are free H100 time that return the **full seven-rung
profile** — with a port of this node's machinery into the competition contract: one file, two arms
sharing every parameter, `control` (terminal CE) vs `closure` (plus `ballistic_depth` §2's
label-free cycle + re-entry via `auxiliary`), loop-on-`T`. **15 hosted runs, all saved.**

Three things came out. **(1) The board is legible.** The `[12,14,16]`-bit no-reduction floor
measures **exactly 6/768** at the gating rung — ranks 2–6's score to the example (a model that
learned the multiply and not the reduce); only 14 twelve-bit moduli exist, so that cell is
necessarily dense and rank 1's 127/768 with 0% OOD is what interpolating it would score; Hard's
split names pin a modulus-grouped generator. **(2) At full Hard budget the control arm memorises
58% of the hidden training set and transfers nothing** — 0/768 at every rung on both profiles,
test loss 11.4 nats — the rule-acquisition wall of
[`variable_modulus/`](variable_modulus/README.md) and [`rule_acquisition/`](rule_acquisition/README.md)
measured on someone else's DGP at 3600 H100-seconds; the organizers' own baseline and ranks
39–107 sit below the floor too. **(3) `ballistic_depth`'s fresh-`x` result (0.001 → 0.32) did not
reproduce in the port**, even with the cycle term rebuilt to the research mechanism and verified
(gradient into the real encoder, bit-exact re-encode, ramp on a fully-memorised model): on a single
seen modulus with 90% of units seen, control / closure-v2 / closure-v3 read **2 / 4 / 1 of 140**
fresh `x` at `T=1` against a floor of 4. Two port defects explain the earlier hosted nulls and are
recorded (a soft rung-selector that could not memorise; a clock-gated aux ramp that froze an
un-memorised model). The only whisper is v3 sitting just above the OOD-`N` floor on two datasets —
counts of 2–4, noted not claimed. Best leaderboard position: rank 31 of 109 via one
correct example. Next-step possibilities — a digit-arithmetic architecture aimed at the floor tier,
an ablation back toward cut1 to find what was load-bearing for fresh-`x`, epochs-vs-updates levers —
are in the node.

## Shared machinery (lives at this node)

**`squaring_mod.py`** — the task, vendored from upstream. Token ids, decimal digit encoding,
field markers and the trapdoor label path (`pow(x, 2^T mod φ(N), N)`) are verbatim; the ~900
lines of split machinery are dropped. Two documented departures, both stated because they are
the kind of thing that silently confounds a depth experiment:

1. **It emits the full trajectory** `x_0 … x_T`. The evaluator never gives you this. It is
   *research-only instrumentation* — no arm trains on an intermediate residue — and it is what
   makes latent veridicality measurable per step.
2. **Fixed-width zero-padded answers**, so exact-match does not tangle answer *length* with
   answer *value*.

`TaskSpec.describe()` reports the depth-periodicity margin, the reachable-state count, and the
unit count; `build_trajectories` cross-checks columns against the independent trapdoor path.

**`shared.py`** — Modal app (`one-layer-deeper`), image, volume (`one-layer-deeper-data`),
`NumpyEncoder`.

## Reproduce

```bash
cd experiments/
MODAL_PROFILE=chromatic modal run --detach \
  one_layer_deeper/<cut>/<script>.py::<entrypoint> --tag <tag> --seed <n>
```

Use `--detach` for anything over ~2 minutes and invoke the function explicitly, not the local
entrypoint. Each cut's README carries its exact commands. Modal volume layout:

```
/ballistic_depth/<tag>/results_seed<N>.json
```

## Next steps

1. ~~**Scale `N`** — *base arm done, grounded arm pending a repair sweep.*~~ — the repair sweep
   is reported; see `ballistic_depth/` §10. It is the **schedule, not the weight**: at warmup 0.6
   the cycle arm learns again (w=3 reads ID 0.996 against 0.003 at warmup 0.3), and with the
   schedule fixed no weight extends the horizon past base's 7. Repair restores learning, not the
   advantage. The `N=9853` baseline is itself unconverged
   ([`horizon_convergence/`](ballistic_depth/horizon_convergence/README.md)), so whether the
   closure advantage survives a larger state space is still open and §9's `p = 0.871` is still
   unseparated. Original framing kept below.
   The base horizon halves (13 → 7) and is capacity-independent; the cycle arm at w=10
   collapses at this scale, so the headline question — does the closure advantage survive a
   larger state space — is open until `scale9853_w{1,3}` / `_w3_warm6` report. §9's `p = 0.871`
   also remains unseparated, because that needs a grounded arm that trained.
2. ~~**The axis the benchmark is actually stuck on is rule acquisition, not depth.**~~ — opened;
   see [`rule_acquisition/`](rule_acquisition/README.md). Our depth results remain about composing
   a *memorised* operator, but the reason the operator is memorised is now measured rather than
   assumed: the atom's reduce stage is starved of coverage by its own composition, and both halves
   generalise when each is given its own input space. The **rule** axis — generalising the reduce
   across moduli, the same axis `variable_modulus` hit from the other direction — is no longer
   the open one: [`staged_reduce/`](rule_acquisition/staged_reduce/README.md) takes held-out `N`
   from floor to 0.98 by decomposing the reduce into bounded-quotient stages *and* widening the
   modulus set, neither of which works alone. What is open now is narrower and more concrete:
   whether breadth keeps paying past 142 moduli (that node's dense arm had a degenerate held-out
   selector and cannot answer it), and whether the composed atom works at 4 digits now that its
   reduce blade does.
3. ~~**Held-out modulus** — the arity-2 cut.~~ — done; see [`variable_modulus/`](variable_modulus/README.md).
   The arity axis turned out to be the *weakest* result in it: a static rule is cheap to carry, so
   `fold` and `cond` differ by 2 depths and share a re-projection ceiling. The connection to the
   length-gen finalizer's *"width amplifies arity in an open loop"* is **not** established here —
   that setting has a rule that changes every step, which this one does not. A substrate with a
   non-static rule is the open version of this question.
4. ~~**Does closure predict the horizon across arms?**~~ — done; see `ballistic_depth/` §6–§7.
   Kept below for the original framing. Twelve-plus configurations are already run;
   closure-at-fixed-`t` against horizon would turn the mechanism claim from a two-arm contrast
   into a slope.

## Would the practice arc help here? (2026-08-25 — assessed, not run)

Asked by Jasper against PR #69[^private] and the
[practice arc](../rhm/practice/README.md): if the learner may **mint new tokens for the phenomena
it encounters** and run the standard loop over them — mine units from its own solved trajectories,
commit, re-read the archive with the climbed vocabulary
([`reread/lm`](../rhm/practice/reread/lm/README.md)), consolidate into planner and executor
([`native`](../rhm/practice/native/README.md), [`spiral`](../rhm/practice/spiral/README.md)) —
could that move the atom? The assessment is **no, for a structural reason**, recorded so the next
agent with the idea finds the argument rather than re-running it.

**Every practice op moves the unit ladder *up*; the wall here is one level *down*.** Practice's
object is composition — `T[ℓ] ⊆ T[ℓ−1]×T[ℓ−1]`, mined from trajectories over exact finer units,
committed verbatim, re-grounded at chunk boundaries. On this substrate that axis is depth `T`, and
depth is the cheap axis: re-projection through the model's own decode gives `p^⌈T/k⌉`
([`ballistic_depth`](ballistic_depth/README.md) §9), the staged reduce chains at test time with an
exact digit-string snap, and the per-digit carrier composes flat to `T=64` with held-out depth at
500/500 ([`second_pass`](dress_rehearsal/second_pass/README.md) §2.3). The *conceptual* content of
practice — ballistic within, discretise at the boundary, select rather than average — is what
solved depth here; `digit_port`'s `s4` (removing the per-application gradient contraction from a
bit-identical forward pass generalises 13× worse) is the same selection-not-averaging fact read
from the loss side. Practice has, in effect, been applied, and it did what it does.

The stuck axis is the atom's `eps`, and the only thing that ever moved it on the rule axis is a
**finer** unit than the one we are handed: [`staged_reduce`](rule_acquisition/staged_reduce/README.md)'s
bounded-quotient long-division stage (held-out `N` floor → 0.98, monotone in stage count — R=10
beats R=100). That level does not self-organise even when the architecture supplies slots for it
([`terminal_only`](rule_acquisition/staged_reduce/terminal_only/NOTES.md): seam collapses to ~12
values, CE flat from the first log point); it had to be *installed* by the single-stage
distribution.

**The disanalogy with RHM that decides it: RHM's hierarchy is in the data; long division's is in
the computation.** In RHM every level is literally present as recurring spans of the token stream,
which is why extraction is bottom-up and why re-reading is renewable as the vocabulary climbs —
the levels are there to be extracted. Here the levels below the atom (quotient digits, partial
remainders) never appear in the stream: the learner observes `x` and `y` (and, in research mode,
the residue trajectory), never `x²`, never a partial remainder. Minting tokens over "phenomena
encountered" can only name what is encountered — residues (naming them *is* the lookup table the
model already learns) and pairs of residues (naming those is depth, already cheap). The token that
would help, "quotient digit at stage *i*", has nothing in the stream to be mined from.
`mine_from = chosen` — the constraint [two_climbings](../../ideas/two_climbings.md) §5.4 reads as
practice's virtue (you cannot mine a chunk you have not executed) — is the blocker here.

The other two components address conditions that are already ideal: allocation manufactures
recurrence (the dataset is fixed and recurs perfectly); metering needs a sighted grader
(exact-match is fully sighted by construction — the reason this substrate was chosen); arrival
and trust ([`census`/`assay`](../rhm/practice/census/README.md)) are about a planner learning to
route among addresses, and there is no planner here — one deterministic thing to do per step.

**What would make this wrong.** Reformulate the substrate so the agent *has* sub-atomic exact
primitives (digit compare, subtract `kN`, shift a digit in) and emits trajectories over them. Long
division is then a level-2 macro over solved programs, and the practice arc would plausibly
reproduce on it — mining, routing, quarantine, trust. But that is program synthesis with a given
calculator: it abandons the question this node was built to ask (can a network *learn* the atom
from terminal labels), is illegal under the competition's rules, and has a static rule, so it does
not even supply the non-static-rule substrate next-step 3 above names as the open version of the
arity question. It would be RHM with an arithmetic grammar.

**The one practice-shaped question with a cheap form** is already queued in
[`second_pass`](dress_rehearsal/second_pass/README.md) next-steps 3: warm-start the stage map on the
single-stage distribution, then fine-tune terminal-only — does a terminal label *maintain* a
decomposition it cannot *discover*? That is the assay's "content is giftable, use is only earnable"
asked in this substrate's coordinates (~16 L4-hours). It scopes the practice claims; it does not
crack the atom. Left queued.

[^private]: Not mirrored: this link points to a document in the private lab repo (the roadmap, the queue, an unrun spec, reading notes, or a conversation). See the top-level README for what is held back and why.
