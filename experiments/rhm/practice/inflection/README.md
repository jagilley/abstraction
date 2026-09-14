# inflection — a rendering rule below the tables: the executor gets a word to say, the reader a word to hear

**Up**: [`../README.md`](../README.md) (practice arc) · **Spec**: [`SPEC.md`](SPEC.md) ·
**Machinery record**: [`FILES.md`](FILES.md) (every gate, run, flag and diagnostic; decisions in
[`DESIGN.md`](DESIGN.md)) · **Offline sizing**: [`sizing/SIZING.md`](sizing/SIZING.md) ·
**Conversations**: `CONVERSATION.md`[^private] (the spec, then this round).
**Machinery donors** (all untouched, forked or imported): [`../tutti/`](../tutti/README.md)
(`inflection.py` forks `tutti.py`; every addition `# [inflection]`-marked, every knob default off,
G-F replay 0.000e+00) · [`../../rhm_sculpt_precheck.py`](../../rhm_sculpt_precheck.py) (the coin
and the any-synonym grader) · [`../crystallize/units.py`](../crystallize/units.py) ·
[`../ratchet/macros.py`](../ratchet/macros.py) · [`../native/span/span_net.py`](../native/span/span_net.py)
· [`../intonation/`](../intonation/FILES.md) (δ_perf, the live executor) ·
[`../maestro/policy.py`](../maestro/policy.py) (A1's rule, imported) ·
[`../../../mjc/practice/tempo/README.md`](../../../mjc/practice/tempo/README.md) (the plant's form
of the question).
**Runs**: 2026-09-09 → 10, fourteen tags, ≈ 24 GPU-h, seed 0 throughout, every fork gated
bit-identical against its donor with the knobs off. **Ranks, signs, mechanisms located by
replay, and multiples of measured floors are the claims.**
**Attribution**: the question (does `tempo`'s factoring speak to practice or to inductive bias;
the intent → word → evaluate picture; the perfect-dictionary reading of the E′/F2 nulls) is
Jasper's, from the conversation that wrote the spec. The ordered-register family and the
shared-morphology bound are the sizing lane's; the fork's mechanics, the `leaf` side table, the
renderer's training signal, the setup-render diagnosis and the A1 replay are the builder's; the
design decisions across rounds and the readings below came out of the orchestrating conversation
and were discussed with Jasper in three rounds.

## One-liner

On every prior RHM practice node the map from a level-1 feature to its surface tuple had no free
parameter in either direction, so the spelling step was never a skill and δ_perf never included
it. This node makes the bottom synonym choice a context-dependent rule the learner must own and
makes the grader report spelling beside meaning. **A learner that stores chunks as features and
fits a renderer arrives at the plant's factoring where it is not free**: fitted renderers spell at
≈ 0.02 written error against 0.42–0.48 for a chunk stored as its own tape or written at synonym 0;
a renderer told the register is a *number* transfers to unpractised registers and improves
monotonically with registers practised (0.68 → 0.71 → 0.875 held-out), where one told it is a
*label* falls back to a single column and does not; a per-feature switch point is identifiable
only once an interior register is practised — `tempo` finding 6 with the axis named. On a matched
clock with the setup rendered as the world spells, correct spelling is worth ≈ 0.08 in meaning
error (4.65× the floor) **through the learner's own planner rather than the grader**. And with a
word to hear, the two execution-currency questions that motivated the spec answer plainly: trust
is habit (π's per-slot mass is use count, in every arm), and δ-silence as advance owner is not
scale-free (a larger-amplitude error component makes its fixed dead zone fire on scale, not on
convergence).

## The question

`tempo/` found on the plant that a chunk stored as forces transfers to no tempo, while a chunk
stored as a path and converted by a body model fitted from the learner's own slow practice
transfers three octaves — provided practice spanned enough speed for the model to be identifiable.
Jasper asked whether that speaks to practice as a phenomenon or to executor inductive bias, and
what the RHM analogue would be. The answer in code: RHM never posed the question. The data pick
each node's synonym by a context-free coin, the grader accepts any synonym, the executor writes
synonym 0 (`canon`), and the reader is pretrained on the coin corpus and frozen at ~1.0. RHM had
the factoring for free, so it could not fail the tempo test — and every execution-currency result
on RHM (the E′ gate-seat nulls, F2's trust-vs-habit) was read inside that condition, where trust
could only ever be proposal mass.

**The one change**, below the tables so every organ above level 1 carries over: the synonym index
at a level-1 node becomes a function of a context variable, and the grader marks spelling beside
meaning — two numbers, never one. The arms: `canon` (synonym 0 always), `given_rule` (the true
rule handed — the ceiling), `leaf` (the chunk carries the spelling it was recorded with — the
force head's twin), `fit_rule` (feature-keyed chunks plus a renderer fitted from the rule-spelled
blocks in the instances the learner solved, with the register as a scalar), `fit_index` (the same
with the register one-hot), and later `fit_shared`, the GF(2)-affine contrast and the
curriculum rungs. Sequence per the spec: Q0 offline sizing, Q1 the fork at level ≤ 3, Q2 the
curriculum ladder, Q3 the crank with the E′/F2 readouts live.

## Findings

### 0. What the rule can be made of, sized offline (`sizing/`)

At m = 2 the per-(feature, context) choice is one bit, and **any fully shared rule
`k = a_f ⊕ g(context)` has at most two effective contexts** — the spec's own `(offset_f + r) mod m`
included — so four contexts force per-feature context sensitivity. The **ordered register with a
per-feature threshold** (`K[f, ρ] = 1{ρ ≥ θ_f}`, ρ ∈ 0…7) is the one family a generic renderer
with a *scalar* register input both represents and extrapolates (by monotonicity); the same head
given the register one-hot pins nothing; its bottom rung ρ = 0 is `canon` itself. A strict grader
would reject 75–94 % of current meaning successes at the deepest damage rung, so the grader is
graded. Context is never needed to parse to the root at `rule_seed 0`: the two bottom-tuple
collisions die at level 2. And `tall/`'s m = 4 verdict re-measures as tuple-space occupancy m/v
rather than m (gradient 2.49× at v = 16, m = 4), though widening still costs the L4 rung.

### 1. The factoring pays where it is not free — on spelling, in every world

Lifetime written-block spelling error, three worlds and three clocks:

| arm | `if_q1` (collision draw) | `if_q1b_cf` (collision-free) | `if_q1c_yk` (yoked clock, setup on-rule) |
|---|---|---|---|
| `canon` | 0.396 | 0.480 | 0.459 |
| `leaf` | 0.237 | 0.420 | 0.427 |
| `fit_index` | 0.020 | 0.024 | 0.016 |
| `fit_rule` | 0.018 | 0.025 | 0.023 |
| `given_rule` | 0.000 | 0.000 | 0.000 |

The order never moves. `leaf`'s tape is the middle register's column (the majority spelling over
the practised registers), so it is right at ρ = 3 and its aliases and wrong at both ends; its side
table records 1.5–2.3 distinct spellings per chunk (max 7), bounded by the number of practised
registers rather than by m^span, which is what a register rule buys a non-factoring learner.

### 2. Number beats label; transfer is decided by the renderer's class, coverage by practice

Held-out registers at three practised (`if_q1b_cf`, no collisions): the scalar head is right on
**0.875** of held-out cells against the one-hot head's **0.750**, and per register the one-hot head
still falls back to a wrong constant at ρ ∈ {5, 6} (0.43–0.64 written error) where the scalar head
is 0.06–0.34. On the **GF(2)-affine world** (`if_q1b_a`, each feature listening to one of two
register bits), a family-aware counter recovers the true (a_f, w_f) at its first fit and a
one-hidden-layer MLP by cycle 11, while the heads with a *shared* context response — one-hot
register, or additive in the bits — cannot fit even the practised registers (0.667) and collapse
to a constant row. So transfer is first a question of whether the rule lies in the renderer's
class; practice decides how much of it coverage pins. (The held-out register r = 3 there is the
family's all-ones column, so any head emitting 1 at an unseen register scores 1.000 — an artefact
noted for the rerun.)

### 3. The curriculum ladder — `tempo` finding 6's twin, with the axis named (`if_q2_*`, `if_q1c_yk`)

| head | practised | acc held-out | offline ceiling | θ̂ (true [1,1,6,4,5,5,5,1]) |
|---|---|---|---|---|
| scalar | {3} | 0.679 | 0.74 | one shared threshold |
| scalar | {0, 7} | 0.708 | 0.79 | one shared threshold |
| scalar | {0, 3, 7} | **0.875** | 0.81 | [2,2,5,5,5,5,5,2] |
| one-hot | {3} | 0.679 | ~0.56 | one shared threshold |
| one-hot | {0, 7} | **0.417** | 0.59 | one shared threshold |
| one-hot | {0, 3, 7} | 0.750 | ~0.60 | [1,1,7,7,7,7,7,1] |

The scalar head is monotone in contexts practised; the one-hot head is not — more practice at the
endpoints reinforced a wrong default. At one register, and at two registers that bracket the
scale, both heads recover **one switch point for all eight features**; only once an interior
register is practised do the features separate. That is "the drag term is below the noise at one
tempo" with the axis stated: coverage of the middle, not the extremes. Transfer error grows
monotonically with register distance for the scalar head from a single practised point
(0.02 → 0.63), while the one-hot head is worst at the near neighbours (0.74) — proximity buys an
index nothing.

### 4. The outcome currency, read properly: spelling costs meaning through the learner's own planner

Meaning is spelling-agnostic by construction (`possible_sets` accepts any synonym), yet on a matched
clock with the setup rendered as the world spells (`if_q1c_yk`: 140 cycles, 0.03 % spend spread,
commits within two cycles) the five arms separate into exactly two clusters:

| cluster | arms | e at end | spread ÷ floor |
|---|---|---|---|
| correct spellers | `fit_rule` 0.607 · `given_rule` 0.612 · `fit_index` 0.617 | 0.61 | 0.6× |
| misspellers | `canon` 0.698 · `leaf` 0.701 | 0.70 | 0.15× |

The gap is **4.65×** the in-tag null-ABBA floor on `e` (v_tol 0.0173); lifetime solves order the
same way (38.6k–40.3k vs 33.6k–34.3k). *Which* correct renderer — handed, fitted-scalar or
fitted-one-hot — is sub-floor. The channel is the planner: the stale value head is trained on
rollouts, and a rollout that spells unlike the world is off its distribution. The sign of the
effect flips with what the value buffer was rendered through (§Apparatus lessons), which is what
locates it in the learner's own models rather than in the grader.

### 5. The reader's job arrived by measurement, in three forms

Not where the spec put it — context is never needed to undo the morphology at this draw — but:
(i) at the collision draw the harvest reads a feature through the last-writer-wins bottom map, so
world-spelled f1 is recorded as f4 and f2 as f7 at the registers where they spell the colliding
codes; the fitted heads learn the rule **modulo the reader's confusions** (7 of `fit_rule`'s 15
deviations from K substitute a readable spelling at exactly a reader-unreadable cell), which is
the gap between 0.70 and the 0.81 ceiling; on the collision-free draw both heads fit all 24
practised cells. (ii) `fit_shared` — the harvest's parse resolved by the renderer's own table —
replays `fit_rule` at 0.000e+00: the learned table inherits the reader's tie-break, so the one
collision it can resolve resolves to the reader's own wrong answer. A fixed point, not a
bootstrap; escaping it needs a reader that notices a feature's mass has vanished from a register.
(iii) At one practised register the frozen reader collapses to 0.42–0.52 at exactly the registers
whose leaf codes it never heard (18 new codes), and at two it degrades to 0.72–0.77 at interior
registers whose codes it knows but whose register pattern it has not seen. The neural reader is not
a lookup.

### 6. Trust is habit (`if_q3_f2`)

On the yoked clock, for 24 committed slots × 19 probe cycles per arm: π's per-slot mass correlates
**0.73–0.78 with use count** in `leaf`, `canon` and `given_rule` alike. Slots in `leaf` and `canon`
carry graded spelling reliabilities (363 and 357 distinct values), yet π's relation to reliability
is ≤ 0 and, in `leaf`, collapses to −0.02 once use is partialled; the small positive relation to
solve-when-used (0.09–0.14) survives partialling but is the same size in `given_rule`, which has no
spelling variation. With execution fallible at the spelling step, trust and habit can be measured
apart, and they are the same thing here.

### 7. δ-silence as advance owner is not scale-free (`if_q3_e`, `if_q3_e_sp`)

Under `tutti`'s mirror (yield commits, δ-silence advances) in pairs differing by whether `e`
includes spelling: the **`fit_rule` pair is inert by mechanism** — the priced beam draws only
practised registers, where the fitted table is exact, so `e_spell ≡ e_feat` on every head execution
(0 of 117 live cycles differ) and the pair's divergence is one edge advance displaced by the
re-derived floor. The **`leaf` pair moves**: the spelling-charged `dsil` sits at 0.41–0.43 against
0.03 feature-only, δ-silence fires two quiet advances where the twin fired none in 140 cycles (at a
*harder* floor, 0.00507 against 0.0046), and the arm ends its ladder at c62 with e 0.969 against
c140 at 0.742. Replaying A1's rule on the off arm's own two series located why: the quiet statistic
is a raw slope in the series' own units against a fixed absolute dead zone behind a
positive-then-flat latch, with no normalisation (a +0.40 offset replays to every digit), and the
spelled series has ~3× larger increments while being relatively smoother, so the same dead zone
both latches `moved` — which the feature-only series never reaches in era 3 — and swings back
inside. The extra advances are the dead zone applied to a larger series, not a plateau. The
replay's first firing is the cycle the on arm actually advanced.

## Apparatus lessons (each caught by an instrument, none by a treatment)

- **The value buffer must be rendered as the world spells.** It had been collected with `canon`
  writes, so every meaning rank in the first three tags was the value head's familiarity with the
  written spelling: `given_rule`'s register gradient in meaning error at L1 flattens from slope
  +0.039 to −0.002 once the setup renders on-rule (`if_q1c_pr`), `canon` unmoved. `tempo`'s "train
  the head on what it will be handed," in our clothes. Those ranks are withdrawn in the record.
- **Yield-paced arms are not comparable on the outcome currency** (14–43 % spread in priced spend,
  eras reached at different cycles); the yoked clock is.
- **A collision-free rule draw deletes the arc's aleatoric channel**, so its mining dynamics are
  not `tutti`'s; every clean number here is within-tag.
- **The governing `tol_dsil` 0.0046 re-derives to 0.00236 feature-only on this world** — ~2×
  conservative, a fact about the transplant; the spelled floor is 0.00507.
- A shared-parse instrument must count *resolutions*, not only *changes*; a harvest that includes
  the learner's own rewritten span carries a weak closed loop (`on_rule_frac` 0.77–0.89, logged).

## Interpretation (discussed with Jasper across the round — argued, not measured)

- **The RHM twin of `tempo`'s factoring holds, and separates the two things Jasper asked about.**
  Whether a renderer transfers at all is decided by whether the rule lies in its hypothesis class
  (number vs label on the ordered register; per-feature vs shared response on the parity world) —
  the inductive-bias half. How much of the rule practice pins is decided by which contexts were
  practised, and the answer has the same shape as finding 6 with a sharper axis: the interior of
  the scale, not its ends — the practice half. Both are true and they are separable.
- **Spelling is a pure execution-currency signal here, and it still reaches the outcome — through
  the speaker, not the listener.** The grader is blind to it by design; the learner's own planner
  is not. That is the two-δ split made live at the spelling step, and it says where a wrong word
  costs: in the plan the speaker can no longer read, not in the listener's verdict.
- **The E′/F2 questions asked with a word to hear.** F2's answer is the same as before, now
  measured rather than confounded: π's trust is proposal mass. E′'s pacer seat gives a
  mechanism-level answer instead of a null: with a learnable renderer the spelling step is solved
  before the seat is live, because the renderer learns from the world's abundant surface far faster
  than the ladder; with an unlearnable one, the seat's statistic is not scale-free and fires on
  amplitude. For the spelling step to be a live signal at the pacer's timescale, the renderer must
  be paced by the meter — learned from scarce own productions, as the plant's body model was — and
  the advance owner needs a scale-free gauge.
- **The reader is where the substrate's remaining free lunch sits.** Its parse of features is
  exact by construction wherever the bottom map is injective; its jobs under a rule are reading the
  register off the surface, sharpening block reads at collisions, and noticing when a feature's
  mass has gone missing — none of which the current reader does.

## Caveats

- The claims are ranks, signs, located mechanisms and floor multiples across fourteen single-seed
  tags; the two results that would become load-bearing under a stronger claim — the two-cluster
  meaning split and the `leaf` pair's early advance — are the ones to seed.
- Every collision number is at `rule_seed 0`, every clean number at `rule_seed 6`; the threshold
  draw leaves two genuinely novel registers of five held out; the Q3 E′ pairs differ in two bits
  (the spelling switch and the re-derived floor), attributed by the in-arm `dsil_sp` vs `dsil_ft`
  identity and the offline replay.
- Cross-rung meaning in Q2 confounds curriculum with corpus (each rung has its own `build_shared`)
  and is not monotone; the identifiability and transfer readouts never route through the value
  head and are the claims there.
- `fit_shared` moved only the harvest's parse; the execution-side reader (the DP through
  `generator.block_logits`) is untouched, and that is where `given_rule`'s Q1 readback cost lived.

## Runs on disk

| tag | what | GPU-h |
|---|---|---|
| `if_gf` | G-F: in-process fork-vs-`tutti.py` replay, 0.000e+00; gate Q-11 | 0.14 |
| `if_smoke` | the ruled `--quick` smoke, 5 arms; `read_acc` 1.000 at every register | 0.35 |
| `if_q1` | Q1: five arms on the E_R8 world, collision draw | 2.37 |
| `if_q1b_sh` | `fit_shared` on the Q1 world — bit-identical to `if_q1/fit_rule` | 0.73 |
| `if_q1b_cf` | the five arms on the collision-free draw | 2.32 |
| `if_q1b_a` | the GF(2)-affine contrast, five heads | 2.30 |
| `if_q1c_pr` | the mechanism probe: `canon` + `given_rule`, setup on-rule | 1.00 |
| `if_q1c_yk` | the five arms on one clock, setup on-rule | 3.09 |
| `if_q2_c1`, `if_q2_c2` | the curriculum rungs, practised {3} and {0, 7} | 1.36 + 1.39 |
| `if_q3_f2` | F2 on the yoked clock, per-slot record | 1.97 |
| `if_q3_e`, `if_q3_e_sp` | the E′ pacer seat under the mirror, off arms then on arms at the re-derived floor | 1.90 + 0.95 |
| smokes (`if_sm_*`) | one per new mechanism | ≈ 0.9 |

Volume `rhm-scaling-data:/data/rhm_practice_inflection/<tag>/`; fetched copies and reductions
under `figures/<tag>/` and `figures/<tag>_reduction.txt`. Every flag per tag is in
[`FILES.md`](FILES.md).

## Reproduce

```bash
cd experiments/          # MODAL_PROFILE=chromatic; CPU steps need no profile

PYTHONPATH=. python3 rhm/practice/inflection/phase0_inflection.py                       # Q0, ~30 s
PYTHONPATH=. python3 -c "from rhm.practice.inflection import inflection as I; I.gates_cpu()"
python3 rhm/practice/inflection/launch_detached.py --fn fidelity_smoke --tag if_gf     # G-F
# the tags, each with its full flag line in FILES.md §Runs / §Q1 … §Q3; e.g. the yoked five arms:
python3 rhm/practice/inflection/launch_detached.py --fn inflection_run --tag if_q1c_yk \
    --rule E_R8 --rule-seed 6 --practiced 0,3,7 --setup-render rule --policy schedule \
    --arms "canon_s,given_rule_s,leaf_s,fit_rule_s,fit_index_s" --eras "1:25:40,2:12:45,3:6:55" ...
python3 rhm/practice/inflection/analyze_inflection.py --tag if_q1c_yk --gf-tag if_gf --fetch
```

## Next steps (queued in `QUEUE.md`[^private], not started)

A renderer paced by the meter (fitted from the beam's own metered productions) so the spelling
step is live at the pacer's timescale · a scale-free quiet statistic, or a per-series floor, for
δ-silence as advance owner · `intonation`'s gate seat with `e` including spelling on the `leaf`
pair · `decode` (the register read off the surface) · the executor-side `fit_shared` or a
missing-mass reader on the collision draw · the GF(2) rerun holding out a discriminating register ·
seeds on the two-cluster split and the `leaf` pair's early advance.

## Files

| file | purpose |
|---|---|
| `inflection.py` | the substrate: `tutti.py` forked; the rule table, the context side channel, the Renderer, the graded grader, `--setup-render`, the schedule mode, the fitted heads, the F2 record, `perf_e_spell`; entrypoints `preflight`, `preflight_q3`, `fidelity_smoke`, `inflection_run`, `gates_cpu` |
| `analyze_inflection.py` | the reduction: `analyze_tutti.py`'s sections plus §1I–§6I |
| `phase0_inflection.py`, `phase0.json` | Q0 (CPU) |
| `launch_detached.py` | the session-isolated launcher |
| `sizing/` | the Q0 fact sheet |
| `FILES.md`, `DESIGN.md` | the machinery record and the decisions, round by round |

[^private]: Not mirrored: this link points to a document in the private lab repo (the roadmap, the queue, an unrun spec, reading notes, or a conversation). See the top-level README for what is held back and why.
