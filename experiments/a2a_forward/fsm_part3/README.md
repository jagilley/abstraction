# Forward self-models, part 3 — does privileged access extend past the current residual? (2026-09-13/14)

**Up**: [`../README.md`](../README.md) (the forward-self-model program) · **Forks**:
[`../confabulation/`](../confabulation/README.md) (paper 2's language battery) and
[`../../rhm/confabulation/`](../../rhm/confabulation/README.md) (paper 2's grammar battery) ·
**Paper**: `papers/forward_self_models_paper2.md`[^private]
· **Files**: [`FILES.md`](FILES.md) · **Children** (code, `DESIGN.md`, `FILES.md`, `figures/`
each; no separate READMEs): [`authorship/`](authorship/FILES.md) ·
[`dispositional_language/`](dispositional_language/FILES.md) ·
[`dispositional_grammar/`](dispositional_grammar/FILES.md)
**Status**: done. One orchestrating conversation, three implementer agents run in parallel;
interpretation discussed with Jasper 2026-09-14. ≈16 GPU-h in total (authorship 1.4,
language 5.1, grammar 9.5). Every number below is from the children's `figures/*_reduction.txt`.
**Ledger**: `ROADMAP_PROGRESS.md`[^private] (2026-09-13, 2026-09-14).

## One-line arc

Paper 2 trained a report head to name the part of a transformer's *current* computation that
no learned self-theory anticipated, and showed no outside observer matches it. This node points
the same head, on the same harnesses, at three other things a system might know about itself.
**Authorship** ("did I write this token?") is public and *more* legible from outside than from
inside: an observer holding the model's likelihoods reads it at 0.76 while the model reading its
own stream manages 0.53. **Behavioral dynamics** ("will my answer here improve, when did it?")
are public in every cut on both substrates. **The residual's past** ("how long ago did my
computation here settle?") and, on the grammar, its future, are private — but only where the
computation is deep and behaviorally silent: on the grammar the deep hierarchy levels carry
+0.10 to +0.29 over every observer on the class readout while the shallow levels, half of all
positions, carry nothing there, so the pooled class number hides it; on the rank readout every
level is positive and the advantage rises with depth to +0.34.

## The question

Paper 2's dissociation is between *implementation* facts, which carry a first-person advantage,
and *behavioral* facts such as own correctness and own entropy, which do not because they are
functions of the input–output map. Jasper's question (2026-09-13) was whether the trained-head
design had been taken to its conclusion: the report head reads an *occurrent* fact at one
position at one training snapshot, and the family it had never been pointed at is
*dispositional* facts about the system's own dynamics — how its computation is changing, has
changed, and will change. Those are still implementation facts, so the privilege might survive;
they are directional rather than scalar, which matters because every scalar collapse of the
residual in this program has nulled; and they are slow, so the head has to be trained across a
training trajectory rather than at one snapshot. Authorship was added as a sharp test of the
taxonomy: a transformer re-reading its own output runs exactly the computation it runs on
anyone else's, so the construction predicts authorship is an input-determined fact that reduces
to likelihood.

The three children are the same battery pointed three ways. Nothing here touches the practice
stack; the question is asked on its own terms so a clean result can later be ported.

## The construction

**Shared with paper 2.** The main model M, its data, the fresh-instrument forward models with
the two junk-residual guards (`ens_cos` across instrument seeds; residual structure against
syntax or the grammar's latents), the report head on M's late stream, the capacity-swept
observer ladder (`O_input` tokens only; `O_io` tokens plus M's output distribution; half-data
control), the activation-access observer `O_act`, the confabulator, and the *advantage*
statistic, self-report minus the best third party. Both harnesses were re-run at their final
checkpoints and reproduce paper 2's rows to three decimals (next section).

**New for the dispositional battery** (language: [`dispositional_language/DESIGN.md`](dispositional_language/DESIGN.md);
grammar: [`dispositional_grammar/DESIGN.md`](dispositional_grammar/DESIGN.md)):

- A **checkpointed wake**: 16 checkpoints along one training run, log-spaced on language (steps
  300–10000) and geometric on the grammar (400–20000, ratio 1.295) so that a backward window
  spans the same step ratio as a forward one. At every checkpoint, fresh instruments at two
  capacities with the guards, on one **fixed report set** of held-out sequences, so targets are
  defined per position across time.
- **A 2×2 of targets** per position `p`, checkpoint `c`, horizon `k`: {implementation,
  behavior} × {prospective, retrospective}. Implementation-prospective is whether the residual
  at `p` reorganizes by `c+k`, `1 − cos(r_c, r_{c+k})` (language), or the direction class of
  `r_{c+k}` and the reorganization amount (grammar); implementation-retrospective is the mirror,
  integrated past reorganization or the class of `r_{c−k}`. Behavior-prospective is the loss
  drop `ℓ_c − ℓ_{c+k}`; behavior-retrospective its mirror. Categorical labels are quantile
  classes **within checkpoint** (and within level on the grammar), so a head pooled across
  checkpoints cannot score by memorizing a per-checkpoint constant. Continuous rank-loss
  variants scored by Spearman ρ ride alongside.
- **The instrument subtlety.** `r_c` and `r_{c+k}` come from different fresh instruments, so a
  naive reorganization measure mixes M's change with instrument variation. Three readings: raw;
  *excess over the per-position no-change floor* (what two instruments at the *same*
  checkpoint disagree by); and *fixed instrument* (the FM trained at `c` applied at both ends,
  floor exactly 0). The grammar's headline reorganization rows are the fixed reading, with the
  other two run through the same cells as a follow-up; the language rows carry all three.
- **The report head** reads M's late stream at `c`, is trained pooled across checkpoints, and is
  scored on held-out sequences at a **held-out block of checkpoints**, so what it learns is a
  rule about states rather than per-checkpoint constants. The strict column below is that one.
- **Two new observers.** Every observer gets a checkpoint-index embedding, without which
  tokens-only observers cannot tell which snapshot they are looking at. `O_hist`, a *historian*,
  additionally sees M's output summaries at the previous checkpoints. For paper 2's occurrent
  target a historian would have been an illegitimate comparator; for a dispositional target it
  is a legitimate third party, and on the language node the conservative advantage is against
  `max(O_io, O_hist)`. `fracpub` = (best third party − chance)/(self − chance) is the fraction of
  the self-report's above-chance score reachable from outside; 1.0 means fully public.
- **An autocorrelation gate before any head is trained.** `committee_head` found a head cannot
  learn a per-round target that is white noise across rounds. Each target's persistence across
  checkpoints is measured first; the grammar's gate carries a correction, because adjacent
  difference windows share an endpoint at lag `k` and read negative whatever the signal, so the
  rule reads disjoint windows. The behavioral targets at the one-checkpoint horizon are close to
  white noise on both substrates, which is why the grammar ran a second horizon (`k=3`) and the
  language node reports `k=4` and smoothed variants alongside `k=1`.
- **Reading within level on the grammar.** Every position's hierarchy level is known and levels
  are learned bottom-up, so age of acquisition is nearly the same fact as level, and level is an
  input fact any observer reads off the token index. All grammar rows are therefore quantile-
  binned and reported within level. Language has no level labels; the stratified re-fit uses
  public proxies (the syntactic taxonomy, output-entropy quartile, position) and the occurrent
  residual magnitude as a private comparison stratifier.

**The authorship node** ([`authorship/DESIGN.md`](authorship/DESIGN.md)): M's own temperature-1
samples are spliced with corpus text in a `mix` set (a 24-token corpus prefix, then alternating
spans with lengths drawn from one shared distribution, starting class randomized, so neither
position, span length nor boundary is a cue) and a `whole` set (the entire remainder either
sampled from M or the document's true continuation, which removes the splice). The target is
per-position "M wrote this". Alongside paper 2's contestants: a likelihood-only observer
`O_lik` (M's six per-token likelihood features, no tokens), an `O_io` handed those features as
well, the parent's `O_io` unchanged (`O_io_nolik`), and a sequence-level self-reporter, since
authorship is a span-level fact and the observers are sequence models. The IMPL row runs on the
same sequences and scored positions as the positive control.

## Harness validation

Both forks are bit-faithful. Final-checkpoint validation loss and paper 2's occurrent rows:

| substrate / arm | val (this run) | published | IMPL adv (published) | BEHAV | ENT | WORLD |
|---|---|---|---|---|---|---|
| language OL (all three nodes) | 5.3204 | 5.320 | **+0.258** (+0.272); IMPL_COS +0.357 (+0.318) | −0.031 (−0.067) | −0.286 (−0.287) | −0.010 (−0.009) |
| grammar CL `ntp_aux_cl` | 2.3829 | 2.3829 | **+0.096** (+0.096) | −0.001 (−0.001) | −0.035 (−0.035) | −0.038 (−0.038) |
| grammar OL `ntp_aux` | 1.5447 | 1.5447 | **+0.065** (≈+0.06) | +0.002 | +0.007 | +0.011 |

Guards along the trajectories: the first three grammar checkpoints (steps 400–674) are in the
junk regime (`ens_cos` ≤ 0.81, d6 η² ≤ 0.012) and are excluded from the `k=3` targets; from step
875 the CL arm runs `ens_cos` 0.847 → 0.938 → 0.902 with final d6 η² 0.323 (published 0.322)
and the OL arm 0.832 → 0.885 → 0.882 with 0.295 (published 0.295). On language `ens_cos` rises
monotonically, 0.61 → 0.86 at the default instrument and 0.70 → 0.91 at the smaller one; the
default instrument is inside paper 2's band only from step ≈2500 and the held-out block
straddles the edge (0.76–0.78), while the smaller instrument is in band from step 605 and gives
the same answers. The forward model's cosine is non-monotonic on the grammar CL arm, bottoming at
0.713 mid-run: M is hardest to self-predict in the middle of learning, not at the start.

## Finding 1 — authorship is public, and more legible from outside than from inside

`mix` set, temperature 1, held-out positions (chance 0.500):

| contestant | accuracy |
|---|---|
| self-report, per-position head on the late stream | 0.533 |
| self-report, sequence model on the same input | 0.523 |
| `O_input` (tokens) | 0.508 |
| `O_io_nolik` (tokens + output distribution, the parent's observer) | 0.502 |
| `O_lik` (likelihood features only, no tokens) | **0.757** |
| `O_io` (tokens + distribution + likelihood features) | 0.718 |
| `O_act` (given M's early activations) | 0.522 |
| **advantage vs best third party** | **−0.224** |

On the same sequences and the same scored positions the residual target reads self 0.461 vs
observer 0.191, **+0.270** (published +0.272), with `ens_cos` 0.861 and η²_norm 0.026 in the
published range. The whole-continuation set gives self 0.606 vs `O_lik` 0.938 (tokens alone
reach 0.608 there: 104 tokens of temperature-1 text are recognizable by their degeneracy, while
8–32-token spans are not); temperature 0.7 gives 0.634 vs 0.931. The parent's observer without
the likelihood join sits at chance, so the whole third-party score *is* the likelihood channel,
and `O_act` sits at the self-report, so the fact is not merely expensive to reconstruct from
outside: it is absent from M's activations and present in M's logits.

The channel ablations are the mirror image of the residual's:

| target | full | `shuffle_r` | `shuffle_p` | confabulator |
|---|---|---|---|---|
| authorship | 0.533 | 0.524 | **0.502** | 0.532 (margin +0.001) |
| IMPL | 0.461 | **0.186** | 0.345 | 0.309 (margin +0.152) |

Whatever trace of authorship M's late stream carries lives entirely in the range of the
self-model, the part a theory of M could have written. Phase 2 of this node, an efference copy
bolted onto the frozen model, was designed and deliberately not built: a private input makes a
private fact by fiat, and the version that would show something trains M with its own
generations in the stream (queued in `QUEUE.md`[^private]).

## Finding 2 — behavioral dynamics are public in every cut

"Will my answer here improve" and "how long ago did it" carry no first-person advantage
anywhere: pooled and within level on the grammar at both horizons and on both arms (all cells
within ±0.03), and in every stratum of every stratifier on language. Where a comparator wins it
is the historian, which holds both checkpoints' outputs and can nearly compute the loss change
(grammar OL retrospective: −0.043 pooled, −0.084 at the shallowest level; language: `fracpub`
1.01–1.09). At the one-checkpoint horizon these targets are close to white noise across
checkpoints, so their null there is uninformative; at the three-checkpoint horizon on the grammar
and at `k=4` or smoothed on language they are persistent, and the null is the same. This is paper
2's ENT row extended along the time axis.

## Finding 3 — the residual's past is private where the computation is deep and silent

**Grammar, open loop, within hierarchy level.** Advantage against `max(O_io, O_hist)`; the
`O_act` comparison alongside. Held-out sequences; the strict held-out-checkpoint column is at
least as large at every deep level (larger by up to +0.08) and is in
`figures/ol_ceilings_reduction.txt`.

| level (positions per sequence) | L0 (32) | L1 (16) | L2 (8) | L3 (4) | L4 (2) | L5 (1) |
|---|---|---|---|---|---|---|
| how long ago did my computation here settle, `k=1` | +0.006 | +0.016 | −0.001 | **+0.087** | **+0.230** | **+0.241** |
| the same, `k=3` | −0.015 | +0.004 | −0.016 | **+0.098** | **+0.181** | **+0.194** |
| will my computation here reorganize, `k=1` | +0.008 | +0.025 | +0.033 | **+0.136** | **+0.270** | **+0.285** |
| the same, `k=3` | −0.025 | +0.005 | +0.019 | **+0.105** | **+0.290** | **+0.279** |
| reorganization amount ahead (fixed instrument), `k=3` | +0.033 | +0.048 | +0.024 | +0.085 | +0.117 | +0.083 |
| reorganization amount behind (fixed instrument), `k=3` | +0.031 | +0.061 | +0.043 | +0.119 | +0.144 | +0.090 |
| will my loss here drop, `k=3` | −0.030 | +0.005 | +0.014 | +0.012 | +0.000 | +0.001 |
| how long ago did my loss here drop, `k=3` | −0.012 | +0.007 | +0.021 | +0.012 | −0.004 | +0.006 |
| what am I computing now (anchor), `k=3` | +0.160 | +0.137 | +0.135 | +0.240 | +0.322 | +0.341 |
| self-report vs `O_act`, settled-how-long-ago, `k=1` | 0.495 / 0.491 | 0.501 / 0.490 | 0.479 / 0.499 | 0.502 / 0.470 | **0.662 / 0.581** | **0.666 / 0.503** |

The self-report rises to 0.58–0.70 at the two deepest levels while every observer stays near
0.39–0.43. `O_act` is a genuine ceiling at L0–L2, where it beats the self-report, and stops being
one at L3–L5, where the self-report leads it by +0.07 to +0.19: an observer handed M's own early
activations cannot, at its budget, reconstruct what M reports about its deep-level dynamics.
Per-level test n at the strict column runs 57,600 at L0 down to 1,800 at L5.

The pooled numbers are +0.011 (reorganize) and +0.028 (settled) at `k=3`, because L0 holds 32 of
63 positions and carries almost nothing on this readout; the pool is a level-weighted average the
deep levels cannot move.

**The rank readout says the shape is a gradient, not a threshold.** Re-running the four
reorganization-amount and loss-change rows as rank-loss models scored by Spearman ρ within level
(`figures/ol_followups_reduction.txt`):

| level | L0 | L1 | L2 | L3 | L4 | L5 |
|---|---|---|---|---|---|---|
| reorganization behind, `k=3`, advantage vs best observer | +0.123 | +0.195 | +0.219 | +0.271 | +0.338 | +0.266 |
| reorganization ahead, `k=3` | +0.132 | +0.189 | +0.118 | +0.123 | +0.272 | +0.294 |
| reorganization ahead, `k=1` | +0.066 | +0.093 | +0.092 | +0.119 | +0.209 | +0.235 |
| loss drop ahead, `k=3` | +0.035 | +0.080 | +0.075 | +0.030 | +0.006 | −0.037 |
| loss drop behind, `k=3` | −0.042 | +0.062 | +0.044 | +0.008 | −0.009 | −0.006 |

On ranks the implementation advantage is positive at every level and rises with depth, driven by
the observers collapsing at the deep levels (on the `k=1` prospective row `O_io` falls 0.39 → 0.09
from L0 to L5 while the self-report falls 0.47 → 0.39). The four-way quartile class inherited from
paper 2, where the target was a k-means class and binning was natural, is a lossy readout of a
quantity whose information is in the ordering, and it understates every continuous dispositional
row here. The behavioral rows stay near zero on ranks too, and at L0 the self-report loses to the
observers on both directions and both horizons (−0.04 to −0.10). The per-level loss trajectory on this arm, first to final checkpoint, is L0 −0.557, L1
−0.468, L2 −0.182, L3 −0.085, L4 −0.073, L5 −0.070: the levels carrying all of the dispositional
advantage are the ones whose behavior barely moves.

**Language, stratified.** Pooled, the strict-column advantage against `max(O_io, O_hist)` is
+0.026 categorical and +0.071 on the rank readout for "how long ago did my computation here
settle", with `fracpub` 0.72–0.79 against 0.20 for the occurrent target. Stratified by the
occurrent residual magnitude (a private stratifier, ~12,200 slots per quartile):

| quartile of \|r_c(p)\| | q1 | q2 | q3 | q4 |
|---|---|---|---|---|
| settled how long ago, advantage | −0.002 | +0.012 | +0.027 | **+0.067** |
| will reorganize (raw), advantage | −0.018 | +0.009 | +0.021 | +0.032 |
| will reorganize (excess / fixed) | −0.005 / −0.018 | −0.004 / −0.009 | +0.002 / +0.002 | +0.008 / −0.007 |
| both behavioral rows | ≤ +0.008 | | | |

The public syntactic taxonomy says the same thing with less power: +0.024 in the 90% bulk,
+0.058 after openers, +0.154 before closing delimiters (648 positions, 216 slots, where every
predictor sits below the within-stratum majority constant). Position bins show no structure;
entropy quartile shows a monotone prospective trend (+0.010 → +0.034) and a flat retrospective
one. Closing delimiters are where paper 2's residual signature is largest, so the retrospective
advantage concentrates where the occurrent residual is large, which on this substrate means
long-range context integration.

## Finding 4 — the residual's future: private on the grammar, instrument variation on language

On language the prospective advantage over the historian lives on the raw reading only:

| reading of "my computation reorganized" | vs `O_io` | vs `max(O_io, O_hist)` | `fracpub` |
|---|---|---|---|
| raw, default instrument | +0.029 | +0.020 | 0.70 |
| excess over the per-position no-change floor | +0.017 | **+0.000** | 1.00 |
| fixed instrument | +0.019 | **+0.001** | 0.99 |
| raw, smaller instrument | +0.046 | +0.029 | 0.69 |
| excess, smaller instrument | +0.024 | +0.006 | 0.94 |

The per-position no-change floor on language is large, 0.42 falling to 0.14 along training and
0.22–0.24 at the held-out block, so most of what looked like "my computation will reorganize
here" was two instruments disagreeing. The retrospective row survives both corrections.

On the grammar all three readings were run through the same cells on the open-loop arm, and the
prospective advantage survives every one of them: pooled +0.018 to +0.050 across the twelve
reading × direction × horizon cells, and within level the three readings agree to within about
0.03, rising from about +0.01 at L0 to +0.07 to +0.13 at L4–L5 (`k=3` prospective: fresh +0.107 /
excess +0.100 / fixed +0.117 at L4). The floor subtraction is not a relabeling: fresh and excess
targets rank-correlate 0.84–0.90 within cells. The fresh-instrument floor falls with level, 0.18
at L0 to 0.05 at L5 (a quarter to a third of the fresh reading at `k=1`, a tenth to a quarter at
`k=3`), so instrument agreement is highest exactly where the advantage lives, the opposite of the
junk signature, and there was never enough instrument variation at the deep levels for the
correction to remove. The two substrates therefore agree in sign on every row and genuinely
disagree on whether a forward-looking advantage survives correction. The language floor is a
larger share of its fresh reading (0.22–0.24 at the held-out block against a reading of the same
order), its deep strata are about 1% of positions against half on the grammar, and its residual
is diffuse; which of these carries the difference is open.

## The closed loop shrinks the dispositional rows

Paper 2 found the loop roughly doubled the occurrent advantage on the grammar. Here the CL arm
keeps its occurrent anchor (+0.082 to +0.091 pooled, +0.137 on held-out checkpoints) and shrinks
every dispositional row to a few hundredths pooled and at most +0.08 within level, against the OL
arm's +0.10 to +0.29. The CL model's *standalone* loss rises over training at five of six levels
(L0 +0.63) as it comes to depend on the injection the battery evaluates without, so its
behavioral targets are partly "my dependence on the injection is growing" and the OL arm is the
one to read for this question. That accounts for the behavioral rows; whether it accounts for
the implementation rows is recorded as open, not explained.

## Reading across the three

The three nodes order a model's self-facts by how legible they are from outside. Behavioral
facts — authorship here, calibration and entropy in paper 2 — are more legible from outside than
from inside, because they are functions of the input–output map that the map's implementer does
not compute for itself. Dispositional implementation facts are private in proportion to how deep and
behaviorally silent the computation is: on the class readout the shallow levels carry nothing
and the deep ones +0.10 to +0.29, on the rank readout every level carries something and the deep
ones carry most, so a pooled class measurement can read as "mostly public" (language `fracpub`
0.7–0.8) while a stratified or rank one reads as strongly private (grammar). The occurrent
residual is mostly private everywhere (`fracpub` 0.20).

Two readings are on the record without being claimed. First, the dispositional advantage sits
precisely where learning is behaviorally invisible, which is the regime in which a behavioral
ledger cannot see that anything is happening and a self-model's residual can; that is the shape
the roadmap's E2 object ("consolidation state as an implementation fact") and the absorption
argument both asked for, measured here on a plain training run. Second, the retrospective
direction is sturdier than the prospective one on both substrates (it survives the instrument
corrections on language and generalizes across held-out checkpoints on the grammar, +0.049 to
+0.056, where the prospective rows' held-out-checkpoint numbers are +0.032 and −0.003): having
changed may generalize across training stages in a way that going-to-change does not. Neither
reading is pre-registered and neither should be cited as established.

## Methods exports

- **Difference-window autocorrelation is biased at lag `k`**: windows `[c, c+k]` and
  `[c+k, c+2k]` share an endpoint with opposite sign, so lag-`k` autocorrelation reads negative
  whatever the signal. Read disjoint windows (lag `k+1`) or lag 1 at `k ≥ 2`.
- **Within-checkpoint (and within-level) quantile classes** are what make a head pooled across
  checkpoints report a rule rather than the calendar; observers need the checkpoint index.
- **A historian observer is a legitimate third party for dispositional targets** and should be
  the comparator; it is not for occurrent ones.
- **Three readings of reorganization** (raw / excess over the same-checkpoint floor / fixed
  instrument) are required before a prospective implementation advantage is read; on language
  the raw reading was mostly instrument variation.
- **Pooling hides level structure.** On a substrate where half the positions carry no
  dispositional privilege, the pooled advantage is a few hundredths while the deep levels read
  +0.29. Stratification is load-bearing, not hygiene.
- **A re-fit reproduction check** (the language stratified pass re-trained all 36 predictor ×
  target pairs and matched the stored pooled scores to 0.0000) is what licenses stratified
  numbers from a run that did not save per-position predictions.
- PNG figures under `experiments/` are gitignored repo-wide; only the `*_reduction.txt` files are
  tracked. The plots exist on disk and on the Modal volumes.

## What this does and does not establish

It establishes, on both substrates, that behavioral self-facts stay public along the time axis
and that a first-person advantage about the residual's own history exists and is largest where
the occurrent residual is large and the behavior is not moving. It establishes that authorship is
not an implementation fact for a plain transformer. It does not establish a forward-looking
privilege on language, and it leaves the flat language ladders undiagnosed: `O_act` does not
beat `O_io` on any language dispositional row, so a real access gap and "only the report site
carries this" are not separated there. Effect signs are triangulated across substrates,
instruments, readouts, horizons and evaluation strictness; the language magnitudes are small
(+0.02 to +0.03 categorical) and the grammar's are large only within level.

## Follow-ups

- **Re-read the language node on ranks and by the floor's share.** The grammar's rank readout
  moved every dispositional row by a factor of three to five over the class readout; the language
  node's rank rows exist pooled (+0.065 / +0.071) but not stratified, and its floor-by-stratum
  share is the number that would say whether the prospective disagreement is instrument variation
  or substrate.
- **Diagnose the language ladder**: a component control on the dispositional targets, or an
  observer given `a_j` rather than `a_0`.
- **The silent-part question**: relate the per-level dispositional advantage to the per-level
  learning rate directly (does the advantage track the *residual's* motion where the *loss* does
  not move), and ask whether what is private is the consolidation state E2 wanted.
- **Authorship phase 2**: train M with its own generations in the stream, tagged at generation
  time by something absent from the tokens (`QUEUE.md`[^private]).

## Reproduce

Each child's `FILES.md` carries its full command set (smoke, main, reductions). Headline runs,
from `experiments/`:

```bash
modal run --detach a2a_forward/fsm_part3/authorship/authorship.py::authorship_test --tag main
modal run --detach a2a_forward/fsm_part3/dispositional_language/dispositional.py::dispositional_test --tag main
modal run --detach a2a_forward/fsm_part3/dispositional_language/stratified.py::stratified_analyze --tag main
modal run --detach a2a_forward/fsm_part3/dispositional_grammar/dispositional.py::dispositional_run --tag main --cond ntp_aux_cl --horizons "1,3"
modal run --detach a2a_forward/fsm_part3/dispositional_grammar/dispositional.py::dispositional_run --tag ol --cond ntp_aux --horizons "1,3"
```

Results: `language-reduction-data:/data/a2a_forward/confabulation/{authorship,dispositional}/`
and `rhm-scaling-data:/rhm_confabulation/v16_s2_L6_m4_distinct/dispositional/{main,ol}/` (volume
paths predate the move and are unchanged).

[^private]: Not mirrored: this link points to a document in the private lab repo (the roadmap, the queue, an unrun spec, reading notes, or a conversation). See the top-level README for what is held back and why.
