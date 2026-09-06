# merge — the merge op, invariance-by-enumeration, and whether free data is free coverage

**Up**: [`../README.md`](../README.md) (practice arc) · **Spec**: [`SPEC.md`](SPEC.md) ·
**Files**: [`FILES.md`](FILES.md) (machinery, arms, gates, calibration records, caveat lists —
read its caveats before trusting any single number) · **Parent idea**:
[`recurrence_manufactures_confounds`](../../../../ideas/recurrence_manufactures_confounds.md)
(§5 the merge op's constraints; §8 the instrument list) and the 2026-08-17 caveat block in
[`meta_learning_under_metered_data`](../../../../ideas/meta_learning_under_metered_data.md)
(reframes i–iii). **Sibling**: [`../fourwall/`](../fourwall/README.md) — an independently-built
narrowed slice of the same spec (index-news and the whittling ops), machinery unshared.
**Runs**: `mg_s0` (round 1) and `sp_s0` + `sp_s0c` (round 2), 2026-08-17; rank
orderings and signs are the reported quantities.

## Round 1 (`mg_s0`) — enumeration vs merge, and the unmetered monolith

**The question.** §18 of
[`practice_manufactures_its_own_credit`](../../../../ideas/practice_manufactures_its_own_credit.md)
calls pretraining "the correct degenerate solution when the meter is off." Is
invariance-by-enumeration (an entry per context cell) the same object as invariance-by-merge (a
quotiented index)? The world: 6 OU-drifting venues × 3 level-2 nodes = 18 keys, a manufactured
venue↔latent confound (σ=3.0, 0.35 nats, calibrated by sweep), paced rotation as manufactured
decorrelation, one node and one venue held out (the fifth wall — certified by gate M-5 as a pure
*index* transform). 14 arms spanning the index policies (`track` / `merge` / `global`), the
rotation-rate axis, and three unmetered monolith variants; `dense_glob` (unmetered, free i.i.d.
variation, one global vocabulary, no index event) is the affordance-matched monolith after the
spec-literal `dense` failed parity for a known structural reason (ratchet: base moves are priced
out of level-2 damage). All six world gates passed; arms and flags: [`FILES.md`](FILES.md).

**Findings.**

1. **Parity**: only the affordance-matched monolith reaches the gate, and only to within noise
   (E2 0.269 vs best metered 0.238, miss 0.031 against a ±0.034 floor), saturating early while
   the merge arms keep descending. `dense`/`dense_wd` fail by 0.28 (structural); weight decay
   changed nothing (0.523 → 0.524 — a weak null on a floor-bound arm).
2. **The fifth wall bites hard exactly once: enumeration without rotation.** `track:rotate=0` is
   rank 2 of 14 at home and **rank 14 — worse than `never_base` — at the held-out key**, its wall
   error rising (0.31 → 0.58) as home error falls; corrected transfer cost +0.228. The cache-miss
   range is the mechanism in one line: its index holds a cell that would have scored 0.156 and it
   executes 0.505 — the content exists, the address is wrong. Merge in the same world recovers
   0.234 of that. With paced rotation on, track ≈ merge on task error: **rotation and merge are
   substitutes for the quotient at the task level.**
3. **They are not substitutes for the next level.** At every rotation rate — including where task
   error shows nothing — `track`'s operative cell supports |T3| = 12 where `merge` supports 67
   and `dense_glob` 72, against `track`'s own *union* of 68: **the content is collectively there;
   the index withholds it**, and the next level is unrepresentable from any address the library
   can actually be queried at. The gap exceeds the offline structural prediction (47.7 vs 93,
   measured pre-run with no learner — see [`FILES.md`](FILES.md)). This is §16's currency claim
   (deep value invisible to within-level signals) measured on the index axis.
4. **The merge op behaves as §5 specified.** Index-only (gate M-6), one representative per merged
   cell; evidence-pooling vs representative-only is a wash. It **refuses genuinely distinguishable
   cells** — off-diagonal forced-transfer success 0.818 without rotation (merge correctly stalls
   at 4 cells) vs 0.930 with rotation (collapse 18 → 1 at 8× less storage). **Paced rotation is
   what renders cells indistinguishable; manufactured decorrelation is the licensing condition
   for merging**, measured as mechanism.
5. **The scaffold's early value did not reproduce** — the no-key `global` arm beats the keyed
   arms in the first 10 cycles (0.177 vs 0.248–0.265). This is the risk the parent doc's §9
   pre-registered (a selector that keys on the true latent cheaply enough), with one missing
   matched cell (`global:rotate=0`) noted in the caveats.

## Round 2 (`sp_s0` + `sp_s0c`) — is free data free coverage?

**The question.** Round 1's monolith was handed free i.i.d. variation over the *whole* space —
exactly the condition self-play lacks: self-play is training where the question model and the
answer model are the same object, so the demand distribution is the policy's own footprint. Does
unlimited data with a self-generated question distribution leave holes the learner cannot see?

**A calibration killed the first design** (recorded in full in [`FILES.md`](FILES.md), because it
is the round's main methodological content): restricting support by *venue* cannot make a hole
here — 19 reachable level-2 entries, every venue demanding 9–15 of them; a collapsed arm at
visitation perplexity 1.46/6 still covered 96.4% of every venue's demand. Narrow volume, no hole.
The support axis moved onto **the latent itself** under uniform world demand: the only difference
between arms is which freely-offered instances each policy chooses to practice on. The
decomposition: `dense_visit` (rich-get-richer on visitation), `dense_selfplay` (the footprint
filtered through *competence*, β=1) and `dense_sp_hard` (β=2), against `dense_full` (exogenous
full support) and exogenous-narrow controls. Exams under three weightings — own footprint /
world i.i.d. / flat — with per-entry excess netted against `given` (which closes round 1's
difficulty-matching caveat). Exclusion/inclusion gates (X-1/X-2) certify the holes and the
coverage from realized supports.

**Findings.**

1. **The β=2 arm is the pathology, fully formed**: rank 1 of 8 on its own footprint (0.113 —
   beating the DGP's own table by 0.086), second-worst on flat coverage (0.324, and *rising*
   across the run while the metered arm's falls), the only positive tail excess (+0.055), a
   vocabulary 6 entries thinner, and 18% less of the next level representable (|T3| 61 vs 74).
   Its support starves the five rarest entries 30–70×, several to literal zero.
2. **The controls attribute the damage to the coupling dynamics, not to narrowness.** A yoked
   exogenous arm fixed to the *same placement* (sp_hard's realized support collapses 88.7% onto
   one head entry) and *strictly narrower* (2 entries ever practiced vs 16) reproduces only
   **22%** of the flat loss, none of the tail excess (−0.005), and about half the minability
   thinning (14/66). The size-matched random control reproduces 71% of the flat loss but as a
   mirror-image lesion (head excess +0.142 — it starves the head instead). **Holding narrowness
   and placement fixed and removing only the competence coupling removes most of the effect** —
   and the coupled arm practiced 8× more distinct entries than the yoked control and still lost,
   so the harm rides on the collapse *trajectory*, not the endpoint support.
3. **The onset is a cliff.** β=1 self-play is a clean null (perplexity narrows 12 → 6.2, 8
   entries starved below 0.2% of draws — yet rank 1 on flat coverage and full minability), and
   `dense_visit` ≈ `dense_selfplay` at β=1. At β=1.5 the support perplexity has already halved
   (→ 2.99) but every readout still sits near the null; the collapse in tail excess and
   minability appears **between β=1.5 and β=2**. Support concentration alone does not predict
   the damage.
4. **One correction recorded against our own earlier reading**: `own − world` self-grading
   inflation is fully reproduced by exogenous narrowness (−0.135 vs −0.146) — a narrowness
   artifact, not a self-play signature. The claims rest on the flat exam (~8σ for sp_hard vs
   ~2σ for the yoked arm) and minability; the tail-excess magnitudes sit at the instrument's
   stated reading boundary.

## What the two rounds jointly say (held with the usual humility)

In-distribution, the unmetered monolith with exogenous full coverage is fine — §18's clause
survives at the task level, *when someone else asks the questions*. The costs live in the two
currencies task error cannot read: **a fractured index withholds the next level** (round 1,
regardless of rotation rate), and **a competence-coupled question distribution manufactures
holes and forecloses the next level past a sharp coupling threshold** (round 2). Scope worth
stating plainly: this world is small enough that exogenous full coverage was *available*; in
domains where it is not, the coupled regime is the reachable one — an argument, not a
measurement. The belief-level integration belongs to the idea docs, not
this node.

## Reproduce

```bash
cd experiments/            # MODAL_PROFILE=chromatic
modal run rhm/practice/merge/merge.py::selfcheck_remote
python3 rhm/practice/merge/launch_detached.py --fn merge --tag mg_s0 --seed 0 \
    --arms "never_base,given,global,track,track:mine_support=1,merge,merge_pool,dense,dense_glob,dense_wd,track:rotate_every=0,merge:rotate_every=0,track:rotate_every=1,merge:rotate_every=1" \
    --cycles 60 --n-venue 6 --train-nodes "1,2,3" --hold-node 0 --l3-node 1 \
    --sigma 3.0 --kappa 0.15 --rotate-every 4 --rotate-start 4 --demand-levels "0/1" \
    --n-pr 72 --n-rt 96 --dense-mult 4 \
    --merge-every 8 --merge-start 16 --merge-tau 0.95 --merge-probe 32 --merge-min-obs 8 \
    --mine-support 3 --budget 4 --pr-width 16 --g-budget 58 --n-grad 4 --gen-steps 20 \
    --n-ex 192 --n-probe 48 --n-e4 1536 --n-mine-post 1536 --probe-every 6
python3 rhm/practice/merge/analyze_merge.py --tag mg_s0 --fetch --figures

# round 2 (self-play) + its control launch, overlaid
modal run rhm/practice/merge/selfplay.py::selfcheck_sp_remote
python3 rhm/practice/merge/launch_detached.py --fn selfplay --tag sp_s0 --seed 0 \
    --arms "never_base,given,metered_glob,dense_full,dense_visit,dense_selfplay,dense_selfplay:sp_beta=2,dense_narrow" \
    --cycles 60 --dense-mult 4
python3 rhm/practice/merge/launch_detached.py --fn selfplay --tag sp_s0c --seed 0 \
    --arms "dense_narrow:narrow_k=2,dense_narrow:narrow_fixed=3-0,dense_selfplay:sp_beta=1.5" \
    --cycles 60 --dense-mult 4
python3 rhm/practice/merge/analyze_selfplay.py --tag sp_s0 --extra sp_s0c --fetch --figures
```

Volumes: `/data/rhm_practice_merge/mg_s0/`, `/data/rhm_practice_selfplay/{sp_s0,sp_s0c}/` on
`rhm-scaling-data`. Figures: `figures/mg_s0/fig1–fig4` (round 1) and
`figures_sp/sp_s0+sp_s0c/fig1–fig3` (round 2, overlaid). Exact per-arm flags for `sp_s0`/`sp_s0c`
are recorded in each tag's `setup.json`; consult it if the launcher defaults drift.
