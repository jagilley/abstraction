# merge — File Index and Calibration Record

Complete file-by-file reference for this node, plus the pre-run calibration. Findings:
[`README.md`](README.md) (written post-discussion, 2026-08-17). Spec: [`SPEC.md`](SPEC.md).
Parent idea:
[`recurrence_manufactures_confounds`](../../../../ideas/recurrence_manufactures_confounds.md)
(§5 the merge op, §8 the instrument list). Machinery donors:
[`../setlist/FILES.md`](../setlist/FILES.md) (OU demand drift, calibration discipline),
[`../crystallize/FILES.md`](../crystallize/FILES.md) (level-indexed action space, hierarchical
damage, exact-DP oracle), [`../ratchet/FILES.md`](../ratchet/FILES.md) (miner, nested tables,
beam, plant/selector online loop). **Nothing in those nodes is modified.**

## What the round installs

`setlist` drifted *what is asked* while the grammar stood still. This round drifts the same
thing for a different purpose: **to manufacture a confound and then pay it down.**

Every task instance is announced with two free observables:

- a **venue** `w`, each carrying its *own* OU demand state (private root prior, private mixture
  weights at `demand_levels = 0/1`). Under narrow demand a venue concentrates, so `w` is a
  nearly-free name for the level-2 feature the damage destroyed — a latent the agent would
  otherwise have to infer. That correlation is not in the DGP; it is an artefact of the sampling
  policy. **Paced rotation** (one OU event every `rotate_every` cycles, all venues) moves every
  taste, so a per-venue table goes out of fashion and the accumulated per-venue tables converge:
  manufactured decorrelation at a paced rate.
- a **node** `j` — which level-2 node the damage hit. The frame.

The key is `(w, j)`; the index is a partition of the key space; **merge** is the op that
coarsens it. Every other op in the arc acts on an entry under a fixed index.

**Held out from every arm**: one level-2 node (`hold_node = 0`) and one venue
(`w = n_venue`). Those are the *fifth wall* — a transform outside the varied set, where a keyed
library takes a genuine cache miss.

## Code files

| File | Purpose |
|---|---|
| `library.py` | The library and the merge op, kept out of the app so they are auditable and gate-testable with no substrate (`setlist/demand.py`'s discipline). `build_venues` / `rotate_venues` are the per-venue OU ensemble over `demand.new_demand` / `typical_demand` / `demand_step`. `node_hist` is the demand over a level's vocabulary **at one node** (`demand.demand_hist` pools over nodes; here the node is part of the key). `venue_latent_joint` / `mutual_info` are the confound's loudness in nats. `KeyedLibrary` is key → cell → committed table with the three index policies (`track` / `merge` / `global`), `cell_of` returning `None` on a **cache miss**, `fallback_cell` (most evidence) as the miss policy, and `storage` as the never-merge arm's carried cost. `merge` coarsens the index and routes both key streams into one miner; `merge_rep="rep"` discards the loser's counts (the merged cell's content at that instant is exactly the representative's — never a blend), `"pool"` inherits them. `merge_pass` is §5's **evidence route (b)**: the K×K forced-transfer matrix `S[a][b] = success of a's table on b's demand`, graded by the grammar's own possible-set DP, with a symmetric alias test at `tau` and greedy agglomeration. `alias_jaccard` is route (a), the offline audit. `next_level` is the ratchet nesting applied to a lower table |
| `merge.py` | The Modal app. Imports `build_shared`, `beam_moves`, `finetune_generator`, `value_steps`, `fit_width`, `push`, `context_instances` from `ratchet`; `demand.*` from `setlist`; `units.*` from `crystallize`. `build_world` draws every venue state and **every exam set once**, shared by all arms. `group_ms` is the per-key action set (base level-1 moves + the cell's table instantiated at every node — position-independent, so a committed unit cannot be contaminated by the key even in principle). `solve` is one priced solve; `exam` grades a battery cell-by-cell with the cache-miss range. `minability` is readout 3; `post_mine_t2` reads a level-2 vocabulary off a representation that holds no library (the only reading a dense arm has); `union_table` is the generous bound on a keyed arm. `run_arm` is the loop. `gate_world` is the six admissibility gates, offline. `cal_loud` is the offline loudness × rate sweep. Entrypoints: `merge` (main), `cal_loud_remote`, `gate_world_remote`, `selfcheck{,_remote}` |
| `analyze_merge.py` | Reduction. `--fetch` pulls from the volume; the report leads with the gates, then the exam table with rank orderings and the **transfer cost** `E3 − E2` (which nets out how good an arm is in the first place), then next-level minability in both readings, then §8's instruments. `--figures` writes fig1–fig4 |
| `launch_detached.py` | `setlist`'s session-isolated launcher, retargeted; forces `MODAL_PROFILE=chromatic` |

## Arms

| arm | meter | demand | index | what it is |
|---|---|---|---|---|
| `never_base` | metered | venue, rotating | — | the arc's floor: base level-1 moves forever |
| `given` | metered | venue, rotating | — | the DGP's own level-2 table, demand-invariant reference |
| `global` | metered | venue, rotating | one cell from birth | **the quotient handed over** — no scaffold, no index event |
| `track` | metered | venue, rotating | finest, forever | **ENUMERATION** — one entry per key cell |
| `merge` | metered | venue, rotating | coarsens by forced transfer | **the quotient, earned** |
| `merge_pool` | metered | venue, rotating | as `merge`, evidence pooled | does the quotient need pooling or only coarsening? |
| `dense` | **unmetered** | uniform (free i.i.d.) | — | the spec's literal monolith: no library at all |
| `dense_glob` | **unmetered** | uniform (free i.i.d.) | one cell | the monolith **with the same affordances**: a vocabulary, no index, no merge event |
| `dense_wd` | unmetered | uniform | — | the capacity meter (weight decay 1e-2 vs 1e-4) |

Rate-axis variants are arm overrides, e.g. `track:rotate_every=0` (no manufactured
decorrelation) and `merge:rotate_every=1` (fast cyclic).

**Why `dense_glob` exists.** `dense` is the spec's literal arm (no library ops), but this
substrate has a standing structural fact from `ratchet`: the base action space is *priced out*
of level-2 damage — the base-move exact-DP oracle at 5.9× the declared budget loses to one
level-2 macro at width 1. So `dense` may fail the in-distribution parity gate for a reason that
has nothing to do with enumeration vs merge. `dense_glob` is the monolith with the action-space
affordances held equal: unmetered, free i.i.d. variation, one global vocabulary, **no index and
no discrete merge event** — which is exactly the "ordinary dense learning under varied demand
achieves the quotient with no index event" case that confounds §9 names.

## The exams (drawn once, shared by every arm)

| exam | drawn from | announced key | what it is |
|---|---|---|---|
| **E1 home** | each training venue's *current* demand, training nodes | that venue | own turf; where concentration and hence enumeration should look best |
| **E2 envelope** | uniform demand, training nodes | each training venue | *the varied set*: the union the rotation exercises, which is also the unmetered arm's home distribution — so **readout 1's parity gate is read here, biased in the monolith's favour by construction** |
| **E3 wall5** | uniform demand, **held-out node** | **held-out venue** | outside every arm's varied set; a keyed index takes a genuine cache miss |
| **E3b** | uniform demand, **held-out node** | a training venue | separates "new frame" from "new key" |
| **E4 minability** | uniform demand, level-3 damage at `l3_node = 1` | — | readout 3 |

`l3_node = 1` spans level-2 nodes 2 and 3, **both training nodes** — otherwise readout 3
(minability) would be silently contaminated by readout 2 (the fifth wall).

## Gates (`gate_world`, offline; run inside every main run and asserted in `selfcheck`)

| Gate | What it asserts |
|---|---|
| **M-1** | **Fidelity.** At σ = 0 the venue sampler is *distributionally* `ratchet.context_instances`: same `d0`, on-grammar 1.000 both, level-2 demand histogram KL ≈ 0. Not stream-identical — the parent draws `rng.integers`, the weighted sampler `rng.random` (setlist's D-1) — which is why bit-identical replay of `ear/er_s0` is **not** the fidelity claim here; the instance draw is restructured into per-(venue, node) groups |
| **M-2** | **Nothing becomes false.** True tables are the same object at every epoch, instances stay 100% on-grammar, `d0` is stationary across rotation events |
| **M-3** | **The confound exists.** I(venue; level-2 latent) > 0 at σ > 0 and ≈ 0 at σ = 0 |
| **M-4** | **Rotation decorrelates.** The *cumulative* joint (pooled over epochs) loses mutual information while the *instantaneous* one does not — the precise sense in which paced rotation pays the confound down |
| **M-5** | **The fifth wall is admissible.** The held-out node's demanded level-2 mass must be covered by the training nodes' support, or nothing could transfer and the transform is a measured null (setlist's level-2 lesson) |
| **M-6** | **Merge is index-only.** After a merge the surviving cell's entries are a subset of the representative's (rep) / of the union (pool). No entry is ever a blend |

Measured at (v=8, s=2, depth=4, m=2, rule_seed=0), all passing:

- **M-1**: `d0` 2.185 (parent) vs 2.153 (venue sampler at σ=0); on-grammar **1.000** both;
  level-2 demand-histogram KL **0.0048**.
- **M-2**: true tables bit-identical across 8 rotation epochs; on-grammar min **1.000**;
  `d0` spread **0.23** (σ=1) / **0.31** (σ=2.5) with no trend.
- **M-3**: I(venue; latent) = **0.138 nats** at σ=1.0, **0.305** at σ=2.5, **0.348** at σ=3.0;
  σ=0 control **0.002**.
- **M-4**: at σ=2.5 over 8 epochs, `I_inst` 0.305→0.281 (no decay) while `I_cum`
  0.305→**0.186** (−39%).
- **M-5**: **1.000** of the held-out node's demanded level-2 mass lies inside the training
  nodes' support (15 entries at the held-out node, 19 across training nodes). The fifth wall is
  a pure *index* transform: the content transfers, the address does not.
- **M-6**: asserted in `selfcheck` for both merge modes.

`selfcheck` additionally asserts the index algebra (I-1 partitions, I-2 cache miss/fallback,
I-3 merge is index-only, I-4 a perfect alias merges and a disjoint pair does not, I-5 mutual
information, I-6 the ratchet nesting still bites: |T3| over a 1-entry T2 = **0** vs **60** over
the true T2).

## Pre-run offline calibration (`cal_loud`, zero GPU)

### The loudness knob: σ sets how loud the free observable is

`setlist`'s lesson inherited verbatim — **sweep, don't solve**. The direct read of "what the key
buys" is the gap between a per-venue demand-matched table's coverage of *its own* venue and of
*another* venue's (level 2, node 1, `cover = 0.8`, 6 venues):

| σ | I(venue; latent) | self-coverage | cross-coverage | **gap** | H(demand)/venue | entries/table |
|---|---|---|---|---|---|---|
| 0.0 | 0.001 | 0.825 | 0.825 | **0.000** | 2.301 | 6.0 |
| 1.0 | 0.139 | 0.846 | 0.694 | **0.153** | 1.961 | 5.0 |
| 2.0 | 0.261 | 0.865 | 0.654 | **0.211** | 1.686 | 4.2 |
| **3.0** | **0.349** | **0.864** | **0.573** | **0.291** | **1.542** | **3.7** |
| 5.0 | 0.442 | 0.892 | 0.538 | 0.354 | 1.414 | 3.5 |
| 8.0 | 0.512 | 0.921 | 0.506 | 0.415 | 1.329 | 3.5 |

At σ = 0 the key buys **exactly nothing** (self = cross), which is the free-i.i.d.-variation
regime and the unmetered arm's world. **σ = 3.0 was chosen** for the main run: a 0.29 coverage
gap and 0.35 nats of confound, past the steep part of the curve and short of the saturation
where a venue's demand collapses onto 3 entries and the whole level-2 vocabulary becomes
trivial.

### The rate axis: κ and the rotation period set how fast the confound is paid down

60 cycles, 6 venues, level 2, `cover = 0.8`. `i_cum` is the mutual information of the *pooled*
sample over the whole run — what a library accumulating across epochs actually faces; `half` is
the number of cycles for a per-venue table frozen at cycle 0 to lose half its coverage.

| σ | κ | period | `i_inst` | `i_cum` | `i_cum / i_inst` | cov 0 → end | half-life |
|---|---|---|---|---|---|---|---|
| 3.0 | 0.15 | 0 (none) | 0.348 | 0.348 | 1.00 | 0.882 → 0.882 | > run |
| 3.0 | 0.15 | 8 | 0.348 | 0.170 | 0.49 | 0.882 → 0.775 | > run |
| **3.0** | **0.15** | **4** | **0.348** | **0.113** | **0.32** | **0.882 → 0.511** | **48** |
| 3.0 | 0.15 | 1 | 0.348 | 0.042 | 0.12 | 0.882 → 0.671 | 12 |
| 5.0 | 0.15 | 4 | 0.442 | 0.130 | 0.29 | 0.892 → 0.427 | 16 |
| 3.0 | 0.50 | 4 | 0.244 | 0.033 | 0.14 | 0.867 → 0.577 | 48 |

**`period = 4`, κ = 0.15 is the main run's rate**: the confound is paid down to 32% of its
instantaneous value over the run and a frozen per-venue table has a 48-cycle half-life — slow
enough that a table is worth mining, fast enough that freezing one is punished inside 60 cycles.
`period ∈ {0, 1}` are carried as arm overrides, which is §4's rate axis.

Note (inherited from setlist): the OU is **mean-reverting**, so a taste can come back into
fashion and endpoint coverage understates erosion — per-epoch trajectories are the honest
readout for anything demand-relative.

### The structural prediction the rate sweep implies, checked offline before any GPU

Exact parses, 240 clean derivations per venue, level-2 mining at nodes 2–3, `support = 3`,
level-3 spans observed from 4096 uniform derivations:

| regime | per-venue \|T2\| | → \|T3\| | union \|T2\| | → \|T3\| |
|---|---|---|---|---|
| σ = 0 (free variation) | 15.0 | 83.7 | 19 | 93 |
| σ = 3.0 (the confound) | 9.8 | **47.7** | 19 | **93** |

Under the manufactured confound a per-venue (enumerated) level-2 vocabulary makes **about half**
of the next level representable that the merged one does — the ratchet's nesting, read as the
cost of the index. Under free variation the gap nearly closes. This is readout 3's mechanism,
present in the substrate before any learner is involved.

## Runs on disk

| tag | what it is |
|---|---|
| `smoke0` | attached quick smoke, 8 arms, 10 cycles, 3 venues. 93 s. Validated every code path incl. merge events firing |
| `smoke1` | attached quick smoke, 5 arms, after adding `dense_glob` and the shared-observation level-3 readout |
| `mg_s0` | **the main run**: 14 arms, seed 0, 60 cycles, 6 venues (+1 held out) × 3 training nodes (+1 held out) = 18 keys, σ=3.0 / κ=0.15 / rotation every 4 cycles from c4, `dense_mult` 4, merge τ=0.95 every 8 from c16, 24 forced-transfer probes. **1153 s.** All six world gates passed in-run |

### Instrument caveats found in `mg_s0` — read these before trusting a number

1. **The fifth wall is not difficulty-matched.** The held-out level-2 node (node 0) is
   intrinsically *easier* than the training nodes: `never_base`, which holds no index at all and
   therefore cannot take a cache miss, scores **E3 − E2 = −0.102**. So the raw `E3 − E2` is
   negative for 12 of 14 arms and its **sign is uninterpretable**. Use the `never_base`-corrected
   column (`(E3−E2)_arm − (E3−E2)_never_base`) or the per-arm E3 *trajectory*, both of which the
   analyzer prints. A future round should either difficulty-match the held-out node (measure
   `d0`/floor per node and pick a matched pair) or hold out a *venue only*, where the frame is
   identical by construction.
2. **`e(T3 macro)` is not a pure representation readout.** The level-3 audition is executed by
   the *arm's own* generator, which has drifted differently per arm — `e(true T3)` (the DGP's own
   level-3 table on the same instances) ranges 0.48–0.66 across arms. Compare `e(T3 …)` only
   against that arm's own `e(true T3)`. The size/recall columns (`|T3| shared`, `t3s_recall`) are
   the arm-comparable ones because they are computed from an oracle observation set that is
   identical for every arm.
3. **Bigger mined tables can score worse.** `never_base` mines |T3| = 61 shared entries and
   auditions at 0.874 against its own true-table reference of 0.659 — the per-key winner's curse
   `crystallize` priced at 3.3×, one level up. `|T3|` measures representability, not quality.
4. **No `global:rotate_every=0` arm was run**, so the keyed-vs-unkeyed comparison inside the
   no-rotation world (the cleanest read of §8's "scaffold value") is missing one cell.

Modal volume (`rhm-scaling-data`): `/data/rhm_practice_merge/<tag>/<arm>/results.json`, with
`setup.json` beside them (config, refs, the full gate record, the training keys). Fetched copies
live in `figures/<tag>/`.

## Reproduce

```bash
cd experiments/            # MODAL_PROFILE=chromatic

modal run rhm/practice/merge/merge.py::selfcheck_remote
modal run rhm/practice/merge/merge.py::cal_loud_remote --tag cal_loud

python3 rhm/practice/merge/launch_detached.py --fn merge --tag mg_s0 --seed 0 \
    --arms "never_base,given,global,track,track:mine_support=1,merge,merge_pool,dense,dense_glob,dense_wd,track:rotate_every=0,merge:rotate_every=0,track:rotate_every=1,merge:rotate_every=1" \
    --cycles 60 --n-venue 6 --train-nodes "1,2,3" --hold-node 0 --l3-node 1 \
    --sigma 3.0 --kappa 0.15 --rotate-every 4 --rotate-start 4 --demand-levels "0/1" \
    --n-pr 72 --n-rt 96 --dense-mult 4 \
    --merge-every 8 --merge-start 16 --merge-tau 0.95 --merge-probe 32 --merge-min-obs 8 \
    --mine-support 3 --budget 4 --pr-width 16 --g-budget 58 --n-grad 4 --gen-steps 20 \
    --n-ex 192 --n-probe 48 --n-e4 1536 --n-mine-post 1536 --probe-every 6

python3 rhm/practice/merge/analyze_merge.py --tag mg_s0 --fetch --figures
```

## Figures

| Path | What |
|---|---|
| `figures/<tag>/fig1_competence_granularity.png` | Metered competence per cycle with rotation events marked, beside **key granularity over time** — the merge-vs-track index trace |
| `figures/<tag>/fig2_exams.png` | E1 / E2 / E3 / E3b per arm, beside the **transfer cost** `E3 − E2` and `E3b − E2` |
| `figures/<tag>/fig3_minability.png` | \|T2\| the reading holds, \|T3\| mined from what the arm solved, and \|T3\| over a **shared** observation set (the nesting constraint alone) |
| `figures/<tag>/fig4_transfer.png` | The last **forced-transfer matrix** of each merge arm: `S[a][b]` = success of cell a's table on cell b's demand |

---

# Round 2 — `selfplay`: is free data free coverage?

**Why.** `mg_s0`'s monolith was handed **free i.i.d. variation over the whole space**, and gate
M-5 certified the fifth wall *inside* the training support. Both are exactly the conditions
self-play lacks: self-play is training where the question model and the answer model are the same
object, so the demand distribution is **the policy's own footprint**. The AlphaZero-shaped hole
was unprobed. This round probes it.

## Calibration that killed a design (recorded because it is the round's main methodological content)

The first build restricted the support by **venue** — each venue an OU demand state, the
self-play arm choosing which venues to practise. **That axis cannot make a hole in this
substrate.** Measured offline over σ, at 12 venues and 4 nodes:

| σ | entries/venue | union | mean pairwise Jaccard | a 3rd venue's mass covered by 2 venues | by 3 | by 6 |
|---|---|---|---|---|---|---|
| 3 | 14.6 | 19 | 0.714 | **0.990** | 0.996 | 1.000 |
| 6 | 11.8 | 19 | 0.656 | 0.939 | 0.976 | 0.992 |
| 10 | 10.2 | 19 | 0.554 | 0.903 | 0.964 | 0.989 |
| 16 | 9.6 | 19 | 0.550 | 0.879 | 0.945 | 0.981 |
| 25 | 9.0 | 19 | 0.532 | **0.870** | 0.926 | 0.980 |

The level-2 vocabulary has only **19 reachable entries** pooled over nodes and every venue demands
9–15 of them, so venue-restriction saturates: even at σ = 25 two venues cover 87% of a third's
demanded mass. Confirmed on GPU (`spsmoke0`): `dense_sp_hard` collapsed to a visitation
perplexity of **1.46 venues out of 6** and still covered **96.4%** of every venue's demand.
Venue-restriction buys narrow **volume**, not a hole.

So the support axis moved onto **the latent itself** — the level-2 entry the damage destroyed,
which the clean derivation names exactly — and the world was set to **uniform demand**, i.e.
literally the free-i.i.d.-variation distribution `mg_s0`'s monolith trained on. The only thing
that differs between arms is **which of the instances the world freely offers each arm chooses to
practise on**. Nothing becomes false; the grammar and the world distribution are untouched
objects.

The entry distribution under uniform demand is strongly skewed, which is what makes a tail
starvable — and what makes the hole invisible to an i.i.d. benchmark:

| | value |
|---|---|
| distinct level-2 entries (pooled over 4 nodes) | **19** |
| world mass, rarest → commonest | **0.0038 → 0.134** |
| entries per node | 15–16 |

## Arms

| arm | meter | support policy | what it isolates |
|---|---|---|---|
| `never_base` | metered | world as-is | the floor |
| `given` | metered | world as-is | **the intrinsic per-entry difficulty profile** everything is netted against |
| `metered_glob` | metered | world as-is | the `mg_s0` practice anchor |
| `dense_full` | **unmetered** | world as-is | exogenous full support (= `mg_s0`'s `dense_glob`) |
| `dense_visit` | unmetered | `p(e) ∝ (n_seen_e + ε)^β` | rich-get-richer on visitation, **no competence coupling** |
| `dense_selfplay` | unmetered | `p(e) ∝ (n_solved_e + ε)^β` | **the footprint filtered through competence** — the self-play arm |
| `dense_sp_hard` | unmetered | same, β = 2 | the amplification knob |
| `dense_narrow` | unmetered | fixed exogenous entry subset, size matched **in-run** to `dense_selfplay`'s realized entry perplexity | narrow support **without** self-selection |

There is **no exogenous seed**: at zero history every policy is the world's own distribution, so
the footprint has to be earned (asserted in `selfcheck_sp` S-2). `p` re-weights a freshly drawn
candidate batch, so the realized distribution is `world_freq(e) × p(e)` — a policy can starve what
the world offers but cannot conjure what it does not.

## Gates

Inherited M-1…M-6 from `merge.gate_world`, plus the two mirror gates, both measured from each
arm's **realized** support (accumulated exactly from the clean derivations it practised on — an
oracle instrument, never consumed):

| Gate | What it asserts |
|---|---|
| **X-2 inclusion** | the exogenous full-support arm realizes a support covering **every** entry — the re-run of `mg_s0`'s M-5, in the direction that made that round's fifth wall admissible |
| **X-1 exclusion** | the self-play arm leaves entries its realized support genuinely starves (< 0.2% of its draws), or there is no hole to find and the round is a measured null |

`selfcheck_sp` asserts the support algebra with no substrate: S-1 policies, **S-2 the competence
coupling** (an entry seen 40× and solved once gets 0.0027 of the draws under `solved` vs 0.0459
under `visit` — a 17× starvation that visitation alone does not produce), S-3 β is the amplifier,
S-4 the resampler realizes `world_freq × p(e)`, S-5 the inherited world gate.

## The difficulty-match fix (`mg_s0` caveat 1, closed)

`mg_s0`'s raw `E3 − E2` was uninterpretable because the held-out node was intrinsically easier
(`never_base` offset −0.102). Here the exam is a **per-entry battery pooled over every node**, so
no held-out-frame offset can arise, and the residual intrinsic difficulty is measured *directly*
by `given` — which holds the DGP's own table and sees every entry — with every per-entry readout
reported as excess over it. Nothing leans on a `never_base` offset.

## The three metrics (readout 1)

Every arm's exam error is reported under three weightings, which is the round's headline
construction:

- **own footprint** `Σ q_e · err_e` — the objective a self-generated demand distribution
  implicitly optimizes;
- **world i.i.d.** `Σ freq_e · err_e` — what *any* test set drawn from the same generator
  measures;
- **flat** `mean_e err_e` — coverage.

`own − world` is self-grading inflation. `world − flat` is the part of a hole an i.i.d. benchmark
**structurally cannot see**, because the starved entries are the rare ones.

## Code files (round 2)

| File | Purpose |
|---|---|
| `selfplay.py` | The Modal app (`rhm-practice-merge-selfplay`). Imports `_cfg`, `solve`, `exam`, `minability`, `post_mine_t2`, `parse_arms`, `gate_world` from `merge.py` and the substrate from `ratchet`/`crystallize`/`setlist`; **`merge.py` is not modified**, so `mg_s0` stays reproducible. `entry_of` reads the destroyed level-2 latent exactly off the clean derivation; `build_world_sp` buckets a large pre-draw by entry into the per-entry exam battery; `support_weights` / `resample` are the four support policies; `baseline_profile` is `s0_e`; `run_arm_sp` is the loop. Entrypoints: `selfplay`, `selfcheck_sp{,_remote}` |
| `analyze_selfplay.py` | Reduction: gates, the three metrics, per-entry holes with a head/tail split by world mass, the self-selection correlations **partialled on log world mass** (the common driver of both visitation and difficulty), and minability. `--figures` writes fig1–fig3 into `figures_sp/<tag>/` |

## Runs on disk (round 2)

| tag | what it is |
|---|---|
| `spsmoke0` | quick smoke of the **venue-axis** design. 140 s. Killed that design: perplexity 1.46/6 venues at 96.4% minimum coverage |
| `spsmoke1` | quick smoke of the entry-axis design |
| `sp_s0` | **the main run**: 8 arms, seed 0, 60 cycles, `dense_mult` 4, 19 level-2 entries, exam 224 instances/entry (pool 30k/node), β = 1.0 (hard arm 2.0), ε = 1.0, oversample 6. **994 s.** All six world gates plus X-1 and X-2 passed |
| `spsmoke2` | quick smoke of the `sp_s0c` arm paths (both narrow variants select the intended entry sets; realized perplexities 2.30 / 1.95) |
| `sp_s0c` | **the control launch** (517 s, 3 arms, overlaid onto `sp_s0` by `analyze_selfplay.py --extra`): `dense_narrow:narrow_k=2` (the caveat-1 size-matched fix; realized perplexity 1.56, lands on two *tail* entries — size-matched but placement-anti-matched), `dense_narrow:narrow_fixed=3-0` (**yoked placement-matched control**: support fixed to `dense_sp_hard`'s own dominant entries, mass-ranks {3, 0}; perplexity 1.98), and `dense_selfplay:sp_beta=1.5` (the amplification midpoint). Overlay world-determinism check: max \|Δfreq\| 0.00e+00, max \|Δs0\| 0.0000, no config diffs. **Caveat 1 below is closed by this launch**; verdict in [`README.md`](README.md) round-2 finding 2 |

### `sp_s0c` code additions (all additive; `sp_s0`/`mg_s0` reproduce bit-identically)

`selfplay.py`: cfg key + entrypoint arg `narrow_fixed` (default `""` → the existing RNG draw is
unchanged; non-empty names entry mass-ranks, `-`-separated because `parse_arms` splits arms on
`,`), plus one log line naming a narrow arm's realized entry set. `analyze_selfplay.py`:
`--extra TAG[,TAG…]` overlays a later launch's arm directories onto a base tag's report (base
supplies `setup.json`), printing the world-determinism check; overlaid figures write to
`figures_sp/<base>+<extra>/` so the base figures stand.

### Instrument caveats found in `sp_s0` — read these before trusting a number

1. **The exogenous control is matched to the wrong arm.** `dense_narrow`'s size is matched
   in-run to `dense_selfplay`'s realized perplexity (6.22 → 6 entries, realized 3.99). But the
   arm that produces the effect is `dense_sp_hard`, whose perplexity collapses to **1.73**. There
   is therefore **no narrowness-matched exogenous control for the arm that shows the result**, so
   `dense_sp_hard`'s coverage loss cannot yet be attributed to self-selection rather than to
   narrowness per se. Fix: a second narrow arm at `--narrow-k 2`. **[CLOSED by `sp_s0c`** — both a
   size-matched and a placement-matched (yoked) exogenous arm were run; the yoked arm reproduces
   only 22% of the flat loss and none of the tail excess, attributing the effect to the
   competence coupling. See the `sp_s0c` row above and `README.md`.]
2. **`world − flat` does not isolate what it was built to isolate.** It is negative for *every*
   arm including `given` (−0.074), because the rare entries are intrinsically harder
   (corr(world mass, `given` error) = **−0.642**). The quantity is dominated by the difficulty
   profile, not by holes. Use the **excess head/tail split** (netted against `given`) instead.
3. **The `given` baseline compresses at both ends.** `given`'s per-entry error spans 0.040–0.469
   and is worst exactly on the rare entries, so "excess over `given`" has a floor on the head and
   a ceiling on the tail. Tail-excess numbers under ~0.06 should not be read.
4. **Single seed**: per-entry SE ≈ √(0.25/224) ≈ **0.033**, so any individual per-entry excess
   below ~0.06 is noise; the 19-entry aggregate columns carry ≈ 0.008.

Volume: `/data/rhm_practice_selfplay/<tag>/<arm>/results.json` + `setup.json`; fetched copies in
`figures_sp/<tag>/`.

## Seeds

`--seed` sets damage draws, mining and probe RNG; `--rule-seed` / `--train-seed` set the DGP
draw and the substrate, defaulted to `ratchet`'s (`0`/`0`/`1`). `--demand-seed` is a separate
stream, so the venues' motion is independent of every arm's randomness. Single seed by default;
absolute error levels are not comparable across rule draws, so **rank orderings, signs and
transfer costs are the reported quantities**.

## Inherited, not copied

`../ratchet/` (`build_shared`, `beam_moves`, `finetune_generator`, `value_steps`, `fit_width`,
`push`, `context_instances`, `macros.py` in full), `../crystallize/units.py` (the level-indexed
action space, hierarchical damage, exact-DP oracle, grading), `../setlist/demand.py`
(`new_demand`, `typical_demand`, `demand_step`, `sample_pool_demand`,
`context_instances_demand`, `demand_table`, `coverage`, `demand_kl`, `demand_entropy`), and
`rhm/rhm_drift.py` through them. Nothing in those files is modified.
