# FILES — `crescendo` (A3: does the earnable range extend with the turn of the crank?)

Machinery record for the node. **No results are interpreted here** — the reduction's numbers go
to the orchestrator and are discussed before any writeup (repo norm).

**Up**: parent arc [`../README.md`](../README.md) · `../../../../ROADMAP.md`[^private] §1.4 and §4.1 (third shape, A3)
**Direct donor** (untouched): [`../maestro/`](../maestro/FILES.md) — A2. `maestro.py` is forked
here; `maestro/policy.py` is **imported, not forked** (this round adds no rule); `ma_s0` is the
cross-tag replay reference and `maestro/floors.json` supplies the measured floors.
**Through it**: [`../conductor/`](../conductor/FILES.md) (A1 — the thermostat, the floors, the
yoke and observation-panel machinery) · [`../census/`](../census/README.md) (the extension op,
finding 1's r² ratchet, finding 7's stream floor) · [`../assay/`](../assay/FILES.md) (the
substrate; "arrival dominates") · [`../spiral/`](../spiral/README.md) (the three clocks;
interpretation (c) named this round) · [`../tall/`](../tall/README.md) (the re-run recipe, and
the standing finding that endogenous pacers cannot traverse a ladder longer than the earnable
range).

## The question, and the one thing that varies

§1.4's distinctive prediction: **the earnable range extends with each turn of the crank.** Every
run in this arc has held `max_macro_level=3` and treated L4 as unearnable — the spiral's
interpretation (c) says in as many words that whether the thin frozen L3 is benign or a
foreclosure "is not measurable at this depth" and is the named question of the next round. This
is that round. L4 is opened to the crank, and the question is whether the value clock holds past
the range the previous turn certified.

`active = era.level + 1` and the mining gate is `min(maxl, era.level + 1)`, so at
`max_macro_level=4` level 4 is **mined from era 3 and committable in era 3** — exactly the deal
L2 got in era 1 and L3 got in era 2. (The brief anticipated "committable from era 4"; the code
says era 3, uniformly with every rung below. Trusting the code.) Nothing else about the world
moves: same substrate, same ladder sequence, same reads, same floors, same rule.

| arm | who paces | L4 committable | extension | role |
|---|---|---|---|---|
| `anchor_long` | schedule (era lengths **=** the caps) | yes | no | the carrier, the cross-tag replay gate, and the **lifetime ceiling** |
| `outer_yield_m4` | A1's thermostat on `yield` | **yes** | no | **the treatment** |
| `ceiling_m3` | clock replay of the treatment | **no** (`commit_max_level=3`) | no | **the ceiling control** — the pair the signature is read on |
| `outer_yield_m4x` | A1's thermostat on `yield` | **yes** | **yes** | the r²-wall bypass |
| `outer_yield_m3` | A1's thermostat on `yield` | **no** (`commit_max_level=3`) | no | the free-pacing ceiling control |

## Code files

| file | purpose |
|---|---|
| `phase0_l4.py` | **The offline phase.** Re-executes the substrate's own `Miner.build` on the logged `keys_at_support` streams in `cd_s0` and `ma_s0` to size the L4 question before any GPU: the true level sizes, the buildable-L4 ceiling under each arm's frozen / live / true L3, the era-3-restart bound, and A1's thermostat replayed on the logged L4 series to find when it would fire. Writes `phase0.json`. No GPU, no Modal, no substrate. |
| `phase0.json` | Phase 0's output, including gate B-1's verdict. |
| `crescendo.py` | The substrate. Forks `../maestro/maestro.py`; every addition marked `# [crescendo]`. Entrypoints: `preflight`, `fidelity_smoke`, `crescendo_run`. |
| `analyze_crescendo.py` | The reduction. A2's, plus §A–§E (the crossing, the signature, the frontier gauge, the pair's twin window, extension). |
| `launch_detached.py` | Session-isolated detached launcher (A2's, retargeted). |

`policy.py` and `floors.json` are **deliberately absent**: the rule and the dead zones are A1's,
reached by importing `rhm.practice.maestro.policy`. That import is the structural form of the
round's claim — *the rule did not change; the rung did.* All 20 offline gates (P-1…P-12,
L-1…L-8) therefore apply unchanged and were re-run for this node: **ALL PASS**.

## What the fork adds (each `# [crescendo]`-marked)

| addition | where | why |
|---|---|---|
| **`commit_max_level`** | `_d6_cfg`, `run_arm` block (g), the panel's `will_commit` | caps what may be **committed**, as against `max_macro_level` which caps what may be **mined / observed / slotted / audited**. `None` → `max_macro_level`, so the file is its donor when unset. See below — this split is the round's real design decision |
| **the five arms + their stream keys** | `ARMS`, `TWIN` | all on the anchor's stream, so every arm is bit-identical to `anchor_long` until its own first action |
| **`CRESCENDO_CAPS` / `_LADDER` / `_ARMS`** | module constants | the caps are sized by Phase 0, and the ladder is set **equal to the caps**, which is what makes `anchor_long` the lifetime ceiling |
| **`max_macro_level` default 3 → 4** | `crescendo_run`, `preflight` | the one constant the round moves |
| **gate C** | `preflight` | the L4 commit path, the `commit_max_level` refusal, the L4 extension, the yoke handoff from an m4 arm, and the lifetime ceiling, all executed at toy sizes before a paid setup |
| **G-F retargeted** | `fidelity_smoke` | this fork's direct donor is `maestro.py` |
| **§A–§E** | `analyze_crescendo.py` | the round's own readouts |

## The one new key, and why the ceiling control needs it

Stated here because, as in A1 and A2, the control is the round's real design decision.

The obvious ceiling control is "the same arm at `max_macro_level=3`". It is **wrong**, and
measurably so: `max_macro_level` also sets the proposal head's slot layout
(`PN.slot_layout(depth, s, maxl)`), the range of the `committed` dict and the `miners` dict, the
observation panel's committable half, and — decisively — the range of the per-cycle **shadow
audition loop**, whose matched-size random control draws `rng.permutation` from the arm's own
shared RNG stream. So an `maxl=3` arm and an `maxl=4` arm diverge in the shared stream from the
first era-3 cycle at which `miners[4]` is non-empty, for reasons that have nothing to do with the
L4 commit.

`commit_max_level` splits the two. `ceiling_m3` and `outer_yield_m3` run at `max_macro_level=4`
— identical head, identical miners, identical panel, identical auditions, identical RNG draws —
and differ from the treatment in **one bit**: whether a firing at the active level may install a
table.

*(The head is safe on the other side too: `ProposalHead.out` is **zero-initialised and
constructed last**, so the trunk's parameters are bit-identical at `n_slots` 56 and 60 and the
four extra rows are zeros the availability mask never selects.)*

**Two ceiling controls, because they answer different questions.** The same `loop.quiet` latch
drives both the commit (block (g)) and the era advance (block (g4)), so an arm forbidden the L4
commit takes an *era advance* on the latch that would have licensed it. That is a real
behavioural difference and it costs lifetime:

- `ceiling_m3` is a **clock yoke** of the treatment, so it holds the treatment's era boundaries
  and its L2/L3 commit cycles cycle-for-cycle. Lifetime-identical by construction, one bit
  different. **This is the pair the signature is read on.** A1 finding 4 and A2 finding 4 both
  measured gauge-and-yoke bit-identity, which is what licenses the substitution.
- `outer_yield_m3` is **gauge-driven**, so it has the treatment's pacing *freedom* but not its
  realised pacing. It answers the other half — what the rung's existence did to the loop's own
  pacing — and it carries the lifetime confound, which is why it is not the signature's pair.

`ceiling_m3` also **is** this round's non-invasiveness check: it must be bit-identical to the
treatment up to the treatment's own L4 commit cycle (reduction §D), which is what a `yoked_m4`
arm would have shown, at no extra arm.

## Lifetime, matched by construction

A1 and A2 both carry the caveat *"arms are not lifetime-matched"* (loop arms reached 125–146
cycles against the anchor's 116), so every deep-era comparison there conflates pacing with
practice time. Here the schedule arm's **scheduled era lengths are set equal to the caps**
(`CRESCENDO_LADDER` ≡ `CRESCENDO_CAPS`): a schedule arm reads `era["cycles"]` and a loop arm
reads `caps[era_i]`, so `anchor_long` runs the longest life any arm may have and every loop arm
can only be shorter. A positive loop delta therefore cannot have been bought with time. The
treatment/ceiling pair is exactly lifetime-identical on top of that.

## The gauge at the frontier — measured, not chosen

A1's commit rule for active level ℓ reads `at_support` at ℓ+1. At era 3, ℓ+1 is **5**, and
Phase 0 measured the L5 read degenerate on this substrate: `obs_hist[5]` reaches a maximum of
**1–2** distinct tuples at support over an entire run and is first non-zero at **c89–124**
(level 5 has 205,824 distinct true tuples against a saturated ~8 observations/cycle). A
paired-interval slope on a series whose whole dynamic range is {0,1} is zero almost everywhere,
so a thermostat driven on it would fire at its first decision point.

**Chosen**: the L4 commit is paced on **the L4 stream's own quieting** — `read_level` clamps to
`gy_level = 4` exactly as it already does, which at era 3 makes the driven read the level *being
earned* rather than one above it. **Logged**: `read_level` per cycle, and the L5 and L6 series
uncharged in every arm's panel; reduction §C prints both. The nearest ancestor of an at-support
read at the committing level is `census`'s G-A admission-rate gauge — **not** `outer_ledger`'s
within-level *task error*, which is the reader A1 finding 2 measured to refuse.

## Phase 0 — the round sized offline, before any GPU

`phase0_l4.py`. **Gate B-1**: the offline replay of `Miner.build` reproduces the runs' own
committed tables entry-for-entry — **18/18 commit events**, matching `n_entries`, `recall` and
`precision` exactly. So the numbers below are the substrate's own arithmetic, replayed, not a
model of it.

**The levels** (distinct flat tuples — the recall denominator): L2 **14**, L3 **56**,
L4 **816**, L5 **205,824**.

**The r² wall binds, and it is not empty.** Buildable L4 at end of run, over each arm's own
frozen L3 / live L3 / the true L3:

| arm | frozen L3 (recall) | L4 at support | buildable \| frozen | true | buildable \| live | true | buildable \| true L3 | true |
|---|---|---|---|---|---|---|---|---|
| `cd_s0/anchor` | 12 (0.071) | 68 | 8 | 1 | 13 | 3 | 17 | 16 |
| `cd_s0/outer_yield` | 21 (0.125) | 87 | **15** | 3 | **25** | 6 | 22 | 21 |
| `ma_s0/learned_yield` | 18 (0.089) | 74 | 13 | 2 | 26 | 6 | 20 | 19 |
| `cd_s0/outer_endo` | — (0.000) | 84 | — | — | 16 | 7 | 26 | 24 |

r² predicts 0.125² × 816 ≈ 13 against the measured 15 — the wall is real and arithmetic.
**Extension roughly doubles the ceiling** (frozen → live: 15→25, 13→26, 8→13), which is what
justifies `outer_yield_m4x`.

**The era-3-restart bound.** `miners[4]` is cold at era 3, so the G-Y cumulative stream is an
upper bound on it. Restricting to L4 keys that first reach support in eras 3+ (~28–36 cycles in
the donors) leaves **2–5** buildable entries over the frozen L3 and **4–8** over the live one —
thin, but non-empty, so the commit can fire. The observation stream runs at a saturated
**7.8–8.0 observations/cycle in every era**, so era length maps linearly onto observations.

**When the rule fires.** A1's thermostat, replayed unchanged on each donor arm's own logged L4
at-support series from era-3 start, first goes quiet at **c_in_era3 = 17 / 18 / 18 / 22 / 28 /
30** across the distinct trajectories, and fires again **9–21 cycles later**. So the L4 commit
should land near c_in_era3 ≈ 18 (with ~144 observations banked, hence a non-empty table) and the
era advance near c_in_era3 ≈ 30–40.

## Caps and arm order

`CRESCENDO_CAPS = 60,50,70,12,9` (= 201), and `CRESCENDO_LADDER` is the same numbers.

- **Eras 1–2 are A1's caps unchanged** (60, 50), so the L2/L3 books — and hence the r² wall the
  L4 commit runs into — are the ones the donors measured, and so the treatment is bit-identical
  to `ma_s0/outer_yield` through both earning eras (the round's free full-scale fidelity gate).
- **Era 3 goes 15 → 70.** Sized by Phase 0: the previous earning eras got 48–60 cycles each, the
  rule is expected to commit near c_in_era3 18 and advance near 30–40, and 70 is ~2× the
  expected exit — headroom for two or three empty-table retries without the cap binding.
- **Eras 4–5 are A1's caps unchanged** (12, 9): they are where the value clock is read, and the
  displacement floors this round reuses (census finding 7; `ma_s1`) were measured at those era
  lengths.

Arm order is load-bearing: `anchor_long` first (the carrier; nothing is readable without it),
`outer_yield_m4` second, `ceiling_m3` immediately after it (it replays the treatment's
**measured** actions, passed in at runtime), then `outer_yield_m4x`, then `outer_yield_m3` last
as the most cuttable.

## Gates

| gate | what | where | status |
|---|---|---|---|
| P-1…P-12, L-1…L-8 | A1's and A2's full offline policy suite, re-run against the imported module | `maestro.policy.policy_gate()` | **ALL PASS** (20/20) |
| **B-1** | **the offline replay of `Miner.build` reproduces the donors' own commit log** entry-for-entry | `phase0_l4.py` | **PASS** (18/18 events) |
| **C-1** | the new rung is reachable: an L4 commit installs a table and grows the action set by one macro per L4 node | `crescendo.py::preflight` | **PASS** — `anchor_long` climbed L2 c6, L3 c12, **L4 c17**; action set 56 → 60 (+4) |
| **C-2** | **`commit_max_level` binds, and only on the level it names**: a clone of the schedule arm carrying that one cfg override commits L2/L3 at identical cycles, never commits L4, and is bit-identical to its twin up to the forbidden commit | `crescendo.py::preflight` | see Runs |
| C-3 | extension reaches the new rung's frozen table | `crescendo.py::preflight` | **PASS** — `census_extend` extend events at levels 2, 3, **4** |
| C-4 | the lifetime ceiling binds — no loop arm outruns the schedule arm | `crescendo.py::preflight` | **PASS** |
| C-5 | neither ceiling arm holds an L4 table | `crescendo.py::preflight` | **PASS** |
| **inertness** | **`commit_max_level` moves nothing when it never binds** — the other half of the one-bit claim | offline, on `_preflight` results | **PASS** — `outer_yield_m4` vs `outer_yield_m3` and vs `ceiling_m3`: **0.000e+00** over 12 series / 27 cycles, first divergence None |
| N-1, N-2, F | A1's endo label-freeness, endo price, floor gate | `crescendo.py` | as A1 |
| G-F smoke | at `max_macro_level=3` with `commit_max_level` unset, this fork replays **`maestro.py`** in process | `crescendo.py::fidelity_smoke` | see Runs |
| **cross-tag (windowed)** | **`cr3_s0/outer_yield_m4` vs `ma_s0/outer_yield` through eras 1–2** — the round's strongest fidelity statement, and free; plus `anchor_long` vs `ma_s0/anchor` and `cd_s0/anchor` through era 1 | `analyze_crescendo.py` §0 | see Runs |
| twin | every loop arm vs `anchor_long` up to its own first action | `analyze_crescendo.py` §0 | see Runs |
| **pair** | **the treatment and `ceiling_m3` bit-identical up to the treatment's L4 commit**, and first divergence == that cycle | `analyze_crescendo.py` §D | see Runs |
| preflight | every call signature and every new branch at toy sizes before a paid setup; the arm list is a parameter (`--arms`) and every assertion is guarded on the arm having run, so a fix can be re-checked without re-paying for all 22. Its tolerances are dust and **no number from it is a measurement** | `crescendo.py::preflight` | see Runs |

**Why the cross-tag window is bounded at era 2.** A1's era-1 and era-2 caps are unchanged here,
and `max_macro_level=4` touches no shared RNG until `miners[4]` is non-empty, which the mining
gate defers to era 3. Beyond era 2 the ladders differ by construction (era-3 cap 15 → 70), so a
full-length comparison would report a designed difference as a fidelity failure.

## Runs

| tag | what | outcome |
|---|---|---|
| `_preflight` | interface + gate C, toy sizes. First pass ran **all 22 arms** without crashing and failed only on a *donor* assertion hardcoded to `maxl=3` (`set(shadow_cert) == {"2","3"}`; at `max_macro_level=4` the certificate is correctly per level 2, 3, **4**) — generalised. Every other donor assertion, and gate C-1/C-3/C-4/C-5, verified offline against that run's own results. Second pass re-checks C-2's GPU clone on the 7-arm subset | `results/preflight.log`, `results/preflight2.log` |
| `cr3_gf` | G-F: in-process fork-vs-`maestro.py` replay at `max_macro_level=3` | PASS — 0.000e+00 vs 0.000e+00 self-replay control, commits equal |
| `cr3_smoke` | five arms end-to-end at `--quick`; mechanics only, and the **s/cycle measurement** that confirms the budget | clean; 10.6 s/cycle |
| `cr3_s0` | the main run: 5 arms, A1's floors, Phase-0's caps | clean; findings in [README.md](README.md), full record `figures/cr3_s0/reduction.txt` |
| `cr3_s1` | stream-displaced twins of the signature pair (`--ref-tag cr3_s0`; the ceiling twin yoked to the *displaced* treatment's realized actions, so the one-bit pair is rebuilt on its own draw) | the two-draw signature (§F) and the in-node displacement floors (§9) of the merged `reduction.txt` |

Volume `rhm-scaling-data:/data/rhm_practice_crescendo/<tag>/`; fetched copies, figures and
`reduction.txt` under `figures/<tag>/`.

## Reproduce

```bash
cd experiments/            # MODAL_PROFILE=chromatic

# offline, no GPU: the policy suite (P-1..P-12 + L-1..L-8), then Phase 0
PYTHONPATH=. python3 rhm/practice/maestro/policy.py
PYTHONPATH=. python3 rhm/practice/crescendo/phase0_l4.py

modal run rhm/practice/crescendo/crescendo.py::preflight   # --arms "..." re-checks a subset
modal run rhm/practice/crescendo/crescendo.py::fidelity_smoke --tag cr3_gf

python3 rhm/practice/crescendo/launch_detached.py --fn crescendo_run --tag cr3_smoke --quick \
    --quick-cycles 12 --quick-probe-clean 512 --quick-gen-steps 20 \
    --endo-price 267 --seed 0

python3 rhm/practice/crescendo/launch_detached.py --fn crescendo_run --tag cr3_s0 \
    --eras "1:25:60,2:12:50,3:6:70,4:3:12,5:1:9" --era-caps "60,50,70,12,9" \
    --arms "anchor_long,outer_yield_m4,ceiling_m3,outer_yield_m4x,outer_yield_m3" \
    --max-macro-level 4 --endo-price 267 \
    --budget 8 --g-budget 482 --n-corrupt 1 --mine-cap 8 --gy-level 4 --entry-rec \
    --span-min-hold 128 --collect-task-matched --tm-episodes 8192 \
    --recert-every 5 --n-aud 192 --probe-every 8 --n-rt 384 --n-score 256 \
    --sil-cv 0.15 --lp-min-drop 0.10 --seed 0

python3 rhm/practice/crescendo/analyze_crescendo.py --tag cr3_s0 --fetch --figures
```

The floors are the defaults (A1's measured values) and are asserted, so no `--tol-*` literals
are passed.

[^private]: Not mirrored: this link points to a document in the private lab repo (the roadmap, the queue, an unrun spec, reading notes, or a conversation). See the top-level README for what is held back and why.
