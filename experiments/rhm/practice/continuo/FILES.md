# FILES — `continuo` (E2: the instrument, its design calls, its gates)

Machinery record for the node. **No results are interpreted here**, and there is deliberately no
`README.md` yet: the node's output is a `reduction.txt` plus figures, and the writeup happens
after those, with Jasper.

**Up**: parent arc [`../README.md`](../README.md) ·
`../../../../ROADMAP.md`[^private] §4.2 shape **E2**, as reframed by §7.1.3 ·
`../../../../QUEUE.md`[^private] § "Track E′" (the E2 entry is the spec, including
its design warnings).

**Donor (untouched, read-only)**: [`../audiation/`](../audiation/FILES.md). `au_s0/anchor` is a
per-decision log of a bit-identical replay of `conductor`'s `cd_s0/anchor` (`0.000e+00` over all
116 cycles), and it carries the three things E2 needs: **117 per-cycle head snapshots**, the
per-cycle action-set context (`ms_slots` / `ms_level` / `ms_node` / `avail_slots`, so the
instruction/data label is checkable per token), and the outer loop's per-cycle panel (so
`at_support` is on the same clock). `analyze_audiation.head_forward` — the numpy re-implementation
of both heads that the donor's own run certified against its live readouts — is *imported*, not
copied. Nothing here writes into `audiation/`, and `audiation/fit.py` is not touched.

**Recipes ported**: `papers/forward_self_models_paper2.md`[^private]
(the object — an implementation fact; criterion 2 — beat the I/O-only twin at matched capacity;
spherical k-means over residual directions; §6's component contrast; §8's junk guards) ·
[`../../../a2a_forward/confabulation/component_control/`](../../../a2a_forward/confabulation/component_control/README.md)
(the `ens_cos` guard and the headroom fraction `frac`, and the reason raw advantage is the wrong
statistic when the self saturates on some targets and not others) · paper 1's saturation law (the
FM saturates at ~the predicted layers' parameter count).

## Code files

| file | purpose |
|---|---|
| `continuo.py` | The whole offline instrument and its reduction. No GPU, no Modal, no substrate, no torch: it reads the donor's fetched bytes, recomputes π and the value head at a fixed probe set from each of the 117 snapshots, fits the sub-saturation FM, and writes `figures/<tag>/{reduction.txt, continuo.json, occupancy.png, decode_guards.png}`. The module docstring is authoritative on the design calls; this table is the map. |

## What the node does, in one pass

1. **Gates.** `_gradcheck()` (analytic vs central differences, float64) as a hard assert; the
   donor's own replay gate read back; a **head-recompute gate** — the numpy π must reproduce the
   run's own logged `t_pi` at the fp16-`z` storage scale `audiation` measured (1.2e-3 on a logit
   scale of ~3.4).
2. **Probe.** A fixed set of practice-beam tip states (`t_z`, `t_root`) stratified over the
   fetched cycles, split by `(cycle, instance)` — never by row.
3. **Primary pass.** For every checkpoint `c ∈ [5, 117]`: π's logits at the probe set over the
   cycle's live slots → FM fit → held-out residual `R`, plus the command-energy share and
   `native`'s can't-decompose readout recomputed offline.
4. **Axis 1 — occupancy.** One spherical-k-means codebook (K=8) over an era-balanced pool of
   residual directions from all checkpoints, applied unchanged to every checkpoint; occupancy
   histograms, their total-variation drift, and where the *named* consolidation events sit in
   that drift distribution.
5. **Axis 2 — the decode.** Per-token instruction/data classification from the slot's residual
   column, leave-one-slot-out, against the observer twin at matched capacity.
6. **Guards.** Capacity × seed sweep (`ens_cos`, `|r|/|t|`, capacity invariance of the decode),
   hierarchy-η² against matched-random groupings, the scalar-norm negative, the shuffled null,
   the slot-label permutation null, the value-head control target.
7. **`at_support`.** Spearman, raw and after residualising both series on cycle and era.

## Design calls (the module docstring is authoritative; this is the index)

| # | call | why, in one line |
|---|---|---|
| 1 | the FM predicts **π's logits** from `(z, root)` | the encoder is frozen (§4.2's "trunk" is the wrong target, as §7.1.3 says); of the two trainable heads only π has a per-**token** output space, and the instruction/data bit is a property of the action vocabulary |
| 1b | the **value head** is a control target in the same FM class | it has no token index, so anything that moves at the commits on both is the state distribution or the calendar, not the command port |
| 1c | **the corridor is re-scoped out of pass 1** | `au_s0` runs `span_mode=False` — there is no span head in this donor at all; consolidation here is routing-only, which is still exactly the command/data object, and an executor-side arm is named as the follow-up rather than smuggled in |
| 2 | probe = the **practice beam's own states**, fixed across checkpoints | `probe_trace.npz` is the era's *metering* set — an eval; §7.1.3 requires a gauge that is not audition-shaped. Fixing the set makes a change in the residual the head changing, not the states changing |
| 3 | FM = a `ProposalHead` with hidden width `H` | same shape of read as the predicted head, so nothing is bought by architecture mismatch — and at `H = 384` the FM *is* the head's architecture, i.e. exactly 100% of the predicted span: the saturation rung, included on purpose |
| 3b | zero-init output layer | with a z-scored target the FM starts at the target mean; with a random output layer the small rungs did not converge inside the iteration budget and `ens_cos` sat in the junk band |
| 3c | `H = 0` is a **ridge** rung | a convex FM has a unique minimiser, so its residual cannot be instrument noise (and `ens_cos` is 1.000 trivially). The floor of the theory ladder, not a substitute for the swept MLP |
| 4 | occupancy scored against **one** codebook | per-checkpoint histograms have to be comparable across time; the codebook is lifted into full 56-slot space so pre- and post-commit directions live in the same space |
| 5i | **degenerate columns excluded** | π's out layer is zero-init and non-live slots get no gradient, so a slot's logit column is *exactly* 0 until a few cycles after its commit — a decoder handed a constant-zero column reads "command" perfectly, and that is wiring |
| 5ii | the **age disambiguator** | a command token is also a young token; L2-macro vs L3-macro (both commands, ~98 vs ~47 cycles old) separates a decoder reading age from one reading command-ness |
| 6 | every side gets a **column over the same probe states**, the same projection, the same folds | matching the *shape* is what makes the twin fair, where `audiation`'s low observer rungs were architecture-limited (they had to rebuild the frozen encoder and sat at ≈0) |
| 6b | headline statistic = **balanced** accuracy | the classes are 32:16 then 32:24; raw accuracy is unreadable. Chance is 0.500 by construction |
| 6c | λ-grid **median**, not the best λ | no selection on the test slots anywhere; capacity invariance is readable off the grid's spread |
| 7 | η² is reported **against a matched-random grouping** | η² is inflated by group count and small groups and is unreadable without its own null |
| — | `OPENBLAS_NUM_THREADS=2` set before the numpy import | the image's OpenBLAS is built `MAX_THREADS=64 NO_AFFINITY`; on 4 cores its spin-wait dominates the small matmuls this file is made of — 8 ms vs 0.2 ms for one `(2148,192)@(192,6)` product, a 40× difference in total runtime |

## The observer ladder, and one degenerate rung worth naming

`R = Π − FM(z, root)`. The observer holds `Π` (or less) and not `z`, so it cannot form the
subtraction; that is the reconstruction cost, and it is exactly "hold the encoder state and refit
a small FM." Paper 2's **activation-access observer** (`O_act`, handed `a_i`) is therefore
*identical to the self side here* rather than an intermediate rung, because the residual is a
deterministic function of `(z, Π)` and the FM is shared. It is not computed; it is stated.

## Reproduce

```bash
cd experiments/            # MODAL_PROFILE=chromatic; no GPU and no Modal job is run

# fetch the donor's bytes (CPU-side; ~120 MB, gitignored under continuo/data/)
mkdir -p rhm/practice/continuo/data/au_s0/anchor
modal volume get --force rhm-scaling-data \
    rhm_practice_audiation/au_s0/anchor/snapshots  rhm/practice/continuo/data/au_s0/anchor/
for s in dec_c0006_c0010 dec_c0021_c0025 dec_c0046_c0050 \
         dec_c0071_c0075 dec_c0096_c0100 dec_c0111_c0115; do
  modal volume get --force rhm-scaling-data \
      rhm_practice_audiation/au_s0/anchor/decisions/$s.npz \
      rhm/practice/continuo/data/au_s0/anchor/decisions/
done
# the two small JSONs are already in the repo under ../audiation/figures/au_s0/anchor/

python3 rhm/practice/continuo/continuo.py --smoke          # ~1 min wiring pass
python3 rhm/practice/continuo/continuo.py                  # the full pass
```

## Runs

| outdir | command | what it isolates |
|---|---|---|
| `figures/au_s0/` | default | **pass 1**: primary rung H=16 on all 117 checkpoints + the full capacity × seed guard sweep |
| `figures/au_s0_agefloor/` | `--h-sweep 16 --n-seeds 2` | follow-up **(a)**, the token-age-floor table; also a determinism check — reproduces pass 1 at every cell |
| `figures/au_s1_perdatum/` | `--tag au_s1 --arm perdatum --h-sweep 0,6,16 --n-seeds 3` | follow-up **(b)**, the within-donor consolidation contrast on its own derived clock (L2 c19, L3 c62) |
| `figures/au_s0_fmseed777/` | `--fm-seed 777` | follow-up **(c1)**, FM initialisation perturbed alone |
| `figures/au_s0_seed1/` | `--seed 1` | follow-up **(c2)**, full independent redraw (probe, split, projection, codebook, FM init) |
| `figures/au_s0_ridge/`, `figures/au_s1_perdatum_ridge/` | `--h-primary 0 --h-sweep 0` | the **matched-instrument** arm pair. (b) at H=16 turned out to compare instruments in different regimes — `ens_cos` 0.718 on `anchor` against 0.577 (junk band) on `perdatum`, because the per-datum head is far more FM-predictable. The ridge rung is deterministic on both arms, so `ens_cos` is 1.000 by construction and the two arms are read with the same instrument. Cheap: the whole pass is seconds of fitting. |

Later additions to `continuo.py`, all additive: `derive_events` / `derive_sweep` (events read off the run's own record, never hardcoded — `au_s0/anchor` and `au_s1/perdatum` do not share a clock; the old table is kept as a regression assert), `slot_live_cycle` + `AGE_FLOORS` (follow-up (a)), and `--fm-seed` (follow-up (c1): perturb the FM's init alone, leaving the draw, split, projection and codebook fixed, so instrument variance is separated from draw variance instead of confounded with it). One crash fixed on the way: a treatment arm legitimately has no replay reference, so `replay_gate.max_abs_delta` is `null` on `au_s1/perdatum`; the gate line now falls back to the tag's in-tag `anchor` inverse gate and says so.

## Children

None. No substrate, no GPU run in pass 1.

---

# PASS 2 — the corridor / span-side consolidation object

Pass 1 re-scoped the executor half out (design call 1c: `au_s0` runs `span_mode=False`, so its
donor has no corridor head at all). Pass 2 is that half. **Asked in**
`../../../../QUEUE.md`[^private] § "Track E′ → E2 pass 2". Still no `README.md`
and still no interpretation here.

**Donor (untouched, NOT edited — another agent is concurrently forking this lineage)**:
[`../intonation/`](../intonation/FILES.md). `ma_s0`'s regime is the configuration reproduced,
and `native/span`'s `SpanHead` is the predicted span.

## The object, and why the executor half is the better-conditioned ask

Pass 1's label was `ms_level[j] >= 2` — a slot is a command iff it is a macro — which is very
nearly "this token is young", and the headline decayed to zero under a 20-cycle token-age floor
with the scalar-norm negative in lockstep (`two_deltas` finding 3). On the executor side the
instruction/data bit is not the slot's LEVEL, it is its **parity gate**:

* gate **open** → the span is materialised by `SpanHead.emit` in one pass, no DP, no table
  consulted at execution time. A **command**: routed, can't-decompose.
* gate **closed** → `macros.apply_any` masks the span, reads per-block infill evidence and runs
  the max-sum DP over `T[ℓ] → … → T[1]`. **Data**: enumerated.

Checked against `ma_s0/mperf_log`'s own record before any of this was built, that bit **varies
within a level** (c60: 10 of 16 L2 slots open, 6 not), **varies within a slot over time** (44
gate events over 153 cycles, re-closures included) and **is not the slot's mint age** (slots
minted together at the L2 commit open anywhere from c49 to c77). So "command" and "new" are
separable in this donor by construction, and the age floors are applied from the first cell
rather than discovered afterwards. `open_tau` (parity ≥ `span_tau` = 0.95, the donor's own
criterion) rides free as a stricter grade of the same state, and is the balanced one late in the
run where `open` saturates.

## Code files (pass 2)

| file | purpose |
|---|---|
| `corridor.py` | The **snapshot-enabled fork** of the `intonation` stack. Adds no arm, no trajectory-changing knob, and **no line inside `intonation.py`**. The whole fork is (i) a monkeypatch of `span_net.parity` — the one function `run_arm` calls exactly once per cycle with generator, head, executor and slots all in hand — which calls the real parity, writes a per-cycle snapshot, and returns the real answer unchanged; (ii) `IN.REMOTE` retargeted to `rhm_practice_corridor`, so no `intonation` tag is written to; (iii) two entrypoints, `corridor_gf` (gate X) and `corridor_run` (the paid run). The `run_arm` frame is read (never written) for `cyc`/`arm`/`shared`/`cfg`. |
| `corridor_fit.py` | The offline instrument and its reduction. No GPU, no Modal, no torch. Reads the snapshots, fits the sub-saturation FM per checkpoint, and writes `figures/<tag>_<arm>/{reduction.txt, corridor.json}`. Imports `continuo.py`'s `_gelu`/`_dgelu`/`_proj`/`_loo_ridge_bal`/`spherical_kmeans`/`unit_rows`/`eta2_vs_random`/`partial_spearman`/`at_support_series` rather than copying them, so pass 1 and pass 2 share one set of guards. |
| `launch_detached.py` | `intonation/launch_detached.py`'s, retargeted at `corridor.py`. |

## Design calls (pass 2; the two module docstrings are authoritative)

| # | call | why, in one line |
|---|---|---|
| 1 | the FM predicts the **span head's block-0 logits** from the head's own read of the state plus the slot identity | the span head is the trainable readable surface on this stack, and block 0 is the one emission every slot has whatever its span (2 / 4 / 8 blocks at L2 / L3 / L4), so every slot's column has the same dimension without padding |
| 1b | the **trunk's per-block feature logits** (`feature_head(pooled)`) are the control target | they carry no slot conditioning anywhere in their computation — pass 1 design call 1b's value-head control, ported. Anything that reorganises at a gate event on both is the state distribution or the calendar |
| 2 | probe = a fixed set of sequences from **`shared["replay"]["x"]`**, with each slot's own span masked per slot | the head's input is a masked observation, and `SpanExecutor.apply` builds it exactly this way. NOT `ex.hold` — that is the parity gate's own metering set, i.e. an eval, and pass 1 design call 2 forbids an audition-shaped gauge. The probe SEQUENCES are shared across slots (only the mask moves), which is what makes one slot's column comparable with another's: the decode's rows are slots, so per-slot probe sets would leak slot identity into the classifier |
| 3 | the FM sees **three reads** — `pooled.mean(all blocks)`, the span's mean, the span's first block — plus a slot one-hot | `span_state` is `ctx(pooled.mean(1)) + slot(sid) + Σ_j in_proj[j](pooled[blk0+j])`, so this is the head's global read exactly and its span read up to the per-offset weighting. **Stated as a cost**: part of the residual is then input the FM never saw rather than structure it failed to theorise. It buys a fixed 3×96 input at every level — the alternative (the full padded 8×96 per-offset read) puts a dimensional signature of the slot's LEVEL into the FM's input, which is the confound this pass exists to control |
| 3b | the slot enters as a **one-hot**, not a learned embedding | a (28, 288) embedding is 8,064 parameters — 6% of the predicted head on its own — so the FM could not be sub-saturation and conditioned at the same time; and `SpanHead`'s slot embedding is part of the span being predicted |
| 3c | the **ridge rung (H=0) carries the full grid**, beside the swept MLP | pass 1's instrument-regime confound (an FM in the `ens_cos` junk band at matched `H` on one arm and not the other) was fixed exactly by reading the arms at the deterministic ridge. Pass 2 has two arms to compare, so the fix is built in rather than applied afterwards |
| 4 | sub-saturation is **computed, not asserted** | `head_params()` derives `SpanHead`'s count from the snapshot's own shapes: 133,736 at `state_dim` 96 / v 8 / `max_span` 8 / 28 slots / `hidden_mult` 4. The rungs are H=0 → 1.90%, H=8 → 2.42%, H=16 → 4.37% (primary), H=32 → 8.3% (over budget, flagged), H=384 → 94% (the saturation demonstration, fitted on a strided checkpoint subset and excluded from every headline) |
| 5 | **two decodes**, because the label has two axes | **A** per checkpoint with rows = slots (pass 1's exact shape, giving a trajectory) and **B** pooled over (slot, checkpoint) rows with **leave-one-SLOT-out** folds, which is where the age floors bite and where the within-slot temporal variation — the thing the routing-side bit did not have — is used |
| 5b | the permutation null permutes labels **within each checkpoint** | rows are (slot, checkpoint); a free row-permutation would destroy the per-checkpoint base rate as well as the slot↔label pairing and would be trivially easy to beat. Permuting within a cycle preserves *how many* slots are open at c and destroys only *which* |
| 6 | **two clocks** for the age floor | cycles since MINT (pass 1's control) and cycles since the slot first OPENED. The floors are 0 / 20 / 50 on both |

## Gates (pass 2) — gate **X**, the instrument is inert

| gate | what it asserts | where |
|---|---|---|
| **X-1** | in-hook, **every cycle**: the torch CPU RNG state and every CUDA RNG state are byte-identical across the snapshot, and the executor's `buf`/`hold` sizes and every slot's `open` flag are unchanged. Always on, so the paid run certifies its own instrument | `corridor.py::_install`'s wrapper |
| **X-2** | the same arm run with the hook OFF and ON, **one process, one shared dict**, log series compared against a **donor self-replay control** (the same arm twice with the hook off) — without which GPU nondeterminism and a real bug look identical | `corridor_gf` |
| **X-3** | the hook actually **fired**, on an arm where it can: `perf_given` holds the true tables and mints every slot at c1, because at smoke sizes a loop arm never commits, no slot is ever minted, and a gate over a hook that never fired would certify nothing (`intonation`'s own reason for carrying the `perf_given` family into preflight) | `corridor_gf` |
| gradcheck | `SpanFM`'s analytic gradients against central differences in float64, a hard assert at the top of the reduction (there is no autograd in `corridor_fit.py`) | `corridor_fit._gradcheck` |

## Runs (pass 2)

| tag | command | outcome |
|---|---|---|
| `co_gf` | `corridor_gf --tag co_gf --cycles 5 --n-probe 32` | **X-2 PASS — 0.000e+00 fork against a 0.000e+00 self-replay control** on all 13 log series; `gate_events` (40), `slot_events`, `events` and `perf_cells` all bit-identical off-vs-on. **X-3 PASS** — 15 snapshots, one per cycle, 28 slots × 32 probe states, every read finite and non-degenerate, both label classes populated. 0.538 MB/cycle at `n_probe=32`. One bug on the way, fixed: the gate's own comparison step read `gate_events` off `run_arm`'s **return** dict, which does not carry it (it goes only into the `results.json` that `write_results` writes) — the three runs had already completed, so X-2/X-3 were computed offline from the written bytes and the reporting line now reads them back. |
| `co_s0` | `launch_detached.py --fn corridor_run --tag co_s0 --arms "mperf_log,mperf_gain" --n-probe 96 --n-pr 24 --pr-width 8 --endo-price 267 --seed 0` | the paid run. `ma_s0`'s configuration exactly — every other flag is already `intonation_run`'s own default, so nothing is restated that could drift. Two arms: the metered baseline and the δ_perf **gain** arm (per-execution, per-attributable-cause credit by construction — the executor-side analogue of pass 1's surviving `perdatum` cell). **Clean, 4696 s = 1.30 GPU-h**; both arms 153 cycles, commits c43/c83/c115, 44 gate events (`mperf_log`); **110 per-cycle snapshots per arm** (c44…c153, 186 MB each), gate X-1 asserted on every one. |
| `co_s0` reductions | `corridor_fit.py --tag co_s0 --arm {mperf_log,mperf_gain} --seed {0,1}` | four reductions, ~1100 s each on 4 CPU cores. Outputs `figures/co_s0_<arm>_s<seed>/{reduction.txt, corridor.json}`. **Numbers are in the reductions; not summarised or interpreted here.** |

Three facts from `co_s0` that belong in the machinery record rather than in a results
discussion, because they are properties of the **instrument and its label**, not of the object:

- **The raw `open` label goes CLASS-DEGENERATE under the age floor**, and does so by losing its
  negative class rather than by losing an effect. Slots take a median of 17 (`mperf_log`) / 15
  (`mperf_gain`) cycles from mint to first opening, range 5–28, and **all 28 of 28 slots
  eventually open in both arms**. So at a 20-cycle mint floor the pooled rows are 1879/1912
  open (98.3%) and 1888/1912 (98.7%), and every cell — self, observer, shuffled and
  scalar-norm alike — returns exactly 0.500 because the classifier predicts one class. The
  50-cycle floor has no admissible cell at all. This is *why* `open_tau` was carried as the
  second grade: it stays balanced at every floor (43.2 / 54.2 / 65.4% positive on `mperf_log`;
  42.6 / 52.2 / 58.5% on `mperf_gain`). Any future pass asking this question on this substrate
  should treat `open` as an era-1-only label and read `open_tau`, or lower `span_tau_fire`
  further to keep slots closed longer.
- **The H=16 MLP rung lands in the `ens_cos` JUNK BAND on all four reductions** — 0.630 / 0.676
  (`mperf_log` seeds 0/1) and 0.625 / 0.670 (`mperf_gain`) against paper 2's 0.65 floor. This is
  pass 1's instrument-regime trap reproducing on the executor side at the same capacity, and it
  is the reason the whole decode grid is run at the deterministic **ridge** rung as well: the
  ridge's residual is a unique minimiser, its `ens_cos` is 1.000 by construction, and it is the
  only rung on which the two arms are compared with the same instrument. The ridge is also the
  *better* FM here (mean held-out MSE 0.0455–0.0482 against the MLP's 0.0472–0.0522), so it is
  the conservative choice as well as the matched one.
- **Occupancy TV drift is near-degenerate at this K and probe size**: the codebook trajectory
  has median TV drift 0.0000 in three of the four reductions (p90 0.0625–0.1875), so a large
  fraction of checkpoint steps are tied at zero and the "percentile among all steps" statistic
  is dominated by ties — a named event landing at "0.0%" means *no drift*, not *the smallest
  drift*. A future pass wanting this axis to resolve needs a larger K, a larger probe, or drift
  measured on the soft assignment rather than the hard histogram.

## Volume layout (pass 2)

`rhm-scaling-data:/data/rhm_practice_corridor/<tag>/{setup.json,summary.json,manifest.json,<arm>/{results.json,snapshots/span_cXXXX.npz}}`.
Fetched copies under `data/<tag>/<arm>/` (gitignored); reductions under `figures/<tag>_<arm>/`.

Each `span_cXXXX.npz` holds, per live slot × probe state: `feat` (the three pooled reads),
`lg0`/`lgm` (the head's block-0 and span-mean logits), `bl0`/`blm` (the trunk's, the control
target), `emit` (what the executor wrote), `dp` (what the DP would have written — the head's own
parity target, so the per-row execution error `e` and its exact-match bit are free on a fixed
distribution every cycle), `err`, `exact`, and the per-slot record `level`/`node`/`blk0`/`span`/
`open_pre`/`open_post`/`open_tau`/`parity`/`parity_n`.

## Reproduce (pass 2)

```bash
cd experiments/            # MODAL_PROFILE=chromatic

modal run rhm/practice/continuo/corridor.py::corridor_gf --tag co_gf --cycles 5 --n-probe 32

python3 rhm/practice/continuo/launch_detached.py --fn corridor_run --tag co_s0 \
    --arms "mperf_log,mperf_gain" --n-probe 96 \
    --n-pr 24 --pr-width 8 --endo-price 267 --seed 0

mkdir -p rhm/practice/continuo/data/co_s0
modal volume get --force rhm-scaling-data rhm_practice_corridor/co_s0 \
    rhm/practice/continuo/data/

# the reduction, per arm, at two independent draws (pass 1's draw variance exceeded its headline)
for a in mperf_log mperf_gain; do for sd in 0 1; do
  python3 rhm/practice/continuo/corridor_fit.py --tag co_s0 --arm $a --seed $sd \
      --out rhm/practice/continuo/figures/co_s0_${a}_s${sd}
done; done
```

[^private]: Not mirrored: this link points to a document in the private lab repo (the roadmap, the queue, an unrun spec, reading notes, or a conversation). See the top-level README for what is held back and why.
