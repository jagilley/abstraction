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

None. No substrate, no GPU run in pass 1. The follow-up that would need one — a `crescendo`
fork with additive per-cycle head snapshots (`# [continuo]`-tagged, G-F bit-identical with the
additions off), buying the live L4 frontier and optionally a `span=True` arm for the
executor-side half — is not built.

[^private]: Not mirrored: this link points to a document in the private lab repo (the roadmap, the queue, an unrun spec, reading notes, or a conversation). See the top-level README for what is held back and why.
