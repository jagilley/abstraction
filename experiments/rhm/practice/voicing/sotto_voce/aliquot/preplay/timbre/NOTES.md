# timbre — NOTES

The implementer's record for this node: every decision and why, every gate and what it was shown
to fail on, every defect beside its correction. Written up in [`../../rubato/README.md`](../../rubato/README.md) §2 (2026-09-24).

**Node**: `experiments/rhm/practice/voicing/sotto_voce/aliquot/preplay/timbre/` (a child of
`preplay/`; the name: the same pitch, the level, played on a different instrument, the reader).
**Up**: [`../README.md`](../README.md) §2 (pp1), §4 (pp3), §6 (pp5) · the pattern:
[`logit_reading/striatum/norm/precision/README.md`](../../../../../../../logit_reading/striatum/norm/precision/README.md)
§8 and §10.
**Runs**: 2026-09-23. `tb1`, app `ap-FeFKiR0oYSJ4zEQObQZ6p5`, eight L4 containers, **0.65 GPU-h**
(2339 container-seconds). See §6.

## 0. The question, in one line

pp1's "shaped above frozen above twin, frozen and twin at chance" and pp5's "the frozen and twin
levels in the gate's seat are worse than no gate" were measured with the linear readout only. On
the other substrate every claim about a reader's level held on richer readers and every claim about
a difference of levels depended on the form. **Re-read pp1, pp3 and pp5's banked fires with richer
readers of the same pooled state, fit on the same rows through the same three trunks, and run the
banked reductions unchanged on each reader's columns.**

## 1. Decisions

1. **Fork, not flag.** `timbre.py` copies `preplay.preplay_arm`'s cell loop, `own.own_arm`'s step
   2 and `readgate.readgate_arm`'s walk loop line for line in their random-draw order, and replaces
   only the read (`readers.read_heads`: one trunk pass per trunk per fire, every head read off it).
   The banked scripts are untouched. Arm 0's columns are then asserted equal to the banked JSON on
   the volume, in-container, before a cell's other columns are written (T-5, T-6; §3).
2. **One set of rows for every reader.** Every reader except `slot` is fit on pp1's own shared
   subsample of the arm's bank (`FIT_SEED_BASE = 7_300_000`, 8192 train / 2048 validation rows),
   so a column differs from pp1's `shaped_refit` / `frozen` / `twin` by the reader's form alone.
   Selection (ridge, MLP stopping step, MLP lr and weight decay) uses the validation rows only;
   the 2048 hold rows are only ever scored ([T1]). The three index sets are asserted disjoint.
3. **The belief is the plant's own block head on the reader's own input.** `SN.trunk` returns
   the pooled per-block hidden and the block logits from one pass; the belief block is the
   log-softmax of those logits over the level-1 alphabet (v = 8), pooled the two ways the hiddens
   are (all blocks; the span), 16 columns. The input is the configuration the projection reads:
   the fired span holds the write and the first block outside the span is masked to -1, so the
   span pool is the belief at the fired slots and the all-block pool includes the masked slot. The
   alternative reading of "the masked / fired slots" -- the executor's own pre-fire belief with the
   span masked (`fire_rec`'s `block_logits(obs)`) -- was not taken: that is the DP's input, the
   prior, and appending it would hand the reader new information (the executor's score) rather
   than a new form of reading the same state.
4. **The standardisation, ported.** `precision/express.py` scales appended columns to the raw
   state's per-dimension RMS through the Gram. Here `pj_design` already z-scores every state column
   to unit sd before a uniform penalty, so the belief columns are appended to the feature matrix
   before `pj_design` and z-scored by the same line on the same slice; each meets the penalty on the
   scale every state column does, and is crossed with the root like them (1881 columns).
5. **A structural fact that bounds `belief`.** The block head is `feature_head(pooled)`, a linear
   map of the per-block hidden that is pooled, so the pooled logits are already in the linear span
   of the pooled hiddens. The log-softmax differs from them by one log-partition per block: the
   belief block adds to the ridge's span only the two pooled log-partitions (each crossed with the
   root), plus whatever the ridge's shrinkage does differently with 16 handed-over directions. This
   is `precision`'s open `+logits` question, answered on this plant by construction.
6. **The MLP.** `striatum/task.py`'s 128 GELU units, one hidden layer, over [z-scored pooled
   hiddens, root one-hot] (200 inputs), BCE with logits, AdamW, cosine schedule, minibatch 512,
   8000 steps, the validation AUC read every 100 steps and the best state kept, the (lr, wd) pair
   chosen on the validation AUC over lr {1e-3, 3e-3} x wd {0, 0.01, 0.1}. The root enters as an
   input, not crossed with the features: the hidden layer can form the interaction. §2 records why
   the learning rate is a ladder.
7. **`bonly`**, the MLP on the 16 belief columns plus the root one-hot: `precision`'s public
   reader ported. **`bonly_lin`**, the ridge on the belief block alone (153 columns), is read in
   rr1 only (free there: no extra fires).
8. **`slot`**, the per-slot form: the projection's design (root interaction, IRLS, the same ridge
   ladder) fit on ALL the bank's trainable rows at the fired cell's own (blk0, span), the u0 split
   as train / validation. The shared subsample holds ~900 rows there, too few for 1737 columns, so
   this reader does not share pp1's rows; it is a different diet slice by construction and is read
   as such. It exists only where the slot holds >= 1000 train and >= 200 validation rows: L2n12
   (2363 / 592 rows at seed 0) and L3n6 (2189 / 541); L4n3 (203 / 42) and L5n1 (21 / 7) are too
   thin. It is not walked in pp5 (a gate defined at two of seven cells would pool unlike cells).
   `within`'s per-slot controls were fit on the probe buffers' counterfactual rows (a different
   diet); that form does not transfer here without changing the diet, which is the other axis.
9. **pp3: step 2 only.** The re-read prices the learner's operative rows one at a time (pp3's §4
   table). pp3's step 3 (the top-k as a table on disjoint test pools) is a selector question and was
   not re-run; `reduce_own.py` needs it and is therefore not run on the views -- [T6] is this node's
   own table in pp3's shape.
10. **pp5's views.** For reader R the walks R@shaped (pooled), R@shaped~pair, R@shaped~m (margin =
    the standard error of R's OWN level on the base, per cell and repeat), R@frozen and R@twin
    stand in pp5's `read`, `read_pair`, `read_m`, `frozen` and `twin` slots. `ridge@frozen` and
    `ridge@twin` ARE pp5's `frozen` / `twin` and are not walked twice; `ridge@shaped` (the refit,
    view `ridge_refit`) is walked beside the banked read because every other reader is a refit.
    The secondary orders run for the banked gates only (so arm 0 reproduces pp5 whole).
11. **Seeds 0 and 2, six arms for rr1, the two of record for rr5**, as pp1 / pp3 and pp5 ran.
    One container per arm; rr5's two settings share a container, because splitting them would buy
    five minutes of wall-clock for a second ~2-minute reader fit.
12. **The delta form is re-read and carries no claim**, exactly as in pp1 (its effective n is
    2-76 of 288 for the world itself).
13. **The leakage count.** Every fired and pool configuration is hashed with its root and looked
    up in the bank's fit rows and in the whole bank; the count is carried per cell ([T0]). An MLP
    can memorise what a ridge cannot, so the disjointness the brief asks for is measured on the
    configurations actually read, not assumed from the seed families.

## 2. The MLP recipe, chosen on validation rows only (`timbre.py::mlp_recipe`, s0_sv)

The first build fit one MLP per weight decay at a fixed lr 1e-3 for 3000 steps. On the smoke the
twin's MLP read its own validation rows BELOW the twin's ridge (0.711 against 0.728) with early
stopping picking the last steps, the sign of an under-trained reader, which would bias the
controls toward "shaping is required". The probe, validation AUC, serial fitter, best over wd:

| reader | ridge | 3000 steps, lr 1e-3 | 8000, 1e-3 | 8000, 3e-3 |
|---|---|---|---|---|
| mlp@shaped | 0.8792 | 0.8787 (step 200) | 0.8787 (200) | 0.8746 (100) |
| bonly@shaped | 0.8792 | 0.8677 (2300) | 0.8696 (4700) | 0.8699 (950) |
| mlp@frozen | 0.7582 | 0.7716 (2500) | 0.7786 (1900) | 0.7844 (3100) |
| bonly@frozen | 0.7582 | 0.7031 (2100) | 0.7115 (3650) | 0.7160 (1600) |
| mlp@twin | 0.7275 | 0.7105 (2950) | 0.7120 (4800) | 0.7306 (4450) |
| bonly@twin | 0.7275 | 0.6562 (2200) | 0.6723 (5250) | 0.6899 (5950) |

No single rate is best everywhere (the shaped state over-shoots at 3e-3 within 100 steps; the
frozen and twin states want the higher rate and the longer schedule), so the rate joined the weight
decay in the per-fit validation ladder, and the six (lr, wd) models are trained as one batched
ensemble (a hand-written AdamW over stacked parameters; one fit of all six costs ~19 s against
~36 s for ONE serial fit). The batched ladder at 8000 steps reads 0.8796 / 0.8694 / 0.7857 /
0.7156 / 0.7286 / 0.6870 on the six rows above, at or above the serial best on each. The weight
decay barely moves any model (AdamW's decoupled decay at these rates); the rate and the stopping
step carry the selection.

## 3. The gates. T-1..T-4 and T-7 closed in `gates_t`; all seven falsified in `falsify_t` (7/7)

| gate | what it asserts | closed at | falsified by |
|---|---|---|---|
| **T-1** | `readers.feats`' hidden block IS `preplay.pj_features`, bit for bit; its belief block IS the generator's own `block_logits` on the same masked input, log-softmaxed and pooled | max\|dFh\| = 0.0 and max\|dFb\| = 0.0 through all three trunks, 2048 hold rows, 16 belief columns | the read with the mask off: max\|dFh\| 1.47, max\|dFb\| 5.43 |
| **T-2** | the banked head through `read_heads` IS `pj_predict`, bit for bit; `fit_lin` on the hidden block IS `pj_fit` on pp1's slice | max\|dp\| = 0.0; the same lambda (32), the same validation AUC to 1e-9, max\|dw\| = 0.0 | the banked head read through the frozen trunk: max\|dp\| 0.998 |
| **T-3** | the MLP fitter can fit: a sign-product (XOR) of two real feature columns is recovered on held-out rows | validation AUC 0.976 (the ridge on the same target 0.662) | the target's labels permuted: 0.509 |
| **T-4** | `gated_walk_g` IS `readgate.gated_walk` on pp5's seven gate names; a reader key under the generic pooled / paired / margin rules IS pp5's rule on the same array; handed -e as its level every generic form IS the world gate; non-vacuous (4 distinct admission sets) | identical on all seven names and all five generic forms | the level handed +e (the sign not flipped): admits 26 of 26 against the world's 20; on the designed sequence `0.50 / 0.40 0.45 0.40 0.50 0.10 0.11` a `cur` pinned at the base admits all six against `[0, 2, 4]` |
| **T-5** | IN-CONTAINER, every rr1 arm: arm 0 (banked, ridge@shaped/frozen/twin) against the banked pp1 JSON per candidate in both forms (world price exact, levels to 1e-5, per-instance AUCs to 1e-4) and against pp3's per operative row; the three refits' lambda, validation AUC and hold AUC | `tb1`: 4/4 pp1 levels and 3/3 pp3 levels on all six arms, every difference 0.0 (bit-identical), the refits' lambda / validation / hold AUC identical | the comparator on (i) one frozen level nudged by 2e-5, (ii) the other seed's cell, (iii) one pp3 banked level nudged by 2e-5: 1, 390 and 1 violations |
| **T-6** | IN-CONTAINER, both rr5 arms: pp5's seven banked gates and four secondary-order walks, every cell and repeat, EXACTLY: admission lists, rejections, confusion counts, audition counts, kept / wrong counts, gate-pool and test errors at every budget | `tb1`: 14/14 (cell, repeat) on both arms, worst \|d\| 0.0; the unchanged `reduce_readgate.py` on the arm-0 view reproduces `../figures/readgate_reduction.txt` [G50]-[G53] line for line, and `reduce_preplay.py` on the ridge view reproduces pp1's [A:single]..[I] line for line | the comparator on the twin gate's admissions reversed, and on one test error nudged by 1e-9: 1 and 1 violations |
| **T-7** | the leakage counter counts: a fired / pool configuration equal to a bank fit row is found | 0 of the 2048 hold rows are fit rows (they cannot be, by the h split) | five fit rows planted in a pool of 64: counted 5 |

`preplay`'s F-1 (the recording DP is `MC.apply_any`) re-runs on the first fire of every level in
rr1; F-2b (the banked hold AUC inside the readout's drift band) re-runs in every container; P-4
(pool families disjoint) re-runs in rr5; pp3's reconstruction gates R-0..R-4 re-run in every
container that reads the learner's tables.

## 4. Defects, and their corrections

**Defect 1 -- the first MLP recipe under-trained the control readers.** A single learning rate
(1e-3, 3000 steps) put the twin's MLP below the twin's ridge on its own validation rows. Caught on
the smoke, before anything was paid for; corrected by the (lr x wd) validation ladder at 8000 steps
(§2). The serial fitter it replaced is not in the code; its probe table is above.

**Defect 2 -- the serial MLP fitter was launch-bound** (~4.5 ms per step for a 26k-parameter net,
so one 8000-step fit took 36 s and the ladder would have cost ~0.5 GPU-h in fits alone across the
sweep). Replaced by the batched ensemble; the per-model result is the same object (one init, one
minibatch stream, its own rate and decay).

**Not a defect, a property of the banked reducers worth carrying**: `reduce_readgate.py` writes to a
fixed path under its own directory and appends to `preplay_reduction.txt` unless told not to, and
`reduce_preplay.py` writes its per-candidate TSVs under its own `figures/<tag>/`. Running them
"unchanged" on this node's views therefore means pointing their module-level `HERE` at a scratch
directory (the only place either reads its location from) and passing `--no-append`; nothing is
written into `preplay/figures/`, and the banked table of record is untouched.

## 5. Two facts about the banked record the re-read surfaced (facts; the parent READMEs are untouched)

1. **pp1's "shaped above frozen above twin on 8 of 8 cells" is 6 of 8 as a strict order in pp1's
   own table**: at L2 on both seeds the frozen column is below the twin's (0.488 < 0.529 at seed 0,
   0.498 < 0.503 at seed 2). "Shaped above both frozen and twin" is 8 of 8.
2. **pp5's pooled capture (0.27 at b = 8, 0.57 at the full walk) is a median over both seeds'
   28 (cell, repeat) pairs; per seed it is 0.58 / 0.93 at seed 0 and 0.14 / 0.03 at seed 2.** The
   seed-2 medians sit on a thin denominator (the median gap between no gate and the world's gate
   is 0.018 at b = 8 there), so `timbre_reduction.txt` [T7] also counts, per (cell, repeat), wins
   / ties / losses against no gate at a 0.005 tolerance: the banked read beats no gate on 7 / 8 of
   14 at seed 0 and 10 / 8 of 14 at seed 2 (b = 8 / full), and loses on 2 / 2 and 3 / 3.

## 6. Reproduction

```bash
cd experiments/            # MODAL_PROFILE=chromatic
T=rhm/practice/voicing/sotto_voce/aliquot/preplay/timbre
modal run $T/timbre.py::gates_t                # T-1..T-4, T-7, the T-5/T-6 comparators on identity
modal run $T/timbre.py::falsify_t              # 7/7
modal run $T/timbre.py::mlp_recipe             # the recipe, validation rows only (§2)
modal run $T/timbre.py::sweep --out-tag tb_smoke --arms1 s0_sv --arms5 s0_sv --smoke 1
modal run --detach $T/timbre.py::sweep --out-tag tb1
python3 $T/reduce_timbre.py --tag tb1 --fetch
```

## 7. Runs

| tag | what | containers | cost | app |
|---|---|---|---|---|
| `tb_smoke` | rr1 + rr5 on s0_sv at smoke sizes (the first MLP recipe) | 3 L4 | 0.09 GPU-h | (attached) |
| `tb1` | rr1 on six arms (183-243 s each), rr5 on the two of record (466 / 505 s) | 8 L4 | 0.65 GPU-h | `ap-FeFKiR0oYSJ4zEQObQZ6p5` |

Peak RSS 5.1 GB per container against the 8 GB request; peak GPU 1.1 GB. The local mirror
`figures/tb1/` holds the per-arm JSON gzipped (1.8 MB; 13 MB plain on the volume) and the logs.
