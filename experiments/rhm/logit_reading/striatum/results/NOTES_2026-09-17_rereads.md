# Three strengthenings of the value-side rounds — 2026-09-17

Facts and commands only. No README was edited; each item lands as a reduced table beside
the node it extends. Up: [`../README.md`](../README.md) (striatum) ·
[`../norm/README.md`](../norm/README.md).

| item | what | where | new compute |
|---|---|---|---|
| (a) | §5's across-level contrast on the **MLP** critic, and the `j`- and `(j, k*)`-stratified re-rank of all three | [`tables_mlp_across.md`](tables_mlp_across.md) | none — a pure re-read |
| (b) | a **second and a third trajectory seed**, `a1_s43` and `a1_s44`, and norm's cells on each | [`../norm/results/tables_s43.md`](../norm/results/tables_s43.md), [`tables_s44.md`](../norm/results/tables_s44.md) (§0b is the three-seed side-by-side) | 2 L4 trajectories + norm's two waves each |
| (c) | a **mean-and-variance critic** on a continuous target | [`../norm/results/tables_varhead.md`](../norm/results/tables_varhead.md) | 3 L4 cells on the banked seed |

---

## (a) The across-level contrast on the MLP critic — a pure re-read

**What it tests.** §5 read the per-level pattern of the value revision against the levels
the edit actually changed, with the state held identical across query levels. It was run
on the **linear** critic only, while the MLP critic beats it by up to 0.08 on the
revision's realised-damage AUC (§2: 0.677 vs 0.601 at `ℓ = 1`), so the null could have
been the readout's rather than the value channel's. The MLP's per-level revisions
`R_mlp_l{l}_a{a}` are already stored on the identical rows in
`stepNNNNNN_striatum_<tag>.npz`; nothing had ever called `across_level` on them.

**Command of record** (CPU, local; `analyze.py` gained an additive `--mlp-across` mode
that writes only `tables_mlp_across.md` and leaves the banked `tables.md` untouched):

```bash
cd experiments   # MODAL_PROFILE=chromatic
D=/v16_s2_L6_m4_distinct/logit_reading
modal volume get rhm-scaling-data $D/traj_a1_s42    <dir>/traj_a1_s42
modal volume get rhm-scaling-data $D/traj_eps01_s42 <dir>/traj_eps01_s42
python -m rhm.logit_reading.striatum.analyze <dir>/traj_a1_s42 --mlp-across \
    --extra-dirs <dir>/traj_eps01_s42 --tags a1,swap65k \
    --out rhm/logit_reading/striatum/results
```

**Reproduction gate.** 125 rows: every banked `across_level`, `across_level_clean` and
`across_level_by_j` win rate recomputed from the `.npz` against the same run's JSON, `n`
matched on every one, **0 mismatches**, largest deviation 0.0028. (The `.npz` stores the
revisions as `float32` while the banked JSON ranked `float64`, so near-ties can flip a
rank in a small stratum; the tolerance is 5e-3.) The §5 headline cells land exactly:
`swap65k` win rate 0.507 / 0.518 at the onset and 0.424 / 0.426 at `t_v`, 64k.

**The numbers**, win rate at `a = 0` — pooled, and re-ranked inside each `j` stratum and
`n`-weighted (`across_level_by_j`'s recipe, which transfers unchanged):

| cell | linear | MLP | clean-only |
|---|---|---|---|
| `a1_s42` `swap65k` `t_v` 64k | 0.424 / 0.467 | **0.462 / 0.497** | 0.426 / 0.454 |
| `a1_s42` `a1` `t_v` 64k | 0.394 / 0.422 | 0.434 / 0.440 | 0.406 / 0.428 |
| `a1_s42` `swap65k` `t_v` 24k | 0.438 / 0.476 | 0.440 / 0.491 | 0.457 / 0.479 |
| `a1_s42` `swap65k` `t_v` 8k | 0.435 / 0.473 | 0.440 / 0.485 | 0.441 / 0.468 |
| `a1_s42` `swap65k` onset 64k | 0.507 / 0.509 | 0.496 / 0.506 | 0.518 / 0.510 |
| `a1_s42` `a1` onset 64k | 0.513 / 0.503 | 0.494 / 0.484 | 0.514 / 0.500 |
| `eps01_s42` `swap65k` `t_v` 64k | 0.384 / 0.508 | 0.403 / 0.497 | 0.414 / 0.509 |
| `eps01_s42` `a1` `t_v` 64k | 0.396 / 0.533 | 0.434 / 0.493 | 0.417 / 0.509 |

At `a = 1` the picture is the same (`swap65k` `t_v` 64k: linear 0.463, MLP 0.430).

**Facts.**

1. The MLP critic does not move the contrast across the null anywhere. At `t_v` it sits at
   0.40–0.46 pooled and 0.44–0.50 within `j`; at the onset it is at 0.48–0.51 and, in four
   of the six onset cells, *below* the linear critic. The extra 0.08 of damage AUC the MLP
   buys on the scalar revision buys nothing on the per-level pattern.
2. **The banked "slightly backwards" reading at `t_v` is a scale artefact of pooling
   across edit widths and detection depths.** `across_level` ranks within level but over
   whatever row set it is given, and both `j` (the edit's width) and `k*` (the depth at
   which it becomes Bayes-detectable) move the *magnitude* of the revision, so the pooled
   row ranks a wide or deep edit's large revisions against a narrow one's. Re-ranking
   inside the cell, at `a = 0`, `t_v` (added 2026-09-17 as the summary table at the top of
   `tables_mlp_across.md`):

   | cell | critic | pooled | by `j` | by `(j, k*)` | coverage |
   |---|---|---|---|---|---|
   | `a1_s42` `swap65k` 64k | linear | 0.424 | 0.467 | **0.482** | 0.99 |
   | `a1_s42` `swap65k` 64k | MLP | 0.462 | 0.497 | **0.502** | 0.99 |
   | `a1_s42` `swap65k` 64k | clean-only | 0.426 | 0.454 | 0.469 | 0.99 |
   | `a1_s42` `swap65k` 24k | linear | 0.438 | 0.476 | 0.499 | 0.99 |
   | `a1_s42` `swap65k` 8k | linear | 0.435 | 0.473 | 0.483 | 0.99 |
   | `eps01_s42` `swap65k` 64k | linear | 0.384 | 0.508 | **0.518** | 0.99 |
   | `eps01_s42` `swap65k` 64k | MLP | 0.403 | 0.497 | 0.491 | 0.99 |
   | `a1_s42` `a1` 64k | linear | 0.394 | 0.422 | 0.446 | 0.93 |
   | `a1_s42` `a1` 64k | MLP | 0.434 | 0.440 | 0.452 | 0.93 |

   `coverage` is the share of the pooled row set the surviving `(j, k*)` cells retain (a
   30-row floor); at `t_v` it is 0.93–0.99, so this is essentially the same rows. On
   `swap65k`, the well-powered venue, controlling width and depth moves the linear critic
   from 0.424 to 0.482 and the MLP to 0.502 — the backwards tilt dissolves and what is
   left is a null. On the ε-trained checkpoint it goes from 0.384 to 0.518, i.e. past the
   null. The `a1` venue does not quite reach 0.5 (0.446 / 0.452) on 465 rows.

   The onset cells are **not** read this way: their `(j, k*)` coverage is 0.54–0.84,
   because the onset row set spreads over `etype` as well, so the stratified row set there
   is not the pooled one.
3. The clean-only critic tracks the trained one throughout, so nothing here is what
   exposure to damage bought.

---

## (b) A second and a third trajectory seed, and norm's cells on them

**What it tests.** Every number in the norm round was a single-seed number, on
`traj_a1_s42`. `traj_a1_s43` (`seed 43`, `data_seed 8`) and `traj_a1_s44` (`seed 44`,
`data_seed 9`) hold the grammar fixed (`rule_seed 0`, `alpha 1.0`, `weight_seed 1`), so
the stimuli, the parse, the glitch world's clean-prefix legal law and the diets'
`(etype, j)` cell histograms are the ones the banked round used and **only the model
moves**. Same ladder, same cells, same diets, same reduction.

**Commands of record** are at the top of
[`../norm/results/tables_s43.md`](../norm/results/tables_s43.md) and
[`../norm/results/tables_s44.md`](../norm/results/tables_s44.md); the three-seed
side-by-side is §0b of the latter. Wall clock per seed: 38 min for the trajectory on one
L4, ~12 min for the banked coeruleus head, ~6 min for the three `_hexcess` columns,
~10 min for norm's six main cells plus the three glitch cells.

`norm/analyze.py` gained `--banked-actor` (the `striatum/` actor gate does not apply across
seeds; `none` drops the comparison rows, or another trajectory's own column can be
supplied) and `--outfile` (so a further seed never overwrites a banked file). Both default
to the banked behaviour.

**The substrate is matched across all three.** Actor clean held-out accuracy at 64k,
`s42 / s43 / s44`: 0.904 / 0.904 / 0.904 (ℓ=1), 0.815 / 0.815 / 0.816, 0.705 / 0.704 /
0.706, 0.524 / 0.515 / 0.523. At 8k: 0.893 on all three at ℓ=1. So the disagreements below
are seed variation, not a worse model.

### What replicates on all three

1. **The norm calibrates to the world, rank one.** ρ of `V_pre` against the diet's
   `E[outcome]` is **+1.00 inside every family, at every level, on every seed**; over all
   eleven diets 0.85–0.99. Per-diet norm levels agree to ~0.01 (`out_lo / out_mid /
   out_hi` at `a1` `t_v` 64k, ℓ=4: 0.343 / 0.334 / 0.342, 0.485 / 0.475 / 0.486, 0.621 /
   0.612 / 0.622).
2. **A shift at init, a rescaling once trained.** Regressing each diet's norm on `full`'s,
   ℓ=1, `a1` `t_v`: at step 0 slopes 0.938–1.012 and sd ratios 0.957–1.030 on all three;
   at 64k `out_lo` 1.161 / 1.176 / 1.164 and `out_hi` 0.816 / 0.787 / 0.792, sd ratios
   1.184 / 1.201 / 1.186 and 0.838 / 0.810 / 0.816.
3. **The outcome surprise orders inversely with the world's expectation** at every level
   on every seed (ℓ=2, `out_lo/mid/hi`: −0.221 / −0.249 / −0.243, −0.272 / −0.298 /
   −0.295, −0.306 / −0.333 / −0.334).
4. **The response does not order with the world**, and the `out` family's ordering is
   stable everywhere.
5. **The twin optimism at ℓ = 1–3 at 64k on `swap65k`**, the round's §3 headline: `+++`
   with all three seeds past 2σ at every one of the three levels (ℓ=1 +2.7 / +3.5 / +3.6;
   ℓ=2 +4.5 / +3.9 / +3.8; ℓ=3 +4.9 / +3.7 / +3.5). This is the round's most robust
   twin row.
6. **Step 0 is nil on all three**, both worlds, every level.
7. **The glitch addendum's persistence dissociation.** `dV` by offset, ℓ=2, 64k: the edit
   world reads +0.066 / +0.066 / +0.041 at offset 1 and stays positive through offset 5 on
   all three; the glitch world reads −0.003 / −0.003 / −0.005 at offset 1 and is nil or
   negative throughout on all three. Sign agrees 3/3 at every offset 1–5 in both worlds.

### The two contested claims, stated across three seeds

**(i) The twin `dR` at ℓ = 4: the sign does not agree, at either checkpoint.**
`swap65k`, each world's own full pair set, `s42 / s43 / s44`:

| step | world | `dR` (sig) | verdict |
|---|---|---|---|
| 8k | edit | −0.0028 (−2.0) / +0.0066 (+5.0) / +0.0082 (+6.0) | `-++`, **sign split** (2+ / 1− at 2σ) |
| 64k | edit | +0.0043 (+2.0) / −0.0042 (−2.3) / −0.0026 (−1.4) | `+--`, **sign split** (1+ / 1− at 2σ) |

Not only do the seeds disagree — the two checkpoints disagree with each other about which
sign the majority takes (8k majority +, 64k majority −). The same instability shows at
`a1` 64k (`+--`, +1.5 / −2.5 / −1.1). **Nothing about ℓ = 4's twin sign is established.**

**(ii) The glitch world's event-level reading: it holds at 8k and fails at 64k.**
Each world's own full pair set, `s42 / s43 / s44`:

| step | world | ℓ | `dR` (sig) | verdict |
|---|---|---|---|---|
| **8k** | glitch | 1 | +0.0075 (+2.7) / +0.0183 (+5.6) / +0.0006 (+0.2) | `+++`, + on 2/3 at 2σ |
| **8k** | glitch | 2 | +0.0172 (+6.0) / +0.0119 (+3.9) / +0.0046 (+1.5) | `+++`, + on 2/3 at 2σ |
| **8k** | glitch | 3 | +0.0117 (+3.8) / +0.0130 (+4.4) / +0.0066 (+2.4) | `+++`, + on 3/3 at 2σ |
| **8k** | glitch | 4 | +0.0056 (+3.0) / +0.0052 (+2.9) / +0.0024 (+1.3) | `+++`, + on 2/3 at 2σ |
| 64k | glitch | 1 | +0.0025 (+0.6) / −0.0018 (−0.4) / −0.0064 (−1.3) | `+--`, none at 2σ |
| 64k | glitch | 2 | +0.0192 (+5.1) / −0.0068 (−1.8) / −0.0102 (−2.3) | `+--`, **sign split** |
| 64k | glitch | 3 | +0.0177 (+4.2) / +0.0061 (+1.4) / −0.0098 (−2.2) | `++-`, **sign split** |
| 64k | glitch | 4 | +0.0072 (+2.2) / −0.0032 (−1.0) / −0.0232 (−6.4) | `+--`, **sign split** |

The edit world over the same cells is `+++` with 3/3 past 2σ at ℓ = 1–3 at **both** 8k and
64k. So:

- **At 8k the addendum's claim holds.** Both worlds are positive at every level on every
  seed — sign agrees 3/3 in twelve of twelve world × level cells — on 2300–4300 pairs.
- **At 64k it does not.** The glitch world is positive on `s42` alone; both new seeds are
  negative, reaching −2.3σ (ℓ=2), −2.2σ (ℓ=3) and −6.4σ (ℓ=4) on `s44`. The banked
  addendum quoted 64k.

**A methodological note that falls out of this.** The addendum's headline table uses the
*common-window* subset (300–1500 pairs), which is noisier still: on `s43` the edit world
reads +0.0105 (+3.9σ) at ℓ=2 on its own 2593 pairs but −0.0039 (−0.7σ) on the 524 common
windows — same seed, same checkpoint, same critic. Where the two subsets disagree, the
own-pairs number is the better-powered one.

> The norm itself — its calibration to the world, its shift-to-rescaling with training, and
> the outcome surprise that restates it — is seed-robust to three decimal places on three
> seeds. The twin contrast is seed-robust at ℓ = 1–3 on the well-powered venue and **not**
> at ℓ = 4, where the sign splits at both checkpoints and the two checkpoints' majorities
> point opposite ways. The glitch-world addendum's *persistence* dissociation is
> seed-robust 3/3. Its *event-level* claim — that the optimism is the same in a world where
> re-parsing is wrong — holds at 8k on all three seeds and **fails at 64k**, where it rests
> on the banked seed alone against two new seeds of the opposite sign.

---

## (c) A mean-and-variance critic

**What it tests.** Xiang, Lohrenz & Montague's insula row is a *variance* prediction error
with a U-shaped response to surprise in either direction, and the norm round's "does not
establish" list named it as having no analogue in a mean-only ridge. On a 0/1 outcome a
variance head is degenerate — `Var = V(1 − V)`, a function of the mean — so a continuous
target is needed. Two were tried, both already in the machinery:

- **`hsum`**: `y_l(t) = Σ_{δ=0..8} o_l(t + δ)`, the outcome summed over the horizon.
  Mean-head held-out R² **0.03–0.06**. Dropped.
- **`lmean`**: `y(t) = mean_{ℓ=1..4} o_ℓ(t)`, the actor being right at each of the four
  absorbed levels at one position, in {0, .25, .5, .75, 1} — the per-position version of
  norm's `out_win` cut variable. Mean-head held-out R² **0.20–0.27**, comparable to the
  banked critic's `l1_d0` 0.21. Carried.

`norm/task.py` gained a `var_target` knob (default `""` = off). When on it fits, on the
same cached states, the same λ ladder and the same val rows: a **mean head** `Vc[ℓ, d]` on
the continuous target, then a **variance head** `Vvar[ℓ, d]` on that head's own squared
training residual. At the event, `dc = y − Vc_pre` and `vpe = dc² − Vvar_pre`.

**Command of record:**

```bash
cd experiments   # MODAL_PROFILE=chromatic
D=/data/v16_s2_L6_m4_distinct/logit_reading
modal run -m rhm.logit_reading.striatum.norm.task::norm_sweep \
    --cells "$D/traj_a1_s42/step064000.pt:swap65k:0,$D/traj_a1_s42/step000000.pt:swap65k:0,$D/traj_a1_s42/step064000.pt:a1:1" \
    --var-target lmean --max-twins 0 --tag var
python -m rhm.logit_reading.striatum.norm.analyze <dir>/traj_a1_s42 \
    --mode varhead --addendum-tag var --outfile tables_varhead.md \
    --out rhm/logit_reading/striatum/norm/results
```

**Reproduction gate on the edit.** The banked `swap65k` 64k cell re-run with the knob off
(`--tag reprod`) produces a JSON **identical on all 250 leaves and on the key set** to the
banked `step064000_norm_swap65k.json`. The edit is inert when off.

**Fits.** Held-out R², `lmean`:

| cell | R² mean (d=0 / d=1) | R² var | R² var given the best quadratic in the mean |
|---|---|---|---|
| `a1` 64k | 0.270 / 0.228 | 0.098 / 0.070 | 0.062 / 0.046 |
| `swap65k` 64k | 0.258 / 0.211 | 0.091 / 0.058 | 0.054 / 0.036 |
| `swap65k` step 0 | 0.054 / 0.040 | 0.006 / 0.005 | 0.002 / 0.001 |

**Facts.**

1. **The variance target is not degenerate and training buys it.** The variance head's
   held-out R² beats the best quadratic in its own mean head's output by a factor of
   ~1.6 at 64k, and collapses to 0.005 at random init. So the state does carry information
   about the critic's squared error beyond what its mean prediction implies.
2. **But on the violation rows the head is flat.** Across seven equal-count bins of the
   signed surprise `dc`, `Vvar_pre` moves by at most 0.01; its U statistic (outer two bins
   minus the middle bin) is −0.004 to +0.001.
3. **So the U in `vpe` is exactly the U in `δ²`.** `swap65k` 64k: `vpe`'s U is 0.2137 at
   the onset against a constant-variance baseline's 0.2095, and 0.2222 at `t_v` against
   0.2221. The learned variance absorbs none of it. The `|dc|` ordering is monotone
   (`swap65k` `t_v` 64k: −0.066 / −0.052 / −0.006 / +0.084 / +0.329 over quintiles) and is
   the same ordering `δ²` alone gives.
4. **`corr(Vvar_pre, dc²)` flips sign with training**: +0.079 / +0.084 at step 0 (onset /
   `t_v`) to −0.173 / −0.058 at 64k. Where the trained critic is most confident is where
   the violation hurts most, so its expected squared error is *anti*-correlated with the
   squared surprise it is about to receive on these rows.
5. **The variance channel does not respond to the event.** `Rvar = Vvar(s_t) − Vvar(s_{t−1})`
   is −0.002 pooled, and at the `a1` onset, where an unedited control exists, it reads
   −0.0006 (swap) / −0.0013 (rare) / **−0.0023 (none, unedited)** — the unedited rows carry
   the largest magnitude, so there is no violation-specific variance response at all. The
   mean head's revision `Rc` over the same rows reads −0.0269 / −0.0165 / −0.0130.

> A variance head is learnable from this state and carries something the mean does not;
> it just does not carry Xiang's row. On the fixed violation rows its prediction is flat in
> the surprise, it does not move at the event, and the U shape in the variance prediction
> error is the arithmetic U of `δ²` with a constant subtracted.
