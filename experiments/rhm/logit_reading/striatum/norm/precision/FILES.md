# Files — logit_reading/striatum/norm/precision

**Parent**: [../FILES.md](../FILES.md) · **Grandparent**: [../../FILES.md](../../FILES.md) ·
**Tables**: [results/tables_20260922.md](results/tables_20260922.md)

No `README.md`: this node is facts only until the reduction has been discussed.

**What the node asks.** Every value-side contrast in the arc is taken at **matched surprisal** —
`junction/analyze.py`'s matcher pins the model's own surprisal at the anchor (0.500 on every banked table) and
the same-prefix twins are selected at a surprisal caliper. None of them has asked whether the reader depends on
`H_pre = H(q_(t0-1))`, the entropy of the trunk's own forecast of the token AT the anchor, made one position
before it. `norm/task.py` has recorded that column on every fixed row since the round was run and no banked table
had used it. The direct behavioural analogue is the surface the music literature measures — pleasure over
(information content × the entropy of the context before the event), whose interaction's sign is disputed
(`reading/Musical pleasure striatum and cerebellum.md`[^private]
§"Pleasure tracks a surprise-by-uncertainty saddle that cortex computes").

**One axis, two names.** `excess_e = nll_e − H_pre` is on file and the identity is checked per cell, so at
matched surprisal the `H_pre` axis is the negative excess axis. `logit_reading/README.md` §2b found `s − H(q)` a
null as a scalar for detection and `striatum/README.md` §5's addendum found the excess head carries nothing
conditioned on the value level; every interaction fit in the tables is therefore printed in both
parameterisations, and every response slope is printed with and without `V_pre` in the controls, so that one
object is not reported as two findings.

## Code files

| file | purpose |
|---|---|
| `reduce.py` | The whole reduction, CPU only, nothing refit and no GPU touched. Reads the banked `norm/` `.json`/`.npz` (plus `hpre.py`'s one recomputed column) and writes `results/tables_<date>.md` and `figs/prec_*.png`. Sections: **0** integrity (every member of every fetched npz decompressed and summed — an earlier round's fetch returned a full-size file with a bad CRC that `np.load` alone did not catch); **0b** reproduction gates against `norm/README.md`'s own published columns (matched cell sizes, `AUC(-R, flip)`, `AUC(-V, flip)`, the token-balanced mean `dR` and the `dR ~ V_pre` slopes); **1** the axis's own guards (range, quantiles, what it correlates with, mean by `k*` / `j` / position parity, and the variance decomposition against the matcher's strata); **2** the level `V_pre ~ H_pre` raw / partial / per-sd, binned, per diet family (does the entropy dependence rescale with the world the way `norm/README.md` §1's calibration did), all seeds side by side; **3** the response at matched surprisal — `R` and `R − Ro` in `H_pre` terciles inside `junction`'s own matched sets with the realised-damage AUC per bin, the partial slope on `H_pre` controlling `(k*, j, t0, nll_e)` and again with `V_pre` added, the `z(s) + z(H) + z(s)·z(H)` interaction fit in both parameterisations, and the 2-D `(surprisal × entropy)` surface with counts; **4** the outcome surprise `δ = outcome − V_pre` by `H_pre` with its two parts separated, and on the same grid; **5** the twins — `dR` binned by `H_pre` (pairs the unit, token-balanced primary), the slope residualised on `sec_twin_kstar`'s own control set, the `(s_v × H_pre)` surface, and `norm/README.md` §3's `dR ~ V_pre` slope with `H_pre` partialled out. Reuses `junction/analyze.py`'s `Rows`, `striatum/analyze.py`'s `_auc` / `match_sign_balanced` / `tbl` / `fmt` and `norm/analyze.py`'s `Twins` / `order_diets` / `_ols` / `_spear` / `_sem` unchanged; the matcher and the `Cell` loads are memoised (deterministic, so no number moves). `--figs-only` regenerates the figures alone. |
| `hpre.py` | Modal, L4, ~8 s per cell. `H_pre` is recorded on the fixed rows but **not** on the twin pairs: `norm/task.py`'s `trace(..., want_lsm=True)` computes the log-softmax at `t_v − 1` for exactly those windows and keeps only the two surprisals. Route (i), the join of `tw__w` against the `tv__` fixed rows, covers only the test-split quarter of the pairs (643 token-balanced at `swap65k` 64k, too thin for the surface). `hpre_cell` closes the gap with one forward pass that reproduces nothing else: the pair set, the selection and the critic are read off the banked npz, and only the entropy at `t_v − 1` is recomputed. Writes `<stem>_norm_<tag>_Hpre.npz` (`tw_w`, `tw_t_v`, `tw_H_pre`, `tw_s_v`, plus `gl_*` when the cell carries the glitch world, whose prefix is the clean one so the join cannot supply it at all). **Gate**: the recomputed violator surprisal is compared to the banked `tw__s_v` on every pair and asserted under `--tol` — measured max 1.2e−05 over all 18 cells, float32 storage. `hpre_sweep` is the CPU coordinator, one container per cell. |
| `mediate.py` | Modal, L4, 25-50 s per cell. The **mediators of the event**, recorded on exactly the banked rows and pairs: `kl_update = KL(q_t \|\| q_(t-1))` (`phasic.py`'s object — the model's forecast movement across the event), `H_next = H(q_t)`, `dstate_<b> = \|\|s_t − s_(t-1)\|\|₂` at `post_block7` (the block `norm/task.py` fits the critic on), `post_block3` and `post_embed`, and for every pair `dstate_pair_<b> = \|\|s_t^flagged − s_t^twin\|\|₂`, the state difference whose linear projection is `dR`. Covers the fixed rows at **both** anchors, the `tw__` pairs (flagged stream = the edited window, twin stream = the same window with `x_c` substituted) and the `gl__` pairs (both streams built from the **unedited** window, which is why the glitch world's `H_pre` cannot come from any join against the fixed rows). Nothing is refit: the rows, the pair sets, the selection and the critics are read off the banked npz. **Gates**, asserted before anything is written: the recomputed `H_pre` and anchor surprisal must equal the banked `<an>__H_pre` / `<an>__nll_e` on every fixed row; the recomputed flagged and twin surprisals must equal `s_v` / `s_c` on every pair; and the two members of a pair must share `H_pre` exactly. Writes `<stem>_norm_<tag>[_<tag2>]_med.npz`. `mediate_sweep` is the CPU coordinator (`--tag glitch` for the `_glitch` cells). |
| `mediation.py` | The reduction for both follow-ups, CPU only; imports `reduce.py`'s `Cell` / `matched_sets` / `_partial` / `_qbin` / `_grid` and `norm/analyze.py`'s `Twins` / `common_windows` unchanged, so the rows and matched sets are the same objects `tables_20260922.md` used. **Part A** (`tables_mediation_<date>.md`): the gate table; whether each mediator depends on `H_pre` at matched surprisal and what it ranks on the matched rows; the **collinearity guard** (`H_next` is a level on `H_pre`'s own scale, not a movement, so it must be read separately from the movement mediators); the partial slope `R ~ H_pre` with each mediator added to the controls one at a time, movement-only, all together, and with `V_pre`; the arc's both-ways conditional (`R`'s entropy slope inside bins of the mediator and the mediator's inside bins of `R`); the `z(s) × z(H)` interaction with the mediators controlled; the 2-D surface of `R` residualised on the mediators; `AUC(H_pre, flip)` conditioned on the mediators and the reverse; and the twins with `dstate_pair` and the pair's `kl` difference in the control set. **Part B** (`tables_glitch_<date>.md`): the across-world entropy guard first, then the two worlds' pair populations, `dR` by `H_pre` bin on each world's own pairs and on the common windows, the residualised slope per world, and the persistence `dV` by offset inside `H_pre` terciles. |
| `express.py` | Modal, L4, ~100 s (`a1`) to a few minutes (`swap65k`) per cell. **The expressivity test**: refits the critic on the SAME trunk, the SAME `full` diet rows (and the clean-only critic's), the SAME split, lambda ladder and held-out per-column lambda selection as `norm/task.py`, with the feature map augmented, and writes the same per-row columns on the identical `split == 2` test rows. Arms, named as critics: `full` (arm 0, the banked ridge refitted — **the gate**), `fullH` (state + `H(q_t)`, primary), `fullQ` (state + `log q_t`, 16 dims, the control for `fullH`), `fullN` / `fullT` (state + `\|\|s_t\|\|` / one fixed random `tanh` projection, placebos), `fullP` (state + `H(q_t)` + `H(q_(t-1))` + `s_t` + `H(q_(t-1)) s_t`, the precision-weighted form), `fullM` (a 128-unit MLP on the state, levels 1–4 at `a = 0`); and `clean`, `cleanH` … `cleanP` for the exposure control. Appended columns are affinely standardised to the state's per-dimension RMS on the training rows by transforming the Gram exactly, so the shared ridge penalty treats each like an average state direction; the state columns and the bias are untouched. Recomputes every base column of the banked npz. **Gates**, asserted before the cell returns: actor accuracy, arm-0 held-out `R2` on every column, the test rows and every outcome label (exact), `nll_e` / `H_pre`, arm 0's `V` / `Vpre` / `R` / `Ro` / `Vpreo` / `Rsh` against the banked `full` and `clean` columns, and the twins' arm-0 `V` at offsets 0 and −1. Writes `<stem>_norm_<tag>_express.{json,npz}`. `express_sweep` is the CPU coordinator. **Extended 2026-09-22, backward-compatibly** (defaults reproduce the committed cells; 456 shared columns checked bit-identical): `--map-arms L,Lp,S` adds the feature-MAP arms of follow-up 4 — `fullL` (`ln_f(s)`, the model's own final layer norm, in place of the state: exactly what the unembedding reads), `fullLp` (`[s, ln_f(s)]`, the ln block scaled as a block to the raw state's per-dimension RMS), `fullS` (affine-free per-row standardisation, in place) and the same on the clean critic; `--arm-set` selects the augmented arms; `--out-sfx` names the output; `--bank-tag` runs a venue with no banked `norm/` cell, where the actor's clean accuracy against the named venue's json is the only gate (the actor is trained on clean windows only); `--do-probes` adds junction's legality ceiling (`probe_swap_ed`, `probe_swap_mlp`) and the same probe on the input embedding at the event (`probe_swap_emb`). **Extended 2026-09-23 (follow-up 6, the `+logits` arm), backward-compatibly** (defaults reproduce every committed cell; the committed block's Gram is untouched): `--logit-arms Z,LE,LX,E,ZE` adds a SEPARATE appended block from the model's raw logits `z` (`lm_head` has no bias, so `z = W_U ln_f(s)` exactly), its `lse = logsumexp(z)` and `max(z)`, one function (`_z_cols`) for the fit rows, the anchors and the twins, with two Grams of its own (`[s, z, lse, max, 1]` and `[ln_f(s), lse, max, 1]`), each appended column standardised through its Gram to the base block's per-dimension RMS by `_standardiser` and mapped back (`_fit_logit_arms`): `fullZ` (`s` + 16 logits), `fullLE` (`ln_f(s)` + `lse`), `fullLX` (`ln_f(s)` + `max z`), `fullE` (`s` + `lse`), `fullZE` (`s` + `z` + `lse`), and the same on the clean critic. It also writes the **span diagnostic** into the json (`span`): held-out `R2` of each logit on `[s]` and of `lse` / `max z` on `[s]`, `[s, z]`, `ln_f(s)`, same ladder and selection. `--check-sfx _express,_express_ln` compares, in-container and before writing, every column the cell shares with the named committed cells (base columns, arm 0, and the refitted `fullH` / `fullQ` / `fullL` / `fullLp`, both anchors and the twins) and gates on it at `tol`. The anchor loop now builds each arm's feature matrix once per stream instead of once per target column (the same array and the same `F @ b[:-1] + b[-1]`, so no number moves; with 20 arms the per-column rebuild dominated the cell). |
| `express_read.py` | The reduction of `express.py`, CPU only; reads the express npz through `reduce.py`'s own `cell_of` / `matched_sets` (the base columns are gated equal, and the matched sets are asserted identical to the banked ones). E0 gates; E1 held-out fit per arm; E2 each arm's `V_pre` slope along the axis beside the realised outcome's, and `delta ~ H_pre`; E3 `R ~ H_pre` at matched surprisal with and without the arm's own `V_pre`, the `z(s) x z(H)` interaction, and `AUC(-R, flip)` / `AUC(-V, flip)` pooled and by `H_pre` tercile; E4 `AUC(H_pre, flip)` conditioned on the arm's `V` and `R` and the reverse, for the `full` arms and the clean-only arms; the twins on each arm; E5 legality at matched consequence (`a1`, `fd`). `--preamble logits` (2026-09-23) adds, after E0, the shared-column gate against the committed cells (`sec_shared`), section **H** (`sec_headline`: README §8's five columns, one row per arm, 64k and 8k, ℓ = 1-2, each venue and anchor, with the marginal `AUC(H_pre, flip)` as each block's first row; the same functions and matched rows as E1/E3/E4) and section **S** (`sec_span`, the span diagnostic from the jsons). |
| `legality.py` | Follow-up 5's reduction, CPU only: **does the arc's law survive a reader that sees the belief?** Reads `express.py`'s refits on the `legal65k` venue at the onset anchor `fd`. Swap against rare among consequential episodes, matched two ways because the two cannot be pinned together (`cons_depth` is `j` for a swap and `j − 1` for a rare edit): `jt` pins the edit's width (junction's recipe) and `ct` pins consequence depth, each with exact strata on `(pinned, t0 // 4)`, 1:1 nearest neighbour on surprisal at caliper 0.30 (and 0.10 as the guard-slip replicate), sign-balanced inside `\|ds\|` bands. Prints every guard (surprisal, position, `j`, consequence depth, an input-embedding probe), the readers' `AUC(-R, swap)` / `AUC(-V, swap)` beside the oracle legality ceiling, the unmatched contrast, the reverse guard (each reader's damage AUC on the same rows) and whether legality predicts the outcome on this venue beyond consequence and surprisal. |

## Artefacts

| path | what |
|---|---|
| `README.md` / `CONVERSATION.md` | The super-writeup for the nine rounds (§1–§10, what it establishes and does not, corrections to banked nodes) and the session behind it, Jasper's prompts verbatim with the orchestrator's responses in summary and the open items at the end. |
| `results/tables_20260922.md` | Every number, facts only. Three trajectory seeds (`traj_a1_s4{2,3,4}`) × two venues (`a1`, `swap65k`) × two anchors (`fd`, `tv`) × three checkpoints (0 / 8k / 64k), seeds side by side and never averaged. |
| `results/tables_clean_20260922.md` | The same reduction with the **clean-only critic** (the one that has never seen an edit) in place of `full`, `swap65k` `t_v` only — the exposure control the arc asks of every value-side reading. Same three seeds, same rows, same matched sets. |
| `figs/prec_axis_<tag>_<anchor>.png` | The axis and its confounds: the `H_pre` histogram against the uniform ceiling, mean surprisal by `H_pre` bin, mean `H_pre` by `k*`, mean `H_pre` by position (the two-position sawtooth). |
| `figs/prec_level_<tag>_<anchor>.png` | `V_pre` against `H_pre` in eight quantile bins, per level, the three checkpoints on one pair of axes with the random-init trunk in grey. |
| `figs/prec_surface_<tag>_<anchor>.png` | The response's 2-D `(surprisal × pre-event entropy)` surface, 3 checkpoints × 4 levels, cell text = n, each panel on its own colour scale. |
| `figs/prec_delta_<tag>_<anchor>.png` | The outcome surprise by `H_pre` bin per level, with the outcome alone dotted beside it. |
| `figs/prec_twins_<tag>_<step>.png` | `dR` by `H_pre` bin per level with all three seeds on one axis, plus the `(s_v × H_pre)` surface of `dR`. |
| `results/tables_mediation_20260922.md` | Part A, facts only: what carries the response's entropy dependence. Three seeds × two venues × two anchors × three checkpoints. |
| `results/tables_glitch_20260922.md` | Part B, facts only: the entropy axis in the glitch world beside the edit world, from the `_glitch`-tagged cells; the across-world entropy guard is section B0 and comes before any cross-world row. |
| `figs/prec_med_<tag>_<anchor>.png` | Each mediator by `H_pre` bin, and `R` by `H_pre` bin raw beside `R` residualised on the five mediators. |
| `figs/prec_worldsH_<step>.png` | `dR` by `H_pre` bin per level in both worlds with all three seeds, plus the persistence `dV` by offset inside `H_pre` terciles per world. |
| volume `.../step*_norm_<tag>[_glitch]_med.npz` | `mediate.py`'s event columns, in the banked rows' and pairs' own order, with the gates in the file's `meta`. |
| `results/tables_express_20260922.md` | The expressivity test, facts only: every arm beside the refitted banked ridge on identical rows, three seeds, both venues, both anchors, three checkpoints. |
| `figs/prec_express_fit_<tag>.png` | Held-out `R2` of `V[l, 0]` per arm and level, three checkpoints, three seeds as markers. |
| `figs/prec_express_axis_<tag>_<anchor>_<step>.png` | Top: each arm's `V_pre` against the realised outcome along the `H_pre` axis; bottom: each arm's `R` on the matched rows by `H_pre` bin. |
| volume `.../step*_norm_<tag>_express.{json,npz}` | `express.py`'s refits: every arm's per-row columns on the banked test rows and the twins' arm columns at offsets 0 / −1, with every gate in the json. |
| `results/tables_express_ln_20260922.md` | Follow-up 4, facts only: `ln` / `+ln` / `std` beside the ridge and `+log q` on identical rows — is the belief's shape just normalisation? |
| `results/tables_legality_20260922.md` | Follow-up 5, facts only: legality at matched consequence and surprisal on the `legal65k` venue, per reader, with every guard, the ceiling, the reverse guard and the outcome regression. |
| `figs/prec_express_ln_{fit,axis}_*.png`, `figs/prec_legality_{jt,ct}.png` | The two follow-ups' figures. |
| volume `.../stimuli_legal65k.{npz,json}`, `parse_legal65k.{npz,json}` | The legality venue: 65,536 windows, half swap and half legal-but-rare, no `none`, seed 2028. |
| volume `.../step*_norm_{a1,swap65k}_express_ln.{json,npz}`, `.../step*_norm_legal65k_express.{json,npz}` | Follow-up 4's and 5's refits. |
| `results/tables_logits_20260923.md` | Follow-up 6, facts only: the log-partition or the ridge's shrinkage? `+logits`, `ln +lse`, `ln +max`, `+lse`, `+logits +lse` beside the refitted ridge, `+H`, `+log q`, `ln`, `+ln` on identical rows; section H is README §8's table with the new arms, section S the span diagnostic. Three seeds, both venues, both anchors, three checkpoints. |
| `figs/prec_express_logits_{fit,axis}_*.png` | Follow-up 6's figures (the E1 fit per arm; each arm's `V_pre` and matched `R` along the axis, s42). |
| volume `.../step*_norm_{a1,swap65k}_express_logits.{json,npz}` | Follow-up 6's refits (18 cells, 80-160 MB each on `swap65k`), every gate and the span diagnostic in the json. |
| volume `.../traj_a1_s4{2,3,4}/step{000000,008000,064000}_norm_{a1,swap65k}_Hpre.npz` | `hpre.py`'s per-pair `H_pre` column, in the banked pairs' own order, with the recomputed `s_v` gate. |

PNGs are gitignored under `experiments/`; the generating code is `reduce.py`.

## Reproduction

```bash
cd experiments            # MODAL_PROFILE=chromatic
# route (ii): H_pre for every banked twin pair, 18 cells, ~8 s each on an L4 (~2.5 GPU-min total)
modal run -m rhm.logit_reading.striatum.norm.precision.hpre::hpre_sweep

# fetch the banked cells and the new column (there is no local mirror)
D=/v16_s2_L6_m4_distinct/logit_reading
for s in 42 43 44; do for st in 000000 008000 064000; do for tg in a1 swap65k; do
  for f in norm_${tg}.npz norm_${tg}.json norm_${tg}_Hpre.npz; do
    modal volume get rhm-scaling-data $D/traj_a1_s$s/step${st}_$f <dir>/traj_a1_s$s/step${st}_$f
done; done; done; done

# the tables and the figures (CPU, ~3 min over the fetched files)
python -m rhm.logit_reading.striatum.norm.precision.reduce \
    <dir>/traj_a1_s42 <dir>/traj_a1_s43 <dir>/traj_a1_s44 \
    --tags swap65k,a1 --anchors tv,fd \
    --out rhm/logit_reading/striatum/norm/precision/results \
    --figs rhm/logit_reading/striatum/norm/precision/figs \
    --outfile tables_20260922.md

# the exposure control: the same reduction on the clean-only critic
python -m rhm.logit_reading.striatum.norm.precision.reduce \
    <dir>/traj_a1_s42 <dir>/traj_a1_s43 <dir>/traj_a1_s44 \
    --tags swap65k --anchors tv --diet clean \
    --out rhm/logit_reading/striatum/norm/precision/results \
    --outfile tables_clean_20260922.md

# the mediators (27 cells: 18 banked + 9 `_glitch`), ~25-50 s each on an L4
modal run -m rhm.logit_reading.striatum.norm.precision.mediate::mediate_sweep
modal run -m rhm.logit_reading.striatum.norm.precision.mediate::mediate_sweep \
    --tags swap65k --tag glitch
# ... fetch `<stem>_med.npz` beside the rest, and the `_glitch` cells themselves, then
python -m rhm.logit_reading.striatum.norm.precision.mediation \
    <dir>/traj_a1_s42 <dir>/traj_a1_s43 <dir>/traj_a1_s44 \
    --tags swap65k,a1 --anchors tv,fd \
    --out rhm/logit_reading/striatum/norm/precision/results \
    --figs rhm/logit_reading/striatum/norm/precision/figs --date 20260922

# the expressivity test: 18 cells (3 seeds x 3 checkpoints x a1, swap65k), one L4 each
modal run --detach -m rhm.logit_reading.striatum.norm.precision.express::express_sweep
# ... fetch `<stem>_norm_<tag>_express.{json,npz}` beside the rest, then
python -m rhm.logit_reading.striatum.norm.precision.express_read \
    <dir>/traj_a1_s42 <dir>/traj_a1_s43 <dir>/traj_a1_s44 \
    --out rhm/logit_reading/striatum/norm/precision/results \
    --figs rhm/logit_reading/striatum/norm/precision/figs --date 20260922

# follow-up 4: the feature-map arms on the same 18 cells (~80 s to a few minutes each)
modal run --detach -m rhm.logit_reading.striatum.norm.precision.express::express_sweep \
    --arm-set Q --map-arms L,Lp,S --out-sfx _express_ln --no-do-mlp
python -m rhm.logit_reading.striatum.norm.precision.express_read \
    <dir>/traj_a1_s42 <dir>/traj_a1_s43 <dir>/traj_a1_s44 --sfx _express_ln \
    --arms full,fullQ,fullL,fullLp,fullS --clean-arms clean,cleanQ,cleanL,cleanLp,cleanS \
    --preamble ln --outname tables_express_ln_20260922.md --figprefix prec_express_ln \
    --out rhm/logit_reading/striatum/norm/precision/results \
    --figs rhm/logit_reading/striatum/norm/precision/figs

# follow-up 5: the legality venue (half swap, half rare, no none), its parse, the refits
modal run --detach -m rhm.logit_reading.stimuli::build_stimuli --n 65536 --seed 2028 \
    --frac 0.5,0.5,0.0 --no-with-reference --tag legal65k
modal run -m rhm.logit_reading.striatum.parse::build_parse --n 65536 --seed 2028 \
    --frac 0.5,0.5,0.0 --tag legal65k            # asserts the replay is bit-identical
modal run --detach -m rhm.logit_reading.striatum.norm.precision.express::express_sweep \
    --tags legal65k --steps 0,8000,64000 --arm-set Q,P --map-arms L --bank-tag swap65k \
    --do-probes --no-do-twins
python -m rhm.logit_reading.striatum.norm.precision.legality \
    <dir>/traj_a1_s42 <dir>/traj_a1_s43 <dir>/traj_a1_s44 \
    --out rhm/logit_reading/striatum/norm/precision/results \
    --figs rhm/logit_reading/striatum/norm/precision/figs --date 20260922

# follow-up 6 (2026-09-23): the +logits arm and its confirming pair, 18 cells, one L4 each.
# Command of record: app ap-EJyyMXgi0jYe1pee43RKOw (chromatic), 10.8 min wall clock at
# max_containers=6, 48.7 L4-min in-cell (swap65k 195-271 s, a1 78-110 s per cell), peak RSS
# 10.5 GB against the 16 GB request; every gate passed, every shared column within 6e-08.
# Smoke: ap-2pcUe2esZItIHzIwE4Gvgg (s42 64k swap65k, --out-sfx _express_logits_smoke; its
# output was bit-identical to the sweep's cell and was removed from the volume).
modal run --detach -m rhm.logit_reading.striatum.norm.precision.express::express_sweep \
    --arm-set H,Q --map-arms L,Lp --logit-arms Z,LE,LX,E,ZE --no-do-mlp \
    --out-sfx _express_logits --check-sfx _express,_express_ln
# ... fetch `<stem>_norm_<tag>_express_logits.{json,npz}` beside the banked `norm_<tag>.{npz,json}`
# and `_Hpre.npz`, then (CPU, ~2 min)
python -m rhm.logit_reading.striatum.norm.precision.express_read \
    <dir>/traj_a1_s42 <dir>/traj_a1_s43 <dir>/traj_a1_s44 --sfx _express_logits \
    --arms full,fullH,fullQ,fullL,fullLp,fullZ,fullLE,fullLX,fullE,fullZE \
    --clean-arms clean,cleanH,cleanQ,cleanL,cleanLp,cleanZ,cleanLE,cleanLX,cleanE,cleanZE \
    --preamble logits --outname tables_logits_20260923.md --figprefix prec_express_logits \
    --out rhm/logit_reading/striatum/norm/precision/results \
    --figs rhm/logit_reading/striatum/norm/precision/figs
```

`modal volume get` writes the remote file AT the local path when that path does not exist, and refuses a
directory that does, so give it the full destination filename rather than a trailing-slash directory.

## Gotchas worth not rediscovering

- **Legality cannot be matched on width and consequence depth at once.** `junction.diets.cons_depth` is `j` for a
  swap and `j − 1` for a rare edit, so a swap and a rare edit never share both; `legality.py` reports the `jt`
  (width pinned) and `ct` (consequence depth pinned) contrasts side by side, each printing the other variable's
  imbalance as a guard.
- **The same identity makes `swap`, `j` and `c_depth` exactly collinear in any regression on edited rows**
  (`c_depth = j - 1 + swap`). `legality.py`'s outcome table (L4) controls only the variable its matched set pins;
  a first draft that controlled all three split the swap coefficient by min-norm `lstsq` and was discarded before
  the committed table was written.
- **`stimuli.build_stimuli` and `striatum/parse.py`'s `build_parse` take `--frac swap,rare,none`** (added
  2026-09-22, empty by default so every banked venue rebuilds unchanged); the parse must be given the same value
  to replay the stimulus RNG, and asserts that it did.
- **An appended column must be standardised before it meets the ridge.** `ridge_solve` penalises every
  non-bias coefficient by one `lam * trace(A) / D`, so a column on its own scale (nats, a norm near 400, log
  probabilities near −20) is crushed or favoured by the penalty for no reason. `express.py` standardises each to
  the state's per-dimension RMS by transforming the Gram exactly and maps the coefficients back.
- **The MLP arm has no lambda selection** (`train_head` runs a fixed 2000 steps, `striatum/task.py`'s setting);
  its held-out `R2` is reported, not selected on, so it is not strictly comparable to the ridge arms' selected
  `R2`.
- **`H_next` is not a movement.** It is a level on `H_pre`'s own scale (`r` printed per cell in the
  collinearity guard), so a mediation column that contains it removes the entropy slope partly for an arithmetic
  reason. The `+ movement only` column (`kl_update` and the three `dstate`) is the control set that answers the
  question the movement hypothesis asks.
- **The two worlds' `H_pre` are not matched to each other.** The edit world's prefix carries the edit and the
  glitch world's is the untouched stream, so their entropies are different distributions — the entropy twin of
  `norm/README.md` §5's surprisal caveat. Within a pair the two members share `H_pre` exactly (asserted 0.0).
- **The random-init trunk is a degenerate floor for this axis.** At step 0 the forecast is near-uniform on every
  row: `H_pre` has mean 2.711 and sd 0.017 against a ceiling of ln 16 = 2.773, so there is almost no spread to
  regress on and a large-looking slope is a tiny effect. Every slope is therefore also printed as `b · sd(H)`,
  the readout's change over one sd of the axis, which is the only column comparable across checkpoints.
- **`H_pre` and the surprisal are anti-correlated** (Spearman −0.31 at `swap65k` `t_v` 64k, −0.45 at `fd`), so
  the 2-D grid's anti-diagonal is thin — 95 rows in the high-entropy / high-surprisal corner against 1,399 in the
  low-entropy / high-surprisal one. Counts are printed in every cell of every surface.
- **`H_pre` is largely orthogonal to the matcher's strata and the surprisal is not.** `(k*, j, t0 // 4)` explains
  0.115 of `Var(H_pre)` and 0.704 of `Var(nll_e)` at 64k, so the entropy axis survives the matching almost intact
  (matched sd 0.65–0.68 against 0.67 on all rows) while the surprisal is nearly used up by it.
- **The twins' entropy coverage is not the fixed rows'.** A legal token of matched surprisal after an identical
  prefix mostly exists where the context is *uncertain*, so the caliper-selected pairs sit at mean `H_pre` 1.57
  against 0.86 on the fixed rows, with a narrower spread. Twin and fixed-row entropy readings are on different
  slices of the axis.
- **The logits and the log-partition are mostly, not entirely, linear in the state.** On these rows each raw
  logit is 0.98 linear in the raw state and `lse` 0.96 on `[s]`, `[s, z]` and `ln_f(s)` alike (held-out `R2`,
  `tables_logits_20260923.md` S; `lse` sd ~3.2 nats), so an appended feature can only act through that last few
  per cent or through the penalty; read the span diagnostic before reading an arm that appends a function of the
  belief.
- **The in-place ln arms carry `ln`'s deep-level fit loss.** `ln +lse` and `ln +max` sit on `ln_f(s)` in place,
  which loses fit at ℓ = 3-4 (the raw stream's scale); compare them with `ln`, not the ridge, at depth. `+lse` and
  `+logits +lse` are the raw-basis versions.
- **With many arms the anchor loop's per-column feature rebuild dominated the cell** (the first attached smoke,
  `ap-LXZKyawIXyEv3R4OZLl6lY`, was killed by a local timeout mid-anchor); `express.py` now builds each arm's
  feature matrix once per stream, bit-identically.
- **The `a1` twin pair sets are too thin for this axis** (270–358 token-balanced pairs at 64k against 2,600 on
  `swap65k`); every `a1` twin cell reads under 2σ. Read `swap65k` for the twins.

## Re-reads of the banked level-level claims under the richer readers

| folder | what |
|---|---|
| [`rereads/`](rereads/FILES.md) (2026-09-22) | The three banked **level-level** claims of the value side, each made with the ridge on the raw state only, re-read on the readers the expressivity rounds found to disagree with it at the sign level (`fullQ`, the belief appended; `fullM`, the MLP; `ln`; `fullP`): `calibration/` (norm §1's rank order with the world, shift-at-init-to-rescaling-when-trained, the outcome surprise's ordering, on the eleven `a1` diets, three seeds), `clock/` (adaptation's running-estimate reads on the five online fitters per reader, both order seeds), `public/` (projection's split with the belief-appended critic's response decomposed exactly into its state and belief parts, a nonlinear public reader, the readout span of the state block, the priced twin, three seeds). Each gated against its banked cells in-container; facts only, no `README.md`; interpretation waits on a discussion. |

[^private]: Not mirrored: this link points to a document in the private lab repo (the roadmap, the queue, an unrun spec, reading notes, or a conversation). See the top-level README for what is held back and why.
