# NOTES — `within`: the within-context candidate-discrimination readout

Decisions, every gate and what it was shown to fail on, and every defect beside its correction.
No README here by instruction; the table of record is
[`figures/within_reduction.txt`](figures/within_reduction.txt) and the orchestrator discusses the
results before any writeup.

**Up**: [`../README.md`](../README.md) (`preplay`, the writeup of this node and its siblings) · [`../../README.md`](../../README.md)
(`aliquot`, the writeup of the three rounds) · **Queue line**: `QUEUE.md`, "Value side",
soundboard follow-ups (ii).

---

## §1 The question, and what the rows let it be

Every AUC in `aliquot`, `duplex` and `soundboard` is pooled over rows, so it mixes "is this
context solvable at all" with "which of this context's candidates is the one to write". Only the
second is the cell the chooser occupies. This node measures the second, on the banked dumps, with
no loop and nothing paid.

A **group** is `(slot key, obs bytes)`: the same masked pre-write context at the same slot. The
**within-context AUC (WC)** is the ordinary rank AUC restricted to (positive, negative) pairs that
lie inside one group. Beside it the table prints **P**, the ordinary pooled AUC *on the same
rows*, so the gap is visible.

Two things the grouping is not, both measured in section [A2] rather than assumed:

1. **A group is not a trajectory.** The grader scored `fin (+) candidate` — the trajectory's
   *final* configuration with the candidate substituted — and `fin` is not in the dump; the
   recorder stores `obs`, the configuration at the *time of the write* with the slot masked
   (`soundboard.py` 8284, `ctx = traj[t]` against `fin = traj[-1]`). So two rows with the same
   `obs` can have been graded on different configurations. The measurable that bounds this is the
   share of rows sharing an `(obs, candidate)` cell whose world verdicts **disagree**: 413–767 of
   51k–62k such rows per arm, **0.8–1.4%** on the six soundboard arms (0.3–1.1% on the two
   overtone dumps). Small, not zero, and stated beside every number.
2. **A group is not a candidate set on the filed diet.** See §3.

The pairs are the claim, not the groups: the headline WC is **pair-weighted**, `WCg` beside it
weights every group alike, and `maxsh` gives the share of all pairs contributed by the single
biggest group (0.010–0.060 on every held-out cell, so no cell is one enormous group).

## §2 Decisions

- **Analysis set**: groups with ≥ 2 rows, **both** world classes, and a finite prior `dp`. The
  `dp` column is NaN where the candidate is not on the final operative table (`n_on_table`
  shortfall): 0.2–6.1% of probe rows. The filter is applied to rows and the group is re-tested for
  both classes afterwards, so filtering cannot manufacture a group.
- **Labels**: probe rows use `|yw`, the world's verdict on the substitution, where it exists. Both
  overtone dumps are **world-graded** (`vo_om` is None) and carry no `|yw`, so there `|y` *is* the
  world's verdict. On every filed buffer `|y` is likewise the trajectory's own verdict. On those
  buffers the `proj` reader **is the label** and is left blank rather than reported at 1.000 —
  defect #2 below.
- **Splits are never averaged and never mixed.** Held-out is `code < 100 · vo_critic_hold = 10`.
  `VoRecorder.hold_code` is a bijection of `obs`, so a group cannot straddle the split (gate H, 0
  violations on 3.4M rows), and `vo_critic_terms` trains the critic on `~hold` only, so a held-out
  cell is held out from the critic too.
- **Seeds are never averaged.** Every cell is per seed, per arm, per level, per split.
- **`dp` is already divided by the span** in the dump (`sb_dump_rows`: `g / move["span"]`), so
  `vo_compose`'s own `/ span` must not be applied a second time. See defect #3 for why this turned
  out not to matter for `comp` and to matter for one guard.
- **The composition is `vo_compose`'s, standardised within the group.** z-scoring is monotone
  inside a group, so `dp`'s and `crit`'s WC columns are unchanged by it and only `comp` is new.
- **`ov_s2` is reduced twice**, whole and on the uniform quarter. Its probe draw is
  3/4 disagreement-driven, which shows up directly: 9531 of 32173 groups carry one candidate
  against ~300 on every uniform arm. `duplex` read that seed's probe cell on the uniform quarter
  for the same reason.
- **What the columns saw is not the same for all of them, and this is not fixable from the dump.**
  `dp` and `crit` are the FINAL core's and the FINAL critic's quantities over `obs` + candidate —
  exactly the chooser's information set. `proj` is the projection's `p` **as filed**, by the
  readout of that cycle, over `fin (+) candidate`. `fin` is not banked, so `proj` cannot be
  recomputed and is not on the same input or the same moment as the other three
  (`overtone/analyze_dump.py` 100–104 states this; `duplex.py` 611–660 lived with it). Every cell
  carries that asymmetry.

## §3 The first finding is structural: the filed diet cannot be asked this question

On every arm, essentially **every filed group of size ≥ 2 carries one and the same candidate**
(22525/22526 at seed 0, 22192/22192 at seed 2). The chooser is `argmax(vo_compose)` with
`vo_eps = 0`, so one context gets one write; the filed record therefore contains **no
within-context candidate contrast at all**. Every reader built from `(obs, candidate)` is
constant across such a group, and the within-context AUC is exactly 0.500 (all ties) on the filed
side of every arm. That is not a null result about the readers — it is the diet having nothing in
it to rank, and it is reported as such.

The same fact holds of the object that matters most: the bank the run's readout is actually fit
on. Section [E]: all 30000 bank rows are `src = 0` (the filed diet), they cover ~29.7k distinct
contexts, and only **11–34 of those contexts carry two candidates with different verdicts**.

## §4 The added columns, and what they are for

Five columns beyond the four the brief asks for. Three are `duplex`'s protocol
(`trunk_features` → `fit_probe_t`, imported, not forked), both on the FINAL trunk over
`obs (+) candidate` (the masked post-write read, which is the run's own `vo_pj_mask = True`), both
fit per slot buffer and scored on held-out rows only:

- **`pw`** — fit on that buffer's own **probe** training rows against the world's verdict.
- **`pf`** — fit on the matching **filed** buffer's training rows instead. Same estimator, same
  trunk, same input, same moment; the only thing that differs from `pw` is the **diet**.
- **`pr`** — `pw` again over a **never-trained twin** of the same architecture, minted
  `torch.manual_seed(20260918)` *after* the trained core is loaded, which is `preplay`'s order
  and `vo_pj_rand_seed` in every one of these arms' own configs (so the floor here is the object
  `aliquot`'s in-loop random-twin arm read, **not** `duplex`/`overtone`'s offline 20260915 twin;
  the two fingerprints are 16198.119 and 16155.366, and gate R1 asserts they differ from each
  other and from the trained 20263.048). `obs ⊕ candidate` puts the candidate's own level-1
  features in the input, so a ridge over *any* feature map can rank candidates by identity
  alone; this column is what says whether the plant's training matters at this cell.

These three are fit per slot, so each carries 28–60 different intercepts and **their pooled
columns are not comparable to the global readers' pooled columns**; only their WC columns are.

Two more take the diet contrast to **the projection's own form** rather than `duplex`'s: one
**global** readout with `VoProjBank`'s design — its own masked read (which is *not*
`mask_outside`'s; `_features` masks the first block outside the span scanning from block 0,
`mask_outside` masks the block immediately after the span with a wrap, and the two differ
whenever `blk0 > 0`), all-block **and** span pooling, a group one-hot and its **interaction**
with the standardised features, IRLS with the ridge chosen on a train-internal split at
`vo_pj_boot = 0.8` over `PJ_RIDGES = (1, 32, 1024)`. `preplay`'s re-implementation of that form
is gated elementwise at 0.000e+00 against `VoProjBank` itself (`preplay.py::fidelity_gate`) and
is imported rather than written again.

- **`gw`** — fit on the probe diet. **`gf`** — fit on the filed diet. Both scored on the **same**
  held-out probe rows, both fit on 19661 train / 4915 validation rows drawn with one seed.

**One forced deviation, stated in the table's own legend.** `VoProjBank` groups by the **root**
— the goal — which is what makes it one linear readout per goal at 1737 columns. **The dump
carries no root.** The recorder banks it only into `vo_bank.npz`, whose rows are the
trajectory's *final* configurations (`push_experience` pushes `p["fin"]`), and those cannot be
joined to a probe row, whose `obs` is the configuration at the *time of the write*. So the
grouping variable in `gw`/`gf` is the slot's **level**: 4 groups, 965 columns. Falsification P1
measures what that substitution costs and finds it small — shuffling the root column costs the
projection's own held-out AUC on the bank **0.0040** (0.8057 → 0.8017), the same order as the
run's banked gap between the interaction design and the additive one (0.8187 against 0.8068).
The per-goal block is very nearly inert on this substrate, so grouping by level instead of root
is a much smaller change than the column count suggests. It is still a substitution, and every
`gw`/`gf` number carries it.

**What the same object cannot be asked.** The projection's form fit on the *bank* — the run's
actual readout — cannot be scored on probe rows for exactly the same reason: probe rows have no
root. Gate P therefore refits the projection's form **verbatim** (root grouping, `v = 8`, 1737
columns) on each arm's own bank at the run's own sizes and checks it against the readout's
banked held-out AUC; it cannot carry that fit onto the probe rows.

**What `pf` can and cannot separate.** `pf` reads 0.53–0.60 WC and 0.52–0.57 pooled on probe rows
— near chance on both. Because it is near chance *pooled* as well as within-context, `pf` alone
does **not** separate "the filed diet has no within-context contrast to learn from" from "a
filed-fit readout does not transfer to counterfactual rows at all". Both are true of that diet and
both would stop the projection doing the chooser's job, but they are different mechanisms. The
direct evidence for the first is section [E], which does not depend on `pf`. Note also that the
run's own projection (`proj`, 30000 bank rows pooled across slots with the root interaction) reads
*above* `pf` at 0.63–0.70 WC, so `pf` is a lower bound on what the filed diet supports at a
better-estimated fit, not an estimate of the projection itself.

**What the two additions returned** (all levels, held out, pair-weighted, probe rows; the table
of record has them per level):

- `pw` − `pr` — the trained trunk over the never-trained twin — is **+0.010 to +0.066** on the
  six soundboard arms, +0.039 and +0.072 on the two overtone dumps, and **−0.008** on the
  `ov_s2` whole-draw cell. The twin itself reads 0.69–0.93. So most of what `pw` reads at this
  cell is available from a random feature map over `obs ⊕ candidate`, which is `overtone`'s
  0.02–0.06 random-init gap reproduced here.
- `gw` − `gf` — the diet at the projection's own global form — is **+0.031 to +0.053** on the six
  soundboard arms and +0.116 to +0.303 on the two overtone dumps. `gf` reads 0.63–0.71, which is
  where the run's own `proj` column reads (0.63–0.70); `gw` reads 0.66–0.76, below `pw`'s
  0.70–0.85 on the same diet and the same trunk, so the per-slot-versus-global axis is worth
  0.09–0.13 here and the per-slot `pf` at 0.53–0.60 is the lower bound §4 said it was.

## §5 The gates, and what each was shown to fail on

Every gate below has been run against a deliberately broken variant and shown to trip;
`within.py::falsify` is the harness, **10/10 required checks trip, plus one reported and not
required (P1)**, and a gate is not reported here until it has failed.
Per-arm gate records are mirrored under `figures/<tag>/<arm>/within_gates.json`.

| gate | what it asserts | shown to fail on | result |
|---|---|---|---|
| **Z** | the reducer's numpy `np_compose` **is** `voicing::vo_compose` | Z1 `ddof = 0` instead of torch's unbiased `.std()`; Z2 the weight `w` on the prior instead of the judge; Z3 no `clamp_min(1e-6)` on a degenerate set | max ∣Δ∣ **1.3e-15** on every arm |
| **A** | the reducer's numpy `np_auc` **is** `voicing::vo_auc` and `duplex::auc_t` | A1 ties ranked by position instead of mid-ranked, on a tie-heavy draw | max ∣Δ∣ **0.0** on every arm |
| **C** | the loaded plant **is** the plant the run ended with | C1 one parameter moved by 1e-3 | ∣Δ∣ **0.000e+00** against `log["sb"][-1]["sig"]` on all six soundboard arms; absent (reported `n/a`, not passed) on the two overtone dumps, which predate the fingerprint series |
| **B** | the batching is not the number | B1 the unmasked post-write read substituted for the masked read of record | **0.0** on both `crit` and `postm`, batch 4096 vs 512, every arm |
| **H** | the hold code is a function of `obs`, so a group never straddles the split | H1 one hold code flipped inside a group | **0** violations, 3.4M rows |
| **D** | the filed-side degeneracy count can move | D1 a second candidate injected into one filed group | the counter falls, so the 22525/22526 is a measurement and not a constant |
| **R** | the twin is a fresh draw and its seed is load-bearing | R1 the twin at seed 20260915, and the trained core | 16198.119 / 16155.366 / 20263.048, all distinct |
| **P** | the projection's form is being driven correctly | P1 the root column shuffled — **reported, not required**; it does not trip, and §4 says why | refit held-out AUC on each arm's own bank 0.759–0.854 against the banked 0.782–0.861, **within 0.030 on all six**; the two overtone dumps banked no readout buffer, so P is reported absent there |

**One consistency check beside the gates, not a gate.** The `proj` column's ordinary pooled AUC
over *all* probe rows of an arm, against the run's own independent instrument — the reservoir
sample `log["vo_probe_rows"]`, 4000 rows drawn across the whole run, which records the
projection's `p` and the world's `yw` at file time: 0.7065 / 0.7190 / 0.7235 from the dump
against 0.7248 / 0.7061 / 0.7183 from the reservoir (`sb_s1/sb_sv_yk`, `sb_s2/sb_sv_yk`,
`sb_s1/sb_so_yk`). Different row sets — the dump is the buffers' capped tail, the reservoir is
the whole run — so they agree only to within sampling noise, which is what they do. This says the
`|y` column is the object it is being read as.

The critic load is `load_state_dict` **strict**: a shape or key mismatch raises rather than
scoring a wrong critic. `voicing::build_critic` with `ctx_mode = "full"` and
`hidden_mult = span_hidden_mult = 4` is numerically the class `soundboard.py` mints (diffed: the
fork only deletes the `ctx_mode` branch and the `hidden_mult = 0` branch, neither of which is on
this path), so the leaf helper is used rather than a second copy.

## §6 Defects, in order, each beside its correction

1. **The first fetch silently collapsed three files into one.** `modal volume get <vol> <src>
   <dst>` with a `<dst>` whose parent directory does not exist writes the file *as* that path, so
   `.../sb_s1/sb_sv_yk` ended up being `vo_heads.pt` rather than a directory holding three files.
   Caught by `np.load` raising `NotADirectoryError`, not by a claim. Corrected by creating the
   per-arm directory first and naming the destination file explicitly.
2. **`proj` was reported at AUC 1.000 on every filed cell and on both overtone probe cells.** On
   those buffers the filed column `|y` *is* the world's verdict, so the "reader" was the label.
   Caught by the number being exactly 1.000 in the first reduction. Corrected with a `label_is_y`
   flag per buffer, derived from whether the buffer carries a `|yw` column; those cells are now
   blank and the legend says why. Nothing else in the table moved.
3. **Gate Z's first Z2 was not a falsification, and the reason is a fact about the composition.**
   Z2 fed the dump's already-divided `|dp` to `vo_compose` without compensating and asserted the
   gate would trip. It did not: `vo_compose` divides by the span and **then** standardises over
   the candidate set, and the standardisation annihilates any positive scale. The span division is
   **inert inside `vo_compose`**; it matters only where the standardisation is skipped — the
   size-1 guard — and for a raw `dp` column pooled *across* levels, which is why the dump divides.
   Corrected by replacing Z2 with the weight-on-the-wrong-organ break, which does bite, and by
   recording the fact in the harness docstring. (The size-1 guard is still real and is handled
   explicitly in `gate_compose`: the donor returns the *raw* `dp_sc` there while the reducer
   returns the already-divided column, so the gate compares them on the same scale rather than
   pretending the guard is not there.)
4. **Falsification P1 does not trip, and the reason is again a fact about the object, not a weak
   harness.** Shuffling the root column of the projection's own design costs its held-out AUC on
   the bank 0.0040. The check is therefore recorded as *reported, not required* (the harness
   asserts only the ten required ones), and the number is carried into §4 because it is what
   licenses the level-for-root substitution in `gw`/`gf`. Second instance in this node of a
   falsification failing for a reason worth keeping; the first was Z2.
5. **The first waiter never waited.** It polled `modal app list` for the string `running`; a
   detached app's State column reads `ephemeral (detached)` while it runs and `stopped` when it
   finishes, so the loop exited on the first iteration and the fetch found an empty volume
   directory. Corrected to wait for `stopped`, and the empty fetch is what caught it.

## §7 Cost and shape

Eight arms across eight CPU containers (`score` starmaps `score_arm`), no GPU, `cpu = 8`: the
plant is 2 layers at width 96 over length 64, one arm is ~430k rows at ~7 forward passes each
(three through the trained trunk, three through the twin, one for the critic), plus three IRLS
fits at 965–1737 columns. **285–577 s per arm**, peak RSS 5.3–5.4 GB (the request is 8192 MB;
the floor is torch's allocator, not the buffers, which are ≤ 8192 rows each). The reduction is
pure numpy, ~8 minutes for all nine rows of the table. Total: three full passes (`wi0` without
`pf`, `wi1` without `pr`/`gw`/`gf`, `wi2` complete) plus four smokes and three falsification
runs.

Volume: `rhm-scaling-data:/rhm_practice_within/wi2/<tag>/<arm>/{within_scores.npz,
within_gates.json, within.log}`. **`wi2` is the tag of record**; `wi0` and `wi1` are earlier and
superseded.

## §8 Reproduction

```bash
cd experiments/            # MODAL_PROFILE=chromatic
modal run rhm/practice/voicing/sotto_voce/aliquot/preplay/within/within.py::falsify   # 10/10 + 1 reported
modal run --detach rhm/practice/voicing/sotto_voce/aliquot/preplay/within/within.py::score \
    --out-tag wi2 --arms all
# then, with the dumps and the score files fetched into <dumps> and <scores>:
python3 rhm/practice/voicing/sotto_voce/aliquot/preplay/within/reduce_within.py \
    --root <dumps> --scores <scores> \
    --out rhm/practice/voicing/sotto_voce/aliquot/preplay/within/figures --arms all
```

`<dumps>` holds `<tag>/<arm>/{vo_rows.npz, vo_rows_meta.json, results.json, vo_bank.npz}` from
`rhm_practice_soundboard` (six arms) and `rhm_practice_voicing` (the two overtone dumps, which
have no bank). Neither directory is committed.

## §9 Files

| file | purpose |
|---|---|
| `within.py` | the Modal CPU scoring pass (`score_arm`, `score`) and the falsification harness (`falsify`) |
| `reduce_within.py` | the numpy reducer and the estimators `within.py` gates against |
| `figures/within_reduction.txt` | **the table of record** — sections [A] [A2] [B] [D] [C] [E] |
| `figures/within_reduction.json` | the same, machine-readable, with every cell's pair counts |
| `figures/<tag>/<arm>/within_gates.json` | the per-arm gate record and per-key fit log |
| `results/launch_wi*.log` | the launch logs of record |
| `NOTES.md` | this file |
