# grokking — NOTES (decisions, gates, defects)

Working notes for the builder and the orchestrator. Not a README: Jasper discusses results before one
is written. Tables of record: `figures/mint_m1_table.txt`.

## Setup as run (Mint, tag `m1`, 2026-09-22)

- Vanilla recipe, forked verbatim (`shared.py` fork notice): p=97, MLP [128,128] ReLU, one-hot (a,b) 194 dims,
  30% split (2822 train / 6587 held-out), full-batch AdamW lr 1e-3 wd 1.0, seed 42, 40k epochs, 53,985 params.
  The net fits the train set at epoch 300 and reaches test ≥0.5/0.99/1.0 at epochs 10,750/16,050/18,550.
- Snapshots every 250 epochs (+ the pre-step init + epoch 39,999): 162 on the true-label run. Shuffled-label run
  (train labels permuted with seed 43, same split and init): every 1000 epochs, 42 snapshots. Untrained random
  inits: seeds 0..3, plus the seed-42 init that heads both runs.
- Modal app `grokking-mint`, volume `grokking-mint-data`, CPU containers only. Training is about 170 epochs/s on 8
  CPUs (235 s for 40k epochs). The whole `mint` coordinator took 674 s.

## Decisions

1. **One gate per compiled form.** The brief's gate compares "the compiled model" with the net's own train
   argmax. Form A (Σ_{w∈K} cos 2πw(a+b−c)/p) has the same argmax for every nonempty K, which follows from the
   brief's own math fact. A gate on A can therefore only decide between an empty K and a nonempty one; which
   frequencies are in K cannot change its answer. So every form runs its own census walk, gated by its own
   agreement with the net, and every form is also evaluated at every other gate's K (the `cross` table).
2. **Tie-aware agreement.** A compiled model whose maximum is shared by m classes scores 1/m if the net's class is
   among them. The empty table of the closed forms is therefore exactly chance (1 − 1/p), not "class 0".
3. **Three tolerances.** `census_walk` admits iff `e_x <= best_e + tol`:
   - `le0`: tol 0, the donor's EXTEND_TOL, verbatim. It admits ties. The effect is that no-op candidates get
     admitted: on B, frequencies with no dominant unit; on A, every frequency after the first.
   - `lt1`: tol = −0.5/n_train. A candidate must fix at least one of the net's train predictions, net of any it breaks.
   - `m3`: tol = −3·sqrt(q(1−q)/n_train) with q = 1/p, which is −0.0057, about 16 pairs. The candidate must beat
     the 3σ binomial noise of chance agreement.
   All three are reported. I have not picked a rule of record. The falsifiers separate them (see below).
4. **Form C was added beyond the brief.** It projects every layer-0 row onto DC plus K's characters, with no
   retraining. It is the frequency-level counterpart of form B, which prunes at the unit level, and the MLP
   analogue of a restricted-loss ablation. I added it because A and ALS cannot measure anything about which
   frequencies are in K (decision 1).
5. **ALS** uses cosine-only amplitudes α_w (the brief's "per-frequency amplitudes"), fit by least squares to the
   net's row-centred logits on the **train pool only** (the gate pool). The design is balanced, so α_w is the
   cosine coefficient of the diagonal-averaged logit g(d), d = a+b−c. Gate G-6 checks this against an explicit
   lstsq fit with per-row intercepts. Every model of the form g((a+b−c) mod p) is translation-equivariant: its
   accuracy on every pair is either 1 or 0 (or a 1/m tie fraction), and its held-out accuracy equals its
   train-label accuracy.
6. **Read.** A unit's dominant frequency is the argmax over w ∈ 1..48 of its combined a+b energy, with w and p−w
   folded together and DC excluded. At the final net, 122 of 128 units have the same dominant w in the a-row
   and the b-row.
7. **Logit-table measure.** The donor's measure (cnb `compute_fourier_energy`, 0.374 on its net) takes the top 5 of
   the *unfolded* diagonal k = 1..96. Conjugate pairs have equal energy, so those top 5 cover about 3 distinct
   frequencies. The measure is reported verbatim (`old_top5`) next to the folded per-frequency diagonal.
8. **SVD baseline.** Following cnb `precompute_svd.py`, all three layers are truncated to the same rank r and the
   biases are kept. Each layer's parameters are counted as min(factored, dense).

## Gates (instrument checks, `mint.py::gates`; all passed before the run)

G-1 `split_pairs` equals the verbatim `generate_data` split. G-2 every single w ∈ 1..48 gives form A 100% on all
pairs, and G-2b the empty A is exactly 1/p. G-3 form B at K = all is bit-for-bit the net, both at init and after
300 steps. G-4 form C at K = all matches to 6.7e-6. G-5 a designed walk sequence where the three rules must
differ gives le0 {1,2,3,4}, lt1 {1,3,4}, m3 {1,4}. G-6 ALS closed form vs lstsq agree to 5e-16. G-7 the read
recovers planted frequencies, conjugate folding included. G-8 filter leak 3e-32.

In-run checks on m1: form B at K = all is bit-exact at 208/208 snapshots; form C at K = all has max |Δlogit|
1.7e-5 and argmax agreement 1.0 everywhere; all 128 units get a dominant w at every snapshot.

## Falsifiers (m1)

- **A** admits nothing under all three rules, on all 42 shuffled-net snapshots and all 5 untrained inits.
- **ALS** admits noise under le0 (k 12–39) and lt1 (k 1–4) on the shuffled and random-init nets, with held-out
  0.000 (it predicts a shifted sum). **Only m3 admits nothing.**
- **B / C** reproduce the net's own spelling, so on the memorizing shuffled net they admit by construction
  (lt1: B 36–44, C 41–47; m3: B 19–41, C 2–8). Their true-label held-out stays at chance (0.007–0.014). On the
  untrained inits under lt1, B admits 0–19 and C admits 0–38 (38 on seed 0).

## Defects and fixes

- D-1 `_load_snaps` indexed the lazy npz once per snapshot. Each state then held views into its own full
  decompressed copy of the stacked arrays, and the analysis containers peaked at 6.1 GB RSS against a 2 GB
  request. Fixed after m1 by decompressing each array once. No effect on any number.
- D-2 the SVD parameter count used the factored count whenever r < min(d_in, d_out), which exceeds dense at high
  ranks. Fixed in `mint.py` (min of the two) and recomputed from rank in `reduce_mint.py`. The matched-accuracy
  ranks (≤ 41) are unaffected.
- D-3 `modal run` lowercases CLI parameter names, so `rung(K=...)` received `k`. Renamed to `--k-list`.
- The "Timed out waiting for final app logs" line from the modal client is harmless (the app completed).

## Rung (built, smoke-tested, not run)

`rung.py` smoke `rsmoke` (4 arms, 2k-epoch caps, K = B/lt1 of m1) runs end to end. Its numbers at 2k epochs are a
code check only. The bilinear learner's width defaults to m = 4k for every alphabet, so the alphabets differ
only in input dims. The recipe is unchanged from Mint, including wd 1.0 on the bilinear net.

## Rung (tag `r1`, run 2026-09-22; control `r1ctrl`)

- The orchestrator's go set these choices:
  - K = the shared B/lt1 = C/lt1 set {10, 11, 18, 24, 30, 33, 36, 38, 39, 44, 45, 46, 47}. The run asserts that
    this is the top 13 of Mint's producer order.
  - lowE = the 13 frequencies with the lowest producer score, none of them in K: {8, 15, 6, 16, 3, 32, 28, 40,
    5, 34, 29, 37, 21}.
  - top1 = 30. The run asserts that this is the K that A-LS under m3 kept.
- Recipe unchanged: lr 1e-3, wd 1.0, full batch, seed 42, same split. Test accuracy is evaluated every 10 epochs,
  and a run stops at test 1.0 or at its cap (40k main, 150k sweep).
- Bilinear width is m = 4·13 = 52 for every alphabet. One extra arm runs top1 at its minimal width m = 4.
- (a+b+c) mod p was skipped. At frac 0.02 it has 18k train triples, which is not nearly free on CPU.
- Consistency check: `main_raw_mlp` is Mint's net again, with a finer eval grid. It first reaches test ≥0.99 / 1.0
  at 16,040 / 18,520; m1 on its 50-epoch grid gave 16,050 / 18,550.
- Arms that never solve are flat by the cap: they memorize train to 1.0 and hold their test level. The exception
  is `main_top1_mlp`, which peaks at 0.714 around epoch 8,300 and then decays to 0.546 as train falls to 0.88.
- Design fact found at reduction: per-input feature norm differs by alphabet (raw and top1 √2, minted and lowE
  √26, all48 √96). The `r1ctrl` control rescales all48 to minted's norm ("all48n"). It still does not solve
  (test 0.0006 MLP, 0.0044 bilinear, train 1.0), so the input scale is not the cause of the all48 failure.
- Defects: none in the run. The reducer's sweep figure had overlapping minor-tick labels and coincident
  "not reached" markers; both fixed before the figures of record.

## Rung controls (tag `r2`, 2026-09-22): why all-48 and top1 fail in r1

Code: `rung_controls.py`, reduced by `reduce_rung_controls.py` into `figures/rung_controls_r2_table.txt`. Seed 42,
train fraction 0.3, CPU, at most 4 containers.

**Margin calculation (the orchestrator's; recomputed here exactly).** The argmax margin of Σ_{w∈K} cos(2πwd/p), peak
at d=0 minus the runner-up over d≠0, is:
- 0.002097 for {30};
- 9.958 for the kept 13;
- 48.50 for all 48.

So a one-frequency solution needs about 4,700 times the per-frequency amplitude of the 13-frequency one to reach
the same margin. By this count all-48 has the largest margin.

**Control A: all-48 as an orthogonal reparameterization of one-hot (MLP only).**
- DC is included for this control, so the per-operand map is square. Q is 97×97 and orthogonal, with rows
  DC/√p and √(2/p)·cos, √(2/p)·sin. Two scales:
  - `all48dc_q`: x = blockdiag(Q,Q)·onehot, per-input norm √2, as for raw.
  - `all48dc_f`: √(p/2)·Q, r1's cos/sin scale plus a DC of 1/√2.
- "Transported" init sets W0 = W0_raw·M⁺ with every other tensor copied from raw's seed-42 init. The initial logits
  then match raw's to 4.5e-8 on all p² pairs.
- A1, under the recipe (AdamW 1e-3, wd 1.0, cap 40k): q-transported, f-transported and q-default all fail to solve
  (final test 0.0005 / 0.0003 / 0.0005, train 1.0).
- A2, needed because A1 failed: raw and q-transported under torch SGD with L2, same init. For plain or momentum GD
  the two are the same trajectory in exact arithmetic. The smoke check at 1,000 epochs agrees: max |Δ test acc|
  1.5e-4, |Δ train acc| 0.
- A2 probe (raw only, momentum 0.9, cap 100k, stop at 1.0), by lr/wd:

  | lr / wd | Raw e99 | Raw final test |
  |---|---|---|
  | 0.1 / 1e-3 | 27,850 | 0.9985 |
  | 0.3 / 3.33e-4 | 25,140 | 0.9951 |
  | 1.0 / 1e-4 | 42,350 | 0.9994 |
  | 1.0 / 1e-3 | never (train 0.015) | 0.0083 |

  No raw arm reaches exactly 1.0 within 100k.
- A2 twins: q-transported at the three settings where raw groks, same init, same SGD:

  | lr / wd | Twin e99 | Twin e100 | Twin final test |
  |---|---|---|---|
  | 0.1 / 1e-3 | 28,060 | never | 0.9989 |
  | 0.3 / 3.33e-4 | 22,210 | never | 0.9998 |
  | 1.0 / 1e-4 | 34,650 | 53,530 | 1.0 |

  All three twins grok.
- Raw vs twin, on the 100-epoch grid:
  - Equal at the start: loss agrees to about 1e-5 relative at epoch 100.
  - Relative |Δloss| first exceeds 1e-4 at epoch 2,000 / 600 / 200 for lr 0.1 / 0.3 / 1.0.
  - |Δtest| first exceeds 0.01 at epoch 14,700 / 11,900 / 18,000, mid-transition.
  - The largest |Δtest| over the run is 0.20 / 0.16 / 0.43.
  - The initial difference is float-level (4.5e-8 in logits).

**Control B: top1 (frequency 30, 4 input dims).**
- B1: the exact one-frequency solution planted in bilinear m=4 (A = t³, U = V = t·selector, W = t·base), then the
  recipe with no early stop for 20k epochs.
  - Every planted amplitude A ∈ {1, 10, 100, 1e3, 1e4} starts at test 1.000.
  - Test stays at 1.000 only while α_30 is large:
    - A = 1e4 holds to epoch 1,100 (α_30 690 at epoch 1,000) and is below 0.99 from 1,200.
    - A = 1e3 holds to epoch 200.
    - A ≤ 100 falls below 0.99 by epoch 100 (A = 1 by epoch 0).
  - From large A, α_30 falls about 4× per 500 epochs (9,969 → 2,470 → 690), close to e^(−3·lr·wd·t) for the
    three factors decaying together.
  - Every arm is at 0.116–0.152 by epoch 5,000 and 0.098–0.113 at 20k.
  - α_30 ends at 2.4–3.1 from every start.
  - The diagonal margin ends at −0.0013 to 0.0041.
- B2: weight-decay sweep from random init (AdamW lr 1e-3, cap 40k).
  - Bilinear m=4 solves at wd 0.1 / 0.01 / 0.0 (e100 14,550 / 7,940 / 7,640, α_30 ≈ 517–522, diagonal margin
    ≈ 1.08–1.09) and fails at wd 1.0 (final 0.098).
  - The MLP never reaches 0.99 at any wd. Final test is 0.546 / 0.984 / 0.985 / 0.974 for wd 1.0 / 0.1 / 0.01 / 0.0,
    and the best peak is 0.989 at wd 0.1.
- B3: r1's minted-13 bilinear arm re-run. It reproduces r1 exactly (e99 2,630, e100 3,760).
  - At the stop: α over K13 has mean 0.80 (min 0.094 at w=10, max 1.23) and sum 10.4; diagonal margin 7.59;
    per-pair test margin min 0.044, mean 4.98.
  - Continued to 40k: α mean 1.50, sum 19.5; diagonal margin 15.2; test margin min 10.6, mean 14.0; diag_frac 0.94.

Run ids: r2 A1/B `ap-Ke2svVwxW6UxfjDbi97qSF` (682 s); A2 probe `ap-3Tj1cVaAZP79H4uWQJQi37` (761 s); A2 twins
`ap-jV9PMNSERcAGRxl8n1wRX8` (473 s); smoke on tag `r2smoke`.

Defects: rung_controls imports `rung` as a module (`RG`), because `Bilinear` exists only where torch is present.
The A2 comparison in the reducer first matched curves by index. That misaligned the lr 1.0 pair, where the twin
stopped at 53,530, so it now matches on shared epochs.
`train_net` gained `optimizer` / `momentum` / `probe` arguments; the defaults reproduce every earlier run, and B3
reproduced r1 bit-for-bit.

## Rename (2026-09-23)

The node was built as `experiments/grokking_mint/` and renamed to `experiments/grokking/` at close-out. Imports,
the Modal local-source mount, command paths and table headers were rewritten; the Modal app (`grokking-mint`),
volume (`grokking-mint-data`) and every `/data/<tag>/` path are unchanged, so all banked runs are still addressed
as recorded. The launch logs under `results/` keep the old path, as records of what ran.
