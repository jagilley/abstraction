# grokking/whittle — NOTES (decisions, gates, defects)

Working notes for the builder and the orchestrator. Not a README: Jasper discusses results before one is written.
Up: [`../NOTES.md`](../NOTES.md) (m1, the net), [`../basis/NOTES.md`](../basis/NOTES.md) (the recovered basis).
Table of record: `figures/whittle_w1_table.txt`; figure `figures/whittle_w1_survivors.png`.

## Setup as run (tag `w1`, 2026-09-23)

- No fresh training. Every MLP arm starts from m1's banked final net (`/data/m1/true/snapshots.npz`, label `e39999`,
  seed 42; 53,985 params, test 1.0). The optional bilinear arm starts from r2's banked minted-13 bilinear
  (`/data/r2/B3_minted13_bil_stop_final.npz`, width 52, the `B3_minted13_bil_stop` arm, test 1.0 at epoch 3,760).
- The basis is `BasisSnapshot(final state)`: 48 recovered 2-D subspaces and the DC vector u0. K is re-walked with
  `_walk_K(snap, "C", lt1)`: candidates (0, 2, 3, 6, 8, 11, 16, 22, 27, 32, 37, 38, 47). Their DFT match (oracle, logged
  only) is {10, 11, 18, 24, 30, 33, 36, 38, 39, 44, 45, 46, 47}, which is m1's K. Form C at this K: held-out 0.99939.
- Modal app `grokking-mint`, volume `grokking-mint-data`, profile `chromatic`. A CPU coordinator (`whittle_run`,
  cpu 1) starmaps one `whittle_arm` container per arm (cpu 4, memory 1.5 GB; smoke peak RSS 0.83 GB).

## Decisions

1. **Feature scale.** Input symbol a is spelled s·Pm[a, :] with s = √(97/2) = 6.964. Each pair's two coordinates are
   then a unit-amplitude cos/sin of the symbol's angle, as in `rung.py`'s `char_features`. Layer 0 is
   A = [Wa Pm, Wb Pm] / s, so A·x = form C's layer 0 exactly and A's entries are the wave amplitudes, the scale of W0's
   entries. The minted head uses the same s: logits = s·Pm·(B h + β), B = Pmᵀ W2 / s, β = Pmᵀ b2 / s. The function does
   not depend on s; Adam's step size relative to the weights does.
2. **Pm by QR, not SVD (the brief's `orth` does not hold up here).** `basis.orth` is an SVD. It returns an orthonormal
   basis of the whole 27-D span that mixes the symbols, so the per-symbol 2-vector groups would be lost. The
   recovered subspaces are not exactly orthogonal to each other: the operator is slightly non-normal, and the Gram
   off-diagonal of [u0 | Q_j] is at most 3.5e-4 (4.3e-4 for all 48). u0 is the operator's Perron vector, which is
   within 4.9e-4 of uniform. So Pm = QR([u0 | Q_j1 | … | Q_j13]) in producer order, with signs fixed so that
   diag(R) > 0. The span is the same, so the projector is form C's. Each column moves by at most 7.7e-5 (1.5e-4 for all
   48). Each symbol's QR columns overlap its own recovered subspace at ≥ 0.9999999 (0.9999996 for all 48).
3. **What is pruned, and the groups.** Only weight matrices are pruned. Biases are not pruned by magnitude; the
   cascade (decision 5) removes the biases of dead units. The groups (brief item a):
   - minted layer 0: one group per (unit, operand, symbol), a 2-vector for a pair and a scalar for DC;
   - minted head: one group per (symbol, layer-1 unit);
   - bilinear U and V: one group per (unit, symbol), 2-vectors, no DC on the input;
   - one-hot layer 0, layer 1 and the plain head: scalar groups.
   Both scalar counts and group counts ("symbol connections") are reported. The group L2 norm is invariant to a
   rotation inside the subspace. With unit-amplitude features, a pair group's norm R and a DC group's |w| both give an
   rms pre-activation contribution of norm/√2. The plain group norm is therefore also the contribution criterion
   inside a minted layer.
4. **Criterion across layers (brief item b): per-layer fractions, round robin.** Each layer keeps its own prune
   fraction f_L (start 0.5). A round prunes ceil(f_L · alive_L) of that layer's smallest surviving groups. On rejection
   the last admitted state is restored and f_L is halved; layers are visited in order layer0 → layer1 → head
   (U → V → W). A layer stops once f_L < 0.005, which takes 7 halvings; near the end every layer tries single-group
   prunes until one fails 7 times. The cap is 400 rounds per arm. There is no global threshold, so the per-layer
   scales never meet.
5. **The exact cascade (added).** After every prune, weights that can no longer reach the output are removed, and so
   are biases of dead units. A layer-0 or layer-1 unit with no surviving outgoing weight loses its incoming weights
   and its bias. A unit with no surviving incoming weight outputs the constant relu(b); that constant times its
   outgoing column is added to the next layer's bias, and the column is removed. For the bilinear, a product unit
   with an empty U row, V row or W column is removed whole. This is function-preserving (G2b, G2c). Without it,
   weights into dead units would receive zero gradient, keep their magnitude, and be counted as survivors.
6. **The retrain budget, and the drift finding that set it.**
   - A round's retrain is a fresh AdamW (lr 1e-3, the arm's weight decay), full batch on the 2,822 train pairs. The
     targets are the original net's train argmax, banked at the start; they equal the train labels.
   - The retrain stops before the step once the pre-step forward has agreed on every train pair for 100 consecutive
     epochs (`settle`), and runs at most 500 epochs (`retrain`). The gate then reads the agreement.
   - The cap was set by G3. With nothing pruned, a single 2,000-epoch retrain at wd 0 moves held-out to 0.923 on
     all48 and to 0.9979 on both_sides (`results/gates_whittle_R2000.log`); all48 is still 0.873 at 1,000. At 500
     every arm passes (`results/gates_whittle_R500.log`).
   - The mechanism is the r2 control A fact: near zero loss Adam's normalized step is about lr per coordinate in
     sign-like directions, and it is not rotation-equivariant.
   - Chained 500-epoch fresh-AdamW chunks with nothing pruned drift on every arm. A local check, 16 chunks, gave
     held-out ranges committed 0.985–1.0, unrestricted 0.980–1.0, both_sides 0.991–1.0, all48 0.874–1.0.
   - Two additions follow from this, both logged only:
     - (a) a **drift twin** per arm: the start state, nothing pruned, retrained with the admitted rounds' epoch
       counts in order. It separates retraining drift from pruning in the held-out series.
     - (b) a **budget-sensitivity arm**, `committed_R2000` (cap 2,000; G3 passes for committed at 2,000). A local
       prototype at a fixed 500-epoch budget had admitted rounds recovering at epochs 246–497, so the cap may bind.
7. **Adam state.** A fresh AdamW for every round keeps rounds independent and makes restore exact. Pruned entries
   get their gradients zeroed before the step and are re-masked after it, so they stay exactly zero (G2).
8. **both_sides recovery.** Projecting the head onto span(Pm) drops train agreement to 0.99894 (3 pairs) and
   held-out to 0.99712. The net is retrained in cap-sized chunks until agreement is 1.0; one chunk was enough in the
   gates and in the smoke. This is recorded per arm as `recover`.
9. **The bilinear is loaded, not retrained.** r2 banked its solved state, so it is loaded instead of re-running the
   ~11 s retrain. Rung trained it on DFT cos/sin features at K13, which is the gifted basis. Its U and V are rotated
   into the recovered pairs by a least-squares map T (26×26; orthogonality error 1.1e-7; feature residual 5.2e-4), and
   its head is minted with DC. G1 vs the banked net projected on span(Pm): max |Δlogit| 9.2e-3, train agreement and
   held-out 1.0.
10. **Readouts** at the start and at every admitted round:
    - survivors per layer (scalars and groups), connected units, live biases, and "effective" (weights + live
      biases + head bias);
    - train agreement and held-out;
    - the probe `rung_controls.make_probe` (gbar margin, per-pair margins, diag_frac, wnorm; its α at the DFT K13,
      oracle).
    Rejected rounds keep their gate numbers and held-out (after the failed retrain, before restore). At the end:
    - the surviving structure (`structure()`): per layer-0 unit, the symbols read on a and on b; single-symbol and
      same-on-a-and-b counts; distinct symbols; per-symbol connection counts; per layer-1 unit, the fan-in and the
      symbols of its inputs; the minted head's writes;
    - `mint.logit_table_energy` (a DFT read, oracle);
    - the final state banked as `/data/w1/<arm>_final.npz`.
    For the one-hot arm, the surviving layer-0 rows are read by projection onto the full recovered basis. DFT matches
    are logged beside every symbol and never consumed.

## Gates (`whittle_gates`; all pass at the cap of record, `results/gates_whittle_R500.log`, Modal, 32 s)

- **G1** (init reproduces the net it rewrites):
  - committed / committed_wd1: max |Δlogit| vs form C 1.1e-5; train agreement 1.0; held-out 0.99939 (= form C).
  - all48: 1.5e-5 vs the original net; unrestricted: 0 (it is the net).
  - both_sides: 1.3e-5 vs form C's logits projected on span(Pm); train agreement 0.99894 (3 pairs), recovered by one
    chunk (decision 8).
  - bilinear_both: 9.2e-3 vs the banked net projected, agreement 1.0 (decision 9).
- **G2** (prune half of every layer's groups, cascade, 200 epochs at wd 0 and at wd 1): pruned entries and dead-unit
  biases are exactly zero after retraining, on every arm. **G2b**: the cascade after that prune changes no logit by
  more than 2.9e-6 (unrestricted, 8 constant units folded) and by 0 elsewhere. **G2c** (designed: 5 layer-0 units
  with no inputs, 5 with no outputs, 3 layer-1 units with no head weights): the fold is exact to 5.7e-6.
- **G3** (zero-prune retrain at the full cap, no early stop): train agreement 1.0 and held-out 1.0 on all six arms at
  500 (both_sides after recovery). At 2,000: all48 0.9235 and both_sides 0.9979 fail, the others pass
  (committed 1.0, unrestricted 0.9994, committed_wd1 1.0, bilinear 1.0).
- **G4** (walk logic on a designed sequence, `MockPruner`): three layers of 100 / 50 / 7 groups whose gate passes iff
  ≤ 73 / 20 / 0 are pruned end at exactly 73 / 20 / 0 in 27 rounds. With scripted spurious failures the walk never
  exceeds a cap and restores on every rejection. The round cap stops at 3 when set to 3.

## Smoke (`wsmoke`, Modal, 27 s wall)

All 7 arms, cap 200, settle 20, 9 rounds each. Every arm completes, both_sides recovers in one chunk, and the drift
twin runs. Profile per arm: loading and the basis read 3–4 s, retrain 4–9 s, probe 0.3–0.6 s, bookkeeping < 0.02 s,
twin 0.7–3 s. Peak RSS 0.80–0.83 GB. The smoke numbers are a code check only.

## Design-stage prototype (local CPU, not of record)

`committed` with a fixed 500-epoch retrain and no early stop (an earlier version of the loop), run on the same banked
net: 68 rounds, 233 s, 36,065 → 689 weights (layer 0 144 scalars / 72 symbol connections, layer 1 165, head 380),
36 / 52 units, held-out 0.9986. Kept here because it set decision 6; the run of record uses the settle-or-cap retrain.

## Defects and fixes

- **W-D1 (walk inefficiency, no effect on the result).** Once ceil(f·alive) = 1, the remaining halvings retry the
  same smallest group from the same restored state with a deterministic retrain. When no other layer was admitted in
  between, the retry repeats the last outcome exactly. The final state is unchanged; only compute is spent
  (bilinear_both: 9 exact repeats, with train disagreements 358 / 356 / 80 on V / W / U each time). A
  cheaper walk would stop a layer at its first single-group rejection, or try the next-smallest group.
- None in the run otherwise. The design went through three changes before launch, each described above: SVD → QR for
  Pm (decision 2), fixed → settle-or-cap retrain (decision 6), and cap 2,000 → 500 after G3.

## Run of record `w1` (app `ap-rmQ9lRGPb9f8rd2HCLlGHZ`, 570 s wall, 7 containers, peak RSS 0.80–0.86 GB)

Facts only. The tables are in `figures/whittle_w1_table.txt` and the figure is `figures/whittle_w1_survivors.png`.

- **Walk mechanics.** Every arm ended by the floor, not the cap (36–89 rounds). Every arm has exactly 21 rejections:
  7 halvings × 3 layers. Every admitted round has 0 train disagreements and every rejected round has ≥ 1. The
  median retrain per admitted round was 99–450 epochs; 1–7 admitted rounds per arm ran to the cap. Retrain is 63–85%
  of arm time, the probe ≤ 1.5%, bookkeeping ≤ 0.1%; the rest is loading and the drift twin.
- **Banked finals reload.** Rebuilding committed, both_sides, bilinear_both and unrestricted from
  `/data/w1/<arm>_final.npz` gives the reported train agreement (1.0), held-out and nonzero-weight count exactly.
- **Final state** (weights = surviving weight scalars; effective = weights + live biases + head bias):

  | arm | start → final weights | layer 0 (scalars / symbol conns) | layer 1 | head | effective | units L0/L1 | held-out | drift twin final |
  |---|---|---|---|---|---|---|---|---|
  | committed | 35,712 → 817 | 148 / 74 | 209 | 460 | 1,017 | 37 / 66 | 0.9924 | 0.9947 |
  | unrestricted | 53,632 → 7,085 | 5,109 | 744 | 1,232 | 7,399 | 95 / 122 | 0.0824 | 0.9997 |
  | both_sides | 26,752 → 162 | 106 / 53 | 30 | 26 / 13 | 230 | 28 / 13 | 0.9979 | 0.9983 |
  | all48 | 53,632 → 786 | 200 / 100 | 192 | 394 | 989 | 50 / 56 | 0.9876 | 0.9236 |
  | committed_wd1 | 35,712 → 1,840 | 310 / 155 | 462 | 1,068 | 2,121 | 78 / 106 | 0.9945 | 1.0000 |
  | bilinear_both | 4,108 → 54 | U 18 / 9 | V 18 / 9 | W 18 / 9 | 81 | 9 product units | 1.0000 | 1.0000 |
  | committed_R2000 | 35,712 → 796 | 220 / 110 | 207 | 369 | 1,012 | 55 / 64 | 0.9429 | 0.9828 |

- **Surviving structure.**
  - committed: all 37 layer-0 units are single-symbol and read the same symbol on a and b. 11 of the 13 symbols
    survive; the two not read are DFT k 47 and 24 (oracle). There are 3 units per symbol (4 for k 18, 6 for k 11). 56
    of 66 layer-1 units read inputs of a single symbol (median fan-in 3); each writes to about 7 classes (head 460 / 66).
  - both_sides: 28 / 28 layer-0 units are single-symbol. 6 symbols are read (k 18, 36, 38, 39, 45, 47) and 5 are
    written (k 18, 38, 39, 45, 47). The k 36 symbol's single unit reads a only. 3 units read only one operand. 12 of 13
    layer-1 units read one symbol, and 13 of 13 write only symbols they read.
  - all48: 50 / 50 units are single-symbol and same-on-a-and-b. 14 symbols are read, all 2-D pairs, no DC. By DFT
    match (oracle), 11 are in the mint's 13 (not 38, 45), and 3 are outside it (12, 13, 22). 13 and 22 are the
    candidates the mint rejected at producer ranks 14 and 15, and 12 is rank 16. 47 of 56 layer-1 units read one symbol.
  - committed_wd1: 78 / 78 units are single-symbol and 77 / 78 are same-on-a-and-b. All 13 symbols survive, with 3–8
    units each. 48 of 106 layer-1 units read one symbol.
  - bilinear_both: 9 product units, each reading and writing one symbol on U, V and W. There are 3 symbols
    (DFT k 18, 24, 38) with exactly 3 units each, 6 weights per unit. The final probe α is 7.6 / 6.6 / 6.1 at those k,
    the gbar margin 5.74 and the test margin min 2.14. All 17 rounds after it reached 9 units were rejected.
    Single-group prunes gave 80–358 train disagreements; 2–3-group prunes gave 181–1,562.
  - unrestricted: 95 layer-0 units with a median 29 / 29 surviving one-hot weights on a / b. Read by projection on the
    recovered basis, their median concentration on the dominant symbol is 0.467, with 18 distinct dominants. 19 of
    122 layer-1 units read one symbol.
- **Held-out along the walk (logged, oracle).**
  - unrestricted falls monotonically: 0.978 at 38,784 weights, 0.877 at 18,959, 0.505 at 14,104, 0.30 at 11,009,
    0.082 at the end. The test margin min goes from +4.9 to −257. Its drift twin stays ≥ 0.981 (final 0.9997), so the
    fall comes with pruning, not with the retraining.
  - committed dips to 0.957 around 3–4k weights and ends at 0.992.
  - both_sides stays ≥ 0.993 throughout.
  - bilinear stays ≥ 0.9997 throughout.
  - committed_wd1 descends slowly to 0.992–0.995.
  - all48 moves between 0.918 and 1.0; its twin moves between 0.86 and 1.0.
  - committed_R2000 falls to 0.92–0.95 from about 5k weights on, while its twin is mostly ≥ 0.97 (min 0.946).
- **Drift twins** (nothing pruned, same epochs): the committed twin's final chunk lost train agreement on 88 pairs and
  held-out 0.876. The all48 twin's median held-out is 0.951 and its minimum 0.86. The wd 1 and bilinear twins sit
  at 1.0 throughout.
- **Logit-table energy at the end** (DFT read, oracle): diagonal / DC energy fraction
  - committed 0.271 / 0.715
  - unrestricted 0.022 / 0.846
  - both_sides 0.874 / 0.039
  - all48 0.232 / 0.753
  - committed_wd1 0.077 / 0.908
  - bilinear 0.972 / 0.003
  - committed_R2000 0.186 / 0.763

  At the start, m1's net read 0.734 / 0.229.

## w2: whittle to the floor in frequencies (`whittle2.py`, tag `w2`, 2026-09-23)

The brief: take the count to its honest floor with the move the w1 walk lacked, removing a whole frequency channel
at once, and with biases on the table. w1 (`whittle.py`) is untouched. Everything new is in `whittle2.py`, which
imports from `whittle.py`, so w1 still reproduces. G5 below reloads w1's banked finals and gets w1's counts and
held-out exactly.

### Decisions (w2)

1. **The channel move.** A channel is one minted symbol t ≥ 1 and everything that reads or writes it:
   - in the MLP, the layer-0 (unit, operand, t) groups, the head (t, unit) groups and t's head-bias group;
   - in the bilinear, the U and V (unit, t) groups, the W (t, unit) groups and t's head-bias group.
   The move removes the channel, runs the cascade (units left without inputs or outputs go; constants are folded),
   retrains with the long budget, and then applies the unchanged gate.
2. **Order: ascending total L2 norm, first success admitted.** Channels are offered in ascending total L2 norm over
   all their groups, which is endogenous. The first one that passes is admitted. A failed channel is restored and the
   next one is tried. The phase ends when every alive channel has failed from the current state, or when one channel
   is left.
3. **Long budget for channel moves and the plant.** The cap is 20,000 epochs, with early stop once train agreement
   has held 100 consecutive epochs. G8 shows a zero-prune 20k retrain keeps agreement 1.0 on both the bilinear and
   the MLP. Group moves keep w1's budget (cap 500, settle 100), so the group walk is w1's walk unchanged.
4. **Biases are first-class groups.** They join the round robin as layers of their own:
   - per unit for layer-0 and layer-1 biases;
   - per symbol for the minted head bias (27 entries = 14 groups: DC plus 13 pairs), so the whole head bias is on
     the table.
5. **Revival on fold.** In w1 a constant unit's fold went into a bias that could not have been pruned. Now it can
   land in a pruned bias entry, so that entry (the whole group, for the head bias) is revived to keep the function
   exact (G6c). The fold removes the unit's bias and its k ≥ 1 outgoing weights and revives at most k bias entries, so
   neither count rises.
6. **Phases alternate.** A channel phase (to exhaustion) is followed by a group phase (w1's walk over the weight and
   bias layers, fractions reset to 0.5, to its floor). If the group phase admitted anything, the channel phase runs
   again. The walk stops when a full cycle admits nothing. bil_from_start therefore meets channel moves first, with
   13 channels.
7. **Plants.** Start from w1's bilinear final and keep only symbol t's groups in U, V and W. Every head-bias group and
   the DC head row are zeroed and masked. The cascade leaves exactly 3 units and 18 weights (G7). One long retrain
   follows; if it passes the gate, the group walk runs on what is left. The symbols are s4, s9 and s13, the three
   alive in w1's final (DFT k 38, 18, 24; oracle). Arms are named by symbol index.
8. **Counts in two currencies.** Scalars (weights, biases, total) and groups (symbol connections in minted layers,
   scalars elsewhere; bias groups as in decision 4). A scalar zero inside a 2-vector depends on the arbitrary rotation
   inside the subspace, so only the group count is a claim.
9. **Explicit read of a one-channel bilinear** (`explicit_bilinear`).
   - For each alive channel: the live units' U and V restricted to the symbol's 2 input coordinates, and W on its 2
     head rows.
   - The 2×2×2 tensor T = Σ_i U_i ⊗ V_i ⊗ W_i, and the least-squares fit T ≈ A·Re(e^{iψ} z_a z_b z̄_c). This is the
     complex product up to a rotation of the frame; a reflected frame lands in the same family.
   - The fit's relative residual.
   - The symbol's eigenvalue phase at g_ref (endogenous) and its DFT k (oracle).
   - The frame phase φ0 at the identity symbol ê (endogenous). An exact solution peaking at c = a+b has ψ = −φ0.
   - The margin on all p² pairs, saved per pair in `/data/w2/<arm>_final.npz`.
10. **Variants.**
    - `mlp_channel_sgd` (SGD lr 0.1, momentum 0.9, every retrain) runs in parallel rather than conditionally. It
      costs no wall-clock and saves a relaunch.
    - `mlp_channel_lr1e-2` (AdamW lr 1e-2) runs because the MLP's amplitude growth was the open question.
    - `bil_channel_lr1e-2` is defined but not run. The brief made it conditional on slow growth, and the bilinear's
      growth was not slow: in the local prototype the last channel removal passed at epoch 2,746.
11. The optional L0-gate route is not built. The brief made it conditional on the walk-based routes stalling.

### Gates (w2; `whittle2_gates`, Modal, 344 s, `results/gates_whittle2.log`; all pass)

- **G5:** reloading w1's banked finals gives bilinear_both 54 weights / 0 train disagreements / held-out 1.0 and
  both_sides 162 / 0 / 0.99787, identical to w1's record.
- **G6a:** on the bilinear final, removing each of the 3 channels removes exactly its 3 units (9 → 6 units, 54 → 36
  weights). The logits then equal the originals minus the channel's own terms to ≤ 5.7e-6.
- **G6b:** on the MLP final, removing each of the 6 channels gives logits equal to hand-zeroing that channel's groups
  to ≤ 1.1e-5, constant units folded included.
- **G6c:** with the whole head bias pruned, 3 layer-1 units made constant are folded exactly (1.5e-5), and 4
  head-bias entries are revived.
- **G7:** each plant starts at exactly 3 units / 18 weights / 0 biases. Train agreement at the plant is 0.122 (s4),
  0.287 (s9) and 0.190 (s13).
- **G8:** zero-prune 20k retrain, no early stop: train agreement 1.0 and held-out 1.0 at every 2.5k checkpoint for
  bilinear AdamW at lr 1e-3 and at lr 1e-2, MLP AdamW and MLP SGD. The MLP's held-out rises from 0.9979 to 1.0. The
  gbar margin plateaus near 17.1 on the bilinear and rises to 39.6 (AdamW) and 21.1 (SGD) on the MLP.
- **G9:** walk2 on a designed case (3 channels, only one removable, norm order 3 < 1 < 2): it tries 3 (fails,
  restores), admits 1, then tries 3 and 2 (both fail), and the group phase ends at the planted cap.
- **Smoke** `w2smoke` (cap 300 / 60, 12 rounds): all 8 arms run end to end in 30 s. The plants do not pass at 300
  epochs; this is a code check only.

### Design-stage prototype (local CPU, not of record)

`bil_channel` at the full budget:
- removed s4 (446 epochs) and s13 (2,845 epochs), leaving s9 (k 18);
- the group walk then pruned the whole head bias (14 groups) and nothing else, ending at 3 units / 18 weights /
  9 groups / 0 biases, held-out 1.0;
- every pair's margin was positive (min 0.214);
- the fit gave A = 146.1, ψ = −2.6869 against frame phase +2.6867, relative residual 0.006.

`bil_plant1_s9` passed after 1,125 epochs.

### Run of record `w2` (app `ap-lkQC8iuPAWsIurZfoQRloL`, 847 s wall, 8 containers, peak RSS ≤ 0.84 GB)

Facts only. The tables are in `figures/whittle_w2_table.txt` and the figure is `figures/whittle_w2_walk.png`. Every
arm ended on its own terms, none at the round cap. Reloading `/data/w2/bil_channel_final.npz` and
`bil_from_start_final.npz` reproduces 18 / 36 weights, 0 disagreements, held-out 1.0, and 0 non-positive margins out
of 9,409.

| arm | final: weight scalars / groups | bias scalars / groups | total scalars / groups | channels (k) | units | held-out | min margin, all pairs |
|---|---|---|---|---|---|---|---|
| bil_channel | 18 / 9 | 0 / 0 | 18 / 9 | s9 (18) | 3 | 1.0000 | 0.2143 |
| bil_plant1_s4 | 18 / 9 | 0 / 0 | 18 / 9 | s4 (38) | 3 | 1.0000 | 0.0069 |
| bil_plant1_s9 | 18 / 9 | 0 / 0 | 18 / 9 | s9 (18) | 3 | 1.0000 | 0.0057 |
| bil_plant1_s13 | 18 / 9 | 0 / 0 | 18 / 9 | s13 (24) | 3 | 1.0000 | 0.0047 |
| bil_from_start | 36 / 18 | 0 / 0 | 36 / 18 | s4 (38) | 6 | 1.0000 | 0.0359 |
| mlp_channel | 68 / 40 | 15 / 13 | 83 / 53 | s7, s8 (39, 47) | 12 / 4 | 0.9979 | train 0.024, held-out −0.694 |
| mlp_channel_sgd | 68 / 40 | 9 / 6 | 77 / 46 | s7, s8 (39, 47) | 12 / 4 | 0.9995 | train 0.115, held-out −0.419 |
| mlp_channel_lr1e-2 | 68 / 40 | 8 / 6 | 76 / 46 | s7, s8 (39, 47) | 12 / 4 | 0.9994 | train 0.006, held-out −0.212 |

- **bil_channel.**
  - Channel moves: s4 (k 38) was removed in 446 epochs, then s13 (k 24) in 2,846 (first 100% at 2,747). No channel
    move was rejected.
  - Amplitudes α_k: before 6.1 / 6.6 / 7.6 (k 38 / 24 / 18); after the first removal 12.0 / 13.6 (k 24 / 18);
    after the second 120.2 (k 18).
  - gbar margin 5.74 → 0.647 → 0.251.
  - The group phase then pruned all 14 head-bias groups (4 admissions). Every weight-group prune was rejected
    (2,513–2,770 train disagreements).
- **Plants.** All three pass the gate after one long retrain, with none of the other channels, no bias and no DC row:
  - s4 (k 38): 2,057 epochs;
  - s9 (k 18): 1,125 epochs;
  - s13 (k 24): 2,057 epochs.
  Train agreement at the plant was 0.12 / 0.29 / 0.19. Afterwards α_k is 52.0 / 27.4 / 49.8 and held-out is 1.0.
  Every group prune after the plant was rejected.
- **bil_from_start.**
  - Channels were removed in ascending norm order: k 10, 47, 33, 46, 44, 18, 24, 45, 30, 39, 36, then 11. The first
    11 took 99 epochs each and agreement never broke; the 12th took 242.
  - After that all 52 product units read only s4 (k 38): 364 weights with the DC head row.
  - The group phase took it to 6 units / 36 weights / 0 biases. It rejects every single-group prune from 6 units
    (673 train disagreements for U or V, 1,817 for W at the 500-epoch group budget).
- **MLP arms (all three optimizers).**
  - The same 4 channels went in the same order (k 36, 45, 38, 18). The last of these took 15,516 (AdamW 1e-3),
    3,937 (SGD) and 2,215 (AdamW 1e-2) epochs.
  - Removing either of the last two (k 39, 47) fails at the 20k cap in every arm: 1,758–2,101 train disagreements,
    held-out 0.27–0.35. Each arm tried both channels twice (4 rejections).
  - All three end in the identical structure: per channel, 6 layer-0 units (all single-symbol, same symbol on a and
    b) feed 2 layer-1 units (fan-in 3 each), which write only that symbol.
  - No weight group was pruned after the channel phase at the 500-epoch group budget; only bias groups went.
- **Explicit one-channel reads (bilinear).** All 5 bilinear finals fit T ≈ A·Re(e^{iψ} z_a z_b z̄_c) with relative
  residual 0.0059–0.0230, and in each ψ + φ0 = 0 to within 0.003 (φ0 = frame phase at ê):
  - bil_channel: 3 units, A = 146.2, ψ = −2.68687, φ0 = +2.68672, relative residual 0.0059; predicted margin
    A·(1 − cos 2π/97) = 0.307, observed minimum 0.214.
  - The plants: A = 52.0 / 27.4 / 49.8, residuals 0.016 / 0.023 / 0.015; observed minimum margins 0.005–0.007 against
    predicted 0.057–0.109.
  - bil_from_start: 6 units, A = 67.3, residual 0.009.
  - The eigenvalue phase at g_ref is endogenous: s9 = 2π·48.00000/97, s4 = 2π·27.99997/97, s13 = 2π·33.00001/97.
    These equal fold(k·62 mod 97) for k = 18, 38, 24.
- **Drift twins.** Every twin kept 0 train disagreements. Held-out minimum was 0.9997 (MLP arms) and 1.0 (bilinear).
  The plants have no twin: no rounds were admitted after the plant.
- **Logit-table energy at the end** (DFT read, oracle):
  - bilinear: diagonal fraction 0.9995–1.0000, n90 = 1;
  - MLP: 0.987–0.988, n90 = 2.
- **Compute.** The MLP arms took 772–836 s, mostly the rejected 20k channel attempts (t_channel ≈ 500–520 s). The
  bilinear arms took 32–143 s.

## Withdrawals

None.
