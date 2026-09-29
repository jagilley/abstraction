# grokking/basis — NOTES (decisions, gates, defects)

Working notes for the builder and the orchestrator. Not a README: Jasper discusses results before one is written.
Up: [`../NOTES.md`](../NOTES.md) (the m1 record this node reads). Table of record: `figures/basis_b1_table.txt`.

## Setup as run (tag `b1`, 2026-09-22)

- No retraining. Nets: m1's banked snapshots on `grokking-mint-data` (`/data/m1/true/snapshots.npz`, 162 snapshots;
  `/data/m1/shuffled/snapshots.npz`, 42) and the untrained inits of seeds 0..3 rebuilt from their seeds. That is 208
  records, the same set m1 analysed. Single seed (42), the banked one.
- One CPU container (`basis_run`, cpu 4, memory 2 GB), app `grokking-mint`, profile `chromatic`. The random-order
  controls (5 permutations) run at the snapshots m1 used: every 1000 epochs, init and last, all shuffled, all inits.
- The operator, per snapshot: the net's softmax over the full p² table (one forward pass), then
  M_g[c, a] = softmax(net(a, g))[c] for the 6 generators.

## Decisions

1. **Operator slot and alphabet.** The generator sits in the b slot (a ↦ net(a, g)). The output class c and
   the input symbol a are identified by index. That is the one identification the task gives: it is the same
   one m1 used when it applied the DFT to both halves of layer 0. No integer order is used anywhere else outside
   the oracle.
2. **Generators.** Six symbols drawn once with `RandomState(42).choice(97, 6, replace=False)`: (62, 40, 93, 18,
   81, 83), fixed across snapshots, with g_ref = 62 (the first drawn). The draw does not contain 0. Symbol 0 is the
   identity, and its operator would be I, whose eigenspaces are fully degenerate.
3. **Candidates** = every real invariant subspace of M_g_ref. Each complex-conjugate eigenpair gives a 2-D
   subspace, orth[Re v, Im v]. Each real eigenvalue other than the one closest to 1 gives a 1-D subspace. The brief
   names only the pairs. The 1-D subspaces exist only when the operator is far from a permutation (1 real
   eigenvalue at the final net; 7–27 on non-grokked nets). Including them keeps the candidates plus the DC spanning
   R^p, so forms B and C at K = all reproduce the net at every snapshot, as in m1.
4. **Recovered DC** = the eigenvector at eigenvalue 1 (the right Perron vector, i.e. the stationary distribution).
   It is not the all-ones vector. Its overlap with the all-ones vector is logged (`dc`). Form C always keeps it, and
   form B excludes it from unit dominance, as m1 did with the DFT DC.
5. **Overlap measure**: ||Q1ᵀQ2||²_F / max(dim1, dim2), which is 1 iff the subspaces are equal. It is used for both
   the certificate and the oracle.
6. **Certificate** (no labels). For each candidate of M_g_ref, take its best overlap with the candidates of
   each other M_g, then the minimum over the 5 other g. It is reported as mean, min and fraction > 0.99 over
   candidates. The commutator ||[M_g, M_h]||_F / ||M_g M_h||_F is logged beside it.
7. **Phase check** (oracle). For each 2-D candidate and every g, the phase of the restricted operator QᵀM_gQ is
   compared with the folded true angle 2π·k·g/p at the candidate's DFT-matched k. For a soft circulant operator the
   eigenvalue is φ̂(k)·χ_k(g)⁻¹, so the phase equals the character's only when the kernel's Fourier coefficient
   φ̂(k) is positive (the net's kernel peaked at the right answer). Designed check D-4 is built that way.
8. **Form A needs a phase reference (an addition to the route).** A recovered eigenvector is a character only up to
   a global complex scalar c. Re[c·χ(a)χ(b)χ̄(c′)] with arg c ≠ 0 predicts a *shifted* sum. The phase is fixed
   at the net's own identity symbol ê = argmax_s mean_a softmax(net(a, s))[a], which is endogenous (it reads the full
   table, no labels), and ψ is scaled to rms 1. With exact characters and ê = 0 this is m1's form A exactly (I-1).
   1-D candidates enter A as ψ(a)ψ(b)ψ(c).
9. **ALS** is refit by joint least squares at every audited K, with per-row intercepts (closed form from a
   precomputed Gram matrix). With exact characters the design is orthogonal and this equals m1's per-frequency
   cosine coefficient (I-1, I-3).
10. **Form C** is the orthogonal projection of every layer-0 row onto span(recovered DC ∪ candidates in K). m1's FFT
    filter is exactly this with the DFT basis. For a non-normal operator the oblique eigen-projection would differ;
    the orthogonal one was chosen.
11. **Random-order controls** permute candidate *indices* (`RandomState(1000+i).permutation(m)`). Candidate index
    order is not DFT-k order, so the random orders differ from m1's even at the final net. Only the producer-order
    walks are comparable to m1.
12. **K comparison.** A kept set of candidates is reported as its set of DFT-matched k (argmax overlap). This set can
    be smaller than k when two candidates match the same frequency. The mean DFT overlap of the kept candidates sits
    beside it (`ov`), so a match made at overlap ~0.05 reads as what it is.

## Gates (`basis_gates`; all passed before the run; `results/gates_basis.log`)

Designed operators (steps 2–4 only):
- **D-1 exact shift** a ↦ a+g (the exact 97-cycle): DFT overlap min 1 − 1e-15, one-to-one; phase error max 3.9e-14;
  certificate min 1 − 1e-15; commutator 0.
- **D-2 random permutation per g** (5–9 cycles): DFT overlap mean 0.061, certificate mean 0.053; phase error up to 3.1.
- **D-2b random 97-cycle per g**: DFT 0.065, certificate 0.054. Its eigenvalues are still exact 97th roots of unity
  (quantization residual 0), so eigenvalue quantization alone does not separate a random cycle from the shift.
- **D-3 relabelled action** σ∘shift_g∘σ⁻¹: certificate 1 − 1e-15, DFT (integer labels) 0.068, and overlap with the
  relabelled characters χ_k∘σ⁻¹ 1 − 1e-15. The certificate certifies a group action. It does not certify that the
  labels are the integers.
- **D-4 soft shift** 0.7·shift_g + 0.3·circulant(even kernel): DFT 1 − 1e-15, phase 5e-14, |λ| down to 0.44.

Instrument checks on banked nets:
- **I-1** the exact DFT basis injected into this pipeline reproduces m1's recorded K for all 12 form × rule gates, and
  held-out to 0.0, at e39999 and at e14000. Producer scores match to 5e-15 and unit dominants are identical.
- **I-2** recovered basis: form B at K = all bit-exact; form C at K = all max |Δlogit| 7.6e-6 (e39999), 9.5e-6
  (e14000), 3e-8 (untrained seed 0).
- **I-3** ALS closed form vs explicit lstsq with per-row intercepts: 5e-17 / 7e-16.

## Defects and fixes

- **B-D1** Launch failure "`results/launch_r1.log` was modified during build process". Importing `grokking.mint`
  makes its Modal functions package-sourced, so Modal adds an entrypoint mount of the whole `grokking/` folder
  with no ignore filter (logs, JSON and PNGs included). The r1 client was still writing its log. Launched with
  `MODAL_BUILD_VALIDATION=warn` (in `RUN_basis.sh`). No code change and no effect on numbers. A durable fix would
  be `include_source=False` on mint.py's functions, or an ignore on the entrypoint mount, both outside this node.

## Run of record `b1` (app `ap-0NfFRsnsjVj5VNdVV56eQi`, 564 s, peak RSS 1.05 GB, 208 records)

Facts only; the tables are in `figures/basis_b1_table.txt`, the figure is `figures/basis_b1_read.png`.

- **In-run instrument checks.** Form B at K = all is bit-exact at 208/208 snapshots. Form C at K = all has max
  |Δlogit| 1.7e-5 and argmax agreement 1.0 everywhere.
- **Final net (e39999).** All 6 hard maps are single 97-cycles and equal to the shift. The read gives 48 pairs and 1
  real eigenvalue. DFT overlap min 0.9999999, one-to-one. Certificate min 0.9999995. Commutator 2.3e-4. |λ| ≥ 0.99998.
  Phase error max 1.4e-5 over all candidates and all 6 g, with p·θ/2π equal to fold(k·g_ref mod p) for all 48. DC
  overlap with ones 0.9999999. Producer scores match m1's at the matched k to 2.8e-4 relative; producer order and all
  128 unit dominants are identical. Kept K (as matched k) equals m1's K at all 12 form × rule gates, and held-out is
  identical (B/lt1 13 freqs 0.9974, C/lt1 13 freqs 0.9994, B/m3 12, C/m3 13; A and ALS k = 1 under lt1 and m3).
- **Along the curve** (first snapshot at which each series stays ≥ 0.99 through the end): net test 16,250; DFT overlap
  mean 17,250, min 17,500; certificate (cross-g) mean 19,500, min 19,750; all 6 hard maps bijective from 18,500.
  - At test 0.978 (15,000) DFT overlap is mean 0.57 / min 0.03; at test 0.990 (16,000) it is 0.62 / 0.12; at 17,000
    (test 0.997) 0.98 / 0.97. Certificate over the same epochs: 0.20 / 0.01, 0.27 / 0.04, 0.60 / 0.39.
  - From 250 to 15,000 (memorisation and early grokking): certificate mean 0.017–0.20, DFT mean 0.05–0.57, 0 of 6
    bijections, 3–25 real eigenvalues.
  - ê = 0 at every true-run snapshot from epoch 250 on (test acc 0.000 there).
- **K vs m1** (matched k set equal to m1's K), over the 95 snapshots from 16,500 on: B/le0 95, B/lt1 95, B/m3 93,
  C/le0 93, C/lt1 92, C/m3 90. The mismatches after 17,000 are single swaps ({22,47} at 17,000–18,000; {12,47} at
  19,250 and 19,750). |held-out(rec) − held-out(m1)| after 17,000 is ≤ 0.0027 (lt1) and ≤ 0.0083 (m3). Before 16,250
  the matched sets differ from m1's, and the kept candidates' mean DFT overlap is ≤ 0.69, so the matching there is
  weak.
- **Form A / ALS along the curve.** Between 16,500 and 25,000 the recovered-basis A and ALS gates keep 2–12 candidates
  (lt1) where m1 keeps 1, with held-out 1.000 (lt1) and ≥ 0.998 (m3) from 17,000. A single recovered candidate reproduces
  only part of the net's train argmax: 27% at 20,000 (its DFT overlap 0.9998) and 70% at 24,000 (0.999999). From
  25,250 on, k = 1, as in m1. Before 16,500, A/lt1 keeps 3–27 with held-out 0.01–0.81.
- **Falsifiers.**
  - Shuffled-label net (40 trained snapshots): certificate mean 0.008–0.033, min ≤ 0.006; DFT mean 0.039–0.052, min
    0.018–0.023; 0/6 bijections everywhere; commutator 1.37–1.41; ê wanders (9, 40, 90, 18). Walks (k, held-out at
    chance 0.007–0.014 throughout): A le0 5–34, lt1 2–12, m3 0–1; ALS le0 2–29, lt1 2–12, m3 0–1; B lt1 25–43, m3 23–34;
    C lt1 53–63, m3 0–2.
  - Untrained inits (seeds 0–3 + the seed-42 init): certificate mean 0.09–0.11, min 0.026–0.037; DFT mean 0.059–0.062;
    0/6 bijections; operator nearly uniform (conf 0.012, median |λ| 0.000). Walks: A le0/lt1/m3 2–7 / 2–7 / 2–5; ALS
    5–7 / 3–6 / 1–4; B lt1 0–17, m3 0–10; C lt1 0–43, m3 0–7; held-out at chance.
  - m1's form A admitted nothing on any falsifier under any rule. Over the recovered basis, A and ALS admit on the
    untrained inits even under m3.
- **Certificate floor.** The untrained inits sit at certificate mean ~0.10, above the memorising true run (0.017–0.05
  at 250–11,000) and the shuffled net (0.008–0.033).

## Flags (places the route as briefed needed an addition or behaves differently from m1)

- Form A over a recovered basis needs a phase reference, the identity symbol (decision 8). The recovered subspaces
  alone do not fix it.
- The m1 closed-form degeneracy (argmax a+b at every nonempty K) holds only when the recovered basis is exact to
  roughly 1e-6 in overlap. Along the curve (16,500–25,000) and on falsifier nets, A / ALS are not degenerate: they keep
  more candidates or admit noise.
- The eigenvector read is sharp. It stays near chance until test accuracy is about 0.99, and the certificate turns on
  about 2,000 epochs after the oracle.
- The certificate certifies a group action, not the integer labelling (D-3).

## Withdrawals

None.
