# Notes — `digit_port`

Running log. What each run was, what it read. No README until the numbers have been
discussed (repo convention).

**Up**: [../README.md](../README.md) (`second_pass` — the writeup) · [../../README.md](../../README.md) (`dress_rehearsal`) · **Files**: [FILES.md](FILES.md)

---

## What this node is

The arm `dress_rehearsal` §5 and next-step 1 point at: replace the parent port's
**pooled-state** operator with a **per-digit** one, keep loop-on-`T`, and read the fresh-`x`
`T=1` rung on the fixed-`N` Medium instruments (`m6`–`m10`, floor ≈4 of 140–180; `m1`/`m2`
for the "`T=1` never trained" shape Hard probably has).

### The design, in one paragraph

The recurrent state is a **fixed-width digit grid**: `S` slots, slot `j` = the `j`-th digit
from the right, zero-padded on the left, each slot a distribution over the vocabulary. The
initial state is `x` right-aligned; the tied operator maps a digit grid to a digit grid; the
answer is the grid after `T` applications, scattered to where `target_positions` reads. The
encoder runs over the **`N` field only** and the operator cross-attends to it, and `N`'s own
digits are added per slot, so the operator is a pure function of `(state, N)` — `x` enters
only as the seed and `T` only as the loop count. The forward pass is exactly `y = f_N^T(x)`.

**Why this might move the rung.** With a canonical carrier, `T = 2` supervision *is*
supervision of `f ∘ f` with the same `f`. The parent's pooled state has no such
factorisation: its rollout passes through vectors no encoder ever produced, so `T = 1`
(the rung that gates the whole ladder) is not identified by the deeper rungs. Here it is.
On `m6`–`m10`, `T = 1` is also directly supervised; on Hard it probably is not, which is
where the factorisation has to carry the weight.

**Positional scheme**: abacus — every token carries a field id (`N`/`x`/`T`) and its index
from the right *within its field*, both derived by counting marker tokens.

**Initialisation**: the operator's residual branches are zero-initialised and the digit
head is tied to the state embedding, so at `t = 0` the step is a near-identity copy. Local
check: grid argmax equals `x`'s digits for 100% of slots after 8 applications with
`head_scale = 3.0`; at `head_scale = 1.0` it is 0.02 (the grid washes out before training
starts). See the sweep under "Local checks".

### Legality — the part worth flagging

Jasper's standing reading for this round: anything not explicitly forbidden is legal, and
tokenizer knowledge used for control flow is fine. This file uses marker token ids for four
things: the `T` parse (iteration count, as in the parent), the abacus positional
embeddings, the encoder's `N`-field attention mask, and the right-aligned gather that seeds
the grid. None of them reads a digit's *value*; digit values are only ever consumed by
learned embeddings. The gather is the most aggressive of the four — it is the hard-attention
limit of a learned copy — so `ARM = "digit_learned"` keeps everything else and replaces it
with a learned slot-query cross-attention, which makes the price of the conservative
reading measurable rather than assumed. **Not yet asked on Discord.**

The loss's padding term asks grid slots above a row's answer width to hold the zero digit.
It uses the evaluator's own `valid_mask` and a constant symbol, and says nothing about the
answer's value; it is what makes the carrier canonical, so one application of the operator
is the same map at `T = 1` as it is inside `T = 2`.

---

## Local checks (CPU, upstream runner at `4ceff95`)

`probe.py` against real generated prompts, all arms emitted by `emit_arm.py`:

| check | result |
|---|---|
| layout parse + right-aligned gather recovers `(N, x, T)` | 64–96/96 rows exact on `m6`, `m5`, `m1`, `m4`, `e5` |
| scatter round-trip (grid slot → `target_positions`) | max &#124;Δ&#124; = 0 on every valid target |
| near-identity at init, `T=8` (`m4`) | copy fidelity 1.000 at `head_scale` 3.0/4.0; 0.977 at 2.0; 0.016 at 1.0 → **init 3.0** |
| model state | 5.82M elements (ceiling 500M) |
| `one-layer validate` | valid, 28,707 bytes |
| upstream CPU smoke (`N=143`, 40 s, batch 32) | 257 steps, ran clean, mean exact 0.088 |

---

## Runs

| # | tier · dataset · arm | id | what it tests | read |
|---|---|---|---|---|
| 1 | medium · m6 · `digit` | `089e96f8` | the per-digit carrier against the parent's `m6` numbers (control v2 **2**/140, closure v2 **4**/140, closure v3 **1**/140 at fresh-`x` `T=1`, floor 4) | *in flight* |


### Run 1 — Medium `m6`, arm `digit` (`089e96f8`), 2026-08-22

600 s, 8,929 steps, batch 512, 5.82M state elements. Train exact 0.006 → 0.99 (crossing
0.5 at 164 s, 0.95 at 406 s, 0.99 at 525 s), train loss 7.38 → 0.022.

| | parent control v2 | parent closure v2 | parent closure v3 | **digit** | floor |
|---|---|---|---|---|---|
| fresh-`x` `T=1` (of 140) | 2 | 4 | 1 | **27** | 4 |
| seen-`N` rungs `T=1…64` | 2 0 2 0 3 1 0 | 4 1 4 1 0 0 0 | 1 0 1 2 3 2 3 | **27 26 27 27 27 28 28** | |
| `ood` (`T=8`, held-out depth, seen `x`, of 500) | 126 | 8 | 95 | **496** | |
| `test` (held-out prompts, of 390) | 30 | 45 | 44 | **275** | |
| OOD-`N` rungs (of 256) | 0 1 3 0 1 1 0 | 1 0 0 1 0 0 0 | 3 2 2 2 2 0 0 | 1 1 0 0 0 0 0 | ~2 |
| mean exact | 0.164 | 0.066 | 0.151 | **0.849** | |

**The two things this says.**

1. **Composition is solved on this modulus.** `T=8` — a depth never trained — reads
   **496/500** on seen `x`, where the parent's best arm read 126/500. And the fresh-`x`
   rung is *flat in depth*: 27, 26, 27, 27, 27, 28, 28 across `T = 1…64`. Sixty-four
   applications of the tied operator lose nothing. Whatever `f` gets right on a fresh `x`
   it gets right at every rung, which is the ballistic property the carrier was built for.
2. **The single-step map partially generalises to fresh `x`, and that is now the only
   thing failing.** 27/140 = 19.3% against a 2.9% floor, ~7× floor and 7× the parent's
   best. Not the "exactly the floor count" target of `dress_rehearsal` §5's tier-2 read —
   far above it — but not the rule either.

So the failure has moved: it is no longer "the rollout goes somewhere no encoder produced",
it is "`f_N` on a fresh `x` is right about one time in five".

**Not carried over:** the OOD-`N` profile (1 1 0 0 0 0 0 of 256) is at/below floor, as
expected — `m6` trains one modulus, so the `N` conditioning has nothing to generalise from.

**Correction carried from the coordinator**: the parent README's "~84%" 12-bit coverage
figure is wrong; measured coverage is **58%** of distinct `(N, x)` pairs (the 84% double-
counted rows across the three `T` settings). The dense cell is still dense, less so.

### Run 2 — Hard `h1`, arm `digit` (`aa3596ee`), 2026-08-22

Today's Hard attempt, spent on the arm run 1 measured. Standing best is the parent's
closure v2 at one correct example (rank 31). `left: 0 today` after acceptance. ~1.5 h.

**Read.** Dead flat: loss 2.219 at step 2,100 and 2.183 at step 30,700 — a drift of 0.036
nats over 28,600 steps, against `ln 10 = 2.303`. Train exact never above 0.004. This is not
slow fitting, it is **stuck**. Two facts frame it:

- **Throughput.** 32,755 steps in 3,600 s = 9.1 steps/s, against the parent's 31 steps/s.
  The parent's control v2 broke its Hard plateau at step ~35k — a step count we did not
  reach. The step is launch-bound, not FLOP-bound: `T = 8` × a 4-layer operator = 32
  transformer layers per forward on `[512, 14, 256]` tensors, ~500 tiny kernels.
- **The missing `T = 1` row.** On `m6` (`T ∈ {1,2,4}`) `f` is supervised directly and the
  model crossed 0.5 train exact in 164 s. On Hard (`T ∈ {2,4,8}` if it is `m5`-shaped) the
  shallowest row is two applications of a **hard digit bottleneck** with an unsupervised
  intermediate — the model must bootstrap `f` from `f ∘ f` with no anchor. The parent's
  pooled state has no bottleneck and can memorise straight through it. This is the
  hypothesis the carrier's own design makes most likely, and it predicts the flat curve
  better than throughput does.

### Sweep s1 — free 600 s H100 runs on our own Modal (2026-08-22 22:5x UTC)

The hosted Medium tier is ~6/day and serialised; every public dataset is generated by
upstream's own script, so a 600 s Medium run reproduces on our own H100 with the same
evaluator, same data, same GPU class. Only Hard's data is hidden. `modal_sweep.py` is that
instrument — the hosted tiers are now needed only for Hard and for confirmation.

| tag | dataset | what it separates |
|---|---|---|
| `s1_m5_base` | `m5` (`[12,14,16]`, `T∈{2,4,8}`) | Hard's faithful proxy at 1/6 clock — control |
| `s1_hD_base` | `hD_t_shallow` (**same widths, same volume, `T∈{1,2,4}`**) | the *only* difference is whether `f` is supervised directly |
| `s1_m5_bs2048` | `m5`, batch 2048, LR 2e-3 | epochs vs updates: if the step is launch-bound a 4× batch is nearly free |
| `s1_m5_carry` | `m5`, arm `digit_carry` | does bypassing the digit bottleneck restore memorisation? |
| `s1_m5_wide2` | `m5`, `D_MODEL` 512, `D_FF` 2048, `N_OP_LAYERS` 2, batch 1024 | same launch count, 4× arithmetic per launch |

`m5` vs `hD_t_shallow` is the controlled pair: same generator, same widths, same
10k/setting, one knob.

### Run 3 — Medium `m1`, arm `digit` (`8ec992f4`), 2026-08-22

`m1` = fixed `N` = 10,403, `T ∈ {4,8,16}`, 30k rows, fresh-`x` cohort 192. Fixed `N`, so
the modulus conditioning is trivial and the *only* difference from `m6` is that `T = 1` is
never trained. Fires the same hypothesis on the other axis: if a fixed-`N` set without a
`T = 1` row also plateaus, the missing anchor is the blocker, not variable `N`. Caveat
recorded in advance: `T` up to 16 makes the step ~2× slower again, so a plateau here is
partly confounded with epochs.

### Run 3 read — Medium `m1` (`8ec992f4`)

4,964 steps (8.3/s — `T` up to 16), train loss **2.3166** against `ln 10 = 2.3026`, i.e. at
or slightly *above* uniform for the whole run. 0/3000 on `test`, 0/3000 on `ood`, **0/192**
at every seen-`N` rung, 0/512 at every OOD-`N` rung. `m1` is **fixed `N`** — no modulus
family to learn — and the model did not learn even the answer's digit marginals.

### Sweep s1 read — all five variants bolted

| tag | steps | loss @900 → end | mean exact |
|---|---|---|---|
| `s1_m5_base` | 8,660 | 2.200 → 2.180 | 0.0031 |
| `s1_hD_base` (`T∈{1,2,4}`) | 13,431 | 2.19 → 2.193 | 0.0025 |
| `s1_m5_bs2048` (116 epochs) | 4,601 | 2.199 → 2.194 | 0.0018 |
| `s1_m5_carry` (no digit bottleneck) | 7,744 | 2.391 → 2.207 | 0.0029 |
| `s1_m5_wide2` (`D_MODEL` 512, 2 op layers) | 9,979 | 2.18 → 2.188 | 0.0023 |

Nothing moves after step ~900. **Both** original hypotheses die here: `hD_t_shallow` has
`T = 1` rows and still plateaus (so the missing anchor is not it), and `s1_m5_carry`
bypasses the digit bottleneck and still plateaus (so the bottleneck is not it). There is no
descent to extrapolate, so "needs more updates" dies too — the parent's curve *descended*
before it broke; ours is flat.

### The actual cause: the rollout has no gradient path

`probe`-style measurement on `m1` (`T = 4`), `|dL/d state|` at each application after 40 and
80 updates:

| state mode | application 1 | 2 | 3 |
|---|---|---|---|
| `soft` (runs 1–3) | 4.0e-06 | 2.0e-04 | 1.0e-02 |
| `st` (straight-through) | 2.0e-08 | 8.4e-06 | 3.3e-03 |
| **`residual`** | **4.1e-01** | **4.1e-01** | **2.5e-01** |

The carrier decayed the gradient **~50× per operator application** (`soft`), so by
application 1 nothing survives. That explains the whole ordering at a stroke:

- `m6` (`T ∈ {1,2,4}`): `T = 1` rows supervise application 1 **directly**, zero applications
  of decay → fits, 27/140.
- `m5`/`hD` (min `T` = 2 for m5, 1 for hD but 204 moduli): one or two applications of decay
  → learns the digit marginals (2.19 < `ln 10`) and stops.
- `m1` (min `T` = **4**, fixed `N`): three applications of decay before any label →
  **exactly uniform**, learns nothing at all.
- Hard (min `T` = 2): 2.20, marginals only, flat over 32,755 steps.

**The fix**: carry the grid as *logits* and make each application add a correction —
`raw_k = raw_{k-1} + head(operator(softmax(raw_{k-1})))`. The identity path delivers the
loss to the operator's parameters with the same magnitude at application 1 as at
application `T`; measured profile is flat (0.24–0.41 at every application). Canonicality is
kept (the digit string is `argmax(raw)`), and the copy-at-init property becomes exact by
construction (delta = 0 at init → grid argmax equals `x`'s digits, verified at `T = 8`).
`STATE_MODE ∈ {"residual", "st", "soft"}`; `"soft"` reproduces runs 1–3.

`HEAD_SCALE_INIT` 3.0 → 1.0 (with `SEED_LOGIT = 4.0` the answer logits already carry scale).
Prior runs' exact files are identified by SHA in their `submission.json`.

### Sweep s2 — the residual carrier, five free 600 s H100 runs

`s2_m1_res` (the deadest set), `s2_m5_res` (Hard's proxy), `s2_m6_res` (regression: is
27/140 preserved?), `s2_hD_res`, `s2_m5_res_lr2` (LR 2e-3).

### Sweep s2 read — the residual carrier trades the bias for the gradient

| tag | steps | train loss | mean exact | seen-`N` rungs | note |
|---|---|---|---|---|---|
| `s2_m6_res` | 6,337 | **3.6e-06** | 0.330 | 0 0 1 4 1 1 1 | train exact 1.000 by step 4,700 |
| `s2_m5_res` | 11,775 | 2.178 | 0.0033 | 1 0 3 3 4 2 3 | still plateaued |
| `s2_m5_res_lr2` | 9,694 | 2.201 | 0.0008 | 1 0 2 1 1 1 1 | LR 2e-3 no help |
| `s2_hD_res` (`T∈{1,2,4}`) | 13,273 | **1.970** | 0.0021 | 0 2 0 1 2 1 0 | **broke** the 2.19 plateau |
| `s2_m1_res` | 7,187 | 2.311 | 0.0010 | 0 0 0 0 0 0 0 | still uniform |

**`m6` regression is the headline.** The residual carrier memorises `m6` *faster and more
completely* than the soft one (train exact 1.000 at step 4,700; final loss 3.6e-06 vs 0.022)
and **generalises far worse**: fresh-`x` `T=1` **27 → 0**, held-out depth `T=8` **496/500 →
106/500**, `test` 275/390 → 175/390, mean exact 0.849 → 0.330. The flat-in-depth profile is
gone.

The reason is structural: `raw_k = raw_{k-1} + delta_k` lets the carried *magnitude* encode
how many applications have run, so the model stops being a tied map of a canonical state and
becomes a depth-indexed one. It buys the gradient and sells the composition.

**And `head_scale` was not the cause of the decay.** Measured `|dL/dstate|` app3/app1 on `m1`
in soft mode: `head_scale` 0.5 → 5e+04, 1.0 → 1e+04, 3.0 → 4e+05. The contraction is
intrinsic to recomputing the state onto the simplex every application, not to saturation.
My earlier "saturated softmax" reading was wrong; straight-through, which I would have
shipped on that reasoning, measured *worse* than soft (4e+05 → and see s1). The cheap
gradient probe is what caught it.

### The synthesis — `STATE_MODE = "norm"`

Residual stream, rescaled to a fixed RMS after every application, so the identity path (and
the gradient) survives but the magnitude cannot carry a depth channel. Measured app3/app1 =
**3** (soft 1e+04, residual 1). Copy-at-init still exact at `T = 8`.

### Sweep s3 — five free 600 s H100 runs, and what picks the Hard submission

`s3_m5_norm` (Hard's proxy), `s3_m6_norm` (**the regression that matters**: is 27/140 and the
flat depth profile preserved?), `s3_hD_norm`, `s3_m5_res_prompt` and `s3_m5_norm_prompt`
(`CTX_SCOPE = "prompt"` lets the operator read `x` and `T`, restoring the parent's
memorisation path at the cost of the operator no longer being a pure function of
`(state, N)`) — the two arms most likely to *fit* a variable-`N` set.

### Sweep s3 read — `norm` separates the two properties, and neither arm moves `m5`

| tag | steps | train loss | mean exact | seen-`N` rungs |
|---|---|---|---|---|
| `s3_m6_norm` | 6,397 | 1.9e-06 | 0.821 | 3 3 5 4 6 5 4 |
| `s3_m5_norm` | 12,393 | 2.181 | 0.0035 | 1 1 1 3 2 3 1 |
| `s3_hD_norm` | 17,331 | 2.091 | 0.0019 | 1 1 0 1 1 1 0 |
| `s3_m5_res_prompt` | 10,505 | 2.185 | 0.0023 | 1 0 2 3 1 2 2 |
| `s3_m5_norm_prompt` | 10,816 | 2.193 | 0.0024 | 1 0 0 1 0 1 0 |

**`m6` under `norm`**: held-out depth `T=8` **500/500** — perfect composition, better than
soft's 496/500 — and fresh-`x` `T=1` **3/140**, the floor. So the depth-channel diagnosis was
right (norm fixed what residual broke) and yet the generalisation did *not* come back. The
two properties are separable, and they sit in different places:

- **composition** survives any carrier that keeps the state canonical (soft 496/500,
  norm 500/500; residual, which lets magnitude encode depth, 106/500);
- **fresh-`x` generalisation** only appears under the *contractive* soft recompute (27/140),
  not under any residual-stream carrier (3–4/140 ≈ floor).

The contraction that kills the gradient is apparently the same thing that produces the
generalisation. That is the tension this node now turns on.

**And `CTX_SCOPE = "prompt"` does not help.** Letting the operator read `x` and `T` — the
parent's memorisation path — leaves `m5` at 2.185/2.193. So `m5`'s plateau is not the
absence of a memorisation shortcut either. Twelve configurations have now failed to move it.
Worth noting for calibration: the parent's own control only reaches 2.13/exact 0.0013 on
`m5` at 600 s, so `m5`-at-Medium is a weak discriminator for "will it fit at Hard budget".

### `STATE_MODE = "highway"` — keep the forward map, add a backward path

`nxt = softmax(...) + G * (state - state.detach())`. The second term is **identically zero
in the forward pass** and contributes an identity to the backward one. Verified: the forward
logits are **bit-identical** to `soft` (max &#124;Δ&#124; = 0.000e+00 over a 96-row `m5` batch), so
the composition and the inductive bias that produced 27/140 are untouched by construction;
only the gradient reaching earlier applications changes. Measured app3/app1 = **1.0**
(soft 4e+05).

Legality: ordinary differentiable tensor algebra inside `forward`, no derivative entry
point, and it *adds* a path from the loss to the parameters rather than breaking one — the
same class of device as a straight-through estimator. Flagged, not asked.

### Quota correction

`one-layer submit --tier hard` rejected with *"Hard daily attempt limit reached (3); resets
at 2026-08-23T00:00:00+00:00"*. **The Hard allowance is 3 per UTC day, not 1** — the parent's
two Hard runs plus run 2 used 2026-08-22's three.

### Sweep s4 — the highway on every instrument

`s4_m6_hw` (the regression: does 27/140 survive?), `s4_m5_hw`, `s4_hD_hw`, `s4_m1_hw` (the
set that was exactly uniform), `s4_m5_hw_fast` (`N_OP_LAYERS` 2, `OP_CROSS` off — throughput
for a Hard shot, since Hard is launch-bound at 9.1 steps/s against the parent's 31).

### Run 4 — Hard `h1`, arm `digit`, `STATE_MODE = "highway"` (`2532aa25`), 2026-08-23

Submitted at the UTC reset, before s4 read. SHA `7ad54199…`. s4 (below) then said this was
the wrong bet.

### Sweep s4 read — the gradient-decay theory is dead

| tag | steps | train loss | mean exact | seen-`N` rungs |
|---|---|---|---|---|
| `s4_m6_hw` | 8,486 | 0.256 | 0.419 | 2 2 2 5 4 1 1 |
| `s4_m5_hw` | 11,539 | 2.186 | 0.0027 | 0 0 0 1 0 2 0 |
| `s4_m5_hw_fast` | **22,016** | 2.183 | 0.0028 | 1 0 0 1 2 2 0 |
| `s4_hD_hw` | 19,871 | 2.169 | 0.0019 | 1 1 1 2 0 2 1 |
| `s4_m1_hw` | 4,880 | 2.313 | 0.0000 | 0 0 0 0 0 0 0 |

**The highway made `m6` worse on every axis**, with a forward pass verified bit-identical to
`soft`:

| `m6` | soft (run 1) | highway |
|---|---|---|
| fresh-`x` `T=1` (of 140) | **27** | 2 |
| held-out depth `T=8` (of 500) | **496** | 311 |
| `test` (of 390) | **275** | 84 |
| train exact @ ~7,900 steps | **0.99** | 0.82 |
| final train loss | **0.022** | 0.256 |

And it unlocked nothing: `m1` still uniform, `m5` still 2.186, `hD` still 2.169 — and
`m5_hw_fast` reached **22,016 steps (1.9×)** and still 2.183, so it is not steps at Medium
budget either.

**So the vanishing gradient was not the blocker, and "fixing" it is actively harmful.** The
same forward map trained with a properly-flowing gradient fits *slower* and generalises 13×
worse. The reading this supports: under the contractive `soft` carrier the loss only
propagates a short way, so each application is trained largely by its *own* local
supervision (the rows whose `T` equals that depth) — greedy, layerwise training of a tied
map, which is what produces a map that composes. The identity path lets the applications
co-adapt across depth instead, and co-adaptation is exactly what a tied ballistic operator
must not do. Two rounds ago I called the decay a bug; it is closer to being the mechanism.

Also, on `m1`'s "exactly `ln 10`": `m1` is fixed `N` with `y` near-uniform on `[1, 10403)`, so
its *marginal* optimum is ≈ `ln 10`. `m5`'s widths vary (12/14/16 bits), so its marginal
optimum is below it. The 2.31-vs-2.18 gap is dataset statistics, not a learning difference —
every failing run is sitting at its marginal optimum, including the parent's own control at
600 s. I over-read that number last round.

### Sweep s5 — throughput on the carrier that actually generalises

`soft` is the only carrier with fresh-`x` generalisation, and the only surviving Hard
hypothesis is step count (parent 112k and memorised; we got 32.7k). `s5_m6_soft` (replicates
run 1 on our own H100 — a calibration never done), `s5_m6_soft_fast` and `s5_m6_soft_ops2`
(does the 1.9× throughput architecture keep 27/140, and which of the two changes costs it),
`s5_m5_soft_fast`.

### Run 4 read — Hard `h1`, `highway` (`2532aa25`)

41,938 steps, train loss **2.212**, flat from step 4,500 (2.213) to the buzzer (2.212).
`test` 4/9,999 · 2.19, `ood_t` **0**/10,002, `ood_n_t` 10/10,002. Seen-`N` rungs
`0 0 0 0 0 0 1`, OOD-`N` `0 0 1 0 0 1 1`. Mean exact 0.00047.

Exactly what s4 predicted: the highway does nothing on Hard. **0/768 at seen-`N` `T=1`**, so
this run does not improve our board standing (the parent's closure v2 still holds our best at
1/768). Two Hard runs have now confirmed that everything sits at its dataset's marginal
optimum — 2.20 (soft, run 2), 2.21 (highway, run 4) — with no descent at 32.7k or 41.9k
steps.

### Sweep s5 read — `OP_CROSS = False` is the finding

| tag | steps | train loss | mean exact | fresh-`x` rungs `T=1…64` (of 140) |
|---|---|---|---|---|
| `s5_m6_soft` (run-1 config) | 6,091 | 0.679 | 0.082 | 2 1 1 2 1 1 1 |
| **`s5_m6_soft_fast`** (2 layers, no cross-attn) | 9,187 | **0.0016** | **0.866** | **35 35 35 35 36 36 36** |
| `s5_m6_soft_ops2` (2 layers, cross-attn kept) | 8,707 | 0.308 | 0.267 | 1 3 1 2 0 0 2 |
| `s5_m5_soft_fast` | 26,854 | 2.181 | 0.0021 | 0 0 0 1 2 1 0 |

`soft_fast` beats hosted run 1 on every axis: fresh-`x` `T=1` **35** vs 27 (floor 4),
held-out depth `T=8` **499/500** vs 496, `test` **286/390** vs 275, ladder flat 35→36 across
`T = 1…64`.

**The win is dropping per-layer cross-attention, not the layer count.** `ops2` has almost the
same step count (8,707 vs 9,187) and does *not* fit (loss 0.308). With `OP_CROSS = False`,
`N` reaches the operator only through the per-slot modulus-digit embeddings — digit-aligned
conditioning, in the same coordinate as the state — instead of through a pooled attention
read. That is the same lesson the carrier gave: this architecture wants everything in digit
coordinates. It also buys **3.1× throughput on `m5`** (44.8 vs 14.4 steps/s).

**Calibration — our own H100 is 0.68× the organizers'.** `s5_m6_soft` is hosted run 1's exact
config and got **6,091 steps against 8,929**. That explains the apparent non-replication (it
never reached memorisation) and means every local sweep number is ~32% under-resourced.
`m5`'s plateau survived 26,854 local steps regardless.

### Run 5 — Hard `h1`, `soft` / 2 layers / no cross-attn (`bafdb828`), 2026-08-23

SHA `50c65072…`. `left: 1 today`.

### Sweep s6 read (four of five; the 3600 s `m5` run is still going)

| tag | steps | train loss | mean exact | fresh-`x` rungs `T=1…64` (of 140) |
|---|---|---|---|---|
| `s6_m6_l1` (**1** op layer) | **13,117** | 0.0019 | 0.864 | 34 34 38 37 37 38 38 |
| `s5_m6_soft_fast` (2 layers) | 9,187 | 0.0016 | 0.866 | 35 35 35 35 36 36 36 |
| `s6_m6_l3` (3 layers) | 10,987 | 0.0050 | 0.875 | 35 35 35 35 35 35 35 |
| `s6_m6_lr2` (2 layers, LR 2e-3) | 12,778 | 0.050 | 0.838 | 18 18 18 18 18 19 19 |
| `s6_m1_fast` | **15,705** | 2.310 | 0.0000 | 0 0 0 0 0 0 0 |

Three things it adds on top of s5.

1. **Operator depth does not matter; 1, 2 and 3 layers are indistinguishable** (34–38 of 140,
   all flat across the ladder — a spread of 1–3 examples is noise at these counts). One layer
   is **1.43× faster** than two. Combined with `OP_CROSS = False` that is ~**4.4×** the
   original throughput, which on Hard projects to ~145k steps against run 2's 32,755 and the
   parent's 112k. So the per-step capacity was never the constraint — the digit-aligned
   conditioning was, and everything else is throughput to be spent.
2. **LR 2e-3 halves the generalisation** (18 vs 35) while fitting nearly as well. Keep 1e-3.
3. **`m1` is still exactly at its marginal optimum with throughput controlled** — 15,705
   steps (3.2× the earlier 4,880, and *more* steps than the 9,187 that fully fit `m6`), same
   architecture, same clock, same batch, also fixed `N`. So `m1`'s failure is not throughput.
   Remaining differences from `m6`: no `T = 1` row (min `T` = 4), a 6.7× larger table (10,200
   vs 1,516 units) and 5× fewer epochs (268 vs 1,340). Those are still confounded with each
   other; `m6` remains the only set this architecture fits at all.

### `s6_m5_fast_3600` — `m5` leaves the plateau, and the other side is the wall

Soft / 2 layers / no cross-attn, 3600 s local (≈2450 hosted-s), SHA `50c65072…`.
**80,365 steps.**

| elapsed | step | train loss | train exact |
|---|---|---|---|
| 182 s | 3,900 | 2.183 | 0.002 |
| 1,091 s | 23,900 | 2.163 | 0.002 |
| 1,273 s | 27,900 | 2.138 | 0.006 |
| 1,451 s | 31,900 | 2.049 | 0.014 |
| 1,630 s | 35,900 | 1.977 | 0.023 |
| 2,336 s | 51,900 | 1.806 | 0.086 |
| 2,870 s | 63,900 | 1.716 | 0.107 |
| 3,580 s | 79,900 | **1.575** | **0.152** |

**The plateau breaks at ~28,000 steps** and the descent is monotonic to the buzzer, train
exact still rising. Every earlier `m5` run — all twelve — topped out between 4,601 and 26,854
steps, i.e. *below the break point*. So "twelve configurations failed to move `m5`" was an
artifact of budget, not a property of the architecture. Correction recorded.

**And what is on the other side is the rule-acquisition wall.** At the buzzer:

| | |
|---|---|
| `test` (held-out prompts) | **13 / 9,000**, loss **2.996** |
| `ood` (held-out depth) | 7 / 3,000, loss 2.233 |
| seen-`N` rungs `T=1…64` (of 768) | 0 1 3 2 2 1 3 |
| OOD-`N` rungs (of 768) | 0 1 1 0 0 2 1 |
| mean exact | 0.0019 |

Train exact 0.152 and `test` loss **2.996 — above `ln 10`**: confidently wrong off the
training prompts. This is precisely the parent's Hard control-v2 signature (58% memorised,
11.4 nats, 0/768 everywhere), reached earlier and from a per-digit architecture. The digit
carrier does **not** rescue a multi-modulus set; it reproduces the wall.

Note also **seen-`N` `T=1` = 0/768 against a floor of 6** — not even the no-reduction cases.
`m5` trains `T ∈ {2,4,8}`, so application 1 is never directly supervised, and the composition
constraint does not identify it once memorisation pressure sets in. That is the `m1` lesson
again, now on the set that proxies Hard.

**So the step hypothesis is answered, and answered negatively**: steps do break the plateau,
and behind it is memorisation without transfer.

### Run 5 read — Hard `h1`, soft / 2 layers / no cross-attn (`bafdb828`)

93,638 steps (2.9× run 2's 32,755), 3.72M state elements. Train loss 2.221 → **2.118**,
train exact 0.002 → 0.018. `test` **2/9,999 · 2.360**, `ood_t` 7/10,002 · 2.212, `ood_n_t`
7/10,002 · 2.256. Seen-`N` rungs **0 1 0 0 0 0 0**, OOD-`N` all zero. Mean exact 0.00053.

| step | 7.5k | 30k | 53k | 68k | 83.5k | 93.6k |
|---|---|---|---|---|---|---|
| train loss | 2.221 | 2.210 | 2.179 | 2.172 | 2.131 | **2.118** |
| train exact | 0.000 | 0.002 | 0.006 | 0.018 | 0.018 | 0.018 |

**Against the prediction**: the direction was right (memorisation beginning, no transfer,
`test` loss 2.36 *above* `ln 10`), the magnitude was badly wrong. `m5` at 80k steps was at
loss 1.575 / train exact 0.152; Hard at 93.6k steps is at 2.118 / 0.018. Hard fits **far**
more slowly than its public proxy — the modulus-grouped split gives ~13/38/133 training
moduli per width, so there is much less per-modulus density than prompt-grouped `m5` has.
`m5`-as-Hard-proxy is calibrated for the *floor* (`dress_rehearsal` §2) but not for the
*fitting rate*; that is a new caveat on the proxy.

**And the sharper comparison is against the parent.** At comparable step counts on the same
hidden data:

| | parent control v2 | digit fast (run 5) |
|---|---|---|
| steps | 111,961 | 93,638 |
| train loss | **0.682** | 2.118 |
| train exact | **0.576** | 0.018 |
| seen-`N` `T=1` | 0/768 | 0/768 |

The parent memorises 58% of Hard; we memorise 2%. That is the digit carrier working as
designed and against us here: the state is `x`'s digits with `N` as the only conditioning, so
there is no per-prompt shortcut — to fit at all it has to compute. On one fixed modulus that
buys 35/140 on fresh `x` with a ladder flat to `T=64`. On 184 moduli it buys an inability to
fit inside 3,600 s.

Board position unchanged: 0/768 at seen-`N` `T=1`, so the parent's closure v2 (1/768,
rank 31) is still our best Hard entry.

### Run 6 — Medium `m5`, arm `digit_closure` (`2e978eaf`), 2026-08-23

The re-entry term rolls on from the model's own decode after one application, which is the
only mechanism here that constrains **application 1** on a set with no `T = 1` row — the
thing runs 2, 4 and 5 all failed at. CPU smoke first (`N=143`, `T∈{1,2,3}`, 40 s): closure
reads `test` 0.467 / held-out-depth 0.790 against the plain arm's 0.233 / 0.520 on the same
manifest. `left: 9 today` on Medium.

### Run 6 read — Medium `m5`, `digit_closure` (`2e978eaf`): the ramp fired mid-plateau again

21,221 steps, 600 s. Train loss flat at 2.18 to step ~14,700, then **2.181 → 3.906 → 4.397**
from step 16,800 on. `test` 17/9,000 · 2.149, `ood` 9/3,000 · 2.147. Seen-`N` rungs
**0 1 0 1 1 2 0** (`T=1` = **0**/768, floor 6). OOD-`N` **1 1 1 3 3 3 3**. Mean exact 0.0024.

**Re-entry did not move the `T=1` rung**, and the test was confounded in a way worth naming.
Two things went wrong, and they compound:

1. **21,221 steps is below `m5`'s ~28,000-step break point.** The run never left the plateau,
   so there was no fitted model for a consistency term to be consistent with.
2. **The aux ramp fired anyway, at 72% of the clock, and dominated the loss.** The v3 gate is
   "train-exact EMA ≥ 0.5, *with a clock fallback at 60%*". Train exact never came near 0.5,
   so the fallback fired mid-plateau — which is exactly the failure the parent's README
   records for closure v2 on `m5`/`h1`, and exactly what v3's progress gate was built to
   prevent. The progress gate works; its fallback re-introduces the bug.

So this measures "closure with a badly-timed ramp", not closure. The mechanism has still
never been tested in the regime it is for. The general point it does establish: **closure is
a post-fit mechanism, and `m5`/Hard never reach post-fit for this architecture** — on Hard the
fallback would fire at 2,160 s, where run 5 was still at loss 2.17 / train exact 0.018, so a
Hard closure run today would reproduce the same confound.

**Last Hard slot: left unspent**, per the rule stated before the run. Closure did not help on
`m5`; the plain arm has been measured on Hard twice; and with 3 slots/day and ~7 days left,
slots are abundant while a null measurement is not worth one.

### Run 7 — Medium `m5`, arm `digit` (`2ea7780e`), 2026-08-23

The matched control run 6 lacked. Every plain-arm `m5` number at this architecture so far is
from **our own H100 at 0.68×**, at a different step count; hosted `m5` reaches ~30-40k steps,
past the break point. Without it, "closure vs plain on `m5`" compares across two machines.

### Run 7 read — Medium `m5`, plain `digit` (`2ea7780e`): the matched control

26,203 steps, train loss **2.178** at the buzzer, flat throughout (2.193 at step 2,500;
2.178 at 26,203). `test` 25/9,000 · 2.162, `ood` 12/3,000 · 2.154. Seen-`N` rungs
**1 0 1 3 2 3 1**, OOD-`N` **3 0 1 3 3 2 3**. Mean exact 0.0034.

| `m5`, hosted, 600 s | plain (run 7) | closure (run 6) |
|---|---|---|
| steps | 26,203 | 21,221 |
| final train loss | 2.178 | 4.363 (aux-dominated) |
| `test` | 25/9,000 · 2.162 | 17/9,000 · 2.149 |
| `ood` | 12/3,000 · 2.154 | 9/3,000 · 2.147 |
| seen-`N` `T=1` (floor 6) | 1 | 0 |
| mean exact | 0.0034 | 0.0024 |

**Neither arm crossed the break.** 26,203 steps is just short of the ~28,000 the 3600 s run
needed, so the organizers' clock at Medium does not reach it either. Both sit at the marginal
optimum and every rung count is 0-3 of 768 — noise. The control confirms run 6 was
uninformative about closure, and adds that closure costs ~19% of the step budget.

**Correction to the hardware calibration.** `m6`: hosted 8,929 vs local 6,091 = 1.47×. `m5`:
hosted 26,203 vs local 26,854 = **0.98×**. The ratio is not a constant — it is dataset-
dependent (dataloader and sequence length differ), so "our H100 is 0.68× the organizers'"
was over-generalised from one dataset and should not be applied as a factor.

### Ramp gate fixed — no clock fallback

`CLOCK_FALLBACK` 0.6 → 1.1 (can never fire). The closure terms now engage only when the
train-exact EMA crosses 0.5. The targets are self-generated, so a model that never fits has
nothing to be consistent with, and the correct behaviour is for the closure arm to degenerate
silently to the control rather than to inject a ramp into a chance-level model. CPU smoke
(`N=143`, 40 s): 450 steps, `test` 0.317 / held-out depth `T=4` **0.530** against the plain
arm's 0.233 / 0.520 on the same manifest.

### Run 8 — Medium `m6`, arm `digit_closure`, gate fixed (`9fbe030c`), 2026-08-23

SHA `fcb18f27…`. `m6` is the only instrument where the model reaches post-fit, so it is the
only place a post-fit mechanism can be measured at all. Baseline to move: the plain arm's
**35 35 35 35 36 36 36** of 140 (hosted run 1 at the older architecture read 27; floor 4),
held-out depth `T=8` 499/500, `test` 286/390. `left: 7 today`.

### Run 8 read — Medium `m6`, `digit_closure` with the gate fixed (`9fbe030c`): no effect

9,798 steps, 600 s. Train exact **0.436 by step 800 (45 s)**, 0.994 by step 7,200 (421 s),
1.000 at the buzzer; final loss 0.023. So the EMA crossed the 0.5 trigger at roughly step
~1,100 (~12% of clock), the ramp completed by ~27%, and **the closure terms were live for
~85% of training on a fully-fitted model** — the regime they were designed for, reached for
the first time in this port.

| `m6` | plain (local `s5_m6_soft_fast`) | closure (hosted, run 8) |
|---|---|---|
| steps | 9,187 | 9,798 |
| fresh-`x` rungs `T=1…64` (of 140) | 35 35 35 35 36 36 36 | **33 34 35 36 35 36 35** |
| held-out depth `T=8` (of 500) | 499 | 497 |
| `test` (of 390) | 286 | 287 |
| mean exact | 0.8657 | 0.8649 |
| OOD-`N` rungs (of 256) | 2 1 1 1 1 1 1 | 2 1 1 1 1 1 2 |

**Identical inside noise on every axis.** Verdict: **drop the closure arm.** It costs steps
(~19% on `m5`) and changes nothing where it can actually run.

The reason is worth stating, because it is not a failure of the idea so much as its
obsolescence here. `ballistic_depth` §2's cycle and re-entry terms exist to stop a *pooled*
state from drifting off the manifold its own encoder produces — the parent port's rollout
passes through vectors no encoder ever emitted, which is what the terms penalise. The digit
carrier makes that drift **impossible by construction**: the state is a digit grid, every
intermediate is a thing the seed path could have produced, and the ladder is already flat to
`T = 64` with 497-500/500 at a held-out depth. There is no inconsistency left for a
consistency loss to remove. **The carrier subsumes the closure terms.** That is a cleaner
answer to `dress_rehearsal` §5's open question ("why didn't fixed-`N` closure transfer?") than
the ablation list it proposed: it transfers, in the sense that the property it was buying is
now structural.

**Caveat I walked into again**: the plain-arm baseline above is from *our* H100 and the
closure run is hosted. Run 9 fixes that.

### Run 9 — Medium `m6`, plain `digit` at the current architecture (`db4c2ac0`), 2026-08-23

The matched hosted control for run 8, and the hosted anchor for the `m7`/`m9` replications.
Hosted run 1's 27/140 is at the *old* architecture (4 layers, cross-attention), so no hosted
number exists for soft/2-layer/no-cross on `m6`. `left: 6 today`.

### Run 9 read — Medium `m6`, plain `digit` at the current architecture (`db4c2ac0`): the matched anchor

11,143 steps, train loss **8.5e-06** (fully memorised), mean exact 0.8731.

| `m6`, hosted, 600 s | plain (run 9) | closure (run 8) | plain, local (`s5`) |
|---|---|---|---|
| steps | 11,143 | 9,798 | 9,187 |
| fresh-`x` rungs `T=1…64` (of 140) | **35 35 38 39 39 39 38** | 33 34 35 36 35 36 35 | 35 35 35 35 36 36 36 |
| held-out depth `T=8` (of 500) | **500** | 497 | 499 |
| `test` (of 390) | **291** | 287 | 286 |
| mean exact | **0.8731** | 0.8649 | 0.8657 |
| OOD-`N` rungs (of 256) | 1 1 0 0 0 0 0 | 2 1 1 1 1 1 2 | 2 1 1 1 1 1 1 |

Now matched machine-for-machine: **closure is slightly worse at all seven rungs and on both
scored splits.** Each gap is 2-4 of 140 — individually noise — but the sign is consistent
across nine independent comparisons, and it costs ~19% of the step budget on top. The drop
verdict stands and is now properly controlled.

The anchor also gives the cleanest statement of what this architecture does on a single
modulus: **fresh-`x` `T=1` = 35/140 (25%) against a floor of 4 (2.9%), a ladder flat and in
fact slightly *rising* out to `T=64`, and held-out depth `T=8` at 500/500 — perfect.**

Hosted/local ratio for `m6` at this architecture: 11,143 / 9,187 = **1.21×** (it was 1.47× at
the old architecture, 0.98× on `m5`). Third distinct value; treat the ratio as unknown per
configuration rather than as a constant.

### Run 10 — Medium `m7`, plain `digit` (`078175f3`), 2026-08-23

First of the replications. `m7` = fixed `N` = 1,763 (43×41), `T ∈ {1,2,4}`, fresh-`x` cohort
**180**, floor 4. The 35/140 result rests on one modulus; `m7` and then `m9` (N=1,927, cohort
~200) say whether it is a property of the architecture or of `N` = 1,517. The parent's
control v2 read **1/180** here. `left: 5 today`.

### Run 10 read — Medium `m7`, plain `digit` (`078175f3`): it replicates, and stronger

9,791 steps, train loss 0.0053, mean exact 0.8536.

| fixed-`N` Medium, hosted, 600 s | `m6` (N=1,517) | `m7` (N=1,763) |
|---|---|---|
| fresh-`x` rungs `T=1…64` | 35 35 38 39 39 39 38 **/140** | **50 51 52 51 51 52 51 /180** |
| fresh-`x` `T=1` as a rate | 25.0% (floor 2.9%) | **27.8%** (floor 2.2%) |
| held-out depth `T=8` | 500/500 | 595/600 |
| `test` | 291/390 | 322/450 |
| parent's control v2, same set | 2/140 | **1/180** |

**The result is a property of the architecture, not of `N` = 1,517.** Two moduli, two
cohorts, the same shape: a fresh-`x` rung ~12× the analytic floor, flat across the entire
`T = 1…64` ladder, with held-out-depth accuracy at 99-100%. The parent port read 2/140 and
1/180 on the same two sets — at or below floor — so this is a ~25-50× improvement on the same
instrument, same evaluator, same budget, same seed.

What it says, stated carefully: on a **single modulus**, the per-digit carrier learns a
single-step map that (a) generalises to `x` values never trained on, one time in four, and
(b) composes exactly — 64 applications lose nothing, and a depth never trained is essentially
perfect. Both of those were open questions for the program; both now have a positive answer
in the competition's own metric. What remains negative is the modulus family: the same
architecture on 204 moduli (`m5`) or ~184 (Hard) cannot fit at all inside the budget.

### Run 11 — Medium `m9`, plain `digit` (`31cd4dff`), 2026-08-23

Third modulus: `N` = 1,927 (41×47), `T ∈ {1,2,4}`, fresh-`x` cohort ~290. `left: 4 today`.

### Run 11 read — Medium `m9` (`31cd4dff`): third modulus, same shape

6,189 steps, train loss 0.0082, mean exact 0.8449. Fresh-`x` rungs **67 63 67 67 66 66 66**
of 290; `ood` (held-out depth `T=8`) 592/600; `test` 327/465.

---

## Where the node stands — 11 hosted runs, 6 free H100 sweeps (2026-08-23)

### The positive result, on three moduli

Floors recomputed exactly from each set's `depth_t_1` split (`x² < N`):

| set | `N` | cohort | fresh-`x` `T=1` | floor | ×floor | ladder `T=1…64` | held-out depth `T=8` | parent port |
|---|---|---|---|---|---|---|---|---|
| `m6` | 1,517 | 140 | **35** (25.0%) | 4 (2.9%) | 8.8× | 35 35 38 39 39 39 38 | 500/500 | 2 |
| `m7` | 1,763 | 180 | **50** (27.8%) | 4 (2.2%) | 12.5× | 50 51 52 51 51 52 51 | 595/600 | 1 |
| `m9` | 1,927 | 290 | **67** (23.1%) | 6 (2.1%) | 11.2× | 67 63 67 67 66 66 66 | 592/600 | — |

On a single modulus the per-digit carrier learns a single-step map that **generalises to
unseen `x` about one time in four (~10× the analytic floor)** and **composes exactly** — the
ladder is flat out to 64 applications, and a depth never trained reads 99-100%. Three moduli,
one seed each, the organizers' evaluator and budget. The parent's pooled-state port reads at
or below floor on the same sets (2/140, 1/180).

### The negative, and it is the one that matters for the board

The same architecture **cannot fit a multi-modulus set inside the budget**. Hard (run 5,
3,600 s, 93,638 steps): train loss 2.221 → 2.118, train exact 0.018, `test` 2/9,999 at loss
**2.36 — above `ln 10`**, seen-`N` `T=1` **0/768**. The parent memorises 58% of Hard at
comparable steps; we memorise 2%. That is the digit carrier working as designed and against
us: with the state fixed to `x`'s digits and `N` the only conditioning, there is no
per-prompt shortcut, so fitting *requires* computing. Board position is unchanged — the
parent's closure v2 (1/768, rank 31) is still our best Hard entry.

`m5` (204 moduli) does leave its plateau, but only at **~28,000 steps** and only into
memorisation-without-transfer (80k steps: train exact 0.152, `test` loss 2.996). Hosted
Medium reaches 26,203 steps, just short, so no 600 s run can see it.

### The closure null

With the ramp gate fixed (no clock fallback) and the terms live for ~85% of training on a
fully-fitted model, closure is **slightly worse than the plain arm at all seven rungs and on
both scored splits** on `m6`, and costs ~19% of the step budget. Read: the digit carrier
**subsumes** the cycle/re-entry terms. They exist to stop a pooled state drifting off its own
encoder's manifold; a canonical digit grid cannot drift, and a ladder already flat to `T=64`
has no inconsistency left to remove. `dress_rehearsal` §5's "why didn't closure transfer?"
resolves as *the property it bought is now structural*. Arm dropped.

### Caveats that cost us runs

- **`m5` is calibrated for Hard's floor, not its fitting rate.** Hard is modulus-grouped
  (~13/38/133 training moduli per width) and fits far more slowly than prompt-grouped `m5`.
- **The hosted/local step ratio is not a constant**: 1.47× (`m6`, old arch), 1.21× (`m6`,
  current), 0.98× (`m5`). Never compare an arm across machines — two of this node's readings
  did, and both had to be redone with a matched hosted control.
- **Hard's allowance is 3/UTC-day, not 1.**
- Two theories were killed by cheap probes that should have run first: "saturated softmax"
  (false — decay is temperature-independent) and "gradient decay is the blocker" (false — a
  verified-identical forward pass with the decay removed fits *slower* and generalises 13×
  worse). The contraction appears to be the mechanism, not the bug.

### The one experiment to run next: where does the modulus family break it?

One modulus → ~25% fresh-`x`; 204 moduli → 0. Hard sits at ~13/38/133 training moduli per
width, i.e. **inside the untested gap**, so this is the question that decides whether the
result can reach the board at all.

The generator gives a modulus-count ladder for free, because the count is fixed by arithmetic
at each width — and widths 10-13 bits all yield **4-digit** moduli, so digit width is
controlled:

| `--modulus_bits` | 10 | 12 | 11 | 13 |
|---|---|---|---|---|
| distinct valid RSA moduli | 6 | 14 | 21 | 57 |

Four single-width datasets, matched `examples_per_setting` and `T ∈ {1,2,4}`, exhaustive-`x`
depth cohorts, `split_group=modulus`. **Cost: generation is CPU-minutes; 4 × 600 s H100 ≈ 40
GPU-minutes, ~80 with a second seed.** It cannot run hosted (the service only serves its own
datasets), which is why it is a proposal rather than a launch.
