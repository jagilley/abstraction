# question_model — File Index and Calibration Record

Machinery and calibration only. Findings go back to the orchestrator for discussion before any
`README.md` is written (repo convention). Design: [`SPEC.md`](SPEC.md).
**Up**: [`../README.md`](../README.md) · **Direct parent (substrate donor)**:
[`../conditional_revision/`](../conditional_revision/README.md) · **Machinery donor**:
[`../practice/setlist/FILES.md`](../practice/setlist/FILES.md).

## What the round installs

`revision_not_surprisal` §4 splits a token's surprisal two ways — news about *this sequence's*
latent structure, and an irreducible realisation residue. This round adds the third term the
`transpose`×`setlist` crossed pair says is missing: news about **the generative mixture itself**,
i.e. *what is being asked*.

```
E[ -log P(x_{t+1}|x_{<=t}) ]  =  E[Dm]  +  E[S_D | theta]  +  E[ H(x_{t+1}|theta, z_{<=D}, x_{<=t}) ]
   total surprisal               DEMAND     STRUCTURE          ALEATORIC
```

All three are exact. `S_D` and `H_irr` are `conditional_revision/oracle.py`'s objects with every
`/m` and `1/v` replaced by a mixture weight; `Dm = KL(P(theta|x_{<=t+1}) ‖ P(theta|x_{<=t}))` comes
from a forward filter that is *exactly* Bayes-optimal because the world's `theta` lives on a finite
grid. The identity is exact in expectation and is checked in-run.

## The one design decision that everything else follows from

`setlist`'s OU drifts the full logit tensor — `theta_root` (v,) and `theta[level]` (v, m). That is
right for the practice arc, whose readout is a table's *coverage* of the demand and needs no
posterior over the demand state. Here the readout **is** the posterior, and a filter over R^80 is
not exact. So the drift is reparametrised as a small number of **scalar OU components riding fixed
random directions in logit space**, each on a K-point grid with a discretised-OU kernel:

- the world's `theta` **is** a grid point, so the discretisation is a property of the world, not an
  approximation in the inference. The filter is exact and the identity closes to Monte-Carlo error;
- the stationary law is an eigenvector, computable exactly, so the prewarm is a draw from the true
  stationary distribution. `setlist`'s `typical_demand` (median-of-48 candidates) exists because a
  continuous OU's stationary law cannot be enumerated; here it can, so the lottery is **absent**
  rather than avoided by heuristic;
- `H(theta)` is finite and known, which **caps the demand channel at `H(theta)/T` nats per token**.

Three components, chosen so the state is level-resolved in the way `setlist`'s demand-levels lesson
says matters — and note the deliberate structural asymmetry, which is the laundering question's
negative control rather than a defect:

| component | what it moves | draws per 64-token block |
|---|---|---|
| `root` | the root prior over r\*, "which feature is asked" | **1** — in-context unidentifiable at T = 64; weights-only currency |
| `hi` | rule mixtures on levels 0–2 | 7 |
| `lo` | rule mixtures on levels 3–5, "which realisation is asked" | 56 — the component a window can actually infer |

## Worlds

| world | what it is | why |
|---|---|---|
| `drift` | one OU step per block | the main world |
| `incoh` | every node redraws from the **theta-marginal** mixture | unigram statistics match `drift` exactly; what is destroyed is within-block *coherence*, i.e. the existence of a theta to infer. The matched control for "did the learner build a question model" — and the loss gap `incoh`-trained minus `drift`-trained on drift data is the **realised value** of the question channel, against `E[Dm]` as its exact ceiling |
| `static` | theta frozen at one stationary draw | the history filter identifies it and its channel decays; the window filter's does not |
| `uniform` | `sigma_s = 0` | dispatches to `rhm_latent_loop._generate_with_traces` **itself**, so the drift-off corpus is bit-identical to every prior run on this regime rather than merely equivalent (`setlist` gate D-1's lesson) |

## Code files

| File | Purpose |
|---|---|
| `demand_state.py` | The world. Grid, discretised-OU kernel, exact stationary law, fixed logit directions, `theta_weights`, per-world corpus sampling (`sample_corpus` dispatches to the parent generator when the demand is off), and the exact per-block `block_kl` / `event_kl` in nats. Inherits the OU form, `weights_from_theta` and `sample_derivations_weighted` from `rhm_drift`; nothing in `setlist/demand.py` or `rhm_drift.py` is modified |
| `oracle_q.py` | Weighted-mixture BP: `conditional_revision/oracle.py` with per-cell mixture weights and a non-uniform root prior, as **new functions** so prior results stay bit-identical. `structure_channel` emits `S_joint` / `S_marg` / `S_chain` / `H_post` / `H_irr` / `H_tot` / `surprisal` with `oracle.py`'s field names, so the drift-off comparison is a field-by-field diff |
| `qfilter.py` | The exact question filter. `prefix_loglik` computes `log P(x_p\|x_{<p}, theta_g)` for every (sequence, grid state, position) in **O(T·L)** per sequence instead of O(T·2^L) by keeping completed subtrees on a binary-counter stack and walking the right frontier — an unobserved subtree's inside vector is exactly all-ones for any weights. `window_posteriors` / `history_priors` / `demand_channel` / `three_way_check` |
| `calibrate.py` | The offline sigma_s × kappa × K sweep, in the currencies the decomposition reads |
| `question_model.py` | The Modal app: `selfcheck`, `calibrate`, `train_world`, `decompose` |
| `analyze_qm.py` | Reduction and figures |

## Gates

| Gate | What it asserts | Where | Measured |
|---|---|---|---|
| **G-1** | **drift-off fidelity**: at uniform weights the weighted BP reproduces `conditional_revision/oracle.py` field-for-field | `oracle_q._test_reduces_to_incumbent` | max\|Δ\| **4.4e-15** |
| **G-2** | weighted BP against brute-force enumeration on tiny trees with **non-uniform** weights (the check that caught the hypertree error in the incumbent, re-run in the weighted regime) | `oracle_q._test_brute_force_weighted` | max\|Δ\| ≤ **1.8e-15** (KL), 4.4e-16 (H), 4.4e-16 (node marginals) |
| **G-3** | the **conditional identity** `E[S_D\|theta] = E[H_tot − H_irr]` — §4's identity holds inside each question state | `oracle_q._test_conditional_identity` | abs err **7e-4 – 1.6e-3** at n = 4000 (Monte-Carlo scale) |
| **G-4** | the O(T·L) frontier walk equals BP's leaf posteriors | `qfilter._test_loglik_matches_bp` | max\|Δ\| ≤ **1.8e-14** over three regimes |
| **G-5** | **drift-off exactness**: at `sigma_s = 0` the demand channel is *identically* zero and the marginal surprisal equals the incumbent's | `qfilter._test_drift_off` | max\|Dm\| = **0.00e+00**, max\|Δsurprisal\| = 4.6e-15 |
| **G-6** | the **demand identity** `E[surp_marg] − E[surp_theta] = E[Dm]` — the D-independent half, which isolates the new term | `qfilter._test_demand_identity` | 0.078736 vs 0.078901, err **1.7e-4** |
| **G-7** | in-run: the three-way identity on the real regime, per D | `question_model.decompose` | see run record |
| **G-8** | in-run: the drift-off control reproduces `conditional_revision`'s published DGP table | `question_model.decompose` | see run record |

Run: `modal run -m rhm.question_model.question_model::selfcheck` (all of G-1…G-6, ~4 min).

## Offline calibration (`calibrate.py`, zero GPU, n = 256 blocks, v16 s2 L6 m4)

Every number below is a **measurement that set a knob**, in the currency the experiment reads.

### The sigma_s trade-off — and what "inadmissible" means here

`setlist`'s admissibility risk was that a loud demand stops the arms separating. The analogue
here is sharper: a loud demand **concentrates the mixture**, which makes the corpus easier and
shrinks the structure channel the decomposition is supposed to read against. News too loud is
weather.

| sigma_s | Dm window | (root / hi / lo) | Dm history | surp_marg | S(d1) | S(d3) | S(d6) | structure vs drift-off |
|---|---|---|---|---|---|---|---|---|
| 0.00 | 0.0000 | — | 0.0000 | 1.3818 | 0.6975 | 0.1919 | 0.0387 | 1.000 |
| 0.40 | 0.0219 | .0009/.0050/.0160 | 0.0134 | 1.3338 | 0.6600 | 0.1832 | 0.0383 | 0.945–0.989 |
| 0.70 | 0.0331 | .0022/.0092/.0216 | 0.0210 | 1.2325 | 0.6056 | 0.1671 | 0.0377 | 0.860–0.974 |
| **1.00** | **0.0387** | **.0038/.0123/.0225** | **0.0243** | **1.1249** | **0.5445** | **0.1550** | **0.0333** | **0.781–0.859** |
| 1.40 | 0.0417 | .0061/.0138/.0217 | 0.0258 | 1.0016 | 0.4783 | 0.1347 | 0.0310 | 0.686–0.802 |
| 2.00 | 0.0462 | .0085/.0166/.0215 | 0.0285 | 0.8551 | 0.4022 | 0.1148 | 0.0263 | 0.577–0.678 |

Two things this measured, both of which set the design:

1. **The demand channel saturates in sigma_s while the structure channel decays monotonically.**
   Dm gains only 2.1× from sigma_s 0.4 → 2.0 while the structure channel loses 40%. There is no
   large-sigma regime worth buying.
2. **The drift rescales the structure channel almost without distorting its level profile.** The
   per-level ratio to drift-off has a spread of only 1.10 at sigma_s = 1.0 (0.859 at d6 → 0.781 at
   d1) and 1.17 even at sigma_s = 2.0. The *shape* every RHM readout uses survives; the scale does
   not. So the binding admissibility constraint is corpus difficulty, not profile distortion.

**Chosen sigma_s = 1.0.** Demand channel 3.4% of total surprisal, structure channel at 0.78–0.86
of drift-off with its profile intact. **sigma_s ≥ 1.4 recorded as inadmissible**: at 0.73× / 0.62×
the drift-off surprisal the corpus is a different, easier substrate rather than the same one under
a moving question.

### kappa is a knob on the HISTORY channel only — a structural finding, not a nuisance

| kappa | Dm window | Dm history | ratio |
|---|---|---|---|
| 0.05 | 0.0417 | 0.0149 | 0.36 |
| **0.15** | **0.0387** | **0.0243** | **0.63** |
| 0.40 | 0.0411 | 0.0368 | 0.90 |
| 1.00 (i.i.d. per block) | 0.0404 | 0.0406 | 1.00 |

`Dm window` is **flat in kappa** (0.0387–0.0417, no trend) and `Dm history` rises monotonically to
meet it at kappa = 1. The reason is structural and worth stating plainly: a T-token-window
transformer re-infers the question from scratch every window, so **its** demand channel is kept
alive by the finite context, not by the drift; the drift is what keeps the channel alive for an
*unbounded-memory* learner, whose channel decays to zero as kappa → 0. The two claims in the SPEC's
"drift is what keeps channel (ii) alive at a knowable rate" are therefore about two different
learners, and the pair of filters separates them by construction.

**Chosen kappa = 0.15** (`setlist`'s value, inherited): an ideal learner with unlimited memory
still faces 63% of the window learner's question uncertainty — fast enough that memory does not
solve the problem, slow enough that memory helps.

### K (grid resolution) is a knob on channel size, and a weak one

| K | n_states | Dm window | Dm history | structure vs drift-off | cell cost |
|---|---|---|---|---|---|
| 5 | 125 | 0.0372 | 0.0206 | 0.813–0.897 | 12 s |
| **7** | **343** | **0.0387** | **0.0243** | **0.781–0.859** | **20 s** |
| 9 | 729 | 0.0414 | 0.0292 | 0.786–0.872 | 39 s |

**Chosen K = 7** — 2× cheaper than K = 9 for a 7% smaller channel.

### The final world spec

```
K = 7, sigma_s = 1.0, kappa = 0.15, half = 2.0 (grid spans ±2 sigma_s), dir_seed = 0
components: root | hi = rule levels 0,1,2 | lo = rule levels 3,4,5
n_states = 343      H(theta) = 5.205 nats      one OU step per 64-token block
```

Derived quantities, measured: per-token demand channel **0.0387 nats** (window) / **0.0243**
(history); the question resolved within one block is **H(theta) 5.205 → 2.53** nats; the demand
channel is the same size as the **root-level** structure channel (0.0387 vs 0.0333), which is the
level where `conditional_revision` measured the model's belief to be weakest (probe at 18% of its
Bayes ceiling, root recovery 0.088).

## Timings (measured)

| operation | cost |
|---|---|
| `structure_channel`, v16 s2 L6 m4 | **54 ms / sequence** (numpy, 1 core) |
| `prefix_loglik`, 343 grid states | **36 ms / sequence** |
| `train_world` (12k base + 12k FM steps, L4) | ~25 min |

## Half 2 — the twin learner

Six arms on **one** drift corpus (`arity_torque`'s idiom: identical data, only the input
differs), identical protocol to Half 1's `train_world` — 8L/8H/256D, 12k steps, same optimiser,
same flat-window sampling, same seed.

| arm | what it gets | rung |
|---|---|---|
| `plain` | nothing — the twin | — |
| `tap_oracle` | the exact history filter's **block-entry** posterior over `root` + `hi` (14 dims) | oracle |
| `tap_shuffled` | the same vectors, block-permuted: same input, same capacity, no information | control |
| `tap_learned` | a learned leaky integrator over the previous **32 blocks**' token histograms (32 dims), rolled out differentiably inside the training step; no theta label anywhere | endogenous |
| `tap_learned_shuffled` | the same estimator on a block-permuted histogram stream — same parameters, same input distribution, no information about the local question | control (added after the first evaluation: `tap_learned` was the only arm that moved, and `tap_shuffled` matches capacity for `tap_oracle` but **not** for the estimator's extra parameters) |
| `loss_alea` | per-position gradient weight from the exact aleatoric label **given theta** | oracle |
| `loss_alea_blind` | the same weighting from the **drift-blind** aleatoric label | control |

**Why the tap carries `root` + `hi` and not `lo`** — forced by a Half-1 measurement. `lo` reads
0.868 against an exact ceiling of 0.943 (92%), so the window already extracts it and supplying it
would blur attribution; `root` (0.239) and `hi` (0.281) sit at the shuffled-label null of ~0.25
with exact ceilings of 0.343 and 0.584. The tap's niche is exactly what the window launders.

**Why the payload is the *block-entry* prior** — it is constant within a block (a slow signal, not
a per-token hint), it is structurally unavailable to a 64-token-window learner, and it is exactly
the object a cross-context estimator would carry, so the oracle rung is a genuine ceiling for the
learned rung rather than a different kind of signal. Measured on the 200k-block pool: its argmax
recovers **root 0.352 / hi 0.420** *before seeing the block at all*, against the Half-1 window
filter's *terminal* 0.343 / 0.584. Complementary, and for `root` the cross-context prior alone
already beats a whole block of in-context evidence.

**Why the loss arm's label is theta-aware** — Half 1 measured `H_irr` falling ~20% once theta is
known (d1 0.684 → 0.553), so a drift-blind aleatoric label down-weights exactly the positions
carrying question-news. `loss_alea_blind` runs that mistake deliberately. On the pool the two
labels have means **0.568 (theta-aware) vs 0.694 (blind)**, correlation +0.943.

### Calibration by measurement (Half 2)

| knob | set by | value |
|---|---|---|
| `alea_gamma` | pinned so the **top/bottom aleatoric-decile weight ratio is 4×**, so the two loss arms differ only in the LABEL and not in how hard either pushes | deciles 0.0000 / 1.3863 (= ln m) → **gamma = 1.0000**; mean weight 0.655 (theta-aware) vs 0.608 (blind) |
| `mem_blocks` | longer than the model's context by construction — the estimator must be able to hold what the window cannot | **32 blocks = 2048 tokens** vs a 64-token context |
| tap components | Half 1's probe-vs-ceiling table (above) | `root` + `hi`, not `lo` |
| tap init | zero-initialised projection, so every tap arm is **exactly** the plain twin at step 0 | verified: all arms read `loss 2.7929` at step 0 |

### Half-2 fidelity

The `plain` arm trained from the cached pool reproduces Half 1's `drift` base **step for step**
(2.7929 at step 0, 1.7093 at step 1000, 1.5808 at 2000 — identical to `launch_train_drift.log`),
so the precomputed corpus is the same token stream the Half-1 base saw and the two halves are
directly comparable.

### The tap-design correction Half 2 forced

The tap was restricted to `root` + `hi` because Half 1 measured `lo` at 92% of its exact
**within-block** ceiling. That conflated two different redundancies. Decomposing Half 1's
per-component demand channel into its cross-context part (`Dm_window − Dm_history`, the exact
prize a block-entry prior can deliver):

| component | window | history | cross-context prize |
|---|---|---|---|
| root | 0.00446 | 0.00295 | +0.00151 |
| hi | 0.01237 | 0.00756 | +0.00482 |
| **lo** | 0.02344 | 0.01598 | **+0.00746** |
| joint | — | — | **0.01426** |

**`lo` carries more than half the available cross-context nats even though it is 92% recoverable
in-context.** In-context redundancy is not cross-context redundancy, and excluding `lo` handed
`tap_oracle` a ceiling of 0.00632 against `tap_learned`'s 0.01426. That, not capacity, is the
leading explanation of the surprise that the endogenous rung beat the oracle rung.

### The ladder correction (11 arms) and the measured noise floor

`precompute_tap_all` adds the block-entry prior over all three components (its root+hi columns
agree with the cached tap at **max|Δ| = 0.00e+00**, so the new arms are on exactly the same
footing), and `--stream-seed` moves **only** the window sampler — same data, same
initialisation, a different stream position, which is `recital`'s convention for a noise floor.

**The floor: |Δ val nll| = 0.00068 (plain pair) and 0.00094 (tap_learned pair)**, two-sided.
Per level, as % of the plain twin's per-level nll: lvl0 **2.30** · lvl1 0.82 · lvl2 0.57 ·
lvl3 1.17 · lvl4 0.60 · lvl5 0.86 · lvl6 0.17. lvl0 has one position per block and is
noise-dominated; **no per-level claim at lvl0 is resolvable.**

| arm | payload | val nll | vs plain | × floor |
|---|---|---|---|---|
| `tap_shuffled` | root+hi, permuted | 1.4015 | +0.0002 | 0.3 (inside) |
| `tap_learned_shuffled` | estimator, permuted stream | 1.4017 | +0.0004 | 0.5 (inside) |
| `plain_str43` | — (noise arm) | 1.4020 | +0.0007 | 0.9 |
| `tap_oracle` | root+hi | 1.3998 | −0.0015 | 1.9 |
| `tap_oracle_full` | root+hi+lo | 1.3966 | −0.0047 | 5.9 |
| `tap_learned` | endogenous, 32 blocks | 1.3942 | −0.0070 | 8.8 |
| `tap_learned_str43` | (noise arm) | 1.3933 | −0.0080 | 10.0 |
| **`tap_oracle_lo`** | **lo only** | **1.3933** | **−0.0079** | **9.9** |

Both information-destroyed controls sit **inside** the measured floor; every informative tap
sits 2–10× outside it.

**Delivered fraction of ceiling** against the exact cross-context prizes
(`Dm_window − Dm_history`): root+hi 0.00632, lo 0.00746, joint 0.01426. **Soft-ceiling caveat**:
the per-component prizes sum to 0.01379 against a joint 0.01426, so the split is approximate —
treat the lo ceiling as ~0.0075–0.0085, not a hard bound.

| rung | ceiling | delivered | fraction |
|---|---|---|---|
| `tap_oracle` (root+hi) | 0.00632 | 0.0015 | 24% |
| `tap_oracle_full` (all three) | 0.01426 | 0.0047 | 33% |
| `tap_learned` (endogenous) | 0.01426 | 0.0070 | 49% |
| `tap_oracle_lo` (lo only) | ~0.0075–0.0085 | 0.0079 | ~93–105% |

**The per-level attribution.** `tap_learned`'s profile correlates **+0.975** with
`tap_oracle_lo`'s (max difference 0.55 pp, inside the per-level floor) and only +0.786 with
`tap_oracle`'s root+hi profile (max difference 3.03 pp). **The endogenous estimator's gain has
the level signature of a lo entry prior, not of a slow-question signal.** The resolvable core of
that signature is lvl1–lvl2 (5.0–5.5× the floor), not lvl0.

Mechanism this suggests, unproven: the tap supplies the *bottom-level* mixture and the payoff
lands on the *top-level* positions — knowing `P(rule | feature)` at the leaves makes each
observed leaf pair sharper evidence about its parent, and that sharpening propagates upward.

### Half-2 gates

| Gate | What it asserts | Measured |
|---|---|---|
| **H-1** | `block_loglik` (one stack pass) equals `qfilter.prefix_loglik(...).sum(-1)` (T passes) | max\|Δ\| ≤ **1.4e-14** |
| **H-2** | the closed-form aleatoric label equals the BP oracle's `H_irr` at D = L−1 | max\|Δ\| **2.2e-16** |

## Dead, confounded and mis-read instruments — do not trust these at face value

Five things this round measured that do not mean what their names suggest. All were caught by a
guard that was in the design; each is recorded so the next round does not rediscover it.

1. **The theta-probe's null is 0.25, not 1/K = 0.143.** The OU stationary law on a 7-point grid
   is far from uniform (mass concentrates at the centre), so a majority-class head already scores
   ≈ 0.25. The shuffled-label control reads **0.249 / 0.256 / 0.250** (root/hi/lo) and *that* is
   the null every probe number must be read against. `figures/*/fig3` plots the shuffled null, not
   nominal chance.
2. **The `incoh`-trained model is a doubly-disadvantaged control and its loss gap cannot price
   the in-context question channel.** Measured exactly: `surp_incoh − surp_marg` = **0.2079**
   nats/token, of which only **0.0411** is the in-context channel (`surp_marg − surp_theta`). The
   other 0.167 is that the `incoh` process's per-node marginals are not the drift process's
   node-*conditional* marginals — a product-form mismatch, not a missing question model. Report
   the exact Bayes ladder, not the transformer gap alone.
3. **A history filter run with the *drift* kernel on a *static* world does not decay.** The
   first attempt at the "demand → 0 under static theta" sanity check kept `P_trans` at κ = 0.15
   while the world's theta was frozen, and read a flat 0.030 nats/token across all ten deciles of
   the corpus. That is a real (and separately interesting) statement — a learner that wrongly
   believes the question drifts never stops paying for it — but it is *not* the sanity check.
   With the **identity** kernel the check passes cleanly (1500 static blocks): the history channel
   goes **0.0006 → 0.0000** nats/token across corpus deciles and its prior `H(theta)` collapses
   0.131 → **0.000** (the question is fully identified), while the window channel stays flat at
   **0.0477**. The mis-specified version sits at 0.0295 → 0.0293 with `H(theta)` pinned at 3.88.
4. **`struct_vs_syn` is not demand-matched, and cannot be made so by banding.** Both families are
   drawn from the same bottom-quartile demand band and the oracle `Dm` guard still reads
   **0.73–0.78** at d6–d3: structure-news and demand-news genuinely co-occur at the token level.
   Only **`demand_vs_syn`** is exactly matched on the other channel (`S` guard **0.500** by
   construction, since both families have `S_chain ≡ 0`), so it is the one clean single-channel
   contrast. `demand_vs_struct` is the maximally-separated contrast (S guard 0.000, Dm guard
   1.000) and is read as a sign test, not as a matched AUC.
5. **Matching strata must be built within position.** Under drift the marginal surprisal is
   continuous rather than atomic, so global quantile edges give uneven occupancy per (position,
   bin) cell — the self-matched guard read **0.73** on the smoke run. Per-position atom-aware
   edges with an explicit offset (no `pos * K + bin` aliasing) bring it to 0.50–0.55.

## Runs on disk

Volume `rhm-scaling-data` (**`chromatic`**), under `/data/v16_s2_L6_m4_distinct/question_model/`.

| tag | what it is |
|---|---|
| `smoke` | 200-step base + FM per world and a tiny `decompose`, for pipeline validation only |
| `w0` | the two 12k-step bases + temporal FMs (`drift` own-world val **1.3969**, `incoh` **1.5567**); the `uniform` arm is `conditional_revision`'s cached `base_8L8H256D_steps12000_seed42.pt`, unmodified |
| `dec0` | the main Half-1 decomposition run — 6500 aligned blocks (4000 probe-train / 500 calib / 2000 test), ~75 min on an L4. **G-7 passes** (three-way identity residual 2.6e-4 – 1.2e-3 nats against a total of 1.152 at every D); **G-8 passes with max\|Δ\| = 0.00e+00** against `conditional_revision/oracle.py` |

| `h2` | Half 2: the shared 200k-block precompute, **eleven** 12k-step arms (six original + `tap_oracle_full`, `tap_oracle_lo`, `tap_learned_shuffled`, and the two `_str43` noise arms), and `evaluate`. Held-out val nll: plain 1.4013 · tap_oracle 1.3998 · tap_shuffled **1.4015** · tap_learned **1.3942** · tap_learned_shuffled **1.4017** · loss_alea 1.4099 · loss_alea_blind 1.4134. Both capacity-matched, information-destroyed controls land within 0.0004 of the plain twin |

Fetched copies and figures under `figures/<tag>/`.

## Reproduce

```bash
cd experiments/            # MODAL_PROFILE=chromatic

modal run -m rhm.question_model.question_model::selfcheck
python3 -m rhm.question_model.calibrate            # local, numpy only, ~4 min

modal run --detach -m rhm.question_model.question_model::train_world --world drift --tag w0
modal run --detach -m rhm.question_model.question_model::train_world --world incoh --tag w0
modal run --detach -m rhm.question_model.question_model::decompose --tag dec0
```

## Inherited, not copied

`../conditional_revision/oracle.py` (`_upward`/`_downward` structure, junction-tree KL, node and
clique marginals, `_leaf_evidence`, `_norm`, `_kl`, `_ent`), `../conditional_revision/gates_ab.py`
(`_partial_r2`, `_strata`, `_stratified_auc`, `_auc`, the atom-aware matching discipline and the
ancestor-chain probe design), `../practice/setlist/demand.py` (the demand framing, the
sweep-don't-solve lesson, the prewarm warning), `rhm/rhm_drift.py` (`weights_from_theta`,
`sample_derivations_weighted`, the OU form and its per-block KL decomposition), `rhm/model.py`,
`rhm/rhm_latent_loop.py` (`_generate_with_traces`), `a2a_forward/forward_model.py`. **Nothing in
those files is modified.**
