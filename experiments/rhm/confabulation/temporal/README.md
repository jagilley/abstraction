# The temporal confabulation test — is the composite's privileged access epistemically charged? (2026-08-11)

**Design doc**: [ideas/temporal_confabulation_test.md](../../../../ideas/temporal_confabulation_test.md) — pre-registered before any code was written
**Parent**: [../README.md](../README.md) — the depth battery, forked here verbatim
**Lineage**: [../../conditional_revision/README.md](../../conditional_revision/README.md) — the temporal FM idiom, the exact BP oracle, and the atom-aware matching discipline, all imported rather than reimplemented
**Code**: [`temporal_confabulation.py`](temporal_confabulation.py) · **Files**: [FILES.md](FILES.md)
**Status**: OL (`ntp_aux`) complete across the full instrument sweep. `cr_base` complete at the
default instrument capacity, remaining capacities still running. CL (`ntp_aux_cl`) unrun.

## One-line arc

The depth battery found the composite (M + FM) has **privileged access to an epistemically inert
quantity**. Shifting the FM's conditioning gap from six blocks to one token gives the residual a
genuinely charged content — but the privileged part is not the charged part. The first-person
advantage survives the axis change at **+0.014 to +0.027** (against a control band of +0.002 to
+0.011, and the depth arm's +0.053 to +0.059), while its correlation with the exact oracle's belief
revision `B_t` at matched surprisal is **0.0000** at every capacity and every level. The decomposed
columns say why: the self-report and a capacity-matched third-party observer track `B_t`
*equally well* (partial R² 0.007 each), so the difference between them carries none of it.

> **The temporal residual is charged, and the charge is public.** The revision content is in M's
> state and is equally recoverable from M's tokens and logits by a 2-layer observer. Whatever the
> composite knows privately about `r_temp`, it is not what the token taught it.

This is the design doc's second pre-registered falsifier, which it flagged as the most live one on
the strength of [`local_loss`](../../conditional_revision/local_loss/README.md)'s finding that
temporal training targets reproduce depth signatures. It is a slightly different statement than that
doc anticipated, and the difference matters — see *What this establishes*.

## The object

```
depth    :  a6[t]     =  FM_d(a0[<=t])[t]                 +  r_depth[t]
temporal :  h6[t+1]   =  h6[t] + FM_T(h6[<=t])[t]         +  r_temp[t]
             ^ actual     ^ the self-theory's forecast of     ^ what the
              next state    the next state, available          token added
                            BEFORE x_{t+1} arrives
```

Both decompositions are measured on **one frozen M**, at **one set of report positions**, with FMs
trained in a shared forward pass — so the inert-vs-charged comparison is at matched machinery rather
than across experiments. `FM_T` uses `conditional_revision` Gate 0's update parametrisation
(`pred = fm(h6) - h6`), matched heads, frozen base.

### Three things the design doc left open, and how they were resolved

**Report position.** `r_temp[t]` does not exist at position `t` — it requires `h6[t+1]`. So the
report is emitted at `p = t+1`, which is the design doc's "the difference is physically present on a
wire inside the system, at the right time" made literal. Every target, observer and oracle column in
this experiment is indexed to `p ∈ [1, T-2]`, so the two arms and all standing controls are
position-matched to the row.

**The temporal confabulator differs in kind from the depth one.** Its access is
`h6[t] + FM_T(h6[<=t])` — the state the self-theory forecasts before the token lands. That is the
exact structural analogue of the depth arm's `FM(a_i)`, but it cannot see the arriving token *at
all*, because that token **is** the conditioning gap. Its margin (+0.48 to +0.51) is therefore not
the depth arm's +0.13 in a different regime, and margins are **not comparable across arms here**.
Only advantages are. (The depth battery's `ENT` row — margin +0.267 with advantage −0.035 — is the
standing cautionary case for exactly this.)

**A real ceiling exists on this axis, and only on this axis.** `h6[<=p]` contains `h6[p]` *and*
everything `FM_T` conditions on, so an observer over `h6` is literally `FM_T` + a subtraction + a
probe. This is the validated upper bound the depth battery's *Next steps* #1 asked for and its
`O_act` failed to provide.

## Harness validation — this is a controlled fork

`ntp_aux` reproduces `val = 1.5447` against the depth battery's published **1.5447**, and the depth
arm — re-run inside this experiment at the temporal report positions — reproduces the published OL
row:

| | published OL | here |
|---|---|---|
| IMPL advantage | +0.06 | +0.053 to +0.059 |
| IMPL margin | +0.08 to +0.17 | +0.058 to +0.170 |
| steer ratio | 1.33–1.68× | 1.32–1.71× |
| BEHAV / ENT / WORLD advantage | +0.002 / +0.007 / +0.011 | +0.002 / +0.006 / +0.011 |
| `ens_cos` (depth) | 0.874–0.911 | 0.845–0.906 |

The oracle independently reproduces `conditional_revision`'s model-free DGP table:
`R²(B ~ exact surprisal)` = **0.072 / 0.112 / 0.223** at d4/d3/d2 against the published
0.071 / 0.115 / 0.230, and `mean B`/`frac B ≡ 0` at d2 = 0.370 / 0.383 against 0.361 / 0.383. The
BP identity self-check passes at rel. error 7e-3 – 1.6e-2 (Monte-Carlo error at n = 2000).

The `cr_base` arm loads `base_8L8H256D_steps12000_seed42.pt` and measures `val = 1.5675` against
gate0's recorded 1.5698 (different eval batch counts) — the same frozen model Gates A and B,
`aleatoric_fraction` and `tracking` all ran on.

## The regime is clean

| instrument | %pred | depth cos / `ens_cos` | temporal cos / `ens_cos` |
|---|---|---|---|
| h4m0.25 | 1.4 | 0.778 / 0.906 | 0.769 / 0.945 |
| h16m1 | 5.6 | 0.836 / 0.882 | 0.827 / 0.939 |
| h64m2 | 16.7 | 0.861 / 0.848 | 0.840 / 0.941 |
| h128m4 | 33.5 | 0.873 / 0.845 | 0.848 / 0.943 |

Not the saturation regime (cosine 0.77–0.87, not 0.99) and not the junk regime (`ens_cos` 0.85–0.95,
not 0.65). **`ens_cos` reads higher on the temporal arm and this is expected rather than reassuring**:
the aleatoric component is input-determined by construction, so every independent FM misses `x_{t+1}`
identically. On this axis `ens_cos` rules *out* FM-idiosyncratic noise; it does not rule *in* a
computational gap. Temporal hierarchy η² is concentrated at d1 (0.045–0.053) rather than d5–d6, the
opposite of the depth residual's profile — consistent with a leaf-adjacent, token-driven object.

## Results — OL (`ntp_aux`, val 1.5447)

| target | inst | self | confab | margin | `O_input` | best `O_io` | **advantage** | base | steer |
|---|---|---|---|---|---|---|---|---|---|
| **TEMP-IMPL** | h4m0.25 | 0.908 | 0.397 | +0.511 | 0.835 | 0.889 | **+0.019** | 0.196 | 1.09 (0.97) |
| **TEMP-IMPL** | h16m1 | 0.899 | 0.389 | +0.509 | 0.823 | 0.884 | **+0.014** | 0.184 | 1.17 (0.98) |
| **TEMP-IMPL** | h64m2 | 0.896 | 0.407 | +0.489 | 0.820 | 0.871 | **+0.025** | 0.173 | 1.15 (0.97) |
| **TEMP-IMPL** | h128m4 | 0.891 | 0.412 | +0.479 | 0.814 | 0.864 | **+0.027** | 0.178 | 1.15 (0.96) |
| IMPL (depth) | h4m0.25 | 0.840 | 0.670 | +0.170 | 0.676 | 0.781 | **+0.059** | 0.185 | 1.62 |
| IMPL (depth) | h16m1 | 0.791 | 0.658 | +0.133 | 0.567 | 0.732 | **+0.059** | 0.175 | 1.71 |
| IMPL (depth) | h64m2 | 0.702 | 0.632 | +0.070 | 0.521 | 0.644 | **+0.058** | 0.154 | 1.32 |
| IMPL (depth) | h128m4 | 0.686 | 0.628 | +0.058 | 0.521 | 0.633 | **+0.053** | 0.167 | 1.43 |
| TEMP-MAG | h16m1 | 0.780 | 0.573 | +0.207 | 0.669 | 0.732 | +0.048 | 0.251 | |
| DEPTH-MAG | h16m1 | 0.724 | 0.633 | +0.091 | 0.570 | 0.658 | +0.066 | 0.251 | |
| BEHAV | h16m1 | 0.772 | 0.746 | +0.026 | 0.760 | 0.770 | +0.002 | 0.584 | |
| ENT | h16m1 | 0.987 | 0.822 | +0.165 | 0.908 | 0.981 | +0.006 | 0.252 | |
| WORLD | h16m1 | 0.559 | 0.488 | +0.071 | 0.529 | 0.547 | +0.011 | 0.121 | |

*advantage = self − best capacity-matched `O_io`. Steering shows residual/prediction flip ratio at
matched KL, with the scale-matched variant in parentheses. Raw `self` is not comparable across
capacity rows (each has its own k-means clustering and baseline).*

**Privileged access survives the axis change, at about a third to a half the depth arm's size.**
`TEMP-IMPL` is above the control band at every capacity, and it is not the junk signature — `ens_cos`
is flat at 0.939–0.945 across the whole 24× sweep, so the advantage is not largest where the residual
is noisiest. It does drift upward with instrument capacity (+0.014 → +0.027) where the depth arm is
flat; unexplained.

**The `O_input` column is where the size difference lives.** Tokens alone reach 0.823 against a
self-report of 0.899 on the temporal axis; on the depth axis the same column reads 0.567 against
0.791. The temporal residual is largely token-determined, which is the exposure the design doc named
("the token *is* visible to the observer, unlike the depth residual's drivers").

### The observer ladder stays flat, and a ceiling finally binds

`TEMP-IMPL`, h16m1 — the design doc's open question 3:

| observer | 1L/64D | 2L/128D | 4L/192D | 8L/256D | half data |
|---|---|---|---|---|---|
| `O_input` (tokens) | 0.796 | 0.823 | 0.815 | 0.811 | — |
| `O_io` (tokens + logits) | 0.871 | 0.884 | 0.873 | 0.868 | 0.858 |

Flat from 64D to 256D, plateauing by 2L/128D, barely moved by halving the data — the same
access-limited-not-resource-limited shape the depth axis showed. `WORLD` remains the internal control
that the ladder *can* climb when the information is in the input (`O_input` 0.226 → 0.412 → 0.529).

**`O_h6` = 0.905 > self 0.899 > `O_io` 0.884 > `O_act` 0.853.** A properly ordered ladder with a
validated ceiling, which the depth battery never had (its `O_act` came in *below* `O_io` at 0.666 vs
0.680 and bounded nothing). The ceiling holds on `cr_base` too (0.873 > 0.862). It also says the
self-report leaves something on the table: there is more `r_temp` information in `h6` than M's own
report head extracts.

### Test 5 — the charge validation

The headline. Per-position `adv = 1[self correct] − 1[best O_io correct]`, regressed on the exact
oracle `B_joint` with exact BP surprisal partialled out, on 2000 held-out sequences (124,000
positions):

| | d4 | d3 | d2 |
|---|---|---|---|
| partial `R²(self-report ~ B \| bp)` | 0.0063 | 0.0078 | 0.0074 |
| partial `R²(observer ~ B \| bp)` | 0.0066 | 0.0074 | 0.0063 |
| **partial `R²(advantage ~ B \| bp)`** | **0.0000** | **0.0000** | **0.0000** |
| shuffled control | 0.0001 | 0.0001 | 0.0001 |
| partial `R²(advantage ~ bp \| B)` | 0.0022 | 0.0021 | 0.0016 |
| AUC(adv), position × exact-surprisal matched | 0.4999 | 0.4976 | 0.4929 |
| guard: exact surprisal against itself | 0.5040 | 0.5002 | 0.4995 |
| guard: adv permuted within position | 0.4977 | 0.4972 | 0.5018 |

Zero to four decimal places at every capacity and every level, while the *same* advantage carries
20–50× more surprisal structure than revision structure. All guards land at 0.497–0.504. The depth
arm, the built-in null, reads 0.0000–0.0013 against shuffled 0.0000–0.0009 — also null, as the
design predicted.

**The decomposed columns are the finding.** Both parties track `B_t`, and they track it equally, so
the difference carries none of it. The revision content is not absent from M's state and is not
absent from the report — it is simply not *private*.

**On power.** The shuffled control sits at 0.0001–0.0002, so this measurement resolves partial R²
down to roughly 0.001. Gate A's positive reading on this substrate was **0.150** at d1 and 0.102 at
d2. An effect two orders of magnitude below Gate A's would still have been visible here.

### Test 3 — steering, and why the scale-matched arm earned its place

At matched KL on the logits (BEHAV-flip controls matched at 0.043–0.045 vs 0.045–0.053):

| arm | residual span | self-theory state span | ratio | vs scale-matched update span |
|---|---|---|---|---|
| temporal, h16m1 | 0.089 | 0.076 | 1.17× | **0.98×** |
| depth, h16m1 | 0.223 | 0.131 | **1.71×** | — |

The literal fork of the depth arm's contrast compares an update-scale residual against a state-scale
forecast, and reads 1.09–1.17× — a weak apparent positive. Against the forecast *update* (same scale)
it reads **0.96–0.98× at every capacity**: no asymmetry at all. The depth arm's 1.32–1.71× is real
and this one is not.

## The `cr_base` arm — the null is substrate-robust

Run on the exact frozen checkpoint Gates A and B used, where the model's belief probe reads 82–90% of
its Bayes ceiling at d1–d2 and Gate B fires at 0.690. Default instrument capacity, h16m1:

| target | self | best `O_io` | advantage | | `ntp_aux` advantage |
|---|---|---|---|---|---|
| TEMP-IMPL | 0.862 | 0.836 | **+0.026** | | +0.014 |
| IMPL (depth) | 0.722 | 0.667 | +0.055 | | +0.059 |
| TEMP-MAG | 0.852 | 0.806 | +0.046 | | +0.048 |
| BEHAV | 0.765 | 0.766 | −0.001 | | +0.002 |
| ENT | 0.985 | 0.976 | +0.009 | | +0.006 |
| WORLD | 0.459 | 0.531 | **−0.072** | | +0.011 |

partial `R²(adv ~ B | bp)` for `TEMP-IMPL`: **0.0000 / 0.0000 / 0.0000** (shuffled 0.0002 / 0.0001 /
0.0002), AUC 0.501 / 0.501 / 0.502. The charge null reproduces on the substrate where the charge was
measured, so "the aux-latent model is not where revision is readable" is not the explanation.

Two things this arm settles about the choice of substrate:

- **The depth steering asymmetry collapses without the latent target**: residual/prediction = **0.87×**
  here against 1.71× on `ntp_aux` and 1.33–1.68× published. The depth *advantage* (+0.055) survives
  fine, but Test 3 does not. Running only `cr_base` would have left the inert-vs-charged comparison
  unreadable — which is why the primary arm is `ntp_aux`, per latent-loop's finding that RHM's
  non-latent targets carry inverted self-knowledge.
- **`WORLD` goes to −0.072** — the third party beats the self-report substantially on world-state, as
  criterion (2) says it should, and more visibly than on a model trained with an ancestor-supervision
  aux head. Closer to the depth battery's published CL row (−0.038).

## What this establishes, and what it does not

**Establishes** (one regime, two substrates):

- Privileged access to the FM residual is **not specific to the depth axis**. Shifting the
  conditioning gap by one token leaves a positive, capacity-swept, guard-cleared first-person
  advantage (+0.014 to +0.027 OL, +0.026 on `cr_base`).
- **That advantage carries no oracle belief-revision structure at matched surprisal**, to a
  resolution two orders of magnitude below Gate A's positive on the same substrate — while carrying
  measurable surprisal structure.
- The reason is symmetry rather than absence: **self-report and capacity-matched observer track
  `B_t` equally well**. This is a different statement than the depth arm's "privileged access to an
  inert quantity" — here the quantity is charged and the charge is simply not the private part.
- The observer ladder's flat-in-capacity, access-limited shape **survives the axis change**.
- A validated ceiling exists on the temporal axis (`O_h6` above the self-report on both substrates),
  the first in this line, and it shows the self-report does not exhaust what `h6` carries.
- The scale-matched steering contrast is **1.00× within noise** on the temporal axis against
  1.32–1.71× on the depth axis.

**Does not establish:**

- Anything about **use**. Every readout here is a trained report head. The design doc scoped this as
  reportable access from the start; whether the wire gets consumed by M's own downstream computation
  when merely available is the separate, later experiment, and this result does not bear on it.
- That the temporal residual is uninformative. It is demonstrably charged
  ([`aleatoric_fraction`](../../conditional_revision/aleatoric_fraction/README.md): 0.665 of the ideal
  arity-1 residual; the oracle here confirms 38–46% of positions have `B ≡ 0` while carrying ~1 nat of
  surprisal). What is null is the *privacy* of that charge, not its existence.
- Anything about the **closed loop**. CL (`ntp_aux_cl`) is unrun. On the depth axis closing the loop
  roughly doubled the advantage; whether it moves a partial R² of 0.0000 is untested, and we should
  not assume the answer.
- **Cross-target comparability of advantage magnitudes.** `TEMP-MAG` (+0.048) exceeds `TEMP-IMPL`
  (+0.014), and `DEPTH-MAG` (+0.066) exceeds `IMPL` (+0.059) — which contradicts the design doc's
  pre-registration of magnitude as a null control. But the magnitude targets are 4-way against the
  direction targets' 8-way, and have more headroom (`TEMP-MAG` self 0.780 / `O_io` 0.732 against
  `TEMP-IMPL`'s 0.899 / 0.884), so compression near ceiling is a live alternative explanation. The
  pre-registered "no advantage beyond `O_io`" for magnitude is falsified; the ordering between
  magnitude and direction should not be read off these numbers.
- Why the temporal advantage **rises with instrument capacity** while the depth arm's is flat.
- Seed robustness, or transfer off this regime.

**A result that did not survive.** In the OL arm alone, all 12 `TEMP-IMPL` stratified mean
differences were negative (depth arm: 6/12), suggesting the advantage *shrinks* where the token
revises beliefs. This does **not** reproduce on `cr_base` (z = −0.1, +1.3, +1.2 at the same cells)
and should be treated as substrate-specific noise, not a finding. It is recorded here because the
sign consistency looked compelling in one arm and was not.

## Child: `epistemics/` — the question this result opened, and its answer

Publicity is a *feature* for the operational question, and this harness is the right instrument for
it: it reads the temporal channel at 0.899 against a validated 0.905 ceiling, where `rule_family`'s
belief probe died at 11–17% of its ceiling, and no privileged access is needed to measure a public
channel. [`epistemics/`](epistemics/README.md) therefore repoints the battery from **privacy** to
**composition** — dropping the report channel, ladder, steering and CL, keeping the frozen M (the same
checkpoint, `val` 1.5447), the report positions, the instrument sweep, the guards and the oracle — and
measures a source × target decode matrix at matched *readout* capacity.

**The pre-registered concentration effect is not there.** The raw update `Δ` decodes the oracle's
belief revision at least as well as the residual `r` at every readout capacity, every level, every
instrument capacity and on both substrates (d2, MLP-64: `Δ` 0.449 vs `r` 0.362 on `ntp_aux`; 0.411 vs
0.366 on `cr_base`), and the subtraction is mildly lossy. This is not forecaster weakness: the arm
carries an **exact ε₁ = 0 innovation** built by counterfactual substitution, which itself only ties
`Δ` on `cr_base`, and on `cr_base` the learned forecast already reaches `cos = 0.989` with the exact
conditional mean. So on this substrate the temporal FM looks **epistemically inert as a signal-former**,
leaving *timing* as its remaining distinct claim — a control-topology question this design cannot
reach. It also runs `conditional_revision`'s never-run **Gate C** end to end: capacity invariance
passes (flat in instrument capacity at the linear rung), and the martingale calibration gets an exact
null (0.99 self-sampled, 1.09 corpus) against which the learned residual's 2.0–4.0 drift is mostly ε₁
bias. Two by-products: 13–17% of surprisal-residualised `B` turns out to be pure **position**, and the
design doc's magnitude negative control — falsified in this experiment's own `TEMP-MAG` row — behaves
exactly as predicted under a direct decode.

## Reproduction

```bash
cd experiments/          # NOT the repo root -- see gotchas

# wiring smoke (minutes; numbers meaningless by construction -- deliberately the artifact regime)
modal run -m rhm.confabulation.temporal.temporal_confabulation::temporal_confabulation_test --smoke

# the pre-registered primary (~4h on an L4 including the 20K-step wake train, which is
# cached into the DEPTH battery's own ckpt path so both experiments share the frozen M)
modal run --detach -m rhm.confabulation.temporal.temporal_confabulation::temporal_confabulation_test \
    --conditions "ntp_aux" --tag ol

# the Gates A/B substrate -- loads the cached plain-NTP base, trains nothing (~2h, measurement only)
modal run --detach -m rhm.confabulation.temporal.temporal_confabulation::temporal_confabulation_test \
    --conditions "cr_base" --tag crbase

# the unrun follow-up arm
modal run --detach -m rhm.confabulation.temporal.temporal_confabulation::temporal_confabulation_test \
    --conditions "ntp_aux_cl" --tag cl
```

Results land on the `rhm-scaling-data` volume (**`chromatic` workspace**) under
`/data/rhm_confabulation/v16_s2_L6_m4_distinct/temporal/`: `{ol,crbase}_results.json`, plus the
cached model-independent oracle `oracle_rs0_seed999_n2000_D2-3-4.npz`. Wake checkpoints are shared
with the parent battery at `/data/rhm_confabulation/v16_s2_L6_m4_distinct/ckpt/`.

## Gotchas worth not rediscovering

- **The temporal residual does not exist at the position it is indexed by.** `r_temp[t]` needs
  `h6[t+1]`, so the report must be emitted at `t+1`. Getting this wrong asks the head to report a
  quantity that is not yet on any wire.
- **The naive residual-vs-prediction steering contrast is scale-confounded on this axis** and reads
  1.15× where the scale-matched version reads 0.97×. On the depth axis the two spans are both
  state-scale and the problem does not arise; do not port the depth contrast over unexamined.
- **The temporal confabulator cannot see the arriving token**, so temporal and depth margins are not
  comparable. Only advantages are, and this is the same lesson the depth battery's `ENT` row taught.
- **`O_h6` is a ceiling for the temporal arm and only for it.** The depth residual needs `a0`, which
  `h6` does not cheaply supply.
- **A high `ens_cos` means less on this axis than on the depth axis.** The aleatoric component is
  input-determined, so independent FMs agree on it by construction.
- **Test 3 on the depth arm is degenerate on a plain-NTP base** (0.87×). The latent aux target is
  load-bearing for the steering asymmetry, though not for the advantage.
- **Launch Modal from `experiments/`, not the repo root** — `modal run -m rhm...` from the root fails
  with `ModuleNotFoundError: No module named 'rhm'`, and a trailing `echo` in the launch script will
  mask the nonzero exit code.
