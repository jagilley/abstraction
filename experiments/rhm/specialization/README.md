# Specialization on RHM: can the depth frontier be moved by *allocation*?

**Status**: WIP. Two controls done, single rule-seed. Both **confirm the prediction**: reallocating or reweighting the *token* objective does **not** move the depth frontier — neither laterally (breadth/subtree, **inert**) nor vertically (level/depth, **actively harmful**). Only a *direct* deep target (oracle-aux) recruits depth. This is the "reweighting ≠ direct target" thesis confirmed on both allocation axes, and it sets up the real payoff: a *non-privileged direct* deep target (cut-2).
**Date**: 2026-07-16
**Parent context**: [../RHM_LATENT_LOOP_README.md](../RHM_LATENT_LOOP_README.md) (frontier-moving-vs-capped; the "two novelties" — sample-novelty refills breadth, only abstraction-novelty moves depth). The framing below (specialization = non-uniform depth allocation; the two axes; the standing-frontier idea) is from the 2026-07-16 discussion that motivated this bucket.

---

## The idea

**Specialization = non-uniform *depth* allocation across the RHM tree.** At high synonymity `m`, next-token gradient at depth is diluted (KFW `vm^{ℓ+2}`), so a fixed-capacity model spends its depth budget ~uniformly and **stalls at ~d3.5** (root recovery ≈ 0.08) — shallow *everywhere*. The specialization question: can you push the frontier deeper *somewhere* by **concentrating** the objective — either onto a **subtree** (breadth axis) or onto **deep-level tokens** (level axis)?

The arc's prior thesis predicts **no**: under plain NTP the deep directions are *unrecruited* (idle capacity), and the gradient that would organize them is diluted before it reaches the surface — so *reweighting a diluted target* cannot recruit them; only delivering a **direct deep target** (a label for the deep feature) does. These two experiments are the load-bearing controls for that claim, on the two allocation axes. They must hold before any *positive* specialization result (a non-privileged direct target, cut-2) is meaningful.

## Shared regime & method

- **DGP**: `v=16, s=2, L=6, m=4`, distinct rules, single rule-seed (`rule_seed=0`). The canonical frontier-stall regime.
- **Model**: causal GPT **8L/8H/256D** (~6.34M), identical init across conditions, 20k steps, AdamW, matched budget.
- **Readout**: per-level ground-truth **ancestor recovery** `d1` (shallow, local) … `d6` (root), best-over-blocks linear + MLP probe — the RHM-only measurement of "how deep the representation reaches."
- **Anchors reproduced exactly** (harness validated): NTP floor `d1 0.98 / d3 0.88 / d4 0.51 / root 0.08`; oracle-aux ceiling `root 0.80`.

---

## Experiment 1 — breadth restriction is INERT ([`rhm_subtree_specialization.py`](rhm_subtree_specialization.py))

**Single controlled variable = breadth of the training distribution.** Two conditions, identical init / steps / optimizer / flat phase-diverse NTP, matched token budget:
- **FULL** — roots uniform over all 16.
- **SUBTREE** — roots restricted to `S = {0,1,2,3}` (25% of the tree).

Both probed on a held-out eval drawn from **S** itself (the tightest test — does concentrating on S deepen S's own domain?).

### Depth recovery on S (MLP-best-over-blocks)

| level | FULL | SUBTREE | Δ (SUB − FULL) |
|---|---|---|---|
| d1 | 0.981 | 0.985 | +0.003 |
| d2 | 0.965 | 0.968 | +0.003 |
| d3 | 0.885 | 0.880 | −0.005 |
| **d4** | 0.541 | 0.554 | **+0.013** |
| **d5** | 0.279 | 0.285 | **+0.005** |
| **d6 (root)** | 0.299 | 0.303 | **+0.004** |

**Every delta is within ±0.013 — probe noise.** Concentrating breadth (matched tokens) yields *identical* depth on the subtree's own domain, including the deep levels where it might have specialized. **Breadth restriction does not convert into depth** — the scaling-era insight (whence "run everything at frontier scale") reproduced on a controlled DGP.

Two riders:
1. **SUBTREE got a marginal NTP edge on S** (val 1.561 → 1.553, −0.008 nats) with **zero depth benefit** — a fresh in-domain restatement of "NTP-loss improvement ≠ compositional depth."
2. **SUBTREE did not *forget* the complement** at the learnable levels (SUBTREE-on-complement `d1–d4` ≈ FULL-on-complement, e.g. d3 0.878 vs 0.881). The low-level rules learned from S are the *same* rules the complement uses (**shared grammar**), so they transfer for free. "Specialize on A, ignore B" is inert in *both* directions — the shared-substrate / "signal from a nominally-unrelated region" phenomenon in its clean form.

**Measurement caveat**: deep-level *absolute* recovery on S is inflated (d5/d6 ≈ 0.28/0.30 vs ~0.16/0.08 on the full tree) because restricting roots to S constrains the high-level features (root chance is `1/|S|=0.25` on S vs `1/16` full). Both models share this, so the FULL-vs-SUBTREE **Δ is unaffected**; and d6 0.30 ≈ its 0.25 chance floor, i.e. the root is still unrecovered even on S.

---

## Experiment 2 — level reweighting is HARMFUL (`ntp_levelfocus@<β>` in [`../rhm_latent_loop.py`](../rhm_latent_loop.py))

**Single controlled variable = how deep-level focus is delivered.** New additive condition family in the latent-loop harness: sequence-aligned NTP (so a target position's hierarchy level is well-defined) with each position's CE weighted by a deep-skew function of its level `f(ℓ)=((ℓ+1)/count[ℓ])^β`, mean-1 normalized. **β=0** = uniform-aligned (batching-only control); **β>0** skews gradient toward the root. Anchored by `ntp` (floor) and `ntp_aux` (oracle direct target, ceiling).

### Final depth recovery (MLP-best-over-blocks)

| level | `ntp` floor | `lf@0` uniform-aligned | `lf@1` deep-skew | `lf@2` hard-skew | `ntp_aux` oracle |
|---|---|---|---|---|---|
| d1 | 0.980 | 0.640 | 0.733 | 0.449 | 0.988 |
| d2 | 0.963 | 0.623 | 0.615 | 0.233 | 0.992 |
| d3 | 0.878 | 0.583 | 0.541 | 0.150 | 0.993 |
| **d4** | 0.507 | 0.340 | 0.286 | 0.160 | 0.979 |
| **d5** | 0.167 | 0.147 | 0.141 | 0.141 | 0.951 |
| **d6 (root)** | 0.078 | 0.073 | 0.070 | 0.063 | **0.796** |

Prediction was "level-focus stalls near the floor"; reality is **stronger — every level-focus arm is at or below the `ntp` floor everywhere**. Two separately-isolated effects:

1. **Aligned-batching penalty** (`lf@0` vs `ntp`): even *uniform* aligned NTP drops d1 0.98 → 0.64 — the phase-diversity artifact the latent-loop harness documents ("aligned batching depresses d1 to ~0.60"). The β=0 arm exists precisely to isolate this from the weighting.
2. **Deep-skew penalty** (within the aligned family, β=0 → 1 → 2, batching held fixed, so confound-free): **skewing gradient toward deep levels monotonically *reduces* depth recovery** — d4 0.340 → 0.286 → 0.160; by β=2 the model collapses to near-chance at d3 (0.150).

**Mechanism** = the bottom-up-learning result made causal: **deep composition is built *on top of* the shallow substrate, so starving L0/L1 of gradient removes the foundation the deep levels compose from.** "Fixate on high-level tokens" is self-defeating. Only the *direct* oracle target — which hands the deep feature as a label rather than reweighting a diluted next-token prediction — breaks the d5/d6 stall (root 0.08 → 0.80).

*(The flat-eval val column is not comparable for aligned-trained arms — aligned-train/flat-eval mismatch, not signal; the probes are the readout.)*

---

## Joint synthesis

| axis | intervention | effect on depth |
|---|---|---|
| **breadth** (exp 1) | restrict to a subtree | **inert** — no depth gain on S; no complement loss (shared rules transfer) |
| **level** (exp 2) | reweight the token loss toward deep positions | **harmful** — monotonically *worse* depth |
| **(reference)** | direct deep target (oracle-aux) | **recruits depth** — root 0.08 → 0.80 |

**Neither reallocating breadth nor reweighting the diluted token target toward depth moves the frontier.** The "boredom ≈ oracle-aux" intuition is **falsified**: oracle-aux's power is the *direct label*, not the fixation on deep tokens — and fixating on those same tokens via reweighting is not merely non-equivalent, it is counterproductive. This is the arc's "reweighting ≠ direct target" thesis confirmed on **both** allocation axes, on the depth-recovery axis directly comparable to the oracle ceiling for the first time.

## What this sets up (next / cut-2)

The load-bearing distinction is now empirical: **a direct deep *target* is the only thing that recruits depth; reweighting the token loss cannot supply one, on either axis.** So the *non-privileged* specialization experiment must generate a **direct self-target** — an EMA / own-lifted-latent target (the untested "drop the privilege" piece the arc keeps deferring, RHM_LATENT_LOOP next-step #2) — and **"boredom" belongs there as the scheduler that *lifts the target's abstraction height* as each level saturates, NOT as a reweighter of the token loss** (which we've now shown backfires). Whether *that* can be pointed selectively (at a subtree, or serially across subtrees → a self-generated "standing frontier") is the specialization payoff.

## Files & reproduction

| file | what |
|---|---|
| [`rhm_subtree_specialization.py`](rhm_subtree_specialization.py) | Exp 1 — FULL vs SUBTREE breadth control (self-contained). |
| [`../rhm_latent_loop.py`](../rhm_latent_loop.py) | Exp 2 — `ntp_levelfocus@<β>` conditions live here (additive to the shared latent-loop harness; `ntp`/`ntp_aux` are the floor/ceiling anchors). |

```bash
cd experiments/    # results volume + comparison data live on the jagilley workspace
# Exp 1 — breadth control (smoke: add --quick):
MODAL_PROFILE=jagilley modal run --detach rhm/specialization/rhm_subtree_specialization.py::subtree_specialization
# Exp 2 — level-focus sweep (floor / uniform-aligned / deep-skew / hard-skew / oracle):
MODAL_PROFILE=jagilley modal run --detach rhm/rhm_latent_loop.py::latent_loop \
    --conditions "ntp,ntp_levelfocus@0,ntp_levelfocus@1,ntp_levelfocus@2,ntp_aux" --ensemble-n 0 --tag levelfocus
```

Results JSON on the `rhm-scaling-data` volume: `rhm_subtree_specialization/v16_s2_L6_m4_distinct_S4of16_r0-1-2-3_N20000/` and `rhm_latent_loop/v16_s2_L6_m4_distinct_8L8H256D_S20000_levelfocus...`.

## Caveats

- **Single rule-seed** (per repo convention; no reason to suspect seed-sensitivity for these directional claims).
- Exp 2's level-focus is tested only as **token-loss reweighting** — the correct test for *this* hypothesis, but not the direct-target version (that is cut-2 by design).
- The aligned-batching confound in Exp 2 is handled by the β=0 arm; the confound-free weighting conclusion is the *within-aligned* β sweep.
- Depth is probed on clean full sequences; the S-conditioning inflation in Exp 1 is noted inline (affects absolutes, not the FULL-vs-SUBTREE Δ).
