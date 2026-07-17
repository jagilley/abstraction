# Specialization on RHM: can the depth frontier be moved by *allocation*?

**Status**: WIP, single rule-seed. Three experiments done. (1) Token-reweighting doesn't move the frontier — breadth-restriction **inert**, level-reweighting **harmful**; only a *direct* deep target recruits depth. (2) Concentrating even that *direct* target onto a subtree (cut-2a) **fails to specialize** — it *hurts* the target subtree (broad beats narrow: the **Student-B > Student-A** result) and can't make the complement shallow, because RHM's single shared ruleset has **no domain-specific deep structure**. This is the first sharp **RHM-vs-language disanalogy**, and it motivates a *partially-heterogeneous* DGP (cut-3).
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

Exps 1–2 killed the *token-reweighting* route on both axes. The load-bearing distinction is now empirical: **a direct deep *target* is the only thing that recruits depth.** Cut-2a asks the sharper question with that direct target.

---

## Cut-2a — concentrating a *direct* deep target: still no specialization (broad beats narrow) ([`rhm_spec_direct_target.py`](rhm_spec_direct_target.py))

**Does concentrating a direct deep target (the oracle ancestor CE) onto subtree A buy *selective* depth (deep-A, shallow-B) and/or an *advantage* on A over applying it uniformly?** The advantage should appear at high synonymity, where even the direct target is capacity-limited when spread over the whole tree.

**Design** — single controlled variable = the A-concentrated term (λ=1). Every condition trains flat NTP on **all** roots (the shared substrate — cut-1 showed A's deep structure needs the rules learned from all data); only the added term varies. A = roots {0,1,2,3}. Swept `m ∈ {4, 8, 12}` (occupancy 0.25 / 0.50 / 0.75).
- `ntp` — floor. · `aux_all` — oracle deep target on **all** roots. · `aux_A` — oracle deep target on **A only** (the specialization condition). · `ntp_A` — extra NTP on A only (token-reweight control).

### Concentration never helps — it *hurts*, most where it should help

Concentration advantage = `aux_A − aux_all` on A-eval (MLP), across the difficulty sweep:

| level | m=4 (aux_all at ceiling) | **m=8 (aux_all capacity-limited)** | m=12 |
|---|---|---|---|
| d4 | +0.000 | **−0.140** | −0.054 |
| d5 | −0.001 | **−0.132** | ≈0 |
| d6 | −0.002 | **−0.098** | +0.001 |

m=8 is the regime *designed* to reveal an advantage (`aux_all` there is genuinely limited — d4 0.58, d5 0.44, far from ceiling). Concentration **hurts most exactly there**. The hypothesis predicted `aux_A > aux_all`; every cell is ≤0.

### Why: the shared grammar, cutting both ways

Depth-on-A at m=8 (MLP), and what happens to **B** (the complement `aux_A` never supervises):

| d4 recovery | ntp | aux_all | aux_A |
|---|---|---|---|
| on **A** | 0.100 | 0.581 | 0.440 |
| on **B** (complement) | 0.108 | 0.527 | **0.408** |

`aux_A` supervises the deep target on **A only**, yet it lifts **B**'s d4 from 0.108 → 0.408 (~75% of the way to `aux_all`'s 0.527) — supervising A's ancestors teaches the *shared rules* that compute B's ancestors too. So:
- **No selectivity** — can't make B shallow; A's deep target floods into B through the shared grammar.
- **Concentration hurts A** — `aux_all`'s broader data (B included) teaches the hard shared rules *better*, deepening A. Restricting to A **discards B-data that was helping A**. The transfer runs both ways.

**Direct ≫ reweight still holds** (`aux_A` d4 0.44 vs `ntp_A` 0.10 at m=8) — cut-1 reconfirmed.

### The reframe: this is *Student B > Student A*, not a null

The result only reads as "failure" against the "specialization = unique depth" hypothesis. Read the other way it is a clean **positive**: the broadly-trained model (`aux_all`) recovers A's *own* deep structure **better** than the model that concentrated on A (`aux_A`) — at m=8, A's d4 is **0.58 (broad) vs 0.44 (narrow)**. That is Sutskever's "broad + light tuning > 10,000 h narrow," in a controlled DGP: **when deep structure is shared, breadth produces more robust deep representations than concentration, even for the target domain.** Homogeneous RHM *demonstrates* the shared-deep-structure phenomenon rather than failing to model it — and "go uniquely deep on A" is the wrong frame when A's depth *is* shared.

### The disanalogy, and where it points (→ cut-3)

This is the first sharp disanalogy between standard RHM and language: RHM's single global ruleset means every subtree shares ~all generative machinery below the root (recovering d1–d5 uses shared rules; only d6 has a thin A-specific slice), so RHM has **no domain-specific deep structure to specialize on** — the shared signal is *total*, which forbids specialization. Language specializes because domains have genuinely non-shared deep structure.

## Next — cut-3: giving RHM genuine domains (context-sensitivity favored)

A 2026-07-16 web-research pass ("where does shared/transferable structure live in a representational hierarchy?") found the question genuinely **unsettled and largely definitional** (three different "shared": transfer-under-finetuning, cross-model convergence, generative-structure-of-the-data). The tension maps onto our discriminative-vs-generative confusion: in a discriminative net "abstract" = the *output/class* end (task-specific by construction), and RHM's root *is* the class — so "specialization at the output end" ↔ "domain-specific at the RHM root." The most-analogous prior art — **Kemp & Tenenbaum, "The discovery of structural form"** (PNAS 2008), a literal hierarchical-Bayesian generative model of knowledge — places a **shared abstract *form*** (tree/grid/ring topology) on top, **domain-specific *structure*** in the middle, and surface data at the bottom. (A caution from the same pass: where nets *actually* specialize, e.g. MoE experts, it's **surface/lexical/operational**, not deep-semantic-domain.)

**The sharpened gap (from the 2026-07-16 discussion).** cut-2a already shows RHM *has* shared-abstraction transfer — supervising A's deep target alone lifts B's d4 **0.11 → 0.41**, and broad beats narrow on A — so the transfer mechanism is **not** missing. What's missing is **genuine domains**: subtrees A and B are the *same generative process with a relabeled root* (they share the surface, mid, *and* deep rules), so RHM's transfer is **intra-domain** (more data for one homogeneous process), never the **cross-domain** transfer of Student B (two genuinely different surfaces sharing deep abstractions). RHM has transfer but no *distance to transfer across*; every cut-3 variant below is a way to give it one.

The evidence does **not** license an a-priori choice, so cut-3 adds a domain-specific slice in variants that share everything else, and measures which reproduces genuine **selective** depth (concentration finally beats breadth on A's domain-specific levels; B stays shallow there):
- **(c) context-sensitive rules — favored; emergent, graded heterogeneity from a *shared* vocabulary.** Standard RHM is **context-free**: a feature expands via a rule that ignores its parent/siblings/context, and leaves are conditionally independent given the tree. Making the DGP **context-sensitive** — a node's expansion depends on its neighbors (the Chomsky-hierarchy step CF→CS, where language actually lives) — lets the *same* low-level rule vocabulary realize differently under different contexts, so A's root induces a different *effective* sub-grammar than B's: genuine A-specific deep statistics **without hard-coding separate rule tables per domain**. This is how language gets transferable-yet-specialized structure (a word means different things in medical vs legal text). More elegant than (a)/(b)'s explicit heterogeneity — though still a graded *form* of it — and whether context alone yields enough A-specificity for concentration to beat breadth is the open empirical question. (Bidirectional / undirected-MRF couplings — parent and child as a symmetric jointly-sampled factor — are a related construction to explore; the context-sensitive reading is the one favored here.)
- **(a) domain-specific-deep** — top-K composition rules per-domain, shared leaf-ward rules (the discriminative-transfer / RHM-root=class reading). The blunt, hard-coded version of (c).
- **(b) K-T-grounded** — shared topology + domain-specific *structure* in the upper-middle + shared surface rendering (Kemp–Tenenbaum's form / structure / data levels).

## Files & reproduction

| file | what |
|---|---|
| [`rhm_subtree_specialization.py`](rhm_subtree_specialization.py) | Exp 1 — FULL vs SUBTREE breadth control (self-contained). |
| [`../rhm_latent_loop.py`](../rhm_latent_loop.py) | Exp 2 — `ntp_levelfocus@<β>` conditions live here (additive to the shared latent-loop harness; `ntp`/`ntp_aux` are the floor/ceiling anchors). |
| [`rhm_spec_direct_target.py`](rhm_spec_direct_target.py) | Cut-2a — concentrate a *direct* deep target (oracle ancestor CE) on A: `ntp` / `aux_all` / `aux_A` / `ntp_A`, all sharing flat-NTP-on-all-roots; m∈{4,8,12}; per-subtree (A/complement/full) probing (self-contained). |

```bash
cd experiments/
# Exps 1–2 ran on jagilley (co-located with the historical RHM depth results):
# Exp 1 — breadth control (smoke: add --quick):
MODAL_PROFILE=jagilley modal run --detach rhm/specialization/rhm_subtree_specialization.py::subtree_specialization
# Exp 2 — level-focus sweep (floor / uniform-aligned / deep-skew / hard-skew / oracle):
MODAL_PROFILE=jagilley modal run --detach rhm/rhm_latent_loop.py::latent_loop \
    --conditions "ntp,ntp_levelfocus@0,ntp_levelfocus@1,ntp_levelfocus@2,ntp_aux" --ensemble-n 0 --tag levelfocus
# Cut-2a — direct-target concentration sweep (ran on chromatic, the default credit-bearing workspace;
#   split per-m for parallelism, or pass all at once):
modal run --detach rhm/specialization/rhm_spec_direct_target.py::spec_direct_target --m-values "4,8,12"
```

Results JSON: exps 1–2 on the **jagilley** `rhm-scaling-data` volume (`rhm_subtree_specialization/…`, `rhm_latent_loop/…_levelfocus…`); cut-2a on the **chromatic** `rhm-scaling-data` volume (`rhm_spec_direct_target/v16_s2_L6_distinct_A4of16_r0-1-2-3_m{4,8,12}_N20000/`).

## Caveats

- **Single rule-seed** throughout (per repo convention; no reason to suspect seed-sensitivity for these directional claims). Cut-2a is single-seed *per m*.
- Exp 2's level-focus is tested only as **token-loss reweighting** — the correct test for *that* hypothesis. Cut-2a is the direct-target version, with the *privileged* oracle latent target (the clean mechanism gate); the *non-privileged* (EMA / own-lifted-latent) direct target remains untested — but cut-2a shows the concentration *mechanism* itself doesn't buy specialization on a shared grammar, so the non-privileged version is deferred behind the cut-3 DGP change.
- The aligned-batching confound in Exp 2 is handled by the β=0 arm; the confound-free weighting conclusion is the *within-aligned* β sweep.
- **Cut-2a magnitude confound (not direction):** `aux_A`'s aux target draws from ~50k A-sequences vs `aux_all`'s 200k, so part of `aux_A`'s deficit is fewer distinct examples; the clean control is an equal-count A-pool. It can't rescue the hypothesis — there is no advantage even at m=4 (ceiling, plentiful data), and the *direction* (concentration ≤ broad) is exactly what the two-way transfer mechanism predicts. Still worth running before hardening the magnitude.
- Depth is probed on clean full sequences; the S/A-conditioning inflation (deep-level absolutes on a root-restricted eval, root chance `1/|A|`) is noted inline — it affects absolutes, not the cross-condition Δ (all conditions share the same eval).
- **Workspaces:** exps 1–2 on jagilley, cut-2a on chromatic (credits) — so the specialization results are split across two `rhm-scaling-data` volumes.
