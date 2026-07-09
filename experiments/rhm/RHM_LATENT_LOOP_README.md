# The latent target is load-bearing for generalizable self-knowledge on RHM (2026-07-04)

**Code**: `rhm_latent_loop.py` — one Modal/GPU (L4) function running the `target × loop` 2×2, a per-condition `λ_local` sweep (via `ntp_cl@<λ>` condition names), and a fresh-FM ensemble agreement test (`--ensemble-n`).
**Origin**: [SLEEP_CHUNKING_RHM_README.md](SLEEP_CHUNKING_RHM_README.md) — the "predict your own lifted latents, not tokens" (Korchinski–Favero–Wyart) line this operationalizes inside the cerebellar loop.
**Sibling**: [../a2a_forward/README.md](../a2a_forward/README.md) — the forward-self-model / closed-loop / self-knowledge machinery (validated on MNIST + language) ported here.
**Depends on**: [RHM_DEEP_COMPOSITION_README.md](RHM_DEEP_COMPOSITION_README.md) — the `oracle_aux` latent signal, the learned frontier (~d3.5), the BP/greedy reference lines, and the occupancy law that fixes the regimes.
**Complements**: [RHM_FM_REGULARIZER_README.md](RHM_FM_REGULARIZER_README.md) — showed self-knowledge is **not** needed for *functional simplification*; this doc is about the separable question of *self-knowledge itself*.

## One-line arc

Closing the cerebellar loop produces **generalizable self-knowledge on RHM for the first time** — but only under a **latent** (own-lifted-DGP) target, **not** a next-token target at any hierarchy depth or loop strength. The mechanism: the FM residual is a **map of the model's computational frontier**; a token target structures it at the shallow levels the model already reaches, a latent target moves it to the deep levels no small FM can predict — making the residual **FM-invariant** and the self-knowledge that encodes it **generalize** across forward models.

## Why this experiment

We had a standing blind spot: the closed-loop self-knowledge that appears robustly on MNIST and language had **never** shown up on RHM (fresh-FM Δ R²(CL−OL) was zero or negative everywhere — [RHM_SPARSE_RATCHET](RHM_SPARSE_RATCHET_README.md) even saw it *invert*). The sleep-chunking line reframed the whole RHM program as implicitly about **latent vs token targets** and isolated the load-bearing variable: **what you train the model to predict**, not what you feed it. This experiment crosses that variable with the self-knowledge loop to ask whether they are orthogonal — and, if not, why.

Two knobs, kept as **separate additive loss terms** so the factorial is clean:

```
wake loss = NTP  +  λ_aux · aux_loss(inter, oracle_ancestor_labels)   [latent target]
                 +  λ_local · local_loss(residual, gate)               [self-knowledge loop]
```

- **target ∈ {token, latent}** — is the model given an *undiluted* deep signal (`oracle_aux`'s per-block ancestor CE, from RHM_DEEP_COMPOSITION) on top of NTP? The latent target uses ground-truth latent **values** (Level-0 privilege), so this is a **diagnostic** ("given the model *can* climb, how does self-knowledge behave"), not a self-supervised escape.
- **loop ∈ {open, closed}** — is the a2a apparatus on (FM predicts the model's own `post_block6` from `post_block0`, injects via a unified gate after block 1, gate-scaled local loss)?

Four cells: `ntp` (token/open), `ntp_aux` (latent/open), `ntp_cl` (token/closed), `ntp_aux_cl` (latent/closed).

## Setup

- **DGP**: distinct-rule RHM, **s=2, L=6**, at two regimes — **v16/m4** (occupancy 0.25; a real learnable-but-unlearned frontier: token-NTP stalls ~d3.5, BP root 0.80) and **v16/m2** (occupancy 0.125; token-NTP reaches the root, BP root 0.95 — no frontier gap). Same rule seed throughout.
- **Model M**: causal GPT **8L/8H/256D** (~6.34M), identical across all conditions/regimes (only `m` changes between regimes — one variable).
- **FM**: `TransformerForwardModel`, 1 layer, 8 heads, d_head 16 (~264K, matched-head so the residual reflects *computational* gaps, not architectural mismatch). `post_block0 → post_block6`, inject after block 1.
- **Measurements** (all on held-out aligned eval sequences, at the last token): per-level latent recovery vs BP/greedy (linear + MLP probes); FM-residual effective rank, per-level feature η², cosine; **self-knowledge** = R² of the FM-residual *direction* from block activations, read off a **fresh** matched FM (apples-to-apples) and the **co-trained** FM; and a **fresh-FM ensemble** (4 FMs, different seeds) → input-centered pairwise residual cosine (high = FM-invariant/DGP-aligned, low = FM-idiosyncratic noise).

**Critical methods fix (phase-diverse NTP).** The first pass inherited `oracle_aux`'s aligned-sequence batching, in which the last position (the probe/η² position) never receives an NTP gradient — the exact artifact RHM_DEEP_COMPOSITION flags for `ntp_only`. It depressed the token baseline (`ntp` recovered d1≈0.60 instead of ~0.98) and contaminated the token residual. The fix: train NTP on **random windows of a concatenated corpus** (every position supervised) while applying the aux term on a **separate aligned-labeled forward pass**. After the fix `ntp` recovers d1=0.98, reproducing `thread_b`. All results below are post-fix.

## Experiment 1 — m4 2×2: the residual tracks the frontier; first generalizable RHM self-knowledge needs the latent target

| cond | target/loop | root(d6) | val | resNorm | d6 η² | d5 η² | **SK b7 (fresh)** | ct b7 | **ens_cos** |
|---|---|---|---|---|---|---|---|---|---|
| `ntp` | token / OL | 0.078 | 1.554 | 49.1 | 0.003 | 0.007 | 0.516 | — | 0.881 |
| `ntp_aux` | latent / OL | 0.796 | 1.545 | 24.1 | 0.295 | 0.381 | 0.557 | — | 0.984 |
| `ntp_cl` | token / CL | 0.073 | 1.746 | 1.2 | 0.003 | 0.006 | **−0.670** | 0.314 | 0.821 |
| `ntp_aux_cl` | latent / CL | 0.807 | 2.383 | 8.9 | 0.322 | 0.421 | **+0.779** | 0.907 | 0.983 |

(BP root 0.80; `ntp` reproduces thread_b's frontier: d1 0.98, d3 0.88, d4 0.51, root 0.08. `ntp_aux`→BP everywhere, root 0.80. Both known cells reproduce — harness validated.)

**Δ SK (CL − OL), the MNIST/language self-knowledge quantity (positive there):**
- **token**: −0.67 − 0.52 = **−1.19** (the RHM inversion, replicated with a matched 8-head FM)
- **latent**: +0.78 − 0.56 = **+0.22** — *positive, first time on RHM, and comparable to language's +0.18.*

Two findings:

1. **Residual structure = frontier map.** The token residual is structured only at the shallow levels the model reaches (d2 η²=0.16, deep ≈0.003); the latent target moves the structure **deep** (d5 η²=0.42, d6=0.32, shallow drops away). Independent of the loop. This is the RHM analog of MNIST's low-rank digit-discriminative residual — *residual structure sits wherever the model's frontier is*, and the latent target pushes the frontier to the root.
2. **Self-knowledge appears only under the latent target.** Closing the loop on a token target inverts SK (−1.19); on a latent target it produces genuine positive self-knowledge (+0.22). Both targets compute the *same* hierarchy at BP under aux, so this is a pure self-knowledge effect, not a "computes-more" effect.

## Experiment 2 — m2 control: a deep residual is **not** sufficient (a falsified prediction)

We predicted: since m2 token-NTP reaches the root, its residual is deep-structured, so token self-knowledge should work at m2 like it does on language. **It doesn't.**

| cond | root | val | resNorm | d6 η² | d5 η² | SK b7 (fresh) | ct b7 |
|---|---|---|---|---|---|---|---|
| `ntp` | 0.819 | 0.826 | 24.6 | 0.186 | 0.347 | 0.463 | — |
| `ntp_aux` | 0.956 | 0.826 | 28.2 | 0.501 | 0.454 | 0.443 | — |
| `ntp_cl` | 0.433 | 1.047 | 0.8 | 0.060 | 0.121 | **−1.078** | 0.414 |
| `ntp_aux_cl` | 0.955 | 1.321 | 4.2 | 0.559 | 0.276 | **+0.505** | 0.961 |

- **Target-axis null confirmed** (the built-in control): `ntp` ≈ `ntp_aux` at every level (both ≈BP, root 0.82/0.96) — the aux adds nothing where there is no frontier to climb.
- **The residual-depth prediction held** (m2 token residual *is* deep-structured: d5 η²=0.35 vs m4 token's 0.007) — which **deconfounds** depth from the aux and shows the sign of Δ SK tracks the **aux/latent target, not the residual depth**: token Δ SK = **−1.54** at m2 despite a deep-structured residual; latent Δ SK = +0.06.

So "a deep-structured residual is sufficient for self-knowledge" is false. The latent target is doing something the residual's depth (measured open-loop) does not capture.

## Experiment 3 — the loop decomposes: injection offloads, local loss consolidates

Sweeping `λ_local` on token-CL at m2 (where there's a root to preserve/strip):

| cond | root | val | resNorm | d6 η² | SK b7 | Δ SK b7 | ens_cos |
|---|---|---|---|---|---|---|---|
| `ntp` (OL) | **0.819** | 0.826 | 24.6 | 0.186 | 0.463 | — | 0.901 |
| `ntp_cl@0.0` (inj-only) | 0.453 | 1.855 | 13.4 | 0.038 | 0.398 | **−0.06** | 0.792 |
| `ntp_cl@0.1` | 0.335 | 1.777 | 2.0 | 0.024 | −0.081 | −0.54 | 0.815 |
| `ntp_cl@0.3` | 0.363 | 1.300 | 1.1 | 0.035 | −0.381 | −0.84 | 0.848 |
| `ntp_cl@1.0` | 0.433 | 1.047 | 0.8 | 0.060 | −1.078 | −1.54 | 0.886 |
| `ntp_aux_cl@1.0` | **0.955** | 1.321 | 4.2 | 0.559 | +0.505 | (+0.06) | 0.933 |

The closed loop bundles **two opposing effects**, which λ_local dissociates:

- **Injection → offloads.** *Every* closed condition, including injection-only (λ=0), strips root from 0.82 → ~0.4. The FM preview lets M not compute the deep levels itself; this is λ-independent. (We initially blamed the local loss for offloading — this run corrected that: it's the injection.)
- **Local loss → consolidates.** As λ rises, standalone val *improves* (1.86 → 1.05, less injection-dependent) and the residual compresses (13.4 → 0.8) — M internalizes the injected computation into its own weights — but that same consolidation monotonically **inverts** the fresh-FM SK (−0.06 → −1.54).

**Key negative result:** loop aggressiveness **cannot substitute** for the latent target. Token SK is non-positive at *every* λ_local — the loop only preserves the pre-existing open-loop shared-basis SK (λ=0) or destroys it (λ>0); it never *generates* self-knowledge. `ntp_aux_cl` remains the sole positive-Δ, deep-η² condition. This rules out the "maybe a gentler loop rescues token SK" confound.

## Experiment 4 — why latent self-knowledge generalizes (fresh-FM ensemble)

Train 4 fresh FMs (different seeds) per final model; measure input-centered pairwise residual cosine. High = the residual is determined by the **input** (FM-invariant, a DGP/capacity gap any FM misses identically); low = determined by the **FM** (idiosyncratic noise). m4 (where the token-CL residual is genuine noise, norm 1.2):

| cond | resNorm | d6 η² | **ens_cos** | SK b7 (fresh) | ct b7 |
|---|---|---|---|---|---|
| `ntp` (token OL) | 49.1 | 0.003 | 0.881 | 0.516 | — |
| `ntp_aux` (latent OL) | 24.1 | 0.295 | **0.984** | 0.557 | — |
| `ntp_cl` (token CL) | 1.2 | 0.003 | 0.821 | −0.670 | 0.314 |
| `ntp_aux_cl` (latent CL) | 8.9 | 0.322 | **0.983** | 0.779 | 0.907 |

**Latent residuals are near-ceiling FM-invariant (0.98)** — any fresh FM leaves essentially the *identical* residual, exactly what a pure inference-depth gap (the root, un-computable by a 1-layer FM) predicts. Token residuals are lower (OL 0.88, CL 0.82), and closing the loop lowers agreement further. (m2's ensemble did *not* separate — 0.79–0.93 for all — because at m2 the model computes the full hierarchy, so *every* residual is input-determined; the test only bites where the token residual is genuinely noise, i.e. m4.)

**But FM-invariance is necessary, not sufficient.** Token-CL has a *moderately* invariant residual (0.82) yet inverted fresh SK (−0.67): the fresh FMs agree on the residual's *shallow* DGP part, but M — having offloaded the deep levels and consolidated onto its *specific* co-trained FM (ct b7=0.31) — doesn't linearly expose it. Generalizing self-knowledge needs **both** (i) an FM-invariant/DGP-aligned residual **and** (ii) M actually encoding that structure. The latent target uniquely supplies both: the aux makes the residual a deep DGP/capacity gap **and** forces M to represent it (ct b7=0.91, fresh SK=+0.78).

## Synthesis

1. **The FM residual is a map of M's computational frontier.** Its per-level structure sits at whatever depth M reaches — shallow under token-NTP (m4), deep under a latent target or wherever token-NTP itself reaches deep (m2, language, MNIST-by-depth-1).
2. **RHM self-knowledge existed but was mis-measured.** Against the *co-trained* FM it is positive in every closed condition (ct b7 0.31–0.96); the "blind spot" was reading it off *fresh* FMs, which only agree with M when the residual is DGP-aligned.
3. **The latent target is the unique, non-substitutable route to *generalizable* positive self-knowledge on RHM** — not a token target at any frontier depth (m2/m4), not any λ_local. It alone makes the residual FM-invariant **and** anchors M to encode it. Language/MNIST get generalizable SK from a plain token target because *their* task densely/shallowly supervises the deep computation; RHM's diluted NTP never does, so it needs the latent anchor.
4. **The closed loop decomposes**: injection = offload/preview (wake-like division of labor), local loss = consolidation/internalization (sleep-like), with a real tension between consolidation and fresh-FM-measured self-knowledge. **[Update 2026-07-06: the distillation ablation refines both labels. "Preview" is unsupported where measured — injected passes carry *less* deep structure, not more (ablation finding 4, ceiling caveat noted there). And internalization of the NTP function turns out to be cheap — a 5K injection-off fine-tune recovers standalone val fully without local loss (finding 1) — so the local loss's distinctive contribution is not function-consolidation but keeping the SK encoding through consolidation (finding 2, pending decomposition).]**

## Connections to our beliefs (discussion, not yet crystallized)

- **[dimensionality_expansion.md](../../beliefs/dimensionality_expansion.md)** frames the residual as the frontier (`R_res` = the slice of `R_act` the self-model hasn't absorbed). "Residual structure tracks the frontier" is the *per-level* refinement of that: the residual is not just *how much* is un-absorbed but *where in the hierarchy*. It also sharpens the belief's "novelty raises `R_act`" — on RHM, i.i.d. sample-novelty is saturated yet the frontier is fixed; only **abstraction-novelty** (a deeper target) raises the achievable `R_act`. Sample-novelty refills breadth; a latent target unlocks depth.
- **Computational, not epistemic, self-knowledge** (a2a OOD_ROBUSTNESS): the FM-invariant latent residual is precisely the "computational / distribution-invariant" self-knowledge the a2a paper argued for. RHM shows the *precondition* for it — a target that makes the residual DGP-aligned.
- **FM-specific vs FM-general SK** (a2a FORWARD_MODEL_SWAP): consistent — token-CL is FM-specific (co-trained ≫ fresh; the post-injection layer carries the FM-specific signature); the latent target is what pulls it into FM-general territory.
- **FM-as-functional-regularizer** (RHM_FM_REGULARIZER) established self-knowledge is *not* needed for functional simplification. This is the mirror: functional simplification is not sufficient for self-knowledge either — self-knowledge needs a target that keeps a substantive DGP-aligned residual, which pure compression removes.

We are **not** crystallizing a belief yet — single-seed (below), and the compounding question (next) is what would elevate this from "self-knowledge appears" to "self-knowledge does work."

## Caveats

- **Single rule seed, single run per cell.** Directionally consistent across four runs and two regimes, and the harness is deterministic (the sweep's `ntp_cl@1.0` reproduced the m2 control's root/SK exactly), but **magnitudes are not hardened** — a 2–3 seed replicate is the precondition for any strong claim. Note the m4 SK magnitudes shift ~0.1 between an A10G and an L4 run (sign/direction stable), underscoring single-seed magnitude fragility.
- **Latent target uses privileged latent values** (`oracle_aux`, Level-0). This is a diagnostic of *whether* self-knowledge can appear given a climbable target — **not** a self-supervised mechanism. The self-supervised version (EMA/data2vec-style own-lifted-latent target) is untested here.
- **`ntp_aux_cl` has degraded standalone val (2.38 at m4)** — the aux+injection combination drives strong dependence; SK is real but read off a somewhat NTP-degenerate standalone model. A λ_local / FM-capacity sweet-spot sweep to get SK *without* the val hit is wanted. **[Resolved 2026-07-06: a 5K injection-off CE+aux sleep phase closes the val gap completely (1.535, marginally better than OL) while keeping SK positive — see the distillation-consolidation ablation, finding 3. No sweet-spot sweep needed.]**
- **The fresh-vs-co-trained and ensemble geometries have subtleties** (SK measured on the injection-removed model; an FM-invariant residual need not be encoded by M). The load-bearing metric is the fresh-FM Δ R²(CL−OL); the co-trained and ensemble numbers are corroboration.

## Reproduction

```bash
cd experiments
# m4 2x2 + ensemble (canonical):
modal run --detach -m rhm.rhm_latent_loop::latent_loop \
    --conditions "ntp,ntp_aux,ntp_cl,ntp_aux_cl" --ensemble-n 4 --tag ens
# m2 control (target-axis null; deep residual not sufficient):
modal run --detach -m rhm.rhm_latent_loop::latent_loop --m 2 \
    --bp-line "d1 1.00 d2 1.00 d3 1.00 d4 1.00 d5 .992 root .954" \
    --greedy-line "(m2: no frontier gap)"
# m2 lambda_local sweep + ensemble (injection-vs-local-loss dissociation):
modal run --detach -m rhm.rhm_latent_loop::latent_loop --m 2 \
    --conditions "ntp,ntp_cl@0.0,ntp_cl@0.1,ntp_cl@0.3,ntp_cl@1.0,ntp_aux_cl@1.0" \
    --ensemble-n 4 --tag lamsweep \
    --bp-line "d1 1.00 d2 1.00 d3 1.00 d4 1.00 d5 .992 root .954" \
    --greedy-line "(m2: no frontier gap)"
```

Results on the `rhm-scaling-data` volume under `/rhm_latent_loop/`.

## Next steps

1. **Seed replicate (precondition).** 2–3 rule seeds of the m4 2×2 to harden the Δ SK sign and the ensemble separation before any belief is crystallized.
2. **Compounding — the "click" (the real next chapter).** `ntp_aux_cl` is the first genuine positive-self-knowledge setup on RHM. This finally makes the abstraction-ratchet question askable: **does this self-knowledge compound across wake–sleep cycles** (injection → distill/consolidate → re-point FM → repeat), or does it plateau like every prior RHM ratchet? Hypothetical, untested — but for the first time the precondition (self-knowledge exists) is met. *[2026-07-06: the cycle recipe is now pinned by the distillation ablation — wake with injection + local loss, sleep with plain CE+aux fine-tuning (no teacher KL needed), seeded from the `@1.0+distill` state.]*
3. **Drop the privilege.** Replace the oracle-latent aux with an EMA self-distilled own-lifted-latent target (data2vec-style, per the sleep-chunking line) and test whether generalizable self-knowledge survives without ground-truth labels — the self-supervised version of Experiment 1.
4. ~~**SK sweet spot.** λ_local / FM-capacity sweep on `ntp_aux_cl` to obtain positive Δ SK without the standalone-val degradation (the consolidation-vs-SK tension from Experiment 3).~~ *Superseded 2026-07-06: the distillation ablation resolves the val degradation directly (sleep phase closes it fully at unchanged SK sign); see finding 3.*

---

# Interpretation (appended 2026-07-06, from discussion)

Theoretical readings of the results above — interpretation and hypotheses, not new data. From a discussion of this experiment against the full a2a_forward + RHM arc.

## 1. Two kinds of residual, and why the FM-invariance requirement is near-definitional

The FM residual comes in two kinds that a scalar (norm, or even rank) cannot distinguish:

- **FM-idiosyncratic (capacity noise)**: *this* FM, with its particular seed and 264K params, misses something a differently-seeded FM would miss differently. From M's point of view there is no stable fact here to learn — "what my self-model gets wrong" is a moving target.
- **DGP-aligned (intrinsic inference gap)**: the computation is un-doable by *any* small FM (the root requires deep joint inference no 1-layer FM can perform), so every FM misses it *identically*. The difficulty is a property of the DGP, not of the FM.

The fresh-FM ensemble cosine (Exp 4) is the operational discriminator between the two. Once unpacked, "generalizable SK requires an FM-invariant residual" is close to a definitional identity: self-knowledge that transfers across FMs must encode something invariant to the choice of FM, and only a DGP-aligned residual is a *model-independent fact about M's own computation*. The empirical content of Exps 1–4 is therefore not that identity but *which training conditions produce a DGP-aligned residual that M also encodes* — and the answer is: only the latent target (necessity shown by the m2 control and the λ sweep; the encoding half shown by token-CL's 0.82-invariant residual with inverted fresh SK).

## 2. Why MNIST and language "just worked": the coupling RHM breaks

On MNIST, the class label **is** the root of the DGP — supervision hits the top of the tree directly, undiluted by construction (depth-1 in the relevant sense). On language, NTP is dense (every position supervised) and the hard semantic composition is exposed at the surface. In both domains the native objective silently drives M's frontier deep, so the precondition for generalizable SK — a deep, DGP-aligned residual that M represents — was satisfied *for free*, and SK looked like a property of the loop. RHM is the one domain where the token objective is structurally diluted at depth (KFW `vm^{ℓ+2}`), which **decouples "train on tokens" from "compute the deep structure"** and reveals the latter as the load-bearing hidden variable. The right reading of the historical RHM SK nulls is not "the a2a phenomena don't transfer to RHM" but "RHM is the controlled setting that exposes the precondition language and MNIST hide."

## 3. A candidate resolution of the MNIST–RHM ratchet gap (meta-learning without multiple tasks)

Root-level supervision keeps MNIST's frontier *moving* throughout training — strokes, then part-compositions, then class-discriminative structure. On a fixed dataset this is an implicit curriculum: the effective learning problem *at the frontier* is non-stationary, so each wake–sleep cycle finds genuinely new structure to consolidate and the residual keeps being refilled from above. That is plausibly why both the bilevel (WS_LG) and first-order (WS_UG) ratchets compounded on MNIST **without any explicit task shift**. On token-NTP RHM the frontier stalls at ~d3.5, the residual stops being refilled, and every ratchet variant exhausted after ~cycle 1 — same mechanism, opposite sign. This displaces the earlier gap candidates (bidirectional-vs-causal FM, classification-vs-NTP, residual rank per se) as the primary explanation, and it makes a sharp prediction for the compounding experiment (next-step 2): `ntp_aux_cl` should compound *while its frontier is still climbing toward the BP ceiling*, and exhaust once it gets there — after which sustained compounding requires abstraction-novelty (a deeper DGP, or new rule sets), not more i.i.d. samples.

## 4. Why token-CL SK goes negative rather than to zero

Closing the loop on a diluted target makes M reorganize around *complementing its specific co-trained FM* (and offloading the shallow structure it used to represent standalone). Its self-map becomes FM-specific — orthogonal-to-anti-aligned with what a *fresh* FM finds surprising. The open-loop model, having specialized to no FM, retains more fresh-FM-readable shared-basis structure. So on a diluted target the loop destroys the pre-existing shared-basis SK without building generalizable SK in its place — worst of both, hence Δ SK of −1.19/−1.54 rather than ≈0. General form: **consolidation and generalizable self-knowledge pull in opposite directions unless a substantive DGP-aligned residual is held open against the compression.** This is the mirror of [RHM_FM_REGULARIZER](RHM_FM_REGULARIZER_README.md): there, SK is not needed for functional simplification; here, functional simplification actively erodes generalizable SK.

## 5. What the latent objective is really doing: a constraint on representations, a freedom in the supervision

It does **not** add representational degrees of freedom. The capacity was always there — [RHM_DEEP_COMPOSITION 4d](RHM_DEEP_COMPOSITION_README.md) is the existence proof (same architecture reaches BP at every level given the signal); under plain NTP the deep directions are not absent but *unrecruited*, because the gradient that would organize them is diluted away before reaching the surface. The latent objective delivers gradient to that idle capacity — if anything it *removes* freedom, pinning previously-unconstrained directions to the hierarchy.

The structural novelty is on the **target side**. A token target is exogenous: a fixed function of the data, pinned to abstraction level 0 forever — dilution is fate. A latent target (in its self-supervised form) is *endogenous*: a function of the model's own representation, whose abstraction height is a free variable that rises with the model. Once level ℓ is consolidated, both target and context lift to level ℓ, and the next level is again the depth-flat local problem (DEEP_COMPOSITION Exp 1's ~0.77 at every rung; KFW's `vm³`). One line: **the latent objective performs a change of coordinates on the supervision, moving the target up the tree in lockstep with the model's frontier so the learning problem at the frontier is always locally strong.** This also makes it a native ratchet — each consolidated level becomes the platform from which the next is locally learnable — which is the mechanistic content of "learning novel knowledge = traversing further up the DGP tree than NTP allows."

Two corollaries to keep honest:

- **This experiment tested only the constraint half.** The aux target values are oracle-given, so target height was set by us, not by the model — undiluted deep signal without endogeneity. The freedom half (EMA/data2vec own-lifted-latent target, next-step 3) is where the characteristic risk enters: an endogenous target is free to *collapse* (predicting a constant is self-consistent), which is what the data2vec/DINO machinery exists to prevent. DEEP_COMPOSITION Exp 5b's invariance failures were early contact with exactly this. Prediction: the EMA-teacher version should succeed where 5b's constructions failed, because the teacher's already-consolidated levels give the target a floor to bootstrap from.
- **The objective and the loop are complementary halves, not separate gadgets.** The latent objective supplies rungs; the loop climbs them — injection explores beyond the standalone frontier (offload/preview), the local loss consolidates that exploration into weights (Exp 3's decomposition), which is what would license the endogenous target to lift another level next cycle. This is the working hypothesis for the compounding experiment. **[Update 2026-07-06: the distillation ablation's with-injection probes found no support for the "explores beyond the frontier" half in the aux regime — injected passes have far *worse* deep-latent recovery than standalone ones (d6 0.18 vs 0.80), i.e. injection displaces depth at the probe position rather than adding it. Scope caveat: the aux conditions are at the BP ceiling, so that cell had no headroom for exploration to show; the fair test (token-CL, root 0.07) is queued. Treat "injection = exploration" as unsupported speculation until then. The consolidation half is also qualified: consolidation of the NTP function needs no local loss (a plain injection-off fine-tune closes the val gap identically — see the ablation's finding 1); what the local loss uniquely appears to preserve is the self-knowledge encoding (finding 2, itself not yet decomposed).]**

## 6. Refinements to [dimensionality_expansion](../../beliefs/dimensionality_expansion.md) (beyond the Connections section above)

- **`R_res` needs a type label.** The health-triple treats residual rank as "frontier," but a high-rank residual can be either FM-idiosyncratic noise or a genuine DGP-aligned frontier (§1). The fresh-FM ensemble cosine is the discriminator the belief was missing; a health meter reading only `rank(residual)` will mistake capacity mush for frontier.
- **Two novelties, only one of which moves depth.** On RHM, i.i.d. *sample*-novelty is unlimited yet the token frontier stays pinned at d3.5 — sample-novelty refills breadth only. What moves the frontier up the hierarchy is *abstraction-novelty*: a deeper target. This changes the belief's prescribed intervention — the curiosity signal should select harder *targets/levels*, not just novel *samples*.

---

# Distillation-consolidation ablation: sleep re-derives, it does not transfer (2026-07-06)

**Code**: `rhm_latent_loop.py` — a `+distill` condition suffix (post-wake sleep phase) and with-injection probes, added to the same harness. Two Modal runs: the main 5-condition ablation (`--tag distill`) and a no-teacher control (`--tag distill-ce0`).
**Motivation**: On MNIST, distillation was load-bearing for the ratchet (GATED_RATCHET: WS compounds, CL_LG stalls), and the MNIST local-loss probes showed local-loss consolidation absorbs meta-knowledge over object-level 3:1. Question here: in the `ntp_aux_cl` setup, is distillation the missing consolidation mechanism — does it close the standalone-val gap (2.38 vs OL 1.55) that local loss doesn't, and what happens to the self-knowledge?

## Setup

All at m4 (v16, 8L/8H/256D, same harness/seed as Exp 1; `ntp_aux` and `ntp_aux_cl@1.0` reproduce the canonical run exactly — val 1.545/2.383, SK 0.557/0.779).

- **Sleep phase** (`+distill`): after the 20K-step wake, 5K steps at lr 1e-4. Teacher = wake-final model WITH its co-trained FM+gate injection (all frozen). Student = same weights continued WITHOUT injection. Loss = α·KL(per-token, student‖teacher) + (1−α)·CE(ground truth) + the aux term (kept on). α=0.5.
- **α=0 control** (separate run, seed-identical wake): the identical sleep phase with the KL term removed — pure injection-off CE+aux fine-tuning. This is the attribution instrument for the +5K-step asymmetry: it distinguishes "the teacher's KL transfers knowledge" from "any standalone fine-tuning closes the gap." (The a2a language distillation result never had this control.)
- **With-injection probes** (new, all closed conditions): val and per-level recovery measured on *injected* forward passes at wake-final, alongside the standalone measurements.

## Results

| condition | root | val (standalone) | resNorm | cos | d6η² | d5η² | SK b7 fresh (Δ vs OL) | ct b7 | ens_cos |
|---|---|---|---|---|---|---|---|---|---|
| `ntp_aux` (OL) | 0.796 | 1.545 | 24.1 | 0.836 | 0.295 | 0.381 | 0.557 | — | 0.984 |
| `ntp_aux_cl@1.0` | 0.802 | 2.383 | 8.9 | 0.882 | 0.322 | 0.421 | 0.779 (+0.22) | 0.907 | 0.983 |
| `ntp_aux_cl@0.0` | 0.800 | 2.577 | 15.1 | 0.877 | 0.313 | 0.396 | 0.746 (+0.19) | 0.593 | 0.984 |
| `@0.0+distill` (α=0.5) | 0.793 | **1.540** | 23.2 | 0.858 | 0.317 | 0.403 | 0.522 (−0.04) | 0.609 | 0.984 |
| `@1.0+distill` (α=0.5) | 0.802 | **1.535** | 4.9 | 0.942 | 0.346 | 0.445 | 0.726 (+0.17) | 0.800 | **0.992** |
| `@0.0+distill` (α=0, control) | 0.795 | **1.537** | 23.6 | 0.851 | 0.316 | 0.404 | 0.530 (−0.03) | 0.599 | 0.984 |

With-injection probes (wake-final; standalone values in parens):

| condition | val_inj | d6 inj | d5 inj | d4 inj | val_inj post-sleep |
|---|---|---|---|---|---|
| `ntp_aux_cl@1.0` | 1.544 (2.383) | 0.153 (0.802) | 0.484 (0.955) | 0.902 (0.979) | 1.543 |
| `ntp_aux_cl@0.0` | 1.546 (2.577) | 0.183 (0.800) | 0.591 (0.959) | 0.913 (0.978) | 1.569 |

## Findings (stated conservatively)

1. **The sleep phase fully closes the dependency gap — and the teacher's KL contributes nothing to that.** Both α=0.5 arms recover standalone val to marginally better than OL (1.535–1.540 vs 1.545), root stays at BP, and the residual returns to the OL-scale frontier map (`@0.0+distill` resNorm 23.2 ≈ OL's 24.1). But the α=0 control recovers **identically** (1.537) with no teacher signal: the gap closes within the first ~1K sleep steps (2.39 → 1.55), and the control's *unoptimized* KL falls to 0.026 on its own. So the standalone degradation was a shallow contextual adaptation to the injection's presence, and removing the injection lets dense CE+aux re-derive the function from data — nothing is transferred from the teacher. On this domain "distillation" reduces to "injection-off fine-tuning."
   - *Consequence for the ratchet*: the sleep phase can be plain CE+aux fine-tuning — no teacher forward pass needed.
   - *Retroactive qualification*: the a2a language distillation result (DISTILLATION_README, "105.6% of the gap closed") is attribution-ambiguous in the same way — it may also have been re-derivation, since dense NTP was in the distillation loss there too.
   - *Hypothesis, untested*: the KL should become load-bearing exactly when supervision is sparse relative to what the teacher knows (MNIST's 1 label/image; masked NTP) — i.e. distillation's value ≈ teacher knowledge − what the data directly supervises. This is the candidate account of why distillation was load-bearing on MNIST but is inert here. Component test queued (sleep-CE-mask, below).

2. **Sleep erases the loop's fresh-FM SK gain unless local loss ran during wake.** `@0.0`'s +0.19 washes out to −0.04 after sleep (α=0 control confirms the erasure is from the fine-tuning itself, not the KL), while `@1.0` keeps +0.17 of its +0.22 (fresh 0.726, ct 0.800). The thing being known persists through sleep (post-sleep residual η² essentially unchanged, ens_cos 0.984) — what washes out is M's *encoding* of it, and the wake-time local loss is what makes that encoding durable. Stated limits: the SK metric is a single aggregate R² (it cannot separate meta- from object-level components, per the MNIST local-loss-probes distinction), and `@1.0+distill`'s much smaller, highly FM-invariant residual (norm 4.9, ens_cos 0.992) plausibly inflates its direction-predictability. **Do not lean on this finding until the decomposition probes are ported** (component test below).

3. **`@1.0+distill` resolves the standalone-val caveat from Exp 1 — the best cell on every axis at once.** Best val (1.535), root at BP (0.802), positive fresh SK (+0.17), the deepest residual η² of any condition (d6 0.346, d5 0.445), the highest FM-invariance (ens_cos 0.992), and the injection is fully redundant post-sleep (val_inj 1.543 ≈ standalone 1.535 — internalization complete). Wake with injection + local loss, sleep with plain CE+aux: this is the natural cycle-1 recipe for the compounding experiment.

4. **The injection displaces deep-latent structure at the probe position rather than adding it.** Injected forward passes at wake-final have far *worse* deep recovery than standalone ones (d6: 0.18 vs 0.80; d5: 0.59 vs 0.96) while val improves (1.55 vs 2.58). Given the training structure (NTP passes injected, aux passes never injected), the deep representation is *mode-specific*: it is built by and lives in the non-injected aux pass. Note the strict scope: because the aux conditions are already at the BP root ceiling (0.80), this cell has **no headroom** and is structurally incapable of detecting positive "exploration beyond the frontier" — it only shows the injection is not depth-neutral where depth already exists. The fair exploration test is token-CL (standalone root 0.07, large headroom); queued below.

## Caveats

- **Single rule seed, single run per cell** (same status as Exp 1–4).
- **+5K-step asymmetry** for the distill arms. Addressed for the val claim by the α=0 control (same step count, no teacher); the OL reference is plateaued by 20K (last 5K of wake bought ~0.01 nats).
- **Single cycle.** The MNIST claim this ablation was aimed at ("distillation is load-bearing") was about *multi-cycle compounding*; this establishes only that the teacher-KL is not needed for single-cycle consolidation here. Whether CE+aux-only sleep sustains compounding across cycles is the ratchet question, untested.
- **SK-survival finding is not yet interpretable** (see finding 2's stated limits).

## Component tests (follow-ups from this ablation)

1. **Injection component — fair exploration test**: token-CL at m4 (`ntp_cl@0.0`, `ntp_cl@1.0`) with the injected probes, where standalone root is 0.07 and there is real headroom. Does injection lift deep recovery above the standalone frontier during wake, or only relocate NTP?
2. **Distillation component — sparse-supervision KL test**: rerun `@0.0+distill` (α=0.5 vs α=0) with the sleep-phase CE masked to 5% of positions while the KL stays dense (`--sleep-ce-mask-rate 0.95`). Wake identical (dense, deterministic). If finding 1's hypothesis is right, α=0.5 should recover standalone val where the α=0 control now cannot.
3. **Local-loss component — SK decomposition**: port the meta-vs-object probe machinery (`a2a_forward/mnist_local_loss_probes.py`: prediction probe vs orthogonalized-residual probe) to interpret finding 2, additionally conditioned on hierarchy level (which MNIST could not offer). Not yet implemented.

## Reproduction

```bash
cd experiments
# main ablation (5 conditions + ensemble):
modal run --detach -m rhm.rhm_latent_loop::latent_loop \
    --conditions "ntp_aux,ntp_aux_cl@1.0,ntp_aux_cl@0.0,ntp_aux_cl@0.0+distill,ntp_aux_cl@1.0+distill" \
    --ensemble-n 4 --tag distill
# no-teacher control (identical sleep, KL removed):
modal run --detach -m rhm.rhm_latent_loop::latent_loop \
    --conditions "ntp_aux_cl@0.0+distill" --distill-alpha 0.0 --ensemble-n 4 --tag distill-ce0
```

Results: `/rhm_latent_loop/v16_s2_L6_m4_distinct_8L8H256D_S20000_distill{,-ce0}.json` on the volume.

---

# Meta/object decomposition: RHM self-knowledge is injection-generated meta-knowledge; object-level is not acquired (2026-07-07)

**Code**: `rhm_latent_loop.py` — new `_meta_object_probes` (+ `_mo_probe`, `_position_levels_np`), wired into the harness (`--meta-object`, on by default) and printed as new summary tables. Ported verbatim from `a2a_forward/mnist_local_loss_probes.py` (family-A four-target probe), extended with a fresh-vs-co-trained axis and per-hierarchy-level conditioning.
**Motivation**: every RHM self-knowledge number to date — the +0.22 latent gain, the −1.19 token inversion, the distill-ablation retention — was read off a **single composite** probe (`R²` of the raw FM residual `target − pred` direction), which **conflates two knowledge types** the a2a arc separates: *object-level* ("what the FM predicts", the `prediction` direction) and *meta* ("where the FM errs, ⊥ its prediction", the `ortho_residual` direction). On MNIST/language, closing the loop builds **meta** (3:1 over object at early layers, `MNIST_LOCAL_LOSS`) and only **distillation** converts meta→orthogonal object-level (`GEOMETRY`, `FORWARD_MODEL_SWAP`). This section ports that decomposition to disambiguate which type each RHM intervention builds.

## The instrument

Four probe targets, native (no-injection) activations, per-token orthogonalization off the FM prediction vector, **linear** probe, **no** input standardization, global-variance `R²` (exact `mnist_local_loss_probes.py` recipe):

```
prediction     = FM(post_block0)                      -> OBJECT-level
residual       = post_block6 − prediction             -> composite (the old SK number)
ortho_residual = residual − (residual·p̂)p̂            -> pure META
target         = post_block6                           -> ceiling
```

Measured at **post_block0 (pre-injection), post_block1 (injection point), post_block7 (deep)**, for **both the fresh matched FM** (apples-to-apples, FM-general) **and the co-trained FM** (the one M learned to complement, FM-specific), and additionally **per hierarchy level** (v_s valuation → d-notation). Target variances (`pred_var`/`res_var`/`ortho_var`/`ortho_over_res`) are reported so a near-zero ortho `R²` in a tiny-residual condition reads as "no residual" (the LL failure mode) rather than "no meta". **Reproduction check**: every prior cell reproduced exactly (val 1.554/1.545/1.746/2.383, composite SK 0.516/0.557/−0.670/0.779, Δ SK token −1.19 / latent +0.22), so the decomposition sits on the validated harness.

## Finding 1 — self-knowledge on RHM *is* meta-knowledge; object-level is a flat, un-acquired baseline

Object-level (`prediction R²`) is **high and nearly condition-invariant** (~0.5–0.95, rising trivially with depth — the "shared reference frame byproduct" it was on MNIST). The entire composite Δ SK lives in the **meta** channel at the deep layer (m4 2×2, fresh FM):

| Δ(CL−OL), fresh | ΔOBJ@b0 | ΔMETA@b0 | ΔOBJ@b7 | **ΔMETA@b7** |
|---|---|---|---|---|
| **token** (ntp_cl−ntp) | +0.464 | −0.037 | −0.120 | **−1.865** |
| **latent** (ntp_aux_cl−ntp_aux) | +0.182 | −0.045 | +0.083 | **+0.284** |

The −1.19 token inversion **is** a deep-meta collapse (−1.865); the +0.22 latent gain **is** deep meta (+0.284). Object barely moves in either. This is the RHM analog of MNIST's 3:1 meta-dominance — here the meta channel is essentially the *whole* story, not just the majority.

## Finding 2 — the latent target converts FM-*specific* meta into FM-*general* meta

The fresh-vs-co-trained split is the mechanism (m4, META@b7):

| META (ortho) @b7 | **fresh FM** | **co-trained FM** |
|---|---|---|
| ntp_cl (token CL) | **−1.389** | −0.137 (but **+0.487 @b0**) |
| ntp_aux_cl (latent CL) | **+0.741** | +0.868 |

- **Token-CL builds FM-specific meta** (co-trained +0.487 @b0) that **does not generalize** — a fresh FM's error is *anti*-predicted (−1.389). The residual there is tiny FM-idiosyncratic noise (norm 1.22, cos 0.983, ens_cos 0.821). Worst of both worlds, now localized to the fresh-meta channel.
- **Latent-CL builds FM-general meta** — fresh (+0.741) and co-trained (+0.868) both strongly positive, ens_cos 0.983. The latent target makes the residual a DGP-aligned inference gap every FM misses identically, **and** M encodes it.

## Finding 3 — m2 control: token inversion is a meta collapse *even with a deep open-loop residual*; latent gives no meta gain without a frontier gap

m2 is the regime where token-NTP already reaches the root (`ntp` root 0.819, deep OL residual d5η²=0.347). Decomposed (fresh Δ):

| Δ(CL−OL), fresh | ΔOBJ@b0 | ΔMETA@b0 | ΔOBJ@b7 | **ΔMETA@b7** |
|---|---|---|---|---|
| token (ntp_cl) | +0.304 | −0.103 | −0.112 | **−2.427** |
| latent (ntp_aux_cl) | +0.160 | +0.032 | +0.075 | **−0.040** |

- Token inversion = deep fresh-**meta** collapse (−2.427), same as m4 — **a deep OL residual is not sufficient** (the doc's Exp 2 falsification, now at the decomposition level). Co-trained meta stays positive (+0.459 @b0): FM-specific meta is still built, it just doesn't generalize.
- **The latent target gives no meta gain at m2** (ΔMETA@b7 −0.040) — because m2 has no frontier gap (`ntp_aux ≈ ntp`, both at root), so closing the loop can't newly represent deep structure. **Generalizable meta requires both (i) a DGP-aligned residual and (ii) a genuine frontier gap the loop helps close.** m4 has both; m2 has only (i). (The m2 co-trained meta is nonetheless huge — ct META@b7 0.927 — pure FM-specific consolidation.)

## Finding 4 — λ_local sweep: injection *generates* meta, local-loss consolidation *destroys the generalizable part*

m2 token, sweeping λ_local (λ=0 = injection-only). This isolates injection from local loss:

| λ_local | val | resNorm | **fresh META@b7** | **ct META@b7** | fresh OBJ@b7 |
|---|---|---|---|---|---|
| ntp (OL) | 0.826 | 24.6 | +0.340 | — | 0.861 |
| **@0.0 (inj only)** | 1.855 | 13.4 | **+0.284** | **+0.706** | 0.873 |
| @0.1 | 1.777 | 2.0 | −0.365 | +0.644 | 0.881 |
| @0.3 | 1.300 | 1.1 | −0.838 | +0.409 | 0.847 |
| @1.0 | 1.047 | 0.8 | −2.087 | −0.356 | 0.749 |

- **Injection alone (λ=0)** roughly preserves the shared-basis fresh meta (0.284 ≈ OL 0.340) **and** builds strong FM-specific meta (co-trained +0.706): **injection is the meta generator**, confirmed with local loss off.
- As **local loss rises**, standalone val improves (1.855→1.047) and the residual compresses (13.4→0.8) — genuine consolidation — but fresh (generalizable) meta is **monotonically destroyed** (+0.284 → −2.087). Object stays high/flat throughout (no object acquisition without distillation). So on a token target, local-loss consolidation trades generalizable meta for standalone competence — the consolidation-vs-generalizable-SK tension, localized to the fresh-meta channel and shown to be λ-driven. (On a *latent* target the same consolidation preserves meta, because the residual it compresses toward is DGP-aligned.)

## Finding 5 — sleep transfers nothing on RHM: the α=0 control kills the KL for representation too

The distillation ablation showed a meta→object *shift* through sleep (co-trained META@b7 down, OBJ@b7 up). The α=0 control (identical sleep, teacher KL removed) reproduces it **identically** — the KL is inert for representation, not just val:

| fresh FM, m4 | @0.0+distill α0.5 | @0.0+distill **α0** | @1.0+distill α0.5 | @1.0+distill **α0** |
|---|---|---|---|---|
| val | 1.540 | 1.536 | 1.535 | 1.531 |
| META@b7 | 0.403 | 0.411 | 0.455 | 0.439 |
| ΔOBJ@b0 (vs OL) | +0.005 | +0.008 | +0.229 | +0.229 |
| ct META@b7 | 0.585 | 0.568 | 0.662 | 0.646 |
| ct OBJ@b7 | 0.796 | 0.787 | 0.725 | 0.715 |
| Δ SK@b7 | −0.035 | −0.028 | +0.169 | +0.162 |

- **The KL relocates nothing.** Mechanistically expected: the FM predicts M's *own* `post_block6`, so there is no external FM knowledge to transfer — removing the injection crutch and re-consolidating on dense CE+aux is all that happens. Distillation on RHM = injection-off fine-tuning, for representation as for val.
- **Refines the earlier "object appears after distillation" reading**: it appears after the **sleep**, KL-irrelevant, and is *not* a meta→object transmutation but **differential durability** — the injection-generated deep meta **washes in both arms** (ΔMETA@b7 → ≈0 once injection is gone), while a **shallow object-accessibility** (ΔOBJ@b0 **+0.229**) present only in the `@1.0` (local-loss-during-wake) arm **survives** consolidation. Consistent with LL→object on MNIST: the durable object signal is a local-loss product, not a distillation product.

## Finding 6 — seed replicate: signs hold, latent magnitude is stable, token-inversion magnitude is not

Second rule seed (m4 2×2), fresh Δ(CL−OL) @b7:

| | seed-0 | **seed-1** |
|---|---|---|
| **latent** ΔMETA | +0.284 | **+0.329** |
| latent ΔSK (composite) | +0.222 | **+0.224** |
| **token** ΔMETA | −1.865 | **−0.836** |
| token ΔSK (composite) | −1.186 | −0.570 |

The headline is robust: latent → positive generalizable meta (+0.22 composite is near-identical across seeds), token → meta collapse (sign robust). Fresh-vs-co-trained reproduces (token co-trained META@b0 +0.234 positive / fresh negative; latent both positive). **Caveat exposed by the seed**: the token-inversion *magnitude* is unstable (−0.836 vs −1.865) because it is a *negative* `R²` fitting the tiny FM-noise residual — the *direction* (meta collapse) is the robust claim, not the number. The positive latent gain fits real DGP-aligned structure and is stable.

## Synthesis — the causal chain, decomposed and seed-checked

- **Injection generates meta-knowledge** (the "where my self-model errs" map). FM-specific by default; FM-general only under a DGP-aligned/latent target **and** a real frontier gap (m4, not m2).
- **Local loss = consolidation.** It internalizes the injected computation into standalone weights (val↓, residual↓, dependency↓) but **erodes generalizable meta on token targets** (monotone with λ) and **preserves it on latent targets**; it independently seeds a durable **shallow object-accessibility**. It does *not* generate self-knowledge — injection does.
- **Sleep/distillation transfers nothing on RHM** — pure re-derivation/reorganization (KL inert, α=0 ≈ α=0.5). Meta evaporates on injection removal; the local-loss shallow object survives.
- **Object-level as an *acquired* knowledge type still has not appeared on RHM** the way MNIST/language get it from distillation — RHM's dense DGP makes the KL redundant. The MNIST prediction (KL load-bearing under *sparse* supervision) remains the open test (`--sleep-ce-mask-rate`, component test #2).

Self-knowledge = meta = injection-generated; object-level = high-but-flat baseline, not built by any wake-only intervention, and not transferred by the (redundant) RHM sleep.

## Caveats

- **The MO composite `R²` slightly under-reads the `_sk_probes` composite** for tiny-residual conditions (300 vs 500 probe steps; MO drops the last, next-token-less position). Decomposition comparisons are internally consistent (all MO probes share settings); cross-comparison to the SK-table composite is not exact.
- **Negative-meta magnitudes are noise-fits** (negative `R²` on a ~0-variance residual): interpret their *sign*, not their value (Finding 6).
- **Two rule seeds** on the m4 2×2; m2 / λ-sweep / distillation are single-seed. Signs directionally consistent; magnitudes not hardened beyond the two-seed m4 headline.
- **Latent target uses privileged oracle latents** (diagnostic, per Exp 1); the self-supervised (EMA) version is still untested.

## Reproduction

```bash
cd experiments
# canonical m4 2x2 + ensemble (decomposition on by default):
modal run --detach -m rhm.rhm_latent_loop::latent_loop \
    --conditions "ntp,ntp_aux,ntp_cl,ntp_aux_cl" --ensemble-n 4 --tag mo
# m2 control:
modal run --detach -m rhm.rhm_latent_loop::latent_loop --m 2 \
    --conditions "ntp,ntp_aux,ntp_cl,ntp_aux_cl" --ensemble-n 4 --tag mo_m2 \
    --bp-line "d1 1.00 d2 1.00 d3 1.00 d4 1.00 d5 .992 root .954" --greedy-line "(m2: no frontier gap)"
# m2 lambda_local sweep (injection-vs-local-loss isolation):
modal run --detach -m rhm.rhm_latent_loop::latent_loop --m 2 \
    --conditions "ntp,ntp_cl@0.0,ntp_cl@0.1,ntp_cl@0.3,ntp_cl@1.0,ntp_aux_cl@1.0" \
    --ensemble-n 4 --tag mo_lamsweep \
    --bp-line "d1 1.00 d2 1.00 d3 1.00 d4 1.00 d5 .992 root .954" --greedy-line "(m2: no frontier gap)"
# distillation + alpha=0 control (KL-inert-for-representation):
modal run --detach -m rhm.rhm_latent_loop::latent_loop \
    --conditions "ntp_aux,ntp_aux_cl@1.0,ntp_aux_cl@0.0,ntp_aux_cl@0.0+distill,ntp_aux_cl@1.0+distill" \
    --ensemble-n 4 --tag mo_distill
modal run --detach -m rhm.rhm_latent_loop::latent_loop \
    --conditions "ntp_aux,ntp_aux_cl@0.0+distill,ntp_aux_cl@1.0+distill" \
    --distill-alpha 0.0 --ensemble-n 4 --tag mo_distill_a0
# second rule seed of the m4 2x2:
modal run --detach -m rhm.rhm_latent_loop::latent_loop \
    --conditions "ntp,ntp_aux,ntp_cl,ntp_aux_cl" --ensemble-n 4 --rule-seed 1 --tag mo_seed1
```

Results: `/rhm_latent_loop/v16_s2_L6_{m4,m2}_distinct_8L8H256D_S20000_mo{,_m2,_lamsweep,_distill,_distill_a0,_seed1}.json` on the volume.

---

# Sparse-supervision sleep: on RHM there is nothing for distillation to transfer (2026-07-08)

**Code**: `rhm_latent_loop.py` — new `--sleep-ce-mask-rate`: during the sleep phase a **fixed** random subset of *absolute corpus positions* is CE-supervised, the rest supervised only by the (dense) teacher KL. Masking by absolute index (not window position) makes it true coverage-sparsity — the un-kept positions **never** receive a hard label.
**Motivation**: the α=0 control (finding 5, prev section) showed the sleep phase transfers nothing on RHM — val/representation re-derive without the teacher KL. Open question: is that because RHM genuinely has **no transferable surplus**, or because *dense* supervision merely *masks* a real transfer? Component test #2: starve the data channel and see if the KL becomes load-bearing. **Framing caveat (from discussion): "KL load-bearing" is only a proxy.** The actual object of interest is whether the FM's contribution gets re-localized into M's weights (the MNIST/language internalization result); the sparse test is an instrument for that, not the goal.

## Setup

- **Regime chosen for a competent teacher + single data channel**: m2 **token** `ntp_cl@1.0` (no aux, so CE is the *only* data channel). Its teacher-with-injection is genuinely competent — injected val **0.824**, root **0.871** — vs the offloaded standalone wake model (val 1.047, root 0.448). So there is real injected competence available to transfer.
- **2×2**: α ∈ {0.0, 0.5} × `sleep_ce_mask_rate` ∈ {0.0 (dense), 0.95 (5% of corpus positions CE-supervised)}. KL stays dense in all cells. Four separate detached runs (α and mask are global; wakes are seed-identical). Readout = **post-sleep standalone val + root(d6) recovery**.
- **Prediction**: dense → α=0 ≈ α=0.5 (KL redundant, reproducing finding 5); sparse → α=0.5 ≫ α=0 (KL fills the 95% the labels never cover, transferring the teacher's root competence).

## Result — the prediction is falsified: every cell recovers, KL adds ~nothing

Post-sleep standalone `ntp_cl@1.0+distill` (ref: teacher-injected val 0.824 / root 0.871; OL `ntp` val 0.826 / root 0.819):

| | mask=0.0 (dense CE) | mask=0.95 (sparse CE) |
|---|---|---|
| **α=0 (no KL)** | val 0.820, root 0.887 | val 0.831, root **0.855** |
| **α=0.5 (KL)** | val 0.820, root 0.894 | val 0.825, root **0.866** |

Sparse-CE-**no-KL** (`a0m95`) recovers to val 0.831 / root 0.855 — essentially matching dense. **95% structural masking did not starve the data channel.** KL's marginal contribution is tiny everywhere (≤ +0.011 root, ≤ +0.006 val). The predicted interaction (KL load-bearing under sparsity) does not appear.

## Why — recoverability, not density, is the load-bearing variable

I withheld 95% of *positions*, but RHM's DGP is a **compact, shared rule set** (m2/v16 ≈ 96 rules) reused at every position. Supervising even 5% of positions gives many examples of *every* rule; M infers the grammar and **generalizes to the 95% it was never labeled on**. The withheld supervision was *recoverable from the retained supervision* via the DGP's regularity, so the teacher had nothing to add. The sharpened principle:

> **Distillation is load-bearing only when the withheld supervision carries information the student cannot reconstruct from the retained supervision.**

MNIST clears this bar — each image's soft label is **per-instance** (this specific 7's ambiguity is not derivable from other images). RHM fails it — its labels are **samples from a shared grammar**, so sparse labels reconstruct the whole thing. RHM's distillation-redundancy is therefore *stronger* than finding 5 established: not merely "the data is dense" but "the DGP is **sample-efficient**, so the data channel can't easily be starved."

## Ambivalent takeaways (from the accompanying discussion)

- **The transfer surplus on RHM is a rounding error — by design.** The FM's entire content is a compression of the compact DGP, which M re-derives from data directly. There is no FM-knowledge M *can't* get elsewhere, so distillation-as-transfer is structurally vacuous here. On language/MNIST, where M's computation encodes something vast and expensive-to-acquire, the transfer question is substantive; on RHM it is not. This is a *feature*: RHM is the clean control proving a regime with **nothing to transfer**.
- **Three senses of "self-knowledge," only one of which is vacuous here.** (1) *Computational meta-knowledge* (frontier map) — **found** on RHM (latent target, +0.28), not vacuous. (2) *Object-level transfer* (FM function internalized) — vacuous on RHM (this result). (3) *Epistemic "I know what I know"* — not produced by the loop even on language (a2a OOD calibration-transfer). So "self-knowledge is the wrong question on RHM" is true only for sense (2).
- **The α=0 control already implied this.** Because every post-sleep number is α-independent, whatever standalone competence M gains through sleep is **re-derived, not absorbed from the FM**. "How much did M absorb *from the FM specifically*" on RHM ≈ nothing.
- **We have not cleanly *measured* object-level absorption on RHM.** The prediction-probe object R² is **baseline-saturated** (~0.86–0.92 even in OL — activations trivially encode a function of themselves), so it does not isolate FM-absorption. The clean instrument — the MNIST cross-model internalization probe (does the *old/co-trained* FM predict the *distilled* model better than it predicts *OL*) — was **not ported** to RHM. Given the α=0 result, porting it to RHM is low-value; the informative place to run it is MNIST/language.
- **The one directionally-correct signal**: under sparsity, KL *did* pull root toward the teacher's exact injected value (0.855 → 0.866, ceiling 0.871). Real transfer, but into a nearly-solved problem — tiny headroom, single-seed, not hardened. Distillation isn't broken on RHM; it's solving an already-solved problem.
- **Recoverability ladder** (the cross-domain synthesis): MNIST (per-instance labels, non-recoverable → genuine transfer) › language (dense NTP in the distill loss → *attribution-ambiguous*; the DISTILLATION_README "105.6% closed" may also be re-derivation) › RHM (shared grammar, fully recoverable → provably no transfer). The sharp follow-up is **not** more RHM but an α=0 control + cross-model internalization probe on **MNIST/language**, to settle whether the internalization we believe in there is genuine transfer or re-derivation.

## Caveats

- **Single seed per cell**; the ~0.01 KL→root nudge is within plausible seed noise.
- **The masking is coverage-sparsity, and RHM defeats it by construction** — this is the finding, but it also means the test never actually reached a starved regime, so it is a null-by-recoverability, not a null-under-genuine-starvation. Level-structured withholding (never CE-supervising the root) could force a starved regime, but that engineers the proxy positive without bearing on real transfer — deliberately not pursued.
- **Token-CL post-sleep models** here carry the expected collapsed generalizable meta (fresh META@b7 −0.38 to −0.79) and baseline-high object — consistent with the prior sections, not the focus.

## Reproduction

```bash
cd experiments
# 2x2: alpha in {0.0,0.5} x sleep_ce_mask_rate in {0.0,0.95}, m2 token ntp_cl@1.0
modal run --detach -m rhm.rhm_latent_loop::latent_loop --m 2 \
    --conditions "ntp,ntp_cl@1.0+distill" --distill-alpha 0.5 --sleep-ce-mask-rate 0.95 \
    --ensemble-n 0 --tag sparsekl_a5m95 \
    --bp-line "d1 1.00 d2 1.00 d3 1.00 d4 1.00 d5 .992 root .954" --greedy-line "(m2: no frontier gap)"
# ...repeat with (--distill-alpha 0.0 --sleep-ce-mask-rate 0.95 --tag sparsekl_a0m95),
#              (--distill-alpha 0.0 --sleep-ce-mask-rate 0.0  --tag sparsekl_a0m0),
#              (--distill-alpha 0.5 --sleep-ce-mask-rate 0.0  --tag sparsekl_a5m0)
```

Results: `/rhm_latent_loop/v16_s2_L6_m2_distinct_8L8H256D_S20000_sparsekl_a{0,5}m{0,95}.json` on the volume.
