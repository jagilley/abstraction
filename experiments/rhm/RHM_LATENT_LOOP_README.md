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
4. **The closed loop decomposes**: injection = offload/preview (wake-like division of labor), local loss = consolidation/internalization (sleep-like), with a real tension between consolidation and fresh-FM-measured self-knowledge.

## Connections to our beliefs (discussion, not yet crystallized)

- **[dimensionality_expansion.md](../../beliefs/dimensionality_expansion.md)** frames the residual as the frontier (`R_res` = the slice of `R_act` the self-model hasn't absorbed). "Residual structure tracks the frontier" is the *per-level* refinement of that: the residual is not just *how much* is un-absorbed but *where in the hierarchy*. It also sharpens the belief's "novelty raises `R_act`" — on RHM, i.i.d. sample-novelty is saturated yet the frontier is fixed; only **abstraction-novelty** (a deeper target) raises the achievable `R_act`. Sample-novelty refills breadth; a latent target unlocks depth.
- **Computational, not epistemic, self-knowledge** (a2a OOD_ROBUSTNESS): the FM-invariant latent residual is precisely the "computational / distribution-invariant" self-knowledge the a2a paper argued for. RHM shows the *precondition* for it — a target that makes the residual DGP-aligned.
- **FM-specific vs FM-general SK** (a2a FORWARD_MODEL_SWAP): consistent — token-CL is FM-specific (co-trained ≫ fresh; the post-injection layer carries the FM-specific signature); the latent target is what pulls it into FM-general territory.
- **FM-as-functional-regularizer** (RHM_FM_REGULARIZER) established self-knowledge is *not* needed for functional simplification. This is the mirror: functional simplification is not sufficient for self-knowledge either — self-knowledge needs a target that keeps a substantive DGP-aligned residual, which pure compression removes.

We are **not** crystallizing a belief yet — single-seed (below), and the compounding question (next) is what would elevate this from "self-knowledge appears" to "self-knowledge does work."

## Caveats

- **Single rule seed, single run per cell.** Directionally consistent across four runs and two regimes, and the harness is deterministic (the sweep's `ntp_cl@1.0` reproduced the m2 control's root/SK exactly), but **magnitudes are not hardened** — a 2–3 seed replicate is the precondition for any strong claim. Note the m4 SK magnitudes shift ~0.1 between an A10G and an L4 run (sign/direction stable), underscoring single-seed magnitude fragility.
- **Latent target uses privileged latent values** (`oracle_aux`, Level-0). This is a diagnostic of *whether* self-knowledge can appear given a climbable target — **not** a self-supervised mechanism. The self-supervised version (EMA/data2vec-style own-lifted-latent target) is untested here.
- **`ntp_aux_cl` has degraded standalone val (2.38 at m4)** — the aux+injection combination drives strong dependence; SK is real but read off a somewhat NTP-degenerate standalone model. A λ_local / FM-capacity sweet-spot sweep to get SK *without* the val hit is wanted.
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
2. **Compounding — the "click" (the real next chapter).** `ntp_aux_cl` is the first genuine positive-self-knowledge setup on RHM. This finally makes the abstraction-ratchet question askable: **does this self-knowledge compound across wake–sleep cycles** (injection → distill/consolidate → re-point FM → repeat), or does it plateau like every prior RHM ratchet? Hypothetical, untested — but for the first time the precondition (self-knowledge exists) is met.
3. **Drop the privilege.** Replace the oracle-latent aux with an EMA self-distilled own-lifted-latent target (data2vec-style, per the sleep-chunking line) and test whether generalizable self-knowledge survives without ground-truth labels — the self-supervised version of Experiment 1.
4. **SK sweet spot.** λ_local / FM-capacity sweep on `ntp_aux_cl` to obtain positive Δ SK without the standalone-val degradation (the consolidation-vs-SK tension from Experiment 3).
