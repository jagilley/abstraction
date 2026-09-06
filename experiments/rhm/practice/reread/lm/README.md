# reread/lm — the LM twin: re-reading a frozen corpus with an endogenous reader

**Up**: [`../README.md`](../README.md) (reread — the sculpting round this node was built to
re-scope) · **Files**: [`FILES.md`](FILES.md) (machinery, gates, calibration record) ·
**Machinery donor**: [`../../../conditional_revision/`](../../../conditional_revision/README.md)
(imported, unmodified) · **Idea**: the reread claim in
[`meta_learning_under_metered_data`](../../../../../ideas/meta_learning_under_metered_data.md)'s
2026-08-17 caveat block — data value is indexed by the reader's current vocabulary, so a
full-loop learner has effectively unlimited training data over a fixed archive.
**Run**: `lm0`, 5 arms × 81.9M tokens, 2026-08-17.

## The question

[`rr_s0`](../README.md) found a frozen archive renewable in *competence* but not in *mining
yield*, and diagnosed the miss as structural: on the sculpting substrate the reader is exogenous
and pinned at 1.000, and mining reads the agent's own canonicalised repairs — the archive is
rewritten by the act of reading it, so vocabulary-gated *perception* is inexpressible there. The
book-reread claim's mechanism (you cannot parse level-ℓ until you own level-(ℓ−1)) lives in the
reader. This node moves to NTP on RHM, where the model's representation is the only reader it
has: **does a fixed corpus keep yielding deeper structure across passes, as the reader climbs —
and at what token penalty relative to endless fresh data?**

## Design in one paragraph

Five arms differ **only** in whether the corpus tensor is redrawn: `fresh` (81.9M distinct
tokens, each consumed ~once) against frozen corpora of 12.8M / 1.05M / 131k / 32.8k tokens —
**6.4× / 78× / 625× / 2500× re-read** at a matched 81.9M-token budget. Extraction is read two
independent ways: the donor's per-level probes against **exact BP ceilings** (measured at the
gate: d1 0.997 … d6 0.798 — every level has real dynamic range), and a probe-free
`excess_over_bayes` (model NLL minus the exact Bayes surprisal, per arrival level), so a probe
failure and a representation failure cannot be confused. The fidelity gate is free:
`frozen_200000` @ step 12000 *is* the donor's Gate-0 base, and it reproduced the published
numbers essentially exactly (d1 0.978 vs 0.979, d3 0.834 vs 0.836, d6 0.085 vs 0.088). Full
gates, protocol notes and flags: [`FILES.md`](FILES.md).

## Findings

**1. Extraction is strictly ordered by level, in every arm.** Tokens-to-reach-50%-of-ceiling is
monotone d1 < d2 < d3 < d4 in all five arms — no arm ever reached level ℓ+1 before level ℓ, on
frozen and fresh data alike. The vocabulary-gating mechanism, read directly.

**2. At 6.4× re-read the frozen corpus is fully renewable — deep extraction at zero token
penalty.** Between 22M and 82M tokens consumed, `fresh`'s val NLL moves only −4.8% and d1/d2 are
saturated, while d3 goes 0.599 → 0.910 and d4 goes 0.218 → 0.648 — deep extraction runs long
after loss flattens. `frozen_200000` does the same on 12.8M frozen tokens (d3 0.623 → 0.888, d4
0.195 → 0.549), with tokens-to-half-ceiling ratios vs fresh of **1.00 / 1.01 / 0.91 / 1.14**
(d1–d4) — nothing outside noise — and it is still extracting at 67–72% of fresh's rate when the
budget ends. The probe-free instrument independently agrees: excess NLL indistinguishable from
`fresh` at every arrival level (Δ ≤ 0.010 nats).

| arm (re-read ×) | d1 | d2 | d3 | d4 | tokens-to-50% ratio d1/d2/d3/d4 |
|---|---|---|---|---|---|
| `fresh` (1×) | 0.983 | 0.974 | **0.910** | **0.648** | — |
| `frozen_200000` (6.4×) | 0.974 | 0.965 | **0.888** | **0.549** | 1.00 / 1.01 / 0.91 / 1.14 |
| `frozen_16384` (78×) | 0.961 | 0.897 | 0.475 | 0.164 | 1.00 / 1.01 / ∞ / ∞ |
| `frozen_2048` (625×) | 0.952 | 0.859 | 0.317 | 0.160 | 1.00 / 1.05 / ∞ / ∞ |
| `frozen_512` (2500×) | 0.944 | 0.783 | 0.274 | 0.160 | 1.00 / 10.9 / ∞ / ∞ |

**3. Renewability has a hard corpus-size wall — a cap, not a tax.** Below a threshold of
distinct tokens (bracketed between 1.05M and 12.8M), re-reading does not slow deep extraction;
it **caps** it: the sub-threshold arms stall at the next level they were climbing and never move
again (their d4 sits at its initialization value at every checkpoint). Achieved depth is roughly
logarithmic in corpus size — **each ~10× of distinct corpus buys about half a level, and passes
buy nothing past the cap.** This is "not all data is created equal" with a coordinate: a corpus
has a depth capacity, and re-reading is free extraction up to it and worthless beyond it.

**4. Overfitting does not destroy deep representation — the wall means it was never built.**
The small-corpus arms collapse onto their corpora (memorisation gap up to +7.4 nats; `frozen_512`
val NLL 7.53 against a Bayes floor of 1.38) while their residual streams *keep acquiring* shallow
and mid structure from held-out data (protocol-clean late d2 +0.055, d3 +0.019). Their deep
levels move exactly 0.000 — never having left initialization.

**5. An instrument dissociation worth keeping** (recorded as suggestive): `frozen_16384`'s
root-level excess NLL improved to 0.259 at 22M tokens and then degraded to 0.976 (3.8× worse)
while its d3 probe stayed flat over the same span. The model lost deep *predictive* competence
it had already acquired while the probe-readable *representation* was retained — the output head
over-committing to the memorised corpus while the residual stream keeps the structure.

## Scope and fragilities

- **d5/d6 are untested, not null**: they never move in any arm (d6 0.077 → 0.087 against a
  ceiling of 0.798) — no dynamic range at this model size and budget. Everything here is a
  d1–d4 statement.
- **The run ends mid-transition at d4** (`fresh` +0.260 and `frozen_200000` +0.174 per late
  interval, both still climbing). Whether the 6.4× arm converges to fresh at d4 or has its own
  wall just past the budget is untested.
- **The wall is bracketed, not located** (one arm above, one below; a ~4M-token arm would pin it).
- The probe protocol switch is a measured, arm-dependent artefact; all cross-checkpoint claims
  here use full-protocol pairs only (see [`FILES.md`](FILES.md)).
- One DGP setting (v16/s2/L6/m4); one model size. Corpus size, not RHM depth, is
  the swept coordinate; the level index ℓ carries the depth axis.

## Reproduce

```bash
cd experiments/            # MODAL_PROFILE=chromatic
modal run -m rhm.practice.reread.lm.lm_reread::gate
python3 rhm/practice/reread/lm/launch_detached.py --fn lm_reread --tag lm0 \
    --arms "fresh,frozen_200000,frozen_16384,frozen_2048,frozen_512" \
    --max-steps 20000 --batch-size 64 --lr 3e-4 --weight-decay 0.01 \
    --data-seed 7 --seed 42 --fresh-every 500 \
    --n-eval-sequences 8000 --eval-seed 999 --n-oracle 512 \
    --probe-steps 600 --probe-lr 1e-2 --mlp-hidden 128 --mlp-steps 800 --thresh-frac 0.5
python3 rhm/practice/reread/lm/analyze_lm.py --tag lm0 --fetch --figures
```

Volume: `/data/rhm_practice_reread_lm/lm0/` on `rhm-scaling-data`. Figures:
`figures/lm0/fig1_levels.png` (per-level recovery vs tokens, BP ceilings dashed), `fig2_loss.png`
(val NLL + memorisation gap), `fig3_excess.png` (BP-referenced excess per arrival level),
`fig4_ordering.png` (tokens-to-reach-level-ℓ).
