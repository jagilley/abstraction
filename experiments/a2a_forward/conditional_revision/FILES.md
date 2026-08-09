# FILES — `a2a_forward/conditional_revision`

**Up**: [README.md](README.md) · **Parent**: [../README.md](../README.md) ·
**RHM sibling**: [`../../rhm/conditional_revision/`](../../rhm/conditional_revision/README.md)

## Code files

| file | purpose |
|---|---|
| `corpus.py` | **The substrate.** "Chronicle": controlled English with an exact belief oracle. Latent scene `z` = 4 slots × 4 single-token values (`\|Z\| = 256`); the clause plan is sampled from prefix-determined eligibility only, so the posterior stays a product of per-slot uniforms and `B_t` is closed form. Defines the analysis families (`synonym`, `offtopic`, `mention`, `reveal`, `reveal_narrow`, `negate_live`, `negate_dead`, `echo`, `deduced`) and the two clauses that make the cut work: the **negation** clause, whose marginal emission is exactly uniform over all 4 values in every posterior state, and the **narrow** clause, which eliminates values by naming the *survivors*. `self_test()` checks the factorised support against brute-force enumeration over all 256 scenes, `H_post` against `log \|support\|`, and the idea doc's §4 identity per position — all exact, all tokenizer-free and Modal-free. |
| `gates.py` | **The driver** (one Modal function). Loads a frozen reader, builds the tokenizer-conditional lexicon, generates and aligns the corpus, caches activations / `nll` / output entropy, trains the belief probe and the temporal FM, runs the ungrounded prompted readout, fits the cross-fit discriminants and their guards, and runs Gates 0 / A / B1 / B2 / B2b / B3 / D / E plus the B1 breakdown by kill mechanism. Writes one JSON per tag. |
| `fm.py` | `TemporalFM` (causal transformer, `FM(h[<=t]) → Δ_t`, the RHM Gate-0 object one substrate over), `SlotProbe` (h → per-slot categorical), `LinearDiscriminant` + `fit_discriminant` (balanced, unit-normalised so the readout is strictly directional). |
| `stats.py` | Matching / AUC / partial-`R²` helpers, ported in behaviour from `rhm/conditional_revision/gates_ab.py` so the numbers are directly comparable. Atom-aware strata, exact stratum crossing (no `a*K+b` aliasing), and `_shuffle_within` for the guards. |
| `shared.py` | Modal app, image (`transformers==4.53.2` pinned) and the `reading-data` volume. Documents why the app/volume keep the `reading` name after the node moved. |

## Results on the volume

`reading-data` (chromatic workspace):

| path | what |
|---|---|
| `/data/chronicle/gates_v3_seed42.json` | **the run the README reports** — all gates, all guards |
| `/data/chronicle/gates_v2_seed42.json` | same gates before the swap-state guard was added |
| `/data/chronicle/gates_v1_seed42.json` | first complete run; probe read at layer 12, no `nment` matching |
| `/data/chronicle/cache/acts_*.pt` | reader activations, `nll`, output entropy |
| `/data/chronicle/cache/prompt_*.pt` | the prompted readout (1390 s of a 1749 s cold run) |

## Children

None.
