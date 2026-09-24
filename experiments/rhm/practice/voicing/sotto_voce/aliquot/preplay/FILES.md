# preplay — machinery record

**Up**: [`README.md`](README.md) (the writeup) · [`../FILES.md`](../FILES.md) (aliquot).

## Code files

| file | purpose |
|---|---|
| `preplay.py` | pp1: the projection's read re-implemented (`pj_features`, `pj_design`, `pj_score`, `pj_predict`, `pj_fit`, `pj_irls`), the recording DP `fire_rec`, `wrong_rows`, the per-arm job `preplay_arm`, the coordinator `sweep`, and `gates` / `fidelity_gate` / `falsify` |
| `pool.py` | `soundboard::context_instances`, copied verbatim and gated on text identity (F-6) |
| `reduce_preplay.py` | pp1's reducer: `figures/preplay_reduction.txt` and `figures/pp1/<arm>_candidates.tsv` |
| `selector.py` | pp2: the read in the selector's seat, pricing on one pool and auditing the built tables on disjoint pools; `gates2`, `falsify2`, `sweep2` |
| `reduce_selector.py` | pp2's reducer: `figures/select_reduction.txt` and the [S] block of `preplay_reduction.txt` |
| `learner_tables.py` | pp3: the exact replay of an arm's operative tables at L2–L4 from `keys_at_support` and `last_build.picks`, with gates R-1, R-1b, R-1c, R-2, R-3 and the digest R-4 |
| `own.py` | pp3: the learner's own rows priced one at a time and the read as their selector |
| `reduce_own.py` | pp3's reducer: `figures/own_reduction.txt` |
| `incremental.py` | pp4: `census_walk`, the loop's `census_extend` gate transcribed and asserted identical on real auditions; `order_by`; the per-arm job `incremental_arm`, `sweep4`, `gates4`, `falsify4` |
| `reduce_incremental.py` | pp4's reducer: `figures/incremental_reduction.txt` and the [W4] block of `preplay_reduction.txt` |
| `readgate.py` | pp5: `decide` and `gated_walk`, the world's gate and the read's level as the gate in pooled, paired and margin forms with the frozen and twin controls; `readgate_arm`, `sweep5`, `gates5`, `falsify5` |
| `reduce_readgate.py` | pp5's reducer: `figures/readgate_reduction.txt` and the [G5] block of `preplay_reduction.txt` |
| `within/within.py`, `within/reduce_within.py` | the within-context readout; see [`within/FILES.md`](within/FILES.md) |

## Docs

| file | purpose |
|---|---|
| `README.md` | the writeup, the super-node for `within` and pp1 to pp5 |
| `NOTES.md` | pp1, pp2 (§7), pp4 (§8) and pp5 (§9): decisions, the gate table with what each was shown to fail on, defects beside corrections |
| `NOTES_own.md` | pp3: the same |
| `CONVERSATION.md` | the session record |

## Runs

| tag | what | containers | cost | app |
|---|---|---|---|---|
| `wi0`, `wi1`, `wi2` | within: the CPU scoring pass, three passes (`wi2` is of record, with the twin and the projection-form diet controls) | 8 CPU each, 272–577 s | no GPU | see `within/NOTES.md` §7 |
| `pp_smoke`, `pp1` | preplay, the single and delta forms, six arms | 6 L4, ~72 s each | 0.12 GPU-h | `ap-8LpmDBVCBYSdjcLwHH7qnS` |
| `pp2_smoke`, `pp2` | the selector, six arms | 6 L4, ~84 s each | 0.14 GPU-h | `ap-91frINqMJ1rwcOoWyxJH2u` |
| `pp3` | the learner's own entries, six arms, two passes (the second adds k = 1) | 6 L4 | 0.09 GPU-h | `ap-t5Qgpba2eGiBlP4f8nw8RD`, `ap-eV7VI14wmOZaKH0CLia7Rp` |
| `pp4` | the read as the order for the loop's gate, the two arms of record, settings (a) and (b) | 2 L4, ~250 s each | 0.14 GPU-h | `ap-ggxCqmIoQ9rab2FRqtoYEr` |
| `pp5` | the read as the gate with `dp_top` as the order, the two arms of record | 2 L4, ~175 s each | 0.10 GPU-h | `ap-iQYLGGBcORtkHtkHckBOrk` |

Volumes: `rhm-scaling-data:/rhm_practice_within/<tag>/<tag>/<arm>/`, `/rhm_practice_preplay/<tag>/`. Inputs read:
`/rhm_practice_soundboard/<tag>/<arm>/{vo_heads.pt, vo_bank.npz, vo_rows.npz, vo_rows_meta.json, results.json}` and
`/rhm_practice_voicing/{ov_s0b/ovt_comp_pr_sh, ov_s2/ovt_comp_pr_dis}/{vo_heads.pt, vo_rows.npz, vo_rows_meta.json}`.
The local mirrors of the soundboard arms (`../soundboard/figures/<tag>/<arm>/{results.json, entry.json.gz}`) are what
pp3's reconstruction is gated against; `entry.json.gz` is not on the volume.

## Gates

| script | gates | falsified |
|---|---|---|
| `preplay.py` | F-1 recording DP == `MC.apply_any` bit for bit; F-2 the re-implemented read == `VoProjBank.predict` elementwise; F-2b the banked hold AUC inside the readout's own drift band; F-3 candidate classes by `MC.grade_table`; F-4 the pool broken at the cell; F-5 the 1737-column design; F-6 `pool.py` text-identical | 7/7 |
| `within/within.py` | Z compose == `vo_compose`; A AUC == `vo_auc`; C the plant's fingerprint == the run's logged final; B batching; H the hold code a function of the context; D the filed degeneracy count can move; R the twin a fresh draw; P the projection's form verbatim on the bank; P1 (reported, not required) the root shuffle costs 0.0040 | 10/10 required |
| `selector.py` | S-1 pricing and test pools disjoint; S-2 a constant price reduces to the random draw; S-3 true/wrong accounting == `MC.grade_table`; S-4 the world selector beats random on its own pool; S-5 `grade_flat` == `MC.grade_table` | 6/6 |
| `incremental.py` | G-1a the walk identical to a transcription of `census_extend` on real auditions at cap 8, the full walk and tol 0.01; G-1b a hand-computed sequence; G-2 the three pool families pairwise disjoint; G-3 a constant price reduces to the tie-break | 4/4 |
| `readgate.py` | P-1a/P-1b the world gate is pp4's; P-2 handed the world's error as its level the read gate reproduces the world gate in all three forms; P-3 a constant level admits everything; P-4 pool disjointness; P-5 the changed set covers every differing instance, asserted non-vacuous | 5/5 |
| `learner_tables.py` / `own.py` | R-1 row count == `n_entries`; R-1b/R-1c pick indices inside the lower table; R-2 precision == the logged; R-3 the per-row truth mask row for row; R-4 the digest of the closed table in every container | 7 cases × 6 arms |

## Children

| folder | summary |
|---|---|
| [`within/`](within/FILES.md) | within-context candidate discrimination on the six soundboard dumps and overtone's two: the grader's filed probability, the prior, the critic, the composed chooser and five fitted controls, on pairs inside one context only. Decisions in [`within/NOTES.md`](within/NOTES.md) |
| [`timbre/`](timbre/FILES.md) | pp1, pp3 and pp5 re-read on richer reader forms of the same pooled state (the belief appended, an MLP on the state, an MLP on the belief alone, a per-slot ridge at L2/L3), refit on pp1's shared bank subsample through the shaped, frozen and twin trunks, the banked reductions run unchanged; arm 0 bit-identical to the banked pp1/pp3/pp5 columns. Facts in `timbre/figures/timbre_reduction.txt`; decisions in [`timbre/NOTES.md`](timbre/NOTES.md). No README yet (2026-09-23) |
