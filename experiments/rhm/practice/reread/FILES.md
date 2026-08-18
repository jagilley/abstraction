# reread — File Index and Calibration Record

Machinery record for this node. Findings: [README.md](README.md) (written post-discussion,
2026-08-17). Spec: [SPEC.md](SPEC.md). Machinery donor:
[`../ratchet/`](../ratchet/README.md) — **imported, never modified**.

## Children

| Child | What |
|---|---|
| [`lm/`](lm/README.md) | **The LM twin** — the follow-up where the reader is endogenous (NTP on RHM against exact BP oracles). Extraction is strictly level-ordered in every arm; a 6.4×-re-read corpus is fully renewable at zero token penalty; below a corpus-size wall, re-reading is capped, not slowed (~10× distinct corpus per half level). Machinery: [`lm/FILES.md`](lm/FILES.md) |

## Code files

| File | Purpose |
|---|---|
| `reread.py` | The Modal app. `build_archive` draws the **frozen** archive once (a declared depth mix over the `level_ladder` cells L1n6 / L2n3 / L3n1, a fixed presentation order, a SHA logged every pass); `archive_content` computes the **ground-truth denominator** — the set of true level-1 flat tuples the archive contains at each level, over each instance's damage-containing span, read off the CLEAN derivation with the exact inverse map, split into `set` (what any reader can see) and `true_set` (the grammatical part); `span_node` generalises `ratchet`'s per-era mining span to a per-instance one; `mine_into` is `ratchet`'s mining step grouped by damage cell; `closure_table` builds `T[l-1]^s`, the literal "re-derivable from `T[l-1]` alone" set; `audit` is `ratchet`'s `audition_macro` plus **which entries the max-sum DP actually selected** (the non-double-counting instrument); `run_cell` is the pass loop (pass = one complete traversal of the archive in `n_arch // chunk` cycles, each cycle `ratchet`'s cycle verbatim) with the **paired novelty probe** at every pass boundary. Entrypoints: `reread` (main), `gate` (C-M / C-R / G-D imported from `ratchet.selfcheck`, plus C-F archive-frozen and C-C archive-content) |
| `analyze_reread.py` | Reduction. `--fetch` pulls from the volume. Sections: **0** the fidelity gate against `ratchet`'s published era-1 numbers; **1–2** yield per pass per level (entries / new / new-and-used, audition, extracted fraction of the archive's own content); **3** the paired novelty probe (frozen vs fresh, same agent, same pass); **4** metering per damage depth and the archive (practice) epoch curve — the monolith baseline; **5** concentration vs coverage (`rand_k` / `comp_k` / `closure`); **6** commits and the plant guard. `--figures` writes fig1–fig4 |
| `launch_detached.py` | Session-isolated detached launcher — `../ratchet/launch_detached.py`'s pattern, copied rather than imported because the donor hardcodes its own module path |

## What is imported from `../ratchet/`, unmodified

`macros.py`: `Miner`, `base_table`, `make_table`, `true_tables`, `make_macro`, `macro_features`,
`to_device`, `parse_features`, `exact_features`, `grade_table`.
`ratchet.py`: `build_shared`, `beam_moves`, `build_ms`, `context_instances`, `era_ctx`,
`fit_width`, `finetune_generator`, `value_steps`, `push`, `plant_probe`, `priced`, `_cfg`,
`selfcheck`.

## Design decisions, and the measurement or argument that forced each

| decision | why |
|---|---|
| **The archive is drawn once, in a fixed presentation order, and SHA-checked every pass** | the spec's first caution. `gate`'s C-F asserts a redraw at the same seed is byte-identical; `run_cell` logs `arch_sha` on every pass record so a silent redraw is visible in the results file |
| **`mine_cap = 0` (uncapped)**, against `ratchet`'s 8 | `ratchet` capped mining per cycle so a 16-tip beam could not saturate a 14-entry table in one cycle. Here a *pass* means the whole archive is read; a cap would make "yield per pass" a sampling artefact rather than a statement about the archive |
| **Mining is not restricted to the era's level.** Every level is mined every pass from every solved instance | `ratchet` mined only `era_level + 1` so era k could not hand era k+1 a finished vocabulary. That schedule *is* the mechanism under test here, so imposing it would beg the question. The only gate left is the structural one (`T[l]` over `T[l-1]` entries, dropped at build time) plus what the agent could solve |
| **Scheduled commits at pass boundaries (L2 after pass 3, L3 after pass 6)**, not the unit-LP certificate | the independent variable is the **vocabulary stage**. A gate whose firing time is itself endogenous would confound stage with gate timing. The certificate is `ratchet`/`ear`'s object, not this node's |
| **Auditions are consumption-matched**: a level-ℓ macro is graded on level-ℓ damage (L2n3 for ℓ=2, L3n1 for ℓ=3), on fixed held-out sets | `ratchet`'s caveat 2 — its level-3 macro was auditioned on era-2 damage where it over-commits by construction, capping its best possible score near 0.51 and causing the certificate's refusal. Free to fix here |
| **Three yield flavours logged, and the headline is `n_new_used`** | the spec's second caution. `n_entries` is raw size; `n_correct` intersects the grammar; `n_used` counts entries the max-sum DP actually *selects* on consumption-matched damage, so an entry re-derivable by composition and never chosen contributes zero. `d_aud` (the pass-over-pass audition change) is the behavioural quantity the count is checked against |
| **Two coverage controls, not one**: `rand_k` (matched-size random subset of the TRUE table, `ratchet`'s instrument) **and** `comp_k` (matched-size random subset of the COMPOSITION CLOSURE `T[l-1]^s`), plus the full `closure` | `rand_k` answers "is the mined table better than a random *legal* table of the same size" (concentration vs coverage). `comp_k`/`closure` answer the sharper question the spec asks — "did the archive supply anything beyond what composing `T[l-1]` already gives you for free" |
| **The paired novelty probe** — at every pass boundary the same agent, held fixed, re-solves the frozen archive AND a fresh matched draw, mining each with a **throwaway per-pass miner** against its own operative lower tables | the between-arm `reread` vs `fresh` contrast carries all the drift of two different training histories. The paired probe puts the contrast inside one agent at one vocabulary stage. It also removes the **repetition-support confound**: the agent's own operative miner accumulates counts across passes, so a tuple present in one archive instance crosses `mine_support = 3` by pass 3 for purely arithmetic reasons; the throwaway miner requires support *within a single pass*, identically on both sides |
| **`n_distinct` (support 1) is logged alongside `n_at_support[3]`** via `Miner.state()` | separates the *solvability* channel (new spans observed because the agent can now repair more of the archive) from the *repetition* channel (old spans crossing the support threshold) |
| **The ground-truth denominator is split `set` vs `true_set`** | `build_inverse_maps` is last-writer-wins over bottom-level synonyms (`ratchet`'s `calp2_s0`: an irreducible **substrate** read error, not the agent's), so the exact parse of a clean derivation can name a tuple the grammar cannot produce. Measured at run setup: the mixed archive's raw level-2 content is 18 tuples of which 14 are grammatical, against a 14-tuple true table |
| **One shared substrate for every (mix, arm) cell** — `build_shared` runs once and every cell forks it | controlling variables. Running the depth mixes in parallel containers would have cut wall clock ~3× at the cost of three independently-trained substrates |

## Gates

| gate | what it asserts | status |
|---|---|---|
| C-M | a macro with the DGP's own table is bit-identical to `units.apply_move` at L2/L3 | passed (imported from `ratchet.selfcheck`, so donor drift breaks here first) |
| C-R | truncating T2 16→6 drops the rebuilt T3 | passed |
| G-D | the nested damage cells are on-grammar 1.000 at all three levels | passed |
| **C-F** | a redraw of the archive at the same seed is byte-identical, and the per-pass `arch_sha` never moves | passed for all three mixes |
| **C-C** | the archive's ground-truth content, against the grammar's | measured: at `n_arch = 192`, level-2 content is **12–14 of the grammar's 14** distinct tuples (essentially complete) and level-3 content is **37–41 of 56** (66–73%), so the archive is content-limited at level 3 and not at level 2 — which is the coordinate the novelty control is about |
| **G-P1** | pass 1 is bit-identical between the frozen and fresh arms (the fresh arm's pass-1 draw uses the archive's own seed), so "reread" differs from "ordinary" only from pass 2 on | built into `run_cell`; read off the results |
| **fid_ratchet** | pass-1 fidelity: `ratchet`'s era-1 configuration (shallow archive, fresh draws, `mine_cap = 8`, level-2 mining only, base action space) reproduces its published era-1 numbers — metered e ≈ 0.349 on L1n6, level-2 table recall ≈ 0.500 / precision ≈ 0.875 | **passed** on `rr_s0`: metered e = **0.336** (ratchet 0.349, Δ −0.013 — inside the measured noise floor), final level-2 table 9 entries, **recall 0.571 / precision 0.889** (ratchet 0.500 / 0.875), audition 0.270–0.316 (ratchet's commit auditions 0.281 / 0.293). Runs FIRST in every `reread` job; section 0 of the analysis |

### The noise floor, measured for free

Pass 1 is a **matched-condition replicate**: the frozen and fresh arms present the identical
archive (the fresh arm's pass-1 draw uses the archive's own seed) from the identical forked
substrate. They are *not* bit-identical, because torch's global RNG is not re-seeded per cell
(inherited from `ratchet`'s `run_arm`), so the six frozen/fresh pass-1 pairs give a direct
read of arm-to-arm noise: **|Δe| = 0.000–0.023 on the metered sets (mean ≈ 0.013)** and
**0–1 entries** on table size. Rank orderings below are read against that.

## Configuration of the main run (`rr_s0`)

Substrate is `ratchet`'s: v=8, s=2, L=4, m=2, rule_seed 0, train_seed 1, seed 0, plant_holdout 0,
declared grounding budget G=58, move budget 4, practice beam width 16, `max_macro_level` 3.

`n_arch = 192`, `chunk = 64` (3 cycles per pass), `n_passes = 9`, commits at passes 3 and 6
(three passes per vocabulary stage), `mine_support = 3`, `mine_cap = 0`, `mine_from = chosen`,
`n_rt = 256` per damage depth, `n_score = 256` per level, `n_rand = 3`, `gen_lr = 1e-4`,
`gen_steps = 20`, `n_grad = 4`, `value_lr_online = 3e-5`. Single seed.

Cells: 3 mixes (`mixed`, `deep`, `shallow`) × 5 arms (`reread`, `fresh`, `dense_reread`,
`dense_fresh`, `given_reread`), plus the `shallow__fid_ratchet` fidelity cell = 16.

## Reproduce

```bash
cd experiments/            # MODAL_PROFILE=chromatic

# structural gates (CPU, ~2 min)
modal run rhm/practice/reread/reread.py::gate
# smoke (attached, ~45 s)
modal run rhm/practice/reread/reread.py::reread --quick --tag smoke0 \
    --mixes mixed --arms "reread,fresh,dense_reread"

# the main run
python3 rhm/practice/reread/launch_detached.py --fn reread --tag rr_s0 \
    --mixes "mixed,deep,shallow" \
    --arms "reread,fresh,dense_reread,dense_fresh,given_reread" \
    --seed 0 --n-arch 192 --chunk 64 --n-passes 9 --commit-l2 3 --commit-l3 6 \
    --arch-seed 424242 --n-rt 256 --n-score 256 --mine-support 3 --mine-cap 0 \
    --budget 4 --pr-width 16 --g-budget 58 --max-macro-level 3 \
    --gen-lr 1e-4 --gen-steps 20 --n-grad 4 --value-lr-online 3e-5

# reduction
python3 rhm/practice/reread/analyze_reread.py --tag rr_s0 --fetch --figures
```

Modal volume (`rhm-scaling-data`): `/data/rhm_practice_reread/<tag>/<mix>__<arm>/results.json`,
with `setup.json` beside them. Figures land in `figures/<tag>/`.

## Results on disk

| tag | what it is |
|---|---|
| `smoke0` | the attached smoke (`--quick`): mixed mix, 3 arms, 3 passes, 48-instance archive |
| `rr_s0` | the main run: 16 cells, seed 0, 9 passes |
