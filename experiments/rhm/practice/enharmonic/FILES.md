# enharmonic — FILES

**Up**: [`../FILES.md`](../FILES.md) (practice). **Spec**: [`SPEC.md`](SPEC.md). **Q2 spec**:
`temperament/SPEC.md`[^private]. Results are facts-only until discussed:
[`sizing/SIZING.md`](sizing/SIZING.md) and `figures/en_*_reduction.txt`.

## Code files

| file | purpose |
|---|---|
| `enharmonic.py` | The fork of `../tutti/tutti.py` (every addition `# [enharmonic]`; G-F 0.000e+00 with the knobs off). Class-keyed miners via one factory, L5 open (`max_macro_level` 5, `gy_level` 6), the slot-resolved fired-path recorder, `entry_rec_cap`, the merge op as the closure `_try_merge` (block (g3.5)), clock yokes from a banked tag (`--yoke-from-tag`), arms `flat` / `given_cat_tok` / `given_cat_min` / `endo_yield` / `endo_ledger` / `endo_yield_force` and their `flat_yk_*` yokes, preflight gates E-2…E-6, `fidelity_smoke` retargeted at `tutti`. |
| `quotient.py` | What a category is in the fork: the class maps (`tok` = the token class, `gen`/`min` = its min element, `singleton` = flat), `ClassMiner` (class-keyed observe and build, flat-materialised cross-product execution, `mass_at_support` logged), the closure of `possible_sets`' composition step, gate E-0 (G-1…G-6). |
| `merge.py` | The merge as an outer-loop action: forced-transfer probe (`fourwall.entry_profile` at the level's own cell, one representative per class, `merge_max_rows`), `merge_candidates`' loss, the two licences (mass rise over a stated floor; `fourwall`'s keep-vs-merge audition — as ported it grades level ℓ, which a level-ℓ merge cannot change: see `QUEUE.md`), `LearnedQuotient`, gate M-0 (M-1…M-6, M-4b). |
| `analyze_enharmonic.py` | The reducer: lifetimes and commits [A], L5 [B], era-4/5 [C], π [D], battery growth [E], precision in both spaces [F], L4 class coverage per cycle [G]/[G2], bill [H], merge events in situ [M]/[M2], yoke realised-vs-planned [Y]. `--bank TAG:ARM` reads a banked arm from another tag. |
| `fetch_compact.py` | Mirrors a tag locally without `log["entry"]` (gzipped beside it): 808 MB → 3 MB on the smoke. |
| `alias_audit.py` | Offline, exact replay of every merge proposal in a tag (operative book, class map, probed set, the probe itself at the proposal's own seed and node): which alias pairs existed, which the probe was shown, which it scored, and their losses. Asserted against the run's own counts. |
| `floors_mass.py` | `tol_mass_l` per level by `null_abba` on a tag's logged `mass_at_support` series (the yield licence's currency), with the stated `support/Σcounts` fallback beside it. |
| `floors_l5l6.py` | `tol_yield_l5/l6` by `null_abba` on a tag's own class-keyed series, with the L3/L4 re-derivation as the calibration check. |
| `launch_detached.py` | The `tutti` launcher, retargeted. |
| `results/RUN_en_s0.sh` … `RUN_en_s9.sh` | The exact commands of record (`en_s6` the composed arm; `en_s7` the row cap; `en_s8` the multi-level sweep, two self-paced arms; `en_s9` `rearm_advance_only`). Logs beside them are untracked. |
| `floors_en_smoke*.json` | The measured floors (`en_smoke`, 3 arms × 77 cycles): `tol_yield_l5 = 0.2197`, `tol_yield_l6 = 0.1836` on the treated arms' series. |

## Children

| folder | what |
|---|---|
| [`sizing/`](sizing/SIZING.md) | Q0, offline: `phase0_cat.py` (the cover, the token class, the alias audit against a shuffled control, the forced-transfer budget, junk under transfer) and `fork_arrival.py` (the fork's own mining path simulated; the canonical re-rendering legality correction). Facts in `SIZING.md`. |
| `figured_bass/`[^private] | Commit the key, not the content: Q0 offline (`sizing/`), the `open_inventory` / `ungate_l5` knobs in this node's fork (`fb_open.patch`), runs `fb_s0` (own clock) and `fb_s1` (the anchor's clock). Facts in `../figures/fb_s{0,1}_reduction.txt`; discussion pending. |
| `temperament/`[^private] | Q2's spec (the endogenous quotient) and its conversation record. The Q2 arms live in this node's fork. |
| `figures/` | Compact mirrors of `en_s0`, `en_s1`, `en_s2` (`given_cat_tok` only), `en_s2b`, `en_s3`–`en_s9`, `fb_s0`, `fb_s1`, their `*_reduction.txt`, and `en_s{3..9}_alias_audit.txt`. `en_smoke` is on the Modal volume only (`rhm-scaling-data:/data/rhm_practice_enharmonic/`). |

[^private]: Not mirrored: this link points to a document in the private lab repo (the roadmap, the queue, an unrun spec, reading notes, or a conversation). See the top-level README for what is held back and why.
