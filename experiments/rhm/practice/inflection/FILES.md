# FILES — `inflection` (the rendering rule below the tables)

Machinery record for the node, in [`../tutti/FILES.md`](../tutti/FILES.md)'s register.
**No results are interpreted here** — facts, gate numbers and the touch-point map only. There
is deliberately no `README.md` yet (repo norm: results are discussed before a writeup).

**Conversations**: `CONVERSATION.md`[^private] (the spec, then Q0 → Q2, this record).
**Up**: parent arc [`../README.md`](../README.md) · [`SPEC.md`](SPEC.md) (the question, the
arms, the sequence, the five open decisions) · [`DESIGN.md`](DESIGN.md) (the decisions as
taken, and the ones left open) · `CONVERSATION.md`[^private]
(how the idea was arrived at)
**Direct donor** (untouched): [`../tutti/`](../tutti/FILES.md) — `tutti.py` is forked here;
`tutti.py` is the in-process G-F reference.
**Sibling lane** (another agent's, not touched here): `phase0_inflection.py` / `phase0.json` /
[`sizing/SIZING.md`](sizing/SIZING.md) — the offline sizing of the rule family (SPEC decision 2
/ Q0). Its gate **I-3** independently reproduces this node's **R-7**: codes 12 and 39, exactly
2 collisions, last writer f=4 and f=7. Its **I-2** independently confirms that no bottom rule
can move the flat keys at L2–L5, which is this node's **R-6** from the other side.
**Imported, not forked**: `../../rhm_sculpt_precheck.py` (`sample_derivations`,
`possible_sets`, `parse_success_and_heuristic`) · `../crystallize/units.py` (`corrupt_hier`,
`apply_move`, `grade`, `oracle_rollout`) · `../ratchet/macros.py` (`apply_any`, `Miner`,
the tables) · `../native/span/span_net.py` · `../native/prop/prop_net.py` ·
`../antiphon/questions.py` · `../maestro/policy.py`.

## The scope condition this node lifts

On every prior RHM practice node the map from a level-1 feature to its surface tuple had no
free parameter in either direction — **verified in the donor's own code, not inherited**:

| where | the fact | site |
|---|---|---|
| data | the synonym is a uniform draw at every node of every level | `rhm_sculpt_precheck.py:63` |
| damage | the same, inside the damaged subtree | `crystallize/units.py:149,153` |
| grader | `possible_sets` reduces over synonyms with `.any(-1)` at the bottom and at every level; `parse_success_and_heuristic` is success on MEANING only | `rhm_sculpt_precheck.py:104,115,120` |
| executor | `canon = rules[depth-1][:, 0, :]` — synonym 0, always | `tutti/tutti.py:2220` |
| reader | `generator0`/`reader` pretrained on the coin-spelled corpus against the exact inverse map, then frozen; `read_acc` ≈ 1.000 | `tutti/tutti.py:2246` |

## The touch-point map

Every site the fork touches, and what it does there.

### The data path (numpy, local RNGs only — nothing reaches the shared torch stream)

| site | change |
|---|---|
| `sample_derivations_ruled` (fork, `inflection.py:1119`) | `sample_derivations` line for line; at the BOTTOM level only, the rule replaces the draw. The draw is still **taken and discarded**, so `rng` is consumed exactly as the donor consumes it |
| `_sample_pool_r` (`:1143`) | `_sample_pool` + the instance's context, drawn on a **disjoint** stream (`seed + 999_331`). `rule=None` calls the donor and returns `ctx=None` |
| `corrupt_hier_r` (`:1159`) | `units.corrupt_hier` line for line; the bottom `choice` becomes the rule at the instance's own context (levels above keep their coins — those are derivational, not spelling). Damage is on-grammar AND on-rule |
| `context_instances` (`:1315`) | `rule`/`n_ctx`/`with_ctx` keywords; with `rule=None` the donor's arrays, from the donor's calls, at the donor's RNG positions |
| `build_shared` (`:2698`) | builds the rule once; the pretraining corpus, `probe_clean` and the holdout filter are rule-spelled; `shared` gains `rule`, `ctx_pool`, `n_ctx` (asserted in `SHARED_KEYS`) |

### The executor (the renderer hook)

| site | change |
|---|---|
| `Renderer` (`:990`) | a **duck for `canon`**: `__getitem__(feats)` is the rule. Reaches all five write sites — `units.apply_move:111`, `macros.apply_any:243`, `span_net.SpanExecutor.apply:305`, `PerfExecutor._fire_only`, `PerfExecutor.apply` — with **no donor edited** |
| `_renderer` (`:1092`) | with `rule=None` returns `shared["canon"]`, the donor's TENSOR, so the write sites take the donor's op. `leaf`/`fit`/`decode` raise `NotImplementedError` here |
| `_bind` (`:1085`) | arms the renderer's cursor with the rows about to be rendered; a no-op on a tensor |
| `beam_moves` (`:1951`), `beam_moves_prop` (`:2057`) | a `ctx` column carried exactly where `roots` is carried (`repeat_interleave(width)`), bound before each `ex.apply` |
| `expand_selected_r` (`:1889`) | a fork of `PN.expand_selected` — the donor's loop plus one `_bind`, because the donor passes `flat[rows]` and the row indices cannot be recovered from the tensor. `ctx is None` runs the **donor's** function |
| `expand_selected_perf` (`:1912`) | the donor's own fork of the same loop, given the same `ctx` argument |
| `plan` (`:2175`), `behavior_step` (`:2570`), `audition_macro` (`:4193`) | `ctx` threaded / bound |

### `run_arm` — where the context comes from and where it goes

`meter[i]` and `shadow[i]` become 3-tuples `(roots, x, ctx)`; the per-cycle practice draw,
`pose_questions`' menu and tail, the commit audition (`seed + 700_000`), the extend audition
(`seed + 900_000`), the recert's `pol`, the width-ladder probe, the all-eras probe, the
no-span twin and the ablation battery each carry their own instances' context. Instrument
sites that were already arm-independent (`measure_refs`, the setup value buffer) keep
rendering at `canon` — [`DESIGN.md`](DESIGN.md) §5.

### The grader

| site | change |
|---|---|
| `rule_ok_table` (`:1197`) | `(v**s, n_ctx)` bool: is this leaf tuple one the rule calls for, for SOME feature that could have produced it. Quantifies over features because `build_inverse_maps` breaks bottom-tuple collisions by last-writer-wins — **2 collisions in 16 legal tuples** at v=8, s=2, m=2, rule_seed 0 |
| `spell_error` (`:1219`) | per row, `(wrong, scored)` over on-grammar blocks |
| `grade_spelled` (`:1239`) | `units.grade`'s two arrays **object for object**, plus `spell` beside them. `spell_weight=0` never executes the strict branch |
| `Renderer.tally` | the WRITTEN-block denominator (the SPEC's definition), exact and needing no write mask, using the same `rule_ok_table` verdict. Carries `n_unbound`, which must be 0 |

Nothing that keys on success sees the second number: mining, π, the certificate, both pacers
and `tacet`'s gate all read `succ`/`ps`, which is the donor's array.

## What the fork adds (each `# [inflection]`-marked; 104 marks, +976 / −98 lines vs the donor)

| addition | where | why |
|---|---|---|
| `Rule` + `make_rule` | module | a rule is a `(v, n_ctx)` synonym table — **family-independent**; four placeholder families (`const`, `offset`, `ctx`, `rand`) pending the sizing lane |
| `Renderer` + `_bind` + `_ictx` + `_renderer` | module | the executor hook (above) |
| `sample_derivations_ruled`, `_sample_pool_r`, `corrupt_hier_r` | module | the data path (above) |
| `rule_ok_table`, `spell_error`, `grade_spelled` | module | the graded grader (above) |
| `expand_selected_r` | module | the one donor render loop whose rows are unrecoverable |
| the `ctx` column | `context_instances`, `pose_questions`, `plan`, both beams, both expanders, `behavior_step`, `audition_macro`, `run_arm` | the context rides where `roots` rides |
| `log["e_sp"]`, `log["e_sp_practice"]`, `log["spell"]` | `run_arm` | the second number, in its own series |
| arms `canon`, `given_rule`, `leaf`, `fit_rule` + their `TWIN` rows | module | all four are `tu_y_exo`'s loop; only `render` moves |
| `rule`, `n_ctx`, `rule_table_seed`, `render`, `render_strict`, `spell_weight`, `perf_e_spell` | `_cfg` | all default OFF, so an unnamed config is `tutti.py` |
| `gates_cpu` | module | the offline suite (below) |
| `--rule`/`--n-ctx` on `preflight`, `inflection_run` | entrypoints | the world knob at run level; `render` is deliberately per-ARM |

## Gates

| gate | what it asserts | status |
|---|---|---|
| **G-F** | with `rule=None` this fork replays **`tutti.py`** in process, on the shared dict this file built, against a donor self-replay control | **PASS — 0.000e+00**, both arms, commits equal (control 0.000e+00) |
| **Q-11** | every selector, every era, on the real substrate — re-run in this fork | **PASS (18 cells)**, exo→bisect 7→10 / 8→18 / 1→9, identical to the donor's |
| **GG-1** | the graded grader at `spell_weight=0` returns `possible_sets`' verdicts EXACTLY — `success` and residual d\* element for element against `units.grade`, at four rule families | **PASS** (`gates_cpu`) |
| **GG-2** | the strict direction is reachable and moves only above weight 0 | **PASS** |
| **R-1a/b/c** | `rule=None` ⇒ the DONOR's sampler, damage and instance draw, array for array from the same seeds — the RNG-consumption half of the fork gate, checked with no GPU | **PASS** |
| **R-2** | a ruled corpus is on-rule and on-grammar in every one of 12,288 blocks, at every family | **PASS** |
| **R-3** | ruled damage is on-grammar AND on-rule in the same context | **PASS** |
| **R-4a/b** | the ruled POOL is the coin pool respelled: same roots; every level-1 feature mismatch is at a colliding code and **0 elsewhere** | **PASS** |
| **R-4c** | *reported, not asserted* — the rejection-sampled instance draw: d\* mean 2.3932 (coin) → 2.3177 / 2.3724 / 2.3646 / 2.3255 (`const`/`offset`/`ctx`/`rand`), roots differ on **7.0–9.1 %** of instances (the `d*==0` redraw). A ruled tag's instance set is not the coin tag's; within a tag every arm shares one world | **reported** |
| **R-5a** | a CANON rendering of a ruled world carries the spelling error the rule predicts, never more (`const` 0.000 = predicted; `ctx` 0.3464 = predicted; `offset` 0.2383 ≤ 0.4959; `rand` 0.4495 ≤ 0.6693 — the gap is R-7) | **PASS** |
| **R-5b** | a GIVEN rendering is on-rule in every block, at every family | **PASS** |
| **R-6** | the bottom inverse map is a function of `rules` alone and does not move with the rule — so `exact_features`, `level2_tuples`, `holdout_set`, `on_grammar_rate` and every level-size fact above L1 are untouched | **PASS** |
| **R-7** | *reported* — 2 of 16 legal bottom tuples share a code at (v=8, s=2, m=2, rule_seed 0); `rule_ok_table` quantifies over features so no spelling verdict depends on the tie-break | **reported** |
| **B-1 (bind)** | on a ruled preflight, `n_unbound = 0` in every cycle of every arm with `render_strict=True` — every executor call site on the run's code path is context-bound | **PASS** |
| P-*, L-*, Q-*, C-*, T-*, I-*, D-*, X-* | the donors' suites, reached by import or re-run in this fork | as donors |

## Volume layout

`rhm-scaling-data:/data/rhm_practice_inflection/<tag>/{setup.json,summary.json,<arm>/results.json,done.txt}`.
Fetched copies and figures under `figures/<tag>/`. The donors' tags are never written to.

## Runs

| tag | what | outcome |
|---|---|---|
| — | `gates_cpu` — the offline suite, CPU, no substrate | **ALL PASS (41 checks)**, ~20 s |
| `_preflight` | interface, `--arms canon`, `rule=None` — every call signature and every `shared[...]` access on the donor path | **PASS**, `results/preflight_canon.log` |
| `_preflight` | interface, `--arms canon,given_rule --rule ctx --n-ctx 3`, `render_strict=True` — every `_bind` site on the executor path, at toy sizes | **PASS**. `canon` wrote 14,704 blocks with **4,400 wrong (0.2992)**; `given_rule` wrote 14,704 with **0 wrong**; `n_unbound = 0` in both. `results/preflight_ruled.log` |
| `_preflight` | interface, `--arms canon,given_rule,leaf,fit_rule,fit_index --rule E_R8 --n-ctx 8 --practiced 0,3,7 --max-macro-level 3` — every Q1 code path (both learned organs, the transfer probe, per-register `read_acc`) at toy sizes | **PASS**, `n_unbound = 0` in all five, `blk_render` tallied in the three learned arms. `results/preflight_q1.log`; the launch file re-checked in `results/preflight_q1b.log` |
| `if_smoke` | the ruled `--quick` smoke, 5 arms, 30 cycles, `--reader-steps 6000` (production reader inside a quick run) | clean, **1253 s = 0.35 GPU-h**. Machinery numbers below; nothing at this scale is a result |
| `if_q1` | **THE Q1 RUN** — 5 arms, `E_R8`, practised {0,3,7}, eras `1:25:50,2:12:45,3:6:55`, `max_macro_level=3`, `--transfer-n 64`, otherwise `tu_s0`'s flags (full command under Reproduce) | clean, **8546 s = 2.37 GPU-h**, app `ap-XEIXOaMhVzEVZOeamqQdZN`. 135/92/82/150/113 cycles at 11.2–12.4 s/cycle. `n_unbound = 0` in all five. Reduction: `figures/if_q1_reduction.txt`; log `results/launch_if_q1.log` |
| `if_gf` | **G-F**: in-process fork-vs-`tutti.py` replay at `max_macro_level=4`, `rule=None`, plus gate Q-11 | **PASS — 0.000e+00** on `anchor` and `given_c1`, against a 0.000e+00 donor self-replay control, commit events equal. **Q-11 PASS (18 cells)**, needy-true-key aiming exo→bisect **7→10 / 8→18 / 1→9** — the donor's own numbers, cell for cell. 490 s ≈ **0.14 GPU-h**. `results/launch_if_gf.log`, `figures/if_gf/gate.json` |

### `if_q1` — the machinery record (facts; the interpretation is the orchestrator's)

**Gates in-tag**: G-F 0.000e+00 (self-replay control 0.000e+00, commits equal) · observation
panel ≡ the donor's G-Y miner on every cycle of all five arms · policy suite 21/21 · endo read
label-free · dead zones measured, matching `floors.json` · difficulty quota 1.000 in every arm
(d\* deviation −0.018…+0.010) · `n_unbound = 0` in all five arms.

**The five arms are NOT lifetime-comparable as run.** Each is yield-paced on its own clock:
82–150 cycles and a **43.1 % spread in priced spend** (20.50M…36.05M). The donors' priced-clock
section (§8T) needs the borrowed `ca_s0/dsil_sched` ceiling and skips itself, so a
matched-spend read is computed here at `t* = 20,500,384` (the shortest arm's lifetime):

| arm | cycles | t_cum(end) | cycle@t\* | e@t\* | e@end | e_sp@t\* | e_sp@end | e_sp(written, life) |
|---|---|---|---|---|---|---|---|---|
| `canon` | 135 | 32.71M | 80 | 0.9036 | 0.9401 | 0.0920 | 0.0932 | **0.3959** |
| `given_rule` | 92 | 23.66M | 80 | 0.7786 | 0.8802 | 0.0000 | 0.0000 | **0.0000** |
| `leaf` | 82 | 20.50M | 82 | 0.8229 | 0.8229 | 0.0199 | 0.0199 | **0.2372** |
| `fit_rule` | 150 | 36.05M | 83 | 0.7370 | 0.7760 | 0.0000 | 0.0000 | **0.0180** |
| `fit_index` | 113 | 26.60M | 89 | 0.7005 | 0.7057 | 0.0000 | 0.0000 | **0.0198** |

Commits: `canon` L2@c6 (5 entries, recall 0.214) · `given_rule` L2@c25 (14, 0.714) + L3@c62
(11, 0.125) · `leaf` L2@c25 (19, 0.786) + L3@c53 (6, 0.036) · `fit_rule` L2@c38 (19, 0.857) ·
`fit_index` L3@c67 (18, 0.107). Lifetime solved instances 3842 / 10800 / 17780 / 30878 / 30412.

**§1I — transfer, per register per level. The number is the renderer's own WRITTEN-BLOCK
spelling error** (the SPEC's definition), now printed as the `wrtn` row of §1I beside the
grader-side `e_sp`; `e_sp` is over EVERY on-grammar block of the graded configuration and is
therefore diluted by the written fraction (the executor rewrites a few blocks of 32 — the
dilution is ~4× at `canon`/ρ=6: 0.23 against 1.00). `*` = practised; the alias registers are
{1, 2, 6} and the genuinely-new ones {4, 5}.

| arm / level | *0 | 1 | 2 | *3 | 4 | 5 | 6 | *7 |
|---|---|---|---|---|---|---|---|---|
| `canon` L1 | 0.00 | 0.21 | 0.19 | 0.20 | 0.20 | 0.73 | 1.00 | 1.00 |
| `canon` L2 | 0.00 | 0.22 | 0.24 | 0.20 | 0.26 | 0.69 | 1.00 | 1.00 |
| `canon` L3 | 0.00 | 0.19 | 0.18 | 0.19 | 0.21 | 0.67 | 1.00 | 1.00 |
| `given_rule` L1/L2/L3 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| `leaf` L1 | 0.09 | 0.00 | 0.00 | 0.00 | 0.09 | 0.47 | 0.53 | 0.55 |
| `leaf` L2 | 0.07 | 0.00 | 0.00 | 0.00 | 0.10 | 0.56 | 0.46 | 0.47 |
| `leaf` L3 | 0.07 | 0.00 | 0.00 | 0.00 | 0.08 | 0.58 | 0.51 | 0.57 |
| `fit_rule` L1 | 0.00 | 0.08 | 0.00 | 0.00 | 0.07 | 0.15 | 0.00 | 0.00 |
| `fit_rule` L2 | 0.00 | 0.11 | 0.00 | 0.00 | 0.08 | 0.54 | 0.00 | 0.00 |
| `fit_rule` L3 | 0.00 | 0.08 | 0.00 | 0.00 | 0.07 | 0.35 | 0.00 | 0.00 |
| `fit_index` L1 | 0.00 | 0.00 | 0.00 | 0.00 | 0.05 | 0.50 | 0.41 | 0.00 |
| `fit_index` L2 | 0.00 | 0.00 | 0.00 | 0.00 | 0.03 | 0.59 | 0.57 | 0.00 |
| `fit_index` L3 | 0.00 | 0.00 | 0.00 | 0.00 | 0.08 | 0.55 | 0.63 | 0.00 |

Meaning (`e`) over the same cells is the `e` row of §1I in `figures/if_q1_reduction.txt`.

### THE COLLISION-SHADOWING DIAGNOSTIC (the §2I / §1I discrepancy, resolved)

The discrepancy — §2I says `fit_rule`'s table is wrong at practised ρ=3 on {1, 2, 4} while §1I
prints a written error of 0.00 there — is **real and fully explained**. The orchestrator's
hypothesis is CONFIRMED, in four parts, all CPU-only from the fetched tag.

**1. The two bottom-map collisions shadow four practised cells out of the harvest.**
`build_inverse_maps` is last-writer-wins: code **12** = (f1,s1) = (f4,s1) → parses as **f4**;
code **39** = (f2,s0) = (f7,s0) → parses as **f7**. Since θ_1 = 1 and θ_2 = 6, f1 spells s1 at
every ρ ≥ 1 and f2 spells s0 at every ρ ≤ 5 — so at the practised registers:

| practised ρ | parsed feature | rows the harvest actually gets | the rule's own call |
|---|---|---|---|
| 0 | f7 | (f2,s0) **and** (f7,s0) → label 0 | K[7,0] = 0 ✓ |
| 0 | f2 | **none** | — |
| 3 | f4 | (f1,s1) **and** (f4,s0) → 94 % label 1 | K[4,3] = **0** ✗ |
| 3 | f7 | (f2,s0) **and** (f7,s1) → 25 % label 1 | K[7,3] = **1**, majority contradicts |
| 3 | f1, f2 | **none** | — |
| 7 | f1 | **none** | — |

Empirically, from `log["spell_rows"]` (fit_rule, per (parsed feature, register), row count and
mean observed synonym): ρ=0 → f2 **0 rows**; ρ=3 → f1 **0**, f2 **0**, f4 3423 rows at mean
**0.94** (true 0), f7 5820 at mean **0.25** (true 1); ρ=7 → f1 **0 rows**. The **4 missing
practised cells** of 24 are exactly **(2,0), (1,3), (2,3), (1,7)** — which is why the offline
verdict fit covered 20–21 of 24.

**2. §2I's practised-register errors are the shadowed cells.** `fit_rule` is wrong at exactly
three practised cells — (1,3), (2,3), (4,3) — of which (1,3) and (2,3) are shadowed and (4,3)
is the cell **poisoned by** the shadow (94 % of its rows are f1's). `fit_index` adds (7,3), also
shadowed. No non-shadowed practised cell is wrong in either head, and three shadowed practised
cells the additive fit nevertheless recovered: (1,7), (2,0) for both, (7,3) for `fit_rule`.

**3. The write side is forgiven by the same collision.** `rule_ok_table` quantifies over
features (it must — see §6 of DESIGN.md), so an f4 written with s1 at ρ=3 is code 12, which is
the tuple the rule calls for at f1/ρ=3, and the verdict passes. Mix-free check: at every
register where **every** feature's `K̂` write is forgiven, the written error must be 0 for any
feature mix — predicted {0, 6, 7} for `fit_rule` and {0, 7} for `fit_index`, and both are
**contained in** the measured zero sets ({0,2,3,6,7} and {0,1,2,3,7}). The extra measured zeros
are at registers whose only unforgiven features are **f1 and f2 — which have zero parsed rows
there**: the same shadowing removes them from the write side as well.

**4. No `K̂` staleness or ρ-scaling bug.** Readout and render path call the SAME method on the
same module with no optimizer step between (`canon.set_khat(rhead.table(device))` immediately
precedes `rule_head_report(rhead, …)`, and both go through `RuleHead._x`, where the ρ/(R−1)
scaling lives), so a scaling mismatch is impossible by construction. `K̂` is not frozen — it
changed on 89/149 (`fit_rule`) and 60/112 (`fit_index`) cycle transitions. The empirical
table-compatibility test is reported as **non-discriminating**: every logged cycle's table is
compatible with the probe's measured zeros, so it neither supports nor refutes staleness; the
containment result in (3) and the code fact are what stand.

**Consequence for reading Q1**: at this rule draw, two of the eight features are invisible to
the learner's own parse at some practised registers, so `acc_practiced` in §2I has a ceiling
below 1.0 that is a property of the SUBSTRATE'S bottom map, not of the head. The same collision
also forgives the resulting writes, so `e_sp`/`wrtn` and `acc` measure different things at
those cells and are not in contradiction.

### THE READBACK SIDE — unreadable writes (CPU, from the fetched tag)

**Definition.** A write of intended feature *f* at register ρ emits the tuple
`bottom[f, T[f, ρ]]` for the arm's own render table *T*. The write is **unreadable** when
`bottom_map` reads that tuple back as a feature other than *f*. Weighted by the **true**
generative level-1 feature marginal (recomputed from the sampler, NOT through the shadowed
inverse map — that marginal is f0…f7 = 0.106, 0.198, 0.165, 0.078, 0.015, 0.131, 0.157, 0.151):

| arm | *0 | 1 | 2 | *3 | 4 | 5 | 6 | *7 |
|---|---|---|---|---|---|---|---|---|
| `canon` | 0.165 | 0.165 | 0.165 | 0.165 | 0.165 | 0.165 | 0.165 | 0.165 |
| **`given_rule`** | 0.165 | **0.363** | **0.363** | **0.363** | **0.363** | **0.363** | 0.198 | 0.198 |
| `leaf` (fallback) | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| `fit_rule` | 0.165 | 0.165 | 0.000 | 0.000 | 0.198 | 0.198 | 0.198 | 0.198 |
| `fit_index` | 0.165 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.198 |

beside the probe's meaning error `e` (last probe at L1 / L2):

| arm | *0 | 1 | 2 | *3 | 4 | 5 | 6 | *7 | corr(unreadable, e) |
|---|---|---|---|---|---|---|---|---|---|
| `given_rule` L1 | 0.44 | 0.97 | 0.86 | 0.88 | 0.86 | **0.53** | 0.64 | 0.69 | **+0.673** |
| `given_rule` L2 | 0.55 | 0.95 | 0.94 | 0.94 | 0.92 | **0.52** | 0.70 | 0.73 | **+0.574** |
| `fit_rule` L1 / L2 | 0.44 / 0.64 | 0.84 / 0.80 | 0.77 / 0.67 | 0.66 / 0.69 | 0.56 / 0.86 | 0.53 / 0.61 | 0.67 / 0.81 | 0.58 / 0.66 | −0.401 / +0.257 |
| `fit_index` L1 / L2 | 0.42 / 0.67 | 0.72 / 0.72 | 0.45 / 0.69 | 0.44 / 0.62 | 0.48 / 0.67 | 0.50 / 0.41 | 0.56 / 0.69 | 0.55 / 0.67 | −0.163 / +0.182 |

(`canon` and `leaf`'s fallback are constant in ρ, so the correlation is undefined for them.)

**The hypothesis is supported on its second half exactly and on its first half PARTLY.**

- **The fitted heads write what the reader can read back.** `fit_index`'s unreadable fraction
  is **0.000 at six of eight registers** and `fit_rule`'s at two, against `given_rule`'s 0.363
  across five. And the coincidence is not incidental: of `fit_rule`'s 15 cells that differ from
  the true K, **7 are exactly the cells where the true spelling would be unreadable, and in all
  7 the head substituted a readable spelling** — cells (1,1) (1,2) (1,3) (2,2) (2,3) (2,4)
  (2,5). For `fit_index` it is **11 of 28**, cells (1,1)…(1,6) and (2,1)…(2,5). The heads have
  learned the rule *modulo the reader's own confusions*.
- **`given_rule`'s unreadable fraction is highest over {1, 2, 3, 4, 5}, not {1, 2, 3, 4}** —
  the block is five registers wide, because f1 spells code 12 at every ρ ≥ 1 and f2 spells code
  39 at every ρ ≤ 5. Its meaning error is worst at {1, 2, 3, 4} (0.86–0.97) as observed, **but
  ρ = 5 has the same 0.363 unreadable fraction and the second-best meaning error in the arm
  (0.53 / 0.52)**. So the unreadable fraction tracks the meaning error (+0.67 / +0.57 across
  the eight registers) but **does not by itself account for it**: ρ = 5 is a clear exception
  and this record does not locate what distinguishes it.
- `canon`'s unreadable fraction is a flat 0.165 in ρ (it always writes synonym 0, and only f2's
  synonym-0 tuple collides), so nothing in `canon`'s per-register meaning profile can be
  attributed to readback.

**Does the parse prefer the last writer?**

- **The reader: YES, directly and by construction of the metric.** `read_acc` is scored against
  `bottom_map`, which maps code 12 → f4 and code 39 → f7, so `read_acc = 1.000` at a register
  IS the statement that the reader returns the last writer on every colliding block there. It
  is 1.000 at all three practised registers and 0.957–1.000 held out.
- **The generator: partly, and it cannot be settled exactly on CPU.** `parse_acc` is scored the
  same way; the share of clean rule-spelled blocks carrying code 12 or 39 is 0.316 / 0.363 ×4 /
  0.378 / 0.213 / 0.213 by ρ. Where `parse_acc > 1 − share` the generator must be agreeing with
  the last writer on at least some colliding blocks — true at ρ = 0, 1, 2, 3, 4 (e.g. ρ=3:
  0.752 against 0.637) and not decidable at ρ = 5, 6, 7. A per-code breakdown of `block_logits`
  needs the trained generator, and **no checkpoint is on the volume for this tag** (only
  `results.json` per arm), so it would take a ~10-minute GPU rebuild of `build_shared` at this
  cfg. Not run.

**§2I — identifiability** (last cycle; `theta_true = [1,1,6,4,5,5,5,1]`):

| head | n rows | acc practised | acc held-out | det θ | monotone | θ̂ |
|---|---|---|---|---|---|---|
| `fit_rule` (scalar ρ) | 57,680 | 0.875 | **0.700** | 0.125 | True | [2,4,2,6,3,6,5,3] |
| `fit_index` (one-hot ρ) | 58,824 | 0.833 | **0.400** | 0.125 | True | [1,7,1,7,1,7,7,7] |

Sizing-lane ceilings for these exact classes at C=3 practised: **0.81** (`additive_scalar`) and
**~0.60** (`additive_1hot`). Held-out trajectory: `fit_rule` 0.38 → 0.68 by c16, flat 0.68–0.75
thereafter; `fit_index` 0.28 → 0.53 by c31, decaying to 0.40.

**The offline `fit_rule_verdict` ceiling**, fitted from `log["spell_rows"]` (logged, never
consumed) — the same two heads on the arm's own **written** blocks with the rule's verdict as
the label (at m = 2 a right/wrong verdict determines the correct synonym):

| ctx | n rows | cells covered | acc practised | acc held-out | det θ | θ̂ |
|---|---|---|---|---|---|---|
| scalar | 27k–49k | 20–21 of 24 | 0.917 | **0.825** | 0.375 | [2,4,3,5,5,5,5,2] |
| one-hot | 27k–49k | 20–21 of 24 | 0.958 | **0.625** | 0.250 | [1,7,7,7,7,7,7,1] |

**Identical to three decimals from all three source arms** (`leaf`, `fit_rule`, `fit_index`) —
the ceiling is a property of the (feature, register) cell coverage, not of the arm.

**§4I — `leaf`'s side table**, end of run: **84 keys** over 19,163 observations, distinct
spellings per chunk mean **1.488**, max **5**, 31 keys with >1, `on_rule_frac` **0.774**.
Growth: 12 keys/224 obs at c1 → 30/7.6k at c31 → 84/19.2k at c82; `on_rule_frac` 0.954 at c1
settling to 0.77. Per-feature fallback [1,0,1,0,1,0,0,0].

**§4I — `read_acc` by register** (frozen reader, `canon`'s probe at c135; `*` = practised):

| ρ | *0 | 1 | 2 | *3 | 4 | 5 | 6 | *7 |
|---|---|---|---|---|---|---|---|---|
| `read_acc` | 1.00 | 1.00 | 1.00 | 1.00 | 0.99 | **0.96** | 1.00 | 1.00 |
| `parse_acc` | 0.72 | 0.76 | 0.75 | 0.75 | 0.69 | **0.61** | 0.71 | 0.72 |

**Unpriced ledger** (`blk_render`, never folded into `t`): 0 / 0 / 4.66M / 6.62M / 5.23M.

### What the smoke measured (machinery, not results)

- **`read_acc` by register, at production `reader_steps`, on the frozen reader — the sizing
  lane's open item 1.** 1.000 at every one of the eight registers except ρ=4 at 0.974,
  practised and **held out alike**. The rule's distribution shift costs the reader nothing at
  this draw. (`parse_acc`, the generator's own unmasked block head, is 0.33–0.54 — the arc's
  long-standing figure, unrelated to the rule.)
- **The transfer probe reads the family's own shape.** `canon`'s written-block spelling error
  by register: 0.000 / 0.112 / 0.048 / 0.091 / 0.132 / 0.612 / 1.000 / 1.000 at ρ = 0…7,
  against the fraction of features with K ≠ 0 of 0.000 / 0.375 / 0.375 / 0.375 / 0.500 / 0.875
  / 1.000 / 1.000. `given_rule` is **0.000 at every register**, held-out included.
- **The organs run and are separable.** `leaf`: 8 keys, `on_rule_frac` 1.000, hit rate 0.236,
  `distinct_spellings_max` 1 — undersampled at 30 quick cycles (n_obs 34), fallback still all
  zeros, so `leaf` ≡ `canon` numerically here. `fit_rule` / `fit_index` fitted from **64 / 128**
  observed blocks (the quick substrate solves almost nothing) and both recovered monotone
  tables; nothing at that sample size is a measurement.
- Every arm wrote the identical number of blocks (516,288) and `n_unbound = 0` throughout.

## Reproduce

```bash
cd experiments/            # MODAL_PROFILE=chromatic

PYTHONPATH=. python3 -c "from rhm.practice.inflection import inflection as I; I.gates_cpu()"

modal run rhm/practice/inflection/inflection.py::preflight --arms "canon"
modal run rhm/practice/inflection/inflection.py::preflight \
    --arms "canon,given_rule" --rule ctx --n-ctx 3

python3 rhm/practice/inflection/launch_detached.py --fn fidelity_smoke --tag if_gf

# the Q1 ladder (single seed)
python3 rhm/practice/inflection/launch_detached.py --fn inflection_run --tag if_q1 \
    --arms "canon,given_rule,leaf,fit_rule,fit_index" \
    --rule E_R8 --n-ctx 8 --practiced "0,3,7" --transfer-n 64 \
    --eras "1:25:50,2:12:45,3:6:55" --era-caps "50,45,55" \
    --max-macro-level 3 --budget 8 --g-budget 482 --question-k 2048 \
    --tol-dsil 0.0046 --endo-price 267 --gate-frac 0.5 \
    --span-tau-fire 0.50 --perf-alpha 0.05 --perf-tau-w 0.20 \
    --n-corrupt 1 --mine-cap 8 --gy-level 4 --entry-rec \
    --span-min-hold 128 --collect-task-matched --tm-episodes 8192 \
    --recert-every 5 --n-aud 192 --probe-every 8 --n-rt 384 --n-score 256 \
    --sil-cv 0.15 --lp-min-drop 0.10 --seed 0

python3 rhm/practice/inflection/analyze_inflection.py --tag if_q1 --gf-tag if_gf --fetch

# --- Q3 (added 2026-09-10) ---
modal run rhm/practice/inflection/inflection.py::preflight \
    --arms "m_pf_off,m_pf_sp,leaf_s" \
    --rule E_R8 --n-ctx 8 --practiced "0,3,7" --transfer-n 8 --setup-render rule

# F2, and stage 1 of E'. Both are `if_q1c_yk`'s design; only `--arms` moves.
for T in "if_q3_f2:leaf_s,canon_s,given_rule_s" \
         "if_q3_e:m_given_rule,m_fit_rule,m_leaf"; do
python3 rhm/practice/inflection/launch_detached.py --fn inflection_run \
    --tag "${T%%:*}" --arms "${T#*:}" \
    --rule E_R8 --n-ctx 8 --practiced "0,3,7" --transfer-n 64 \
    --rule-seed 6 --setup-render rule --reader-steps 6000 \
    --eras "1:25:40,2:12:45,3:6:55" --era-caps "40,45,55" \
    --max-macro-level 3 --budget 8 --g-budget 482 --question-k 2048 \
    --tol-dsil 0.0046 --endo-price 267 --gate-frac 0.5 \
    --span-tau-fire 0.50 --perf-alpha 0.05 --perf-tau-w 0.20 \
    --n-corrupt 1 --mine-cap 8 --gy-level 4 --entry-rec \
    --span-min-hold 128 --collect-task-matched --tm-episodes 8192 \
    --recert-every 5 --n-aud 192 --probe-every 8 --n-rt 384 --n-score 256 \
    --sil-cv 0.15 --lp-min-drop 0.10 --seed 0
done

# stage 2 takes the SAME line with --tag if_q3_e_sp, --arms "m_fit_rule_sp,m_leaf_sp"
# and --tol-dsil <the floor section 6I derives from if_q3_e's own dsil_sp>
```

## Q1 — the ordered register (added 2026-09-09)

**The rule**: `E_R8` — `K[f, ρ] = 1{ρ ≥ θ_f}`, ρ ∈ {0..7}, θ replicated verbatim from the
sizing lane's `phase0_inflection.fam_threshold(8, 8, PARAM_SEED=11)`. Recovered
**θ = [1, 1, 6, 4, 5, 5, 5, 1]**, `R_eff = 5`, **ρ = 0 IS `canon`** (every θ ≥ 1).
**Practised {0, 3, 7}**, **held out {1, 2, 4, 5, 6}**.

**The held-out ladder is not homogeneous, and this governs every transfer reading**: columns
1, 2, 3 are identical and 6, 7 are identical, so the held-out set splits into

| | registers | what they are |
|---|---|---|
| **aliases** of a practised column | **1, 2, 6** | transfer is free — a built-in NULL CONTROL |
| **genuinely new** columns | **4, 5** | the only real transfer rungs |

**The five arms** (one world, `tu_y_exo`'s loop, only `render` moves): `canon` (synonym 0) ·
`given_rule` (true K handed — the ceiling) · `leaf` (a side table keyed by the chunk's whole
feature tuple, replayed; table and DP untouched) · `fit_rule` (one linear logistic over
[one-hot feature ; **scalar** ρ] = the lane's `additive_scalar`) · `fit_index` (the same head
with ρ **one-hot** = `additive_1hot`; `tempo` finding 3 in one bit).

**What Q1 cuts against `tu_s0`, and why**: `max_macro_level` 4 → 3 (the coordinator's "level ≤
3"), so eras 1–2 earn L2/L3 and era 3 is a consumption era; era cycles 60/50/70 → 50/45/55; 5
arms instead of 8. Everything else — `budget`, `g_budget`, `n_pr`, `n_rt`, `n_score`, the
floors, the port — is `tu_s0`'s. `tu_s0` measured 13.1–14.7 s/cycle at maxl=4; at maxl=3 plus
the transfer probe (8 registers × 64 instances at `probe_every`) the estimate is ~12 s/cycle,
so 5 × 150 cycles ≈ **2.5–3.0 GPU-h**.

**New machinery** (all `# [inflection]`-marked): `Rule` gains the `E_R4`/`E_R8` branch and
`PARAM_SEED` · `practiced_set` and the `practiced` / `fixed_ctx` keywords on every draw ·
`SpellStore` (`leaf`'s side table) · `_build_rule_head` / `rule_head_fit` /
`rule_head_report` (the fitted renderer and its identifiability readout) ·
`observed_synonyms` · the transfer probe and its per-(era, register) fixed sets ·
`plant_probe`'s `by_register` cells · `log["render"]`, `log["transfer"]`,
`log["spell_rows"]` · `analyze_inflection.py` §§1I–4I.

### Q1 gates

| gate | what it asserts | status |
|---|---|---|
| **E-1** | the replicated `E_R8` table equals the sizing lane's own `fam_threshold` at `PARAM_SEED=11`, so §2's ceilings apply to THIS table | **PASS** |
| **E-2** | `R_eff = 5` of 8, canon rung `{0}`, every θ strictly interior | **PASS** |
| **E-3** | *reported* — the held-out ladder's split: aliases {1, 2, 6}, genuinely new {4, 5} | **reported** |
| **E-4** | lexical invisibility: **0** new leaf codes at every held-out register given practised {0, 3, 7} | **PASS** |
| **GG-1** | still holds with `E_R8` in the family list | **PASS** |
| **B-1 (bind)** | `n_unbound = 0` in all five arms on a ruled preflight with `render_strict=True` | **PASS** |

## Q1b — the collision-free draw, `fit_shared`, and the parity contrast (added 2026-09-09)

Three tags, built to separate the rule question from the reader question that Q1's readback
diagnostic surfaced. Full rationale in [`DESIGN.md`](DESIGN.md) "Q1b".

| tag | world | arms | what moves against its comparator |
|---|---|---|---|
| `if_q1b_cf` | `rule_seed 6` (bottom map **injective**), `E_R8`, practised {0,3,7} | `canon`, `given_rule`, `leaf`, `fit_rule`, `fit_index` | **the rule draw alone**, against `if_q1` |
| `if_q1b_sh` | `rule_seed 0` (the Q1 world — same seed, same instances) | `fit_shared` (+ `fit_rule` as its in-tag twin) | **the harvest's parse**: colliding codes resolved by the renderer's OWN `K̂` |
| `if_q1b_a` | `rule_seed 6`, **`A_2class`** (GF(2)-affine, R=4), practised {0,1,2} hold {3} | `given_rule`, `fit_index`, `fit_scalar`, `fit_gf2`, `fit_mlp` | **the family and the head class** |

**`if_q1b_cf` carries a caveat that must travel with it**: a collision-free draw deletes the
arc's native aleatoric channel (junk mass 0 at every level, sizing fact 5), so the L2+ mining
dynamics are not `if_q1`'s either. Its claims are **within-tag arm ranks** and the
identifiability readout **against its own ceiling** — not a cross-tag delta on anything the
mining path touches.

### Q1b runs

| tag | what | outcome |
|---|---|---|
| `_preflight` | `--arms fit_shared --rule E_R8 --practiced 0,3,7` — the shared-parse path on the launch file | **PASS**, `results/preflight_q1b.log` |
| `if_sm_sh` | `--quick` smoke, `fit_rule` + `fit_shared`, `rule_seed 0` | clean, **552 s = 0.15 GPU-h**. The shared parse ran (18 colliding blocks seen over the smoke, from ~2 solved instances); it exposed that counting only CHANGES reports 0 on a table that is working, so the instrument gained a `resolved` column (below) |
| `if_sm_a` | `--quick` smoke, all five A arms, `rule_seed 6`, `A_2class` | clean, **1099 s = 0.31 GPU-h**, `n_unbound = 0` in all five, every head fitted. At 126–352 rows nothing is a measurement |
| **`if_q1b_cf`** | the collision-free draw, 5 arms | clean, **8353 s = 2.32 GPU-h**, app `ap-ByUskmKF5rXWhHndxdGCh1`. Reduction `figures/if_q1b_cf_reduction.txt` |
| **`if_q1b_sh`** | `fit_shared` on the Q1 world | clean, **2641 s = 0.73 GPU-h**, app `ap-aeY4JOTruacuNxQe6JJu6f`. **Bit-identical to `if_q1/fit_rule`** — see below. Reduction `figures/if_q1b_sh_reduction.txt` |
| **`if_q1b_a`** | the GF(2)-affine contrast, 5 arms | clean, **8287 s = 2.30 GPU-h**, app `ap-mE2rjNwLB2BhXARUBXZhdD`. Reduction `figures/if_q1b_a_reduction.txt` |

All three carry `if_q1`'s flags verbatim except the three noted knobs; artifacts land at
`rhm-scaling-data:/data/rhm_practice_inflection/<tag>/`.

**The shared-parse instrument, as shipped** (the smoke's lesson): `n_ambiguous` (colliding
blocks seen) · **`n_resolved`** (the renderer's table picked a single owner, whether or not that
changed the reader's answer) · `n_relabel` (it changed the answer) · `n_determined` /
`n_resolved_determined` / `n_resolved_correct` / `n_relabel_determined` /
`n_relabel_correct` (against the TRUE rule) · **`n_reader_correct_determined`**, the baseline
the shared parse has to beat. All oracle readouts, logged and never consumed.

### `if_q1b_sh` — the machinery record

**`fit_shared` replayed `if_q1`'s `fit_rule` exactly.** Max |delta| = **0.000e+00** over 15
series across all 150 cycles (`e`, `succ`, `dres`, `t_cum`, `n_moves`, `width`, `g_per_solve`,
`e_practice`, `vloss`, `gloss`, `n_solved`, `n_mined`, `m_per_solve`, `e_sp`, `e_sp_practice`),
commit and advance events identical. So every Q1 number for `fit_rule` is this arm's number:
150 cycles, `e` 0.7370 at `t*` / 0.7760 at end, `e_sp`(written, life) 0.0180, `blk_render`
6,618,881, 30,878 lifetime solves, θ̂ = [2, 4, 2, 6, 3, 6, 5, 3], acc practised 0.875 /
held-out 0.700, the same three wrong practised cells **(1,3), (2,3), (4,3)** — none repaired —
and the same per-register written-block profile (0.00 / 0.08 / 0.00 / 0.00 / 0.07 / 0.35 /
0.00 / 0.00 at L3).

**The shared parse fired, and it changed nothing.** Lifetime ledger over 149 cycles with a row:

| quantity | count |
|---|---|
| `n_ambiguous` (colliding blocks seen) | 16,881 |
| `n_resolved` (K̂ picked a single owner) | **7,958** |
| `n_relabel` (that pick differed from the reader) | **0** |
| `n_determined` (the TRUE rule determines the owner) | 48,246 |
| `n_resolved_determined` | 7,568 |
| **`n_resolved_correct`** | **0** of 7,568 |
| `n_reader_correct_determined` | 40,575 (**0.8410** of the determined set) |

**Why, exactly.** With this K̂ (θ̂ above), at the practised registers:

| code | owners | reader | ρ=0 | ρ=3 | ρ=7 |
|---|---|---|---|---|---|
| 12 | (f1,s1), (f4,s1) | f4 | 0 survivors — unresolved | **resolves → f4, i.e. the reader's own answer; the true owner is f1 — WRONG** | 2 survivors — ambiguous |
| 39 | (f2,s0), (f7,s0) | f7 | 2 survivors — ambiguous | 0 survivors — unresolved (true owner f2) | 0 survivors — unresolved |

So the only cell the table resolves at all is code 12 at ρ = 3, and it resolves to the reader's
answer — which is the wrong one. **The shared parse in this form is a fixed point at
`fit_rule`'s own solution**: K̂ was fitted on labels the reader produced, so its wrong cells are
precisely the ones that make the resolution reproduce the reader, and the bootstrap has no
direction to leave from. Gate SH-1 shows the mechanism is correct when handed the TRUE table
(0.837 → 1.000 on the determined set); the run shows the learned table never reaches the basin
where that would start.

**What this does NOT test.** Only the harvest's readback moved; the execution-side reader is
untouched by construction (DESIGN.md Q1b.2) — the DP still resolves a macro through
`generator.block_logits` and the miner still keys on `MC.parse_features(shared["reader"])`. A
`fit_shared` that also serves the executor's intent is untried.

### `if_q1b_a` — the machinery record (GF(2)-affine, `rule_seed 6`, practised {0,1,2}, held out {3})

`K` rows (v=8 × R=4): four features on `w = (1,0)` and four on `w = (0,1)`, `a_f = 0`
throughout; `R_eff = 4`; ρ = 0 is the canon rung. Gates: G-F 0.000e+00, panel ≡ G-Y on every
cycle of all five arms, policy suite 21/21, quota 1.000 in every arm, `n_unbound = 0` in all
five. Comparability: 99–125 cycles, **priced spread 22.16 %** (24.84M…31.91M) — not
lifetime-matched; `t*` = 24,836,595. Commits: two per arm in every arm.

| arm | cycles | t_cum(end) | e@t\* | e@end | solves | e_sp(written, life) | acc practised | acc held-out |
|---|---|---|---|---|---|---|---|---|
| `given_rule` | 105 | 26.73M | 0.8932 | 0.8620 | 32,086 | **0.0000** | — | — |
| `fit_index` (one-hot) | 99 | 24.84M | 0.9245 | 0.9245 | 24,701 | 0.2750 | 0.667 | **0.000** |
| `fit_scalar` (bits as reals) | 107 | 26.98M | 0.9297 | 0.8750 | 33,083 | 0.2630 | 0.667 | **1.000** |
| **`fit_gf2`** (family-aware) | 101 | 26.39M | **0.7266** | **0.7005** | 27,993 | 0.0146 | **1.000** | **1.000** |
| `fit_mlp` (hidden 16) | 125 | 31.91M | 0.9323 | 0.8151 | 33,902 | 0.0179 | **1.000** | **1.000** |

**`fit_gf2` recovered the family exactly at its FIRST fit** (cycle 1, 1088 rows) and held it for
all 101 cycles: `(a_f, w_f)` = (0,(1,0)) ×3 for f0/f1/f3/f7 and (0,(0,1)) for f2/f4/f5/f6 —
**identical to the true parameters**, `K̂ ≡ K`, `acc` 1.000 at every register including the
held-out one, `n_undetermined = 0` on every cycle. That is the lane's `affine_gf2` ceiling
(1.00/1.00 at C = 3) reached, not approached.

**`fit_mlp` also reached 1.000/1.000**, from cycle 11 (13,184 rows) onward, loss → 0.0000.

**`fit_scalar` never fits the practised registers**: `acc_practiced` ranges 0.542–0.792 over the
run and is **never 1.000**; its final `K̂` is the constant row (0,0,1,1) for **every** feature —
it collapsed to "spell by register, ignore the feature", which is the best a monotone function
of the two bits can do against a parity rule. Its `acc_heldout` of **1.000** is an artefact of
that collapse: ρ=3 is the one register where the constant row happens to be right for all
eight features. The lane's `--` (cannot fit two practised contexts of a parity rule) is
confirmed in the direction that matters — the practised fit — and the held-out column is not
evidence of transfer.

**`fit_index`** likewise collapses to a constant row (0,0,1,0) and scores **0.000** at the
held-out register: the one-hot has no way to place ρ=3 and puts it at the wrong constant.

Written-block spelling error per register per level (`*` = practised; ρ=3 held out):

| arm / level | *0 | *1 | *2 | 3 |
|---|---|---|---|---|
| `given_rule` L1/L2/L3 | 0.00 | 0.00 | 0.00 | 0.00 |
| `fit_index` L1 / L2 / L3 | 0.00 | 0.30 / 0.07 / 0.47 | 0.48 / 0.49 / 0.49 | **1.00 / 1.00 / 1.00** |
| `fit_scalar` L1 / L2 / L3 | 0.00 | 0.48 / 0.43 / 0.39 | 0.73 / 0.48 / 0.40 | 0.00 |
| `fit_gf2` L1/L2/L3 | 0.00 | 0.00 | 0.00 | 0.00 |
| `fit_mlp` L1/L2/L3 | 0.00 | 0.00 | 0.00 | 0.00 |

### `if_q1b_cf` — the machinery record (collision-free draw, `rule_seed 6`)

Same `E_R8` θ, same practised {0,3,7}, same aliases {1,2,6} / novel {4,5}. Gates: G-F
0.000e+00, panel ≡ G-Y on every cycle of all five arms, policy 21/21, `n_unbound = 0`.
Comparability: 98–113 cycles, **priced spread 14.51 %** (24.52M…28.68M) — the tightest of the
three tags but still not lifetime-matched; `t*` = 24,521,146.

| arm | cycles | t_cum(end) | e@t\* | e@end | solves | e_sp(written, life) | commits |
|---|---|---|---|---|---|---|---|
| `canon` | 113 | 28.68M | **0.7943** | 0.8516 | 20,157 | 0.4797 | L2 c10, L3 c63 |
| `given_rule` | 106 | 26.76M | 0.9010 | **0.8177** | 30,118 | **0.0000** | L2 c24, L3 c57 |
| `leaf` | 101 | 25.66M | 0.9193 | 0.9010 | **35,443** | 0.4195 | L2 c37, L3 c81 |
| `fit_rule` | 112 | 27.63M | 0.9479 | 0.9375 | 29,202 | 0.0247 | L2 c24 |
| `fit_index` | 98 | 24.52M | 0.9323 | 0.9323 | 25,306 | 0.0239 | L2 c22 |

**With the collisions gone, both heads fit the practised registers perfectly.** `fit_rule`
`acc_practiced` **1.000** from cycle 10 onward (0.917 at c1) and `acc_heldout` **0.875**;
`fit_index` `acc_practiced` **1.000**, `acc_heldout` **0.750**. **Neither has a single wrong
cell at a practised register** (against Q1's three for `fit_rule` and four for `fit_index`), and
the harvest covers **24/24** practised cells in all three harvesting arms (Q1: 20/24). θ̂:
`fit_rule` [2,2,5,5,5,5,5,2], `fit_index` [1,1,7,7,7,7,7,1], against true [1,1,6,4,5,5,5,1].
Both above the lane's C=3 ceilings (0.81 scalar, ~0.60 one-hot), which the lane states as
averages over practised subsets rather than for this one.

Written-block spelling error per register per level:

| arm / level | *0 | 1 | 2 | *3 | 4 | 5 | 6 | *7 |
|---|---|---|---|---|---|---|---|---|
| `canon` L1 / L2 / L3 | 0.00 | 0.47/0.40/0.42 | 0.45/0.43/0.49 | 0.43/0.42/0.42 | 0.44/0.42/0.50 | 0.89/0.82/0.94 | 1.00 | 1.00 |
| `given_rule` L1/L2/L3 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| `leaf` L1 / L2 / L3 | 0.07/0.05/0.07 | 0.32/0.32/0.35 | 0.35/0.32/0.35 | 0.38/0.28/0.31 | 0.40/0.29/0.27 | 0.78/0.82/0.85 | 0.92/0.89/0.91 | 0.93 |
| `fit_rule` L1 / L2 / L3 | 0.00 | **0.47/0.36/0.40** | 0.00 | 0.00 | 0.02/0.04/0.00 | 0.34/0.11/0.06 | 0.00 | 0.00 |
| `fit_index` L1 / L2 / L3 | 0.00 | 0.00 | 0.00 | 0.00 | 0.02 | 0.43/0.47/— | 0.59/0.64/— | 0.00 |

**`given_rule`'s meaning error still varies by register with no collisions**: spread 0.281 at L1
(0.30 / 0.45 / 0.31 / 0.31 / 0.36 / 0.58 / 0.56 / 0.56, sd 0.116), 0.125 at L2, 0.141 at L3.
So register-dependent meaning cost is **not** an artefact of unreadable writes alone.

`leaf`'s side table: **44 keys** over 21,019 observations, distinct spellings/chunk mean
**2.136**, max **7**, 34 keys with >1, `on_rule_frac` **0.893**; fallback [0,1,0,0,0,0,0,0].
`read_acc` by register (frozen reader, c113): 1.00 / 1.00 / 1.00 / 1.00 / 0.98 / **0.94** /
1.00 / 1.00; `parse_acc` 0.58–0.66.

### Q1b gates (all in `gates_cpu`, CPU, no substrate)

| gate | what it asserts | status |
|---|---|---|
| **CF-1** | the bottom map's injectivity, by seed: **0 colliding tuples of 16 at rule_seed 6 and 10**, 2 at the arc's seed 0. (Recorded aside: seeds 6/10 carry 6/14 duplicate tuples in the layers ABOVE the bottom; those enter no map this arc reads — `build_inverse_maps` is consumed only at `inverse_maps[-1]` and the DP reads `rules` directly — so the lane's "collision-free" claim, which is about parse ambiguity, stands) | **PASS** |
| **A-1** | the replicated `A_2class` equals the sizing lane's `fam_two_class` at `PARAM_SEED=11` | **PASS** |
| **A-2** | `R_eff = 4` of 4 and the table contains a 2×2 **XOR** submatrix — `additive-real? = no`, so no real-valued additive renderer can represent it | **PASS** |
| **A-3** | lexical invisibility at C=3: **0 new leaf codes** at the held-out register, for **every** choice of it | **PASS** |
| **SH-1** | `shared_parse` with the TRUE table in `K̂`'s place: **7148 of 8192** blocks are determined by the true rule (every non-colliding block plus the collisions it resolves); the reader alone is right on **0.837** of them and the shared parse on **1.000**. 2208 blocks collide, the rule resolves **1164**, and at this table **every resolvable collision is one the last-writer map gets wrong**, so all 1164 are re-labels; the rest are left untouched | **PASS** |
| GG-1, R-*, E-* | Q1's suite, re-run | **ALL PASS — 61 checks** |

### New machinery (all `# [inflection]`-marked)

`make_rule` += `A_2class` and `Rule.bits_of` · `RuleHead` += `ctx_rep="bits"` and
`kind="mlp"` · **`GF2Head`** (a counting solver over the eight `(a_f, w_f)` triples per feature;
ties and unobserved features fall back to canon, `n_undetermined` logged) · `code_owners`,
**`shared_parse`**, `relabel_truth` · `fit_head` / `fit_hidden` / `fit_parse` knobs · arms
`fit_shared`, `fit_scalar`, `fit_gf2`, `fit_mlp` with their `TWIN` rows · the shared-parse
instrument in `log["render"]` (`n_ambiguous`, `n_relabel`, `n_relabel_determined`,
`n_relabel_correct`).

## Q1c — the meaning-currency confound, and `--setup-render` (added 2026-09-10)

**The diagnosis** (orchestrator's, from `if_q1b_cf` §1I at L1): the probe's meaning error rises
with register for exactly the arms that write synonym-1 tuples (`given_rule` 0.30 → 0.56,
`fit_rule` 0.25 → 0.58, `fit_index` 0.27 → 0.48 from ρ=0 to ρ=7) and is flat for the two that
never do (`canon` 0.17–0.31 while misspelling **100 %** of its writes at ρ=6,7; `leaf`
0.14–0.33). Meaning is spelling-agnostic by construction, and `rule_seed 6` has no bottom
collisions, so the gradient enters through the **planner's organs** — the stale value buffer,
collected with the `canon` executor (DESIGN §5 named the suspect in advance).

**Consequence, on the record: the MEANING ranks in `if_q1`, `if_q1b_cf` and `if_q1b_a` are not
interpretable as run.** The spelling and identifiability readouts never route through the value
head and are unaffected.

**The knob**: `--setup-render {canon,rule}`. `canon` is the default and reproduces every
`if_q1`-era config; `rule` makes every pre-arm organ that writes tuples render through the TRUE
rule at each row's own register. Audited scope: the only such path is the value-buffer
collector in both forms (`collect_value_buffer`, `_collect_task_matched`), via `behavior_step` →
`units.apply_move`. `_train_edit_controller` / `_train_generator` / `train_reader` consume the
(already rule-spelled) corpus under masking and render nothing; `measure_refs` renders but
trains nothing and stays at `canon`, stated rather than hidden.

| gate | what it asserts | status |
|---|---|---|
| **SR-0** | `setup_render_mode` is **inert at `rule=None`** (a coin world has no rule to spell with) and defaults to `canon` | **PASS** (`gates_cpu`, now 62 checks) |
| **SR-1** | with `--setup-render rule` on a ruled world, the value buffer's on-grammar blocks are 100 % on-rule at the register each row was written at — recorded per run as `setup.json → setup_render_check.buffer_on_rule` | measured per run |

The buffer now carries the register each state was written at (`replay["c"]`), so SR-1 is a
measurement of the buffer rather than of the corpus behind it.

**New arms**: `canon_s`, `given_rule_s`, `leaf_s`, `fit_rule_s`, `fit_index_s` — the five
renderers with `commit="delta_prov"` and no `loop`, i.e. the arc's own **schedule** idiom
(provisional commit at each era boundary, advance at the era cap), so every arm commits and
advances at the same cycles.

### Q1c runs

| tag | what | outcome |
|---|---|---|
| `if_sm_yk` | `--quick` smoke of the schedule mode + the knob, 2 arms, 6-cycle eras | clean, **380 s = 0.11 GPU-h**; `setup_render=rule` fired, `policy=schedule` in both arms. No commit at this scale — the L2 miner reached only 1 key at support 3 in 6-cycle eras, so the boundary commit had an empty table |
| `if_sm_yk2` | the same at 20-cycle eras, to exercise the boundary commit before a 2.3 GPU-h launch | clean, **1262 s = 0.35 GPU-h**. **The boundary commit fires**: both arms committed L2 at **c20** — the era-1 boundary — identically, on a 2-key table. (The `[arm]` summary line prints `commits@[]` because that field lists LOOP-driven commits only; the event record carries the real one.) It also exposed that `setup_render_check` was skipping `buffer_on_rule` whenever `--collect-task-matched` is on, because `_spiral_shared` REPLACES `shared["replay"]` with the task-matched buffer — fixed by carrying the register through that collector too |
| **`if_q1c_pr`** | the mechanism probe: `canon` + `given_rule`, `cf` world, yield-paced, `--setup-render rule` | clean, **3607 s = 1.00 GPU-h**, app `ap-SjJwaNxH5HLK6wpkCIJ6aD`. **The L1 gradient flattened** — see below. Reduction `figures/if_q1c_pr_reduction.txt`. **Caveat**: launched before the `_collect_task_matched` fix, so the FIX is active (both collectors already took `canon_setup`) but this tag's `setup_render_check` carries only `buffer_on_grammar` (1.000 over 8192 rows), not `buffer_on_rule` |
| **`if_q1c_yk`** | the five schedule-paced arms, `cf` world, `--setup-render rule`, eras `1:25:40,2:12:45,3:6:55` | clean, **11113 s = 3.09 GPU-h**, app `ap-SmMPE1CGDROskvziAlIpx3`. **SR-1 PASS: `buffer_on_rule` = 1.000 over 262,144 scored blocks.** Reduction `figures/if_q1c_yk_reduction.txt` |

### `if_q1c_pr` — the mechanism probe's verdict

**The ρ-gradient of the probe's meaning error, `if_q1b_cf` (setup at `canon`) against
`if_q1c_pr` (setup at `rule`).** Same world, same two arms, same flags but the knob.

| tag | arm | L | cyc | ρ = 0…7 | spread | sd | slope |
|---|---|---|---|---|---|---|---|
| `if_q1b_cf` | `canon` | L1 | c46 | 0.30 0.23 0.23 0.17 0.31 0.27 0.30 0.31 | 0.141 | 0.046 | +0.0078 |
| **`if_q1c_pr`** | `canon` | L1 | c34 | 0.39 0.39 0.31 0.34 0.36 0.28 0.34 0.44 | 0.156 | 0.046 | **+0.0002** |
| `if_q1b_cf` | `given_rule` | L1 | c50 | 0.30 0.45 0.31 0.31 0.36 **0.58 0.56 0.56** | **0.281** | **0.116** | **+0.0387** |
| **`if_q1c_pr`** | `given_rule` | L1 | c50 | 0.38 0.44 0.33 0.41 0.38 **0.41 0.33 0.41** | **0.109** | **0.037** | **−0.0015** |
| `if_q1b_cf` | `canon` | L2 / L3 | c72 / c113 | — | 0.172 / 0.125 | 0.060 / 0.044 | −0.0134 / −0.0050 |
| `if_q1c_pr` | `canon` | L2 / L3 | c79 / c88 | — | 0.203 / 0.172 | 0.069 / 0.050 | −0.0206 / +0.0015 |
| `if_q1b_cf` | `given_rule` | L2 / L3 | c95 / c106 | — | 0.125 / 0.141 | 0.046 / 0.045 | +0.0069 / −0.0035 |
| `if_q1c_pr` | `given_rule` | L2 / L3 | c95 / c120 | — | 0.250 / 0.094 | 0.079 / 0.026 | −0.0091 / +0.0054 |

**Verdict: the L1 gradient FLATTENED, and did not invert.** `given_rule`'s L1 slope goes
**+0.0387 → −0.0015** and its sd **0.116 → 0.037**; the ρ ≥ 5 tail that carried it (0.58 / 0.56
/ 0.56) is gone (0.41 / 0.33 / 0.41). `canon` — the arm that never writes synonym 1 — is
unmoved and flat (slope +0.0078 → +0.0002, sd 0.046 either way): it does **not** now pay at
ρ ≥ 1, so the effect did not invert. At L2/L3 neither arm carries a monotone ρ-gradient in
either tag and the spreads move both ways; the L1 rung is where the confound lived and where it
is gone.

The same thing read as a correlation across the eight registers, corr(written-block spelling
error, meaning error) for `canon` (the only arm whose written error varies with ρ; `given_rule`
is 0.000 everywhere so the correlation is undefined for it):

| tag | L1 | L2 | L3 |
|---|---|---|---|
| `if_q1b_cf` | **+0.247** | −0.711 | −0.220 |
| `if_q1c_pr` | **−0.065** | −0.846 | +0.037 |

**Headline numbers.** `t*` = 22,580,550 (`if_q1c_pr`), 26,761,773 (`if_q1b_cf`).

| tag | arm | cycles | t_cum(end) | e@t\* | e@end | solves | e_sp(written) | commits |
|---|---|---|---|---|---|---|---|---|
| `if_q1b_cf` | `canon` | 113 | 28.68M | 0.8802 | 0.8516 | 20,157 | 0.4797 | L2 c10, L3 c63 |
| `if_q1b_cf` | `given_rule` | 106 | 26.76M | 0.8177 | 0.8177 | 30,118 | 0.0000 | L2 c24, L3 c57 |
| `if_q1c_pr` | `canon` | 88 | 22.58M | 0.9271 | 0.9271 | 25,779 | 0.4723 | L2 c24, L3 c64 |
| `if_q1c_pr` | `given_rule` | 120 | 29.40M | **0.7578** | 0.8750 | **33,823** | 0.0000 | L2 c26 |

`setup_render_check` = `{mode: rule, n_rows: 8192, buffer_on_grammar: 1.000}` — **no
`buffer_on_rule`**, per the caveat above; `if_q1c_yk` carries that number on the same substrate
build.

**Two limits on this reading, stated rather than left implicit.** Single seed, and the probe is
64 instances per (era, register) cell, so a single cell moves on noise — the load-bearing
numbers are the L1 slope and sd, not any one register. And the two tags' arms are yield-paced
on their own gauges, so the probes are at different cycles (`canon` L1 c34 here against c46
there); `if_q1c_yk` is the matched-clock version.

### `if_q1c_yk` — the matched-clock read

**SR-1 PASS**: `setup_render_check` = `{mode: rule, n_rows: 8192, buffer_on_grammar: 1.000,
buffer_on_rule: 1.000, n_blocks_scored: 262144}`.

**The clock is matched to a degree the loop-paced tags never reached**: all five arms ran
**140 cycles**, eras c1–40 / c41–85 / c86–140, and priced spend **35,966,880…35,977,248 — a
0.03 % spread** (against 14.5–43.1 % in the loop-paced tags).

**Correction to the expectation: the commits are NOT at the ladder's boundaries.** In every arm
the certificate fired first, so `delta_prov` committed on the certificate — L2 at **c17 / c17 /
c17 / c19 / c19** and L3 at **c58 / c57 / c58 / c57 / c58** — within two cycles of each other,
which is what makes the table ages matched in practice:

| arm | L2 @cyc | entries | recall | L3 @cyc | entries | recall |
|---|---|---|---|---|---|---|
| `canon_s` | 17 | 10 | 0.467 | 58 | 9 | 0.161 |
| `given_rule_s` | 17 | 7 | 0.467 | 57 | 11 | 0.196 |
| `leaf_s` | 17 | 9 | 0.533 | 58 | 13 | 0.179 |
| `fit_rule_s` | 19 | 7 | 0.467 | 57 | 11 | 0.196 |
| `fit_index_s` | 19 | 7 | 0.467 | 58 | 11 | 0.196 |

**Meaning per era, end, solves:**

| arm | era 1 | era 2 | era 3 | e(end) | e(last 10) | solves |
|---|---|---|---|---|---|---|
| `canon_s` | 0.3832 | 0.7525 | 0.8036 | 0.6979 | 0.6953 | 34,326 |
| `given_rule_s` | 0.4188 | 0.8378 | 0.7386 | **0.6120** | **0.5961** | 38,579 |
| `leaf_s` | 0.3550 | 0.8025 | 0.8048 | 0.7005 | 0.6958 | 33,575 |
| `fit_rule_s` | 0.4071 | 0.8521 | 0.7358 | **0.6068** | 0.6344 | 39,825 |
| `fit_index_s` | 0.4119 | 0.8733 | 0.7535 | **0.6172** | 0.6469 | **40,309** |

**Spelling ranks reproduce `if_q1b_cf`'s.** Lifetime written-block error, `yk` against `cf`:
`canon` 0.4592 / 0.4797 · `given_rule` 0.0000 / 0.0000 · `leaf` 0.4269 / 0.4195 · `fit_rule`
0.0231 / 0.0247 · `fit_index` 0.0158 / 0.0239 — same order, same magnitudes. Per register the
two tags agree to ±0.05 in almost every cell (`canon` L1: 0.00 0.48 0.40 0.45 0.46 0.90 1.00
1.00 against 0.00 0.47 0.45 0.43 0.44 0.89 1.00 1.00). The one visible drift is `leaf_s` at L3
(0.38 0.16 0.16 0.15 0.22 0.62 0.76 0.74 against `cf`'s 0.07 0.35 0.35 0.31 0.27 0.85 0.91
0.92) — its side table is larger here (52 keys over 30,101 observations against 44 over 21,019)
and its ρ = 0 cell is worse while every other cell is better.

**Identifiability is unchanged by the pacing** (as it should be — it never routes through the
clock): `fit_rule_s` acc practised **1.000** / held-out **0.875**, θ̂ [2,2,5,5,5,5,5,2], 5 cells
differ and **none at a practised register**; `fit_index_s` **1.000** / **0.750**, θ̂
[1,1,7,7,7,7,7,1], 10 cells differ, none practised. Identical to `if_q1b_cf`'s.

**The register gradient, on a matched clock and an on-rule setup.** L1 ρ-slope / sd:
`canon_s` −0.0054 / 0.044 · `given_rule_s` **+0.0009 / 0.047** · `leaf_s` −0.0102 / 0.045 ·
`fit_rule_s` +0.0078 / 0.050 · `fit_index_s` +0.0009 / 0.064. Every arm is within
sd 0.044–0.064 and |slope| ≤ 0.011 — against `if_q1b_cf`'s `given_rule` at sd 0.116, slope
+0.0387. **No arm carries a register gradient at L1 any more**, including the two that
misspell 90–100 % of their writes at ρ = 6, 7.

`leaf_s`'s side table: **52 keys** over 30,101 observations, distinct spellings/chunk mean
**2.327**, max **6**, 37 keys with >1, `on_rule_frac` **0.882**; fallback [1,1,0,0,0,0,0,0].
`read_acc` by register (c140): 1.00 ×4, 0.98, **0.94**, 1.00, 1.00; `parse_acc` 0.64–0.69.

**Does the outcome currency separate the arms?** Yes, into two clusters and not five. On a
matched clock (140 cycles, 0.03 % spend spread, commits within two cycles) the three arms that
spell correctly end at `e` **0.6068 / 0.6120 / 0.6172** (`fit_rule_s`, `given_rule_s`,
`fit_index_s`; spread 0.0104) and the two that misspell end at **0.6979 / 0.7005** (`canon_s`,
`leaf_s`; spread 0.0026). The **gap between the clusters is 0.0807**, which is **4.65 × the
pooled in-tag null-ABBA floor on `e`** (sd(N) = 0.0347 over 500 windows, span 1, W 4, era
boundaries and commit cycles dropped → v_tol = 0.0173); the within-cluster spreads are
**0.60 ×** and **0.15 ×** that floor. Lifetime solves order the same way (38.6k–40.3k against
33.6k–34.3k, a 15–20 % gap). So spelling correctly is worth ~0.08 in meaning error here, and
*which* correct renderer — handed, fitted-scalar or fitted-one-hot — is not distinguishable at
this floor.

## Q2 — the curriculum ladder (added 2026-09-10)

`tempo` finding 6's twin: the body model was identifiable only once practice spanned two tempi.
Here the rungs are **contexts practised**, on exactly `if_q1c_yk`'s design — `cf` world
(`rule_seed 6`), `E_R8`, `--setup-render rule`, schedule-paced, the same 140-cycle ladder,
single seed — with only `--practiced` moving. Each rung needs its own `build_shared` (the
pretraining corpus is drawn in the practised registers), hence two tags.

| tag | practised | held out | arms | app |
|---|---|---|---|---|
| `if_q2_c1` | **{3}** (C=1) | {0,1,2,4,5,6,7} | `fit_rule_s`, `fit_index_s` | `ap-QC5eSZXVi2kWDHiW4vejF0` |
| `if_q2_c2` | **{0, 7}** (C=2) | {1,2,3,4,5,6} | `fit_rule_s`, `fit_index_s` | `ap-tCkGaXHygtQ7lvZgzOtP5D` |
| `if_q1c_yk` | {0, 3, 7} (C=3) | {1,2,4,5,6} | five arms | banked |

`{3}` rather than `{0}` for C=1: ρ=0 is the family's canon rung (`K[:,0] ≡ 0`), so practising it
alone would be `canon`'s world; at ρ=3 each feature is heard in exactly one of its two
spellings, and **both synonym indices are practised across features** (3 features spell 1, 5
spell 0). `{0, 7}` for C=2: the two ends of the scale, so every (feature, synonym) pair is
heard and every threshold is bracketed on (0, 7].

**No GPU smoke was run and none was needed**: nothing new was built. `--practiced` is an
existing knob, gated since Q1 (`practiced_set`, threaded through every draw), and both arms are
`if_q1c_yk`'s, already run at full scale. The only addition is the CPU gate below.

### Gate E-5 — what each rung makes lexically visible (`rule_seed 6`, the Q2 world)

| rung | leaf codes heard | new codes at a held-out register | verdict |
|---|---|---|---|
| C=1 {3} | **8 of 16** | **18**, by ρ: 0:3, 1:0, 2:0, 4:1, 5:4, 6:5, 7:5 | **REPORTED, not asserted** |
| C=2 {0,7} | 16 of 16 | 0 at every held-out register | **PASS** |
| C=3 {0,3,7} | 16 of 16 | 0 at every held-out register | **PASS** |

**At C=1 the held-out registers are lexically VISIBLE by construction** — each feature is heard
in exactly one of its two spellings, so 8 of the 16 leaf codes are never in the corpus at all.
`read_acc` by register is therefore a **readout** at that rung, not a gate, and a drop there
**is the phenomenon** (finding 6's "the drag term sits below the practice noise at one tempo"),
not a bug. ρ = 1 and 2 show 0 new codes because their columns alias the practised ρ = 3.
`gates_cpu` is 65 checks, ALL PASS, at `rule_seed` 0 and 6 alike.

### Q2 runs

| tag | what | outcome |
|---|---|---|
| `if_q2_c1` | C=1, practised {3} | clean, **4892 s = 1.36 GPU-h**, app `ap-QC5eSZXVi2kWDHiW4vejF0`. Reduction `figures/if_q2_c1_reduction.txt` |
| `if_q2_c2` | C=2, practised {0,7} | clean, **4993 s = 1.39 GPU-h**, app `ap-tCkGaXHygtQ7lvZgzOtP5D`. Reduction `figures/if_q2_c2_reduction.txt` |

### The ladder, read

**Two structural facts first, because they govern how every row below is read.**

1. **At C=1 the two heads are the SAME arm.** With one practised register a single one-hot
   column carries exactly the information a single scalar value does, so only the per-feature
   bias is identifiable. Measured: `max|fit_rule_s − fit_index_s| = 0.000e+00` over 15 series
   across all 140 cycles, commits equal, **and identical `K̂`**. The C=1 row is one arm printed
   twice.
2. **At C=2 the two heads have identical RUN traces but different tables.**
   `max|delta| = 0.000e+00` over 15 series, commits equal, **but `K̂` differs**. The reason is
   clean: both are correct at every practised register (acc practised 1.000), so during the run
   — which only ever visits practised registers — they write identically; they part company
   only in the unpriced transfer probe. At C=3 they genuinely diverge (max|delta| 5.15e+04,
   commits differ).

**Identifiability ladder** (last cycle; θ_true = [1,1,6,4,5,5,5,1]; ceiling = the sizing lane's
`E_R8` §2 value for that class at that C):

| head | C | practised | acc prac | acc held | ceiling | det θ | monotone | θ̂ |
|---|---|---|---|---|---|---|---|---|
| `fit_rule_s` | 1 | {3} | 1.000 | **0.679** | 0.74 | 0.000 | True | [0,0,8,8,8,8,8,0] |
| `fit_rule_s` | 2 | {0,7} | 1.000 | **0.708** | 0.79 | 0.125 | True | [4,4,4,4,4,4,4,4] |
| `fit_rule_s` | 3 | {0,3,7} | 1.000 | **0.875** | 0.81 | 0.375 | True | [2,2,5,5,5,5,5,2] |
| `fit_index_s` | 1 | {3} | 1.000 | **0.679** | ~0.56 | 0.000 | True | [0,0,8,8,8,8,8,0] |
| `fit_index_s` | 2 | {0,7} | 1.000 | **0.417** | 0.59 | 0.000 | True | [7,7,7,7,7,7,7,7] |
| `fit_index_s` | 3 | {0,3,7} | 1.000 | **0.750** | ~0.60 | 0.375 | True | [1,1,7,7,7,7,7,1] |

The scalar head is **monotone in C** (0.679 → 0.708 → 0.875) and crosses its ceiling only at
C=3; the one-hot head is **not** (0.679 → 0.417 → 0.750) and is below its ceiling at C=2. Both
recover a **degenerate constant θ̂** at C=1 and C=2 — every feature given the same threshold —
and only at C=3 does either produce a table with more than one distinct row. `det θ` is 0.000
at C=1 for both, 0.125 / 0.000 at C=2, 0.375 for both at C=3.

**Interpolated vs extrapolated** (the lane's second axis). {0,7} brackets the scale, so at C=2
**every** held-out register is interior; at C=1 all seven are outside the practised point
(ρ=0–2 on one side, 4–7 on the other); at C=3 all five are interior. Per-register accuracy:

| head | C | interior | outside |
|---|---|---|---|
| `fit_rule_s` | 1 | — | {0,1,2,4,5,6,7} 0.679 |
| `fit_rule_s` | 2 | {1..6} 0.708 | — |
| `fit_rule_s` | 3 | {1,2,4,5,6} 0.875 | — |
| `fit_index_s` | 2 | {1..6} 0.417 | — |
| `fit_index_s` | 3 | {1,2,4,5,6} 0.750 | — |

**Written-block spelling error by register**, and by distance from the practised set:

| head | C | mean written error at distance d = 1 / 2 / 3 / 4 |
|---|---|---|
| `fit_rule_s` | 1 | 0.02 (n6) / 0.25 (n6) / 0.53 (n6) / 0.63 (n3) |
| `fit_rule_s` | 2 | 0.24 (n6) / 0.31 (n6) / 0.49 (n6) / — |
| `fit_rule_s` | 3 | 0.12 (n12) / 0.10 (n3) / — / — |
| `fit_index_s` | 1 | 0.02 / 0.25 / 0.53 / 0.63 (identical — same arm) |
| `fit_index_s` | 2 | **0.74** (n6) / **0.70** (n6) / 0.48 (n6) / — |
| `fit_index_s` | 3 | 0.15 (n12) / **0.41** (n3) / — / — |

At C=1 the scalar head's error grows **monotonically with register distance** (0.02 → 0.63) —
the monotone extrapolation working from a single point in the near field and failing in the
far. At C=2 the scalar head is flat-ish across the interior (0.24–0.49) while the one-hot head
is **worst at the near neighbours** (0.74, 0.70) — it cannot place an unseen register on a
scale at all, so proximity buys it nothing.

**`read_acc` by register** — a **readout** at C=1 (gate E-5: the held-out registers are
lexically visible there by construction) and a **gate** at C=2:

| rung | ρ=0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|---|
| C=1 {3} | 0.68 | 1.00 | 1.00 | **1.00\*** | 0.89 | **0.52** | **0.42** | **0.43** |
| C=2 {0,7} | **1.00\*** | 0.73 | 0.75 | 0.77 | 0.72 | 0.91 | 1.00 | **1.00\*** |
| C=3 {0,3,7} | **1.00\*** | 1.00 | 1.00 | **1.00\*** | 0.98 | 0.94 | 1.00 | **1.00\*** |

At C=1 the reader collapses to **0.42–0.52** at ρ = 5, 6, 7 — the registers whose spellings it
never heard — exactly the shape E-5 predicted (18 leaf codes first seen at a held-out register,
concentrated at ρ = 5, 6, 7). At C=2 every code is heard and `read_acc` holds at 0.72–1.00, and
at C=3 at 0.94–1.00. `parse_acc` (the generator's own head) tracks it: 0.31–0.75 at C=1,
0.60–0.67 at C=2, 0.64–0.68 at C=3.

**Meaning per era on the matched clock** — all rows are 140 cycles with priced spend
35.967M–35.977M (a 0.03 % spread):

| arm | era 1 | era 2 | era 3 | e(end) | solves |
|---|---|---|---|---|---|
| C=1 `fit_*_s` (one arm) | 0.3510 | 0.7685 | 0.7625 | 0.6432 | 40,281 |
| C=2 `fit_*_s` (identical traces) | 0.4048 | 0.8128 | 0.7421 | 0.6562 | 41,547 |
| C=3 `fit_rule_s` | 0.4071 | 0.8521 | 0.7358 | **0.6068** | 39,825 |
| C=3 `fit_index_s` | 0.4119 | 0.8733 | 0.7535 | 0.6172 | 40,309 |
| C=3 `canon_s` (floor) | 0.3832 | 0.7525 | 0.8036 | 0.6979 | 34,326 |
| C=3 `given_rule_s` (ceiling) | 0.4188 | 0.8378 | 0.7386 | 0.6120 | 38,579 |

**A caveat that has to travel with that table**: each rung has its own `build_shared`, so the
pretraining corpus, controller, generator, reader and stale value differ between rungs. A
cross-rung meaning comparison therefore confounds the curriculum with the corpus, and the
`e(end)` ordering (C=3 0.607 < C=1 0.643 < C=2 0.656) is not monotone in C. The **within-rung**
comparisons — and the identifiability ladder, which never routes through the value head — are
the clean readings.

**Harvest cell coverage is complete at every rung**: 8/8 at C=1 (57,390 rows), 16/16 at C=2
(63,168), 24/24 at C=3 (60,928 / 60,544). No rung is starved of a practised cell.

Readouts on landing, per rung with C=3 from `if_q1c_yk` beside: the identifiability ladder
(acc practised / held-out, θ̂, det) against the sizing lane's `E_R8` ceilings by C — **0.74 /
0.79 / 0.81** for `additive_scalar` and **~0.56 / 0.59 / 0.60** for `additive_1hot` at C = 1 /
2 / 3 — and the lane's interpolated-vs-extrapolated split; the written-block spelling error per
register per level, read as a function of register distance from the practised set; meaning per
era on the matched clock; `read_acc` by register.

**A design note for a later rerun, not acted on**: `if_q1b_a`'s held-out register r=3 is the
**all-ones column** of `A_2class`, so any head that outputs 1 at an unseen register scores 1.000
there — the artefact behind `fit_scalar`'s held-out 1.000. A rerun should hold out r=1 or r=2.

# Q3 — the two readouts the SPEC never had a seat for (added 2026-09-10)

Two independent questions, two tags, one shared piece of new machinery. Full design record in
[`DESIGN.md`](DESIGN.md) "Q3". Nothing here is consumed by any decision: both readouts are
logged and read offline.

## What was built

| piece | where | consumed by |
|---|---|---|
| `Renderer.last_ok` | the per-block spelling verdict of the LAST render, kept rather than recomputed | `PerfExecutor.apply` (E' only) |
| `Renderer.slot_tab` / `slot_tally()` | per-(macro slot, register) written/misspelled counts, a `bincount` over the register column already in hand | `log["f2"]` only |
| `_bind(canon, ctx, move)` | labels the write with its macro slot; all 6 call sites pass the move | the tally |
| `perf_e_mode(cfg, rule)` / `perf_e_compose(e_feat, ok_frac)` | E's decision logic and composition, isolated as pure functions so the gates need no GPU | gates PE-0 / PE-1 |
| `PerfExecutor(meter_sp=, e_spell=)` | the shadow meter and the one bit | `log["perf_sp"]`, `panel["dsil_sp"]` |
| `log["f2"]` | one row per cycle per arm; cells keyed by macro slot | NOBODY |
| `log["perf_sp"]`, `panel["dsil_sp"]` / `panel["dsil_ft"]` | the shadow benchmark and its open-slot mean, **named by content**: `dsil_sp` is always the spelling-charged reduction, `dsil_ft` always the feature-only one, `dsil` always the DRIVEN one | NOBODY (the floor is derived offline from it) |
| arms `m_given_rule`, `m_fit_rule`, `m_leaf`, `m_fit_rule_sp`, `m_leaf_sp` | `tu_m_exo`'s mirror seat x renderer x the one bit | — |
| arms `m_pf_off`, `m_pf_sp` | **preflight only** — the mirror seat holding the TRUE tables, so a slot opens at c1 and both new paths actually run on a forty-step substrate | — |
| gates PE-0, PE-1 | `gates_cpu`, now **67 checks, ALL PASS** | — |

`--perf-e-spell` exists at run level (default False, inert at `rule=""`); the `_sp` arms carry
it as an arm-level cfg key, which is what lets an off/on pair share one tag.

## The F2 row's columns

`n_used` / `n_calls` (from `out["seq"]`), `solve_used` / `solve_unused` (from `ps`),
`written` / `wrong` **per register** (from the renderer's tally), `pi` (from
`probe["pi"]["mean_mass"]`, so present on probe cycles only), plus `age`, `open`, `level`,
`node`, and the cycle's `base_written` / `base_wrong` for the writes outside every macro.
`solve_unused` is there so **use count can be partialled out at read time**.

## Q3 runs

| tag | what | outcome |
|---|---|---|
| `preflight_q3` | `--arms "m_given_rule,m_leaf,m_leaf_sp,leaf_s"`, ruled | clean, EXIT 0, app `ap-FpJoNfz48Ua7OJHdHqDAVI`. **And it found the hole**: the three mirror arms never commit at preflight scale, so `pmeter.n_call = 0`, `dsil = None` in all 27 cycles and `log["f2"]` is empty — both new paths untested. `leaf_s` DID exercise them: 21 F2 rows, 16-28 cells each, all 21 carrying `pi`; `dsil_sp` live on all 21 cycles and **> `dsil` on every one** (last pair 0.393 / 0.575), which is `perf_e_compose`'s inequality on the live executor |
| `preflight_q3b` | `--arms "m_pf_off,m_pf_sp"` — the two new preflight-only arms | clean, app `ap-3rZY6BkzTC0iJ8UFrK4K9F`. The executor is live from c2 in both (`fired` ~490/cycle) |
| `preflight_q3c` | `--arms "m_pf_off,m_pf_sp,leaf_s"` on the FINAL code (shadow carries the other currency, `dsil_ft` added) | clean, app `ap-e9VUJcehUFJ4x6AcECRmR1`. **Everything the design claims, measured**: in the OFF arms driven `dsil == dsil_ft` exactly and `dsil_sp` is the shadow; in the ON arm driven `dsil == dsil_sp` exactly and `dsil_ft` is the shadow; **`dsil_sp > dsil_ft` on 100 % of live cycles in all three arms** (`perf_e_compose`'s inequality on the live executor). F2 rows 27 / 25 / 21. The one bit moves the run: `m_pf_off` ran 27 cycles and `m_pf_sp` 25, and the driven `dsil` series part at **c1** — the first execution — while `log["e"]` never parts, because at preflight scale nothing is solved. **The gate is the driven `dsil`, not `e`**: the bit moves the EXECUTION currency, and meaning only moves downstream once the pacer decides differently |

### Q3 launches

| tag | arms | app | GPU-h (est.) |
|---|---|---|---|
| **`if_q3_f2`** | `leaf_s,canon_s,given_rule_s` (schedule-paced, so π's mass on the SAME slot at the SAME cycle is comparable across arms) | `ap-Fia7IZCRwrhZhrgvJEDPmw` | ~1.9 |
| **`if_q3_e`** (stage 1) | `m_given_rule,m_fit_rule,m_leaf` — the mirror seat, E' OFF, at the governing `--tol-dsil 0.0046` | `ap-oi17vzNW5gM1HLeBMSoyR2` | ~1.9 |
| **`if_q3_e_sp`** (stage 2) | `m_fit_rule_sp,m_leaf_sp` — E' ON, at **`--tol-dsil 0.00507228`**, the floor re-derived from stage 1's own `dsil_sp` | `ap-ICOQFJoIE9FQuRQNHwMZdm` | ~1.3 |

Both are `if_q1c_yk`'s design exactly (`cf` world at `rule_seed 6`, `E_R8`, `--setup-render rule`,
practised {0,3,7}, the 140-cycle ladder `1:25:40,2:12:45,3:6:55`, single seed). Stage 2 is a
separate tag because the floor cannot be derived before stage 1 runs; it is merged into stage
1's reduction with `--merge-tag`, which asserts substrate identity first.

### `if_q3_e` stage 1 — clean, 6832 s = 1.90 GPU-h, app `ap-oi17vzNW5gM1HLeBMSoyR2`

**THE SPELLED DEAD ZONE.** Protocol as pinned: null-ABBA, span 1, W 4, era-boundary (advance)
and commit cycles dropped, each arm's series taken from its own first live cycle. Floor is
`sd(N)/sqrt(W)`.

| arm | live@ | n_win | floor on `dsil_ft` (feature-only) | floor on `dsil_sp` (spelled) | mean D (sp) |
|---|---|---|---|---|---|
| `m_given_rule` | c32 | 87 | 0.002265 | 0.002265 | +0.00040 |
| `m_fit_rule` | c24 | 100 | 0.001756 | 0.001756 | +0.00064 |
| `m_leaf` | c43 | 84 | 0.002990 | **0.008599** | +0.00022 |
| **POOLED** | | **271** | **0.00235975** | **0.00507228** | |

The tag's own re-derivation of the FEATURE-ONLY floor is **0.00235975** against the governing
**0.0046** it actually ran under — so the governing floor was ~2x conservative on this world,
which is a fact about the transplant, not about E'. **The spelled floor is 0.00507228**, and
stage 2 was launched at exactly that.

**THE FLOOR IS CARRIED BY ONE ARM, and the reason is mechanical.** `dsil_sp` is *identical* to
`dsil_ft` at every live cycle in two of the three arms:

- `m_given_rule` — by construction: a given renderer never misspells (`e_sp` 0.0000 all run).
- `m_fit_rule` — **because the priced beam only ever runs at PRACTISED registers.** Instances
  are drawn from `practiced_set` = {0, 3, 7}, and the fitted head recovers those cells exactly,
  so once it has fitted (before `dsil` goes live at c24) it never misspells a head-written
  block. Measured: **0.000 written-block spelling error at all three registers over 152,072
  blocks.** Its 0.0269 lifetime figure is the transfer probe's held-out writes, and probes are
  excluded from the meter by `_ENTRY_REC["phase"]` (they take `_fire_only`).
- `m_leaf` — the only arm with a gap: written-block error **0.208 / 0.172 / 0.820** at rho
  0 / 3 / 7 over 649,244 blocks, and `dsil_sp` **0.4304** against `dsil_ft` **0.0324** at the
  end. Its own `dsil_sp` floor alone is 0.0086.

So E' can only bite where the renderer misspells inside the practised set, and on this world
that is `leaf` alone. Stage 2 runs `m_fit_rule_sp` (expected inert by this mechanism — the
control) and `m_leaf_sp` (where the bit can move something).

**Stage-1 trace.** Every arm rode the cap in eras 2 and 3; only era 1 was chosen.

| arm | cyc | commits | advances | era 1 | era 2 | era 3 |
|---|---|---|---|---|---|---|
| `m_given_rule` | 137 | c26 (L2), c44 (L3) | **c37 quiet** (V/tol 0.55), c82 cap, c137 cap | c1-37 | c38-82 | c83-137 |
| `m_fit_rule` | 137 | c18 (L2) | **c37 quiet** (V/tol 0.99), c82 cap, c137 cap | c1-37 | c38-82 | c83-137 |
| `m_leaf` | 140 | c38 (L2), c54 (L3) | c40 cap, c85 cap, c140 cap | c1-40 | c41-85 | c86-140 |

| arm | era | e(end) | succ(end) | e_sp(end) | dsil | dsil_sp | dsil_ft |
|---|---|---|---|---|---|---|---|
| `m_given_rule` | 1 / 2 / 3 | 0.378 / 0.805 / 0.638 | 0.622 / 0.195 / 0.362 | 0.000 | 0.0639 / 0.0280 / 0.0163 | same | same |
| `m_fit_rule` | 1 / 2 / 3 | 0.375 / 0.753 / 0.789 | 0.625 / 0.247 / 0.211 | 0.000 | 0.0367 / 0.0163 / 0.0106 | same | same |
| `m_leaf` | 1 / 2 / 3 | 0.867 / 0.630 / 0.742 | 0.133 / 0.370 / 0.258 | 0.133 / 0.037 / 0.057 | - / 0.0513 / 0.0324 | - / **0.4192** / **0.4304** | - / 0.0513 / 0.0324 |

### `if_q3_f2` — clean, 7080 s = 1.97 GPU-h, app `ap-Fia7IZCRwrhZhrgvJEDPmw`

All three arms ran 140 cycles on the same schedule clock and committed at the same cycles
(L2 c17, L3 c58 / c58 / c57), so a cross-arm read on matched (cycle, slot) is well posed.
`e(end)` 0.7005 / 0.6979 / 0.6120 and `e_sp(end)` 0.0513 / 0.0562 / 0.0000 for
`leaf_s` / `canon_s` / `given_rule_s`.

**What is behind every correlation below**: **24 committed slots** (16 at L2, 8 at L3) x **19
probe cycles** (c24..c140) = **408 cells** per arm; 404 / 406 / 407 of them were actually used
by the beam at least once, and all 408 wrote at least one block.

**Is there variation to select on?** Yes, and it is graded rather than binary.

| arm | per-cell spelling reliability (1 - misspelled/written) | per-slot, pooled over cycles |
|---|---|---|
| `leaf_s` | mean 0.594, sd 0.081, min/p25/med/p75/max 0.339 / 0.546 / 0.595 / 0.653 / 0.833, 363 distinct values | mean 0.591, sd 0.038, range 0.519..0.672 |
| `canon_s` | mean 0.565, sd 0.084, 0.284 / 0.509 / 0.569 / 0.619 / 0.799, 357 distinct | mean 0.568, sd 0.042, range 0.470..0.638 |
| `given_rule_s` | **1.000 exactly, sd 0** (one distinct value) | 1.000 |

Solve-rate-when-used is also graded but far more skewed: `leaf_s` mean 0.106 (median 0.000,
p75 0.191), `canon_s` 0.134 (median 0.100), `given_rule_s` 0.184 (median 0.167).

**The correlations.**

| arm | cells | `pi~solve` | `pi~use` | `pi~solve\|use` | `pi~spellok` |
|---|---|---|---|---|---|
| `leaf_s` | 408 (used 404) | 0.164 | **0.736** | 0.105 | -0.147 |
| `canon_s` | 408 (used 406) | 0.136 | **0.750** | 0.140 | -0.259 |
| `given_rule_s` | 408 (used 407) | 0.138 | **0.728** | 0.093 | undefined (sd 0) |

**Partialling, on the 404 / 406 / 407 cells that were both used and wrote:**

| arm | `pi~spellok` | `\|use` | `\|solve` | `\|use,solve` | `pi~solve` | `\|use` | `\|use,spellok` |
|---|---|---|---|---|---|---|---|
| `leaf_s` | -0.146 | **-0.019** | -0.141 | **-0.017** | 0.164 | 0.105 | 0.104 |
| `canon_s` | -0.261 | -0.258 | -0.251 | **-0.248** | 0.136 | 0.140 | 0.120 |
| `given_rule_s` | - | - | - | - | 0.138 | 0.093 | 0.093 |

**With cycle and level fixed effects absorbed** (the cleaner read — pi's overall scale drifts
across the run and L2/L3 slots differ systematically):

| arm | `pi~solve\|cyc` | `pi~solve\|cyc,use` | `pi~spellok\|cyc` | `pi~spellok\|cyc,use,solve` | `pi~use\|cyc` |
|---|---|---|---|---|---|
| `leaf_s` | 0.198 | 0.122 | -0.209 | **-0.035** | **0.780** |
| `canon_s` | 0.146 | 0.165 | -0.346 | **-0.238** | **0.777** |
| `given_rule_s` | 0.128 | 0.086 | - | - | **0.740** |

**Cross-arm, against `given_rule_s` on matched (cycle, slot), 408 matched cells each:**

| arm | corr(pi) | mean d(pi) | corr(solve) | mean d(spellok) |
|---|---|---|---|---|
| `leaf_s` | 0.884 | -0.0029 | 0.192 | -0.4057 |
| `canon_s` | 0.911 | -0.0023 | 0.171 | -0.4355 |

**Per-slot spelling parity by register** (pooled, * = practised): `leaf_s` 0.201 / 0.228 /
0.857 at rho 0 / 3 / 7 over 212,658 written blocks; `canon_s` 0.000 / 0.348 / 1.000 over
234,520 (canon IS the rule at rho 0, and rho 7 is above every threshold, so it is wrong
everywhere); `given_rule_s` 0.000 / 0.000 / 0.000 over 221,374.

### `if_q3_e_sp` stage 2 — clean, 3430 s = 0.95 GPU-h, app `ap-ICOQFJoIE9FQuRQNHwMZdm`

Merged into `if_q3_e` with `--merge-tag`, which asserted substrate identity first:
**refs identical = True, stale buffer identical = True**.

**THE PAIRS DIFFER IN TWO BITS, NOT ONE, AND THAT IS FORCED BY THE PROTOCOL.** The E'-ON arms
had to run at the re-derived floor (`--tol-dsil 0.00507228` vs the off arms' `0.0046`), so
`perf_e_spell` and `tol_dsil` both move between an arm and its twin. Every claim below is
therefore attributed to one or the other explicitly, and the attribution is decidable because
`dsil_sp` vs `dsil_ft` says whether the BIT could have changed anything at all.

**The twin gate, on the driven `dsil`.** Both pairs are bit-identical up to the parting cycle:

| pair | driven `dsil` parts | `e` / `succ` / `t_cum` over c1..parting | `dsil_sp != dsil_ft` in the ON arm |
|---|---|---|---|
| `m_fit_rule` / `m_fit_rule_sp` | **c38** | identical over c1..c37 (37 cycles), all three | **0 of 117 live cycles** |
| `m_leaf` / `m_leaf_sp` | **c43** | identical over c1..c42 (42 cycles), all three | **20 of 20 live cycles** |

**Actions, and what licensed each.**

| arm | tol | cyc | commits | advances (why, V/tol) |
|---|---|---|---|---|
| `m_given_rule` | 0.0046 | 137 | L2 c26, L3 c44 | c37 **quiet** 0.55, c82 cap −0.24, c137 cap −0.05 |
| `m_fit_rule` | 0.0046 | 137 | L2 c18 | c37 **quiet 0.99**, c82 cap 0.25, c137 cap −0.10 |
| `m_fit_rule_sp` | 0.00507 | 140 | L2 c18 | c40 cap 0.40, c85 cap 0.47, c140 cap 0.12 |
| `m_leaf` | 0.0046 | 140 | L2 c38, L3 c54 | c40 cap, c85 cap 0.07, c140 cap 0.09 |
| `m_leaf_sp` | 0.00507 | 62 | L2 c38 | c40 cap, **c52 quiet 0.76**, **c62 quiet 0.93** |

**Era values.**

| arm | era | span | e(end) | succ | e_sp | dsil | dsil_sp | dsil_ft |
|---|---|---|---|---|---|---|---|---|
| `m_given_rule` | 1/2/3 | c1-37 / c38-82 / c83-137 | 0.378 / 0.805 / 0.638 | 0.622 / 0.195 / 0.362 | 0.000 | 0.0639 / 0.0280 / 0.0163 | same | same |
| `m_fit_rule` | 1/2/3 | c1-37 / c38-82 / c83-137 | 0.375 / 0.753 / 0.789 | 0.625 / 0.247 / 0.211 | 0.000 | 0.0367 / 0.0163 / 0.0106 | same | same |
| `m_fit_rule_sp` | 1/2/3 | c1-40 / c41-85 / c86-140 | 0.372 / 0.760 / 0.737 | 0.628 / 0.240 / 0.263 | 0.000 | 0.0319 / 0.0156 / 0.0087 | same | same |
| `m_leaf` | 1/2/3 | c1-40 / c41-85 / c86-140 | 0.867 / 0.630 / 0.742 | 0.133 / 0.370 / 0.258 | 0.133 / 0.037 / 0.057 | – / 0.0513 / 0.0324 | – / **0.4192** / **0.4304** | – / 0.0513 / 0.0324 |
| `m_leaf_sp` | 1/2/3 | c1-40 / c41-52 / c53-62 | 0.867 / 0.844 / 0.969 | 0.133 / 0.156 / 0.031 | 0.133 / 0.030 / 0.033 | – / **0.4320** / **0.4116** | – / 0.4320 / 0.4116 | – / 0.0967 / 0.0580 |

**δ_perf, magnitude and sign, per (slot, cycle) cell.** The shadow column is the same statistic
on the currency the arm is NOT driving on.

| arm | E′ | cells | driven mean δ | frac δ>0 | shadow mean δ | frac δ>0 |
|---|---|---|---|---|---|---|
| `m_given_rule` | off | 2375 | −0.00562 | 0.492 | −0.00562 | 0.492 |
| `m_fit_rule` | off | 1808 | −0.01152 | 0.319 | −0.01152 | 0.319 |
| `m_fit_rule_sp` | ON | 1856 | −0.01134 | 0.320 | −0.01134 | 0.320 |
| `m_leaf` | off | 2225 | −0.00684 | 0.434 | **−0.00010** | **0.503** |
| `m_leaf_sp` | ON | 318 | −0.00450 | 0.440 | −0.00941 | 0.421 |

δ_perf is negative on average in every arm. In `m_leaf` the spelling-charged shadow is
essentially **zero-mean and sign-balanced** (−0.0001, 50.3 % positive) where the driven
feature-only δ is −0.0068 / 43.4 % — the spelled benchmark tracks a level the executor is not
improving on, so the residual has no drift.

**The 2x2 (executed-as-intended x solved), lifetime, with the spelling column.**

| arm | int_solved | int_failed | bad_solved | bad_failed | no-exec | intent frac | written-block spelling error |
|---|---|---|---|---|---|---|---|
| `m_given_rule` | 18,242 | 75,537 | 1,483 | 8,675 | 36,351 | 0.902 | **0.0000** (0 / 43,565) |
| `m_fit_rule` | 18,373 | 81,628 | 1,303 | 8,294 | 30,690 | 0.912 | **0.0000** (0 / 39,241) |
| `m_fit_rule_sp` | 21,080 | 82,054 | 1,257 | 8,136 | 30,833 | 0.917 | **0.0000** (0 / 39,534) |
| `m_leaf` | 11,601 | 63,826 | 1,794 | 17,030 | 49,109 | 0.800 | **0.4224** (18,898 / 44,735) |
| `m_leaf_sp` | 1,818 | 11,886 | 365 | 4,130 | 45,289 | 0.753 | **0.4170** (14,718 / 35,298) |

`n_unbound = 0` in all five. `leaf` hit rate 31,321 / 1,319 miss (`m_leaf`), 31,946 / 694
(`m_leaf_sp`).

**Lifetime spend and solves.**

| arm | cyc | t_cum | solves (life) | succ(end) | e(end) |
|---|---|---|---|---|---|
| `m_given_rule` | 137 | 35,311,411 | 37,472 | 0.362 | 0.638 |
| `m_fit_rule` | 137 | 33,165,734 | 33,426 | 0.211 | 0.789 |
| `m_fit_rule_sp` | 140 | 33,830,765 | 36,189 | 0.263 | 0.737 |
| `m_leaf` | 140 | 35,971,450 | 40,685 | 0.258 | 0.742 |
| `m_leaf_sp` | **62** | **15,231,994** | 28,884 | 0.031 | 0.969 |

### The E′ read

**The `fit_rule` pair: the bit is inert, and the divergence is entirely the floor — confirmed,
not refuted.** In `m_fit_rule_sp`, `dsil_sp` equals `dsil_ft` on **0 of 117** live cycles and
the written-block spelling error is **0.0000 over 39,534 blocks**, so `e_spell == e_feat` on
every head execution and `perf_e_spell` could not have changed a single number the arm read.
The pair nevertheless parts at c38, and the mechanism is visible in one line: `m_fit_rule`'s
era-1 quiet advance fired at c37 at **V/tol = 0.99** — on the edge — and at the higher floor
the same V is 0.40, so `m_fit_rule_sp` rode the cap to c40 instead. Everything downstream (3
extra cycles, +2,763 solves, e(end) 0.789 → 0.737) follows from that one displaced advance.
This is the practised-register mechanism as predicted: the priced beam draws only from
{0, 3, 7}, where the fitted table is exact.

**The `leaf` pair: the bit moved the advance clock, and the floor moved the other way.** The
driven `dsil` goes from 0.032 (feature-only) to 0.41–0.43 (spelling-charged), `dsil_sp` differs
from `dsil_ft` on 20 of 20 live cycles, and the pacer's behaviour inverts: `m_leaf` never fired
a quiet advance in 140 cycles (3 of 3 capped, best V/tol 0.09), while `m_leaf_sp` fired **two**
(c52 at V/tol 0.76, c62 at 0.93) and finished the ladder in **62 cycles at 15.2M g against
140 cycles at 36.0M**. The floor cannot explain that direction: it moved **up** (0.0046 →
0.00507), which makes a quiet advance strictly **harder**, so the extra quiet advances are
attributable to the spelling-charged series and not to the tolerance. What the shortened clock
bought is stated with it: era 3 lasted 10 cycles instead of 55 and e(end) is 0.969 against
0.742, i.e. the arm advanced off eras it had not learned.

### How A1's quiet statistic reads the spelled series as quiet (CPU, `if_q3_e` logs only)

Replayed on **`m_leaf`'s own logged panel**, both series, same cycles, same arm, same re-arm
schedule (era starts c1/c41/c86; commits c38, c54; advances c40, c85, c140 — both policies
re-arm on every action, per `_acted_all`). So **no floor and no bit differs between the two
columns**; only which error the benchmark was fed.

**The statistic, as `maestro/policy.py::QuietPolicy.step` actually computes it:**

```
pts   = the last W+1 = 5 buffer values, stride span = 1, oldest -> newest
u[k]  = pts[k] - pts[k+1]                    ( > 0 == the error FELL )
D     = mean(u)                              RAW SLOPE, in the series' OWN units
N     = mean(u | C) - mean(u | K)            the null contrast — LOGGED, never driven
V     = EWMA(D, alpha = 0.5), reset on every action and at every era start
moved = latches True the first time V > tol  <-- the "positive THEN flat" precondition
quiet = (V is not None) and moved and n_since > burn and V <= tol
```

**There is no normalisation — not by level, not by the series' own scale.** `D` is a mean of
first differences of the raw series, and `tol` is an absolute number.

**Per era, both series, both floors.**

| series | era | n defined | mean D | sd(D) | max V | max V/tol @0.0046 | @0.00507 | mean N | `moved` latched | licensed at |
|---|---|---|---|---|---|---|---|---|---|---|
| `dsil_ft` | 2 (c41-85) | 43 | +0.00353 | 0.00552 | +0.01681 | 3.65 | 3.31 | +0.00033 | **yes, c47** | **never** |
| `dsil_ft` | 3 (c86-140) | 55 | +0.00020 | 0.00192 | +0.00306 | 0.67 | 0.60 | +0.00015 | **NO** | **never** |
| `dsil_sp` | 2 | 43 | +0.00206 | 0.00644 | +0.02149 | 4.67 | 4.24 | +0.00263 | yes, c47 | **c52**, then c77-85 |
| `dsil_sp` | 3 | 55 | −0.00089 | 0.00558 | +0.00860 | 1.87 | 1.70 | +0.00120 | yes, c119 | c120-129, c133-140 |

At the higher floor `dsil_sp` licenses at c52/c53 and c133-140; `dsil_ft` still never. **The
replay's first firing, c52, is exactly the cycle `m_leaf_sp` actually fired its era-2 quiet
advance** — reproduced from the off arm's own shadow series.

**What blocks the feature-only series is different in each era, and neither is the floor:**

- **era 2**: both series latch `moved` at c47. `dsil_ft`'s V then never returns to the dead
  zone (min V after the latch **+0.00802** > tol on all 35 cycles) — it is still falling
  monotonically. `dsil_sp`'s V does return (min **−0.00910**, ≤ tol on 10 cycles).
- **era 3**: `dsil_ft` **never latches `moved` at all** (max V/tol 0.67). Its increments are too
  small for the fixed dead zone, so however flat it becomes it can never be licensed — the
  precondition is unreachable. `dsil_sp` latches at c119 and goes quiet at c120.

**The decisive test: the offset does NOT do it.** Replaying `dsil_ft + 0.40` reproduces
`dsil_ft` **to every digit** at both floors (same mean D, same sd(D), same max V, same latch,
still never licensed). `D` is a mean of first differences, so a constant offset cancels exactly.
**The mechanism is not level-normalisation.**

**What in the spelled series' shape produces the quiet reading — its ABSOLUTE increment scale
against a fixed dead zone.**

| era | series | mean level | sd(level) | **sd(diff)** | mean abs diff | ac1(level) | ac1(diff) | sd(diff)/level |
|---|---|---|---|---|---|---|---|---|
| 2 | `dsil_ft` | 0.0891 | 0.0383 | **0.00705** | 0.00565 | +0.988 | +0.542 | 0.079 |
| 2 | `dsil_sp` | 0.4311 | 0.0245 | **0.01697** | 0.01280 | +0.728 | −0.168 | 0.039 |
| 3 | `dsil_ft` | 0.0413 | 0.0071 | **0.00575** | 0.00424 | +0.669 | −0.322 | 0.139 |
| 3 | `dsil_sp` | 0.4539 | 0.0267 | **0.01651** | 0.01431 | +0.807 | −0.332 | 0.036 |

Ratios spelled / feature-only: level x4.8 (era 2) and x11.0 (era 3); **sd(diff) x2.41 and
x2.87**; mean abs diff x2.27 and x3.37. corr(`dsil_ft`, `dsil_sp`) is +0.788 in era 2 and
**−0.039** in era 3 — in era 3 the two series are essentially unrelated in level; their
first differences correlate +0.268 / +0.377.

Where the dead zone sits relative to each series' own movement:

| era | series | tol / sd(diff) @0.0046 | @0.00507 | frac abs(diff) > tol |
|---|---|---|---|---|
| 2 | `dsil_ft` | 0.65 | 0.72 | 0.429 |
| 2 | `dsil_sp` | **0.27** | **0.30** | 0.667 |
| 3 | `dsil_ft` | 0.80 | 0.88 | 0.370 / 0.315 |
| 3 | `dsil_sp` | **0.28** | **0.31** | 0.870 |

**The finding is about the statistic.** A1's `V` is a raw slope in the series' own units
compared against a fixed absolute dead zone, behind a positive-then-flat latch. Charging for
spelling raises the series' absolute amplitude roughly threefold in increments (and 5-11x in
level) **without making it relatively noisier** — by `sd(diff)/level` the spelled series is
actually the *smoother* of the two (0.036-0.039 vs 0.079-0.139). That absolute rescaling is
what (i) lets the `moved` latch fire at all in era 3 and (ii) lets the EWMA swing below the same
fixed `tol` afterwards. So `m_leaf_sp`'s two quiet advances are not the spelled executor being
"more converged"; they are the same dead zone applied to a series that is an order of magnitude
larger, on a statistic that never divides by scale.

## Q3 reduction

`analyze_inflection.py` gains three sections, run after the four `if_q1`-era ones:

- **§5I `slot_trust`** — F2. Per arm: `pi~solve` (mass against the slot's solve rate when
  used), `pi~use` (mass against how often the beam took it), **`pi~solve|use`** (the first with
  the habit term partialled out) and `pi~spellok`. Then the cross-arm read against
  `given_rule_s` on matched (cycle, slot), and the per-slot spelling parity by register.
- **§6I `spelled_floor`** — null-ABBA on `dsil_sp`, per arm and **pooled over the E'-OFF arms**,
  the way `tutti` pooled its four. Prints the `--tol-dsil` to pass stage 2.
- **§6I.b `e_prime`** — the pair table (both currencies, commits, advances, the clock) and the
  in-tag twin gate, read on the **driven `dsil`** series.

`PANEL_KEYS` gains `dsil_sp` and `dsil_ft`, so §6's own floor table carries them too.

## Open at halt

- **`read_acc` under the rule: MEASURED in `if_smoke`** at production `reader_steps` — 1.000 at
  every register, held-out included (one cell at 0.974). The SPEC's Q0 item is closed for the
  reader; whether it stays closed at Q1's scale is read again from `if_q1`'s probes.
- `analyze_inflection.py` forks `analyze_tutti.py` and adds **§1I** (the transfer probe, register x level), **§2I** (identifiability: recovered K vs true K, `theta_hat`, `acc_heldout`), **§3I** (spelling beside meaning, and the `blk_render` ledger) and **§4I** (`leaf`'s side table and `read_acc` by register). The donors' sections are kept and run first.
- `leaf` and `fit_rule` raise `NotImplementedError` at `_renderer`; see
  [`DESIGN.md`](DESIGN.md) §9 for exactly what each still needs.
- `QUEUE.md` / `ROADMAP_PROGRESS.md` rows are **not** written (the orchestrator commits).

## Children

None yet. `leaf`'s unit representation is expected to become a sibling module
(`leafunits.py`) rather than an edit — see [`DESIGN.md`](DESIGN.md) §9.

[^private]: Not mirrored: this link points to a document in the private lab repo (the roadmap, the queue, an unrun spec, reading notes, or a conversation). See the top-level README for what is held back and why.
