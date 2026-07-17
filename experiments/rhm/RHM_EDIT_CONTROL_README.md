# RHM as a control task: editing, off-manifold belief-gaming, and the on-manifold generative fix

**Status**: **WIP.** Part 1 (editing-as-control) and Part 2 (on-manifold generative planning) done, single seed per setting, m ∈ {2,3,4}. Part 3 (planning in *latents* — the stated payoff) not yet built. Results discussed but not yet crystallized as beliefs.
**Date**: 2026-07-11
**Scripts**: [`rhm_edit_control.py`](rhm_edit_control.py) (Part 1: editing arity battery + reveal contrast + ground-truth judge), [`rhm_generative_planner.py`](rhm_generative_planner.py) (Part 2: generator-defined on-manifold moves + cerebellar self-consistency veto). New grammar utilities in [`rhm_data.py`](rhm_data.py): `generate_rules_invertible`, `build_inverse_maps`, `parse_leaves`.
**Origin / sibling**: this is the execution of [ACTIVE_RHM_README.md](ACTIVE_RHM_README.md) **next-step #4** ("to see the internalization/self-model phenomenon in the RHM *domain* it must be a genuine control task — an agent that edits/writes tokens toward a target root, not information-gathering"). The reaching-control analog it is measured against: [a2a_forward/reaching/REACHING_INTERNAL_README.md](../a2a_forward/reaching/REACHING_INTERNAL_README.md), [a2a_forward/reaching/ACTIVE_VISION_README.md](../a2a_forward/reaching/ACTIVE_VISION_README.md).

---

## The question

The active-**query** RHM line (ACTIVE_RHM_README) established that active inference on RHM is "inference in disguise": a query does not change the world, it only sharpens a belief about a fixed hidden root, so a mean-Δ forward model is a **planning null** (its payoff lives in the *variance* over hidden content, which the mean discards) and internalizing the forecast has no headroom (act ≈ plan). That arc's own conclusion was that to exhibit the reaching-style self-model phenomenon, RHM must become a genuine **control** task where *acting ≠ planning* — an agent that **edits** tokens toward a target root, so a forward model has to simulate an edit's downstream cascade.

This README covers building that task and what it revealed: the dynamics **are** plannable (reversing the query null), but doing so surfaced a *second*, independent obstacle — **belief faithfulness off the data manifold** — which turns out to be the real crux, and which a learned generator largely fixes.

---

## Setup shared by both parts

- **Grammar**: `generate_rules_invertible` (new) samples **collision-free** rules — at each level the `m·v` produced s-tuples are all distinct, so the bottom-up parse (`parse_leaves`, new) is **unique and exact**. This gives well-defined ground truth for any (on-grammar) sequence, closing the "the controller defines its own target" loophole flagged in ACTIVE_RHM. Canonical setting: `v=8, s=2, L=4` (leaf length 16, `n_blocks=8`), m ∈ {2,3,4}, single rule seed.
- **Action primitive — block edit**: set leaf block `j` to the canonical valid tuple of level-1 feature `g` (action = `(j,g)`, 64 total). Keeps the leaf→level-1 layer on-grammar always; *upper*-level validity is the planning challenge. (Single-leaf edits were rejected: a 1-leaf perturbation leaves only ~10% of sequences on-grammar, so it drowns the task in off-grammar noise.)
- **Task — corrupt-and-repair**: start from a valid `r*` expansion, corrupt `n_corrupt` random blocks to random features, and edit back toward `r*`. Guaranteed solvable in `n_corrupt` edits (the ground-truth oracle hits 1.0), with a dense feature-match metric. Target root `r*` is given.
- **Judge / belief**: a frozen `BeliefController` (the ACTIVE_RHM controller, trained full-observation-weighted here so it is a near-perfect parser on valid sequences: full-seq root acc **1.000** at m=2/3, **0.772** at m=4). Belief `b = C.state(x)`; the planner scores edits by the controller's `log P(root=r*)`.
- **Two oracles**: `controller-oracle` (greedy-best edit by the controller's *own* belief — the ceiling for belief-space planning) and `gt_oracle` (greedy-best by the *true grammar* — the true achievable ceiling, always 1.0 here).

---

## Part 1 — editing IS plannable in belief space, but the belief is gameable off-manifold (`rhm_edit_control.py`)

Two layers, both m-robust.

### Layer 1 (belief space): the arity impossibility *and its usability* port to control

Fraction of the random→controller-oracle gap closed, scoring by the controller's belief `P(r*)`:

| m | edit arity-1 `F(b)` | **edit arity-2 `F(b,edit)`** | edit arity-2, shuffled-action | reveal arity-2 (same mean-Δ FM) |
|---|---|---|---|---|
| 2 | +0.00 | **+0.98** | +0.17 | +0.09 |
| 3 | +0.00 | **+0.93** | +0.04 | +0.06 |
| 4 | +0.00 | **+0.71** | +0.10 | +0.12 |

`cmd_cos` (edit-conditional structure of the FM): arity-1 = **0.000** at every m (edit-blind — the arity impossibility ports); arity-2 = 0.76 / 0.70 / 0.57. **The identical mean-Δ arity-2 FM is a decisive planner on editing (+0.71–0.98) and a near-null on the epistemic reveal task (+0.06–0.12), on the same controller and grammar.** This is the mirror image of the query null: editing's consequence is deterministic-given-the-observed-state (you know what you're writing), so the mean *is* the exact consequence; revealing's is variance-in-hidden-content, so the mean discards it. Degrades gracefully with m (tracking the controller's own parsing quality), no collapse.

### Layer 2 (ground truth): belief-planning controls the belief, not the world

Grading the *same runs* by the true grammar instead of the controller:

| m | arity-2 gt-success | controller-oracle gt-success | gt_oracle | arity-2 gt-gap-closed |
|---|---|---|---|---|
| 2 | 0.003 | 0.012 | 1.000 | +0.00 |
| 3 | 0.010 | 0.016 | 1.000 | +0.01 |
| 4 | 0.005 | 0.015 | 1.000 | +0.00 |

Belief-space planning drives `P(r*)` → ~1.0 while **~99% of the final sequences are off-grammar and true success is ~0**, at every m. The `gt_oracle` reaches 1.0, so the task is solvable — the failure is entirely that the planner reaches configs the controller *mislabels* as `r*`.

### The mechanism — a second axis beyond act≠plan

The controller is a *perfect parser on valid sequences* but **uncalibrated off the valid manifold** (it was only ever trained where revealing-true-tokens keeps you in-distribution). Editing a root *requires* passing through off-grammar "word-salad" intermediates (you break the parse to rebuild it), and there the belief is exploitable. So:

> Reaching achieves *true* control because its state is always on the reachable manifold → its learned belief is **faithful**. Editing-toward-a-root is plannable-in-belief but **forces off-manifold excursions**, where a root-classifier belief is **gameable** → no true control. **The bottleneck is belief faithfulness, not planning or arity** — a distinct axis from the control-vs-inference (act≠plan) axis. (It is also why ACTIVE_RHM could safely use the controller as its own judge: revealing never leaves the manifold; editing is the first task that does.)

*(A strict off-grammar "verifier" head — extra INVALID class — was built and rejected: root validity is a global property, so its signal is ~0 until the final fix — too sparse for a myopic planner. The dense lenient controller + a ground-truth check is the honest instrument.)*

---

## Part 2 — the on-manifold fix: generator-defined moves + a cerebellar veto (`rhm_generative_planner.py`)

Jasper's proposal: don't robustify the belief off-manifold (an unbounded space); **constrain the actions to stay (softly) on-manifold** by drawing moves from a learned **generator** — the cortex proposing on-manifold moves, the controller judging only near-manifold sequences it understands, and *direction* toward `r*` coming from the planner's **search** (select proposals that raise `P(r*)`), not from conditioning the generator on `r*` (which would let one big regeneration solve it and dissolve planning).

- **Generator `G`** (`BlockInfiller`): a block-level masked infiller — mask a region, predict each masked block's level-1 feature from the rest (+ optionally `r*`), render via canonical tuples. Leaf-valid by construction; *upper*-grammar consistency is **learned, not enforced** — so proposals are **soft** on-manifold (Jasper's "not perfectly on-manifold, but the divergence is structured and useful" — that residual is exactly the cerebellar signal below).
- **Move**: pick a region (here single blocks), regenerate it via `G`. **Cerebellar self-consistency veto**: score each proposal by `log P(r*)` **plus** `G`'s leave-one-out self-consistency (mean log-prob it assigns to each block given the rest) — high = a near-fixed-point of `G` = on-manifold; low = `G` is surprised = off-manifold. This is `G`'s own residual/novelty signal used as an on-manifold veto (cortex proposes, cerebellum calibrates).

### The arc (fraction of the random→gt_oracle **true-success** gap closed — learned models only, no grammar access)

| planner | m=2 | m=3 | m=4 |
|---|---|---|---|
| raw controller-oracle *(games the belief)* | +0.01 | +0.02 | +0.01 |
| generator moves (unconditional) | +0.38 | +0.35 | +0.19 |
| generator moves (r\*-conditioned) | +0.44 | +0.48 | +0.20 |
| **generator + cerebellar veto (uncond.)** | **+0.65** | +0.52 | +0.36 |
| **generator + cerebellar veto (r\*-cond.)** | +0.66 | **+0.71** | **+0.43** |

Full m=2 detail (true success / on-manifold rate): raw controller-oracle 0.006 / 0.006 → generator 0.39 / 0.39 → **generator+veto 0.65 / 0.67**; gt_oracle 1.0 / 1.0.

### What Part 2 establishes

1. **The gaming is fixable, with only learned models.** Just changing the action space from raw edits to generator proposals takes true success **0.006 → ~0.4** (60×); the cerebellar veto adds another **~0.4 → ~0.65**. The first true-control positive on RHM editing.
2. **`gt_success ≈ gt_valid` at every row** — whenever the planner lands on-manifold, it lands on the *right root*. So the residual gap is purely **off-manifold drift**, not wrong-root; the generator (soft) leaves some residue and greedy myopia + a 6-move budget leave some more. This is what a faithful latent value + amortized lookahead (Part 3) should recover.
3. **Planning does not dissolve.** Once the veto is in, r\*-conditioning barely beats unconditional at m=2 (0.66 vs 0.65) — the *search* does the steering. Conditioning helps more as m rises (m=3: +0.52 → +0.71) because search-alone steering weakens with more synonyms.
4. **m=4 is weaker (+0.43 best)** but largely because the *judge itself* is degraded there (controller full-seq acc 0.77; the gt_oracle scores only 0.77 in the controller's eyes vs 1.0 in truth) — a capacity limit, not a mechanism failure.

---

## What this arc establishes (so far)

1. **The arity impossibility and its *usability* both port to control.** A mean-Δ arity-2 FM that was a null on epistemic queries is a decisive planner on edits (belief-space gap closed 0.71–0.98 vs 0.06–0.12), because an edit's consequence is deterministic-given-observed-state, not variance-in-hidden-content.
2. **Belief faithfulness off-manifold is a second, independent axis of controllability**, orthogonal to act≠plan. Editing is a control task (act≠plan) yet fails to yield *true* control until the off-manifold gaming is addressed — an obstacle reaching never hit because its states are always on-manifold.
3. **Constraining actions to a learned generator's (soft) manifold + a cerebellar self-consistency veto converts total gaming (gt≈0) into majority true control (gt≈0.65)**, using only learned models — a concrete cortex-proposes / cerebellum-calibrates instantiation.

---

## What this does NOT show / caveats

- **Single seed per setting**; m=4 is judge-capacity-limited; the residual third of the gap (off-manifold drift + greedy myopia + budget) is unclosed.
- All Part-2 planning is **token-space**: each move is materialized to tokens and re-encoded (the "CoT-like" regime — the model never plans in its own latents). Whether **planning in latents** helps is exactly Part 3, untested.
- The self-consistency veto uses `bottom_map` to name the observed feature of each (valid) block — a decode of observable tokens, **not** target/ground-truth leakage; the veto is `G`'s own surprise on the current config.
- Internalization / self-model reorganization (the original REACHING_INTERNAL question) has **not** been run in this domain yet — Part 1 shows editing clears the *first* prerequisite (act≠plan, plannable dynamics); the faithfulness obstacle had to be understood first.

---

## Next steps

1. **Part 3 — plan in latents (the payoff).** A cerebellar FM predicting a move's *latent* consequence `z→z'` without decoding (amortized short-lookahead, attacking the greedy-myopia residual) + a **faithful value head** grounded in true progress (attacking the drift residual). Head-to-head **latent vs token-space** planning is the direct test of the CoT-vs-latent-planning distinction; the FM residual is the structured-divergence signal.
2. **Close the token-space residual** as a control: more budget / stronger veto / de-confound m=4 (a stronger controller), to see the ceiling of on-manifold-moves + veto before adding latents.
3. **Seeds** on the headline numbers.
4. **Then** the internalization question in-domain: co-train an endogenous edit-planner and ask whether it reorganizes the editor toward plannability, now that both prerequisites (act≠plan, on-manifold faithfulness) are met.

---

## Reproduction

```bash
cd experiments/
# Part 1: editing-as-control arity battery + reveal contrast + ground-truth judge
modal run --detach rhm/rhm_edit_control.py::edit_control --m 2   # (and --m 3, --m 4)
# Part 2: on-manifold generator moves + cerebellar self-consistency veto
modal run --detach rhm/rhm_generative_planner.py::generative_planner --m 2  # (and 3, 4)
```

Both take `--quick` for fast smoke tests. Results JSON on the `rhm-scaling-data` volume under `rhm_edit_control/` and `rhm_generative_planner/`.
