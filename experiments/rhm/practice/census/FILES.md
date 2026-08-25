# FILES — `census` (can an endogenous gauge buy the coverage the certificate can't see?)

Machinery record. The writeup is [`README.md`](README.md) — the single super-writeup for the
census + forensics + assay unit (post-discussion, 2026-08-25); this file is the census node's
per-file, per-gate, per-decision reference.

**Up**: [`SPEC.md`](SPEC.md) (the record of what was asked) · parent arc
[`../README.md`](../README.md) · direct parent [`../spiral/`](../spiral/FILES.md) (findings 5
and 6 are the two measured facts this round connects) · gauge donors
[`../teacher_slot/`](../teacher_slot/README.md), [`../recital/`](../recital/README.md).

## Status

**Phase 0 complete (no GPU spend). Phase 1 built and launched** — R1–R7 adopted, with four
coordinator modifications recorded below.

## Code files

| file | purpose |
|---|---|
| `phase0_replay.py` | **Phase 0**: replays every candidate gauge offline over `spiral/figures/sp_s0/`'s logs — all four earning arms, both levels, over a (window × threshold × support) grid. Pure local analysis; reads the fetched spiral logs and writes `figures/phase0/`. Contains the observability check that had to come first (`t4_observability`), the saturation detectors (`replay_gauge`), the extension-opportunity series (`replay_extend`), and the four structural analyses the replay turned up: `replay_validity`, `ratchet_ceiling`, `extension_windows`, `perfect_l2_control` |
| `census.py` | **Phase 1's Modal app**. Forks `../spiral/spiral.py` verbatim with four `# [census]` insertions: the G-Y instrument, commit-then-extend, the L2 conjunction gate, and the yoked control. Entrypoints `census_run`, `preflight`, `fidelity_smoke`, `gy_soundness` |
| `forensics.py` | **Step 1 forensics** (free, no GPU): five checks over `cs_s0`'s logs asking why 2.5x coverage closed no bracket — the forced-window check (from the code, then log-confirmed), the true/junk composition of every admission, the e-trajectory around each admitting event, the cycle-aligned era-4 decomposition, and the lifetime traces that separate `given_route` from the earning arms before any table exists. Writes `figures/forensics/` |
| `analyze_census.py` | Phase 1 reduction: both halves of G-F, the twin gates, the coverage table, the eras-4–5 bracket, the timing price, the grader bill, the G-Y readout, plant guard and cost. `--figures` writes `fig1_competence.png`, `fig2_coverage.png` |
| `launch_detached.py` | the arc's session-isolated launcher, retargeted |

## Phase 1 — the four mechanisms, and where each lives

| mechanism | where | note |
|---|---|---|
| **G-Y instrument** | `run_arm`, at miner construction and in the mining block | an unpriced, read-only `Miner(4, s)` in **every** arm, observing the era's own level-4 node in **every** era. `gy_level` is deliberately above `max_macro_level`: that constant caps what may be *committed*, not what may be *observed*. Asserted sound by `gy_soundness()` against hand-enumerated counts before any paid setup |
| **commit-then-extend** | `run_arm` block (g3), scoped by `spec["extend"]` | the recert loop widened to **every committed level in every era** (Phase 0 §5: the inherited loop runs in eras 2–3 and not at all in eras 4–5, where the payoff is measured). Selection, not append: each candidate is auditioned on fresh held-out instances of the era's own cell and admitted only if it does not raise the table's audition error. Priced at the commit audition's rate |
| **L2 conjunction gate** | `run_arm`, the `delta_prov` branch, scoped by `cfg["gate_level"]` | commit iff the certificate has fired **and** the G-A gauge is quiet (at-support series, W=12, θ=0). A conjunction, so the op is strictly "hold longer" and no setting can accelerate a commit |
| **yoked control** | `run_arm`, the `yoked` branch + `census_run`'s arm loop | the commit cycle is `census_gate`'s **measured** one, injected at runtime; `summary["yoke"]["case"]` records whether the gate fired or fell back to the boundary |

`extend_candidates` and `refresh_upper` are the two module helpers extension needs: candidates
are recomputed against the *current* lower table so indices are correct by construction, and
because extension only ever **appends**, the level above keeps its child indices and its
flattened expansion when its lower table grows.

## Phase 1 gates

| gate | what it asserts | how |
|---|---|---|
| **G-F, full scale** | this tag's in-tag `spiral_route` ≡ `sp_s0`'s over the 32 cycles the two ladders share | cross-tag diff in `analyze_census.py`. Free: era 1 is 48 cycles here against 32 there, and `ec` reaches only the loop bound, `at_boundary` (c18's certified commit precedes both) and the era-end probe trigger (fires at c32 either way; probes consume no RNG) |
| **G-F, smoke scale** | with the census ops off, `census.run_arm` ≡ `spiral.run_arm` for the whole loop shape | `fidelity_smoke`, in-process against the imported donor with a donor self-replay control. Smoke scale on purpose: a full-scale in-process replay is three 116-cycle depth-6 arms plus setup, ~1.5 GPU-h, to re-derive what the in-tag anchor gives free |
| **twins** | each treated arm ≡ the anchor until its own single treatment | `analyze_census.py` §0 |
| **G-Y soundness** | span, hand-enumerated counts at every support threshold, and node indexing over the real ladder | `gy_soundness()`, run in `preflight` **and** at the top of `census_run` before the paid setup, and stored in `setup.json` |
| **scoping** | the anchor and the gate arms carry **no** extension events | asserted in `preflight` |

## Phase 0 outputs

`figures/phase0/`:

| file | what |
|---|---|
| `DECISION.md` | **the decision document** — gauge-by-gauge verdicts, the four structural findings, and seven numbered recommendations with numbers attached. This is what the orchestrator reads before Phase 1 is sized |
| `report.txt` | the full replay printout (9 sections, all grids) |
| `phase0.json` | machine-readable: every grid cell, every recert opportunity, every derived table |

## Phase 0 headline numbers

| | |
|---|---|
| G-Y (next-level yield) | **not observable** — no level-4 miner exists in `sp_s0` (miners span levels 2–3), and per-key counts carry no adjacency, so the T4 stream cannot be reconstructed offline |
| G-C vs G-A at L2 | **bit-identical in all four arms** (max gap 0) — one gauge, not two; they separate only at L3 |
| replay-valid window | cert cycle == commit cycle in **8 of 8** (arm × level) ⇒ "gate-later" is exactly "post-commit", where the log stops being a counterfactual |
| L3 ratchet ceiling | frozen-L2 recall `r` caps buildable L3 at ≈ `r²`: **0.617** (enum arms) / **0.327** (routed arms) |
| perfect-L2 control | `given` mines L3 for 84 cycles over a complete L2, never freezes, ends at recall **0.286** / precision **0.727** — the empirical ceiling for any L3 gate in a ladder this long |
| L2 coverage on offer | frozen 0.571 → live **0.929** recall in the routed arms (**+0.357**) |
| extension windows | 11 at L2 (era 2), 2 at L3 (era 3), **none in eras 4–5** — where the payoff is measured |

## Coordinator modifications to R1–R7, on the record

1. **Six arms**: `spiral_route` (anchor), `census_gate`, `yoked_delay`, `census_extend`,
   `given_route` (**new** — true tables + π only, because the round is routing-only per the
   finding-9 quarantine, so the ceiling must be routing-only too), `given_native` (kept for
   cross-tag bracket continuity with `sp_s0`'s eras-4–5 numbers).
2. **The widened recert is scoped to the extension arm only.** The anchor must stay a
   bit-for-bit `sp_s0` replay and the gate arms must differ from it by exactly one thing each.
3. **`yoked_delay` takes `census_gate`'s measured commit cycle**, so `census_gate` runs earlier
   in the arm order and the cycle is injected at runtime rather than hard-coded.
4. **G-Y soundness is part of the record**, not an assumption — the spiral README promised the
   readout conditionally on the instrument being sound.

## Step 1 forensics — headline numbers

| check | verdict | number |
|---|---|---|
| (1) extensions trigger a forced window | **REFUTED** | `k_eff`/width move at **0 of 12** extensions vs **8 of 8** commits |
| (2) truth composition of admissions | measured | 22 admitted = **10 true / 12 junk** (54.5%); L2 5 = 3/2 (40%), L3 17 = 7/10 (58.8%) |
| (2) entry-level pi split | **NOT COMPUTABLE** | pi's action space is (level, node) slots; entries of a level share one action |
| (3) e-excursions at admitting events | **REFUTED** as an extension effect | 1 of 12 exceeds 2x the arm's own mean \|Δe\|, and it is the era 2→3 boundary |
| (4) era-4 deficit concentrated after c105 | **REFUTED** | +0.0443 before, +0.0462 after; already +0.039 at c101, four cycles before the event |
| (5) given_route's deep-era edge predates earning | **CONFIRMED** | **80.3%** of its final era-4 advantage is present at c16, before any arm has committed (first commit c18) |

## Reproduce

```bash
cd experiments/                     # Phase 0: no Modal, no GPU
python3 rhm/practice/census/phase0_replay.py

# Phase 1                            # MODAL_PROFILE=chromatic
modal run rhm/practice/census/census.py::preflight
modal run rhm/practice/census/census.py::fidelity_smoke
modal run rhm/practice/census/census.py::census_run --quick --tag smokeC

python3 rhm/practice/census/launch_detached.py --fn census_run --tag cs_s0 \
    --eras "1:25:48,2:12:40,3:6:12,4:3:9,5:1:7" \
    --arms "spiral_route,census_gate,yoked_delay,census_extend,given_route,given_native" \
    --budget 8 --g-budget 482 --n-corrupt 1 --mine-cap 8 --gy-level 4 \
    --gate-win 12 --gate-theta 0 --extend-cap 8 --extend-tol 0.0 \
    --span-min-hold 128 --collect-task-matched --tm-episodes 8192 \
    --probe-every 8 --probe-widths "1,2,4" --n-rt 384 --n-score 256 --n-aud 192 \
    --recert-every 5 --sil-cv 0.15 --lp-min-drop 0.10 --seed 0

python3 rhm/practice/census/analyze_census.py --tag cs_s0 --fetch --figures

# Step 1 forensics                   # no Modal, no GPU
python3 rhm/practice/census/forensics.py
```

Requires `../spiral/figures/sp_s0/` to be present locally (fetch with
`python3 rhm/practice/spiral/analyze_spiral.py --tag sp_s0 --fetch`).

## Inherited, not copied

`../spiral/` (the substrate and the logs Phase 0 replays), `../recital/` (the admission-rate
pacer this round aims at the frontier), `../teacher_slot/` (the one-level-up gauge idea and
`endo_yield`'s label-free currency). Nothing in those folders is modified.
