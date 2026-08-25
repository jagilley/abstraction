# FILES — `spiral` (the live re-earning spiral on the depth-6 substrate)

Machinery record. The writeup is [`README.md`](README.md) (post-discussion, 2026-08-23); this
file is the complete per-file, per-gate, per-decision reference.

**Up**: [`SPEC.md`](SPEC.md) (the record of what was asked) · parent arc
[`../README.md`](../README.md) · direct parents [`../native/`](../native/README.md) (the
ports) and [`../tall/`](../tall/FILES.md) (the depth-6 substrate, and the re-run recipe this
node is built to satisfy).

## Fork base — the choice, on the record

`spiral.py` forks **`../native/full/full.py`** verbatim and adds three things, each marked
`# [spiral]` at its insertion point. Nothing is copied out of `tall/`, `ear/` or `recital/`;
what those contribute is *policy*, re-derived here:

| what | where it came from | why it is written rather than imported |
|---|---|---|
| the depth-6 transplant (`_spiral_shared`, `preflight`, `SPIRAL_ERAS`, `_d6_cfg`) | `tall/tall.py`'s `_tall_shared` / `preflight` **idiom** | `tall` builds its shared dict out of `ear._shared_plus`; this file's donor has its own self-contained `build_shared`, so the *lesson* transfers and the code does not |
| certify-else-provisional-at-boundary + the live recert | `ear/ear.py` (carried by `recital/recital.py`) | importing `ear.run_arm` would mean abandoning `full.py`'s two ports, which is the whole point of the round |
| the entry-identity miner instrument | `tall/tall.py::_install_identity_miner` | `tall` monkey-patches the shared `MC.Miner` class; depending on another node's import side effects is not a dependency worth having |

`native/`, `tall/`, `ear/`, `recital/` and `ratchet/` are **not modified**.

## Code files

| file | purpose |
|---|---|
| `spiral.py` | The Modal app. Donor arm loop plus: `_spiral_shared` (complete-key-set assert), `_install_identity_miner`, the `delta_prov` / `prov` commit branches and the provisional skip of the priced pre-commit audition, the live recert block `(g2)` with `pol()`/`port_for()` (the recert is graded **through the arm's own port**), the spiral arms, and the Phase-A instruments (`_descent`, `_pricing_rows`, `_task_matched_buffer`). Entrypoints below |
| `analyze_spiral.py` | Reduction. Phase A: the gate roll-up, the descent table with `dens0`'s reference row, the commit-cycle pricing readout in units of each arm's own cycle-to-cycle \|Δe\|, cost per cycle, and the task-matched collection number |
| `launch_detached.py` | Session-isolated `modal run --detach` launcher (the arc's, retargeted). Logs to `results/launch_<tag>.log`, opened `"w"` — a relaunch overwrites, so that file is a live view, not an archive |

## Entrypoints

| entrypoint | GPU | what it asserts / does |
|---|---|---|
| `gates_remote` | no | the arc's C-M / C-R / G-D **at depth 6**, plus `tall`'s T-1…T-4 re-derived, plus the pricing tables |
| `gates_d6_remote` | L4 | `native`'s own P-1…P-7 / C-1…C-5 **on this substrate** (56 moves, 64 tokens) |
| `preflight` | L4 | every call `phase_a` makes, at toy sizes, in the order it makes them — including the transplanted commit/recert path. Run before any paid setup |
| `fidelity_d4` | L4 | **G-F**: with the transplant off, this file's `run_arm` replays the donor's, bit for bit, with a donor-self-replay control |
| `phase_a` | L4 | the paid depth-6 block (setup, descent gate with routing live, commit-cycle pricing, in-tag depth-6 twin, task-matched collection) |
| `spiral_run` | L4 | Phase B's main entry (the six-arm spiral ladder). Not yet run |

## Gates

| gate | what it asserts | measured |
|---|---|---|
| C-M / C-R / G-D | the arc's: macro ≡ true level move, the ratchet bites, nested damage is on-grammar | pass at depth 6, m=2 |
| T-1 | the ladder nests at all five levels; every node index in range | pass |
| T-2 | damage 100% on-grammar at every ladder level | pass (accept 0.90–1.00) |
| T-3 | oracle repair distance monotone in ladder depth | pass, **gradient 3.313×** |
| T-4 | action set and grounding price over 32–62 moves | 32/48/56/60/62; `n32_b8_w2`=482, `n48_b8_w2`=722, `n56_b8_w2`=842 |
| P-1…P-7 | Port 1's selection / expansion / stream / pricing properties **at depth 6** | pass; 56 moves, 56 slots, injective |
| **P-3** | `beam_moves_prop` at k = n_moves ≡ `beam_moves` bit-for-bit | pass; extra groundings **64** = one root encode per instance |
| C-1 / C-3 / C-4 / C-5 | executor seam is a no-op; an open slot writes only its own span; two heads cost the shared stream nothing; the practice call shape | pass (`C1` max\|Δ\| = 0.0, `C3` outside max\|Δ\| = 0.0 on all 24 macros) |
| **C-2** | composed fidelity: proposal beam at k = n_moves with a span executor present-but-closed ≡ enumerating beam | pass; extra groundings **64** |
| **G-F** | with the tall transplant OFF, `spiral.run_arm` ≡ `native/full.run_arm` at depth 4, carried against a donor **self-replay control** so a non-zero delta is interpretable | `fidelity_d4` |
| in-tag depth-6 twin | `fid` (both ports wired, both shut) ≡ `given` over the run's behaviour series; `g/solve` differs by exactly the declared root encode | inside `phase_a` |

## Pricing, and the choice of `g_budget` (on the record)

`g_budget = 482` = `beam_ground(32, 8, 2)`, i.e. exactly width 2 over the base action set, and
exactly `tall/dens0`'s declared budget. Three reasons, in order:

1. the enum arm is then an **in-run reproduction** of `dens0`'s −0.004/cycle reference rather
   than a comparison against a differently-priced run;
2. under enumeration this budget **cannot** hold width 2 past the L2 commit (n=48 needs ≥ 722,
   n=56 needs ≥ 842) — which is the pricing inversion that voided `tall`, and therefore the
   thing routing is claimed to dissolve. Raising G would remove the effect being tested;
3. under routing the width refits under **k, not |ms|**. Measured by `gates_remote` at G=482,
   budget 8:

| | width |
|---|---|
| enum, n = 32 / 48 / 56 / 60 / 62 | **2 / 1 / 1 / 1 / 1** |
| route, k = 1 / 2 / 4 / 8 / 20 / 32 | **1 / 75 / 18 / 8 / 3 / 1** |

So at k = 4 the beam is 18 wide at the same declared price, and the L2 commit does not move it.
The one transient to watch, and the pricing readout logs it: for `prop_new_cycles` cycles after
a commit the newly committed slots are **forced** into the expansion, so `k_eff` jumps by the
number of new nodes (4 → 20 at the L2 commit, i.e. width 18 → 3) before falling back.

## Volume layout

`rhm-scaling-data:/data/rhm_practice_spiral/<tag>/{setup.json, gate.json, <arm>/results.json,
fid/<arm>/results.json, done.txt}`. Fetched copies under `figures/<tag>/`.

## Runs

| tag | what | outcome |
|---|---|---|
| `sp_s0` | **Phase B**: the main run — seven arms on the sized five-era ladder `1:25:32,2:12:56,3:6:12,4:3:9,5:1:7` (116 cycles/arm), commit policy certify-else-provisional-at-boundary fixed across every earning arm, task-matched collection, `span_min_hold=128` | **complete**, 15,509 s = **4.31 GPU-h**; all 7 arms 116/116; every commit in every earning arm CERTIFIED (no boundary-provisional fallback); `fid` == `given` bit-for-bit over all 116 cycles; L3 span slots cleared tau in `spiral` (5-7 of 8) and `given_native` (3-7 of 8). Figures `figures/sp_s0/fig{1,2,3}*.png` |
| `smokeB` | `spiral_run --quick`, all seven arms, the full Phase-B entrypoint path | complete, 1201 s; every arm 15/15 cycles; `given_native` opened all 24 span slots |
| `fid_d4b` | G-F re-run after the Phase-B additions | **PASS**, still 0.000e+00 against a 0.000e+00 control |
| `fid_d4` | G-F, the donor-fidelity gate at depth 4 | **PASS**: max\|fork − donor\| = 0.000e+00 over 2 arms × 13 series, donor self-replay control 0.000e+00, commit events identical |
| `pa0` | **Phase A**: depth-6 setup, the descent gate with routing live, the commit-cycle pricing readout, the in-tag depth-6 twin, task-matched collection | complete, 1454 s = **0.40 GPU-h**. In-tag twin `given` ≡ `fid` behaviour max\|Δ\| = 0.000e+00 over 5 cycles, with the declared root-encode surcharge on `g/solve` appearing (as +1.0) only from the cycle the filter switches on. Setup 495 s, refs 11 s, 8.1–13.1 s/cycle |

## Reproduce

```bash
cd experiments/            # MODAL_PROFILE=chromatic

modal run rhm/practice/spiral/spiral.py::gates_remote
modal run rhm/practice/spiral/spiral.py::gates_d6_remote
modal run rhm/practice/spiral/spiral.py::preflight
modal run rhm/practice/spiral/spiral.py::fidelity_d4

python3 rhm/practice/spiral/launch_detached.py --fn phase_a --tag pa0 \
    --era "1:25" --cycles 22 --commit-at 16 \
    --arms "practice_late,practice_late_prop_k4,practice_late_native" \
    --fid-arms "given,fid" --fid-cycles 5 \
    --budget 8 --g-budget 482 --max-macro-level 3 --n-corrupt 1 --mine-cap 8 \
    --probe-every 8 --probe-widths "1,2,4" --n-pr 64 --n-rt 384 --n-score 256 --seed 0

python3 rhm/practice/spiral/analyze_spiral.py --tag pa0 --fetch

# --- Phase B -----------------------------------------------------------------
modal run rhm/practice/spiral/spiral.py::spiral_run --quick --tag smokeB

python3 rhm/practice/spiral/launch_detached.py --fn spiral_run --tag sp_s0 \
    --eras "1:25:32,2:12:56,3:6:12,4:3:9,5:1:7" \
    --arms "given,fid,enum_live,spiral_route,spiral,enum_live_g722,given_native" \
    --budget 8 --g-budget 482 --max-macro-level 3 --n-corrupt 1 --mine-cap 8 \
    --span-min-hold 128 --span-tau 0.95 --collect-task-matched --tm-episodes 8192 \
    --prov-offset 0 --recert-every 5 --n-aud 192 \
    --probe-every 8 --probe-widths "1,2,4" --n-pr 64 --n-rt 384 --n-score 256 \
    --sil-c 0.06 --sil-cv 0.15 --sil-win 5 --sil-hold 2 --sil-min-cycle 6 --lp-min-drop 0.10 \
    --prop-warmup 4 --prop-new-cycles 2 --seed 0 --rule-seed 0 --train-seed 1
```

## Phase B: what changed, and the measured reason for each change

| decision | measured reason |
|---|---|
| commit policy = `delta_prov` (certify-else-provisional-at-boundary) for every earning arm, fixed | the arc's approved policy for this substrate; `tall`'s design |
| **the SHADOW CERTIFICATE**, per level, every cycle, in every arm | Phase A's `sil run` was 0 at every cycle, which looks like a dead detector and is not — its arms committed at c16 under `commit="late"` and the donor's certificate stops being evaluated once a level commits. Cycles-to-certification is this round's primary rate readout, so it cannot be conditional on the commit rule that consumed it. Read-only: nothing in it reaches `do_commit`, so the commit path stays bit-identical to the donor's |
| era 1 = 32, era 2 = 56 cycles | **offline detector replay** on Phase A's own audition series at this round's constants: the L2 certificate fires at **c19** (enum) / **c21** (routed), against `cal0`'s clock-paced prediction of c12 — depth-6 certification is ~1.6–1.75× later than calibration suggested. c21 × 1.5 = 32. Scaling `cal0`'s L3 (c22) by the same 1.7× and applying the same margin gives 53–58 |
| `span_min_hold` 256 → 128 | Phase A accumulated 184–218 held-out observations per slot in its 6 post-commit cycles (~30–35/cycle), so at 256 the parity gate was never **evaluated** — `parity` returned `None` on every slot and 0 slots opened in all 22 cycles. τ = 0.95 re-checked every cycle is unchanged |
| task-matched collection damage, adopted uniformly, matched to **era 1's cell** | Phase A measured terminal success 0.1541 against 0.1269 random-block (1.21×) at 41 s. Matched to the first era rather than a mixture of all five: a mixture folds in the L4/L5 cells where nothing succeeds and would *lower* the positive rate |
| certificate constants from `ear` (`sil_cv` 0.15, `lp_min_drop` 0.10, `mine_cap` 8) rather than the donor's (0.10 / 0.05 / no cap) | the transplanted commit policy is `ear`'s and these are the values it was calibrated with; the cap is also what `tall`'s coverage law is stated at |
| `enum_live_g722` carries `width_cap_ref_g = 482`, not a raw budget raise | G=722 holds width 2 at n=48 as intended — but it also buys width **3** at n=32, so a bare budget raise would hand the control a wider beam through all of era 1 and confound the cycles-to-certification comparison in its own favour. The cap is stated as the rule it is (never wider than the matched-budget arm at the base action set; only decline to collapse), because a literal `width_cap=2` is a constant of `budget=8` and becomes a 6× *handicap* at budget 2 — which `preflight` caught |
| arm ORDER `given, fid, enum_live, spiral_route, spiral, enum_live_g722, given_native` | a budget-risk decision: ceiling and fidelity assertion first (nothing is readable without them), treatments before controls, and `given_native` last because the SPEC names it most cuttable. If the run is cut short, what is lost is what was cheapest to lose |

Instruments the Phase-B log carries beyond Phase A's: `shadow_cert` (per level: fire cycle, first-candidate cycle, cycles-to-cert, the audition trajectory the silence test ran on); `committed_grade` (the FROZEN table's recall/precision per cycle, beside the live candidate's); `aud_oracle` on every commit (an unpriced oracle audition of the committed table, so calibration is readable at both levels even for provisional commits, which buy no priced audition by design); and `width_before/after`, `k_eff_before/after`, `n_moves_before/after`, `g_budget` on every commit event at both levels.

**Flagged, not silently accepted**: `enum_live_g722` holds width 2 through the L2 commit (n=48) but **not** through the L3 commit — n=56 needs G ≥ 842, so both enum arms run width 1 in the consumption eras. The L3 *earning* trajectory happens in era 2 at n=48, so the rate claim is un-handicapped, and the consumption-era comparison stays matched between the two enum arms. An `enum_live_g842` would de-confound the consumption eras too; it was not added, per the instruction to spend headroom on era 2 rather than on more arms.

## Phase-A design notes worth keeping

- **The descent-gate arms commit at a FIXED cycle** (`commit="late"` with
  `late_offset = cycles - commit_at`), not on the certificate. The transplanted `delta_prov`
  policy is what the main run uses and is exercised by `preflight`; here a certificate firing
  at different cycles in different arms would confound the one number the block exists to read.
- **`n_corrupt = 1` is folded into `build_shared`**, not bolted on. `dens0` had to retrain a
  value head because it was sweeping variants; there is nothing left to sweep, so the densified
  buffer is simply the buffer this file builds.
- **The recert suspends span capture** for the duration of its beams. It is a grading step;
  letting it feed the corridor's self-imitation buffer would give a treated arm strictly more
  head training data on recert cycles than its twin.
- **`preflight_seed_miner`** is a preflight-only cfg key. A substrate trained for forty steps
  solves nothing, the miner never observes, and the empty-table guard turns every commit off —
  which left the transplanted commit path untested on the first preflight attempt. Seeding the
  miner from the DGP's own tuples makes the path reachable; it is scientifically meaningless by
  construction, which is why no real config carries the key.
