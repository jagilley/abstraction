# FILES — `antiphon/trap` (the installed trap: what is a worthless answer here?)

Machinery record for the node. **No results are interpreted here** — the reduction's numbers go
to the orchestrator and are discussed before any writeup (repo norm).

**Up**: [`DESIGN.md`](DESIGN.md) (the brief, the offline sizing, the arms and gates) ·
parent node [`../README.md`](../README.md) · [`../FILES.md`](../FILES.md) §1–§3 (the question
parameterization and the controls this node inherits unchanged).
**Direct donor** (extended additively, never forked): [`../antiphon.py`](../antiphon.py) and
[`../questions.py`](../questions.py) — every addition marked `# [trap]`, defaulted OFF, and
gated bit-identical with the knob off.

## 1. What the trap is, in one paragraph

Every true level-L flat's two halves are true level-(L−1) flats (0 exceptions at L3/L4/L5 on the
DGP's own tables), so a key whose half is not a true row is junk **with certainty** and
`Miner.build`'s ratchet refuses it forever. The half the *question* fixes is the one the damage
cell does not cover, and it reaches the miner as the question set it. So: **a distractor is a
menu candidate whose clean half parses to a key that is not a true row**, and the knob is the
fraction `f` of the K = 2048 menu built from such candidates, presented spread uniformly over the
junk-half pool. Rejected with reasons in `DESIGN.md` §1: a foreign grammar (the action space is
world-typed and mining is success-filtered, so it can never mint keys) and an unlearnable channel
at matched difficulty (the reachable junk-half pool is smaller than the era's budget over
`mine_support`, so everything closes).

## 2. Code files

| file | purpose |
|---|---|
| `trap_menu.py` | **The trap channel**, out of the Modal app so every rule is auditable with no GPU (`../questions.py`'s convention). The atom (`is_distractor`, `clean_keys_of`, `trap_applies`, `clean_level`), the per-era `DistractorBank` (keyed by clean half, so the trap can be presented uniform over the junk-half pool), `build_bank`'s rejection sampler, `install_trap`'s slot substitution, `trust_weights`, and the offline gate suite **T-3** (9 checks). |
| `phase0_trap.py` | **The offline phase.** [0]/[1] geometry and the junk-half pools; [1b] the breadth-first take share; [2] the worthlessness proof; [3] d\*-orthogonality and quota feasibility; [4] construction cost; [5] the selector simulation on real menus with a delivery model calibrated to the banked stream; [5b] the selection premium vs f; [5c] the incumbent ledger's discrimination; [6] closure arithmetic; [7] the banked-log reads; [8] the rejected channels; [9]/[9b] the use record as a guard. No GPU, no Modal, no substrate. |
| `phase0_trap.json` / `.txt` | Phase 0's output, and its printed tables. |
| `analyze_trap.py` | **The reduction.** §0 the era-1 twin gate, §1 controls, §2 dose, §3–§6 delegated to `../analyze_antiphon.py` (sound because the substrate is certified identical), §5 the premium join, §7 the two poles against the pool-share bound, §8 the guard (use-weighted precision at commit+20 and end of run; the realised per-cycle take). One figure. |
| `figures/tr_s0_trap.png` | The climb at f = 0.90 with the native-dose twins dotted, and what each selector took against the menu rate and the pool share. |
| `figures/tr_s0_reduction.json` | The reduction's output. |

Additions in the donor files, all `# [trap]`-marked and defaulted OFF:
`questions.py` — `select_trust`, `SUBW_FLOOR`, `_FORBIDDEN` extended (**T-4**), gates **Q-12/Q-13**.
`antiphon.py` — the `trap_menu` import; `pose_questions`'s trap block and its `trap` dose row;
the trust bundle's `subw`; `run_arm`'s `tstate` and the cumulative beam-use accumulator;
`trap_mined`; nine `t*_*` arms and their `TWIN` entries; `trap_frac`/`trap_spread`/
`trap_bank_mult` in `_cfg`; `trap_gate` in `preflight` and `fidelity_smoke`; `q_interface_check`
extended to the `trust` mode.

## 3. Gates

| gate | what | where | status |
|---|---|---|---|
| **T-1** | worthlessness: halves of a true level-L flat are true level-(L−1) flats | `phase0_trap.py` [2] | **PASS** — 0 exceptions at L3/L4/L5 |
| **T-2** | d\*-orthogonality and quota feasibility at every f | `phase0_trap.py` [3] | **PASS** — TV 0.074 / 0.039; ≥ 32× the per-stratum need |
| **T-3** | menu construction, 9 checks incl. "at f = 0 the inputs are returned untouched" | `trap_menu.py::trap_gate` | **PASS (9/9)** |
| **T-4** | containment: `trap`/`distractor`/`is_d`/`bank` refused in an agent bundle; `subw` allowed | `questions.py::containment_gate` | **PASS** |
| **Q-1…Q-13** | the port's offline suite plus the trust selector's two | `questions.py::question_gate` | **PASS (13/13)** |
| **P-1…P-12, L-1…L-8** | A1's/A2's policy suite against the imported module, re-run after the edits | `maestro.policy.policy_gate()` | **PASS (21/21)** |
| **G-T** | with the port and the trap off, this file replays **`crescendo.py`** in process | `antiphon.py::fidelity_smoke` (`tr_gf`) | **PASS — 0.000e+00** on `anchor_long` and `given_c1`, against a 0.000e+00 donor-donor control; commits equal |
| **Q-11** | every selector, every era, on the real substrate | inside `tr_gf` | **PASS (21 cells**, up from `an_s0`'s 18 — the `trust` mode adds three) |
| **§0 era-1 twin (full scale)** | the trap atom does not exist at era 1, so every trap arm must be bit-identical to its `an_s0` counterpart for 60 cycles and diverge on the first cycle of era 2 | `analyze_trap.py` §0 | **PASS — 0.000e+00** in all five arms, first divergence **c61 = expected c61** |
| **within-tag twin** | `t90_trust` vs `t90_novel` — the guard degenerates to novelty until it has a non-uniform table one level down | `analyze_trap.py` | **0.000e+00 through c110, diverges at c111** = the first cycle of era 3 |
| **substrate identity** | `--ref-tag an_s0` asserted before any cycle ran | in-run `[ref]` | **PASS** — `refs_identical: true`, `read_acc` 1.0/1.0, stale buffers identical |
| **quota** | per-cycle d\* histogram bin-for-bin | §1 | **1.000** in all five arms (201/201); selected d\* 2.774 vs menu 2.773 |
| **volume** | mined == `mine_cap` | §1/§2 | **0.945–0.980** (`an_s0`'s band was 0.955–0.985) |
| **priced budget** | cumulative groundings | §1 | matched to **0.009%**: 50,687,206–50,691,776 |
| **dose** | offered / taken / landed, per era per arm | §2 | menu f **0.000** at era 1 (the atom is absent), **0.900** at eras 2–3 |

## 4. Runs

| tag | what | outcome |
|---|---|---|
| `tr_gf` | G-T + Q-11 | **PASS**, app `ap-8QAwLfhXC8xX5oNuVSJo1c` |
| `tr_smoke` | 3 arms at `--quick`, `--question-k 512`: T-3/T-5/T-6 shakedown | wired end to end; banks built to the offline pool sizes (9 / 96 junk halves); **T-5 uninformative at quick scale** (`mine_cap` fill 0.000 in the f = 0 control too — `an_smoke`'s documented starved-substrate caveat), 0.12 GPU-h, app `ap-SYl2X8gACFbLcVagKkZBfA` |
| `tr_s0` | the main run: 5 arms at f = 0.90, 201 cycles each | **complete**, 2026-09-01, app `ap-UBFJ7MprZIjzubCuCIySvT`. 11552 s, **3.21 GPU-h**, 10.7–10.9 s/cycle. All gates above pass |

Volume `rhm-scaling-data:/data/rhm_practice_antiphon/{tr_gf,tr_smoke,tr_s0}/`; fetched mirrors
under `../figures/`, reduction and figure under `figures/`.

## 5. Reproduce

```bash
cd experiments/            # MODAL_PROFILE=chromatic

python3 rhm/practice/antiphon/trap/phase0_trap.py                    # offline, ~60 s, no GPU
python3 rhm/practice/antiphon/trap/trap_menu.py                      # T-3, offline
python3 rhm/practice/antiphon/questions.py                           # Q-1..Q-13, offline

python3 rhm/practice/antiphon/launch_detached.py --fn fidelity_smoke --tag tr_gf --cycles 4
python3 rhm/practice/antiphon/launch_detached.py --fn antiphon_run --tag tr_s0 \
    --arms "t90_exo,t90_bisect,t90_trust,t90_novel,t90_endo" --ref-tag an_s0 --seed 0
python3 rhm/practice/antiphon/trap/analyze_trap.py --tag tr_s0 --merge-tag an_s0 \
    --fetch --figures
```
