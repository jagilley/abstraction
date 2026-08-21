# FILES — `native/prop` (Port 1: routing / proposal)

Machinery record for Port 1 of the [`native`](../SPEC.md) node. No `README.md` here yet —
results go to Jasper for discussion first.

**Fork convention** (inherited from [`../../teacher_slot/handle/`](../../teacher_slot/handle/)):
`prop.py` is a verbatim copy of [`../../ratchet/ratchet.py`](../../ratchet/ratchet.py) with the
routing port added. `ratchet/` and `teacher_slot/` are not modified; `macros.py` is imported from
`ratchet/` rather than copied.

## Code files

| file | purpose |
|---|---|
| `prop_net.py` | The port's own machinery: the `(level, node)` slot layout, `ProposalHead` (pi(move \| z, r\*), zero-init output), `isolated_rng`/`build_head` (the head consumes none of the shared torch stream), `select_moves` (stable-sort top-k; exact no-op at k = n_moves; `forced` and `explore` extensions), `expand_selected` (ragged expansion — move j materialised only on the tips that chose it), `decompose_probe`/`base_mass_probe` (the can't-decompose signature), and `beam_ground_k`/`fit_width_k` (the pricing under branching k). |
| `prop.py` | The fork. New: `beam_moves_prop` (the filtered beam, with the kept children's encoder states cached and reused as the next step's proposal read), `plan` (one entry point so the port is either everywhere the beam plans or nowhere), `prop_pairs`/`prop_train` (self-imitation on the beam's own solved surviving trajectories, masked CE over the live action set), the `*_prop_k*` arms + `TWIN`/`STREAM`, `prop_selfcheck` (gates P-1..P-7), entrypoint `prop_run`. Everything else is ratchet's. |
| `launch_detached.py` | Session-isolated `modal run --detach` launcher (handle/'s, retargeted). Logs to `results/launch_<tag>.log`. |
| `analyze_prop.py` | Reduction. Retargets `ratchet/analyze_ratchet.py`'s volume prefix/figure root for the standard readouts, then adds: the twinning + fidelity gates, the per-solve ledger (groundings / materialisations / proposal reads, effective branching, width), the cost-to-match-the-twin's-success read off the probe width ladder, the can't-decompose signature, the next-level currency native vs exogenous, and the plant guard. Figures `fig1`–`fig4`. |

## The port in one paragraph

A proposal head pi(move | z, r\*) sits beside the value on the **same encoder read**
(`controller.state(x)` plus the root target). It is trained by self-imitation on the beam's own
chosen trajectories — `beam_moves(collect=True)` already returns them (`traj` = per-step states of
the surviving trajectories, `tips_seq` = their move ids) — restricted to the survivors that
terminally **solved**, so selection happens before the regression. At plan time only the top-k
proposed moves per tip are materialised and value-scored, instead of all `n_moves`. Output logits
live in a fixed `(level, node)` slot vocabulary (14 slots at L=4, s=2, max_level=3), so the head
survives the action set growing at a commit.

## Pricing, stated (the spec asks for this out loud)

A grounding is "score one state" (encoder pass + value read; `beam_moves` charges one per
materialised child).

* Every proposal read **below the root** is a head-only forward on an encoder state the value
  already paid a grounding for when it scored that state as a child. The port caches those
  states and the ledger charges nothing for re-reading them.
* The **root** state has never been anybody's child, so the port pays **one full grounding per
  instance per solve** for it, and `fit_width_k` sees that cost when it picks the width.
* The head-only forwards are counted separately (`counts["prop"]`, logged per cycle and per
  metering beam) and priced at `c_prop`, **default 0.0**; the reduction can re-price them.
* The freed groundings are **reinvested, not banked**: every arm runs the widest beam that fits
  the same declared G = 58 under *its* effective branching factor — ratchet's matched-pricing
  idiom with `n_moves` -> `k`. The banked reading (same width, fewer groundings) is recovered
  from the width ladder logged at every probe (`analyze_prop.matched_success`).

Consequence worth noting: under a top-k filter the grounding cost depends on **k, not on how many
moves the arm holds**, so `given_prop_k4`, `never_base_prop_k4` and `practice_late_prop_k4` spend
the identical budget at the identical width. The chunk-vs-primitive control is exactly matched.

Price table at budget 4, G = 58 (`prop_selfcheck` P-7):

| | enum n=8 | enum n=12 | enum n=14 | k=1 | k=2 | k=4 | k=8 |
|---|---|---|---|---|---|---|---|
| width | 2 | 1 | 1 | 1 | **16** | 4 | 1 |
| g/solve | 58 | 49 | 57 | 6 | 47 | 57 | 34 |

## Design decisions (choices, not derivations)

| decision | choice | why |
|---|---|---|
| online vs offline training | **online, inside the loop**, after a `prop_warmup` of 4 cycles during which the head trains but never gates | the freed-budget question is a within-run question, and self-imitation wants on-policy data (an offline head trained on an enum run's trajectories is off-policy for the rollout it then gates). Cost admitted: proposal quality and the commit schedule interact for the `practice_late_*` arms. The warmup means the filter never gates from a random init, and doubles as the twin gate's boundary. |
| self-imitation loss | masked cross-entropy over the live action set, on the surviving tips that **solved** | selection before regression (§18's "gradient descent is averaging" objection does not bite when the targets are selected first). Masked, so a slot that does not exist yet is neither target nor competitor. |
| no solved tips this cycle | fall back to all surviving tips | a policy that cannot generate a success cannot be improved by imitating an empty set; without the fallback a bad filter freezes itself in. Logged per cycle (`prop.fallback`). |
| practice explores, performance does not | `prop_explore = 1` extra uniformly-random move per tip, **practice beam only** | the substrate's own convention (`collect_value_buffer` runs its behaviour policy at `explore_eps` = 0.3). Without it the filtered beam is a closed loop: a move the head does not propose never appears in a solved trajectory, so it never becomes a target, so the head never learns it — measured in `smoke0`, where `given_prop_k2` locked L3n1 out of its top-5 while its k = n_moves twin ranked it first. Uses its own numpy stream. Disabled when k >= n_moves, so the fidelity gate is untouched. |
| a newly committed move | zero-init output rows (neutral entry) **plus** force-expansion for `prop_new_cycles` = 2 cycles | an untrained logit should not be able to lock a just-earned macro out of the action set. Priced at its true cost: the effective branching is k + \|forced\| those cycles, and the width is refit accordingly. |
| k ladder | k ∈ {1, 2, 4} + k = n_moves (fidelity) | picked off the price table: k=1 is the no-search extreme (6 g), k=2 buys width 16 at 0.82x enum's budget, k=4 is exactly budget-matched to `given`'s enum (57 g). k=8 is off the ladder because at branching 8 the root encode tips the base arm's width from 2 to 1 and would confound the control. |
| the poison twin (`practice_early` natively routed) | **not run in round 1** | it costs two arms, not one (the untreated twin is required for the twinning gate), and its question — does routing to a bad L2 foreclose L3 *worse* — is only interpretable once routing is known to do anything at all. |

## Gates

`prop_selfcheck` (P-1..P-7, no training) and the in-tag arm gates:

| gate | what it asserts |
|---|---|
| P-1 | `select_moves` at k = n_moves returns `arange(n_moves)` for every row **including exact ties** (a zero-init head produces exactly that) — the tie-ordering hazard, closed by a stable sort rather than `topk`. |
| P-2 | `expand_selected` at k = n_moves is bit-identical to the enumerated `stack([apply_any(flat, ms[j])], 1)`. |
| P-3 | `beam_moves_prop` at k = n_moves reproduces `beam_moves` bit-for-bit (x, seq, tips, traj, materialisation count); the only delta is the declared root encode, asserted to equal exactly one grounding per instance. |
| P-4 | building the head consumes **none** of the global torch stream. |
| P-5 | the head is exactly uniform at init (zero-init output layer). |
| P-6 | at k < n_moves the selection is exactly the k highest logits, ascending; with `forced` it is k + \|forced\| wide and contains every forced index. |
| P-7 | the `(level, node)` slot layout is injective over the full 14-move action set; prints the price table above. |
| **twinning** (in-tag) | every treated arm bit-identical to its untreated twin through the warmup: max\|Δe\| = 0.0. |
| **fidelity** (in-tag) | `given_prop_kN` (k = n_moves) reproduces `given` for the **whole** run: max\|Δe\| = 0.0. |

## Reproduce

```bash
cd experiments/            # MODAL_PROFILE=chromatic

modal run rhm/practice/native/prop/prop.py::prop_selfcheck_remote     # P-1..P-7
modal run rhm/practice/native/prop/prop.py::prop_run --quick --tag smoke1   # smoke

python3 rhm/practice/native/prop/launch_detached.py --fn prop_run --tag np_s0 --seed 0 \
    --arms "never_base,given,practice_late,given_prop_kN,given_prop_k1,given_prop_k2,given_prop_k4,never_base_prop_k1,never_base_prop_k2,never_base_prop_k4,practice_late_prop_k2,practice_late_prop_k4" \
    --eras "1:6,2:3,3:1" --era-cycles 30 \
    --budget 4 --pr-width 16 --g-budget 58 --max-macro-level 3 \
    --n-pr 64 --n-rt 384 --n-score 512 --n-grad 4 --value-lr-online 3e-5 \
    --gen-lr 1e-4 --gen-steps 20 --plant-holdout 0 \
    --mine-from chosen --mine-cap 8 --mine-support 3 \
    --sil-c 0.06 --sil-cv 0.15 --sil-win 5 --sil-hold 2 --sil-min-cycle 6 --lp-min-drop 0.10 \
    --early-offset 1 --late-offset 3 --probe-every 4

python3 rhm/practice/native/prop/analyze_prop.py --tag np_s0 --fetch --figures
```

Substrate flags are `rr_s0`'s exactly (v=8, s=2, L=4, m=2; eras `1:6,2:3,3:1`; era_cycles 30;
G = 58), so the vocabulary is held fixed and the port is the only new variable.

Modal volume (`rhm-scaling-data`): `/data/rhm_practice_native_prop/<tag>/<arm>/results.json` with
`setup.json` beside them. Fetched copies + figures land under `figures/<tag>/`.

## Runs on disk

| tag | what it is |
|---|---|
| `smoke0` | first attached `--quick` smoke, 7 arms. Fidelity gate passed (max\|Δe\| = 0.0). Diagnosed the closed-loop exploration failure that added `prop_explore`. |
| `smoke1` | attached `--quick` smoke with exploration on |
| `np_s0` | the main run: 12 arms, seed 0, 3 eras x 30 cycles |
