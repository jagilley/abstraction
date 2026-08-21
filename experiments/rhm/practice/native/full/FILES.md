# FILES — `native/full` (Phase 2: the composed ports, and the ablation battery)

Machinery record. No `README.md` here by design — READMEs come after the numbers are discussed.

**Up**: [`../SPEC.md`](../SPEC.md) · Port 1: [`../prop/FILES.md`](../prop/FILES.md) ·
Port 2: [`../span/FILES.md`](../span/FILES.md) · substrate donor:
[`../../ratchet/`](../../ratchet/README.md).

## Fork base — the choice, on the record

`full.py` forks **`../prop/prop.py`** (Port 1), which is itself a verbatim fork of
`../../ratchet/ratchet.py`, and grafts Port 2 by **importing** `../span/span_net.py`. Nothing is
copied from `span/`; `macros.py` is still imported from `ratchet/`. `ratchet/`, `teacher_slot/`,
`prop/` and `span/` are untouched.

The alternative was re-forking `ratchet.py` and importing both nets. Forking `prop.py` was chosen
because Port 1's insertions are the larger and more delicate of the two — a rewritten beam with a
cached-encoder-state contract and a fidelity gate already verified bit-for-bit in-tag — while
Port 2's are a clean seam: one executor object threaded through the beam's materialisation call.
Re-deriving Port 1 by hand would have risked the already-gated half to save nothing. The cost of
this choice is that Port 2's insertions are re-applied here rather than inherited, so
`finetune_generator_span` is duplicated from `span.py` (with one addition, below) and the
`mint`/`bind_slots`/parity-gate block is re-pasted; both are asserted against `span_net`'s own
gates (S-3 via C-1) rather than trusted.

## Code files

| file | purpose |
|---|---|
| `full.py` | The composition. New relative to `prop.py`: the executor seam (`beam_moves`, `beam_moves_prop` and `plan` all materialise through `span_net.PlainExecutor` / `SpanExecutor`); `finetune_generator_span` (plant loss + span self-imitation in the same optimizer steps, **plus a `lam == 0` short-circuit** that restores the original graph exactly, which is what makes the composed fidelity arm a real gate); the native arms, the poison pair and `given_fid`; `ablation_battery` (nested `battery()` at the end of `run_arm`); `full_selfcheck` (P-1…P-7 inherited + C-1…C-4). Entrypoints `full_run`, `full_selfcheck_remote`. |
| `launch_detached.py` | Session-isolated `modal run --detach` launcher. Logs to `results/launch_<tag>.log`. |
| `analyze_full.py` | Reduction. Retargets `ratchet/analyze_ratchet.py` and reuses `prop/analyze_prop.py`'s Port-1 readouts, then adds the twinning + composed-fidelity gates, the composed ledger printed against phase 1's numbers, the ablation battery, the structural/informative split of next-level currency, the poison contrast, span diagnostics, and the plant guard. Figures `fig1`–`fig3` + `summary.json`. |

## Arms

| arm | vocabulary | Port 1 | Port 2 | role |
|---|---|---|---|---|
| `given` | true table | — | — | untreated twin / in-tag enum reference |
| `given_native` | true table | k = 4 | span head | the full native primitive |
| `given_fid` | true table | k = n_moves | wired, `span_lam=0`, `span_tau=2.0` | **the composed fidelity gate** |
| `practice_late` | mined (c27/c57) | — | — | untreated twin |
| `practice_late_native` | mined (c27/c57) | k = 4 | span head | the full native primitive on an earned table |
| `practice_early` | 1-entry bogus L2, frozen at c1 | — | — | the poison baseline (ratchet's finding 7) |
| `practice_early_prop_k4` | same bad table | k = 4 | — | **the poison twin**: a natively *routed* bad L2 |

`k = 4` is Port 1's measured sweet spot — at G = 58 it is exactly budget-matched to `given`'s
enumeration (57 groundings per solve either way), so composition changes nothing in the headline
ledger's denominator. Span flags are `sp_s0`'s (`--span-lam 1.0 --span-lr 1e-3 --span-tau 0.95
--span-min-hold 256`).

The poison twin is routing-only on purpose: the SPEC's question is whether a bad address gets
*called* more when the policy routes to it, and adding the corridor port would make the contrast
two-variable.

## The ablation battery (end-of-run probe, per arm, per era)

Run **after** the 90 cycles on the final trained state rather than as a mid-run switch, so the
trajectory every other readout is measured on is never corrupted. Unpriced (it is an instrument),
but each condition records its own per-solve cost so the conditions stay comparable.

| condition | action set | execution | asks |
|---|---|---|---|
| `a_full` | as it ran | as it ran | the reference |
| `b_table` | macros **stay callable** | the table may not be consulted: the span head serves **every** macro call | routing-not-pruning — is the address book still needed at the current level? |
| `b_span` | base moves only | plain | table *and* corridor deleted: pure enumeration fallback |
| `c_prims` | macros only | as it ran | can the policy solve with chunks alone? |

**Policy under `b_table`, stated because it is a choice**: the head serves all macro calls
**regardless of the parity gate**, so a parity shortfall shows up as error rather than being
hidden behind a fallback to the very DP path the condition is meant to have removed. The
end-of-run parity of every slot is recorded beside the numbers so the shortfall is readable. An
arm with no span head has no execution path under `b_table` at all and is reported as such — for
those arms deleting the table deletes the macro, which is the contrast the prediction wants.

**Next-level currency is split**, because the mining-collapse half of the prediction is partly
structural in this substrate:

- **structural** — `built`: entries `Miner.build` can assemble over the surviving lower table.
  `T3 ⊆ T2 × T2`, so deleting T2 drops T3 to **0 by construction**. This number carries no
  information on its own and is reported only so it can be subtracted.
- **informative** — `at_support`: distinct level-1 tuples observed at least `mine_support` times
  in that condition's own chosen trajectories, keyed by the flat tuple and gated by **no table at
  all**. This is the real question: does the observation stream of chunk-shaped spans survive
  when the routing is native?

## Gates

`full_selfcheck` (L4) carries Port 1's P-1…P-7 unchanged and adds the composition's own:

| gate | what it asserts | result |
|---|---|---|
| C-1 | `PlainExecutor` and a non-firing `SpanExecutor` are `macros.apply_any` on every move — the executor seam is a no-op by itself | max\|Δ\| = **0.0** |
| C-2 | **composed fidelity**: the proposal beam at k = n_moves with a span executor present but closed reproduces the enumerating beam bit-for-bit (x, seq, tips, traj, materialisation count) | all equal; extra groundings = **64** = exactly one root encode per instance |
| C-3 | an OPEN span slot rewrites exactly the macro's own token positions and nothing else, through the composed beam's executor | outside max\|Δ\| = **0.0** on all six macros; head fired 384× |
| C-4 | minting **both** heads leaves the shared torch stream exactly where it was | untouched |

In-tag arm gates: twinning (bit-identical to the untreated twin until the **earliest** port
switches on — the prop filter at c5, or the span head's first stepped slot), and `given_fid` vs
`given` over the whole run.

Note on the twin gate for `given_native`: its span slots exist from cycle 1 (the true table is
committed at construction), so its earliest port-on is c1 and it has **no pre-window by
construction**. That is why `given_fid` is carried: it is the in-tag assertion that the composed
loop reproduces enumeration, which the `given_native` twin gate cannot supply.

## Volume layout

`rhm-scaling-data:/data/rhm_practice_native_full/<tag>/{setup.json,<arm>/results.json,done.txt}`.
Fetched copies and figures under `figures/<tag>/`.

## Runs

| tag | what it is |
|---|---|
| `smoke0` | detached `--quick` smoke, 7 arms; quick mode forces the span gate open (`span_tau=0`, `span_min_hold=32`) so the firing path is exercised, while `given_fid`'s per-arm cfg keeps its own gate nailed shut |
| `nf_s0` | the main run: 7 arms, seed 0, 3 eras × 30 cycles, plus the ablation battery |

## Reproduce

```bash
cd experiments/            # MODAL_PROFILE=chromatic
modal run rhm/practice/native/full/full.py::full_selfcheck_remote
modal run rhm/practice/native/full/full.py::full_run --quick --tag smoke0

python3 rhm/practice/native/full/launch_detached.py --fn full_run --tag nf_s0 --seed 0 \
    --arms "given,given_native,given_fid,practice_late,practice_late_native,practice_early,practice_early_prop_k4" \
    --eras "1:6,2:3,3:1" --era-cycles 30 \
    --budget 4 --pr-width 16 --g-budget 58 --max-macro-level 3 \
    --n-pr 64 --n-rt 384 --n-score 512 --n-grad 4 --value-lr-online 3e-5 \
    --gen-lr 1e-4 --gen-steps 20 --plant-holdout 0 \
    --mine-from chosen --mine-cap 8 --mine-support 3 \
    --sil-c 0.06 --sil-cv 0.15 --sil-win 5 --sil-hold 2 --sil-min-cycle 6 --lp-min-drop 0.10 \
    --early-offset 1 --late-offset 3 --probe-every 4 \
    --span-lam 1.0 --span-lr 1e-3 --span-tau 0.95 --span-min-hold 256

python3 rhm/practice/native/full/analyze_full.py --tag nf_s0 --fetch --figures
```

Substrate flags are `rr_s0`'s exactly, so the vocabulary is held fixed and the ports are the only
new variables.
