# teacher_slot — File & Child Index

Summarized in [README.md](README.md) (the single writeup for the whole node);
[SPEC.md](SPEC.md) is the record of what was asked. The children below carry machinery and
outputs only — per the node's convention, they do not have individual READMEs.

## Children

| Child | What |
|---|---|
| [`decision/`](decision/) | Rungs A + A½ machinery on the `fourwall/lm` substrate (donor imported verbatim, untouched). `slot_lm.py` — the Modal app: donor training loop + per-checkpoint reversible condition choice, ABBA paired-trial value rule, priced reads, `gate` (bit-identity vs fwlm0/fwlm1 twins, max\|Δ\| = 0.0); `policy.py` — FixedPolicy (step-function ≡ donor merge arms) and the paired-trial policies with offline gates P-1…P-10; `launch_detached.py`, `analyze_slot.py` (fetch/reduction/figures incl. decision traces and the d5 trajectory). Tags: `tsd_gate`/`tsd_gate2` (gates), `tsdA` (round 1 — voided decision contrast, on the record), `tsdB` (round 2 — the reported results). Figures under `figures/<tag>/`, reductions under `results/`. |
| [`verbal/`](verbal/) | Rung B1 machinery (local, no GPU). `records.py` — record access, readout conditions, splice/session state machine over measured fwlm0/fwlm1/tsdB trajectories; `vignette.py` — the blinded neutral-vocabulary template + programmatic `check_blinding()`/`check_symmetry()`; `run_sessions.py` — pinned headless `claude-opus-5` driver, full verbatim logging; `reduce.py` — decision tables, timing, hold-vs-bail with splice-mismatch discounting, justification excerpts. Tag `b1`: `outputs/sessions_b1.json` (all 106 prompts + responses), `outputs/reduction_b1.txt`, reviewed example vignettes. |
| [`handle/`](handle/) | The separable backprop-handle probe on the ratchet substrate (`ratchet/` untouched; forked runner). `handle_net.py` — generator wrapper adding zero-init macro-symbol embeddings + per-span prediction heads at commit time; `handle.py` — forked ratchet runner with per-arm torch streams (twin pairs bit-identical to the commit cycle) and gates H-0…H-5; `launch_detached.py`, `analyze_handle.py`. Tag `hr_s0`: 6 arms (`never_base`, `given`, `given_handle`, `practice_late`, `practice_late_handle`, `practice_late_handle_level`); figures + `summary.json` under `figures/hr_s0/`. |

## Code files

No code at this level; every script lives in a child (see the table above — each child's scripts
are enumerated in its row). `decision/` imports `../fourwall/lm/` and `handle/` imports
`../ratchet/`; neither donor is modified.
