# Cut #5 — dimensionality expansion under drift: file index

Full file-by-file reference for this node. Summarized in [README.md](README.md); this is the
lookup material. Belief under test:
[`beliefs/dimensionality_expansion.md`](../../../beliefs/dimensionality_expansion.md).
Instrument of record: [`rhm/residual_decomposition/`](../../rhm/residual_decomposition/README.md)
— **β**, **`R_res_participation`**, **frontier mass**. The `(R_act, R_comp, R_res)` triple this cut
was originally specified against was falsified 2026-07-26 and is not used anywhere here; **β is
computed and stored but never read**, because Piece 1 measured it failing its own
capacity-invariance control by 6–59× on this substrate.

## The three pieces

| Piece | Question | Answer | Script |
|---|---|---|---|
| **1 — saturation gate** | Is the instrument in the saturated dead zone here, and does it respond to a *wrong model*? | Not saturated (rel. residual 0.107 vs 0.003–0.008); separates a stale FM **2.7×** at k=14 | `saturation_gate.py` |
| **2 / 2b — support-fixed null** | Does drift that only moves the target function open the frontier? | **No** — ±0.08 directions across a de-confounded 2×2 | `support_fixed.py` |
| **3 — DOF accretion** | Does the instrument respond to *added directions* at all? | Yes — **+0.72 ± 0.42**, 3/3 seeds | `support_growing.py` |

## Code files

| File | Purpose |
|---|---|
| `saturation_gate.py` | **Piece 1.** Sweeps the FM's k-step-displacement prediction gap (k = 1…20) and measures the frontier instrument at every cell across four axes: **k** (gap width — on this substrate *is* the reactive↔ballistic commitment axis), **n_links** (3 = capacity slack / 5 = capacity binds), **FM variant** (`matched` / `stale` = known-wrong positive control / `small` = β capacity-invariance check), **probe** (`task` = matched-FM reach transitions / `broad` = teleport pool). Carries a pre-registered READS/LATENT/CLOSE decision rule with a **two-sided** readable window (`0.02 < frontier mass < 0.90`) — the upper degeneracy (diverged rollout ⇒ frontier → 1, nothing explained) is unique to rolling a model out and the rhm domains never hit it. Local helper `shadow_law_flex` = rhm's `shadow_law` with its `min_dirs=8` guard relaxed and `n_dirs` always reported, since the arm's state target has only 2n = 6 or 10 directions. Also records `resid_mean_frac` (the mean-blindness diagnostic — `repaired_triple` mean-centers, so a constant-offset residual is invisible by construction) and per-probe joint excursion. |
| `saturation_gate_agg.py` | Piece 1 cross-seed aggregator (local, CPU-only): the k-sweep, the stale-vs-matched positive control with a per-seed sign count, and the instrument checks (β + `n_dirs` + capacity-invariance gap; mean-blindness fraction). |
| `support_fixed.py` | **Pieces 2 and 2b.** Eight rounds of perpetual OU drift on curl gain(s) with the FM re-fitting each round, graded on a **frozen probe** (fixed (s₀, k-step command) pairs re-executed in each round's plant, so only the world and the model differ). Conditions form a 2×2 from the name — anything ending `static` does not drift, anything starting `regions` uses the four-gated-region geometry: `static` / `global` / `regions_static` / `regions`. Three variants graded every round: `live` (fine-tuned), `frozen` (round-0 snapshot, the drift-tracking control), `rail` (field-free FM = Piece 1's stale yardstick). `buffer_mode` defaults to `fresh` (bit-identical to the first pass, tags `sf_s*`); `accumulate` is the 2b protocol and is **wrong under drift** — see the README's contamination box. |
| `support_fixed_agg.py` | Pieces 2/2b cross-seed aggregator, in reading order: (0) did the drift happen, (1) was the instrument awake — `frozen`/`rail` separation **plus** the per-seed corr(\|gain−mu\|, frozen frontier) tracking control, (2) the 2×2 with each drift arm differenced against **its own geometry's** no-drift control, (3) the `R_act` confound check and probe out-of-box diagnostics. `--k` re-reads any stored horizon (this is how the k=8 correction was obtained for free). |
| `support_growing.py` | **Piece 3.** DOF accretion as the dimensionality-axis calibration. `locked` / `accretion` / `full` over 8 rounds, all from one shared random init. The probe is drawn from and executed on the **fully-unlocked arm, once, for every round of every condition**, so `R_act` is constant by construction (asserted, measured spread `0.0e+00`) and the "unlocking a joint mechanically adds directions to the plant's covariance" tautology cannot occur. `--probe-only` prints per-joint variance shares and exits — that is how the lock set was chosen rather than guessed. |
| `support_growing_agg.py` | Piece 3 cross-seed aggregator: lock-deviation check, the staircase, the pass/fail calibration verdict, Spearman of each readout against free-DOF count, and the readable-window / separation-sign table across all k. |
| `train.sh` · `train_piece2.sh` · `train_piece2b.sh` · `train_piece3.sh` | Per-piece 3-seed launchers. Seeds run **sequentially** — detached Modal clients launched from one shell evict each other, and a client killed before its function's final `volume.commit()` loses `results.json` entirely (`ballistic/arm/train.sh`, `arm_substrate` gotcha (ii), `on_policy` §Gotchas) — with the grep-for-completion-marker retry that `ballistic/arm` needed for the client-side connection flake. `train_piece2.sh` is kept unchanged so the first (pre-2×2) pass stays re-runnable. |
| `__init__.py` | Package marker, so `.add_local_python_source("mjc")` ships this folder. |

## Shared machinery this node changed

| Change | Where | Why it is safe |
|---|---|---|
| `locked_joints` knob (+ `lock_solref` / `lock_solimp`) | [`../arm_env.py`](../arm_env.py) | An `equality/joint` constraint pinning named joints, **plus `gear="0"` on their motors** (soft constraints lose to a gear-8 motor: a first pass drifted pinned joints 0.16 rad per control step). Keeps the state 2n-dim and the command n-dim so one FM spans every body. **Additive and off by default** — with no `locked_joints` key nothing is emitted, no gear is altered, and the XML is byte-identical to the unlocked arm. Gated by `on_policy/verify_backcompat.py`. |
| `.add_local_python_source("rhm")` | [`../shared.py`](../shared.py) | Lets this node import the frontier instrument from `rhm.residual_decomposition.decomposition` rather than vendoring a second copy that could drift from the instrument of record. Pure numpy, empty package `__init__`s, no Modal app pulled in. No existing cut imports `rhm`, so every prior code path is unchanged. |

## Modal volume layout

```
/data/expansion_gate/<tag>/            results.json + fig_saturation_gate.png     (Piece 1)
/data/expansion_support_fixed/<tag>/   results.json + fig_support_fixed.png       (Pieces 2, 2b)
/data/expansion_support_growing/<tag>/ results.json + fig_support_growing.png     (Piece 3)
```

Tags: `gate_s{0,1,2}` (Piece 1) · `sf_s{0,1,2}` (Piece 2, fresh buffer) · `sf2_s{0,1,2}` (Piece 2b,
2×2 + accumulate) · `sg_s{0,1,2}` (Piece 3) · plus `calib`/`calib2`/`endpt`/`probe_diag` (the
sizing runs described in the README's gotchas). Local mirrors: `figures/<script>_<tag>/`.
Logs: `logs/`.

## Two design decisions worth reading before extending this node

1. **The basis is the plant's, not a model's.** `R_res_participation` counts the frontier "in the
   model's own basis, weighted by the computation actually done", which is why it reads 7 on RHM
   where naive rank reads 84. Here the basis is A's principal directions — the *plant's*
   displacement covariance on the probe, an ordering of the **data**, not of a computation. That is
   a weakened instrument, accepted on purpose. Note the spec in `mjc/README.md` before 2026-07-27
   ("basis = the FM's hidden covariance, residual = k-step rollout error against the plant") is
   **not computable as written**: `repaired_triple` needs A and P in the same space, and those are
   256-d and 2n-d. The coherent options are the state-space basis used here and a learned-latent
   basis.
2. **β needs a learned latent; so, independently, does honesty about the prediction target.**
   β is a log-log slope across *participating* directions and the arm's state target supplies 6 or
   10. It failed capacity-invariance by 6–59× here. Separately, predicting `qpos/qvel` is access to
   the simulator's own generalized coordinates that no embodied learner has. Both point the same
   way, and the cheap version is FK-derived observations (link-endpoint Cartesian positions)
   through a learned encoder — no renderer required.
