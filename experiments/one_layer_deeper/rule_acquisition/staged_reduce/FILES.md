# Files — `staged_reduce`

**Up**: [README.md](README.md) (this node) · [../README.md](../README.md) (rule_acquisition) · [../../README.md](../../README.md) (one_layer_deeper)

## Code files

| file | purpose |
|---|---|
| `staged_reduce.py` | The cut. One Modal entrypoint, `staged_reduce`; an arm is `BASE_ARM` plus its overrides in `ARMS`. Trains only the single-stage reduce (`(N, y') -> y' mod N`, `y'` uniform on `[0, radix·N)`) and chains it **at test time** as schoolbook long division, `r ← (r·R + c_i) mod N`, re-grounding the remainder through the model's own decode. Knobs: `radix` (0 ⇒ the monolithic `S=1` control over `[0, N^2)`, same code path and pools), `n_digits`, `min_margin`, `max_moduli`, `dense_n`, `inner_steps` (the compute-matched control — `S` operator applications per query, no re-grounding), plus the usual capacity set. `persist()` writes results, checkpoint and a `volume.commit()` at every log point with a `complete` flag, so a killed run keeps its whole `eps`-vs-budget curve. Carries `certifiable_T(eps) = ln2/(768·eps)` next to every number. |
| `reprobe.py` | Two checkpoint probes; trains nothing. (1) `reprobe` stratifies chain accuracy on `k`, the number of stage queries on the true stage path falling in the arm's training half, and builds a rejection-sampled `k = 0` pool — composite problems none of whose stage queries was ever trained on. This is what separates generalisation from coverage, and it is apples-to-apples against a monolithic arm because `S = 1` makes its ordinary held-out pool a `k = 0` pool. `P(k=0) = 2^-S`, so it is reachable at `S ≤ 8` and skipped at `S = 20`. (2) `dense_probe` scores a `dense_n` arm and a reference arm on **one shared pool**, per modulus, and aggregates by modulus class (parity, divisibility by 5, minimum prime factor, factor count, magnitude, and membership of the reference arm's own held-out set). It is what caught the degenerate dense selector in README §4. |
| `analyze.py` | Merges the per-tag result files into the four cuts' tables. Reports `eps` with exact error counts and pool resolution rather than accuracy alone, `certifiable_T`, peak-with-step alongside the endpoint and a tail slope for both the stage and chain curves (with a decay flag, since both monolithic controls over-train), the radix sweep, the rule-axis 2×2, and the chain-vs-`p^S` comparison. |

## The arms

| arm | cut | what it is |
|---|---|---|
| `sr3_r2` / `sr3_r10` / `sr3_r100` / `sr3_r1000` | §1 | the radix sweep at 3 digits — `S` = 20 / 6 / 3 / 2, quotient ≤ 1 / 9 / 99 / 999. `r1000` defined, not run; `r2` stopped at 125k |
| `sr3_mono` | §1 | `radix=0`, `S=1` over `[0, N^2)` — the control. Reproduces `exact_atom`'s `divqfull` at matched budget |
| `sr3_mono_inner6` | §1 | 6 tied-operator applications per query, no re-grounding — the compute-matched control |
| `sr4_r10` / `sr4_r100` | §1 | 4 digits, `S` = 8 / 4 — `exact_atom` §5's scissors cell |
| `sr4_mono` / `sr4_mono_inner8` | §1 | 4-digit controls; `sr4_mono` reproduces `div4_qfull`'s 0.0004. `inner8` defined, not run |
| `sr3_r10_many` / `sr3_mono_many` | §3 | 142 train / 36 held-out moduli (`min_margin=2, max_moduli=0`) — the rule axis. `r10_many` is the headline cell, run at seeds 0 and 1 |
| `sr3_r10_m8` | §3 | **the family-matched control** — 8 moduli subsampled from the same 178-modulus family the `_many` arms use, so only the count varies. Establishes that staging alone does not move the rule axis |
| `sr3_r10_dense` / `sr3_mono_dense` / `sr3_r2_dense` | §4 | `N` drawn from all 900 three-digit integers instead of ~178 semiprimes. `mono_dense` and `r2_dense` were cancelled for compute |

## Children

| child | one-liner |
|---|---|
| [`terminal_only/`](terminal_only/NOTES.md) | **The staged forward pass under terminal-only supervision** (2026-08-22/23; written up in [`dress_rehearsal/second_pass/`](../../dress_rehearsal/second_pass/README.md) §3). Same task, pools and parameter count as `sr3_mono*`; `S=6` tied stage applications with the remainder re-grounded through the model's own decode; loss on the final remainder only; 2×2 of 8/142 moduli × straight-through/soft seam. All four cells at floor, CE flat from 25k to 600k where the monolithic control descends; the seam collapses to ~12 distinct values. Architecture alone does not discover the decomposition. `terminal_only.py` (verbatim fork + diff table), `reprobe.py` (depth/restart/stagefn/crossprobe), `analyze.py`, `NOTES.md`, `FILES.md`, `results/<tag>/`. |

## Results layout

Modal volume `one-layer-deeper-data`; results at `/staged_reduce/<tag>/results_seed<N>.json`,
checkpoints at `/staged_reduce/<tag>/ckpt/<arm>_seed<N>.pt`, probes at
`/staged_reduce/<tag>/reprobe_seed<N>.json` and `/staged_reduce/<tag>/dense_probe_seed<N>.json`.
Local copies under `results/<tag>/`. Tags: `sr3a`, `sr3a2`, `sr3b`, `sr3c`, `sr3m8`, `sr4a`,
`sr4a2`, `sr4a3`, `sr4b`, `srb1`, `srb1r_s1`, `srb2`, `srb2r`.

## Gotchas

- **The `dense_n` held-out selector is degenerate.** `(N * 2246822519) % 10 == 0` was intended as
  a pseudo-random 10% hash, but `2246822519 ≡ 9 (mod 10)` and 9 is invertible mod 10, so the
  predicate is exactly `N ≡ 0 (mod 10)`. Every dense arm therefore holds out precisely the final
  digit class it never trains on. It is left unfixed so the finished runs stay reproducible; fix
  it before running any new `dense_n` arm. Scoped to `dense_n` — other arms take held-out moduli
  from `ModulusFamily.split()`, and the `(N, y)` split uses a different, sound hash.
- **Arms run sequentially inside a job.** A two-arm 600k job is ~3h against ~90min for singles,
  and killing a job mid-run forfeits any arm that has not started. Prefer one arm per job when
  you may want to stop early.
- **A tag rewrites its whole results file from the in-process dict.** Relaunching a second arm
  under an existing tag overwrites the first arm's curve. Use a new tag; `analyze.py --tags`
  reassembles.
- **Killing early is cheap.** `persist()` commits at every log point with `complete: false`, so a
  stopped run keeps its full curve and a usable checkpoint. Three arms in this node are partials.
- **`per_modulus` is only stored at `≤ 16` moduli**; min/median/max are always stored. Use
  `dense_probe` when the full breakdown is needed on a larger pool.
- **The chain model `p^S` is not calibrated.** It under-predicts when the per-stage rate is high
  and over-predicts badly when it is low (10× on held-out `N`). Measure the composed number.
- **Local launcher shells are reaped** long before these jobs finish. Launch with `--detach` and
  detect completion by fetching the result file. Inherited from every parent cut.
- **`modal volume get` needs `--force`** and a full per-file destination path.
