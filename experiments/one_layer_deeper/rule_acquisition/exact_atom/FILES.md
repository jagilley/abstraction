# Files — `exact_atom`

**Up**: [README.md](README.md) (this node) · [../README.md](../README.md) (rule_acquisition) · [../../README.md](../../README.md) (one_layer_deeper)

## Code files

| file | purpose |
|---|---|
| `exact_atom.py` | The cut. One Modal entrypoint, `exact_atom`; an arm is `BASE_ARM` plus its overrides in `ARMS`. Three task families behind one shared instrument so error rates are comparable across them: `div` (`(N, y) -> y mod N`, `y` uniform with `q_cap` bounding the quotient), `mul` (`x -> x^2`, `x` uniform over `[0, 10^w)`), and `sqdiv` (the composed atom, with a task token selecting the SQ or DIV branch through one shared encoder/operator/decoder). Knobs: `q_cap`, `train_frac`, `dense_n` (draw `N` from every `w`-digit integer rather than the semiprime family — legal because the reduce needs no factorisation and no periodicity margin), `div_aux`, `seam` + `seam_metric` + `seam_warmup`, `global_x_split`, and the usual capacity set. Carries `certifiable_T(eps) = ln2/(768·eps)`, the ladder-rung consequence printed next to every number. |
| `reprobe.py` | Two checkpoint probes the training-time instrument could not supply; trains nothing. (1) Re-evaluates `div` arms **per modulus** and reports the modulus-uniform mean, because the exhaustive training-time pool weights a modulus by its own space size (`N^2` at full quotient range, a 32× spread) while training samples moduli uniformly. (2) For `sqdiv`, measures seam MSE, seam **cosine**, and encoder-output norms on held-out `x` — the three together distinguish "the constraint installed" from "the loss was minimised by shrinking" — plus the **oracle seam** readout, which feeds the operator and decoder the true `Enc(DIV, N, x^2)` and asks whether the rest of the pipeline works. That is `ballistic_depth` §8's cold-start probe moved from the rollout seam to the multiply/reduce seam. |
| `analyze.py` | Merges the per-tag result files into the five cuts' tables. Reports `eps` with exact error counts and pool resolution rather than accuracy alone, the ladder-rung table, a power-law fit to the tail of each `eps`-vs-budget curve (slope ≈ 0 ⇒ asymptote) with the extrapolated step count to reach rungs T=1 and T=64, the memorisation-boundary grid with params and training-set size, and the composed-atom 2×2. |

## The arms

| arm | cut | what it is |
|---|---|---|
| `mul3` / `mul4` / `mul5` | §1 | `x -> x^2` at `w = 3/4/5`, half the space held out — 500 / 5k / 50k training inputs |
| `mul5_sparse` | §1 | ~500 training inputs across a 100× larger space; the control separating "small space" from "real transfer" |
| `mul4_sparse` | §1 | the intermediate point on the same axis (defined, not run) |
| `memb_d128` / `memb_d512` / `memb_d512L8` | §2 | capacity ladder at fixed 50k inputs — 0.55M / 8.5M / 27.4M params |
| `memb_n5k` / `memb_n20k` | §2 | training-set size at fixed capacity, same `w=5` space |
| `divq8` / `divq64` / `divqfull` | §3 | the reduce with quotient capped at 8 / 64 / `N-1`, 600k steps, `eps` logged 24× |
| `divq64_wide` | §3 | capacity control for the reduce (defined, not run) |
| `divq64_densen` / `divqfull_densen` | §3 | `N` drawn from all 900 three-digit integers instead of ~178 semiprimes — the rule-axis attempt |
| `sqpad` | §4 | the composed atom, no intervention; the parent's `sq` with the `x` field padded to 2w |
| `sqpad_div` | §4 | + dense uniform-`y` division auxiliary through the same modules |
| `sqpad_seam` / `sqpad_div_seam` / `sqpad_div_seam_densen` | §4 | the 2×2 with the **scale-dependent** MSE seam; kept verbatim because they are the record of a constraint that did not install (MSE 0.0010 at cosine 0.081) |
| `sqpad_seamcos` | §4 | seam with no division auxiliary — the control showing an unanchored target makes the statistic vacuous (cos 0.9997 on held-out `x`, oracle 0.004) |
| `sqpad_div_seamcos` / `_w10` | §4 | the seam that installs, at weight 1 and 10 |
| `sqpad_div_seamcos_gx` | §4 | the same arm under the clean global `x` split |
| `div4_q64` / `div4_qfull` | §5 | the reduce alone at 4 digits — the diagnostic that makes a composed null assignable to a half |
| `sq4_plain` / `sq4_div_seamcos` | §5 | the composed atom at 4 digits, clean `x` split |

## Results layout

Modal volume `one-layer-deeper-data`, results at `/exact_atom/<tag>/results_seed<N>.json`,
checkpoints at `/exact_atom/<tag>/ckpt/<arm>_seed<N>.pt`, checkpoint probes at
`/exact_atom/<tag>/reprobe_seed<N>.json`. Local copies land under `results/<tag>/`. Tags:
`divq8`, `divq64`, `divqfull`, `divdensen`, `mul_a`, `mul_b`, `sq_a`, `sq_b`, `sq_c`, `sq_cos`,
`sq_cos10`, `sq3gx`, `memb_cap`, `memb_cap8`, `memb_n`, `div4a`, `div4b`, `sq4a`, `sq4b`.

## Gotchas

- **A split on `(N, x)` is contaminated for anything `N`-independent.** The parent cut split
  each modulus's unit group by its own permutation; `x -> x^2` does not depend on `N`, so a
  held-out `x` under one modulus was a trained pair under another with probability `~1-0.5^k`.
  That is the whole difference between `sqnomod`'s 0.920 and `mul3`'s 0.068. `global_x_split`
  keys on `x` alone and is the only honest setting for the multiply.
- **A closure loss must be scale-free.** The encoder's final LayerNorm gain is shared across
  branches, so an MSE seam is minimised by shrinking both sides; everything downstream
  re-normalises, so nothing pushes back. Always report cosine *and* norms, never MSE alone.
- **A closure statistic is only interpretable if its target is independently grounded.**
  `sqpad_seamcos` reaches cos 0.9997 on held-out `x` and means nothing, because with no
  division loss the target branch is free to be whatever the source branch is. The
  in-distribution-competence version of this is `ballistic_depth` §10.
- **Exhaustive enumeration is not the same weighting as training.** Enumerating every `(N, y)`
  weights a modulus by `N^2`; training draws moduli uniformly. At `q_cap=inf` that is a 32×
  spread across a 3-digit family and it visibly moves `divqfull`. `reprobe.py` reports both.
- **Resolution bounds the claim.** `eps ~ 1e-5` cannot be read off a 4096-example pool. Pool
  size and `1/n` are printed with every number for this reason.
- **`modal volume get` needs `--force`** and a full per-file destination path. Inherited from
  every parent cut.
- **Local launcher shells get killed well before these jobs finish.** Launch with `--detach`
  and detect completion by fetching the result file; `sq_b` lost its client at step 287.5k/300k
  and saved normally.
