# Files — `rule_acquisition`

**Up**: [README.md](README.md) (this node) · [../README.md](../README.md) (one_layer_deeper)

## Code files

| file | purpose |
|---|---|
| `rule_acquisition.py` | The cut. Trains the **one-step** map at depth 1 with **half of every modulus's bases held out**, so lookup cannot cover the test set, and scores held-out `x` / held-out `N` against a **no-reduction floor computed on each arm's own eval pool** rather than quoted. One Modal entrypoint, `rule_acquisition`; an arm is `BASE_ARM` plus its overrides in `ARMS`, so the arm table *is* the complete statement of what varies. Four knob families — task (`sq`/`mul2`/`sqnomod`/`redmod`/`redmod_u`), representation (`number_base`, `abacus`), compute (`n_enc_layers`, `inner_steps`, `d_ff`, `d_op_ff`), pressure (`min_margin`, `max_moduli`) — plus `aux_product` (auxiliary CE on the un-reduced product, decoded off the encoder state) and `q_cap` (bounds the quotient for the uniform-`y` reduce arms). Also carries a per-modulus group-structure readout reusing `ballistic_depth/rule_structure/`'s two discriminating statistics, so a representation change is readable even when accuracy does not move. |
| `analyze.py` | Merges the per-family result files (the sweep is split across tags so parallel jobs do not clobber a shared output path) into one table: accuracy vs floor with the lift, the group-structure statistics against their permutation nulls, and held-out-`x` against training step as the grokking check. |

## Children

| child | summary |
|---|---|
| [`exact_atom/`](exact_atom/README.md) ([FILES](exact_atom/FILES.md)) | **How far the one-step map is from being *exactly* right.** Re-reads this cut against the exactness target Hard gates on: under `ballistic_depth` §9's re-projection at `k=1`, rung `T` needs `eps <~ 9e-4/T`, so each rung costs one factor of two and the whole ladder is 64×. **Retracts this cut's `sqnomod` 0.920** as a per-modulus-split artefact (clean split: **0.068**, seen-`x` exactly 1.000; a matched-count control across a 100× larger space reads 0.00003). Converts §3's coverage inference into a measurement: fed the true `Enc(DIV, N, x^2)` from inside a trained composed model, the reduce reads **0.953** on held-out `x` where `redmod` read 0.022. A pre-registered mechanism — generalisation appears where memorisation becomes infeasible — **failed in both directions**; 27.4M parameters gives the best transfer in the node (**0.909** at 5 digits, slope −3.70) and an arm that failed to memorise generalised at 0.001. The two widths tested bracket the problem: at 3 digits the reduce works and the multiply does not; at 4 digits the multiply improves (seam cos on held-out `x` 0.625 → 0.945) and the **full-range** reduce collapses to 0.0004 while bounded-quotient reads 0.986 — the reduce's limit is the *quotient range*, which `x^2` spans by construction. Held-out `N` is at or below floor everywhere, including under an oracle state. Method notes: a closure loss must be scale-free (MSE seam: 0.0010 at cosine **0.081**), and a closure statistic is only interpretable if its target branch is independently grounded (`sqpad_seamcos` reaches cos 0.9997 on held-out `x` and means nothing). Single seed; four pre-registered predictions failed. |

| [`staged_reduce/`](staged_reduce/README.md) ([FILES](staged_reduce/FILES.md)) | **Decomposing the reduce until its stages are coverable.** Acts on `exact_atom` §5's localisation of the reduce's failure to the *quotient range*: schoolbook long division makes every stage a bounded-quotient reduce, chained **at test time only** with the remainder re-grounded through the model's own decode — `ballistic_depth` §9's re-projection one level down, and exact rather than a manifold projection because the remainder is a digit string. Step-matched at 600k: staged **0.9968** vs monolithic 0.4841 at 3 digits, and **0.9899** vs **0.0005** at 4 digits (the control reproducing `div4_qfull`'s 0.0004). It is generalisation, not coverage — rejection-sampled chains where *no* stage query was ever trained on read **0.9925**. **The rule axis moves for the first time**: held-out `N` 0.9826 / 0.9790 across two seeds, per-modulus min 0.851 over 36 unseen moduli, against `redmod_q64_many`'s 0.058. But only as an **interaction** — a family-matched 8-modulus arm reads 0.0042 on held-out `N` while reading 0.9943 on held-out `y`, and breadth without staging moves 0.0012 → 0.0025. The compute control is **not** null (`inner6` recovers ~a third of the gap in `eps`). A reading that 774 moduli generalise worse than 142 is **retracted**: the dense held-out selector `(N·2246822519) % 10 == 0` is exactly `N ≡ 0 (mod 10)`, so it holds out the one final-digit class never trained on; on matched moduli the 774-arm is marginally *better* (0.9888 vs 0.9822). That bug exposed that transfer is uniform across minimum prime factor and magnitude but collapses on an unseen final digit. No cell certifies rung `T=1` (best `eps` 3.16e-3 against 9.0e-4). Headline is 2 seeds; the rest single seed. |

## The arms

| arm | family | change from `sq` |
|---|---|---|
| `sq` | T baseline | — (`x -> x^2 mod N`); a replicate of `variable_modulus/` §3's null |
| `mul2` | T | `x -> 2x mod N` — reduction with a quotient of at most 1 |
| `sqnomod` | T | `x -> x^2` — multiplication with no reduction |
| `redmod` | T | `x^2 -> x^2 mod N` — the general reduction, on exactly the inputs `sq`'s reduce sees |
| `redmod_q8` / `_q64` / `_qfull` | T | the same reduction with `y` drawn **uniformly**, quotient capped at 8 / 64 / `N-1`, input width fixed at 2w so only the quotient distribution moves. Separates "division is unreachable" from "division was undersampled" |
| `auxprod` | T | `sq` plus a head that must decode `x^2` off the encoder state — the two-stage structure handed to the model |
| `binary` | R | `N`, `x` and the answer in base 2 |
| `abacus` | R | one shared place-value embedding across the `N` and `x` fields |
| `enc8` / `inner8` / `wide4k` | C | 8 encoder layers / operator applied 8x per task step / `d_ff` 4096 |
| `manymod` | D | 8 -> 142 train moduli |
| `stack` | — | `binary` + `enc8` + `inner8` + `wide4k` + `manymod` |

## Results layout

Modal volume `one-layer-deeper-data`, results at `/rule_acquisition/<tag>/results_seed<N>.json`,
checkpoints at `/rule_acquisition/<tag>/ckpt/<arm>_seed<N>.pt`. Local copies land under
`results/<tag>/`. Tags: `cut1_task`, `cut1_repr`, `cut1_compute`, `cut1_stack`,
`cut2_decomp`, `cut3_quotient`.

## Gotchas

- **One output path per tag.** Every job writes `results_seed<N>.json` under its tag, so
  parallel jobs must use *different tags* or they overwrite each other. The sweep is split by
  knob family for this reason, and `analyze.py --tags` reassembles it.
- **`modal volume get` needs `--force`** and a full per-file destination path; without it an
  existing local file is silently skipped, and a directory download into a non-existent local
  path writes a *file* rather than a tree. Inherited from both parent cuts.
- **Local shells are capped well below job length here.** Launch with `--detach` and detect
  completion by fetching the result file; a killed launcher shell does not mean a killed job.
- **The floor is per pool and per task**, not a constant. `sq` and `redmod` share a floor by
  construction (same bases, same free cases) and are directly comparable; `mul2`'s is ~0.49
  because half its cases need no reduction; `sqnomod` has none.
