# Files — `staged_reduce/terminal_only`

**Up**: [../README.md](../README.md) (staged_reduce) · [../../README.md](../../README.md) (rule_acquisition) · [../../../README.md](../../../README.md) (one_layer_deeper)

No `README.md` yet — the numbers have not been discussed. Running notes are in
[NOTES.md](NOTES.md).

## What this node asks

The donor [`../staged_reduce.py`](../staged_reduce.py) trains **only on the single-stage
distribution** (uniform `y' ∈ [0, R·N)`, label `y' mod N`) and chains at *test time*. That
distribution is self-made — legal under the competition's rules, but data augmentation the
participant invents, so the result is not portable to a setting where the only supervision is
the terminal label.

This node keeps the terminal label and moves the decomposition into the **forward pass**:
`S` applications of one tied stage map, remainder re-grounded at each seam through the model's
own decode, loss on the final remainder only. The question is whether a staged forward pass
can *self-organise* long division from terminal labels alone, given breadth of moduli.

## Code files

| file | purpose |
|---|---|
| `terminal_only.py` | The cut. A fork of `../staged_reduce.py` (donor unedited) — prompt layout, architecture, parameter count (2,122,782 at w=3), modulus family, hash split, `_div_pool` / `_chain_pool` and `persist()` are verbatim, so every pool is bit-identical to the donor's for a matching modulus set. Changed: training distribution is `y ~ U[0, N²)` (the whole task, as `sr3_mono`); supervision is the **terminal** remainder only; the forward pass is `S` tied stage applications with the remainder re-grounded between them. Two re-grounding modes — `st` (argmax one-hot forward, straight-through gradient; the seam channel is exactly `w` decimal digits and the forward is bit-identical to the donor's test-time chain) and `soft` (convex combination of the real digit embeddings written in place into the remainder's own digit slots; `ballistic_depth.forward_soft_digits` / `dress_rehearsal` v3's mechanic one level down). Optional label-free seam terms (`ent`, `cos`), off by default. |
| `reprobe.py` | Checkpoint probes; trains nothing. `reprobe` runs three: **depth** (pad the dividend with extra leading zero radix digits and run `S+k` stages — a genuine tied stage operator treats them as no-ops), **restart** (inject the true partial remainder before stage `j` and run the rest natively — separates a poor per-stage rate from error propagation, `ballistic_depth` §8's cold-start probe at the stage seam), and **stagefn** (exhaustive read of the tied map over `y' ∈ [0, R·N)` for every train and held-out modulus, broken down by the true quotient digit). `crossprobe` scores a terminal-only arm and a donor arm on one shared pool, per modulus. |
| `analyze.py` | Merges this node's tags with the donor cells read straight out of `../results/` (the pools are bit-identical, so they belong in one table). Four tables: the 2×2 (`chain_heldout_y` / `chain_heldout_n` × {8, 142 moduli} × {monolithic, staged-supervised, terminal-only}); the stage probe; the per-stage diagnostics; and tail power-law slopes with a decay flag. |

## The readouts

| readout | what it is |
|---|---|
| `chain_*` | the arm's **native** staged forward on the donor's chain pools (`y ~ U[0, N²)`, modulus-uniform, same hash split). `chain_heldout_y` and `chain_heldout_n` drop straight into the donor's tables. |
| `chainhard_*` | (soft arms only) the same weights scored with an **argmax** chain — separates "the decomposition is discrete" from "the seam is a continuous side-channel". |
| `stage_all` / `heldout_y` / `seen_y` / `heldout_n` | the tied map applied **once** to a bounded-quotient query `(N, y')`, `y' ∈ [0, R·N)`. These are the donor's `sr3_r10` *training* pools, which a terminal-only arm never sees. `sr3_r10` reads 0.9978 on `heldout_y` having trained on it. The strongest single probe in the node. |
| `final_stage_probe` | per stage index, on the chain pool: `true_match` (decoded == `(y // R^i) mod N`), `in_range`, `identity` (== its predecessor), `modal_share`, and `purity` / `inv_purity` against a **within-modulus permutation null**. Purity is trivially 1.0 at the first stages and cheap whenever the decode is near-constant, hence the null. High purity with low `true_match` is a *relabelled* residue — a decomposition discovered outside the schoolbook coordinate. |
| `final_chain_teacher` | the donor's oracle re-projection: true partial remainder injected at every stage, giving the per-stage rate free of propagation. |

## The arms

| arm | moduli | re-ground | against |
|---|---|---|---|
| `to3_st10` | 8 (`min_margin=20`) | straight-through hard | `sr3_mono` 0.4841 / 0.0012, `sr3_r10` 0.9968 / 0.0025 |
| `to3_soft10` | 8 | soft | same |
| `to3_st10_many` | 142 (`min_margin=2`) | straight-through hard | `sr3_mono_many` 0.2527 / 0.0025, `sr3_r10_many` 0.9943 / **0.9826** |
| `to3_soft10_many` | 142 | soft | same |

Defined and held in reserve: `to3_mono` / `to3_mono_many` (the `S=1` endpoint, identical to the
donor's `sr3_mono*`), `to3_st10_m8` (family-matched 8, against `sr3_r10_m8`), `to3_st100_many`
(`S=3`), `to3_soft10_ent_many` / `to3_soft10_cos_many` (the label-free seam terms),
`to4_st10_many` (4 digits — only if 3 digits moves).

## Results layout

Modal volume `one-layer-deeper-data`; results at
`/staged_reduce/terminal_only/<tag>/results_seed<N>.json`, checkpoints at
`/staged_reduce/terminal_only/<tag>/ckpt/<arm>_seed<N>.pt`, probes at
`.../reprobe_seed<N>.json` and `.../crossprobe_seed<N>.json`. Local copies under
`results/<tag>/`.

## Gotchas

Inherited from the donor and confirmed here:

- **Arms run sequentially inside a job**, so one arm per tag when arms may want stopping
  independently. A 600k-step `S=6` arm is ~7.5–8 h on an L4 (~22 step/s against the donor's
  ~110 at `S=1`), which is why `TIMEOUT` defaults to 79200 s here rather than 43200.
- **`persist()` commits at every log point** with a `complete` flag, so a stopped run keeps its
  whole curve and a usable checkpoint.
- **A tag rewrites its whole results file** from the in-process dict; relaunching a second arm
  under an existing tag overwrites the first.
- **`modal volume get` needs `--force`** and a full per-file destination path.
- **The `dense_n` selector is degenerate** in the donor (`(N*2246822519) % 10 == 0` is exactly
  `N ≡ 0 mod 10`). It is carried into the fork unfixed so the fork stays a fork; **no arm here
  sets it**, and any new one needs a real hash first.
- **`radix` must be a power of ten** in this fork — the soft re-encode is digit-aligned
  (`r·R + c` is `r`'s digit string followed by `c`'s, right-aligned in the `2w` field), which
  is what makes the in-place re-encode exact. The donor's `radix=2` arms have no analogue here.
- **Purity needs its null.** At stage 0 and 1 the stage input is a deterministic function of
  `(N, true partial remainder)`, so purity is 1.0 by construction; and a near-constant decode
  scores high purity for free. Always read `purity` against `purity_null`.
