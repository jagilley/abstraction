# Confabulation Test (language) — File & Auxiliary-README Index

Full file-by-file reference for the sub-experiment. Summarized in [README.md](README.md);
this is the complete listing.

## Code files

| File | Purpose |
|---|---|
| `confabulation.py` (`residual_diagnostics`) | Standalone cheap re-read of the two junk-residual discriminators (`ens_cos`, `_residual_structure`) from saved wake checkpoints — no report heads, no observer ladder, no steering. Exists because those numbers gate the interpretation of every IMPL number but cost almost nothing to compute, so correcting them should not require re-running the whole battery |
| `confabulation.py` | **The whole battery.** Forks `a2a_forward/stages.py::a2a_loop_train`'s wake recipe (Run-6 controlled-retrain shape: shared init/seed/data order, only the loop differs), then asks whether a self-report tracks M's *implementation* or a *self-theory*. Report targets IMPL / IMPL_COS (FM-residual direction, categorical + continuous) / BEHAV / ENT / WORLD emitted from `post_block3`; four tests: capacity-swept third-person observer ladder (`O_input`/`O_io`/`O_act` + half-data control), channel ablation (shuffle/zero of `r` vs `FM(a_i)`), matched-KL residual-vs-prediction span steering, and a trained-under-ablation confabulator control. **Every IMPL number is swept over instrument-FM capacity (`d_head` × `mlp_mult`, ~5%→42% of the predicted blocks) and gated on `ens_cos` + syntactic-η²** — an over-capacity FM leaves a junk residual that M can report and no observer can predict, which fakes the headline. Design: [ideas/confabulation_test.md](../../../ideas/confabulation_test.md) |
| `__init__.py` | Package marker so `a2a_forward.confabulation` imports under Modal's `add_local_python_source("a2a_forward")` |

## Key entrypoints

| Command | What it does |
|---|---|
| `modal run a2a_forward/confabulation/confabulation.py::confabulation_test --smoke` | Wiring check (~minutes). Tiny everything; numbers meaningless |
| `modal run --detach a2a_forward/confabulation/confabulation.py::confabulation_test --conditions "cl,ol" --tag main` | Headline run: both arms, full battery, instrument sweep (~4 h on an L4) |
| `modal run a2a_forward/confabulation/confabulation.py::residual_diagnostics --conditions "cl,ol" --tag diag` | Just the two junk-residual discriminators (`ens_cos` + residual structure) re-read from saved wake checkpoints — ~10 min instead of the ~4 h battery |

**Workspace gotcha**: the FineWeb-Edu token shards are on the **`jagilley`** volume, not
`chromatic`. Run `modal profile activate jagilley` first.

## Notable flags

| Flag | Default | Why you would change it |
|---|---|---|
| `--predict-to` | `post_block2` | `post_block3` runs the canonical a2a gap, but puts the report site *at* the FM target, degenerating Tests 2/3 into a linear readout |
| `--inst-caps-str` | `16:0.5,4:0.25,32:1.0,64:2.0` | `(d_head:mlp_mult)` instrument sweep. Index 0 is the default capacity, where the FM-independent targets and the full observer ladder run |
| `--ens-n` | `3` | Number of independent fresh FMs per capacity point, for `ens_cos` |
| `--observer-causal` | off | Restores RHM's causal observer stack; the default bidirectional observer is strictly more generous to the third party |
| `--full-ladder-every-cap` | off | Runs the full observer ladder at every instrument capacity (~4× observer cost) instead of only at the default |
| `--report-native` | off | Additionally re-reads the CL arm through its native injected forward pass rather than standalone |
| `--obs-topk` | `64` | Size of the top-k output summary handed to `O_io` (four full-distribution scalars are always included alongside) |
| `--component-control` | off | Test 1b. Adds `PRED`/`PRED_COS` (the theory-visible component `FM(a_i)`) and `AJ`/`AJ_COS` (the whole state) as report targets alongside `IMPL`, with identical k-means/head/ladder, and prints the headroom-normalized `frac` column. See [component_control/README.md](component_control/README.md) |
| `--skip-fixed-targets` | off | Skips `BEHAV`/`ENT`/`WORLD`. They do not depend on the instrument, so a control run that only needs the IMPL-family ladders can drop them |
| `--skip-steering` | off | Skips Test 3. Steering needs the `BEHAV` head as its matched-behaviour control, so it is additionally gated on `--skip-fixed-targets` |

## Auxiliary READMEs

| File | Summary |
|---|---|
| `README.md` | **The writeup.** Positive on the core dissociation (implementation targets +0.21…+0.37 advantage over the best third-party observer at every instrument capacity; I/O-map and input controls at 0 or *negative* — `ENT` is −0.287, the observer predicts M's entropy better than M reports it), clean channel ablation (`shuffle_r` collapses the report, `shuffle_p` does not), modest matched-KL steering (1.4–2.1×). **Null on loop-necessity** (CL−OL ≤0.004 everywhere) with the leading confound named: the CL arm is net *worse* than OL here, and its co-trained FM is at 42% of the predicted blocks, i.e. saturation. **Disagrees with the RHM sibling**, which found CL > OL on every secondary measure — unresolved, with three candidate explanations and the tests that would separate them. Also documents the location-vs-scale η² gotcha and the `after_punct` taxonomy difference |

## Children

| Folder | Summary |
|---|---|
| [`component_control/`](component_control/README.md) | **Test 1b — is the advantage about the residual, or about having the state?** Varies which function of `a_j` is reported (`PRED` = `FM(a_i)`, `AJ` = the whole state) against `IMPL` = the residual, with identical machinery. The best I/O observer reaches ~0.87 / ~0.84 of the achievable headroom on `PRED` / `AJ` and only 0.19–0.26 on `IMPL`, so the forward-model decomposition is load-bearing for Finding 1 and not only for the steering test. Also: a tokens-only observer nearly solves `PRED` at 1L/64D while staying near the floor on `IMPL` at every capacity. Weaker on the continuous targets, where the separation from `AJ` depends on the headroom normalization. Language only, two of the four instrument capacities |
| [`../fsm_part3/`](../fsm_part3/README.md) | **Part 3 (2026-09-13/14): does the privilege extend past the current residual?** Three nodes that fork this battery: authorship (public, more legible from outside than inside), and dispositional targets on language and grammar (the residual's past is private where the computation is deep and behaviorally silent; behavior is public in every cut). Lives in its own folder because it spans both substrates; see its README and [`../fsm_part3/FILES.md`](../fsm_part3/FILES.md). |

## Related, outside this folder

| Path | Relation |
|---|---|
| [ideas/confabulation_test.md](../../../ideas/confabulation_test.md) | The design doc this implements |
| [experiments/rhm/rhm_confabulation.py](../../rhm/confabulation/rhm_confabulation.py) | Sibling instantiation on RHM; source of the instrument-capacity sweep and the `ens_cos` gate |
| [../stages.py](../stages.py) | `a2a_loop_train` — the wake recipe this forks |
| [../README.md](../README.md) | Parent experiment; Run 6 (controlled retrain) and OOD_ROBUSTNESS (self-knowledge is computational, not epistemic) are the load-bearing priors |
