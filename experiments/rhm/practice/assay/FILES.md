# FILES — `assay` (which ingredient of the complete vocabulary is the currency?)

Machinery record. The unit writeup lives at [`../census/README.md`](../census/README.md)
(post-discussion, 2026-08-25); [`README.md`](README.md) here is the pointer. This file is the
assay node's per-file, per-gate, per-decision reference.

**Up**: [`SPEC.md`](SPEC.md) · parent arc [`../README.md`](../README.md) · direct parents
[`../census/`](../census/FILES.md) (`cs_s0` and `figures/forensics/report.txt`, required
reading) and [`../spiral/`](../spiral/FILES.md).

## Fork base

`assay.py` forks **`../census/census.py`** verbatim and adds four things, each marked
`# [assay]`. `census/`, `spiral/` and all donors are untouched.

| addition | where |
|---|---|
| **oracle surgery at commit** | `apply_surgery` / `rows_from_flats`, applied in `run_arm` *before* the commit body — so the priced pre-commit audition grades what is actually committed, and an emptied table cancels the commit through the donor's existing guard (the `strip`-empties-L3 risk the SPEC named) |
| **per-execution entry identity** | `macro_features_rec` + `_install_entry_recorder`. `macros.macro_features` ends at `best = cur.argmax(-1)` and discards it; the recording copy accumulates a per-entry histogram **on-device** (no host sync), never touches `counts`, and draws no RNG. Phase-tagged `beam` (priced) vs `probe` (unpriced) |
| **the realistic junk pool** | `build_junk_pool`, reading `log["miner"][...]["keys_at_support"]` from `census/cs_s0` and `spiral/sp_s0` on the volume and keying against `MC.true_tables` |
| **auditability** | every surgery logs its own table verbatim (flat tuples + truth mask), the mined table's grade beside it, and the pool + draw go into `setup.json` |

## Arms

| arm | table committed at each level | axis |
|---|---|---|
| `anchor` | own mined (`spiral_route` replica) | reference **and the full-scale fidelity carrier** |
| `complete` | own ∪ all missing true | amount ↑, truth ↑ |
| `exact` | the full true table | complete content at commit-time **arrival** |
| `junk_dose` | own ∪ K realistic junk, K = `complete`'s addition count | amount ↑, truth ↓ |
| `given_c1` | true tables from c1 (`given_route` replica) | the c1-arrival ceiling |
| `strip` | own minus its junk | truth ↑, amount ↓ (most cuttable, run last) |

Arm order `anchor,complete,exact,junk_dose,given_c1,strip` is a budget-risk decision: the
fidelity carrier first, the two cells the round exists to price next, then the mechanism dose,
then the bracket's top, and the most cuttable arm last.

## Gates (all measured before launch)

| gate | result |
|---|---|
| **entry-recorder bit-identity** | max\|Δ\| **0.0** on both `feats` and `pos` over 48 macro applications, recorder off *and* on; 1,152 selections recorded at levels 2–3 |
| **G-F smoke scale** (in-process vs `census.py`, instrument installed, 2 arms) | **0.000e+00**, self-replay control 0.000e+00, commits identical |
| **G-F full scale** | `as_s0/anchor` vs `cs_s0/spiral_route` over **all 116 cycles** — same spec, same ladder, same stream, so unlike `cs_s0` (32-cycle prefix) the whole run is comparable. Computed in the reduction |
| **twins with the instrument ON** | all four surgery arms **0.000e+00** vs the anchor up to their own first surgery — the assertion the SPEC asks for by name |
| **surgery does what its name says** | `strip` precision 1.0 and adds nothing; `complete` adds only true; `exact` installs the full true table; `junk_dose` adds **zero** true entries; dose matched at L2 (K=7, added 7, shortfall 0) |
| **`exact` ≡ `given_c1` content** | identical end-of-run grade (L2 14/14, L3 56/56, recall and precision 1.0) — the arrival axis is clean |
| **G-Y soundness** | PASS (hand counts == measured, node indexing in range all five eras) |

**Junk pool**: L2 **8** mined-but-false of 21 distinct; L3 **45** of 72. Provenance `cs_s0`
(6 arms, 78 keys) + `sp_s0` (7 arms, 15 new keys), logged in `setup.json`.

**Caveat on the arrival axis, stated**: `exact` carries the anchor's torch stream and
`given_c1` carries `given`'s, exactly as `spiral_route` and `given_route` did in `cs_s0` — so
the pair differs in arrival *and* in stream position. The stream's own size is bounded by
`cs_s0`'s twin data rather than zero, and the reduction says so.

## Runs

| tag | what | outcome |
|---|---|---|
| `gf_smoke` | G-F, in-process vs `census.py` with the instrument installed | PASS, 0.000e+00 |
| `smokeA` | `assay_run --quick`, all six arms | complete, 829 s; `given_c1` recorded 27,676 L2 / 15,160 L3 beam entry selections |
| `as_s0` | **the main run**, six arms | complete, 8,254 s = 2.29 GPU-h; full-length G-F vs `cs_s0/spiral_route` **0.000e+00 over all 116 cycles** |
| `smokeJ` | 4-arm quick smoke of the displaced twins | complete, 567 s; `e` diverges from c1 in both pairs |
| `as_s1` | **the stream-displaced twins** `given_c1_j` / `exact_j` | complete, 3,291 s = 0.91 GPU-h; substrate asserted identical to `as_s0` before any arm ran |

## The stream-position floor (`as_s1`)

Each twin is its original in every config bit, with the per-arm RNG stream displaced by a
**1024-draw burn on both the CPU and CUDA generators**, placed immediately after the per-arm
seeding. Both generators are burned because both are consumed every cycle by the plant's
masking in `finetune_generator` (`torch.randint` on CPU, `torch.rand(..., device=device)` on
CUDA); dropout is **not** a consumer here — every net is built with `dropout=0.0` — so the burn
point was chosen by reading what actually draws.

| pair | eras 3–5 \|Δ recovered fraction\| | max | mean |
|---|---|---|---|
| `exact_j` vs `exact` (anchor stream) | 0.079 / 0.015 / 0.087 | **0.087** | 0.061 |
| `given_c1_j` vs `given_c1` (given stream) | 0.083 / 0.148 / 0.344 | **0.344** | 0.192 |

Both twins diverge from **c1**. The arrival residual `given_c1 − exact` is +0.139 / +0.482 /
+0.525 in eras 3/4/5, i.e. **1.67× / 3.24× / 1.52×** the larger measured floor.

**The bracket's top is itself stream-dependent**, so the denominator moves with it:
`given_c1` +1.152 vs `given_c1_j` +1.003 in era 4, and +0.934 vs +0.590 in era 5. Bracket
fractions must be quoted against both (e.g. `complete` era 5: 0.425 with `given_c1` as the top,
0.803 with `given_c1_j`).

**Certification cycles move under displacement alone**: `given_c1` L3 c72 → `given_c1_j` c61
(11 cycles); `exact` L3 c70 → `exact_j` c56 (14 cycles).

## Reproduce

```bash
cd experiments/            # MODAL_PROFILE=chromatic
modal run rhm/practice/assay/assay.py::preflight
modal run rhm/practice/assay/assay.py::fidelity_smoke
modal run rhm/practice/assay/assay.py::assay_run --quick --tag smokeA

python3 rhm/practice/assay/launch_detached.py --fn assay_run --tag as_s0 \
    --eras "1:25:48,2:12:40,3:6:12,4:3:9,5:1:7" \
    --arms "anchor,complete,exact,junk_dose,given_c1,strip" \
    --budget 8 --g-budget 482 --n-corrupt 1 --mine-cap 8 --gy-level 4 --entry-rec \
    --span-min-hold 128 --collect-task-matched --tm-episodes 8192 \
    --recert-every 5 --n-aud 192 --probe-every 8 --n-rt 384 --n-score 256 \
    --sil-cv 0.15 --lp-min-drop 0.10 --seed 0

python3 rhm/practice/assay/analyze_assay.py --tag as_s0 --fetch --figures

# the stream-displaced twins (bounds the arrival-vs-stream confound)
python3 rhm/practice/assay/launch_detached.py --fn assay_run --tag as_s1 --ref-tag as_s0 \
    --arms "given_c1_j,exact_j" --eras "1:25:48,2:12:40,3:6:12,4:3:9,5:1:7" \
    --budget 8 --g-budget 482 --n-corrupt 1 --mine-cap 8 --gy-level 4 --entry-rec \
    --span-min-hold 128 --collect-task-matched --tm-episodes 8192 \
    --recert-every 5 --n-aud 192 --probe-every 8 --n-rt 384 --n-score 256 \
    --sil-cv 0.15 --lp-min-drop 0.10 --seed 0

python3 rhm/practice/assay/analyze_assay.py --tag as_s0 --merge-tag as_s1 --fetch --figures
```

`--merge-tag` joins `as_s1`'s arms into `as_s0`'s reduction after asserting the two tags trained
the same substrate (refs element-wise plus the stale-buffer pair). A separate tag rather than a
write into `as_s0`, because `summary.json` is rebuilt per invocation and the reduction is driven
by `summary["order"]` — writing the twins into `as_s0` would have orphaned its six originals.
