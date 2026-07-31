# FILES — `rhm/minting`

Complete file-by-file index for this node. The writeup is [README.md](README.md); the parent node is
[`../README.md`](../README.md).

## Code files

| file | what |
|---|---|
| [`mint_loop.py`](mint_loop.py) | The whole experiment. `prep` (one warm checkpoint per breadth, plus the `--sweep` mint-yield diagnostic), `arm` (one `(breadth, acceptance)` cell: rounds of train → frozen generative probe → mint → accept → fold in, then the depth probes), `minting` (orchestrator: spawns preps then arms in parallel, collects, prints the tables). Acceptance rules live in `_accept`; the dose knob is `_make_mixed_batcher`. |
| [`analyze.py`](analyze.py) | Re-derive one run's tables from saved arm JSONs without re-running: depth recovery + §7's falsifier contrasts, rule-violation rate, generative validity, mean parse depth, support expansion, pair entropy, duplication, acceptance sharpening, verifier yield. |
| [`aggregate.py`](aggregate.py) | Pool replicate seeds. Paired-within-seed contrasts with mean ± sem, per-seed sign counts and t, recovery fraction against the `real_data` ceiling, and the pool-level mechanism check. This is what produces the §4 tables. |

Shared primitive touched (lives at the parent node, purely additive): `../rhm_data.py`
**`possible_set_parse`** — exact CYK-style possible-set parse. Needed because `parse_leaves` builds
last-writer-wins inverse maps and false-rejects 99.98% of genuine sequences at v=16/m=4, so it could
not serve as the verifier. Every prior caller keeps the cheap approximate parse it was written
against.

## Results

| path | what |
|---|---|
| [`figures/q8_m3_s42/`](figures/q8_m3_s42) | headline run, seed 42 — all 10 arms (includes `verifier@4`, `mirror_dedup`, `broad:static`) |
| [`figures/q8_m3_s1/`](figures/q8_m3_s1), [`figures/q8_m3_s2/`](figures/q8_m3_s2) | replicate seeds — the essential 7 arms |
| [`figures/dose_m3_w0.5/`](figures/dose_m3_w0.5) | the over-concentrated dose run (README §3, middle row) — kept because the failure is the evidence |
| [`figures/AGGREGATE_3seed.txt`](figures/AGGREGATE_3seed.txt) | `aggregate.py` output over the three headline seeds |

Live results also on the `rhm-scaling-data` volume under `rhm_minting/`. Warm checkpoints are keyed
by regime, pool, warmup steps **and seed** — replicate seeds must not share one.

## Gotchas

- **`--mint-weight -1` is pooled sampling** (minted share = its size share). Values `>= 0` fix the
  gradient share instead, which README §3 shows damages every arm equally, `real_data` included.
- **The acceptance quota must bind** or pools diverge and the arms stop being matched on volume.
  Check `quota met every round` in `analyze.py`'s last line; size `n_candidates` from the yield
  diagnostic with ≥1.3× margin.
- **`mirror_dedup` is only distinguishable from `mirror` when the pool is being memorised.** At a
  healthy pool size they are bit-identical, which is itself the diagnostic.
- **`modal run --detach` log streaming died ~1h before job completion twice.** Poll the volume for
  expected files instead of tailing logs.
