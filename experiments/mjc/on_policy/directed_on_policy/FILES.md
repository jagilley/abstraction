# `directed_on_policy/` — file index

Up: [README.md](README.md) · parent: [`../README.md`](../README.md) · node: [`../../README.md`](../../README.md)

## Code files

| file | purpose |
|---|---|
| `directed_on_policy.py` | **E3 / the prize** — the retracted `ballistic/directed` S2 directed-collection loop (inner FM-readapt + outer `lprog×visits` where-to-collect value loop + ballistic control), on the ON-POLICY arm. Regions are a tip-space ladder (1 on-reach reducible target + off-reach reducible + off-reach noise distractors), classified by actual path visitation; reducible gains drift under a continuous OU walk. The per-region learning-progress SURVEY is itself on-policy and **charged to the same `Body` budget** (monitor:collect ~1.8×, not S2's 22×). Policies: `uniform,oracle,value,lprog-only,visits-only,error-only`. |
| `directed_on_policy_agg.py` | Cross-seed aggregator: the ladder table (region-A / off / ballistic per policy), ceiling-normalised region-A recovery, the value-vs-ablation seed tallies, the noisy-TV comparison (value/lprog vs error-only + budget-share-to-noise), the abandonment per-region errors, and `fig_recovery.png` (region-A error + ballistic vs round). |

## Modal volume layout

```
/data/directed_on_policy/<tag>/results.json
```
Local mirror + figures: `figures/directed_on_policy_<tag>/` (PNGs regenerable, git-ignored per the repo convention).
