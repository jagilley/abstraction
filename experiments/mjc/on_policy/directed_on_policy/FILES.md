# `directed_on_policy/` — file index

Up: [README.md](README.md) · parent: [`../README.md`](../README.md) · node: [`../../README.md`](../../README.md)

## Code files

| file | purpose |
|---|---|
| `directed_on_policy.py` | **E3 / the prize** — the retracted `ballistic/directed` S2 directed-collection loop (inner FM-readapt + outer `lprog×visits` where-to-collect value loop + ballistic control), on the ON-POLICY arm. Regions are a tip-space ladder (1 on-reach reducible target + off-reach reducible + off-reach noise distractors), classified by actual path visitation; reducible gains drift under a continuous OU walk. The per-region learning-progress SURVEY is itself on-policy and **charged to the same `Body` budget** (monitor:collect ~1.8×, not S2's 22×). Policies: `uniform,oracle,value,lprog-only,visits-only,error-only`. |
| `directed_on_policy_agg.py` | Cross-seed aggregator: the ladder table (region-A / off / ballistic per policy), ceiling-normalised region-A recovery, the value-vs-ablation seed tallies, the noisy-TV comparison (value/lprog vs error-only + budget-share-to-noise), the abandonment per-region errors, and `fig_recovery.png` (region-A error + ballistic vs round). |
| `render_ladder.py` | **Communication artifact** — renders the ballistic reach EARLY vs LATE side by side (top-down MuJoCo, OSMesa headless), with the FM's own forecast of the trajectory ghosted against what the body actually does, drift regions drawn as discs (blue = reducible curl, red = irreducible noise; region A haloed), and a static region-A-error-vs-round strip. **Pure replay** — no model, no CEM: it consumes `render_pack.json`. Entrypoints `smoke` (1 composed frame → PNG) and `video` (→ mp4). |

## Modal volume layout

```
/data/directed_on_policy/<tag>/results.json
/data/directed_on_policy/<tag>/render_pack.json    # only when --render-rounds is set
/data/videos/e3_<tag>.mp4                          # render_ladder.py::video
```
Local mirror + figures: `figures/directed_on_policy_<tag>/` (PNGs regenerable, git-ignored per the repo convention); videos land in `videos/`.

## The render pack (`--render-rounds`, off by default)

`directed_on_policy.py --render-rounds "-1,0,29" [--render-policies value] [--render-n 4]` records, per named round, the ballistic reaches that round's FM plans: `starts`, `goals`, the executed `actions` (H×n torques), the `actual_tips` the body traced, the `pred_tips` **the FM itself forecast for that same plan**, and the fully-resolved `dgp` (so the renderer rebuilds the exact plant, no re-derivation). Round `-1` = the stale base FM before the policy has collected anything — the strongest "early" frame, since round 0 has already been fine-tuned once.

It is **side-effect-free by construction**: a fresh `ArmEnv` (hence its own `_noise_rng`) and fresh plan generators, and it never trains — so the graded rollout that follows is unchanged. With the default `--render-rounds ""` it is never called, so `ladder_s{0,1,2}` reproduce exactly.
