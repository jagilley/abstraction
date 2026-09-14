# Dispositional self-knowledge — file index

Full file-by-file reference for this sub-experiment. Design decisions live in
[`DESIGN.md`](DESIGN.md); the writeup is `README.md` (written after results are discussed).

**Up**: [`../../confabulation/README.md`](../../confabulation/README.md) — the occurrent confabulation battery this forks ·
[`../../confabulation/FILES.md`](../../confabulation/FILES.md)

## Code files

| File | Purpose |
|---|---|
| `dispositional.py` | **The whole node.** Forks the parent battery from *occurrent* self-knowledge ("what am I computing right now") to *dispositional* ("how is my computation changing"). Trains the OL arm once with 16 log-spaced checkpoints, then at every checkpoint forms the forward-model residual on a **fixed** held-out report set and builds four target families — IMPL/BEHAV × prospective/retrospective — as within-checkpoint quantile classes plus rank-scored continuous variants. A report head reading `a_c(p)` is trained pooled across checkpoints and evaluated on held-out sequences at a held-out contiguous checkpoint block; the observer ladder is the parent's (`O_input`/`O_io` capacity-swept, half-data control, `O_act` ceiling) plus a checkpoint-index embedding for every observer and a new `O_hist` history observer. Gated on per-checkpoint `ens_cos` + residual structure and on a pre-head **autocorrelation gate**. Four Modal functions: `dispositional_test` (CPU coordinator), `wake_trajectory`, `checkpoint_probe` (one container per checkpoint), `analyze` |
| `stratified.py` | **Stratified re-analysis**: does the pooled dispositional result hide structure, as the grammar sibling's hierarchy-level cut does? Re-fits six targets' heads and observers from the parts on the volume (no retraining of M or the instruments), dumps per-position predictions on the strict set, and scores self / best `O_io` / `O_hist` / `O_act` **within stratum** under four proxies for computational depth: the parent's 5-way syntactic taxonomy, M's output-entropy quartile, position-in-sequence bins (all public) and the occurrent residual-magnitude quartile (private, for comparison only). Prints a reproduction check of every pooled score against the stored run before any stratified number |
| `reduce_stratified.py` | CPU post-hoc reduction of `stratified.json` → `figures/stratified_reduction.txt` (the reproduction check, then all four stratifiers' tables for all six re-fit targets with n, share, chance, self, `O_io`, `O_hist`, `O_act`, ADVio, ADVpub, then a cross-stratifier ADVpub summary of the four headline rows) and `figures/stratified_advantage.png`. No interpretation |
| `reduce_dispositional.py` | CPU post-hoc reduction of `results.json` → `figures/<tag>_reduction.txt` (guards, gate, headline advantage table, observer ladders, the three instrument readings, every laddered target, the occurrent controls) and four PNGs. No interpretation |
| `__init__.py` | Package marker so `a2a_forward.fsm_part3.dispositional_language` imports under Modal's `add_local_python_source("a2a_forward")` |

## Modal functions in `dispositional.py`

| Function | GPU | What it does |
|---|---|---|
| `dispositional_test` | none (CPU) | Coordinator. Runs the wake trajectory, fans `checkpoint_probe` out over checkpoints in waves of `max_dop`, then runs `analyze`. This is the entrypoint to invoke |
| `wake_trajectory` | L4 | Trains M (OL arm) for `n_steps`, saving a checkpoint at each scheduled step. Replicates `confabulation_test`'s OL path including RNG consumption order, and also writes the harness's standard `wake_ckpt/{ck_key}.pt` so a sibling can load the same model instead of retraining |
| `checkpoint_probe` | L4 | One checkpoint's work: M's side of the report set (report-site state, `a_0`, top-k output summary + the four full-distribution scalars, per-position loss/entropy/correctness), `ens_n` fresh instrument FMs per capacity, the residual, the junk-residual guards, and the per-position instrument no-change floor. Writes one part file, caches its FM weights |
| `analyze` | L4 | Loads the parts, folds in the cross-checkpoint quantities through a sliding window, runs the autocorrelation gate, the dispositional battery and paper 2's occurrent controls at the final checkpoint. Writes `results.json` |

## Notable flags

| Flag | Default | Why you would change it |
|---|---|---|
| `--n-ckpt` / `--ckpt-first` | `16` / `300` | Number of log-spaced checkpoints and where the schedule starts. Starting earlier walks into the saturated-instrument regime (a barely-trained M is trivial to predict) |
| `--hold-ckpts-str` | `""` → middle 3 | The held-out contiguous checkpoint block. The strict evaluation is held-out sequences at these checkpoints |
| `--horizons-str` | `1,2,4` | Prospective horizons in checkpoints. All are computed and all get the autocorrelation gate reading; only the first and last get an observer ladder |
| `--retro-w` | `4` | Retrospective window, in checkpoints. Matched to `--hist-lags` so `O_hist` sees exactly the stretch the retrospective targets integrate over |
| `--n-qclass` | `4` | Quantile classes for the categorical targets. Boundaries are fit **within checkpoint** on train sequences, which is what removes basis drift and per-checkpoint constants |
| `--inst-caps-str` | `16:0.5,4:0.25` | Instrument-FM capacity sweep. Index 0 is the default, where the full ladder and the three instrument readings run |
| `--full-cap-targets` | off | Runs every target family at the non-default instrument capacity too (~2× observer cost) |
| `--n-report-sequences` | `2000` | Below the parent's 3000 because the aggregator holds every checkpoint's report set resident |
| `--obs-steps` / `--head-steps` | `3000` / `3000` | Kept equal: an observer step sees `obs_bs × (T−1) = 4064` positions and a head step `head_bs = 4096` rows, so equal counts give matched exposure |
| `--hist-lags` | `4` | How many earlier checkpoints `O_hist` sees, beyond the current one |
| `--max-dop` | `8` | Concurrent GPU containers in the per-checkpoint fan-out |
| `--refresh-wake` / `--refresh-fm` / `--refresh-parts` | off | Force retraining / recomputation of each cached phase |
| `--skip-occurrent` | off | Skips paper 2's occurrent battery at the final checkpoint |
| `--smoke` | off | Wiring check: 300 wake steps, 8 checkpoints, N=128, tiny everything. Numbers meaningless (`ens_cos` lands in the 0.3–0.55 junk band, which is the correct reading for a 300-step model) |

## Reproduction

```bash
cd experiments/            # chromatic workspace; token shards are at /data/tokens

# wiring check (~6 min cold, ~4 min with parts cached)
modal run a2a_forward/fsm_part3/dispositional_language/dispositional.py::dispositional_test \
    --smoke --tag smoke

# headline
modal run --detach \
    a2a_forward/fsm_part3/dispositional_language/dispositional.py::dispositional_test \
    --tag main

# reduction (local, CPU)
modal volume get language-reduction-data \
    /a2a_forward/confabulation/dispositional/main/results.json .
python3 -m a2a_forward.fsm_part3.dispositional_language.reduce_dispositional \
    --results results.json --tag main
```

## Figures (`figures/`, written by `reduce_dispositional.py`)

| File | Contents |
|---|---|
| `<tag>_reduction.txt` | The full reduction: run config, the per-checkpoint junk-residual guard tables at both instrument capacities, the autocorrelation gate, the headline 2×2 under both advantage statistics, the observer ladders, the three instrument readings, every laddered target, and paper 2's occurrent battery |
| `<tag>_trajectory_guards.png` | M's val-loss trajectory with the held-out checkpoint block marked; `ens_cos` and FM cosine per checkpoint per instrument against paper 2's lowest published `ens_cos`; the per-position instrument no-change floor and \|r\| |
| `<tag>_advantage_2x2.png` | Self / best `O_io` / `O_act` / `O_hist` on the four primary targets, and the advantage under both statistics |
| `<tag>_autocorrelation_gate.png` | Lag-1 autocorrelation and between/total variance share for every target |
| `<tag>_readings_and_occurrent.png` | The three instrument readings of IMPL-prospective, and paper 2's occurrent battery on this model |
| `stratified_reduction.txt` | The stratified re-analysis: reproduction check, then the syntactic / entropy-quartile / position-bin / \|r\|-quartile tables for all six re-fit targets, then the cross-stratifier ADVpub summary |
| `stratified_advantage.png` | ADVpub per stratum for the four headline rows across all four stratifiers, with each stratum's share of positions on the axis |

## Two advantage statistics

`ADVio = self − best single-snapshot O_io` is paper 2's statistic, carried over unchanged.
`ADVpub = self − max(O_io, O_hist)` is the conservative one, and it is the one to read for a
**dispositional** target: `O_hist` reads only M's outputs, just over several snapshots, so it
is a legitimate third party here in a way it would not have been for an occurrent target.
`fracpub = (best third party − chance) / (self − chance)` normalises by how well the self can
report the target at all — `component_control`'s `frac`, with the self as ceiling, which
matters because the self is far from saturation on every dispositional target.

## Volume layout (`language-reduction-data`)

```
/a2a_forward/confabulation/dispositional/<tag>/
    ckpt/step<N>.pt          M's state at each scheduled wake step
    fm/<cap>_step<N>_s<j>.pt each fresh instrument FM (cached; a rerun skips training)
    parts/ck<NNN>.pt         one checkpoint's report-set cache, residual unit directions,
                             instrument floor and guards
    results.json             the whole reduction input
    stratified/stratified.json   per-stratum tables from `stratified.py`
    stratified/predictions.pt    per-position predictions on the strict set, for every
                             target x predictor, plus the stratum labels and the
                             (checkpoint, sequence, position) frame -- any further cut
                             is CPU-only from this file
/a2a_forward/confabulation/wake_ckpt/<ck_key>.pt
                             the parent harness's standard wake checkpoint, written by
                             `wake_trajectory` in the parent's own key format and content
                             shape so a sibling can load it
```

## Gotchas

- **The three instrument readings are not interchangeable.** `r_c` and `r_{c+k}` come from
  different fresh FMs, so raw `1 − cos` mixes M's change with instrument variation. Read
  `IMPL_PROSP` (raw), `IMPL_PROSPEXC` (minus the per-position no-change floor) and
  `IMPL_PROSPFIX` (one instrument at both ends) together; the floor column in the guard
  table says how large the confound is at each checkpoint.
- **Within-checkpoint quantile classes are load-bearing, not cosmetic.** They are what stops
  a pooled head from scoring by memorising a per-checkpoint constant, and what removes any
  uniform residual-stream basis drift between snapshots.
- **`O_hist` is a ceiling, not a competitor.** The advantage statistic stays
  `self − best O_io`, as in paper 2. `O_hist` answers "is this fact knowable from M's
  behavioural history at all", which is the retrospective analogue of paper 2's `ENT` row.
- **The observers need the checkpoint-index embedding.** The report tokens are identical at
  every checkpoint, so without it `O_input` cannot tell which snapshot it is looking at and
  its score degenerates to the position's average rank across training.
- **Early checkpoints are the junk-residual risk here**, not over-capacity: a nearly
  untrained M is trivial to predict and the instrument saturates. The schedule starts at
  step 300 and the per-checkpoint `ens_cos` / `eta2` table is printed so rows can be dropped
  after the fact.
- **The wake phase is deliberately fp32 with TF32 off**, matching the parent. Enabling TF32
  would speed it up several-fold but would produce a *different* model, breaking the promise
  that the standard `wake_ckpt` is the object the parent battery's OL arm would produce.

## Stratified re-analysis: reproduction and two deviations

`stratified.py` re-fits rather than reloads (the parent run saved aggregate scores, not
per-position predictions), so it prints a reproduction check first: re-fit pooled strict
score against the stored one for every target x predictor. On the `main` run the max
absolute difference was **0.0000** — every seed on the head/observer path is explicit, so
the re-fit models are the same models. Two deliberate deviations, both conservative:
`IMPL_PROSPEXC` / `IMPL_PROSPFIX` were light targets in the parent run and got only the
4L256D `O_io`; here they get the full three-capacity sweep. And within each stratum,
"best `O_io`" is the max over the three capacities *evaluated in that stratum*, which lets
the observer pick its best capacity per stratum while the self cannot.

**Read `chance` in every stratified row.** It is the within-stratum majority-class share,
and in the rare high-reorganization strata it exceeds every predictor's score — no pooled
head or observer is allowed to condition on the stratum label, so a within-stratum
constant is not a baseline any of them could have used, but a large advantage in a stratum
where nobody beats `chance` is a different object from one where they all do.

## Related, outside this folder

| Path | Relation |
|---|---|
| [`../../confabulation/confabulation.py`](../../confabulation/confabulation.py) | The parent battery; this imports its k-means, ensemble-cosine, residual-structure, head and syntactic-label helpers rather than copying them |
| [`../../confabulation/README.md`](../../confabulation/README.md) | The occurrent result this extends |
| `../../../../papers/forward_self_models_paper2.md`[^private] | The paper (§2–§5, §8) |
| [`../../../mjc/committee_head/README.md`](../../../mjc/committee_head/README.md) | Phase B: the autocorrelation gate's origin — "probe-able does not imply learnable from a given target" |
| [`../../../rhm/residual_decomposition/trajectory/README.md`](../../../rhm/residual_decomposition/trajectory/README.md) | The residual reorganizing across training: the raw material of the IMPL targets, and the source of the early-checkpoint saturation warning |
| [`../../OOD_ROBUSTNESS_README.md`](../../OOD_ROBUSTNESS_README.md) | Exp A: output entropy beats every activation probe at predicting M's own loss — why the behavioural rows are the built-in dissociation |

[^private]: Not mirrored: this link points to a document in the private lab repo (the roadmap, the queue, an unrun spec, reading notes, or a conversation). See the top-level README for what is held back and why.
