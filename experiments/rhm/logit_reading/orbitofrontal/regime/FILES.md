# regime — file index

**Up**: [`../README.md`](../README.md) (orbitofrontal, the writeup) · **Donor**: [`../../striatum/README.md`](../../striatum/README.md) (striatum) · **Tables**: [`results/tables.md`](results/tables.md)

A world in which a rule violation is evidence about a persistent hidden state that predicts
future cost, built so that a value reader has a reason to read legality. The interpretation is the
super-node's ([`../README.md`](../README.md) §2b); [`README.md`](README.md) here is a pointer.

## Code files

| file | purpose |
|---|---|
| [`world.py`](world.py) | The world and its reference. `regime_path` / `regime_windows` are the two-state Markov corruption process (mean dwell `dwell_c` = 288 clean, `dwell_n` = 32 noisy, so `p_noisy` = 0.10; `eps_c` = 0.002, `eps_n` = 0.12, marginal 0.0138) — stream-level for training, window-level for evaluation, identical in distribution because the chain is started stationary. `corrupt` applies `altitude/train_noisy.py`'s uniform redraw at the regime's rate. `regime_filter` is the exact forward filter over the two regimes whose emission model is the ε-observer predictive at each regime's rate (`flat_oracle.flat_predictive(..., noise_eps=eps_r)` at `k = 5`, the model's own altitude) — the running form of `basalis/hold.py`'s one-shot log Bayes factor — returning the belief before the token (`b_pre`), the forward-looking belief (`b_fwd`), its revision (`db`), the same in log-odds (`dlogbf`) and the token's own evidence (`llr`). `world_build` writes one evaluation venue: the corrupted windows, their clean counterparts, the level-ℓ answers of the (uncorrupted) generating tree, the exact legal support under `p_L` at `k = L`, and the filter. **The filter is always the `burst` reader**, in both worlds, so the same statistic is defined on the control venue — where, by construction, it predicts nothing. |
| [`train.py`](train.py) | One next-token trajectory per world: `train_noisy`'s recipe (8L/8H/256D, AdamW 3e-4, batch 64, fresh data every chunk, the same 13-checkpoint ladder) with the regime process in place of the i.i.d. ε, applied to the stream before it is windowed and before the `(x, y)` split. `--world iid` is the same marginal rate with no regime; it is rebuilt here rather than read from the banked `traj_eps01_s42` so both worlds share this module's RNG discipline and the rates match to four digits. |
| [`task.py`](task.py) | The value-side battery: `striatum/task.py`'s actor (16-way level-ℓ query head on `post_block7`, trained on clean windows and frozen), its streamed-Gram closed-form ridge critic `V[ℓ, δ]` at every horizon δ = 0…12 fitted on realised outcomes only, the shuffled-outcome floor and the clean-only critic. Events are the positions whose token the corruption changed, plus a quiet control at a corruption-free position of a quarter of the windows. Two twins, both with a bit-identical prefix: the **natural twin** (this corruption undone, every other token kept — so the damage `o_nat − o_x` is what *this* corruption did), and the phasic **legal twin** (the corrupted token replaced by a legal token of nearest model surprisal, defined only where the prefix is still legal). Every event carries the filter's `b_pre`, `b_fwd`, `db`, `dlogbf`, `llr`, the realised corruption in the horizon ahead, oracle probes for the regime / legality / damage on the same states, and a position-level regime probe. `regime_all` is the CPU coordinator that runs the whole node in one detached launch: wave 1 builds both venues and trains both trunks (4 containers), wave 2 runs the readout cell for every (world, checkpoint). |
| [`analyze.py`](analyze.py) | The local reduction → [`results/tables.md`](results/tables.md) and `figs/`: the world census, the trunks and actor, the regime probe, the event census, `R` against the filter's revision (with the partial correlation given the model's own surprisal), the revision inside terciles of the pre-event belief, the cross-world "corruption ahead" reading, the matched legality contrast, the twins, and the regime-switch traces. |

### Shell / helper files in `results/`

| file | purpose |
|---|---|
| `fetch.sh` | pull this round's artefacts off the volume into a local mirror, one file at a time (`modal volume get` on a directory returns a zip that has come back truncated) |
| `verify.py` | touch every member of every fetched `.npz` (a silent truncation and a bad CRC on a single member have both slipped past `np.load` before) |
| `reduce.sh` | verify the mirror, then write the tables and figures |

## Artefacts on the volume

`rhm-scaling-data` (**`chromatic`** workspace), under `/data/v16_s2_L6_m4_distinct/logit_reading/`:

- `regime_world_{burst,iid}.{npz,json}` — the two evaluation venues (65,536 windows each)
- `traj_regime_{burst,iid}/step{000000…064000}.pt`, `train_log.json` — the two trunks
- `traj_regime_{burst,iid}/step{000000,008000,024000,064000}_regime_{burst,iid}.{json,npz}` —
  the readout cells; the `.npz` is the per-event table (every event, with its split label)

## Reproduction

```bash
cd experiments            # MODAL_PROFILE=chromatic
modal run --detach -m rhm.logit_reading.orbitofrontal.regime.task::regime_all \
    --worlds burst,iid --n 65536 --steps 64000 --read-steps 0,8000,24000,64000
bash rhm/logit_reading/orbitofrontal/regime/results/fetch.sh  <local-mirror>
bash rhm/logit_reading/orbitofrontal/regime/results/reduce.sh <local-mirror>
```

## Gotchas worth not rediscovering

- **`p_L` is undefined after the first violation in a window.** `flat_predictive` at `k = L`
  normalises a zero posterior to zero everywhere, so *every* token after the first illegal one
  reads as illegal (about 18% of all positions here, against a 1.3% corruption rate). Legality
  is only a label on events whose within-window prefix is still legal (`legal_prefix`), which is
  about a third of corrupted events because corruptions come in bursts — most are not the first
  of their burst. Among those, the illegal / legal split is a workable 0.59 / 0.41.
- **The filter's reader must be the same in both worlds.** With the control world's own
  (degenerate) process the filter returns zeros and every cross-world row is empty. Running the
  `burst` reader on both venues makes `db`, `llr` and `b_fwd` the same statistic; it reads 0.80
  AUC for a corruption in the next 12 tokens on `burst` and 0.51 on `iid`.
- **`db` and `llr` are different currencies.** An illegal token carries a fixed `llr` (≈ 2.7
  nats at `k = 5`) whatever the reader already believed; its revision `db` of the forward belief
  collapses from ≈ 0.60 to ≈ 0.17 across terciles of the pre-event belief. Which of the two a
  value revision tracks is the "weighted by expectation" question, and they must be reported
  side by side.
- **Only offset 0 is a clean twin reading.** After the event the window and either twin share
  the same continuation, corruptions included, so the later offsets are not a decay of the event.
- **A LEGAL corruption costs more than an illegal one, at every horizon, in both worlds**
  (§7c: at ℓ = 2 the natural-twin damage is 0.30 after an illegal token and 0.39 after a legal
  one at `a = 0`, 0.16 against 0.32 at `a = 2`). A token that is off-grammar is recognisable as
  noise; a legal-but-wrong token is a plausible member of another constituent and the model
  re-parses around it. So "illegal" is not "expensive" here — it is "visible", and any table
  that reads a legality contrast as a cost contrast reads it backwards.
- **The pre-event belief has no range inside the legal-prefix subset.** `legal_prefix` means no
  earlier violation in the window, which is exactly the condition under which the filter has not
  yet raised its belief (`b_fwd_pre` terciles 0.025 / 0.028 / 0.053). The "weighted by
  expectation" analysis has to run on all corrupted events with the token's own evidence held
  fixed (`llr ≥ 3`), where the terciles are 0.03 / 0.38 / 0.83.
- **`n_corrupt_ahead` is truncated by position** near the end of the window, and the state knows
  the position, so any reading of it must be restricted to events with a full horizon
  (`t + 12 ≤ 63`); without that the control world reads 0.57–0.60 on a quantity it cannot know.
- **A damage label read as an AUC is compressed by ties.** `o_nat − o_x` is mostly 0, so the
  AUC of a matched legality contrast against it sits near 0.5 whatever the truth; report the
  paired mean difference with its standard error instead.
- **Outcome arrays are `uint8`**; cast before differencing (`o_nat − o_x` underflows to 255).
- **A sibling node's `.py` file being written races Modal's source mount.** The documented
  `ignore` does not cover it (the failure was `... was modified during build process`); stage
  `rhm/`'s `.py` files into the scratchpad and run `modal run` from there.
- **`modal volume get` on a directory returns a zip** that has come back truncated; fetch
  members individually and run `results/verify.py`.
- **A coordinator can only `.spawn` a function its entry-point module imported at module level.**
  Importing `train.py` inside `regime_all`'s body left `train_regime` unhydrated
  (`Function has not been hydrated with the metadata it needs to run on Modal`).
- **Modal's CLI lower-cases flag names**, so a keyword argument spelled `skip_pL` is unreachable
  from `modal run` (`--skip-pl` arrives as `skip_pl`).
