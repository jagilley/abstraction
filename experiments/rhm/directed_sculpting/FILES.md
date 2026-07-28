# directed_sculpting — File index

Complete file-by-file reference for this node. Summarized in [README.md](README.md).

The *primitives* this node exercises live one level up at [`../`](../FILES.md), because they are
shared across RHM experiments (per [`STRUCTURE.md`](../../../STRUCTURE.md) — code lives at the
lowest node that shares it): [`../rhm_channels.py`](../rhm_channels.py),
[`../rhm_drift.py`](../rhm_drift.py), [`../rhm_repair_cost.py`](../rhm_repair_cost.py),
[`../verify_backcompat.py`](../verify_backcompat.py). What lives here are the experiments that
certify them.

## Code files

| File | Purpose |
|---|---|
| `verify_distractors.py` | P1–P5 on the multi-channel distractor DGP: structural irrelevance of distractor content, exact irreducibility of the noise channels, the `error-only` and `lprog-only` traps, and unigram-statistic matching. Reports each channel's saturation step, which sets the drift cadence needed to keep allocation zero-sum |
| `verify_drift.py` | D0–D4 on support-fixed rule drift: back-compatibility at uniform weights, on-grammar invariance under heavy drift, the closed-form KL against the realised log-likelihood ratio, per-level σ calibration for matched-magnitude sweeps, and stationarity of the OU walk |
| `verify_repair_cost.py` | R1–R3 on the repair-cost instrument: the zero-drift null, the attributable gap's response to drift magnitude (and its cross-check against the closed-form KL), and the demonstration that raw cross-entropy is blind to support-fixed drift. Carries the paired matched-compute control arm |

## Results

| Path | What |
|---|---|
| `figures/distractor_dgp_l5_v1/results.json` | P1–P5 at L=5, 20k steps — the headline distractor verification |
| `figures/drift_l5_v1/results.json` | D0–D4 at L=5 |
| `figures/repair_cost_l5_fresh/results.json` | R1–R3 at L=5 with fresh per-step sampling — the trustworthy repair-cost run |
| `figures/repair_cost_l5_v1/results.json` | The same sweep on the broken fixed-pool harness. **Instrument validation only** — absolute numbers are invalid (matched reference collapsed to CE 1.31); retained because the difference-in-differences recovered R1–R3 through it |

## Children

| Folder | What |
|---|---|
| [`full_loop/`](full_loop/README.md) ([FILES](full_loop/FILES.md)) | E3's inner + outer loop wired onto this substrate: the multi-channel sculpting environment and its E1–E6 certification, the seven-policy allocation ladder, the drift-level climb sweep, and the grader-type × drift expansion 2×2 |
