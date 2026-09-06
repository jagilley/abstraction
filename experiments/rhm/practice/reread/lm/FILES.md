# reread/lm — File Index and Calibration Record

The **LM twin** of [`../`](../FILES.md) (the sculpting round, `rr_s0`). Machinery donor:
[`../../../conditional_revision/`](../../../conditional_revision/README.md) — imported, never
modified. Substrate conventions: [`../../../CLAUDE.md`](../../../CLAUDE.md). Findings:
[`README.md`](README.md) (written post-discussion, 2026-08-17).

## Why this node exists

`rr_s0` found the frozen archive **renewable in competence but not in mining yield**, and the
diagnosis was structural rather than substantive: on the sculpting substrate the reader is
**exogenous** — handed over, trained once, pinned at 1.000 block accuracy — and mining reads the
agent's own repair (`mine_from="chosen"`, canonicalised through `canon`). The archive is
therefore rewritten by the act of reading it, and **vocabulary-gated perception is structurally
inexpressible**. The book-reread claim's mechanism (you cannot parse level-ℓ until you own
level-(ℓ−1)) lives in the reader, so this node moves to a substrate where the reader is
endogenous: autoregressive NTP on RHM, where the only representation is the learned one and
per-level structure recovery is read against exact BP oracles.

## Code files

| File | Purpose |
|---|---|
| `lm_reread.py` | The Modal app. `parse_arms` maps `frozen_<n>` / `fresh` to a corpus policy; `ladder` builds the log-spaced checkpoint schedule with the fidelity-gate step forced in. The training loop is `conditional_revision.gate0`'s base loop verbatim (same GPT, AdamW(3e-4, wd 0.01), batch 64, and the same flat-window sampler `corpus[ix + arange(T)]` off a concatenated corpus) — the arms differ **only** in whether the corpus tensor is redrawn. Per checkpoint: `per_level` (the donor's `_probe_acc` on last-position activations, best over blocks, repo convention d1 shallowest → d6 root), `flat_nll` on held-out fresh windows and on the arm's own corpus, and `excess_over_bayes` (model per-position NLL minus the exact Bayes surprisal, bucketed by `pos_top_level`). Entrypoints: `lm_reread` (main), `gate` (C-F / C-O / C-C, CPU) |
| `analyze_lm.py` | Reduction. `--fetch` pulls from the volume. Sections: **0** the fidelity gate against `conditional_revision`'s published Gate-0 base; **1** per-level recovery vs tokens against the BP ceiling; **2** val NLL and the memorisation gap; **3** BP-referenced excess loss per arrival level; **4** the ordering readout — tokens-to-reach-level-ℓ, its monotonicity in ℓ, and the frozen/fresh ratio (the "effectively unlimited data" quantification); **5** late movement, i.e. whether an arm is still extracting when the budget ends. `--figures` writes fig1–fig4 |
| `launch_detached.py` | Session-isolated detached launcher, `../launch_detached.py`'s pattern with modal's `-m` module form |

## What is imported from `../../../conditional_revision/` and `rhm/`, unmodified

`oracle.py`: `prefix_beliefs` (the exact per-level ceiling), `revision_and_entropy` (the exact
Bayes surprisal floor), `self_check`.
`rhm/`: `model.GPT`, `rhm_data.generate_rules_distinct`, `rhm_latent_loop._generate_with_traces`
and `._probe_acc`, `shared.{DATA_DIR, volume, image, setting_key, NumpyEncoder}`.

## Design decisions, and the measurement or argument that forced each

| decision | why |
|---|---|
| **NTP on RHM rather than sculpting** | the whole point of the follow-up: the reader must be endogenous. Here there is no handed-over parser and no canonicalising rewrite — the model's representation is the only reader it has, and the corpus is never modified by being read |
| **The arms differ only in corpus redraw.** Same model init (`torch.manual_seed(seed)` before `GPT(...)`), same optimiser, same sampler generator, same steps, same tokens consumed | tokens-consumed is matched by construction, so "extra passes" is the only variable. `rr_s0`'s matched-pricing discipline transposed |
| **`frozen_200000` @ step 12000 IS `conditional_revision`'s Gate-0 base** — v16/s2/L6/m4, rule_seed 0, pool 200 000 sequences at data_seed 7, seed 42, 8L/8H/256D, 12 000 steps, batch 64 | the fidelity gate costs nothing: the reference arm doubles as a data point on the corpus-size axis. Their base co-trains a depth FM in an open loop on detached activations, which consumes global torch RNG but never touches the model, so the model trajectory should be a replay rather than a re-derivation. Read as a reproduction, not a bit-identity claim |
| **Corpus size, not RHM depth, is the swept coordinate** — 512 / 2 048 / 16 384 / 200 000 sequences = 32.8k / 131k / 1.05M / 12.8M tokens against 81.9M consumed (2 500× / 625× / 78× / 6.4× re-read) | `rr_s0`'s surprise #3 transposes as "how much distinct structure does the archive hold". Changing RHM `L` changes the sequence length and therefore the model and the reference lines; corpus size moves the same variable while keeping the fidelity anchor intact. **The per-level index ℓ is the depth coordinate**, and tokens-to-reach-level-ℓ is exactly the requested per-depth quantification |
| **`fresh` redraws a pool of `batch_size × fresh_every` sequences every `fresh_every` steps** | sized so each pool is consumed almost exactly once (4 096 tokens/step × 500 steps = 2.05M consumed; 32 000 sequences × 64 = 2.05M drawn), so `fresh` is genuinely novel rather than a large-but-finite corpus. 40 redraws = 81.9M distinct tokens for 81.9M consumed |
| **Two independent extraction readouts, one probe-based and one probe-free** | the probe (`_probe_acc`) is the donor's own instrument and carries its known weakness (`probe_diag` exists because a probe reached 0.068–0.182 against a Bayes ceiling of 0.371–0.614). `excess_over_bayes` needs no probe at all — it is the model's own NLL against an exact floor — so a probe failure and a representation failure cannot be confused |
| **Every recovery number is charted against the exact BP ceiling**, not against 1.0 | `probe_diag`'s discipline: "the probe is weak" and "the model cannot know this" must stay distinguishable. Measured at the gate (below), the ceilings are 0.798–0.997, so **every level has real headroom** and a flat curve is a finding rather than a saturation artefact |
| **Probes run on 5 blocks at most checkpoints and all 9 at {12 000, final}** | the full best-over-all-blocks × max(linear, MLP) protocol is the donor's and is what the published reference numbers were measured under, but it costs ~4× a reduced pass. Full protocol where the gate and the endpoint need it; reduced elsewhere, where only the trajectory shape matters |
| **The eval apparatus is drawn once and shared by every arm** (8 000 aligned sequences at `eval_seed` 999 — the donor's own value — plus a held-out fresh flat val corpus and 512 oracle sequences) | controlling variables; also makes the fidelity comparison exact, since the donor measured its level profile on the same 8 000-sequence aligned set |

## Gates (`modal run -m rhm.practice.reread.lm.lm_reread::gate`)

| gate | what it asserts | status |
|---|---|---|
| **C-F** | the frozen corpus is byte-identical on redraw at the same seed, and a `fresh` draw at a different seed differs | **passed** |
| **C-O** | `oracle.self_check` — `E[B_D] == E[H_tot − H_irr_D]` — so the BP instrument this node reads its Bayes floor from is the one `conditional_revision` validated | **passed**: D0 rel_err 0.011, D2 rel_err 0.024; mean exact surprisal 1.382 nats |
| **C-C** | the exact per-level ceiling `P(z_ℓ | x_{1..T})`, the denominator every recovery number is charted against | **measured** (8 000 eval sequences): d1 0.997 · d2 0.999 · d3 0.996 · d4 0.984 · d5 0.957 · **d6 0.798**. Every level is near-determined given the full sequence, so the donor's published d6 = 0.088 is a **model/probe limit with ~0.71 of headroom**, not a task limit — all six levels have dynamic range |
| **fidelity** | `frozen_200000` @ step 12000 reproduces the published d1 0.979 / d3 0.836 / d6 0.088 | **passed on `lm0`, essentially exactly**: d1 **0.978** (−0.001) · d3 **0.834** (−0.002) · d6 **0.085** (−0.003), under the full best-over-all-blocks × max(linear, MLP) protocol. A replay, not a re-derivation |

### Known instrument artefact: the probe protocol switch

Most checkpoints probe **5 blocks with a linear head**; the two `full_probe` checkpoints (steps
12 000 and 20 000) use **all 9 blocks and max(linear, MLP)** — the donor's protocol, and what
the published reference numbers were measured under. The full protocol reads **systematically
higher**, and the offset is *arm-dependent*: ≤0.042 for `fresh` and `frozen_200000`, but
**+0.216 to +0.235 at d2 and +0.061 to +0.098 at d3** for the small-corpus arms. So the reduced
probe understates recovery precisely where the model is heavily overfit, and any quantity that
spans the two protocols measures the protocol change rather than learning. `analyze_lm.py`
section 5 therefore compares **only full-protocol checkpoints with each other**, and prints the
offset itself so the size of the artefact is on the record. The trajectory *shape* within the
reduced-protocol checkpoints is internally consistent; levels are not comparable across the two.

Also measured at the gate: the exact Bayes surprisal per arrival level (0 = the token closes the
root … 6 = it closes only a leaf pair) — **0.135 / 0.138 / 0.193 / 0.251 / 0.464 / 0.932 /
2.156 nats**. This is the floor `excess_over_bayes` subtracts, and it says the deep-boundary
positions are where nearly all the extractable uncertainty lives.

## Configuration of the main run (`lm0`)

v=16, s=2, L=6, m=4, rule_seed 0 (the donor's regime, so its reference lines transfer);
GPT 8L/8H/256D (6.34M params), T=64; AdamW lr 3e-4, wd 0.01, batch 64;
`max_steps` 20 000 = **81.9M tokens consumed**; data_seed 7, seed 42.
Checkpoints: 250, 387, 601, 931, 1443, 2236, 3466, 5372, 8326, **12000**, 12904, 20000.
Arms: `fresh`, `frozen_200000`, `frozen_16384`, `frozen_2048`, `frozen_512`.
Measurement: 8 000 aligned eval sequences (eval_seed 999), 512 oracle sequences,
probe 600 steps @ lr 1e-2 (+ MLP 128×800 in the full protocol), `thresh_frac` 0.5.

## Reproduce

```bash
cd experiments/            # MODAL_PROFILE=chromatic

# structural gates (CPU, ~1 min)
modal run -m rhm.practice.reread.lm.lm_reread::gate
# smoke (attached, ~80 s)
modal run -m rhm.practice.reread.lm.lm_reread::lm_reread --quick --tag smoke0 \
    --arms "fresh,frozen_2048"

# the main run
python3 rhm/practice/reread/lm/launch_detached.py --fn lm_reread --tag lm0 \
    --arms "fresh,frozen_200000,frozen_16384,frozen_2048,frozen_512" \
    --max-steps 20000 --batch-size 64 --lr 3e-4 --weight-decay 0.01 \
    --data-seed 7 --seed 42 --fresh-every 500 \
    --n-eval-sequences 8000 --eval-seed 999 --n-oracle 512 \
    --probe-steps 600 --probe-lr 1e-2 --mlp-hidden 128 --mlp-steps 800 \
    --thresh-frac 0.5

# reduction
python3 rhm/practice/reread/lm/analyze_lm.py --tag lm0 --fetch --figures
```

Modal volume (`rhm-scaling-data`): `/data/rhm_practice_reread_lm/<tag>/` with `setup.json`
and one `<arm>.json` per arm. Figures land in `figures/<tag>/`.

## Results on disk

| tag | what it is |
|---|---|
| `smoke0` | the attached smoke (`--quick`): 2 arms, 600 steps, 2 checkpoints |
| `lm0` | the main run: 5 arms, seed 0, 20 000 steps / 81.9M tokens |
