# Dress rehearsal — the organizers' evaluator, a submission port, and the first full-budget Hard measurements

**Up**: [../README.md](../README.md) (one_layer_deeper) · **Files**: [FILES.md](FILES.md)
**Upstream**: [tilde-research/one-layer-deeper](https://github.com/tilde-research/one-layer-deeper) at **`4ceff95`** (Aug 14) · competition deadline **Mon 2026-08-31 22:00 PT**
**Status**: harness complete; port at v3; **15 hosted runs** (4 Easy, 9 Medium, 2 Hard) on the
organizers' H100s, all saved under `results/hosted/`. Single seed throughout (the evaluator's
`[74]`). **Date**: 2026-08-22.

---

## One-liner

Every other node here uses the upstream benchmark as a controllable DGP. This one uses it as a
benchmark: it runs the **organizers' evaluator unmodified** on the **organizers' data** — first on
our own Modal H100 (smoke only), then on the hosted Easy/Medium/Hard tiers, which turn out to be
free H100 time with full per-rung results returned — to find out where a submission built from
this program's machinery actually lands. Three things came out.

1. **The leaderboard's structure is now legible.** The `[12,14,16]`-bit no-reduction floor measures
   **exactly 6/768** at the gating rung, which is exactly the score of ranks 2–6; the 12-bit cell
   is necessarily dense (only 14 such moduli exist, and `m5`'s training split already covers ~84%
   of all possible 12-bit prompts); and rank 1's 127/768 ≈ half of one 256-example cell with 0% OOD
   is what "interpolates the dense cell, nothing else" would score. Hard's scored-split names pin
   it to a **modulus-grouped** generator config. (§2, §4.)
2. **At full Hard budget, our control arm memorises 58% of the hidden training set and transfers
   nothing** — 0/768 at every rung on both profiles, test loss 11.4 nats/token — which is the
   rule-acquisition wall [`variable_modulus/`](../variable_modulus/README.md) and
   [`rule_acquisition/`](../rule_acquisition/README.md) described, measured on someone else's DGP
   at 3600 H100-seconds. (§4.2.)
3. **`ballistic_depth`'s fresh-`x` result did not reproduce in the port**, even after the cycle
   term was rebuilt to the research mechanism and verified (gradient into the real encoder,
   bit-exact re-encoding, ramp active on a fully-memorised model): on a single seen modulus with
   every unit seen except the fresh cohort, control / closure-v2 / closure-v3 read **2 / 4 / 1 of
   140** at `T=1` against a floor of 4, where the research code read 0.001 → 0.32. (§4.3.)

The port went through three versions, each fixing a defect the previous hosted run exposed (a
soft rung-selector that could not memorise anything; a clock-gated aux ramp that froze an
un-memorised model; a `ReEncoder` that never grounded the encoder fresh inputs pass through).
Those are recorded in §3 because each one explains a hosted number that would otherwise read as
a fact about the idea rather than about the port.

---

## 1. The rules and the grader, as verified (Aug 21–22)

The local clone at `~/Code/one-layer-deeper` is at `885fb4c` (Jul 18) and is **nine
commits stale**; `4ceff95` includes the deadline, a rules rewrite, `token_training_loss`, multiple
backward passes, and `79f0a09` "Updating scoring to be based on extrapolation in T", which rewrote
`benchmark/runner.py` and `data/squaring_mod.py`. A fresh checkout lives at
`~/Code/one-layer-deeper-head` with its own `uv` venv (see Reproduce).

- **Hard ranking** (rule 13): largest consecutively-certified `T` on *seen* moduli with fresh
  `(N, x)` prompts → then the same on *unseen* moduli → then exact accuracy at each profile's
  first uncertified rung. Ladder `T = 1,2,4,8,16,32,64`; a rung certifies only at **100% exact**;
  certification must be a consecutive prefix. Splits `depth_t_{T}` / `depth_ood_n_t_{T}`,
  **768 per rung** (3 widths × 256).
- **Rules that bind a port** (6–8, 12–14): no hard-coded weights; no hard-coded algorithm in the
  forward pass; an unbroken autograd path from loss to the parameters producing the logits;
  custom loss via legacy `training_loss` or `token_training_loss(TokenLossBatch)` (which carries
  `.auxiliary`); **no data augmentation**, no task-specific solvers, no `backward`/`autograd.grad`
  in participant code, no nested model/loss calls from the loss; ≤500M model-state elements;
  GPU only. "Recurrence, adaptive computation, and depth curricula are allowed"; depth is
  deliberately unconstrained. These rules are why
  [`staged_reduce/`](../rule_acquisition/staged_reduce/README.md)'s two enabling moves — a
  self-made single-stage training distribution and an argmax re-grounding schedule — were not
  ported.
- **Contract** (`runner.py::_loss_and_accuracy`, `squaring_mod.py::collate_squaring_mod`): prompt
  `[N] digits [X] digits [T] digits` (`DIGIT_OFFSET=7`, `VOCAB_SIZE=17`); separate input/output;
  labels are the unpadded decimal digits of `y` and `target_positions` are the **last `len(y)`
  positions of the input**, so the model emits `[B, L_in, vocab]` and the evaluator gathers the
  right-aligned tail; padding (not causal) mask; `(logits, auxiliary)` return; bf16 AMP, grad clip
  1, batch 512 default, one seed; wall-clock budget 60 / 600 / 3600 s with half again for
  evaluation; construction and compile count against the clock.
- **Hard's data is modulus-grouped.** Its three scored splits are named `test` (9,999),
  `ood_t` (10,002), `ood_n_t` (10,002). In the generator `separate_ood_splits` requires
  `split_group ∈ {x, modulus}` and a depth profile requires `split_group ∈ {prompt, modulus}`, so
  Hard is `split_group=modulus`: moduli at each width are shuffled and **partitioned into disjoint
  train/test pools** (~90/10); `test` = unseen moduli at trained depths, `ood_t` = seen moduli at
  held-out depth, `ood_n_t` = unseen moduli at held-out depth. Every public dataset, and every
  plausible-Hard config in §2, is prompt-grouped. For *ranking* this matters less than it sounds —
  rule 13 ranks on the depth profiles, not on those splits — but it means Hard has fewer training
  moduli per width (≈13/38/133 at 12/14/16 bits) and a hidden-modulus test pool.
- **What the hosted service returns.** `one-layer status <id> --json` returns the **complete
  per-rung profile on both profiles** (`correct_examples` / `example_count` / `status` for all
  seven rungs), every seed-level field (steps, seconds, state elements, per-split counts and
  losses) — **for Hard as well as Medium**; nothing is redacted below what the board shows.
  `one-layer metrics` is only the `log_every=100` training curve at 3 decimals. Observed quotas:
  Easy 60/day, Medium **9/day** (README says 6), Hard reported `left: 0` after one acceptance and
  `left: 1` after the next day's — the allowance may exceed 1/day near the deadline. One
  queued/running job per account; a Medium run is ~20 min end to end, a Hard run ~1.5 h. Easy
  and Medium are therefore **free, full-resolution H100 instruments** at 1/60 and 1/6 of Hard's
  budget.

---

## 2. Generator arithmetic, the plausible-Hard neighbourhood, and the floor calibration

`data/squaring_mod.py` draws `(p, q)` fresh per record from all primes of half the target width,
so the modulus population at a width is fixed by arithmetic:

| bits | 10 | 11 | 12 | 13 | 14 | 15 | 16 | 17 | 18 |
|---|---|---|---|---|---|---|---|---|---|
| distinct valid RSA moduli | 6 | 21 | 14 | 57 | 42 | 169 | 148 | 584 | 543 |

Consequences, measured by `audit_dataset.py` on the public `m5` (`[12,14,16]`, `T∈{2,4,8}`,
10k/setting — the anchor, because its floor matches the board):

- **A small cell is necessarily dense, a large one sparse.** Per-modulus base coverage
  **0.589 / 0.063 / 0.004** at 12 / 14 / 16 bits.
- **Data volume at 12 bits is capped.** 35,624 `(N, x)` unit pairs exist across all fourteen
  12-bit moduli; each `T` cell needs distinct pairs and the depth cohort needs 256 that appear
  nowhere, so `examples_per_setting ≲ 28k` (30k fails generation). At 10k the training split
  already holds **~84%** of every possible 12-bit prompt.
- **Deep rungs are partly degenerate.** `x^(2^T)` is eventually periodic in `T`; on `m5`'s seen-`N`
  profile 191/768 prompts at `T=16` and **357/768 at `T=64`** repeat a lower rung's answer.
  Certification needs 100%, so this hands nobody a rung.

Plausible-Hard neighbourhood (`hard_configs.py`; each moves one knob off `m5`) and the measured
no-reduction floor (`x^(2^T) < N`) at the rung that gates:

| config | bits | train `T` | /setting | seen-`N` `T=1` floor | OOD-`N` `T=1` floor |
|---|---|---|---|---|---|
| `m5` (public anchor) | 12,14,16 | 2,4,8 | 10k | **6 / 768** | 11 / 768 |
| `hB_width_down` | 11,13,15 | 2,4,8 | 10k | 17 / 768 | 8 / 768 |
| `hC_width_up` | 13,15,17 | 2,4,8 | 10k | 1 / 768 | 6 / 768 |
| `hD_t_shallow` | 12,14,16 | **1,2,4** | 10k | 6 / 768 | 11 / 768 |
| `hE_t_deep` | 12,14,16 | **4,8,16** | 10k | 6 / 768 | 11 / 768 |
| `hF_data_25x` | 12,14,16 | 2,4,8 | 25k | 9 / 768 | 4 / 768 |

The live board (2026-08-21, 107 ranked): rank 1 **16.5365% = 127/768** seen-`N` `T=1`, 0.00% OOD;
ranks 2–6 **0.7813% = 6/768** seen-`N` and **0.3906% = 3/768** OOD; 7–10 at 2/768; 11–28 at 1/768;
29–38 at 0 seen / 1 OOD; 39–107 at zero. Nobody has certified `T=1`. The `[12,14,16]` seen-`N`
floor matching ranks 2–6 to the example is the calibration this node was built for; one bit
either way gives 17 or 1. The OOD-`N` floor is **not** a usable constraint (11 vs 4 on the same
widths across two seeds — Poisson noise on a count of ~5).

**Caveat on the neighbourhood**: it is prompt-grouped; Hard is modulus-grouped (§1). The widths
inference stands, but regenerating `hB–hF` with `--split_group modulus --separate_ood_splits true`
is one flag and should precede any further local use.

---

## 3. The port — three versions

`submission.py` is one file, two arms selected by the `ARM` constant (`emit_arm.py` rewrites it):
`control` = terminal CE; `closure` = plus [`ballistic_depth/`](../ballistic_depth/README.md) §2's
label-free *cycle* and *re-entry* terms as a `token_training_loss` over tensors the model hands out
through `auxiliary`. Both arms share every parameter. A bidirectional prompt encoder pools a state
`h₀` and a rule vector `r` ([`variable_modulus/`](../variable_modulus/README.md)'s `cond` arm); a
tied operator is applied; a decoder emits a right-aligned digit grid scattered to where
`target_positions` reads. ~9.9M (control) / ~12.3M (closure) state elements; wall-clock LR schedule.

| | v1 | v2 | v3 |
|---|---|---|---|
| depth | fixed 64-step unroll + **learned soft rung selector** | **loop-on-`T`**: `T` parsed from the prompt (walk left from the last token over digit tokens), operator applied `T` times | same |
| aux ramp | step at 30% of clock, unnormalised | linear 15→45% of clock, MSEs normalised by ‖h₀‖² | **progress-gated**: starts when an EMA of train exact crosses 0.5 (clock fallback 60%), ramps over 15% |
| cycle re-encode | learned `ReEncoder` grounded by `R(Dec(h₀), r)=h₀` | same | **through the real encoder**: decoded soft digits written in place, right-aligned into the `x` span; terms masked to width-matching rows |
| what it could do | could not memorise anything (§4.1) | memorises; closure froze on m5/h1 (§4.2) | memorises; mechanism verified; fresh-`x` unmoved (§4.3) |

Legality readings, written into the file's docstring: parsing `T` and locating the `[X]` marker by
token id are tokenizer knowledge used for control flow (an iteration count; where to write the
model's own decode), never for output computation — rule 7 forbids hard-coding the *arithmetic*,
and the problem page advertises "recurrent depth, adaptive computation". This is a reading, not a
ruling; rule 15 invites asking on Discord, which has not been done.

v3's verification (CPU smoke, `N=143`): cycle-term gradient norms into `blocks[*]` and
`token_embedding` are nonzero (in v2 they were structurally zero); feeding the true `x`'s one-hot
digits back through the re-encode path reproduces `h₀` **bit-for-bit** on m5 and m6 rows; both
arms at train exact 1.000, held-out `test` 0.083 → 0.133 and loss 9.6 → 5.2; max training loss 3.37
(v1's ramp spiked to 30,331, v2's to 18). One silent null was caught on the way: the decoded-width
scan read answer slots no row supervises, giving 0% term eligibility, now capped at the modulus's
digit width.

The v1 and v2 artefacts are identified by SHA-256 in each run's `submission.json`; the canonical
`submission.py` is v3 and a copy of the exact v3 file sits in `results/hosted/medium_m6_closure_v3/`.
v1/v2 source was edited in place and is not preserved; the hosted records are the record.

---

## 4. Results

All hosted, organizers' H100s, seed 74, batch 512. Each row's `status.json`, `metrics.jsonl` and
`submission.json` (id, SHA, change note) are under `results/hosted/<tier>_<dataset>_<arm>[_vN]/`.
Our own Modal smoke runs (e1 at 60 s, `hD` at 120 s, all arms) are on the volume and carry no
signal beyond "the pipeline works"; they are not tabulated.

### 4.1 Medium `m5` — 600 s, 81k training prompts, floor 6/768 seen-`N`, 11/768 OOD-`N`

| arm | id | steps | train loss | mean exact | `test` | `ood` | seen-`N` rungs T=1…64 | OOD-`N` rungs |
|---|---|---|---|---|---|---|---|---|
| upstream baseline | `21a7b6a9` | 72,713 | 2.049 | 0.0009 | 8/9000 · 2.26 | 3/3000 · 2.52 | 0 1 0 1 2 1 1 | 1 1 1 2 0 0 0 |
| control v1 | `e2e93fa5` | 13,880 | 2.129 | 0.0013 | 11/9000 · 2.13 | 4/3000 · 2.16 | 0 0 2 2 2 3 3 | 0 0 1 0 1 0 2 |
| closure v1 | `1bcafed2` | 10,115 | 4.293* | 0.0027 | 21/9000 · 2.14 | 9/3000 · 2.17 | 0 0 1 2 4 1 1 | 0 0 0 1 3 3 2 |
| control v2 | `c5ea6ae7` | 32,969 | **1.953** | 0.0013 | 14/9000 · 2.30 | 3/3000 · 2.54 | 1 2 1 2 0 0 1 | 0 0 0 0 0 0 0 |
| closure v2 | `16bfde04` | 24,094 | 4.289* | 0.0024 | 22/9000 · **2.14** | 7/3000 · 2.39 | 1 0 2 0 3 2 2 | 2 0 0 1 2 2 2 |

\* closure's recorded loss is the total including aux terms; its CE is not separable.

Uniform over a digit is ln 10 = 2.303. v1 arms and the baseline end within 0.15–0.3 nats of it.
**v1 could not memorise at all** — 13,880 steps × 512 ≈ 263 passes over the training set with train
loss still at chance; the soft rung-selector (temperature-8 softmax over 65 states) never sharpened
and the blurred state made digits unlearnable. **v2 starts to**: control v2 sits at ~2.13 until
~step 16–20k (≈300–360 s, ≈120 epochs) then falls steadily to 1.95 at the buzzer with train exact
0.01–0.02, test loss at chance, OOD loss *above* chance and 0/768 at every OOD rung — memorisation
beginning, confidently wrong off the training moduli. Closure v1 spiked to 30,331 at the 30%
switch-on and spent the rest of the run aux-dominated; closure v2's total is flat at 4.29 after
its ramp. The baseline creeps down monotonically (−0.017 nats per 200 s) and never accelerates.

### 4.2 Hard `h1` — 3600 s, hidden data, the measurement that matters

| | control v2 `2d450fea` | closure v2 `a6e0275a` |
|---|---|---|
| steps | 111,961 | 103,481 |
| train loss / max train exact | **0.682 / 0.600** | 4.232* / 0.008 |
| mean exact (3 scored splits) | 0.00017 | 0.00060 |
| `test` (unseen moduli, trained depths) | 3 / 9,999 · loss **11.38** | 11 / 9,999 · 2.25 |
| `ood_t` (seen moduli, held-out depth) | 2 / 10,002 · 9.74 | 3 / 10,002 · 2.28 |
| `ood_n_t` (unseen moduli, held-out depth) | **0** / 10,002 · 11.72 | 4 / 10,002 · 2.26 |
| seen-`N` rungs T=1…64 (of 768, floor 6) | 0 0 0 1 0 0 0 | 1 0 0 1 1 0 0 |
| OOD-`N` rungs (of 768, floor ~3) | 0 0 0 0 0 0 0 | 0 0 1 0 0 0 1 |
| Max T / OOD-`N` Max T | null / null | null / null |

Control leaves its 2.18 plateau at step ~35k (1131 s) and descends all the way: 1.62 at 2100 s,
1.11 at 2700 s, 0.68 at 3600 s, train exact 0.004 → 0.086 → 0.330 → 0.576 and still rising. **It
memorised 58% of the hidden training set and transferred nothing** — every rung 0/768 on both
profiles, not even the six `x²<N` cases, eval losses 4–5× ln 10 (confidently wrong). Closure v2
never left the plateau (train exact 0.008 after 103k steps): its clock-gated ramp (15–45%) landed
before the CE broke its plateau (control needed 31% of the clock), and the aux terms froze it at
chance — its eval losses are all ≈2.25, *better* than control's on every scored split, but because
it learned nothing rather than because it generalised. The single correct example at seen-`N`
`T=1` placed this entry at **rank 31 of 109** (0.1302%, tied with ~18 others); control v2 placed
109th.

### 4.3 Fixed-`N` fresh-`x` — Medium `m6` (N=1517=37×41, T∈{1,2,4}, 3,510 train rows, 90% of units seen) and `m7` (N=1763)

The seen-`N` depth cohort here is **every unit not used in training** — `ballistic_depth`'s
held-out-`x` axis in the competition's own metric. Cohort 140 (m6) / 180 (m7), floor 4.

| arm | id | steps | train exact 0.9 / 1.0 at | mean exact | `test` | `ood` (seen `x`, T=8) | **seen-`N` T=1** (of 140) | seen-`N` rungs | OOD-`N` rungs (of 256, floor 2) |
|---|---|---|---|---|---|---|---|---|---|
| m6 control v2 | `ba38a557` | 4,597 | 60 s / 460 s | 0.164 | 30/390 · 7.96 | **126/500** · 11.15 | **2** | 2 0 2 0 3 1 0 | 0 1 3 0 1 1 0 |
| m6 closure v2 | `71ac33ff` | 4,297 | 170 s / 416 s | 0.066 | 45/390 · 6.20 | 8/500 · 10.90 | **4** | 4 1 4 1 0 0 0 | 1 0 0 1 0 0 0 |
| m6 closure v3 | `d0544392` | 11,347 | 23 s / 382 s | 0.151 | 44/390 · 7.81 | 95/500 · 10.87 | **1** | 1 0 1 2 3 2 3 | **3 2 2 2 2** 0 0 |
| m7 control v2 | `ebdd7c29` | 5,370 | — / (loss 0.0) | 0.053 | 10/450 · 9.91 | 50/600 · 13.76 | **1** (of 180) | 1 0 0 1 1 0 1 | 2 2 3 1 0 1 3 |

All arms memorise completely. On fresh `x` at `T=1` every arm is at or below the floor; the spread
1–4 of 140 is one band of noise. `ballistic_depth` cut1's 0.001 → 0.32 (3 seeds, flat in depth) did
not reproduce — and did not reproduce *after* the reason we thought it hadn't (v2's `ReEncoder`
never grounding `Enc`) was removed and the fix verified. Two further readings, offered with the
weight the counts support: (a) closure v2's collapse on held-out depth (126 → 8/500) was v2's
mid-fit ramp, not the consistency idea — v3 restores 95/500; (b) v3 is the only arm above the
OOD-`N` floor at `T=1` (3 vs 2) and ≥ every other arm on the first five OOD-`N` rungs, the same
direction as the Hard closure arm's 4 vs 0 on `ood_n_t` — two or three examples out of 256, single
seed, two datasets agreeing; not a result, noted.

### 4.4 Easy `e5` sanity (60 s, ~500–1000 steps, nothing memorises)

control v1 / closure v1 / control v2 / closure v2: mean exact 0.0050 / 0.0063 / 0.0042 / 0.0063;
seen-`N` `T=1` 1 / 2 / 4 / 1 of 512 against a floor of 22. A v3 sanity run (`519ccba7`, 0.0092)
ran but was not saved. These established that the hosted evaluator accepts and runs the port; they
measure nothing else.

---

## 5. Reading

- **The instrument is calibrated and cheap.** Medium m5 is a faithful size/shape proxy for Hard
  (≈81k vs ≈90k training prompts, matching floor), costs nothing, takes 20 minutes, and returns the
  full ladder. At 1/6 of Hard's clock it sees the *onset* of memorisation and nothing after; Hard
  sees memorisation complete (58%, still rising) and is the only tier where post-memorisation
  behaviour is observable. Each daily Hard attempt is a full-resolution measurement.
- **The zero is the rule-acquisition wall, not a budget or a port defect.** Once the port trains
  (v2), control does at Hard exactly what a monolithic operator does in
  [`variable_modulus/`](../variable_modulus/README.md) §3 and [`rule_acquisition/`](../rule_acquisition/README.md):
  per-prompt recall with no transfer to fresh `(N, x)`, on seen or unseen moduli. The organizers'
  own baseline sits below the floor too, as do ranks 39–107.
- **The board decomposes along this program's own axis.** Ranks 2–6 score precisely the
  no-reduction cases on seen *and* unseen moduli — the profile of a model that learned the
  **multiply** (`x → x²` as digits, generalising to fresh small `x`) and not the reduce, the
  decomposition [`rule_acquisition/`](../rule_acquisition/README.md) measured. Rank 1's 127/768
  with 0% OOD is what interpolating the structurally-dense 12-bit cell would score. Neither is a
  claim about what those entrants built; both are consistent with every number here.
- **Closure on fixed `N` did not transfer to the port**, with the mechanism implemented as in the
  research code and checked. Candidates for what was load-bearing in `ballistic_depth` cut1 and
  absent here: training depths 1..6 vs {1,2,4}; fixed-width zero-padded fields and no `N`/`T` in
  the prompt; the specific architecture (d_model 256, VQ-capable state) vs a 384-wide pooled
  encoder; a 2.1M-parameter model at 8k steps vs 9.9M at 4–11k; something about `N=893`. This is an
  open question for the program, not a verdict on the idea; it is also not a nine-day question.
- **What "performing well" would now take, by tier.** Tie for rank 2 (the floor): a model whose
  small-`x` squaring generalises — digit-arithmetic inductive bias, which a pooled-state recurrent
  operator lacks; the fixed-`N` sets read this directly (success = exactly the floor count on fresh
  `x`). Rank 1: something that works on the dense cell. Certifying `T=1`: the rule, which nothing
  legal in the tree has. On current evidence the first is plausible in days, the second is not in
  reach, the third is not a competition outcome.

---

## Reproduce

```bash
# upstream at the pinned commit, its own venv (needs uv >= 0.12: ~/.local/bin/uv), its own tests
cd ~/Code && git clone https://github.com/tilde-research/one-layer-deeper.git one-layer-deeper-head
cd one-layer-deeper-head && git checkout 4ceff95 && ~/.local/bin/uv venv && ~/.local/bin/uv sync
.venv/bin/python -m unittest discover -s tests                         # 129 tests, OK
bash scripts/generate_datasets.sh                                      # all 20 public sets
python <node>/hard_configs.py generate                                 # hB..hF (prompt-grouped; see §2 caveat)
python <node>/audit_dataset.py data/generated/<name> [...] [--json]    # moduli, coverage, floors, periodicity

# local CPU smoke of an arm (upstream's smoke_cpu.json allows 0.1 s — too short)
python <node>/hard_configs.py manifest /tmp/smoke.json --cpu --seconds 30
python <node>/emit_arm.py all /tmp/arms
.venv/bin/python -m benchmark.runner --manifest /tmp/smoke.json --submission-file /tmp/arms/closure/submission.py

# hosted (free; needs `one-layer login` — Jasper's GitHub identity, never from an agent)
uv tool install git+https://github.com/tilde-research/one-layer-deeper.git
one-layer validate /tmp/arms/closure/submission.py
one-layer submit /tmp/arms/closure/submission.py --tier medium --dataset m6      # or --tier hard
one-layer status <id> --json | sed -n '/^{/,$p' > status.json                    # full rung profile
one-layer metrics <id> --output metrics.jsonl                                    # coarse train curve

# our Modal harness (app old-dress-rehearsal, volume old-dress-rehearsal-data — separate from shared.py's)
cd experiments/
MODAL_PROFILE=chromatic modal run one_layer_deeper/dress_rehearsal/modal_rehearsal.py::generate_public
MODAL_PROFILE=chromatic modal run one_layer_deeper/dress_rehearsal/modal_rehearsal.py::generate_hard
MODAL_PROFILE=chromatic modal run one_layer_deeper/dress_rehearsal/modal_rehearsal.py::audit
MODAL_PROFILE=chromatic modal run one_layer_deeper/dress_rehearsal/modal_rehearsal.py::evaluate \
    --dataset m5 --arm control --seconds 120 --tag smoke_m5_control     # H100; --arm control|closure|baseline
MODAL_PROFILE=chromatic modal run one_layer_deeper/dress_rehearsal/modal_rehearsal.py::evaluate_l4 ...   # tags prefixed l4_, never tier-faithful
MODAL_PROFILE=chromatic modal run one_layer_deeper/dress_rehearsal/modal_rehearsal.py::collect
```

Volume layout: `/data/generated/<dataset>/`, `/data/audit/<dataset>.json`,
`/data/results/<tag>/{result.json,runner.log}`. Use `--detach` past a couple of minutes, per
`/run-experiment-on-modal`[^private]. Only smoke
budgets (≤120 s) were ever run on our own Modal; every real-budget number above is hosted.

**Gotchas.** On small datasets the DataLoader, not the model, bounds steps (e1: 600 rows = one
batch per epoch with worker respawn; m6: 4.6k steps in 600 s) — `Submission.batch_size` is the
knob. Evaluation gets half the training budget and the depth profile runs after the scored
splits; the port used ≤17 s of it. Hard and Medium are serialised behind one job per account.
`one-layer status` prints a human table before the JSON; strip it with `sed -n '/^{/,$p'`.

---

## Next steps (possibilities, not a plan)

1. **The floor tier, deliberately.** Replace the pooled-state operator with a per-digit
   bidirectional transformer carrying digit-position structure (the arithmetic-transformer
   toolkit), keep loop-on-`T`, and read `m6`–`m9`'s fresh-`x` `T=1` rung: the target is *exactly*
   the floor count, which would mean the multiply generalises. Free on Medium; the realistic
   ceiling for the remaining days.
2. **Why didn't fixed-`N` closure transfer?** One ablation at a time back toward
   `ballistic_depth` cut1 — depths 1..6, fixed-width fields, the research architecture — on
   `m6`/`m9` with the hosted evaluator as the grader (or the research harness with `m6`'s data).
   A program question worth a cut of its own; the candidates are listed in §5.
3. **The OOD-`N` whisper.** v3 above the OOD-`N` floor on `m6` and Hard's `ood_n_t` 4 vs 0: a
   variable-`N` Medium set (`m3`/`m4`/`m5`) with control vs closure v3, two seeds, would say whether
   it is anything.
4. **Epochs vs updates.** Every break-through in the port's probes tracked passes over the data;
   `Submission.batch_size = 2048` (358 epochs in 600 s vs 199) and the evaluator's legal
   `should_reuse_batch` (≤8 updates per batch) are untested levers.
5. **Regenerate `hB–hF` modulus-grouped** before any further local use (§2 caveat), and ask
   `#one-layer-deeper` about loop-on-`T` if a submission is going to be pushed.
6. **Hard attempts are free samples of the hidden distribution** and return full ladders; there
   is no reason to leave a day's attempt unused while anything new is ready.

[^private]: Not mirrored: this link points to a document in the private lab repo (the roadmap, the queue, an unrun spec, reading notes, or a conversation). See the top-level README for what is held back and why.
