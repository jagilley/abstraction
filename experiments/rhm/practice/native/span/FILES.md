# FILES — `native/span` (Port 2: the chunk's corridor, as a native executor primitive)

**Up**: [`../SPEC.md`](../SPEC.md) (the record of what was asked) ·
[`../../README.md`](../../README.md) (rhm/practice).
**Substrate donor**: [`../../ratchet/`](../../ratchet/README.md) — `ratchet.py` forked verbatim
(fork notice at the top of `span.py`), `macros.py` **imported**, never copied; `ratchet/` and
`teacher_slot/` are not modified.
**Measured negative this port is built against**:
[`../../teacher_slot/handle/`](../../teacher_slot/handle/) (chunk *identity* into the generator;
plant moved negatively, dose-ordered) — `handle.py`/`handle_net.py`/`launch_detached.py`/
`analyze_handle.py` are also the fork and reduction convention followed here.

No README in this folder by design: READMEs come after the numbers are discussed.

## Code files

| file | purpose |
|---|---|
| `span_net.py` | The port itself. `SpanHead` (macro-slot-conditioned, joint autoregressive emission of the span's level-1 features); `trunk` (a bit-identical split of `BlockInfiller.block_logits` returning the pooled block hiddens the head reads); `dp_features` (`macros.macro_features`' max-sum DP on already-computed logits — the parity target); `build_head` (mints off a dedicated `torch.Generator`, restoring the shared stream); `PlainExecutor` / `SpanExecutor` (what the beam calls to materialise a move; closed slots run `macros.apply_any` itself); `parity` (held-out exact-match against the DP); `span_train_terms` (the self-imitation loss). Module header carries the design decisions. |
| `span.py` | Verbatim fork of `../../ratchet/ratchet.py` + the port. New: `finetune_generator_span` (plant loss + span self-imitation in the SAME optimizer steps), per-arm torch streams keyed by the twin (`STREAM`), the `span_true` / `span_mined` arms, the per-macro parity gate, the block-level counterfactual tally, the span-head-bypassed competence probe (`e_nospan`), and `span_selfcheck` (gates S-1..S-6). Entrypoints: `span_run`, `span_selfcheck_remote`, plus ratchet's `cal_ladder` / `cal_parse` / `cal_stale` / `selfcheck_remote` inherited unchanged. |
| `launch_detached.py` | Session-isolated `modal run --detach` launcher (handle/'s, retargeted). Logs to `results/launch_<tag>.log`. |
| `analyze_span.py` | Reduction. Retargets `ratchet/analyze_ratchet.py` for every ratchet-standard readout, then adds: the twin bit-identity gate, the per-macro parity trajectories and first-open cycles, the gate-flapping tally (transitions, fraction of cycles open, fraction of the parity series inside a ±0.02 band around τ, and e-vs-twin split by whether the head fired that cycle), the arm x era table (e / priced time / groundings & materialisations per solve) with the counterfactual block ledger printed beside it, the plant-guard contrast against `handle/`'s numbers, the shadow-audition contrast (both at the era's active level and pinned to a fixed level, which is the form `handle/` reported), and the span-head-bypassed competence twin. Figures + `summary.json` under `figures/<tag>/`. |

## Gates (`span_selfcheck`, CPU)

| gate | what it asserts |
|---|---|
| S-1 | minting a span head draws nothing from the shared torch stream (own generator) |
| S-2 | `span_net.trunk` is a bit-identical split of `block_logits`, and `span_net.dp_features` reproduces `macros.macro_features` exactly (max\|d\| = 0.0) |
| S-3 | a CLOSED slot executes through `macros.apply_any` itself — `PlainExecutor` and a non-firing `SpanExecutor` are bit-identical to `apply_any` on every move |
| S-4 | an OPEN slot writes canon-rendered leaves at exactly the macro's own positions and nowhere else |
| S-5 | the span loss reaches the shared trunk (non-zero core gradient) — the treatment is real |
| S-6 | the gate can open at all: distilling the DP on a frozen trunk drives held-out exact-match parity from 0 towards 1.0 at both levels |

## Pricing, stated explicitly

The headline ledger is ratchet's, unchanged: `t = n_ground*d_fb + n_mat*c_mat`, `d_fb=1.0`,
`c_mat=0.05`, **one materialisation per move application per row whatever the span**, and a
grounding is a value/feedback call (the head never calls the value). So in the ledger the run is
priced by, a span-head call and a DP call cost the same and the port buys nothing on `t`;
head training is unpriced, exactly as plant training is unpriced for both arms alike. Separately
and never folded in, the runner tallies `mat_{base,dp,head}` and `blk_{base,dp,head}` over the
priced beams only (practice + metering; probes excluded at the snapshot), where a level-`l` DP
execution is charged its `span = s**(l-1)` blocks of infill and a head execution 1 — the SPEC's
"one-pass materialisation against s blocks" reading, computable from the log as an explicit
counterfactual column.

## Volume layout

`rhm-scaling-data:/data/rhm_practice_native_span/<tag>/{setup.json,<arm>/results.json,done.txt}`.
Fetched copies and figures under `figures/<tag>/`.

## Runs

| tag | what it is |
|---|---|
| `smoke0` | attached `--quick` smoke; in quick mode the gate is forced open (`span_tau=0`, `span_min_hold=32`) so the firing path is exercised end-to-end |
| `sp_s0` | the main run: 5 arms (`never_base`, `given`, `span_true`, `practice_late`, `span_mined`), seed 0, 3 eras x 30 cycles |

## Reproduce

```bash
cd experiments/            # MODAL_PROFILE=chromatic
modal run rhm/practice/native/span/span.py::span_selfcheck_remote
modal run rhm/practice/native/span/span.py::span_run --quick --tag smoke0

python3 rhm/practice/native/span/launch_detached.py --fn span_run --tag sp_s0 --seed 0 \
    --arms "never_base,given,span_true,practice_late,span_mined" \
    --eras "1:6,2:3,3:1" --era-cycles 30 \
    --budget 4 --pr-width 16 --g-budget 58 --max-macro-level 3 \
    --n-pr 64 --n-rt 384 --n-score 512 --n-grad 4 --value-lr-online 3e-5 \
    --gen-lr 1e-4 --gen-steps 20 --plant-holdout 0 \
    --mine-from chosen --mine-cap 8 --mine-support 3 \
    --sil-c 0.06 --sil-cv 0.15 --sil-win 5 --sil-hold 2 --sil-min-cycle 6 --lp-min-drop 0.10 \
    --early-offset 1 --late-offset 3 --probe-every 4 \
    --span-lam 1.0 --span-lr 1e-3 --span-tau 0.95 --span-min-hold 256

python3 rhm/practice/native/span/analyze_span.py --tag sp_s0 --fetch --figures
```

## Decisions on the record (the SPEC left these open)

- **Conditioning form.** The head is conditioned on the macro **slot** `(level, node)` — which
  committed macro is being called — and never on a table entry index. The entry a call resolves
  to is precisely what the DP computes, so conditioning on it would presuppose the thing being
  replaced, and in a run nothing supplies it (entry proposal is Port 1's). The head's output
  alphabet is level-1 **features**, so no arbitrary label is ever a target: that is the
  identity/corridor distinction made operational against `handle/`.
- **New parameters** (`span_net.SpanHead`): slot embedding, span-step embedding, emitted-feature
  embedding, one input projection per span offset, a context projection, and a 2-layer MLP head.
  The trunk is structurally untouched; the head reads it and its loss backpropagates into it.
- **Loss and schedule.** Cross-entropy on the span's features, teacher-forced on the DP's own
  emission, added to the plant's loss at `span_lam` in the *same* optimizer steps (`n_steps`
  untouched). Continuous training from the commit cycle, firing only where parity passes — not
  a two-phase collect/train/fire — because the head's target is a deterministic function of the
  current trunk and the trunk moves; a frozen collect phase would train against a stale executor.
  The head's own parameters use `span_lr` (10x `gen_lr`) since they start from random init; the
  **trunk stays at `gen_lr`**, so the plant-guard contrast is not confounded by a different plant
  rate.
- **Training distribution.** Self-imitation on the executor's own materialisations, sampled from
  the **beam's own macro-call distribution** (every tip the beam materialises that macro on),
  with targets recomputed from the current executor. The SPEC's narrower phrasing was "selected
  solved traces"; widened deliberately, because parity is gated on the deployment distribution
  and training only on selected traces manufactures a train/deploy mismatch the gate would then
  simply refuse. §18's selection-before-regression concern does not bite here: the target is a
  deterministic operator output, not an outcome, so there is nothing to average over.
- **Parity gate.** Held-out exact-match of the emitted span leaves against what `apply_any`
  writes on the same instances, per macro, `tau = 0.95`, minimum 256 held-out rows, re-checked
  every cycle (it can close again — the recert idiom). Held-out is a ~10% split of the
  executor's own call buffer keyed by a **bijective code of the observation**, so a context that
  recurs (the metering set is fixed per era) can never straddle the split. `tau = 0.95` is set
  against the substrate's own read floor — the plant's clean-config parse accuracy is 0.62-0.65,
  so a 5% mismatch budget is an order of magnitude tighter than the executor's own error and
  below the twin-pair noise floor on `e` (0.01-0.03) — and the full parity trajectory is logged
  so a different threshold can be read off the record.
- **Below parity → DP+infill.** A macro whose slot is closed is executed by `macros.apply_any`
  itself, not by a re-implementation (gate S-3), so the fallback cannot introduce drift.
- **Mining.** Unchanged, and it keeps working as instrumentation. The miner parses the beam's
  chosen configuration with the **reader**, not the generator, and the head emits level-1
  features rendered through the same `canon` the DP path uses — so a span the head wrote is the
  same kind of object in the same alphabet, and the miner cannot tell the difference. Where the
  head is wrong (at most 1 - tau of its calls) the miner records a different tuple, which is
  exactly what already happens when the DP is wrong. Below parity the head does not fire at all.
- **Merge licensing deferred**, per the SPEC's own reasoning: ratchet has no venue/index, so
  demand-invariance is not expressible; `span_mined` vs `span_true` is the proxy. If the answer
  turns out to matter, `../../merge/` is where it becomes expressible.
- **Arms.** The SPEC's table lists both a `span` row and a `span_mined` vs `span_true` row.
  Resolved as one family, not three arms: `span` names the treatment, and `span_true` /
  `span_mined` split it by which vocabulary supplies the corridor content. `never_base` is
  carried so the earned-vs-given fraction is computable in-tag.
