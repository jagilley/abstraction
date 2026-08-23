# Second pass — a per-digit carrier, a legality-bounded lever, and a note on the board

**Up**: [../README.md](../README.md) (dress_rehearsal) · **Files**: [FILES.md](FILES.md)
**Children**: [`digit_port/`](digit_port/NOTES.md) (track A — the per-digit port; `NOTES.md` is the run-by-run record, [`FILES.md`](digit_port/FILES.md) the code index) · [`board_note/`](board_note/README.md) (track C — the externally-readable note).
**Sibling cut, linked not moved**: [`../../rule_acquisition/staged_reduce/terminal_only/`](../../rule_acquisition/staged_reduce/terminal_only/NOTES.md) (track B — it forks `staged_reduce.py` by absolute module path and is semantically that node's question, so it stays under its donor).
**Upstream**: `4ceff95` · deadline **2026-08-31 22:00 PT** · board read 2026-08-22/23.
**Status**: track A — **11 hosted runs** (8 Medium, 3 Hard) + **29 Modal H100 runs** in six 600 s sweeps and one 3600 s run (≈ $30); track B — **4 Modal L4 arms** at 600k steps (≈ $26); track C — written, **not posted**. Single seed throughout (hosted: the evaluator's `[74]`). **Date**: 2026-08-23. Orchestrated as one conversation with three Opus implementer agents; this README is the writeup for the whole scope, the children carry the records.

---

## One-liner

Nine days before the deadline, a state-of-the-line review read the competition as a log-error ladder on one starved atom ([`exact_atom/`](../../rule_acquisition/exact_atom/README.md)), the board as decomposing into a no-reduction floor tier and a dense-12-bit-cell tier ([`../README.md`](../README.md) §2, §5), and the only legal supervision as terminal CE plus label-free self-consistency. Three tracks were launched in parallel against that reading. What came out:

1. **On a single modulus, a per-digit carrier learns a step map that generalises to unseen `x` about one time in four and composes exactly.** Fresh-`x` `T=1` **35/140, 50/180, 67/290** (23–28%, ~10× the analytic floor) on `m6`/`m7`/`m9`, the ladder **flat out to `T=64`**, held-out depth `T=8` at **500/500, 595/600, 592/600**. The parent's pooled-state port reads 2/140 and 1/180 on the same sets. Three moduli, the organizers' evaluator and budget. (§2.3)
2. **The same architecture cannot fit a multi-modulus set in budget.** Hard (93,638 steps): train-exact 0.018, test loss 2.36 (above ln 10), **0/768 at `T=1`**; the parent memorises 58% of Hard at comparable steps, this memorises 2%. `m5` (204 moduli) leaves its plateau only at ~28k steps and only into memorisation without transfer. The board position is unchanged — the parent's closure v2 at 1/768 (rank 31) remains our best Hard entry. (§2.4)
3. **A staged forward pass under terminal-only supervision is a degenerate fixed point.** All four cells (8/142 moduli × straight-through/soft re-grounding) at floor; CE flat from the first log point where the *monolithic* control on identical data descends; the seam collapses to ~12 distinct values over the whole pool. Architecture alone does not make `staged_reduce`'s decomposition discoverable from terminal labels. (§3)
4. **Closure is null on the digit carrier.** With the ramp gate fixed and the terms live for ~85% of a fully-fitted run, closure is slightly worse at all seven rungs and both scored splits, at ~19% of the step budget. The carrier subsumes the cycle/re-entry terms; [`../README.md`](../README.md) §5's "why didn't fixed-`N` closure transfer?" resolves as *the property it bought is now structural*. (§2.5)
5. **The board note exists** ([`board_note/`](board_note/README.md) + `discord.md`), every number re-derived from source; it caught and corrected an error in our own record (12-bit coverage is 58% of distinct `(N,x)` pairs, not the ~84% previously stated). Not posted; Jasper's call. (§4)

Two corrections to the orchestrating reading are worth stating up front: Hard's allowance is **3 attempts per UTC day**, not 1; and `m5` is a faithful proxy for Hard's *floor* but not its *fitting rate* (Hard is modulus-grouped and far sparser per modulus).

---

## 1. Why these three tracks

The review ([`../README.md`](../README.md) §5 and the program's [`exact_atom/`](../../rule_acquisition/exact_atom/README.md)) left three facts standing: certifying rung `T` needs one-step error `eps ≲ 9e-4/T` and depth is cheap once a rollout can be re-grounded through its own decode, so the whole competition is "one exact modular squaring"; ranks 2–7 sit on the `[12,14,16]`-bit no-reduction floor (6/768) and rank 1 (129/768, 0% OOD) is consistent with interpolating the 12-bit cell alone, which is 256 of 768 and structurally dense (14 moduli exist); and the rules close every supervision channel except the terminal label and label-free consistency of the model's own states, which is why [`staged_reduce/`](../../rule_acquisition/staged_reduce/README.md)'s rule-axis result — the one thing in the program that ever moved held-out `N` — could not be ported.

So: **A** — the architecture the board's two tiers both point at (digit-position structure; the floor tier is small-`x` squaring as digits, the rank-1 tier is fixed-`N` rule acquisition on a dense cell), with the free fixed-`N` Medium sets as the instrument and Hard's free daily attempts as samples of the hidden distribution. **B** — the one experiment that could open a legal lever: the staged forward pass with loss on the final remainder only, which had never been run. **C** — the calibration facts written for participants and organizers.

Jasper's standing rule for the round: anything not explicitly forbidden is legal; tokenizer knowledge used for control flow is fine; arithmetic on digit values in the loss or forward pass is not.

---

## 2. Track A — the per-digit port ([`digit_port/`](digit_port/NOTES.md))

### 2.1 Design

The recurrent state is a **fixed-width digit grid** — slot `j` is the `j`-th digit from the right, zero-padded left, each slot a distribution over the vocabulary. The initial state is `x` right-aligned (a gather on the `x` field's token positions); a tied operator maps grid → grid and is applied exactly `T` times (`T` parsed from the prompt tail as in the parent port); the grid after `T` applications is scattered to where `target_positions` reads. `N` enters the operator only through per-slot embeddings of its own digits (the final configuration; the initial one also cross-attended to an `N`-only encoder). Abacus positional scheme (field id + index-from-the-right within field). Residual branches zero-initialised and the digit head tied to the state embedding, so the step starts as a near-identity copy. Loss = terminal CE + a zero-padding convention term on slots above the answer width (uses the evaluator's `valid_mask` and a constant symbol; says nothing about the answer). The forward pass is literally `y = f_N^T(x)`: `x` enters only as the seed, `T` only as the loop count.

The bet, stated before the runs: with a canonical carrier, `T=2` supervision *is* supervision of `f∘f` with the same `f`, so the `T=1` rung is identified by the deeper rungs; the parent's pooled state has no such factorisation. Legality, flagged in the docstring and `NOTES.md`: the gather that seeds the grid is the most aggressive use of tokenizer knowledge (the hard-attention limit of a learned copy; no digit *value* is read except through a learned embedding); `ARM="digit_learned"` replaces it with a learned slot-query cross-attention so the conservative reading's price is measurable — not run. Not asked on Discord.

### 2.2 What the six sweeps falsified

Each sweep was five or four 600 s H100 runs on our Modal against the public sets (`modal_sweep.py`, the organizers' evaluator and data; only Hard's data is hidden). Recorded here because each read overturned the one before it.

| sweep | arm(s) | what it tested | what it read |
|---|---|---|---|
| `s1` | soft carrier, 4-layer op + cross-attn; `m5`, `hD_t_shallow` (`T∈{1,2,4}`), batch 2048, no-bottleneck `carry`, wide | why Hard run 2 never left its plateau | every arm flat at its dataset's marginal optimum (2.18–2.19) after ~900 steps; neither the missing `T=1` row nor the digit bottleneck is the cause |
| probe | `|dL/dstate|` at applications 1/2/3 on `m1` | is the carrier passing gradient? | soft: 4e-6 / 2e-4 / 1e-2 — **~50× decay per application**; straight-through worse (2e-8 → 3e-3); a residual stream flat |
| `s2` | `STATE_MODE=residual` (logits carried additively) | fix the gradient | `m6` memorises faster and **fresh-`x` 27 → 0, held-out depth 496 → 106/500**: the carried magnitude encodes depth and the map stops being tied; `hD` leaves its plateau (1.970), `m5`/`m1` do not |
| `s3` | `STATE_MODE=norm` (residual stream rescaled to fixed RMS); `CTX_SCOPE=prompt` | keep the identity path, remove the depth channel | `m6` held-out depth **500/500** and fresh-`x` **3/140** — composition and fresh-`x` generalisation separate, and only the contractive soft carrier has the latter; `m5` unmoved under either |
| `s4` | `STATE_MODE=highway` (`+G·(state − state.detach())`: forward **bit-identical** to soft, app3/app1 gradient ratio 1.0 vs 4e+05) | is the decay the blocker? | `m6` fresh-`x` **27 → 2**, depth 496 → 311, test 275 → 84, train-exact at 7.9k steps 0.99 → 0.82; `m1`/`m5` unmoved. **Removing the decay from an identical forward pass fits slower and generalises 13× worse** — the contraction reads as the mechanism (greedy per-application training of a tied map), not a bug |
| `s5` | soft carrier; 2 layers ± cross-attn (`ops2` vs `fast`) | throughput on the carrier that generalises | `s5_m6_soft` = hosted run 1's config at **6,091 vs 8,929 steps** (calibration: local ≠ hosted); **`soft_fast` 35/140 flat, 499/500, 286/390** while `ops2` (cross-attn kept, same steps) reads 1–3/140 — **`OP_CROSS=False` is the win** (digit-aligned conditioning), and 3.1× throughput |
| `s6` | op depth 1/2/3; LR 2e-3; `m1` at matched throughput; `m5` at **3600 s** | what else moves, and does `m5` fit at full budget | 1/2/3 layers indistinguishable (34–38/140; 1 layer is 1.43× faster); LR 2e-3 halves generalisation (18/140) while fitting; `m1` at 15,705 steps still at ln 10; **`m5` leaves its plateau at ~28k steps** then memorises without transfer (§2.4) |

The two theories the orchestrator and the implementer would both have shipped — "the softmax is saturated" (false: the decay is temperature-independent) and "the gradient decay is the blocker" (false: `s4`) — died to cheap probes. Worth carrying as method.

### 2.3 Results — a single modulus, hosted

Fixed-`N` Medium sets, `T∈{1,2,4}` trained, 600 s, seed 74; the fresh-`x` cohort is every unit not in training; floors recomputed from each `depth_t_1` split (`x² < N`).

| set | `N` | cohort | fresh-`x` `T=1` | floor | ×floor | rungs `T=1…64` | held-out depth `T=8` | `test` | parent port |
|---|---|---|---|---|---|---|---|---|---|
| `m6` | 1,517 | 140 | **35** (25.0%) | 4 | 8.8× | 35 35 38 39 39 39 38 | **500/500** | 291/390 | 2 |
| `m7` | 1,763 | 180 | **50** (27.8%) | 4 | 12.5× | 50 51 52 51 51 52 51 | 595/600 | 322/450 | 1 |
| `m9` | 1,927 | 290 | **67** (23.1%) | 6 | 11.2× | 67 63 67 67 66 66 66 | 592/600 | 327/465 | — |

Hosted run 1 — the initial 4-layer, cross-attention configuration on `m6` — read 27 26 27 27 27 28 28 of 140 with `T=8` at 496/500; the `OP_CROSS=False` configuration above is the canonical `submission.py` (`STATE_MODE="soft"`, `N_OP_LAYERS=2`, `OP_CROSS=False`, `head_scale=3.0`).

Two properties, and they separated cleanly across the sweeps: **composition** (a flat ladder to `T=64`, an untrained depth at 99–100%) survives any canonical carrier (soft 496, norm 500, residual 106 of 500); **fresh-`x` generalisation** appears only under the contractive soft recompute. The first was an open question for the program in the competition's own metric ([`ballistic_depth/`](../../ballistic_depth/README.md) §9 reached it only with test-time re-projection on seen bases); the second is the rank-1 tier's regime — fixed-`N` rule acquisition — at a rate nobody in this program had measured.

### 2.4 Results — many moduli, and Hard

| run | set | config | steps | train loss / exact | `test` | seen-`N` rungs `T=1…64` | OOD-`N` rungs |
|---|---|---|---|---|---|---|---|
| hosted 2 | Hard `h1` | soft, 4-layer + cross | 32,755 | 2.200 / ≤0.004 | 4/9,999 · 2.205 | 0 0 0 0 1 0 0 | 0 1 0 0 0 0 0 |
| hosted 4 | Hard `h1` | highway | 41,938 | 2.212 / — | 4/9,999 · 2.19 | 0 0 0 0 0 0 1 | 0 0 1 0 0 1 1 |
| hosted 5 | Hard `h1` | soft, 2-layer, no cross | **93,638** | 2.118 / 0.018 | 2/9,999 · **2.360** | 0 1 0 0 0 0 0 | 0 0 0 0 0 0 0 |
| parent control v2 | Hard `h1` | pooled state | 111,961 | 0.682 / 0.576 | 3/9,999 · 11.38 | 0 0 0 1 0 0 0 | 0 0 0 0 0 0 0 |
| local `s6` | `m5`, 3600 s | soft, 2-layer, no cross | 80,365 | 1.575 / 0.152 | 13/9,000 · **2.996** | 0 1 3 2 2 1 3 | 0 1 1 0 0 2 1 |
| hosted 7 | `m5`, 600 s | same | 26,203 | 2.178 / — | 25/9,000 · 2.162 | 1 0 1 3 2 3 1 | 3 0 1 3 3 2 3 |
| hosted 3 | `m1` (fixed `N`=10,403, `T∈{4,8,16}`) | soft, 4-layer | 4,964 | 2.317 / — | 0/3,000 | 0/192 at every rung | 0/512 |

On `m5` the break comes at ~28,000 steps (loss 2.183 at 3.9k → 2.138 at 27.9k → 2.049 at 31.9k → 1.575 at 79.9k); every earlier `m5` run, local or hosted, topped out below it, so "twelve configurations failed to move `m5`" was a budget artefact and is retracted in the record. What is behind the break is the rule-acquisition wall of [`variable_modulus/`](../../variable_modulus/README.md) and [`../README.md`](../README.md) §4.2, now from a second architecture: test loss *above* ln 10 (confidently wrong off the training prompts) and **0/768 at seen-`N` `T=1` — not even the six no-reduction cases**. Hard fits more slowly still (modulus-grouped; ≈13/38/133 training moduli per width): at 93.6k steps it is at 2.118 / 0.018 where the parent's pooled port memorised 58%. That is the digit carrier working as designed and against us here — with the state fixed to `x`'s digits and `N` the only conditioning there is no per-prompt shortcut, so fitting requires computing.

`m1` is the other instructive null: fixed `N`, no modulus family to learn, and still exactly at its marginal optimum with throughput controlled (15,705 steps in `s6`, more than the 9,187 that fully fit `m6`). What separates it from `m6` — no `T=1` row (min `T`=4), a 6.7× larger unit table, 5× fewer epochs — remains confounded.

### 2.5 The closure null

`digit_closure` (label-free sharpness + re-entry through the model's own hard decode, the parent's v3 mechanics on the grid) was run three times. On `m5` the ramp's **clock fallback fired at 72% of the clock on an unfitted model** — the parent README's v2 failure re-introduced by v3's fallback; run 6 therefore measured a mistimed ramp, not closure, and the matched plain control (run 7) confirms both arms sat at the marginal optimum. With the fallback removed (the terms now engage only when the train-exact EMA crosses 0.5, and otherwise the arm degenerates to the control), run 8 on `m6` had the terms live for ~85% of training on a fully-fitted model:

| `m6`, hosted | plain (run 9) | closure (run 8) |
|---|---|---|
| fresh-`x` rungs `T=1…64` (of 140) | 35 35 38 39 39 39 38 | 33 34 35 36 35 36 35 |
| held-out depth `T=8` (of 500) | 500 | 497 |
| `test` (of 390) | 291 | 287 |
| steps | 11,143 | 9,798 |

Slightly worse at all seven rungs and both splits — individually noise, consistently signed — at ~19% of the step budget. Read: the cycle/re-entry terms exist to stop a pooled state drifting off its own encoder's manifold; a canonical digit grid cannot drift, and a ladder already flat to `T=64` has no inconsistency left to remove. The carrier subsumes the terms. Arm dropped.

### 2.6 Caveats

- **Single seed**, hosted and local; ranks and the flat-ladder shape are the claims, a spread of 1–4 of 140 is noise.
- **Hosted vs local is not a constant factor**: 1.47× (`m6`, 4-layer), 1.21× (`m6`, 2-layer), 0.98× (`m5`). Two reads in `NOTES.md` compared arms across machines and had to be redone with a matched hosted control (runs 7 and 9).
- **`m5` proxies Hard's floor, not its fitting rate** (prompt-grouped vs modulus-grouped).
- The `digit_learned` legality arm is defined, not run; the loop-on-`T` and gather readings have not been asked on Discord.
- The closure null is on the set where the model fits; on Hard the terms would never engage, which is the same conclusion by a different route.

---

## 3. Track B — the staged forward pass under terminal-only supervision ([`terminal_only/`](../../rule_acquisition/staged_reduce/terminal_only/NOTES.md))

**Question.** [`staged_reduce/`](../../rule_acquisition/staged_reduce/README.md) moved held-out `N` from floor to 0.98 by training a tied stage map on a *self-made* single-stage distribution and chaining it at test time. Under the competition's rules that distribution is data augmentation, and every composed arm in the program trained terminal-only sits at floor — but every one of those had a monolithic reduce. Can a **staged forward pass** — `S=6` applications of one tied stage map, each consuming the modulus, the current remainder, and the next radix digit, with the remainder re-grounded through the model's own decode at every seam — self-organise the decomposition from the terminal label alone, given breadth of moduli?

**Design.** Task identical to the donor's `sr3_mono` (`(N, y) → y mod N`, `y ~ U[0, N²)`, 3 digits), loss on the final remainder only; architecture, parameter count (2,122,782), family, hash split, budget (600k steps) and eval pools bit-identical to the donor's and verified at launch. A 2×2: 8 vs 142 moduli × straight-through hard re-grounding vs soft in-place re-encode. In `st` mode the seam is a genuine bottleneck (3 decimal digits against a prefix of up to five), so whatever is carried must compress the prefix.

| arm | moduli | seam | chain held-out `y` | floor | held-out `N` | floor | `stage_all` (donor's training distribution, never seen) | floor |
|---|---|---|---|---|---|---|---|---|
| `to3_st10` | 8 | st | 0.0034 | 0.0018 | 0.0013 | 0.0016 | 0.0025 | 0.100 |
| `to3_soft10` | 8 | soft | 0.0033 | 0.0018 | 0.0019 | 0.0016 | 0.0025 | 0.100 |
| `to3_st10_many` | 142 | st | 0.0052 | 0.0027 | 0.0022 | 0.0020 | 0.0035 | 0.100 |
| `to3_soft10_many` | 142 | soft | 0.0053 | 0.0027 | 0.0026 | 0.0020 | 0.0036 | 0.100 |
| donor `sr3_mono` / `sr3_mono_many` | 8 / 142 | — | 0.4841 / 0.2527 | | 0.0012 / 0.0025 | | | |
| donor `sr3_r10` / `sr3_r10_many` (stage-supervised) | 8 / 142 | | 0.9968 / 0.9943 | | 0.0042 / **0.9826** | | 0.9975 | |

**Not undertrained — dead.** Training CE is flat at 2.07 (8 moduli) / 1.93 (142) from the first log point (25k) to 600k, where `sr3_mono` descends 1.467 → 0.541 and `sr3_mono_many` 1.921 → 0.766 on identical data; every tail slope 0.00. The staged forward is *worse* than the monolithic control on the same task.

**Where it dies, by probe.** The decoded seam takes **12 distinct values** over the whole 8-modulus pool (~90 across 142 moduli, under one per modulus) where a residue needs ~600; `true_match` 0.002–0.027 at every stage; relabel purity barely above its within-modulus permutation null (0.977 vs 0.935). Not identity-plus-monolithic-last-stage, and not a relabelled residue — an almost contentless channel, which is why the `st` and `soft` curves agree to three decimals (a seam carrying nothing passes no gradient, and both reduce to the same `(N, c₀)`-only predictor). `restart` (inject the true partial remainder before stage `j`) is flat in `j`, `depth` (extra leading-zero stages) is flat, and `stagefn` is flat at 0.0036 across every quotient digit *including `q=0`* — the map does not return `y'` when `y' < N`. The re-encode is positionally exact by construction, so this is not a prompt bug.

**Read** (the implementer's, which I share): the donor's decomposition is not discoverable by architecture alone. Early in training the decode is uninformative, a stage learns to ignore its seam input, and once ignored there is no gradient to make it informative — a degenerate fixed point, not an expressivity limit. Terminal labels never localise blame to a stage. The interaction the donor found (staging × breadth) needs the staging to be *installed*, and the only thing in the tree that installs it is the single-stage distribution. One scoping note where I differ from the implementer's record: manufacturing that distribution inside a submission is data augmentation under rule 14 (and needs nested model calls if done in the loss), so I read it as closed for the competition, not merely unported.

Single seed, one digit width, one radix; the null is flat and identical across four arms and two seam modes, so seeds were not warranted.

---

## 4. Track C — the board note ([`board_note/`](board_note/README.md))

`README.md` (~1,300 words) and `discord.md` (≤300 words), for participants and organizers: the `[12,14,16]`-bit no-reduction floor is exactly 6/768 at `T=1` (ranks 2–7 to the example; one bit either way gives 17 or 1; the floor collapses to 1 at `T=2` and 0 from `T=4`); the 12-bit cell is dense by arithmetic (14 moduli, 35,624 `(N,x)` pairs, **58%** of them in `m5`'s train split; per-modulus coverage 0.589/0.063/0.004 at 12/14/16 bits), so rank 1's 129/768 with 0% OOD is consistent with interpolating that cell alone; deep rungs are partly periodic (191/768 at `T=16`, 357/768 at `T=64` repeat a lower rung) which changes no ranking; and under test-time re-grounding the ladder is a log-error ladder, `eps ≲ 9e-4/T`, 64× end to end, with the caveats from [`exact_atom/`](../../rule_acquisition/exact_atom/README.md). It says nothing about this program's approach. Every number was re-derived from the public generator and the live board (111 entries on 2026-08-22) — which is how the **~84% → 58%** coverage correction in [`../README.md`](../README.md) §2 was found. Not posted.

---

## 5. Reading

**What the review had as argument is now measured, from both sides.** The "beat everyone" side: the rank-1 tier's mechanism — fixed-`N` rule acquisition — is real and reachable, at ~25% fresh-`x` on a single modulus with the composition axis fully solved in the competition's own metric; that is the first time this program has put a positive number on the atom inside the grader. The "cursed" side: a second, digit-native architecture reproduces the rule-acquisition wall on Hard; architecture alone cannot discover the decomposition that moves the rule axis from terminal labels; and the single-modulus result does not survive contact with the modulus family at any budget tried.

**The one untested thing between this and a board number is the modulus count.** One modulus → 25%; 204 → 0; Hard's 12-bit cell has ~13 training moduli. The generator gives a clean ladder for free — widths 10/12/11/13 bits yield **6/14/21/57** moduli at a fixed 4-digit width — so four single-width sets with matched rows, `T∈{1,2,4}`, exhaustive-`x` cohorts, and `split_group=modulus` would locate the transition. ~40 GPU-minutes (80 with a second seed); generation is CPU-minutes. It cannot run hosted, so it is a proposal, not a launch. It is also the program's rule-axis question asked of the new carrier: in [`staged_reduce/`](../../rule_acquisition/staged_reduce/README.md) breadth *helped* (each stage the same function across moduli); here breadth kills fitting (each `f_N` a different function). Whether a per-digit composed model crosses from the first regime to the second as the family grows is the question, and it is not pre-read here.

**The three methodological exports** are the ones that cost runs: (i) probe gradients and forward-identity before shipping a theory — two plausible mechanisms died to probes that take a minute; (ii) never compare an arm across machines without a matched hosted control — the hosted/local step ratio varied 0.98–1.47× across datasets; (iii) a self-generated consistency target needs a progress gate and no clock fallback, or it re-introduces the failure it was built to prevent.

**Relation to the rest of the program.** The single-modulus result is `ballistic_depth`'s horizon and held-out-`x` questions answered together on the competition's metric by changing the carrier rather than the loss — and `s4`'s finding that a flowing gradient *hurts* a tied map is, suggestively, the competition-side form of what `variable_modulus` and `ballistic_depth` §8 measured about a rollout needing to stay inside its own input domain: the contraction keeps each application trained on its own supervision. That is a reading, not a result. The closure null is the cleanest: what the loss was buying at fixed `N` is structural in a canonical digit carrier.

---

## Honest caveats

- Everything is single seed. The three-modulus replication is the only replication in the node.
- Track A's sweep reads are on our Modal at a dataset-dependent fraction of the hosted clock; every number that carries weight was re-measured hosted (runs 7, 9, 10, 11).
- Track B is one digit width, one radix, one architecture; it closes "architecture alone" for this construction, not every construction.
- The legality readings (loop-on-`T`, the seeding gather, highway-style gradient paths) are readings; Discord was never asked.
- Hard's data is hidden; the modulus-grouped, `[12,14,16]`-bit, `T∈{2,4,8}` shape is inferred from its floor and split names ([`../README.md`](../README.md) §1–2).
- No README was written for the implementer nodes; their `NOTES.md` files are the records and carry more detail than this writeup (every run id, SHA, curve).

## Reproduce

Track A: [`digit_port/FILES.md`](digit_port/FILES.md) and the run table in [`digit_port/NOTES.md`](digit_port/NOTES.md); hosted runs are `one-layer submit <emitted submission.py> --tier medium --dataset m6` etc. (the exact file's SHA is in each `results/hosted/<run>/submission.json`; canonical `submission.py` is the `OP_CROSS=False` configuration, SHA `50c65072…`); local sweeps are `modal_sweep.py::sweep` (app `old-digit-port`, volume `old-dress-rehearsal-data:/digit_results/`). Track B: the command block at the end of [`terminal_only/NOTES.md`](../../rule_acquisition/staged_reduce/terminal_only/NOTES.md) (volume `one-layer-deeper-data:/staged_reduce/terminal_only/<tag>/`). Track C: [`board_note/README.md`](board_note/README.md) §"Reproducing" (`../audit_dataset.py` over public `m5`).

## Next steps (possibilities, not a plan)

1. **The modulus-count ladder** (§5): 6/14/21/57 moduli at fixed 4-digit width, matched rows, exhaustive-`x` cohorts; ~40 GPU-min. The one experiment that says whether the single-modulus result can reach the board.
2. **Post the note**, or not — [`board_note/discord.md`](board_note/discord.md) is ready.
3. Track B's follow-ups, scientifically interesting and competition-irrelevant: where terminal-only dies in `S` (`S=2`, `S=3`; ~7 L4-h); warm-start the stage map on the single-stage distribution and fine-tune terminal-only (does a terminal label *maintain* a decomposition it cannot discover; ~16 L4-h).
4. `m1`'s three confounds (no `T=1` row, table size, epochs) are separable on generated data if the fixed-`N` result is pursued further.
5. Hard attempts are free at 3/day; nothing currently measured is worth one. Spend them on whatever the ladder says.
