# Authorship — design decisions

Node: `experiments/a2a_forward/fsm_part3/authorship/`.
Parent battery: [`../../confabulation/confabulation.py`](../../confabulation/confabulation.py) and [`../../confabulation/README.md`](../../confabulation/README.md).
Paper: `papers/forward_self_models_paper2.md`[^private], §2 (the criteria) and §4–§5 (the ladder and the dissociation).
File index: [`FILES.md`](FILES.md).

This file is the record of *why the experiment is shaped the way it is*, written before the
run. It is not a writeup and contains no results.

---

## 1. The question, and why the framework's answer is a prediction rather than a guess

Can a transformer tell which parts of its context it wrote itself?

The construction answers before the experiment does. M re-reading its own output runs
**exactly the same computation** as reading anyone else's: the state at every layer on the
re-read pass is a deterministic function of the token sequence, and of nothing else. There
is no register that was set when M sampled the token and is still set when M reads it back.
Nothing about "having written it" survives into the re-read except through the tokens.

So in the paper's taxonomy, `MINE` is an **input-determined** fact — the same category as
the `WORLD` control (§3.2), not the `IMPL` category. Criterion 1 (the report is a function
of M's internal state) holds vacuously, because *everything* on the re-read pass is.
Criterion 3 (M's behaviour is causally sensitive to it) is not even in play. Criterion 2 —
not *cheaply* recoverable from M's input–output map — is the whole question, and the
construction says it should fail: authorship should reduce to **"is this token likely under
me"**, a quantity an observer holding M's own output distribution computes directly.

That makes the experiment worth running rather than assuming, for two reasons:

- **It is falsifiable and cheap to falsify.** If the self-report beats the best I/O
  observer, the framework is wrong about something specific, and the same battery says
  *where*: the `O_act` ceiling separates "the observer is short of access" from "the
  observer is short of capacity", and the capacity/data ladder separates both from "we
  under-built the observer".
- **The literature has a positive result that this bears on.** Panickssery et al. (2024)
  report that LLM evaluators recognize and favour their own generations. If, on a system
  where the boundary is explicit, recognition is fully matched by an observer holding the
  model's logits, then self-recognition of that kind is **self-inference on a public fact**,
  not introspection — the same conclusion the paper reaches for calibration (§5), reached
  the same way.

Either outcome is the result. The design below is built so that neither is favoured.

---

## 2. What "M wrote this" is made of

### 2.1 Sampling

M has no sampling loop in `a2a_forward/model.py`; one is written here
(`_sample_into`). At each position flagged as M's, M samples from `softmax(logits/τ)`
conditioned on **everything already in the sequence** — corpus spans and self-written spans
alike. Temperature 1 is the honest case: the token M writes is drawn from M's own
next-token distribution, so it is not identifiable by being "more probable than the corpus"
in any per-token sense. It is identifiable only to the extent that M's distribution differs
from the text distribution — which is exactly the quantity the framework says is public.

A lower temperature (0.7) is run for the record. At low temperature M writes near its mode,
and the text is trivially recognizable by likelihood; that run is not evidence about
introspection, it is a check that the detectors respond the way a likelihood story predicts.

### 2.2 Report sequences

Two constructions, both on held-out (val-split) FineWeb-Edu documents, T = 128:

**`mix`** (the main set) — a 24-token corpus prefix, then spans that alternate class until
the sequence is full. The starting class is drawn per sequence, and every span length is
drawn from a shared distribution `{8, 12, 16, 24, 32}` used identically for both classes.
Consequently: absolute position does not predict the class; span length does not predict
the class; the existence of a boundary does not predict which side is which. ~5–6 spans per
sequence, so the per-position label really does alternate within a sequence rather than
being a sequence-level label in disguise.

**`whole`** (the splice-free control) — a 24-token corpus prefix, then *either* the whole
remainder sampled from M, *or* the document's own true continuation, 50/50 across
sequences.

Why both. The `mix` construction has one asymmetry that cannot be designed away: an M span
is conditioned on the visible context, while a corpus span that follows an M span does not
continue it. The corpus class therefore carries a splice discontinuity that the M class
does not. This does **not** invalidate the comparison — a discontinuity is a *public* cue,
visible in M's own logits, so it helps the observer at least as much as the self-report —
but it does make the task easier for everyone and could compress the advantage against a
ceiling. `whole` has no splice at all on either side: both classes are genuine
continuations of the same prefix. If the two sets agree, the splice is not driving
anything.

The prefix is excluded from scoring in both sets. It is always corpus and always at the
front, so scoring it would hand every contestant a free positional cue. It is context, not
evidence.

### 2.3 The re-read

Report sequences are re-read by M in an ordinary teacher-forced forward pass — one pass,
no injection, the `ol` arm. The activations cached from that pass (`post_block0`,
`post_block2`, `post_block3`) are what every first-person contestant reads. This is the
structural point of the experiment made concrete: it is the same pass M would run on text
from anywhere.

---

## 3. The contestants

`advantage = best self-report − best I/O observer`, as in the paper (§3.3).

| contestant | reads | why it is in the battery |
|---|---|---|
| `SELF_mlp` | M's `post_block3` at position *t* | the parent's report head, unchanged, so the `IMPL` control is comparable to the published numbers |
| `SELF_seq` | M's `post_block3`, whole sequence | `MINE` is a span-level fact and the observers are sequence models; see §3.1 |
| `O_input` | tokens | the paper's tokens-only rung |
| `O_lik` | M's six per-token likelihood features, **and nothing else** | the explicit "is this likely under me" baseline the framework says everything reduces to |
| `O_io` | tokens + top-*k* output summary + 4 full-distribution scalars + likelihood features | everything the I/O map has to offer; see §3.2 |
| `O_io_nolik` | the parent's `O_io` exactly | shows how much of the observer's score the likelihood channel carries |
| `O_act` | tokens + M's `post_block0` | the activation-access ceiling (paper §4): distinguishes "expensive to reconstruct" from "not learnable" |
| `LIK_pos` | the six likelihood features at position *t* alone, MLP | the no-aggregation floor of the likelihood route |

All sequence models share one architecture, one optimizer, one data budget and one
capacity ladder (1L/64D, 2L/128D, 4L/256D — the last matching M). The ladder therefore
measures access, not architecture. Observers are bidirectional by default, as in the
parent, because the argument rests on giving the third party every advantage we can
afford. The strongest `O_io` is retrained at half data as the budget control.

### 3.1 Why the self-report gets a sequence model

The parent's report target (`IMPL`) is a per-position fact and its report head is a
per-position MLP. `MINE` is not a per-position fact: evidence for "this span is mine"
accumulates across the span. Comparing a per-position MLP against bidirectional sequence
observers would be a rigged comparison, and a null would be uninterpretable ("you crippled
the head"). `SELF_seq` is the same Observer architecture at the same capacities, handed M's
`post_block3` stream instead of tokens — i.e. the report site read with the observers' own
aggregation power. `SELF_mlp` is kept alongside so the `IMPL` row stays comparable to the
paper.

Is `SELF_seq` a "self-report"? It is a trained network reading M's own late stream, which
is exactly what the paper's report head is; the paper's caveat that the report head is
trained and that the introspection belongs to the composite applies identically. The
self/observer distinction is about *what is read*, not about what does the reading.

### 3.2 Why `O_io` is handed the likelihood features — a deliberate deviation

The parent's `O_io` sees, at position *t*, M's output distribution over token *t+1* (top-*k*
ids and probabilities plus four full-distribution scalars). To compute `log p(x_t | x_<t)`
— the single most relevant public quantity for this target — it would have to **join** the
distribution emitted at *t−1* with the identity of the token realized at *t*, across
positions, through attention it was never trained to use. Withholding that join would
manufacture a self-advantage out of an input-format artifact and would be exactly the
mistake the parent already guards against elsewhere (the four full-distribution scalars
exist because without them the observer could not compute M's entropy and the `ENT` control
showed a fake advantage).

`log p(x_t | x_<t)` is a function of M's I/O map and the input sequence. It is public by
construction. So it is handed over. The six features, all aligned to the token *at* that
position and all derived from the distribution emitted at *t−1*:

```
log p_M(x_t | x_<t)          log1p(rank of x_t)          H[p_M(· | x_<t)]
max_v p_M(v | x_<t)          surprisal − entropy          log p(x_t) − log max_v p(v)
```

standardized on train positions only. `O_io_nolik` keeps the parent's likelihood-free
observer in the same run, so the size of the deviation is itself a reported number rather
than an assumption.

The mirror-image question — does M's *own* state at position *t* contain its surprisal at
*t*? — is left to the data. M computed `p(x_t | x_<t)` at position *t−1*, not at *t*, and
was never trained to carry it forward. `SELF_seq` can attend to *t−1* and recover it if it
is there; `SELF_mlp` cannot. That difference is informative and is reported, not designed
away.

---

## 4. The positive control

The parent's `IMPL` / `IMPL_COS` targets are run **on the same report sequences, with the
same splits, on the same scored positions**, with a freshly trained instrument forward
model at the parent's default capacity (`d_head=16, mlp_mult=0.5`, ~10.6% of the two
predicted blocks) and its two junk-residual guards: `ens_cos` over three independently
seeded instruments, and the residual-structure statistics (`eta2_norm`, `eta2_dir`,
`eta2_vec`, per-category Cohen's *d* on |r| against the published table).

Running it on the authorship sequences rather than on clean corpus sequences is the
stronger form of the control: if `IMPL` shows its advantage on the very same
half-synthetic text and the very same positions where `MINE` shows none, then a null on
`MINE` is not an artifact of the data, the splits, the report site, or the pipeline. The
number may differ from the paper's (whose report set is pure corpus text), and that is
expected rather than a discrepancy.

The four channel ablations and the fair confabulator come along for free from the parent's
report-head recipe, and they are run on `MINE` too. The prediction there is the mirror
image of `IMPL`: if authorship is theory-visible, the `MINE` report should survive
`shuffle_r` (destroy the residual) and suffer under `shuffle_p` (destroy the self-theory's
component), and its confabulator margin should be near zero. `IMPL`'s published pattern is
the opposite. Both are measured in the same job.

## 5. What is deliberately *not* here

- **One arm (`ol`), one seed.** The CL/OL contrast and the instrument-capacity sweep are
  the parent's questions. Nothing in this design turns on the loop: the authorship signal,
  if any, is a property of the re-read pass, which is identical in both arms. The wake
  checkpoint is loaded from the parent's `wake_ckpt/` key when present.
- **No steering (Test 3).** Criterion 3 asks whether M's behaviour is causally sensitive to
  the reported signal. For an input-determined fact there is no separate signal to steer:
  perturbing the state perturbs the input's representation. Test 3 would not mean here what
  it means for `IMPL`.
- **No `BEHAV`/`ENT`/`WORLD`.** They do not depend on this construction and the paper
  already has them; each costs a full observer ladder.
- **Descriptive statistics instead of extra ladders** where a ladder would add nothing:
  mean `log p_M` per class, and the AUC of raw surprisal used directly as an untrained
  detector. The latter is the cheapest possible public detector and is the number the
  framework's claim is really about.

## 6. Reporting

Accuracy (comparable to the parent) **and** AUC (the right statistic for a binary target
with a possibly-imperfect class balance) for every contestant, plus per-position test
predictions saved to `predictions.npz` so the reduction can plot accuracy against
*offset within span* — how much of the span a contestant has seen. A detector that needs
evidence to accumulate and one that reads a per-token cue look different on that curve.

---

## 7. Phase 2 (designed, not built): making authorship an implementation fact

Phase 1 measures a system in which authorship *cannot* be an implementation fact. The
constructive half asks for the smallest addition that makes it one. The requirements, from
the brief and from the framework:

1. the signal must be a function of something **absent from the tokens** — the sampling
   noise, or the pre-sampling logit state at generation time;
2. it must be available **only to the generating system**;
3. it must be carried in **M's state**, not handed to the report head as an input flag (a
   flag the head reads directly is trivially decodable and shows nothing).

**Construction.** At generation time, at each position M samples, draw the Gumbel noise
`g_t` used for the sample and add `ε · W g_t` (a fixed random projection, `W` frozen,
unknown to every contestant) into M's residual stream at that position; cache the
generation-time state and replay it on the re-read, so that on self-written spans M's state
differs from a fresh read by a signal only the generator had. Re-run the battery. The
ceiling observer is handed `g_t` itself.

**Two honest caveats to settle before building it.**

- *It risks being a watermark rather than an efference copy.* `ε·Wg_t` is, to any reader
  who does not know `W`, an extra random component; detecting it is detecting added
  variance, which is a statement about the perturbation, not about self-knowledge. The
  version that is not a watermark requires M's computation to *use* the copy — i.e.
  criterion 3 — which means training M with the efference copy present, not bolting it on
  to a frozen M. That is a different and larger experiment.
- *The comparison to Phase 1 must be like-for-like.* A Phase-2 positive with a frozen M
  says only "an unpredictable component was added at some positions and a probe found it".
  The interesting Phase-2 claim is conditional: the advantage should appear *and* the
  observers' ladder should stay flat *and* the noise-access ceiling should close it — the
  same three-part shape the paper demands of `IMPL`.

Phase 1 is the standalone result; Phase 2 is only worth building if the budget clearly
allows the trained-with-efference-copy version, not the bolt-on.

[^private]: Not mirrored: this link points to a document in the private lab repo (the roadmap, the queue, an unrun spec, reading notes, or a conversation). See the top-level README for what is held back and why.
