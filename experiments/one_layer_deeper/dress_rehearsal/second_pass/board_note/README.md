# What the One Layer Deeper leaderboard is measuring

*A note for participants and organizers of [One Layer Deeper](https://github.com/tilde-research/one-layer-deeper)
(Core Automation × Tilde Research). Board reading and dataset measurements: 2026-08-22, upstream
commit `4ceff95`.*

**Up**: [../README.md](../README.md) (second_pass) · [../../README.md](../../README.md) (dress_rehearsal) · [../../../README.md](../../../README.md) (one_layer_deeper)

---

The competition asks for `y = x^(2^T) mod N` given `(N, x, T)`, and ranks its Hard tier by the
largest consecutively certified rung on the ladder `T = 1, 2, 4, 8, 16, 32, 64`. A rung certifies
when all 768 fresh prompts at that depth are exactly right, first on moduli seen during training
and then on unseen ones, with accuracy at the first uncertified rung as the tiebreak.

Nine days from the deadline, with 111 ranked entries, nobody has certified `T=1`. Every score on
the board is a small multiple of 1/768. This note is about what those multiples mean.

Four facts about the task's arithmetic account for most of the board's current shape, and all four
are checkable from the public repository in a couple of minutes. Hard's data is hidden, and the
problem page warns that Hard "may change aspects of the recurrence itself," so the first three are
properties of the *public* generator and carry over only if Hard is the same family. The fourth
follows from the ranking rule alone.

## 1. There is a no-reduction floor, and it measures exactly 6/768

Some prompts never touch the modulus. When `x^(2^T) < N` the answer is the unreduced power, so a
model that has learned to square small integers as digit strings answers them knowing nothing at
all about reduction. The share of such prompts at a given width is about `1/√N`.

On the public `m5` config, whose moduli are 12, 14 and 16 bits wide, that comes to exactly
**6 of 768** at `T=1` on the seen-`N` profile, and 11 of 768 on the unseen-`N` profile. Ranks 2
through 7 currently read **0.7813% = 6/768** seen and 0.3906% = 3/768 unseen.

So the second tier of the leaderboard is, to the example, the score of a model that learned the
multiply and never learned the reduce. That is a real thing to have learned, and it reads more
usefully as itself than as partial credit on the rule.

Two caveats. Hard's widths are inferred rather than known: `[12,14,16]` is the anchor because it
reproduces 6/768, while one bit either way gives 17/768 (`[11,13,15]`) or 1/768 (`[13,15,17]`).
And the unseen-`N` floor is not a usable constraint at all, since the same widths give 11 on one
draw and 4 on another. Those are Poisson fluctuations on a count near five.

The floor also collapses with depth: 6 prompts at `T=1`, 1 at `T=2`, none from `T=4` onward. The
free cases live only at the rung that gates the whole ladder, and six out of 768 does not open it.

## 2. The 12-bit cell is dense by construction

The generator draws `p` and `q` fresh for every record from all primes of half the target width,
so the modulus population at each width is fixed by arithmetic rather than by dataset size:

| bits | 10 | 11 | 12 | 13 | 14 | 15 | 16 |
|---|---|---|---|---|---|---|---|
| distinct valid RSA moduli | 6 | 21 | **14** | 57 | **42** | 169 | **148** |

Fourteen moduli exist at 12 bits, and 35,624 `(N, x)` unit pairs across all fourteen of them. A
dataset drawing 10,000 examples per setting therefore trains on **58%** of every 12-bit pair there
is. Mean per-modulus base coverage runs **0.589 / 0.063 / 0.004** at 12 / 14 / 16 bits.

Three widths in one dataset therefore behave like three different problems. At 12 bits a model can
see most of the space it will be tested on, at 14 bits a few percent of it, at 16 bits almost none.
Each depth rung draws 256 of its 768 prompts from each width.

That gives the top of the board an arithmetic reading. Rank 1's **129/768 = 16.7969%** with
0.0000% on unseen moduli is about half of one 256-example cell and nothing whatsoever off the
training moduli. Interpolating the dense cell alone would look like that. This is not a claim about
what that entry does. It is a statement about which scores are cheaply available, which is worth
knowing before reading any nonzero number as evidence about the rule.

One more structural note. Hard's three scored splits come back named `test`, `ood_t` and
`ood_n_t`, and in the generator's own vocabulary that combination requires whole moduli to be
partitioned between train and test. If Hard's widths are `[12,14,16]`, roughly 13 / 38 / 133
moduli per width are available for training and the rest are held back.

## 3. Deep rungs are partly degenerate, and it hands nobody a rung

`x^(2^T)` is eventually periodic in `T`, so beyond some depth the ladder starts revisiting answers.
On `m5`'s seen-`N` profile the number of prompts whose answer at a rung already appeared at a lower
rung is 74/768 at `T=8`, 191 at `T=16`, 304 at `T=32` and **357 at `T=64`**. Nearly half the
deepest rung is free once a shallower one is solved.

Certification is all-or-nothing, so this changes nothing about the ranking. It matters for reading
diagnostics: an accuracy of 0.4 at `T=32` is not a measurement of composition to depth 32, because
up to 40% of that rung can be answered by returning a shallower answer unchanged.

## 4. The ladder is a log-error ladder, and depth is the cheap axis

The rungs double, which makes the ladder look like a depth ladder. Under one assumption it is an
error ladder instead, and a short one.

The assumption is that a depth-`T` rollout can be turned into `T` depth-1 problems at test time, by
re-grounding the model's state through its own decode at every step. On a fixed modulus this is
measurable without retraining anything: an ungrounded model goes from 0.000 to 0.377 exact at
`T=60` under that intervention, a model trained with a state-consistency constraint holds 1.000 at
every depth out to `T=60`, and accuracy tracks `p^⌈T/k⌉` in the number of restarts, where `p` is
the per-restart rate and `k` the period. Nothing there is specific to this competition's data. It
needs only that the composition be restartable, which for a serial recurrence it is.

Grant that, and certifying rung `T` needs a one-step exact error `eps` small enough that all
`768·T` steps land: `(1 - eps)^(768·T) ≳ 1/2`, which is

**`eps ≲ 9.0e-4 / T`**

| rung `T` | 1 | 2 | 4 | 8 | 16 | 32 | 64 |
|---|---|---|---|---|---|---|---|
| tolerable one-step `eps` | 9.0e-4 | 4.5e-4 | 2.3e-4 | 1.1e-4 | 5.6e-5 | 2.8e-5 | **1.4e-5** |

Each doubling of `T` costs one factor of two in `eps`. The entire ladder, from certifying nothing
to certifying `T=64`, is 64× in one-step error. A model that gets one modular squaring right on
99.91% of fresh `(N, x)` certifies `T=1`; one at 99.9986% certifies all seven rungs. Essentially
all of the difficulty sits at rung zero.

Two caveats on the formula. It assumes per-step errors are independent across a trajectory, which
is false in detail, since the error set is indexed by state and a trajectory that enters it tends
to stay wrong. And the re-projection result it rests on was measured at fixed `N` on bases seen
during training, so the rate `p` there is not a rate on fresh `(N, x)`.

## What is being ranked right now

Of the 111 ranked entries, 69 score zero on both profiles, 10 more score 1/768 on unseen moduli
only, 21 score 1/768 on seen moduli and 4 score 2/768. The organizers' own AdamW baseline, run at
the Medium allowance on public `m5`, reads 0/768 at `T=1` against that dataset's floor of 6. The
six-entry tier at 6/768 is the free cases, and the single entry above it is consistent with the
dense cell.

Budget does not look like the binding constraint. At the full 3,600-second Hard allowance, a
9.9M-element recurrent operator trained on terminal cross-entropy reached 58% exact on its own
training batches at the buzzer and still scored 0/768 at `T=1` on both profiles, with evaluation
loss of 11.4 nats against the 2.30 a uniform digit distribution would give. It memorised its
training set and transferred none of it.

Put together: what the ladder asks for is one exact modular squaring on a fresh `(N, x)`, to about
one part in a thousand, after which depth comes comparatively cheap. That is a narrower target
than seven rungs make it look, and probably a more encouraging one nine days out.

## Reproducing the floor, coverage and periodicity numbers

Everything in §1–§3 comes from one script over the public datasets, and needs no GPU:

```bash
git clone https://github.com/tilde-research/one-layer-deeper.git && cd one-layer-deeper
git checkout 4ceff95 && uv venv && uv sync
bash scripts/generate_datasets.sh     # writes the 20 public datasets under data/generated/

python <path>/audit_dataset.py \
  data/generated/squaring_mod_new11_medium_bidirectional_variable_b121416_t248
```

`audit_dataset.py` lives at
[`experiments/one_layer_deeper/dress_rehearsal/audit_dataset.py`](../../audit_dataset.py) in this
repository. It reports, per bit width, the distinct training moduli and per-modulus base coverage,
and then for each of the seven rungs on both depth profiles the no-reduction count, the number of
prompts repeating a lower rung's answer, and the exact accuracy a floor-scoring model would show.

`m5` is the anchor because its widths are the ones that reproduce the board's 6/768; it holds
81,000 training prompts spread over the three widths. The modulus counts in §2 come from the
generator's own enumeration in `data/squaring_mod.py`, which is exhaustive below 20 bits.

Worth knowing if you want to calibrate against the ladder yourself: Easy and Medium expose the
same `Max T` and `OOD N Max T` fields as Hard, on the same seven-rung ladder, as diagnostics that
do not affect their exact-accuracy scores. So the full rung profile on both the seen-`N` and
unseen-`N` sides is measurable on the practice tiers, at 1/60 and 1/6 of Hard's clock and without
spending the one Hard attempt a day.
