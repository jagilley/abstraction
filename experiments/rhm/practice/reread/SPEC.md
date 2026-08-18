# SPEC — reread: is a fixed corpus renewable to a learner whose vocabulary has climbed?

**Status**: spec, unbuilt (2026-08-17). Written from a 2026-08-17 conversation with Jasper on the
practice arc's long-run scope. Sibling spec from the same conversation:
[`../merge/SPEC.md`](../merge/SPEC.md).
**Up**: [`../README.md`](../README.md) (practice arc) · machinery donor:
[`../ratchet/`](../ratchet/README.md) (mining, nested tables, the depth ladder)
**Attribution**: the seed observation is Jasper's — a person reads a book once, learns one set of
things, rereads it six months later, and gleans an entirely new set from the same content, because
the underlying representations changed; hence *full cognitive-loop learners have effectively
unlimited training data*, though not all data is equally expressive about the underlying DGP. The
formalization via the ratchet's nesting result, and the design below, came out of the exchange.

## Why this experiment

The claim under test: **a token's extractable news is not intrinsic to the token — it is indexed by
the reader's current vocabulary.** For a monolithic learner, data is fuel: a second epoch over the
same corpus yields a vanishing gradient, and the field's "data wall" discourse treats the archive as
consumed. For a loop learner, a second pass after a re-chunking cycle is a *different extraction
operator applied to the same ore*: structure that was unparseable on pass one is minable on pass
two, because the units it is written in now exist.

The formal ground is already measured, just never read this way.
[`ratchet`](../ratchet/README.md)'s nesting result: `T[l]` is defined over `T[l−1]` entries, and a
missing level makes the next level **unrepresentable, not merely worse** — therefore *unminable from
data that contains it*. If that is right, mining yield over a fixed archive should be gated on the
learner's vocabulary stage rather than on data novelty. Nothing in the arc has held the archive
fixed to check; every round drew fresh tasks per cycle.

Stakes, in ascending order of reach: (i) both lines of the repo-root
`OPEN_QUESTIONS.md`[^private] are the two directions of one fact — declining
to learn from a sample now (the aleatoric null,
[`../../question_model/SPEC.md`](../../question_model/SPEC.md)) and extracting more from it later
(the reread) both say data value is state-relative; (ii) "not all data is created equal" gets a
coordinate — a corpus's hierarchical depth *relative to the reader's current level* — so deep data
should support many rereads and shallow data should exhaust in one; (iii) if yield is
vocabulary-gated, the data wall is **footprint-exhaustion mistaken for world-exhaustion**, and
"effectively unlimited training data" has a measurable form: effective data ≈ corpus × ratchet
passes. The [`question_model` spec](../../question_model/SPEC.md)'s decomposition defines a token's
news relative to a *posterior*; this experiment tests the stronger dependence — relative to what is
*representable at all*.

## Reading list (ordered)

1. [`practice_manufactures_its_own_credit`](../../../../ideas/practice_manufactures_its_own_credit.md)
   §14 (the ratchet round; nesting), §18 (the monolith clause this bears on).
2. [`../ratchet/README.md`](../ratchet/README.md) — the mining machinery you will reuse, and the
   foreclosure result (premature commitment makes the next level unrepresentable) that is this
   spec's mechanism read in the destructive direction.
3. [`meta_learning_under_metered_data`](../../../../ideas/meta_learning_under_metered_data.md) —
   the doc whose "correct degenerate solution" clause now carries a bracketed 2026-08-17 caveat;
   this experiment is one of its two named instruments.
4. [`../recital/README.md`](../recital/README.md) — the stream-position noise convention (rank
   orderings over fractions; measure a noise floor if you need one).

## Substrate

The ratchet world (RHM sculpting, depth-laddered damage), because the vocabulary is an **explicit
object** there and mining yield is a direct readout — and because the DGP's per-level structure of a
fixed archive is exactly known, so "what fraction of the archive's level-ℓ structure has been
extracted by pass k" can be charted against ground truth. An LM/NTP instantiation (fixed RHM corpus,
per-level BP-probe recovery as the extraction readout, [`../../conditional_revision/`](../../conditional_revision/README.md)
oracles) is the natural twin if you find the sculpting version confounded — your call; note why if
you switch.

The one structural change from `ratchet`: **the archive is drawn once and frozen.** Same damage
instances, same configurations, re-presented pass after pass. Everything else imports.

## Design (held loosely — the instrument list is the commitment)

Arms, in priority order:

- **The reader**: the practice loop re-exposed to the fixed archive across its vocabulary stages,
  mining permitted every pass. The readout is **mining yield per pass, per level** — new table
  entries, and repair success at depths previously unaffordable — as a function of vocabulary stage.
- **The monolith control**: a dense learner given matched extra passes over the same archive
  (epoch-2, epoch-3, …). Its per-pass improvement curve is the baseline shape the reader's yield
  curve is read against.
- **The novelty control** (the load-bearing one): at each vocabulary stage, a matched-size *fresh*
  archive versus the re-read of the frozen one. If fresh and re-read yield similar mining at stage
  k, novelty is not the binding variable — vocabulary is. If fresh dominates everywhere, the reread
  claim is wrong as stated and the interesting question becomes what the fresh draw carries that the
  archive doesn't.
- **The depth coordinate**: sweep the archive's hierarchical depth. The theory's prediction
  (recorded per repo norms as a prediction, not a constraint on interpretation): shallow archives
  exhaust in one pass for both arms; deep archives separate them, with the reader's yield
  re-arming after each ratchet cycle.

Sanity discipline: pass-1 must reproduce the ordinary ratchet round on this archive (fidelity
check); yield accounting must not double-count entries re-derivable from `T[l−1]` composition alone
(a mined entry counts only if it changes repair behavior at its level — reuse `ratchet`'s
concentration-vs-coverage instruments).

## Practicalities

- Invoke `/run-experiment-on-modal` first; profile `chromatic`; smoke-test before long runs; follow
  the halting procedure for anything >5 min.
- Single seed first, per `experiments/CLAUDE.md`. Rank orderings over fractions where noise is
  uncalibrated.
- Structure per `experiments/CLAUDE.md`: this folder gets its `README.md` post-discussion and a
  `FILES.md`; import `../ratchet/` machinery, don't copy or modify it.
- Bring the numbers back for discussion before writing any README (repo norm).

[^private]: Not mirrored: this link points to a document in the private lab repo (the roadmap, the queue, an unrun spec, reading notes, or a conversation). See the top-level README for what is held back and why.
