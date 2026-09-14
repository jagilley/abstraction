# SPEC — figured_bass: commit the key, not the content

**The question in one sentence**: when the vocabulary is keyed by category, should a commit
freeze the *inventory* — the spellings the book holds, and so which classes it covers — or only
the *partition* and the executor's adoption of the level, leaving the inventory and the class
coverage to keep filling from the stream?

**Status**: spec, 2026-09-10. Nothing built. **Parent**: [`../SPEC.md`](../SPEC.md)
(`enharmonic`); sibling of [`../temperament/`](../temperament/SPEC.md), whose merge op composes
with this one. The name: a figured bass commits the harmony and leaves the voicing to the
player. **Origin**: the Q1 runs (`en_s0`, `en_s1`; facts in
[`../figures/en_s0_reduction.txt`](../figures/en_s0_reduction.txt) §[G2]) read against
[`../../tutti/README.md`](../../tutti/README.md) finding 4. **Attribution**: the commit-time
table is the Q1 reduction's; the reading that a commit freezes the wrong object in category
coordinates came out of the 2026-09-10 exchange with Jasper. **Machinery**:
[`../enharmonic.py`](../enharmonic.py) (`operative()`, the class-keyed miners, the yokes) and
[`../quotient.py`](../quotient.py); nothing new is needed below them.

## Why this node exists

Q1 solved arrival and moved the wall. With `T[ℓ]` keyed by token class the L5 and L6 gauges
carried 37 and 34 keys at support where the flat key's were identically 0 — and L5 still did not
serve demand, because the L4 book the L5 build looks halves up in was frozen at the commit:

| arm | L4 commit | rows at commit | L4 classes at commit (of 13) | L4 classes the live miner reached |
|---|---|---|---|---|
| `given_cat_tok` | c95 | 64 | **3** | **9** |
| `given_cat_min` | c87 | 144 | 4 | 5 |
| `flat_yk_tok` (flat key, same clock) | c95 | 1 | 1 | 4 |

The yield-quiet licence fired when L4's class-pair arrival plateaued on era 3's damage cell; era
4's cell then showed the classes that matter and the frozen book could not admit them. The flat
key has little to lose from freezing (1 → 4); the class key loses most of what it was for.

The precedent is already in the arc. `tutti`'s mirror won its lifetime ceiling because δ-silence
advanced the era before yield could quiet on L3, so **L3 was never frozen** and L4 was committed
over a live L3 set holding 28 keys at support against the frozen arm's 10. `ratchet` measured
the foreclosure of an early flat commit (−0.372, worse than never) — a fact about freezing a
*spelling* table. In category coordinates what the level above needs from the level below is
its *classes*, and a class is a schema the stream can keep filling after the executor has
adopted it. This is the drain reading one step further: the hippocampal index consolidates into a
schema, and a schema is committed as a key, not as a list.

## What a commit does today, and what this node separates

A commit does three things at once: (a) mints the level's π slots and starts the corridor head
on it — the executor adopts the level; (b) freezes the table the DP executes over; (c) freezes
what the level above can be built from. Under a class key, (b) and (c) can stay live while (a)
happens: the operative table of a committed level becomes the frozen partition applied to the
live at-support keys, the DP maxes over whatever inventory is held, the corridor imitates the
DP's current intention and keeps training as it does across eras anyway. In the fork this is one
branch of `operative()`. What is *not* known, and is the experiment: whether an executor can
adopt a level whose content keeps moving under it (parity, misfires, slot open/close churn across
the commit), and whether the mirror loop's licences — which read arrival and δ-silence, neither
of which depends on freezing — behave the same.

## First shapes (sketches; per repo norms no outcome is interpreted in advance)

- **Q0, offline, before any GPU**: replay `en_s0`'s logs. At every cycle after the L4 commit,
  the live miner's L4 at-support keys under the token class: classes covered, L5 class-pair keys
  `buildable()` over them, and the cycle at which L5 would have reached support — against the
  frozen book's 2–7 keys. The same for L3 → L4. This says whether an open inventory reaches L5 at
  the existing budget before anything is built (`ostinato`'s discipline).
- **Arms**: `given_cat_tok` (banked; the frozen-inventory anchor) · `given_cat_tok_open` (the
  commit adopts; inventory and coverage stay live) · `flat_open` (the same bit in flat
  coordinates — `tutti` says a live level helps there too) · the house-style clock yokes
  (`flat_yk_open` replaying the open arm's schedule). If cheap, after `en_s3` lands: `endo_open`,
  `temperament`'s merge acting on the live book past the commit — the two ops compose, and the
  executor's expansion choice under an open inventory is where
  [`../../inflection/`](../../inflection/SPEC.md) meets this node.
- **Readouts**: L4 class coverage per cycle past the commit · L5 build, at-support, and commit ·
  the matched-clock era-4/5 error · the corridor's parity, misfire rate and slot churn across the
  commit · π's per-slot mass · the deletion battery with growth · the oracle-read count.
- **Gates**: the fork bit-identical with the open bit off (`given_cat_tok_open` ≡ `given_cat_tok`
  at 0.000e+00) · Q0 first.

## Norms

The parent's and `temperament`'s: volume, difficulty mix, priced budget and lifetime held fixed
across arms with the yokes; single seed first; Modal per `/run-experiment-on-modal`; an
implementer subagent reads `/subagent-instructions` and never touches git; results discussed
before a README; `QUEUE.md` / `ROADMAP_PROGRESS.md` on landing.

**First pass landed (2026-09-11, `fb_s0` own clock, `fb_s1` the anchor's clock)**, facts only:
[`sizing/SIZING.md`](sizing/SIZING.md), `../figures/fb_s{0,1}_reduction.txt`, index
[`FILES.md`](FILES.md). The committable L5 miner is era-gated; the open bit re-paces the loop
through δ-silence on its own clock and, on the anchor's, reaches Q0's 9 of 13 L4 classes at
token precision 1.000 while costing consumption-era error. Interpretation pending discussion.
